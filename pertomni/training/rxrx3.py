"""Contrastive training on RxRx3-core cell painting (compound and CRISPR wells of HUVEC cells)."""

from dataclasses import dataclass

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .. import paths
from ..config import field
from ..data.cellpainting import (
    CHANNEL_MODES,
    IMG_DIM,
    N_CHANNELS,
    RxRx3Dataset,
    aggregate_by_pert,
    load_img_embeddings,
    pert_key,
)
from ..data.datasets import sequential_loader
from ..io import load_obj, save_obj
from ..models.rxrx3 import RxRx3MultiCLIPModel
from .common import device, fit, make_trainer

RXRX3 = paths.SCRATCH / "cellclip_data"
SPLIT_DIR = RXRX3 / "rxrx3_results" / "split"


@dataclass
class RxRx3Config:
    h5_path: str = field(
        str(RXRX3 / "rxrx3_dinov2.h5"), help="well image embeddings (N x 6 instances x 5 channels x 1536)"
    )
    meta_path: str = field(str(RXRX3 / "rxrx3" / "metadata_rxrx3_core.csv"))
    channel_mode: str = field(
        "mean_all",
        help="aggregation of the 5 channels: mean_all, concat, per_channel (shared projection)",
        choices=CHANNEL_MODES,
    )
    compound_only: bool = field(False, help="train on the compound wells only (no CRISPR text encoder)")
    enc_cl: bool = field(False, help="append the HUVEC cell line text embedding to every text embedding")
    losstype: str = field("siglip", choices=["siglip", "clip"])
    temperature: float = 0.07
    weighted_loss: list[float] = field([1.0, 1.0], help="weights of the compound and CRISPR losses")
    projection_dim: int = 128
    lr: float = 1e-4
    batch_size: int = 128
    min_epochs: int = 100
    max_epochs: int = 500
    es_mode: str = field(
        "es_on_acc", help="early stopping on validation accuracy or loss", choices=["es_on_acc", "es_on_loss"]
    )
    es_patience: int = 10
    seed: int = 0
    num_workers: int = 4


def load_text_embeddings(compound_only):
    """{name: embedding} of the compounds and (unless compound_only) the CRISPR genes, and the HUVEC embedding."""
    as_float = lambda d: {k: np.array(v, dtype=np.float32) for k, v in d.items()}
    drug = as_float(load_obj(paths.CELLPAINTING / "ensem_emb_drug.pickle"))
    gene = None if compound_only else as_float(load_obj(paths.CELLPAINTING / "ensem_emb_gene.pickle"))
    huvec = np.array(load_obj(paths.CELLPAINTING / "ensem_emb_cellline.pickle")["HUVEC"], dtype=np.float32)
    return drug, gene, huvec


def well_text_embeddings(txt_embs, pert_type, well_to_pert, well_to_type, cell_emb=None):
    """{well_id: text embedding of its perturbation [+ cell line]} for the wells of one perturbation type."""
    out = {}
    for wid, pert in well_to_pert.items():
        if well_to_type.get(wid) == pert_type and pert_key(pert) in txt_embs:
            emb = txt_embs[pert_key(pert)]
            out[wid] = emb if cell_emb is None else np.concatenate([emb, cell_emb])
    return out


@torch.no_grad()
def export_aggregated(model, loader, idx):
    """Perturbation-level (image, text) embeddings of the test wells."""
    model.eval()
    imgs, txts, perts = [], [], []
    for batch in loader:
        perts.extend(batch.pop("pert_label"))
        batch = {k: v.to(model.device) for k, v in batch.items()}
        img_emb, txt_emb = model.encode(batch, idx)
        imgs.append(img_emb.cpu())
        txts.append(txt_emb.cpu())
    return aggregate_by_pert(torch.cat(imgs), torch.cat(txts), perts)


def train_rxrx3(config, run_dir, checkpoint=None):
    """Train (or, with checkpoint, only evaluate) and write the test embeddings to run_dir:
    test_{cmp,kd}_{img,txt}_embeds.pkl and test_{cmp,kd}_perts.pkl (compound / CRISPR, one row per perturbation)."""
    c = config
    L.seed_everything(c.seed)
    img_embs = load_img_embeddings(c.h5_path, channel_mode=c.channel_mode)
    image_dim = IMG_DIM * N_CHANNELS if c.channel_mode == "concat" else IMG_DIM

    drug_embs, gene_embs, huvec = load_text_embeddings(c.compound_only)
    has_crispr = gene_embs is not None
    meta = pd.read_csv(c.meta_path, low_memory=False)
    well_to_pert = load_obj(
        paths.CELLPAINTING / "rxrx3_results" / "mapping.pkl"
    )  # labels carry a well suffix, e.g. 'Phloretin-AD37'
    well_to_type = dict(zip(meta["well_id"], meta["perturbation_type"]))
    cell_emb = huvec if c.enc_cl else None
    txt = {"COMPOUND": well_text_embeddings(drug_embs, "COMPOUND", well_to_pert, well_to_type, cell_emb)}
    if has_crispr:
        txt["CRISPR"] = well_text_embeddings(gene_embs, "CRISPR", well_to_pert, well_to_type, cell_emb)
    cl_dim = len(huvec) if c.enc_cl else 0
    drug_text_dim = len(next(iter(drug_embs.values()))) + cl_dim
    crispr_text_dim = (len(next(iter(gene_embs.values()))) + cl_dim) if has_crispr else None

    splits = {
        s: pd.read_csv(SPLIT_DIR / f"cellpainting-split-{s}-imgpermol.csv")["SAMPLE_KEY"].tolist()
        for s in ("train", "valid", "test")
    }
    loader_kw = dict(
        batch_size=c.batch_size, num_workers=c.num_workers, pin_memory=True, persistent_workers=c.num_workers > 0
    )
    loaders = {}
    for pert_type in txt:
        for split, wells in splits.items():
            dataset = RxRx3Dataset(
                [w for w in wells if well_to_type.get(w) == pert_type],
                img_embs,
                txt[pert_type],
                pert_type,
                well_to_pert,
            )
            loaders[pert_type, split] = DataLoader(dataset, shuffle=split == "train", **loader_kw)
            print(f"{pert_type:<9} {split:<5} {len(dataset):>6} wells")

    model_kwargs = dict(
        image_dim=image_dim,
        drug_text_dim=drug_text_dim,
        crispr_text_dim=crispr_text_dim or 3072,
        projection_dim=c.projection_dim,
        lr=c.lr,
        losstype=c.losstype,
        temperature=c.temperature,
        weighted_loss=tuple(c.weighted_loss),
        channel_mode=c.channel_mode,
        has_crispr=has_crispr,
    )
    if checkpoint is None:
        model = RxRx3MultiCLIPModel(**model_kwargs)
        monitor, mode = ("val_acc", "max") if c.es_mode == "es_on_acc" else ("val_loss", "min")
        trainer = make_trainer(run_dir, c.min_epochs, c.max_epochs, monitor=monitor, mode=mode, patience=c.es_patience)
        val_loader = sequential_loader([loaders[t, "valid"] for t in txt])
        checkpoint, _ = fit(trainer, model, [loaders[t, "train"] for t in txt], val_loader, run_dir)

    model = RxRx3MultiCLIPModel.load_from_checkpoint(checkpoint, map_location=device(), **model_kwargs)
    for idx, (pert_type, prefix) in enumerate((("COMPOUND", "cmp"), ("CRISPR", "kd"))):
        if pert_type in txt:
            imgs, txts, perts = export_aggregated(model, loaders[pert_type, "test"], idx)
            save_obj(imgs, run_dir / f"test_{prefix}_img_embeds.pkl")
            save_obj(txts, run_dir / f"test_{prefix}_txt_embeds.pkl")
            save_obj(perts, run_dir / f"test_{prefix}_perts.pkl")
            print(f"test {pert_type}: {len(perts)} perturbations")
    return checkpoint
