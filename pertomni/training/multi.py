"""PertOmni: joint contrastive training on Tahoe and CRISPRi, and its drug-gene interaction (DGI) fine-tuning."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import lightning as L
import numpy as np
import pandas as pd
import torch

from ..config import field
from ..data.datasets import make_dataset, make_eval_loader, make_train_loader, sequential_loader
from ..data.inputs import METHODS, load_expression, load_id_maps, load_text_embeddings, text_dim
from ..data.splits import CELL_KEY, FOLD_SPLITS, split_obs_names
from ..evaluation.retrieval import benchmark, benchmark_intra
from ..io import save_obj
from ..models.multi_clip import DATASETS, DGI_LOSSES, MultiCLIPModel
from .common import checkpoint_paths, device, export_embeddings, fit, make_trainer


@dataclass
class MultiConfig:
    split_mode: str = field(help="target, target_seed{n}, cv_target / cell_type (with --fold), pair, pair_{suffix}")
    fold: Optional[int] = field(None, help="cross-validation fold of the cv_target / cell_type splits")
    method: str = field("pseudobulk", help="expression input; baseline = random text embeddings", choices=METHODS)
    losstype: str = field("clip", choices=["siglip", "clip", "clip_reg"])
    temp_tahoe: list[float] = field([1.0], help="Tahoe temperature, or [expr->text, text->expr] with --diff_temp")
    temp_crispr: list[float] = field([1.0], help="CRISPRi temperature, or [expr->text, text->expr] with --diff_temp")
    diff_temp: bool = field(False, help="a separate temperature for each retrieval direction")
    hardmask: Optional[str] = field(
        None, help="both: only samples of the same cell line are negatives", choices=["both"]
    )
    mask_norm: bool = field(False, help="with --hardmask: average the loss over the unmasked pairs")
    drug_target_loss: Optional[str] = field(
        None, help="add / fine-tune on a loss over interacting drug-gene pairs", choices=DGI_LOSSES
    )
    ft_retrieval: Optional[str] = field(
        None,
        help="fine-tune --init_from on the DGI loss; text: freeze the expression encoder, all: train every layer",
        choices=["text", "all"],
    )
    init_from: Optional[str] = field(
        None, help="pretrained run folder (its checkpoints/best.ckpt) or checkpoint file for --ft_retrieval"
    )
    ablate: Optional[str] = field(
        None,
        help="drop the perturbation (pert), cell line (ct) or dose (dosage, Tahoe) from the text input",
        choices=["pert", "ct", "dosage"],
    )
    export_train: bool = field(False, help="also export the train + valid rows ({bench}_trainside_*) as kNN references")
    export_full: bool = field(False, help="also export every row of both datasets (full_*) with the best checkpoint")
    lr: float = 1e-4
    min_epochs: int = 100
    seed: int = 0
    num_workers: int = 3

    def __post_init__(self):
        if (self.fold is not None) != (self.split_mode in FOLD_SPLITS):
            raise ValueError(f"--fold is required by, and only defined for, the {FOLD_SPLITS} splits")
        if (self.ft_retrieval is not None) != (self.init_from is not None):
            raise ValueError("--ft_retrieval and --init_from go together")
        if self.ft_retrieval is not None and self.drug_target_loss is None:
            raise ValueError("--ft_retrieval fine-tunes on --drug_target_loss")


def init_checkpoint(init_from):
    path = Path(init_from)
    return checkpoint_paths(path)[0] if path.is_dir() else path


def load_inputs(c):
    """Expression AnnData and text embeddings of both datasets; pseudobulk inputs are restricted to the shared genes."""
    pseudobulk = c.method in ("pseudobulk", "baseline")
    adatas = {name: load_expression(name, c.method) for name in DATASETS}
    if pseudobulk:
        common_genes = np.intersect1d(adatas["tahoe"].var_names, adatas["crispr"].var_names)
        adatas = {name: adata[:, common_genes] for name, adata in adatas.items()}
    embeddings = {name: load_text_embeddings(name, c.method, c.seed) for name in DATASETS}
    return adatas, embeddings


def train_multi(config, run_dir, checkpoints=None):
    """Train (or, with checkpoints=(best, last), only evaluate) PertOmni and write embeddings and scores to run_dir.

    Embeddings: {prefix}_{tahoe,crispr}_{gene,text}_embeds.pkl (gene = expression), for the checkpoints "test" (best
    validation) and "last":
      default         prefix {bench}: test rows; scores {bench}_{tahoe,crispr}_scores.pkl and intra_test_{tahoe,crispr}_scores.pkl
      --export_train  also prefix {bench}_trainside: train + valid rows, in AnnData order
      --fold          prefix "testing": test rows (best checkpoint), scored as testing_{tahoe,crispr}_scores.pkl;
                      prefix {bench}: train + valid rows, the kNN reference set of perturbation prediction
      --export_full   also prefix "full": every row of both datasets, in AnnData order (best checkpoint)
    """
    c = config
    L.seed_everything(c.seed)
    adatas, embeddings = load_inputs(c)
    drug_id, gene_id = load_id_maps(c.drug_target_loss)
    id_maps = {"tahoe": drug_id, "crispr": gene_id}

    loaders, test_obs = {}, {}
    for name in DATASETS:
        adata, emb = adatas[name], embeddings[name]
        train_ids, valid_ids, test_ids = split_obs_names(adata, name, c.split_mode, c.fold)
        train_set, valid_set, test_set = (
            make_dataset(name, adata[ids], emb, id_maps[name]) for ids in (train_ids, valid_ids, test_ids)
        )
        test_obs[name] = test_set.df
        loaders[name] = {
            "train": make_train_loader(train_set, c.num_workers, persistent_workers=True),
            "valid": make_eval_loader(valid_set, c.num_workers, persistent_workers=True),
            "test": make_eval_loader(test_set, c.num_workers, persistent_workers=True),
        }
        if c.fold is not None or c.export_train:
            train_side = adata.obs_names.isin(pd.Index(train_ids).append(pd.Index(valid_ids)))
            loaders[name]["trainside"] = make_eval_loader(
                make_dataset(name, adata[train_side], emb, id_maps[name]), c.num_workers
            )
        if c.export_full:
            loaders[name]["full"] = make_eval_loader(make_dataset(name, adata, emb, id_maps[name]), c.num_workers)
    combined = {
        split: sequential_loader([loaders[name][split] for name in DATASETS])
        for split in loaders["tahoe"]
        if split != "train"
    }

    model_kwargs = dict(
        text_embedding1=text_dim("tahoe", embeddings["tahoe"], c.ablate),
        text_embedding2=text_dim("crispr", embeddings["crispr"], c.ablate),
        image_embedding=adatas["tahoe"].shape[1],
        temperature_tahoe=c.temp_tahoe,
        temperature_crispr=c.temp_crispr,
        diff_temp=c.diff_temp,
        lr=c.lr,
        losstype=c.losstype,
        drug_target_loss=c.drug_target_loss,
        hardmask=c.hardmask,
        mask_norm=c.mask_norm,
        ft_retrieval=c.ft_retrieval is not None,
        ablate=c.ablate,
    )
    if checkpoints is None:
        model = MultiCLIPModel(**model_kwargs)
        if c.ft_retrieval is not None:
            state = torch.load(init_checkpoint(c.init_from), map_location="cpu")["state_dict"]
            print("initialised from", c.init_from, model.load_state_dict(state, strict=False))
            if c.ft_retrieval == "text":
                for param_name, param in model.named_parameters():
                    if param_name.startswith("image_projection"):
                        param.requires_grad = False
        trainer = make_trainer(run_dir, c.min_epochs)
        train_loaders = [loaders[name]["train"] for name in DATASETS]
        checkpoints = fit(trainer, model, train_loaders, combined["valid"], run_dir)

    best, last = checkpoints
    exported = {}
    for bench, ckpt in (("last", last), ("test", best)):  # "test" = best validation checkpoint
        model = MultiCLIPModel.load_from_checkpoint(ckpt, map_location=device(), **model_kwargs)
        if c.fold is not None:
            exports = {bench: "trainside"} | ({"testing": "test"} if bench == "test" else {})
        else:
            exports = {bench: "test"} | ({f"{bench}_trainside": "trainside"} if c.export_train else {})
        if c.export_full and bench == "test":
            exports["full"] = "full"
        for prefix, split in exports.items():
            exported[prefix] = export_embeddings(model, combined[split])
            for name, (expr_emb, text_emb) in exported[prefix].items():
                save_obj(expr_emb, run_dir / f"{prefix}_{name}_gene_embeds.pkl")
                save_obj(text_emb, run_dir / f"{prefix}_{name}_text_embeds.pkl")

    # fold runs: "{bench}_*" hold the train-side rows (too many for a full ranking), so score the held-out "testing_*"
    for bench in ["last", "test"] if c.fold is None else ["testing"]:
        for name, (expr_emb, text_emb) in exported[bench].items():
            scores, full_scores = benchmark(expr_emb, text_emb)
            save_obj(scores, run_dir / f"{bench}_{name}_scores.pkl")
            save_obj(full_scores, run_dir / f"{bench}_{name}_fullscores.pkl")
            if bench == "test":
                _, intra_gene = benchmark_intra(test_obs[name], expr_emb, text_emb, CELL_KEY[name], side="gene")
                _, intra_text = benchmark_intra(test_obs[name], text_emb, expr_emb, CELL_KEY[name], side="text")
                save_obj(intra_gene | intra_text, run_dir / f"intra_test_{name}_scores.pkl")
    return best, last
