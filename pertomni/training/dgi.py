"""Training of the drug-gene interaction classifier (models.dgi.TargetModel)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .. import paths
from ..config import field
from ..data.dgi import RETRIEVAL_DIRS, DrugGenePairDataset, paper_inputs, run_inputs
from ..io import load_obj, save_obj
from ..models.dgi import TargetModel
from .common import device, make_trainer

EMBEDDINGS = ("gene", "text", "all")  # expression half, text half or both halves of the [expression, text] embedding
ES_MODES = {"es_loss": ("val_loss", "min"), "es_acc": ("val_acc", "max")}


@dataclass
class DGIConfig:
    retrieval: str = field("drug2gene", help="task, see pertomni.data.dgi", choices=list(RETRIEVAL_DIRS))
    emb_dir: Optional[str] = field(None, help="per-pair embeddings from scripts/make_dgi_embeddings.py")
    pairs_dir: Optional[str] = field(None, help="pair files from scripts/make_dgi_pairs.py (default: the task folder)")
    paper_model: Optional[str] = field(
        None,
        help="instead of --emb_dir: the published inputs of this model "
        "(bothmask_concat_pathway_shared_text_es_acc, vision_scores, ...) and --seed",
    )
    use_emb: list[str] = field(
        list(EMBEDDINGS), help="embedding halves to train on, one classifier each", choices=EMBEDDINGS
    )
    es_modes: list[str] = field(
        list(ES_MODES), help="early stopping on validation loss and / or recall", choices=list(ES_MODES)
    )
    lr: float = 1e-5
    dropout: float = 0.1
    batch_size: int = 32
    hidden_dim: int = 64
    norm: str = field("batch", choices=["batch", "layer"])
    seed: int = 0

    def __post_init__(self):
        if (self.emb_dir is None) == (self.paper_model is None):
            raise ValueError("give either --emb_dir or --paper_model")

    def inputs(self):
        if self.paper_model is not None:
            return paper_inputs(self.retrieval, self.paper_model, self.seed)
        pairs_dir = Path(self.pairs_dir) if self.pairs_dir else paths.DATA / RETRIEVAL_DIRS[self.retrieval]
        return run_inputs(pairs_dir, Path(self.emb_dir))


def load_split(files, use_emb):
    """(drug embeddings, gene embeddings, labels) of one split; files = (pairs, drug embeddings, gene embeddings)."""
    pairs, drug_emb, gene_emb = (load_obj(f) for f in files)
    drug_emb, gene_emb = np.asarray(drug_emb, dtype=np.float32), np.asarray(gene_emb, dtype=np.float32)
    if use_emb == "gene":
        drug_emb, gene_emb = drug_emb[:, :128], gene_emb[:, :128]
    elif use_emb == "text":
        drug_emb, gene_emb = drug_emb[:, 128:], gene_emb[:, 128:]
    return drug_emb, gene_emb, list(pd.DataFrame(pairs)[2])


@torch.no_grad()
def predict(model, loader):
    model.eval()
    logits = torch.cat([model([x.to(model.device) for x in batch])[1].cpu() for batch in loader])
    return logits, (torch.sigmoid(logits) >= 0.5).float()


def train_snn(
    inputs,
    run_dir,
    seed=0,
    use_emb="all",
    es_mode="es_loss",
    lr=1e-5,
    dropout=0.1,
    batch_size=32,
    hidden_dim=64,
    norm="batch",
):
    """Train one classifier and save its validation / test logits and predictions to run_dir:
    snn_{es_mode}_{use_emb}_{logits,pred}.pkl (test) and val_snn_{es_mode}_{use_emb}_{logits,pred}.pkl.

    inputs: {split: (pairs file, drug embedding file, gene embedding file)}, see data.dgi.run_inputs / paper_inputs.
    """
    data = {split: DrugGenePairDataset(*load_split(files, use_emb)) for split, files in inputs.items()}
    loaders = {
        split: DataLoader(dataset, batch_size=batch_size, shuffle=split == "train") for split, dataset in data.items()
    }
    input_dim = data["train"].drug_emb.shape[1]

    L.seed_everything(seed)
    model_kwargs = dict(lr=lr, input_dim=input_dim, dropout=dropout, hidden_dim=hidden_dim, norm=norm)
    model = TargetModel(**model_kwargs)
    name = f"{es_mode}_{use_emb}"
    monitor, mode = ES_MODES[es_mode]
    trainer = make_trainer(
        run_dir,
        min_epochs=100,
        max_epochs=None,
        monitor=monitor,
        mode=mode,
        keep_last=False,
        num_sanity_val_steps=2,
        name=name,
    )
    trainer.fit(model, loaders["train"], loaders["valid"])

    model = TargetModel.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path, map_location=device(), **model_kwargs
    )
    for prefix, split in (("", "test"), ("val_", "valid")):
        logits, pred = predict(model, loaders[split])
        save_obj(logits, run_dir / f"{prefix}snn_{name}_logits.pkl")
        save_obj(pred, run_dir / f"{prefix}snn_{name}_pred.pkl")
