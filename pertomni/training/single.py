"""Contrastive model of one dataset (PertOmni-chem on Tahoe, PertOmni-cri on CRISPRi, and their baselines)."""

from dataclasses import dataclass
from typing import Optional

import lightning as L

from ..config import field
from ..data.datasets import make_dataset, make_eval_loader, make_train_loader
from ..data.inputs import DATASETS, METHODS, load_expression, load_text_embeddings, text_dim
from ..data.splits import CELL_KEY, split_obs_names
from ..evaluation.retrieval import benchmark, benchmark_intra
from ..io import save_obj
from ..models.clip import CLIPModel
from ..models.losses import LOSSES
from .common import device, export_embeddings, fit, make_trainer


@dataclass
class SingleConfig:
    dataset: str = field(help="tahoe (drugs) or crispr (CRISPRi)", choices=DATASETS)
    split_mode: str = field(
        help="target, target_seed{n}, cv_target, cell_type (with --fold), pair, pair_all5, pair_sim (crispr)"
    )
    fold: Optional[int] = field(None, help="cross-validation fold of the cv_target / cell_type splits")
    method: str = field("pseudobulk", help="expression input; baseline = random text embeddings", choices=METHODS)
    losstype: str = field("clip", choices=LOSSES)
    temp: list[float] = field([1.0], help="temperature, or [expr->text, text->expr] with --diff_temp")
    diff_temp: bool = field(False, help="a separate temperature for each retrieval direction")
    proj_dim: int = field(128, help="embedding size")
    cellline_mask: Optional[str] = field(
        None, help="hard: only samples of the same cell line are negatives", choices=["hard"]
    )
    cells_per_batch: Optional[int] = field(
        None, help="build each training batch from ~this many cell lines (default: random batches)"
    )
    ablate: Optional[str] = field(
        None,
        help="drop the perturbation (pert), cell line (ct) or dose (dosage, tahoe) from the text input",
        choices=["pert", "ct", "dosage"],
    )
    lr: float = 1e-4
    min_epochs: int = 100
    seed: int = 0
    num_workers: int = 3

    def __post_init__(self):
        if self.dataset == "crispr" and self.ablate == "dosage":
            raise ValueError("the CRISPRi data have no dose to ablate")


def train_single(config, run_dir, checkpoints=None, resume_ckpt=None):
    """Train (or, with checkpoints=(best, last), only evaluate) and write embeddings and scores to run_dir.

    Exports {bench}_gene_embeds.pkl / {bench}_text_embeds.pkl (expression / text embeddings of the test rows) and
    {bench}_scores.pkl / {bench}_fullscores.pkl for bench = "test" (best checkpoint) and "last", plus the
    within-cell-line scores of the best checkpoint, intra_test_scores.pkl.
    """
    c = config
    L.seed_everything(c.seed)
    adata = load_expression(c.dataset, c.method)
    embeddings = load_text_embeddings(c.dataset, c.method, c.seed)
    train_ids, valid_ids, test_ids = split_obs_names(adata, c.dataset, c.split_mode, c.fold)
    train_set, valid_set, test_set = (
        make_dataset(c.dataset, adata[ids], embeddings) for ids in (train_ids, valid_ids, test_ids)
    )
    test_loader = make_eval_loader(test_set, c.num_workers)

    model_kwargs = dict(
        text_embedding=text_dim(c.dataset, embeddings, c.ablate),
        image_embedding=adata.shape[1],
        projection_dim=c.proj_dim,
        temperature=c.temp,
        diff_temp=c.diff_temp,
        lr=c.lr,
        losstype=c.losstype,
        dataset=c.dataset,
        cellline_mask=c.cellline_mask,
        ablate=c.ablate,
    )
    if checkpoints is None:
        model = CLIPModel(**model_kwargs)
        train_loader = make_train_loader(train_set, c.num_workers, CELL_KEY[c.dataset], c.cells_per_batch, c.seed)
        trainer = make_trainer(run_dir, c.min_epochs)
        checkpoints = fit(
            trainer, model, train_loader, make_eval_loader(valid_set, c.num_workers), run_dir, resume_ckpt
        )

    best, last = checkpoints
    for bench, ckpt in (("last", last), ("test", best)):  # "test" = best validation checkpoint
        model = CLIPModel.load_from_checkpoint(ckpt, map_location=device(), **model_kwargs)
        expr_emb, text_emb = export_embeddings(model, test_loader)[c.dataset]
        save_obj(expr_emb, run_dir / f"{bench}_gene_embeds.pkl")
        save_obj(text_emb, run_dir / f"{bench}_text_embeds.pkl")
        scores, full_scores = benchmark(expr_emb, text_emb)
        save_obj(scores, run_dir / f"{bench}_scores.pkl")
        save_obj(full_scores, run_dir / f"{bench}_fullscores.pkl")

    obs = test_set.df
    _, intra_gene = benchmark_intra(obs, expr_emb, text_emb, CELL_KEY[c.dataset], side="gene")
    _, intra_text = benchmark_intra(obs, text_emb, expr_emb, CELL_KEY[c.dataset], side="text")
    save_obj(intra_gene | intra_text, run_dir / "intra_test_scores.pkl")
    return best, last
