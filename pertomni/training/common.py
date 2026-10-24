"""Run folders, Lightning trainers and embedding export shared by the training entry points.

A run writes everything to {results_dir}/{save_name}/:
  config.json             the run configuration, package version and git commit
  checkpoints/            best.ckpt (best validation metric) and last.ckpt (last epoch)
  logs/                   Lightning CSV logs (metrics.csv)
  train.json              checkpoint paths and training time
  *_embeds.pkl, *_scores.pkl  exported embeddings and scores
"""

import dataclasses
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from .. import __version__
from ..io import save_json


def git_commit():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def prepare_run_dir(results_dir, save_name, config, overwrite=False, existing_ok=False):
    """Create {results_dir}/{save_name}/ and save config.json (config: a config dataclass or a dict).

    A folder that already holds a run is only reused with overwrite=True (its checkpoints and logs are deleted) or
    existing_ok=True (evaluation or resumed training of that run; its config.json is kept).
    """
    run_dir = Path(results_dir) / save_name
    config_path = run_dir / "config.json"
    if config_path.exists():
        if existing_ok:
            return run_dir
        if not overwrite:
            raise FileExistsError(f"{run_dir} already holds a run; choose another --save_name or pass --overwrite")
        for sub in ("checkpoints", "logs"):
            shutil.rmtree(run_dir / sub, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "config": config if isinstance(config, dict) else dataclasses.asdict(config),
            "pertomni_version": __version__,
            "git_commit": git_commit(),
            "created": datetime.now().isoformat(timespec="seconds"),
        },
        config_path,
    )
    return run_dir


def checkpoint_paths(run_dir):
    """(best, last) checkpoints written by make_trainer."""
    return Path(run_dir) / "checkpoints" / "best.ckpt", Path(run_dir) / "checkpoints" / "last.ckpt"


def make_trainer(
    run_dir,
    min_epochs,
    max_epochs=500,
    monitor="val_acc@1",
    mode="max",
    patience=10,
    keep_last=True,
    num_sanity_val_steps=0,
    name="",
):
    """Trainer with early stopping on `monitor` and checkpoints in {run_dir}/checkpoints[/name]/{best,last}.ckpt."""
    ckpt_dir = Path(run_dir) / "checkpoints" / name
    best = ModelCheckpoint(
        dirpath=ckpt_dir, filename="best", monitor=monitor, mode=mode, save_top_k=1, enable_version_counter=False
    )
    callbacks = [EarlyStopping(monitor=monitor, mode=mode, patience=patience, min_delta=1e-4), best]
    if keep_last:
        callbacks.append(
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="last",
                monitor="step",
                mode="max",
                save_top_k=1,
                enable_version_counter=False,
            )
        )
    trainer = L.Trainer(
        callbacks=callbacks,
        logger=CSVLogger(run_dir, name="logs", version=name or None),
        default_root_dir=run_dir,
        num_sanity_val_steps=num_sanity_val_steps,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
    )
    return trainer


def fit(trainer, model, train_loaders, val_loader, run_dir, ckpt_path=None):
    """trainer.fit, then record the checkpoint paths and the training time in train.json; returns (best, last)."""
    start = time.perf_counter()
    trainer.fit(model, train_loaders, val_loader, ckpt_path=ckpt_path)
    best, last = checkpoint_paths(run_dir)
    best_score = trainer.checkpoint_callback.best_model_score
    save_json(
        {
            "best_checkpoint": best,
            "last_checkpoint": last,
            "best_score": None if best_score is None else float(best_score),
            "epochs": trainer.current_epoch,
            "train_seconds": time.perf_counter() - start,
        },
        Path(run_dir) / "train.json",
    )
    return best, last


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def export_embeddings(model, loader):
    """{dataset: (expression embeddings, text embeddings)} of every batch, in loader order.

    The dataset of a batch is "tahoe" if it holds a drug, else "crispr".
    """
    model.eval()
    out = {}
    for batch in loader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        expr_emb, text_emb = model.embed(batch)
        parts = out.setdefault("tahoe" if "drug" in batch else "crispr", ([], []))
        parts[0].append(expr_emb.cpu())
        parts[1].append(text_emb.cpu())
    return {name: (torch.cat(expr), torch.cat(text)) for name, (expr, text) in out.items()}
