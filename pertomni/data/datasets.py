"""PyTorch datasets and data loaders of the Tahoe and CRISPRi samples."""

import ast
from itertools import chain

import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse
from torch.utils.data import DataLoader, Dataset, IterableDataset, Sampler

BATCH_SIZE = 128


class TahoeDataset(Dataset):
    """One sample per (cell line, drug, dose): expression x, cell line / drug text embeddings and the dose (uM)."""

    def __init__(self, adata, emb_drug, emb_cell, id_drug=None):
        self.adata = adata
        self.df = adata.obs.copy()
        self.emb_drug = emb_drug
        self.emb_cell = emb_cell
        self.id_drug = id_drug  # drug -> id(s), only needed by the DGI loss

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        drug, dose = ast.literal_eval(row["drugname_drugconc"])[0][:2]
        drug_id = -1 if self.id_drug is None else self.id_drug[row["drug"]]
        return {
            "x": torch.FloatTensor(self.adata[idx].X),
            "exp": torch.FloatTensor([dose]),
            "cell": torch.as_tensor(self.emb_cell[row["cell_name"]], dtype=torch.float32),
            "drug": torch.as_tensor(self.emb_drug[drug], dtype=torch.float32),
            "id": torch.tensor(drug_id, dtype=torch.long),
        }


class CrisprDataset(Dataset):
    """One sample per (cell type, target gene): expression x, cell type text embedding and the target gene's text
    embedding concatenated with its pathway embedding ("kd_target")."""

    def __init__(self, adata, emb_gene, emb_cell, emb_path, id_gene=None):
        self.adata = adata
        self.df = adata.obs.copy()
        self.emb_gene = emb_gene
        self.emb_cell = emb_cell
        self.emb_path = emb_path
        self.id_gene = id_gene  # gene -> id(s), only needed by the DGI loss

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = self.adata[idx].X
        if issparse(x):
            x = np.array(x.todense())
        gene_id = -1 if self.id_gene is None else self.id_gene[row["target_gene"]]
        kd_target = np.concatenate((self.emb_gene[row["target_gene"]], self.emb_path[row["target_pathway"]]), axis=0)
        return {
            "x": torch.FloatTensor(x),
            "cell": torch.as_tensor(self.emb_cell[row["cell_type"]], dtype=torch.float32),
            "kd_target": torch.as_tensor(kd_target, dtype=torch.float32),
            "id": torch.tensor(gene_id, dtype=torch.long),
        }


def make_dataset(dataset, adata, embeddings, ids=None):
    if dataset == "tahoe":
        return TahoeDataset(adata, embeddings["pert"], embeddings["cell"], ids)
    return CrisprDataset(adata, embeddings["pert"], embeddings["cell"], embeddings["pathway"], ids)


class CellLineBatchSampler(Sampler):
    """Each batch holds ~cells_per_batch cell lines (batch_size // cells_per_batch samples each), so a within-cell-line
    mask keeps enough negatives per anchor. Every sample is used once per epoch; batches are reshuffled every epoch."""

    def __init__(self, cell_labels, batch_size=BATCH_SIZE, cells_per_batch=2, seed=0):
        labels = pd.Series(np.asarray(cell_labels))
        self.groups = [np.asarray(idx) for idx in labels.groupby(labels).indices.values()]
        self.batch_size = batch_size
        self.chunk = max(1, batch_size // cells_per_batch)
        self.rng = np.random.default_rng(seed)
        self.n_batches = int(np.ceil(len(labels) / batch_size))

    def __iter__(self):
        chunks = [
            chunk
            for idx in self.groups
            for chunk in np.array_split(self.rng.permutation(idx), int(np.ceil(len(idx) / self.chunk)))
        ]
        order = self.rng.permutation(len(chunks))
        flat = np.concatenate([chunks[i] for i in order])  # consecutive single-cell-line chunks form a batch
        for start in range(0, len(flat), self.batch_size):
            yield flat[start : start + self.batch_size].tolist()

    def __len__(self):
        return self.n_batches


def _worker_kwargs(num_workers, persistent_workers):
    if num_workers == 0:
        return {"num_workers": 0}
    return {
        "num_workers": num_workers,
        "prefetch_factor": max(num_workers - 1, 1),
        "persistent_workers": persistent_workers,
    }


def make_train_loader(dataset, num_workers, cell_key=None, cells_per_batch=None, seed=0, persistent_workers=False):
    """Shuffled training batches; with cells_per_batch, batches grouped by cell line (CellLineBatchSampler)."""
    workers = _worker_kwargs(num_workers, persistent_workers)
    if cells_per_batch is None:
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, **workers)
    sampler = CellLineBatchSampler(dataset.df[cell_key].values, BATCH_SIZE, cells_per_batch, seed)
    return DataLoader(dataset, batch_sampler=sampler, **workers)


def make_eval_loader(dataset, num_workers, persistent_workers=False):
    return DataLoader(dataset, batch_size=BATCH_SIZE, **_worker_kwargs(num_workers, persistent_workers))


class SequentialIterableDataset(IterableDataset):
    """Iterates over several data loaders one after the other (e.g. all Tahoe, then all CRISPRi batches)."""

    def __init__(self, loaders):
        super().__init__()
        self.loaders = loaders

    def __iter__(self):
        return chain(*self.loaders)


def sequential_loader(loaders):
    return DataLoader(SequentialIterableDataset(loaders), batch_size=None)
