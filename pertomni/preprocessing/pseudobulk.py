"""Pseudobulk expression: one row per (perturbation, cell line) sample, the summed counts of its cells."""

import anndata
import numpy as np
import scanpy as sc
from scipy.sparse import issparse


def pseudobulk(adata, keys):
    """Sum the counts of the cells of each sample (unique combination of obs[keys]).

    Rows follow the first appearance of each sample and keep the obs of its first cell; X stays raw counts.
    """
    keys = list(keys)
    obs = adata.obs.drop_duplicates(keys)
    rows = adata.obs.groupby(keys, observed=True, sort=False).indices
    X = np.zeros((len(obs), adata.n_vars), dtype=np.float64)
    for i, sample in enumerate(obs[keys].itertuples(index=False, name=None)):
        block = adata.X[rows[sample if len(keys) > 1 else sample[0]]]
        X[i] = np.asarray(block.sum(axis=0)).ravel() if issparse(block) else block.sum(axis=0)
    return anndata.AnnData(X, obs=obs.copy(), var=adata.var.copy())


def tahoe_plate_pseudobulk(path):
    """Pseudobulk of one Tahoe-100M plate file: samples = (drugname_drugconc, cell_name)."""
    return pseudobulk(sc.read_h5ad(path), ["drugname_drugconc", "cell_name"])


def crispr_pseudobulk(path, cell_type, min_genes=100, min_cells=3):
    """Pseudobulk of one CRISPRi Perturb-seq file: cells with >= min_genes genes and genes in >= min_cells cells are
    kept, samples = (target_gene, cell_type).

    The target gene is obs["gene"] (Replogle et al.) or obs["target_gene"] (HCT116 of the Virtual Cell Challenge);
    the cell line is always set to cell_type, replacing any cell line column of the file.
    """
    adata = sc.read_h5ad(path)
    if "gene" in adata.obs:
        target = adata.obs.pop("gene")
    elif "target_gene" in adata.obs:
        target = adata.obs.pop("target_gene")
    else:
        raise KeyError(f"{path}: no target gene column (gene or target_gene) in obs")
    adata.obs["target_gene"] = target.astype(str).replace("Non-Targeting", "non-targeting")
    adata.obs["cell_type"] = cell_type
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    return pseudobulk(adata, ["target_gene", "cell_type"])


def normalize(adata):
    """Keep the counts in layers["counts"], then normalize_total + log1p (layers["logcounts"])."""
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata.layers["logcounts"] = adata.X.copy()
    return adata


def keep_embedded_targets(adata, embeddings, key="target_gene", control="non-targeting"):
    """Drop samples whose perturbation has no text embedding (the control is always kept)."""
    keep = set(embeddings) | {control}
    return adata[adata.obs[key].isin(keep)].copy()


def concat(adatas):
    """Concatenate pseudobulk parts (plates, cell lines) on their shared genes, with unique row names."""
    adata = anndata.concat(adatas)
    adata.obs_names_make_unique()
    return adata
