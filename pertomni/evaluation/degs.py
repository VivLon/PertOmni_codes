"""Top DEGs of each test perturbation from the raw single-cell Perturb-seq data (rpe1: Replogle 2022;
hepg2 / jurkat: GSE264667), used by the *_top20 perturbation-prediction metrics."""

import anndata
import numpy as np
import pandas as pd
import scanpy as sc

from .. import paths

RAW = {
    "rpe1": "rpe1_raw_singlecell_01.h5ad",
    "hepg2": "GSE264667_hepg2_raw_singlecell_01.h5ad",
    "jurkat": "GSE264667_jurkat_raw_singlecell_01.h5ad",
}
CTRL = "non-targeting"


def load_cells(path, targets, chunk=20000):
    """Cells of the given targets and the non-targeting controls, read chunk by chunk from the backed dense X."""
    a = anndata.read_h5ad(path, backed="r")
    keep = a.obs["gene"].astype(str).isin(targets | {CTRL}).to_numpy()
    blocks = []
    for start in range(0, a.n_obs, chunk):
        m = keep[start : start + chunk]
        if m.any():
            blocks.append(np.asarray(a.X[start : start + chunk])[m])
    obs = a.obs[keep].copy()
    obs["target_gene"] = obs["gene"].astype(str)
    return anndata.AnnData(
        np.vstack(blocks).astype(np.float32), obs=obs[["target_gene", "gem_group"]], var=a.var[["gene_name"]].copy()
    )


def find_degs(cell_line, targets, genes, n_genes=20, chunk=20000):
    """Wilcoxon DEGs of each target against the controls, ranked by |score|.

    normalize_total + log1p on all measured genes; Ensembl ids are mapped to symbols (duplicated symbols, HSPA14,
    are averaged) and restricted to `genes` (the pseudobulk genes). Returns (rank_genes_groups DataFrame, uns dict).
    """
    adata = load_cells(paths.SCRATCH / "arc_challenge" / RAW[cell_line], targets, chunk)
    print(cell_line, "cells", adata.shape, "test perturbations", adata.obs["target_gene"].nunique() - 1, flush=True)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    names = adata.var["gene_name"].astype(str).to_numpy()
    X = pd.DataFrame(adata.X, columns=names)
    if pd.Index(names).duplicated().any():
        X = X.T.groupby(level=0).mean().T
    adata = anndata.AnnData(X[list(genes)].to_numpy(np.float32), obs=adata.obs, var=pd.DataFrame(index=genes))
    groups = sorted(set(adata.obs["target_gene"]) - {CTRL})
    sc.tl.rank_genes_groups(
        adata, groupby="target_gene", groups=groups, reference=CTRL, method="wilcoxon", rankby_abs=True, n_genes=n_genes
    )
    return sc.get.rank_genes_groups_df(adata, group=None), adata.uns["rank_genes_groups"]
