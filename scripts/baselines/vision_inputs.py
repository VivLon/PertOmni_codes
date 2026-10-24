"""Inputs of vision_scores.R for the drug-gene interaction baselines (tahoe_blog.py, vision_pair_inputs.py).

For the genes shared by both datasets, writes {tahoe,crispr}_expr.txt.gz (pseudobulk counts, genes x samples:
layers["counts"] of make_pseudobulk.py) and {tahoe,crispr}_metadata.csv (cell line and perturbation of each sample,
same order) to --out_dir.
"""

import argparse
from pathlib import Path

import anndata
import numpy as np
import pandas as pd

from pertomni import paths
from pertomni.data.inputs import DATASETS, expression_path
from pertomni.data.splits import CELL_KEY, PERT_KEY


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", type=Path, default=paths.RESULTS / "vision")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    adatas = {ds: anndata.read_h5ad(expression_path(ds)) for ds in DATASETS}
    genes = np.intersect1d(*(adata.var_names for adata in adatas.values()))
    for ds, adata in adatas.items():
        X = adata[:, genes].layers["counts"]
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        expr = pd.DataFrame(X.T, index=genes, columns=adata.obs_names)
        expr.to_csv(args.out_dir / f"{ds}_expr.txt.gz", sep="\t", compression="gzip")
        adata.obs[[CELL_KEY[ds], PERT_KEY[ds]]].to_csv(args.out_dir / f"{ds}_metadata.csv")
        print(ds, adata.n_obs, "samples,", len(genes), "genes", flush=True)


if __name__ == "__main__":
    main()
