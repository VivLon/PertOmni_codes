"""Pre-define the cell-type cross-validation folds (run once, before any cell_type experiment).

Tahoe: the cell lines shuffled into 5 folds (seed 0); CRISPRi: leave one cell type out.
"""

import anndata

from pertomni import paths
from pertomni.data.splits import CRISPR_CELLTYPE_FOLDS, celltype_folds, make_celltype_cv


def main():
    obs = anndata.read_h5ad(paths.TAHOE_PSEUDOBULK, backed="r").obs
    make_celltype_cv(celltype_folds(obs["cell_name"].astype(str), n_folds=5, seed=0), "tahoe")
    obs = anndata.read_h5ad(paths.CRISPR_PSEUDOBULK, backed="r").obs
    assert set(obs["cell_type"].astype(str)) == {c for fold in CRISPR_CELLTYPE_FOLDS for c in fold}
    make_celltype_cv(CRISPR_CELLTYPE_FOLDS, "crispr")


if __name__ == "__main__":
    main()
