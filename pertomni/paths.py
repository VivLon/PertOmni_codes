"""Locations of the inputs and of the run folders, relative to the repository.

  data/     processed datasets, text embeddings, id maps and splits      (PERTOMNI_DATA)
  scratch/  large raw / intermediate files: raw single cells, padded
            PaSCient inputs, cell painting embeddings                   (PERTOMNI_SCRATCH)
  results/  run folders, one per --save_name                            (PERTOMNI_RESULTS)
Each folder may be a symlink, or be moved with the environment variable in brackets.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("PERTOMNI_DATA", ROOT / "data"))
SCRATCH = Path(os.environ.get("PERTOMNI_SCRATCH", ROOT / "scratch"))
RESULTS = Path(os.environ.get("PERTOMNI_RESULTS", ROOT / "results"))

# Tahoe-100M (chemical perturbations)
TAHOE = DATA / "tahoe_100m"  # splits, text embeddings
TAHOE_PSEUDOBULK = DATA / "tahoe_adata.h5ad"  # make_pseudobulk.py --dataset tahoe

# CRISPRi Perturb-seq (Replogle 2022 + GSE264667, from the Arc Virtual Cell Challenge)
CRISPR = DATA / "arc_challenge"  # text embeddings, pathway annotations
CRISPR_PUBLIC = CRISPR / "public"  # splits
CRISPR_PSEUDOBULK = DATA / "crispr_adata.h5ad"  # make_pseudobulk.py --dataset crispr

MULTI = DATA / "multi_dataset"  # drug / gene id maps for the drug-gene interaction (DGI) loss
CELLTYPE_CV = DATA / "cv_celltype"  # cell-type cross-validation folds
TARGET_CV = DATA / "langpert" / "cv_data"  # perturbation cross-validation folds
CELLPAINTING = DATA / "cellclip_data"  # RxRx3-core text embeddings and well -> perturbation map

GENEPT = DATA / "GenePT"  # GenePT gene embeddings (GenePT_emebdding_v2)
