"""Per-pair inputs of the SNN + VISION baseline of drug-gene interaction: each pair's drug (gene) gets the mean VISION
signature scores (vision_scores.R) of all its Tahoe (CRISPRi) samples (pertomni.data.dgi.pair_features).

Writes {split}_{tahoe,crispr}_emb.pkl to --out_dir, the --emb_dir of train_dgi.py (with --use_emb all).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pertomni import paths
from pertomni.data.dgi import RETRIEVAL_DIRS, pair_features
from pertomni.data.inputs import DATASETS
from pertomni.data.splits import PERT_KEY


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vision_dir", type=Path, default=paths.RESULTS / "vision", help="output of vision_scores.R")
    parser.add_argument("--retrieval", default="drug2gene", choices=list(RETRIEVAL_DIRS))
    parser.add_argument("--out_dir", type=Path, help="default: {vision_dir}/dgi_{retrieval}/")
    args = parser.parse_args()

    features = {}
    for ds in DATASETS:
        labels = pd.read_csv(args.vision_dir / f"{ds}_metadata.csv", index_col=0)[PERT_KEY[ds]].to_numpy()
        scores = pd.read_csv(args.vision_dir / f"{ds}_raw_scores.csv", index_col=0).to_numpy(np.float32)
        features[ds] = (labels, scores)
    pair_features(features, args.retrieval, args.out_dir or args.vision_dir / f"dgi_{args.retrieval}")


if __name__ == "__main__":
    main()
