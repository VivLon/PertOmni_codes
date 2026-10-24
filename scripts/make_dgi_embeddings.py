"""Per-pair inputs of the drug-gene interaction classifier from a PertOmni run trained with --export_full.

Writes {split}_{tahoe,crispr}_emb.pkl (rows aligned with the task's pair files) to --out_dir, by default
{run_dir}/dgi_{retrieval}/.
"""

import argparse
from pathlib import Path

from pertomni.data.dgi import RETRIEVAL_DIRS, pair_embeddings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, type=Path, help="PertOmni run folder holding full_*_embeds.pkl")
    parser.add_argument("--retrieval", default="drug2gene", choices=list(RETRIEVAL_DIRS))
    parser.add_argument("--out_dir", type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir or args.run_dir / f"dgi_{args.retrieval}"
    pair_embeddings(args.run_dir, args.retrieval, out_dir)


if __name__ == "__main__":
    main()
