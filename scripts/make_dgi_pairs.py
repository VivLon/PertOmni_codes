"""Labelled (drug, gene) pairs of a drug-gene interaction task: {train,valid,test}_drug_gene_pairs.pkl in the task
folder of pertomni.data.dgi.RETRIEVAL_DIRS."""

import argparse

from pertomni.data.dgi import RETRIEVAL_DIRS, make_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", default="drug2gene", choices=list(RETRIEVAL_DIRS))
    parser.add_argument("--neg_ratio", default=1, type=int, help="negative pairs per positive pair (train / valid)")
    parser.add_argument(
        "--neg_gene",
        action="store_true",
        help="sample negative genes for each drug instead of negative drugs for each gene",
    )
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    make_pairs(args.retrieval, args.neg_ratio, neg_drug=not args.neg_gene, seed=args.seed)


if __name__ == "__main__":
    main()
