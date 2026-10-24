"""Random-matching baseline of text-transcriptome retrieval: within each cell line, every test sample ranks the text
embeddings of the cell line's test samples in random order.

Scores have the keys of the models' intra_test_scores.pkl (gene_* = expression -> text, text_* = text -> expression)
and are averaged over cell lines weighted by their number of test samples. Writes {dataset}_intra_scores_seed{seed}.pkl
to {results_dir}/{save_name}/.
"""

import argparse
import collections
from pathlib import Path

import anndata
import numpy as np
import pandas as pd

from pertomni import paths
from pertomni.data.inputs import DATASETS, expression_path
from pertomni.data.splits import CELL_KEY, PERT_KEY, load_target_split
from pertomni.evaluation.retrieval import eval_acc, eval_mrr, eval_ndcg
from pertomni.io import save_obj
from pertomni.training.common import prepare_run_dir


def random_matching(obs, group_key, seed, side, ks=(1, 5, 10)):
    """Within-group scores of a random ranking, averaged over groups weighted by their size."""
    rng = np.random.default_rng(seed)
    groups = collections.defaultdict(list)
    for i, g in enumerate(obs[group_key]):
        groups[g].append(i)
    table = {}
    for grp, idxs in groups.items():
        n = len(idxs)
        ranks = np.stack([rng.permutation(n) for _ in range(n)])
        true = np.arange(n)
        acc, ndcg = eval_acc(n, true, ranks, max_k=10), eval_ndcg(n, true, ranks, max_k=10)
        table[grp] = {f"{side}_{m}@{k}": v[k - 1] for k in ks for m, v in (("acc", acc), ("ndcg", ndcg))}
        table[grp][f"{side}_mrr"] = eval_mrr(true, ranks)
    df = pd.DataFrame(table)
    return {key: sum(len(idxs) * df.loc[key, grp] for grp, idxs in groups.items()) / len(obs) for key in df.index}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save_name", default="random_matching")
    parser.add_argument("--results_dir", type=Path, default=paths.RESULTS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--datasets", default=list(DATASETS), nargs="+", choices=DATASETS)
    parser.add_argument("--seeds", default=[0, 1, 2, 3, 4], type=int, nargs="+")
    args = parser.parse_args()

    out_dir = prepare_run_dir(args.results_dir, args.save_name, vars(args), args.overwrite)
    for ds in args.datasets:
        obs = anndata.read_h5ad(expression_path(ds), backed="r").obs
        test_obs = obs[obs[PERT_KEY[ds]].isin(load_target_split(ds)[2])]
        for seed in args.seeds:
            scores = {
                key: value
                for side in ("gene", "text")
                for key, value in random_matching(test_obs, CELL_KEY[ds], seed, side).items()
            }
            save_obj(scores, out_dir / f"{ds}_intra_scores_seed{seed}.pkl")
            print(ds, f"seed {seed}:", len(test_obs), "test samples, gene_acc@1", round(scores["gene_acc@1"], 4))


if __name__ == "__main__":
    main()
