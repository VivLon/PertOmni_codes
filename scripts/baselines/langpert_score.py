"""Score the LangPert passes of langpert_run.py with the metrics of predict_perturbation_split.py
(pertomni.evaluation.perturbation.score). Runs in the LangPert environment (the results pickle LangPert objects).

Top-20 DEGs: test = the Wilcoxon DEGs of find_test_degs.py (--deg_dir), val = the 20 genes with the largest
|true - ctrl|. Writes per_perturbation.csv and mean_scores.csv to --langpert_dir and the mean table as
langpert_scores.csv to --out_dir, next to the scores of predict_perturbation_split.py.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pertomni import paths
from pertomni.evaluation.perturbation import score
from pertomni.io import load_obj

METRICS = ["corr", "corr_delta", "corr_top20", "corr_delta_top20"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--langpert_dir", type=Path, default=paths.RESULTS / "langpert", help="langpert_run.py --out_dir"
    )
    parser.add_argument("--deg_dir", type=Path, default=paths.DATA / "langpert" / "deg_split_test")
    parser.add_argument("--out_dir", type=Path, default=paths.RESULTS / "split_masked")
    args = parser.parse_args()

    inputs = load_obj(args.langpert_dir / "inputs.pkl")
    var = pd.Index(inputs["var_names"])
    rows = []
    for path in sorted(args.langpert_dir.glob("*/*/*/llm_response_results.pkl")):
        if not (path.parent / "expr_results.pkl").exists():
            continue
        phase, variant, cl = path.relative_to(args.langpert_dir).parts[:3]
        inp, results = inputs[(phase, cl)], load_obj(path)
        queries = inp["queries"]
        pred = np.stack([results[g].prediction for g in queries])
        true = np.stack([inp["true"][g] for g in queries])
        deg = None
        if phase == "test":
            table = pd.read_csv(args.deg_dir / f"{cl}.csv", index_col=0)
            degs = table.groupby("group")["names"].apply(list).to_dict()
            deg = np.stack([var.get_indexer(degs[g]) for g in queries])
            assert (deg >= 0).all()
        non_ctrl = np.tile(inp["fallback_mean"], (len(queries), 1))
        for method, scores, n_knn in (
            (
                f"LangPert {variant}",
                score(pred, true, inp["ctrl"], 20, deg),
                [len(results[g].knn_genes) for g in queries],
            ),
            ("non-ctrl mean", score(non_ctrl, true, inp["ctrl"], 20, deg), 0),
        ):
            rows.append(pd.DataFrame(scores).assign(phase=phase, cl=cl, method=method, target=queries, n_knn=n_knn))
    per_pert = pd.concat(rows).drop_duplicates(["phase", "cl", "method", "target"])
    per_pert.to_csv(args.langpert_dir / "per_perturbation.csv", index=False)
    mean = per_pert.groupby(["phase", "method", "cl"])[METRICS].mean().reset_index()
    mean.to_csv(args.langpert_dir / "mean_scores.csv", index=False)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mean.to_csv(args.out_dir / "langpert_scores.csv", index=False)
    table = mean.pivot_table(index=["phase", "method"], columns="cl", values=["corr_delta", "corr_delta_top20"])
    with pd.option_context("display.width", 200):
        print(table.round(4))


if __name__ == "__main__":
    main()
