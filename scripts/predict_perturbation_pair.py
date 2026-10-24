"""kNN perturbation prediction on the CRISPRi pair split (held-out (target, cell line) pairs), within each cell line.

  val:  queries = the cell line's valid rows, references = its train rows  -> select K (best mean corr)
  test: queries = the cell line's test rows,  references = its train + valid rows
Embeddings: runs of train_multi.py --split_mode pair --export_train ({bench}_crispr_* = test rows in test_pair.pkl
order, {bench}_trainside_crispr_* = train + valid rows in AnnData order). Test top-20 DEGs: find_test_degs.py --split pair.
"""

import argparse
from pathlib import Path

import anndata
import numpy as np
import pandas as pd

from pertomni import paths
from pertomni.evaluation.perturbation import CTRL, knn_predict, per_target_mean, score
from pertomni.io import load_obj, run_file, save_obj
from pertomni.training.common import prepare_run_dir

PAPER_RUNS = [
    str(paths.DATA / f"multi_dataset/siglip/pair/pseudobulk/export_{m}concat_pathway_shared_text_es_acc/seed{{seed}}")
    for m in ("bothmask_", "")
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save_name", default="pair_masked_vs_unmasked")
    parser.add_argument("--results_dir", type=Path, default=paths.RESULTS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--runs", default=PAPER_RUNS, nargs="+", help="run folder of each model, with a {seed} placeholder"
    )
    parser.add_argument("--labels", default=["masked", "unmasked"], nargs="+", help="name of each model in --runs")
    parser.add_argument("--seeds", default=[0], type=int, nargs="+")
    parser.add_argument("--bench", default="test", help='"test" = best validation checkpoint, "last" = last epoch')
    parser.add_argument(
        "--Ks", default=[1, 3, 5, 8, 10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000], type=int, nargs="+"
    )
    parser.add_argument("--cell_lines", default=["rpe1", "hepg2", "jurkat"], nargs="+")
    parser.add_argument("--temp", default=0.07, type=float)
    parser.add_argument("--n_top_degs", default=20, type=int)
    parser.add_argument("--deg_dir", type=Path, default=paths.DATA / "langpert" / "deg_pair_test")
    return parser.parse_args()


def load_pair_embeddings(run_dir, e, bench, obs_names, split):
    """(n_obs, d) embeddings aligned with crispr_adata.h5ad rows; rows outside train / valid / test stay NaN."""
    test = load_obj(run_file(run_dir, f"{bench}_crispr_{e}_embeds.pkl")).detach().cpu().numpy()
    side = load_obj(run_file(run_dir, f"{bench}_trainside_crispr_{e}_embeds.pkl")).detach().cpu().numpy()
    side_rows = np.where(obs_names.isin(split["train"] | split["valid"]))[0]
    assert test.shape[0] == len(split["test_order"]) and side.shape[0] == len(side_rows), (
        test.shape,
        side.shape,
        len(side_rows),
    )
    out = np.full((len(obs_names), test.shape[1]), np.nan, dtype=np.float32)
    out[obs_names.get_indexer(split["test_order"])] = test
    out[side_rows] = side
    return out


def main():
    args = parse_args()
    assert len(args.runs) == len(args.labels), "one label per run"
    out_dir = prepare_run_dir(args.results_dir, args.save_name, vars(args), args.overwrite)
    order = load_obj(paths.CRISPR_PUBLIC / "test_pair.pkl")
    split = {s: set(load_obj(paths.CRISPR_PUBLIC / f"{s}_pair.pkl")) for s in ["train", "valid", "test"]} | {
        "test_order": list(order)
    }
    adata = anndata.read_h5ad(paths.CRISPR_PSEUDOBULK)
    X = np.asarray(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X, dtype=np.float32)
    obs, names = adata.obs, adata.obs_names
    tg, cls = obs["target_gene"].astype(str).to_numpy(), obs["cell_type"].astype(str).to_numpy()
    in_split = {s: names.isin(split[s]) for s in ["train", "valid", "test"]}
    emb = {
        (label, seed, e): load_pair_embeddings(Path(run.format(seed=seed)), e, args.bench, names, split)
        for run, label in zip(args.runs, args.labels)
        for seed in args.seeds
        for e in ["text", "gene"]
    }

    val_rows, test_rows = [], []
    for cl in args.cell_lines:
        in_cl = cls == cl
        ctrl = X[in_cl & (tg == CTRL)].mean(axis=0)
        for phase, ref_sets, q_set in [("val", ["train"], "valid"), ("test", ["train", "valid"], "test")]:
            ref_rows = in_cl & np.logical_or.reduce([in_split[s] for s in ref_sets])  # includes the non-targeting rows
            q_rows = in_cl & in_split[q_set] & (tg != CTRL)
            q_targets = np.unique(tg[q_rows])
            assert not set(q_targets) & set(tg[ref_rows]), "query target seen in the same cell line's references"
            true = per_target_mean(X[q_rows], tg[q_rows], q_targets)
            deg_idx = None
            if phase == "test":
                degs = (
                    pd.read_csv(args.deg_dir / f"{cl}.csv", index_col=0).groupby("group")["names"].apply(list).to_dict()
                )
                deg_idx = np.stack([adata.var_names.get_indexer(degs[t]) for t in q_targets])
                assert (deg_idx >= 0).all() and deg_idx.shape[1] == args.n_top_degs
            ref_nc = ref_rows & (tg != CTRL)
            nc_mean = per_target_mean(X[ref_nc], tg[ref_nc], np.unique(tg[ref_nc])).mean(axis=0)
            preds = {
                ("-", "non-ctrl mean", None, None): np.tile(nc_mean, (len(q_targets), 1)),
                ("-", "control mean", None, None): np.tile(ctrl, (len(q_targets), 1)),
            }
            for (label, seed, e), E in emb.items():
                q_text = emb[(label, seed, "text")][q_rows]
                for K, p in knn_predict(q_text, E[ref_rows], X[ref_rows], args.Ks, args.temp).items():
                    preds[(label, f"PertOmni_{e}", seed, K)] = per_target_mean(p, tg[q_rows], q_targets)
            for (label, method, seed, K), p in preds.items():
                sc = score(p, true, ctrl, args.n_top_degs, deg_idx)
                base = dict(cl=cl, model=label, method=method, seed=seed, K=K)
                if phase == "val":
                    val_rows.append(base | {m: float(np.nanmean(v)) for m, v in sc.items()})
                else:
                    test_rows += [
                        base | dict(perturbation=t) | {m: float(v[i]) for m, v in sc.items()}
                        for i, t in enumerate(q_targets)
                    ]
            print(cl, phase, len(q_targets), "queries,", int(ref_rows.sum()), "reference rows", flush=True)

    val, test = pd.DataFrame(val_rows), pd.DataFrame(test_rows)
    km = val.dropna(subset=["K"]).groupby(["model", "method", "K"])["corr"].mean().reset_index()
    best = {(r.model, r.method): r.K for r in km.loc[km.groupby(["model", "method"])["corr"].idxmax()].itertuples()}
    best100 = {
        (r.model, r.method): r.K
        for r in km[km["K"] <= 100].loc[lambda x: x.groupby(["model", "method"])["corr"].idxmax()].itertuples()
    }
    val.to_csv(out_dir / "val_scores.csv", index=False)
    test.to_csv(out_dir / "test_scores_all_K.csv", index=False)
    save_obj(dict(best=best, best_upto100=best100), out_dir / "selected_K.pkl")
    metrics = ["corr", "corr_delta", "corr_top20", "corr_delta_top20"]
    summary = (
        test.groupby(["model", "method", "K", "cl", "seed"], dropna=False)[metrics]
        .mean()
        .groupby(["model", "method", "K", "cl"], dropna=False)
        .mean()
        .reset_index()
    )
    summary.to_csv(out_dir / "test_summary_all_K.csv", index=False)
    print("val-selected K:", best, "| within K<=100:", best100)
    for title, sel in [("val-selected K", best), ("val-selected K within 1..100", best100)]:
        keep = summary[summary.apply(lambda r: pd.isna(r["K"]) or sel.get((r["model"], r["method"])) == r["K"], axis=1)]
        print(
            f"\n== test, {title}\n",
            keep.pivot_table(index=["method", "model"], columns="cl", values=metrics).round(4).to_string(),
        )


if __name__ == "__main__":
    main()
