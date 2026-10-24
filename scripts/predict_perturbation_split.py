"""Perturbation prediction on the CRISPRi target split (train / valid / test target genes), within each cell line.

Each method's hyperparameter is selected on the validation perturbations (references / fit = train perturbations),
then the test perturbations are predicted with references / fit = train + valid perturbations.
  PertOmni_{text,gene}  the query's text embedding retrieves the top-K reference rows by cosine similarity (text or
                        expression embedding); prediction = softmax(sim/temp)-weighted mean expression; K tuned
  non-ctrl mean         mean expression of the reference non-control perturbations
  control mean          non-targeting mean ("no change"; its delta correlation is undefined)
  GenePert              Ridge on L2-normalised GenePT (ada) target-gene embeddings; alpha tuned
  STRING GNN            StandardScaler -> PCA(100) -> KNeighborsRegressor on STRING GNN embeddings; n_neighbors tuned
Targets without an embedding fall back to the non-ctrl mean. Metrics (pertomni.evaluation.perturbation.score) per
perturbation; the top-20 DEGs are the test-set Wilcoxon DEGs of find_test_degs.py with --deg_dir, otherwise (and
always on validation) the 20 genes with the largest |true - ctrl|.
PertOmni embeddings: full_crispr_{text,gene}_embeds.pkl of runs trained with --export_full.
"""

import argparse
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pertomni import paths
from pertomni.data.splits import load_target_split
from pertomni.evaluation.perturbation import CTRL, feature_predict, knn_predict, per_target_mean, score
from pertomni.io import load_obj, run_file, save_obj
from pertomni.training.common import prepare_run_dir

PAPER_RUNS = str(
    paths.DATA / "multi_dataset/siglip/target/pseudobulk/bothmask_concat_pathway_shared_text_es_acc/seed{seed}"
)
STRING_GNN = (
    paths.DATA
    / "string_gnn/dim256_layers8_gcn_weighted_link_prediction_lr0.0001_wd0.0_dropout0_bs2048_thresh700/embeddings/final_embeddings.npz"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save_name", default="split_masked")
    parser.add_argument("--results_dir", type=Path, default=paths.RESULTS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--runs", default=PAPER_RUNS, help="PertOmni run folder of each seed, with a {seed} placeholder"
    )
    parser.add_argument("--seeds", default=[0, 1, 2, 3, 4], type=int, nargs="+")
    parser.add_argument(
        "--Ks", default=[1, 3, 5, 8, 10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000], type=int, nargs="+"
    )
    parser.add_argument("--cell_lines", default=["rpe1", "hepg2", "jurkat"], nargs="+")
    parser.add_argument("--temp", default=0.07, type=float)
    parser.add_argument(
        "--symbol2ensembl",
        type=Path,
        default=paths.DATA / "langpert" / "symbol2ensembl.pkl",
        help="pickle {symbol: [ensembl ids]} for the STRING GNN lookup",
    )
    parser.add_argument("--n_top_degs", default=20, type=int)
    parser.add_argument(
        "--deg_dir", type=Path, help="test-set DEGs from find_test_degs.py ({cl}.csv); default: top |true - ctrl| genes"
    )
    return parser.parse_args()


def load_feature_embeddings(symbol2ensembl):
    genept = load_obj(paths.GENEPT / "GenePT_gene_embedding_ada_text.pickle")
    genept = {k: np.asarray(v, dtype=np.float32) for k, v in genept.items()}
    genept = {k: v / np.linalg.norm(v) for k, v in genept.items()}
    z = np.load(STRING_GNN)
    string = {}
    for sym, ids in load_obj(symbol2ensembl).items():
        hit = [i for i in ids if i in z.files]
        if hit:
            string[sym] = z[hit[0]]
    return genept, string


def main():
    args = parse_args()
    out_dir = prepare_run_dir(args.results_dir, args.save_name, vars(args), args.overwrite)
    split = dict(zip(("train", "valid", "test"), map(set, load_target_split("crispr"))))
    assert (
        not (split["train"] & split["valid"])
        and not (split["train"] & split["test"])
        and not (split["valid"] & split["test"])
    )
    adata_b = anndata.read_h5ad(paths.CRISPR_PSEUDOBULK, backed="r")
    keep = np.where(adata_b.obs["cell_type"].isin(args.cell_lines).to_numpy())[0]
    adata = adata_b[keep].to_memory()
    X = np.asarray(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X, dtype=np.float32)
    obs = adata.obs
    print("adata", adata.shape, flush=True)

    pertomni = {}  # full_crispr_* rows are aligned with crispr_adata.h5ad
    for seed in args.seeds:
        run_dir = Path(args.runs.format(seed=seed))
        for e in ["text", "gene"]:
            emb = load_obj(run_file(run_dir, f"full_crispr_{e}_embeds.pkl")).detach().cpu().numpy()
            assert emb.shape[0] == adata_b.n_obs
            pertomni[(seed, e)] = emb[keep]
    genept, string = load_feature_embeddings(args.symbol2ensembl)

    alphas, nns = [0.01, 0.1, 1.0, 10.0, 100.0], [1, 5, 10, 20, 40, 60, 80, 100, 150, 200]
    val_rows, test_rows, coverage = [], [], []
    tg = obs["target_gene"].to_numpy()
    for cl in args.cell_lines:
        in_cl = (obs["cell_type"] == cl).to_numpy()
        ctrl = X[in_cl & (tg == CTRL)].mean(axis=0)
        for phase, ref_sets, q_set in [("val", ["train"], "valid"), ("test", ["train", "valid"], "test")]:
            ref_targets = set().union(*[split[s] for s in ref_sets])
            ref_rows = in_cl & np.isin(tg, list(ref_targets))  # includes the non-targeting rows
            q_rows = in_cl & np.isin(tg, list(split[q_set]))
            q_targets = np.unique(tg[q_rows])
            true = per_target_mean(X[q_rows], tg[q_rows], q_targets)
            deg_idx = None
            if phase == "test" and args.deg_dir is not None:
                degs = (
                    pd.read_csv(args.deg_dir / f"{cl}.csv", index_col=0).groupby("group")["names"].apply(list).to_dict()
                )
                deg_idx = np.stack([adata.var_names.get_indexer(degs[t]) for t in q_targets])
                assert (deg_idx >= 0).all() and deg_idx.shape[1] == args.n_top_degs
            # per-target reference means for the feature-based baselines (non-control only)
            ref_nc = ref_rows & (tg != CTRL)
            ref_t = np.unique(tg[ref_nc])
            ref_y = per_target_mean(X[ref_nc], tg[ref_nc], ref_t)
            nc_mean = ref_y.mean(axis=0)

            preds = {
                ("non-ctrl mean", None): np.tile(nc_mean, (len(q_targets), 1)),
                ("control mean", None): np.tile(ctrl, (len(q_targets), 1)),
            }
            for seed in args.seeds:
                q_text = pertomni[(seed, "text")][q_rows]
                for e in ["text", "gene"]:
                    for K, p in knn_predict(
                        q_text, pertomni[(seed, e)][ref_rows], X[ref_rows], args.Ks, args.temp
                    ).items():
                        preds[(f"PertOmni_{e}", (seed, K))] = per_target_mean(p, tg[q_rows], q_targets)
            out, cov_gp = feature_predict(lambda a: Ridge(alpha=a), alphas, genept, ref_t, ref_y, q_targets, nc_mean)
            preds.update({("GenePert", a): p for a, p in out.items()})
            make_knn = lambda k: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=100, random_state=0)),
                    ("knn", KNeighborsRegressor(n_neighbors=k)),
                ]
            )
            out, cov_sg = feature_predict(make_knn, nns, string, ref_t, ref_y, q_targets, nc_mean)
            preds.update({("STRING GNN", k): p for k, p in out.items()})
            coverage.append(
                dict(cl=cl, phase=phase, n_query=len(q_targets), n_ref=len(ref_t), genept=cov_gp, string=cov_sg)
            )

            for (method, hp), p in preds.items():
                sc = score(p, true, ctrl, args.n_top_degs, deg_idx if phase == "test" else None)
                seed, h = hp if method.startswith("PertOmni") else (None, hp)
                if phase == "val":
                    val_rows.append(
                        dict(cl=cl, method=method, seed=seed, hp=h, **{m: float(np.nanmean(v)) for m, v in sc.items()})
                    )
                else:
                    test_rows += [
                        dict(
                            cl=cl,
                            method=method,
                            seed=seed,
                            hp=h,
                            perturbation=t,
                            **{m: float(v[i]) for m, v in sc.items()},
                        )
                        for i, t in enumerate(q_targets)
                    ]
            print(cl, phase, len(q_targets), "queries", flush=True)

    val, test = pd.DataFrame(val_rows), pd.DataFrame(test_rows)
    # one hyperparameter per method: best validation corr averaged over cell lines (and seeds)
    sel = val.groupby(["method", "hp"], dropna=False)["corr"].mean().reset_index()
    best = sel.loc[sel.groupby("method")["corr"].idxmax()].set_index("method")["hp"].to_dict()
    test_sel = pd.concat(
        [
            test[(test["method"] == m) & ((test["hp"] == h) if pd.notna(h) else test["hp"].isna())]
            for m, h in best.items()
        ]
    )
    val.to_csv(out_dir / "val_scores.csv", index=False)
    test.to_csv(out_dir / "test_scores_all_hp.csv", index=False)
    test_sel.to_csv(out_dir / "test_scores_selected.csv", index=False)
    pd.DataFrame(coverage).to_csv(out_dir / "coverage.csv", index=False)
    save_obj(best, out_dir / "selected_hp.pkl")
    print("selected hyperparameters:", best)


if __name__ == "__main__":
    main()
