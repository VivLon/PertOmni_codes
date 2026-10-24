"""kNN perturbation prediction from the perturbation cross-validation runs (train_multi.py --split_mode cv_target).

For each seed and fold, the text embedding of every held-out perturbation retrieves its top-K most similar
training-side samples within the same cell line; the prediction is their softmax(sim/temp)-weighted mean expression.
Writes seed{seed}/{cl}_top{K}_{use_train_emb}_scores.pkl (per perturbation) and {cl}_{use_train_emb}_scores.pkl
({metric: {K: [mean over perturbations, one per seed]}}) to {results_dir}/{save_name}/.
"""

import argparse
from pathlib import Path

import anndata
import pandas as pd

from pertomni import paths
from pertomni.data.splits import load_target_cv
from pertomni.evaluation.perturbation import cv_knn, cv_predict, cv_scores
from pertomni.io import load_obj, run_file, save_obj
from pertomni.training.common import prepare_run_dir

METRICS = ["mae", "mse", "corr", "corr_delta", "mae_top20", "mse_top20", "corr_top20", "corr_delta_top20"]
PAPER_RUNS = str(
    paths.DATA / "langpert/siglip/cv_target/pseudobulk/concat_pathway_shared_text_es_acc_fold{fold}/seed{seed}"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save_name", default="pertomni_unmasked_cv")
    parser.add_argument("--results_dir", type=Path, default=paths.RESULTS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--runs", default=PAPER_RUNS, help="run folder of each fold and seed, with {fold} and {seed} placeholders"
    )
    parser.add_argument("--n_folds", default=5, type=int)
    parser.add_argument("--seeds", default=[0, 1, 2, 3, 4], type=int, nargs="+")
    parser.add_argument("--Ks", default=[1, 3, 5, 8, 10, 25, 50, 75, 100], type=int, nargs="+")
    parser.add_argument("--cell_lines", default=["rpe1", "hepg2", "jurkat"], nargs="+")
    parser.add_argument(
        "--use_train_emb",
        default=["text", "gene"],
        nargs="+",
        choices=["text", "gene"],
        help="embedding of the training-side samples that the test text embeddings are matched against",
    )
    parser.add_argument("--temp", default=0.07, type=float)
    parser.add_argument(
        "--deg_dir", type=Path, default=paths.DATA / "langpert" / "deg", help="top DEGs per cell line, {cl}.csv"
    )
    return parser.parse_args()


def fold_embeddings(adata, runs, seed, use_train_emb, n_folds):
    """Per fold: (train-side AnnData, their embeddings, held-out AnnData, held-out text embeddings)."""
    for fold in range(n_folds):
        train_side = load_obj(paths.TARGET_CV / f"train_crispr_targets_fold{fold}.pkl")
        test = load_target_cv("crispr", fold)[2]
        train_adata = adata[adata.obs["target_gene"].isin(train_side)]
        test_adata = adata[adata.obs["target_gene"].isin(test)]
        run_dir = Path(runs.format(fold=fold, seed=seed))
        train_emb = (
            load_obj(run_file(run_dir, f"test_crispr_{use_train_emb}_embeds.pkl")).detach().cpu().numpy()
        )  # train-side rows
        test_emb = load_obj(run_file(run_dir, "testing_crispr_text_embeds.pkl")).detach().cpu().numpy()  # held-out rows
        assert train_emb.shape[0] == train_adata.n_obs and test_emb.shape[0] == test_adata.n_obs, (
            fold,
            train_emb.shape,
            test_emb.shape,
        )
        yield train_adata, train_emb, test_adata, test_emb


def main():
    args = parse_args()
    out_dir = prepare_run_dir(args.results_dir, args.save_name, vars(args), args.overwrite)
    adata = anndata.read_h5ad(paths.CRISPR_PSEUDOBULK)
    degs = {}
    for cl in args.cell_lines:
        deg_path = args.deg_dir / f"{cl}.csv"
        degs[cl] = (
            pd.read_csv(deg_path, index_col=0).groupby("group")["names"].apply(list).to_dict()
            if deg_path.exists()
            else None
        )
        if degs[cl] is None:
            print(f"[warn] {deg_path} missing: *_top20 metrics skipped for {cl}")
    adata_cls = {cl: adata[adata.obs["cell_type"] == cl].copy() for cl in args.cell_lines}

    for use_train_emb in args.use_train_emb:
        all_scores = {cl: {m: {K: [] for K in args.Ks} for m in METRICS} for cl in args.cell_lines}
        for seed in args.seeds:
            print(use_train_emb, "seed", seed, flush=True)
            ranks, sims = cv_knn(
                adata,
                fold_embeddings(adata, args.runs, seed, use_train_emb, args.n_folds),
                max(args.Ks),
                args.cell_lines,
            )
            for K in args.Ks:
                preds = cv_predict(adata, ranks, sims, K, args.temp, args.cell_lines)
                for cl in args.cell_lines:
                    scores = cv_scores(adata_cls[cl], preds[cl], degs=degs[cl])
                    save_obj(scores, out_dir / f"seed{seed}" / f"{cl}_top{K}_{use_train_emb}_scores.pkl")
                    mean_scores = pd.DataFrame(scores).set_index("perturbation").mean(axis=0)
                    for m in METRICS:
                        if m in mean_scores:
                            all_scores[cl][m][K].append(float(mean_scores[m]))
        for cl in args.cell_lines:
            save_obj(all_scores[cl], out_dir / f"{cl}_{use_train_emb}_scores.pkl")


if __name__ == "__main__":
    main()
