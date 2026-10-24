"""Score drug-gene interaction predictions (pertomni.evaluation.dgi) over the seeds of a run.

  --kind snn        classifier runs of train_dgi.py: (val_)snn_{es_mode}_{use_emb}_logits.pkl
  --kind embedding  PertOmni runs trained with --export_full (e.g. the DGI fine-tuned text encoders): cosine
                    similarity of the mean drug and gene embeddings ({use_emb}: gene = expression, text, all = both)
Writes scores.pkl ({use_emb: {metric: [one value per seed]}}) and scores.csv to {results_dir}/{save_name}/.
Example:
  python scripts/evaluate_dgi.py --kind embedding --retrieval drug2gene --use_emb text \
      --runs "$PERTOMNI_RESULTS/pertomni_dgi_drug_seed{seed}" --save_name dgi_drug2gene_ft_scores
"""

import argparse
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import torch

from pertomni import paths
from pertomni.data.dgi import RETRIEVAL_DIRS
from pertomni.evaluation.dgi import mean_embeddings, pair_cosine, score, tune_cutoff, without_controls
from pertomni.io import load_obj, run_file, save_obj
from pertomni.training.common import prepare_run_dir


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save_name", required=True)
    parser.add_argument("--results_dir", type=Path, default=paths.RESULTS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--kind", required=True, choices=["snn", "embedding"])
    parser.add_argument("--runs", required=True, help="run folder of each seed, with a {seed} placeholder")
    parser.add_argument("--retrieval", default="drug2gene", choices=list(RETRIEVAL_DIRS))
    parser.add_argument("--pairs_dir", type=Path, help="pair files (default: the task folder)")
    parser.add_argument("--test_pairs", default="test_drug_gene_pairs.pkl", help="test pair file name in --pairs_dir")
    parser.add_argument("--seeds", default=[0, 1, 2, 3, 4], type=int, nargs="+")
    parser.add_argument("--use_emb", default=["gene", "text", "all"], nargs="+", choices=["gene", "text", "all"])
    parser.add_argument(
        "--es_mode", default="es_loss", choices=["es_loss", "es_acc"], help="--kind snn: classifier to score"
    )
    return parser.parse_args()


def snn_probs(run_dir, prefix, es_mode, use_emb):
    logits = load_obj(run_dir / f"{prefix}snn_{es_mode}_{use_emb}_logits.pkl")
    return torch.sigmoid(logits).detach().squeeze().numpy()


def embedding_probs(run_dir, pairs, obs, use_emb):
    """Per-pair cosine similarity of the mean drug / gene embeddings of a run's full_* exports."""
    means = {}
    for name in ("tahoe", "crispr"):
        parts = {
            kind: load_obj(run_file(run_dir, f"full_{name}_{kind}_embeds.pkl")).detach().cpu().numpy()
            for kind in ("gene", "text")
        }
        emb = np.concatenate([parts["gene"], parts["text"]], axis=1) if use_emb == "all" else parts[use_emb]
        means[name] = mean_embeddings(emb, obs[name])
    return pair_cosine(means["tahoe"], means["crispr"], pairs)


def main():
    args = parse_args()
    out_dir = prepare_run_dir(args.results_dir, args.save_name, vars(args), args.overwrite)
    pairs_dir = args.pairs_dir or paths.DATA / RETRIEVAL_DIRS[args.retrieval]
    all_pairs = {
        "valid": load_obj(pairs_dir / "valid_drug_gene_pairs.pkl"),
        "test": load_obj(pairs_dir / args.test_pairs),
    }
    kept = {split: without_controls(pairs) for split, pairs in all_pairs.items()}
    obs = None
    if args.kind == "embedding":
        obs = {
            "tahoe": anndata.read_h5ad(paths.TAHOE_PSEUDOBULK, backed="r").obs["drug"].to_numpy(),
            "crispr": anndata.read_h5ad(paths.CRISPR_PSEUDOBULK, backed="r").obs["target_gene"].to_numpy(),
        }

    scores, rows = {}, []
    for use_emb in args.use_emb:
        probs = {"valid": [], "test": []}
        for seed in args.seeds:
            run_dir = Path(args.runs.format(seed=seed))
            for split, prefix in (("valid", "val_"), ("test", "")):
                idx = kept[split][0]
                if args.kind == "snn":
                    probs[split].append(snn_probs(run_dir, prefix, args.es_mode, use_emb)[idx])
                else:
                    probs[split].append(embedding_probs(run_dir, [all_pairs[split][i] for i in idx], obs, use_emb))
        cutoff = tune_cutoff(kept["valid"][1], probs["valid"])
        per_seed = [score(kept["test"][1], prob, cutoff) for prob in probs["test"]]
        scores[use_emb] = {m: [s[m] for s in per_seed] for m in per_seed[0]}
        rows += [dict(use_emb=use_emb, seed=seed, cutoff=cutoff, **s) for seed, s in zip(args.seeds, per_seed)]
        print(use_emb, {m: round(float(np.mean(v)), 4) for m, v in scores[use_emb].items()}, flush=True)
    save_obj(scores, out_dir / "scores.pkl")
    pd.DataFrame(rows).to_csv(out_dir / "scores.csv", index=False)


if __name__ == "__main__":
    main()
