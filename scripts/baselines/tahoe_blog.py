"""Tahoe blog baseline of drug-gene interaction, reproduced from the Tahoe blog post "Target deconvolution through
data" (https://blog.tahoebio.ai/p/target-deconvolution-through-data): a (drug, gene) pair is scored by the similarity
of the drug's and the gene knockdown's differential gene-set signatures.

Within each cell line, a perturbation's differential signature is its mean VISION score (vision_scores.R) minus the
control's. The cosine similarity of a drug's and a gene's differential signatures is averaged over the CRISPRi cell
types, then over the Tahoe cell lines; the cutoff of the recall / F1 scores is tuned on the validation pairs.
Writes similarity.csv (drugs x genes) and scores.pkl ({metric: value}) to {results_dir}/{save_name}/.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from pertomni import paths
from pertomni.data.dgi import RETRIEVAL_DIRS
from pertomni.data.splits import CELL_KEY, PERT_KEY
from pertomni.evaluation.dgi import CONTROLS, score, tune_cutoff, without_controls
from pertomni.io import load_obj, save_obj
from pertomni.training.common import prepare_run_dir


def differential_scores(vision_dir, ds):
    """{cell line: DataFrame perturbation x signature} of the mean VISION score minus the control's mean score."""
    obs = pd.read_csv(vision_dir / f"{ds}_metadata.csv", index_col=0)
    scores = pd.read_csv(vision_dir / f"{ds}_raw_scores.csv", index_col=0).set_index(obs.index)
    control = next(c for c in CONTROLS if c in set(obs[PERT_KEY[ds]]))
    out = {}
    for cl, sub in scores.groupby(obs[CELL_KEY[ds]], sort=False):
        means = sub.groupby(obs.loc[sub.index, PERT_KEY[ds]]).mean()
        out[cl] = means.drop(index=control) - means.loc[control]
    return out


def drug_gene_similarity(vision_dir):
    """DataFrame drugs x genes: cosine similarity of the differential signatures, averaged over the CRISPRi cell types
    of each Tahoe cell line, then over the Tahoe cell lines."""
    tahoe, crispr = differential_scores(vision_dir, "tahoe"), differential_scores(vision_dir, "crispr")
    per_line = []
    for drugs in tahoe.values():
        sims = []
        for genes in crispr.values():
            shared = drugs.columns.intersection(genes.columns)
            sims.append(pd.DataFrame(cosine_similarity(drugs[shared], genes[shared]), drugs.index, genes.index))
        columns = sorted(set().union(*(s.columns for s in sims)))
        mean = np.nanmean([s.reindex(columns=columns).to_numpy() for s in sims], axis=0)
        per_line.append(pd.DataFrame(mean, index=drugs.index, columns=columns))
    rows = sorted(set().union(*(d.index for d in per_line)))
    return sum(d.reindex(rows) for d in per_line) / len(per_line)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save_name", required=True)
    parser.add_argument("--results_dir", type=Path, default=paths.RESULTS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--vision_dir", type=Path, default=paths.RESULTS / "vision", help="output of vision_scores.R")
    parser.add_argument("--retrieval", default="drug2gene", choices=list(RETRIEVAL_DIRS))
    parser.add_argument("--pairs_dir", type=Path, help="pair files (default: the task folder)")
    parser.add_argument("--test_pairs", default="test_drug_gene_pairs.pkl", help="test pair file name in --pairs_dir")
    args = parser.parse_args()

    out_dir = prepare_run_dir(args.results_dir, args.save_name, vars(args), args.overwrite)
    similarity = drug_gene_similarity(args.vision_dir)
    similarity.to_csv(out_dir / "similarity.csv")
    pairs_dir = args.pairs_dir or paths.DATA / RETRIEVAL_DIRS[args.retrieval]
    probs, labels = {}, {}
    for split, file in (("valid", "valid_drug_gene_pairs.pkl"), ("test", args.test_pairs)):
        pairs = load_obj(pairs_dir / file)
        idx, labels[split] = without_controls(pairs)
        probs[split] = np.array([similarity.loc[pairs[i][0], pairs[i][1]] for i in idx])
    result = score(labels["test"], probs["test"], tune_cutoff(labels["valid"], [probs["valid"]]))
    save_obj(result, out_dir / "scores.pkl")
    print(args.retrieval, {k: round(float(v), 4) for k, v in result.items()})


if __name__ == "__main__":
    main()
