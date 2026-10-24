"""Inputs of LangPert (langpert_run.py) on the CRISPRi target split, per cell line.

  val   queries = valid target genes, reference pool / observed effects = train target genes
  test  queries = test target genes, reference pool / observed effects = train + valid target genes
Observed effect of a reference gene = its mean pseudobulk expression in the cell line (the pool includes the
non-targeting control). PertOmni_text + LangPert narrows the candidate genes of each query: the query's PertOmni text
embedding retrieves pool rows of the same cell line by cosine similarity, and the candidates are the first
--max_narrow_K unique target genes in rank order. Writes one pickle, --out.
"""

import argparse
from pathlib import Path

import anndata
import numpy as np
import pandas as pd

from pertomni import paths
from pertomni.data.inputs import expression_path
from pertomni.data.splits import load_target_split
from pertomni.evaluation.perturbation import CTRL, l2_normalize
from pertomni.io import load_obj, run_file, save_obj

PAPER_RUN = paths.DATA / "multi_dataset/siglip/target/pseudobulk/bothmask_concat_pathway_shared_text_es_acc/seed0"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run_dir", type=Path, default=PAPER_RUN, help="PertOmni run holding full_crispr_text_embeds.pkl"
    )
    parser.add_argument("--cell_lines", default=["rpe1", "hepg2", "jurkat"], nargs="+")
    parser.add_argument("--max_narrow_K", default=500, type=int)
    parser.add_argument("--out", type=Path, default=paths.RESULTS / "langpert" / "inputs.pkl")
    args = parser.parse_args()

    split = dict(zip(("train", "valid", "test"), map(set, load_target_split("crispr"))))
    backed = anndata.read_h5ad(expression_path("crispr"), backed="r")
    keep = np.where(backed.obs["cell_type"].isin(args.cell_lines).to_numpy())[0]
    adata = backed[keep].to_memory()
    X = np.asarray(adata.X, dtype=np.float32)
    tg, cls = adata.obs["target_gene"].to_numpy(), adata.obs["cell_type"].to_numpy()
    emb = l2_normalize(load_obj(run_file(args.run_dir, "full_crispr_text_embeds.pkl")).detach().cpu().numpy()[keep])

    out = {"var_names": list(adata.var_names), "max_narrow_K": args.max_narrow_K}
    for cl in args.cell_lines:
        in_cl = cls == cl
        ctrl = X[in_cl & (tg == CTRL)].mean(axis=0)
        for phase, pool_splits, query_split in (("val", ["train"], "valid"), ("test", ["train", "valid"], "test")):
            pool = set().union(*(split[s] for s in pool_splits))
            pool_rows = np.where(in_cl & np.isin(tg, list(pool)))[0]
            query_rows = np.where(in_cl & np.isin(tg, list(split[query_split])))[0]
            observed = pd.DataFrame(X[pool_rows]).groupby(tg[pool_rows]).mean()
            observed_effects = {t: observed.loc[t].to_numpy(np.float32) for t in observed.index}
            non_ctrl_mean = observed.drop(index=CTRL, errors="ignore").to_numpy().mean(axis=0).astype(np.float32)
            queries = sorted(set(tg[query_rows]))
            true = pd.DataFrame(X[query_rows]).groupby(tg[query_rows]).mean()
            order = np.argsort(-(emb[query_rows] @ emb[pool_rows].T), axis=1)
            narrow = {}
            for i, r in enumerate(query_rows):
                ranked = list(dict.fromkeys(str(t) for t in tg[pool_rows][order[i]]))
                narrow.setdefault(str(tg[r]), ranked[: args.max_narrow_K])
            assert not set(queries) & set(observed_effects), "query in the reference pool"
            out[(phase, cl)] = dict(
                queries=queries,
                observed_effects=observed_effects,
                fallback_mean=non_ctrl_mean,
                ctrl=ctrl,
                true={t: true.loc[t].to_numpy(np.float32) for t in queries},
                narrow=narrow,
            )
            print(cl, phase, len(queries), "queries,", len(observed_effects), "pool genes", flush=True)
    save_obj(out, args.out)


if __name__ == "__main__":
    main()
