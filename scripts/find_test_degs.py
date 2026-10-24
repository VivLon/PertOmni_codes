"""Top-20 DEGs of each test perturbation from the raw single-cell CRISPRi data (pertomni.evaluation.degs.find_degs).

Test perturbations: --split target = the test target genes; --split pair = the cell line's targets in test_pair.pkl.
Writes {cl}.csv (rank_genes_groups table) and {cl}.pkl (its uns entry) to --out_dir.
"""

import argparse
from pathlib import Path

import anndata

from pertomni import paths
from pertomni.data.splits import load_target_split
from pertomni.evaluation.degs import CTRL, find_degs
from pertomni.io import load_obj, save_obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell_lines", default=["rpe1", "hepg2", "jurkat"], nargs="+")
    parser.add_argument("--split", default="target", choices=["target", "pair"])
    parser.add_argument("--n_genes", default=20, type=int)
    parser.add_argument("--out_dir", type=Path, default=paths.DATA / "langpert" / "deg_split_test")
    parser.add_argument("--chunk", default=20000, type=int, help="rows read at a time from the backed raw data")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pseudobulk = anndata.read_h5ad(paths.CRISPR_PSEUDOBULK, backed="r")
    if args.split == "target":
        test = {cl: set(load_target_split("crispr")[2]) for cl in args.cell_lines}
    else:
        obs = pseudobulk.obs[pseudobulk.obs.index.isin(load_obj(paths.CRISPR_PUBLIC / "test_pair.pkl"))]
        test = {cl: set(obs.loc[obs["cell_type"] == cl, "target_gene"].astype(str)) - {CTRL} for cl in args.cell_lines}
    for cl in args.cell_lines:
        df, uns = find_degs(cl, test[cl], pseudobulk.var_names, args.n_genes, args.chunk)
        df.to_csv(args.out_dir / f"{cl}.csv")
        save_obj(uns, args.out_dir / f"{cl}.pkl")
        print(cl, "done", df.shape, "frac down-regulated", float((df["logfoldchanges"] < 0).mean()), flush=True)


if __name__ == "__main__":
    main()
