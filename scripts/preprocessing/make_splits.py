"""Train / validation / test splits of the pseudobulk data (pertomni.data.splits).

  target  held-out perturbations: 2/3 train, 1/6 valid, 1/6 test (controls in train); --seed > 0 writes the
          reshuffled _seed{n} files used by --split_mode target_seed{n}
  pair    held-out (perturbation, cell line) pairs (Tahoe: drugs seen in >= 20 cell lines, 10 held out each;
          CRISPRi: targets seen in >= 4 cell types, 1 held out each)
Cell-type folds: make_celltype_folds.py.
"""

import argparse

import anndata

from pertomni import paths
from pertomni.data.splits import check_pair_holdout_split, make_target_split, pair_holdout_split
from pertomni.io import save_obj

DATASETS = {  # pseudobulk, perturbation key, cell key, control, split folder, target-split file suffix
    "tahoe": (
        paths.TAHOE_PSEUDOBULK,
        "drug",
        "cell_name",
        "DMSO_TF",
        paths.TAHOE,
        "",
    ),
    "crispr": (
        paths.CRISPR_PSEUDOBULK,
        "target_gene",
        "cell_type",
        "non-targeting",
        paths.CRISPR_PUBLIC,
        "_new",
    ),
}
PAIR_SETTINGS = {"tahoe": dict(min_cells=20, n_holdout=10), "crispr": dict(min_cells=4, n_holdout=1)}


def save_new(obj, path, overwrite):
    """save_obj, refusing to replace existing inputs (e.g. the published ones) unless overwrite is set."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    save_obj(obj, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("split", choices=["target", "pair"])
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--overwrite", action="store_true", help="replace existing files")
    args = parser.parse_args()

    for name in args.datasets:
        path, pert_key, cell_key, control, out_dir, suffix = DATASETS[name]
        obs = anndata.read_h5ad(path, backed="r").obs
        if args.split == "target":
            seed_suffix = f"_seed{args.seed}" if args.seed else ""
            for split, perts in zip(
                ("train", "valid", "test"), make_target_split(obs[pert_key].astype(str), control, args.seed)
            ):
                save_new(perts, out_dir / f"{split}_target{seed_suffix}{suffix}.pkl", args.overwrite)
                print(name, split, len(perts), "perturbations")
        else:
            settings = PAIR_SETTINGS[name]
            res = pair_holdout_split(
                obs, pert_key, cell_key, controls=(control,), val_frac=0.15, seed=args.seed, **settings
            )
            check_pair_holdout_split(obs, res, pert_key, cell_key, controls=(control,), n_holdout=settings["n_holdout"])
            for split in ("train", "valid", "test"):
                save_new(res[split], out_dir / f"{split}_pair.pkl", args.overwrite)


if __name__ == "__main__":
    main()
