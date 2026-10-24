"""Train / validation / test splits of the Tahoe and CRISPRi pseudobulk data.

Split modes (`split_obs_names`):
  target, target_seed{n}  held-out perturbations (Tahoe drugs, CRISPRi target genes); _seed{n} = reshuffled split
  cv_target               perturbation cross-validation fold (needs fold)
  cell_type               cell-type cross-validation fold (needs fold): Tahoe 5-fold on cell lines,
                          CRISPRi leave one cell type out
  pair, pair_{suffix}     held-out (perturbation, cell type) pairs; the suffix (_all5, _sim) only changes the CRISPRi
                          split, Tahoe always uses its pair split
The pre-computed split files live in paths.TAHOE, paths.CRISPR_PUBLIC, paths.TARGET_CV and paths.CELLTYPE_CV.
"""

import collections
import pickle

import numpy as np
import pandas as pd

from .. import paths
from ..io import load_obj

PERT_KEY = {"tahoe": "drug", "crispr": "target_gene"}
CELL_KEY = {"tahoe": "cell_name", "crispr": "cell_type"}
FOLD_SPLITS = ("cv_target", "cell_type")

# leave-one-cell-type-out folds of the CRISPRi data, ordered by size (fold 1 = k562)
CRISPR_CELLTYPE_FOLDS = [["hct116"], ["k562"], ["hepg2"], ["jurkat"], ["rpe1"]]


def split_obs_names(adata, dataset, split_mode, fold=None):
    """obs_names of the (train, valid, test) rows of `adata` for one dataset ("tahoe" or "crispr")."""
    assert (fold is not None) == (split_mode in FOLD_SPLITS), (
        f"fold is required by, and only defined for, {FOLD_SPLITS}"
    )
    pert_key, cell_key = PERT_KEY[dataset], CELL_KEY[dataset]
    if split_mode == "target" or split_mode.startswith("target_seed"):
        suffix = split_mode[len("target") :]  # "" for the default split, "_seed{n}" for the reshuffled ones
        return _select(adata, pert_key, load_target_split(dataset, suffix))
    if split_mode == "cv_target":
        return _select(adata, pert_key, load_target_cv(dataset, fold))
    if split_mode == "cell_type":
        return _select(adata, cell_key, load_celltype_cv(dataset, fold))
    if split_mode == "pair" or split_mode.startswith("pair_"):
        if dataset == "tahoe":
            return tuple(load_obj(paths.TAHOE / f"{s}_pair.pkl") for s in ("train", "valid", "test"))
        suffix = split_mode[len("pair") :]
        return tuple(load_obj(paths.CRISPR_PUBLIC / f"{s}_pair{suffix}.pkl") for s in ("train", "valid", "test"))
    raise ValueError(f"unknown split_mode {split_mode!r}")


def make_target_split(perturbations, control, seed=0):
    """(train, valid, test) perturbations: 2/3 of the non-control perturbations for training, the rest halved into
    validation and test; the control is always in train."""
    perts = sorted(set(perturbations) - {control})
    rng = np.random.default_rng(seed)
    train = rng.choice(perts, size=int(2 * len(perts) / 3), replace=False)
    rest = sorted(set(perts) - set(train))
    valid = rng.choice(rest, size=len(rest) // 2, replace=False)
    test = np.array(sorted(set(rest) - set(valid)))
    return np.append(train, control), valid, test


def load_target_split(dataset, suffix=""):
    """(train, valid, test) perturbations; the controls are always in train."""
    if dataset == "tahoe":
        return tuple(load_obj(paths.TAHOE / f"{s}_target{suffix}.pkl") for s in ("train", "valid", "test"))
    return tuple(load_obj(paths.CRISPR_PUBLIC / f"{s}_target{suffix}_new.pkl") for s in ("train", "valid", "test"))


def load_target_cv(dataset, fold):
    """(train, valid, test) perturbations of a perturbation cross-validation fold; valid is carved out of train."""
    train_side = load_obj(paths.TARGET_CV / f"train_{dataset}_targets_fold{fold}.pkl")
    valid = load_obj(paths.TARGET_CV / f"valid_{dataset}_targets_fold{fold}.pkl")
    test = load_obj(paths.TARGET_CV / f"test_{dataset}_targets_fold{fold}.pkl")
    return np.setdiff1d(train_side, valid), valid, test


def _select(adata, key, groups):
    return tuple(adata[adata.obs[key].isin(group)].obs_names for group in groups)


# cell-type cross-validation folds


def celltype_folds(cell_types, n_folds=5, seed=0):
    """Shuffle the cell types (sorted first, so the folds only depend on the seed) into n_folds near-equal groups."""
    perm = np.random.default_rng(seed).permutation(np.array(sorted(set(cell_types))))
    return [sorted(chunk.tolist()) for chunk in np.array_split(perm, n_folds)]


def make_celltype_cv(folds, name, save_dir=paths.CELLTYPE_CV):
    """Fold k: test = folds[k], valid = folds[(k+1) % n], train = the rest. Saves {split}_{name}_celltypes_fold{k}.pkl."""
    save_dir.mkdir(parents=True, exist_ok=True)
    n = len(folds)
    for k in range(n):
        splits = {
            "test": folds[k],
            "valid": folds[(k + 1) % n],
            "train": sorted(c for j in range(n) if j not in (k, (k + 1) % n) for c in folds[j]),
        }
        for split, cell_types in splits.items():
            with open(save_dir / f"{split}_{name}_celltypes_fold{k}.pkl", "wb") as f:
                pickle.dump(cell_types, f, pickle.HIGHEST_PROTOCOL)
        print(
            f"{name} fold{k}: test {splits['test']} | valid {splits['valid']} | train {len(splits['train'])} cell types"
        )


def load_celltype_cv(name, fold, save_dir=paths.CELLTYPE_CV):
    """(train, valid, test) cell types of a pre-defined fold; name is "tahoe" or "crispr"."""
    return [load_obj(save_dir / f"{split}_{name}_celltypes_fold{fold}.pkl") for split in ("train", "valid", "test")]


# (perturbation, cell type) pair split


def pair_holdout_split(
    obs,
    pert_key="target_gene",
    cell_key="cell_type",
    controls=("non-targeting", "DMSO_TF"),
    min_cells=4,
    val_frac=0.15,
    n_holdout=1,
    seed=0,
):
    """Split (perturbation, cell type) pairs for cross-context evaluation.

    The split unit is a whole (perturbation, cell type) pair: all rows of a pair (every dose, plate and replicate)
    go to the same split. Perturbations observed in at least `min_cells` cell types are eligible; they are shuffled
    and divided into validation perturbations (`val_frac`) and test perturbations (the rest), so validation and test
    never share a perturbation. Each eligible perturbation holds out `n_holdout` of its cell types, chosen to balance
    held-out pairs across cell types; its other cell types stay in training. Controls and all other perturbations
    are training-only. Returns a dict of obs_names per split ("train", "valid", "test").
    """
    assert n_holdout < min_cells, "every held-out perturbation must keep at least one training cell type"
    pert = obs[pert_key].astype(str).to_numpy()
    cell = obs[cell_key].astype(str).to_numpy()
    pairs = pd.DataFrame({"pert": pert, "cell": cell}).drop_duplicates()
    cells_per_pert = pairs.groupby("pert")["cell"].apply(lambda x: sorted(set(x)))

    # sort before shuffling so the split only depends on the seed
    eligible = sorted(p for p, cs in cells_per_pert.items() if len(cs) >= min_cells and p not in set(controls))
    rng = np.random.default_rng(seed)
    eligible = [eligible[i] for i in rng.permutation(len(eligible))]
    n_val = int(round(val_frac * len(eligible)))
    split_perts = {"valid": eligible[:n_val], "test": eligible[n_val:]}

    # hold out n_holdout cell types per perturbation, each time preferring the cell type with the fewest held-out
    # pairs so far (random tie-break)
    held_out = {}
    for split, perts in split_perts.items():
        counts = collections.Counter()
        for p in perts:
            cs = list(cells_per_pert[p])
            for _ in range(n_holdout):
                least = min(counts[c] for c in cs)
                c = rng.choice([c for c in cs if counts[c] == least])
                held_out[(p, c)] = split
                counts[c] += 1
                cs.remove(c)

    row_split = np.array([held_out.get((p, c), "train") for p, c in zip(pert, cell)])
    return {split: obs.index[row_split == split].tolist() for split in ["train", "valid", "test"]}


def check_pair_holdout_split(
    obs, split_ids, pert_key="target_gene", cell_key="cell_type", controls=("non-targeting", "DMSO_TF"), n_holdout=1
):
    """Assert the properties of pair_holdout_split and print per-split counts."""
    split_of = pd.Series("", index=obs.index)
    for split, ids in split_ids.items():
        assert split_of.loc[ids].eq("").all(), f"rows assigned to more than one split ({split})"
        split_of.loc[ids] = split
    assert split_of.ne("").all(), "rows missing from the split"
    df = pd.DataFrame({"pert": obs[pert_key].astype(str), "cell": obs[cell_key].astype(str), "split": split_of})

    # every row of a (perturbation, cell type) pair is in the same split
    assert df.groupby(["pert", "cell"])["split"].nunique().eq(1).all(), "a pair is split across splits"
    assert df.loc[df["pert"].isin(controls), "split"].eq("train").all(), "controls outside training"
    train_pairs = set(map(tuple, df.loc[df["split"] == "train", ["pert", "cell"]].drop_duplicates().to_numpy()))
    train_perts, train_cells = {p for p, _ in train_pairs}, {c for _, c in train_pairs}
    held = {s: df.loc[df["split"] == s, ["pert", "cell"]].drop_duplicates() for s in ["valid", "test"]}
    for s, h in held.items():
        assert h["pert"].value_counts().eq(n_holdout).all(), (
            f"a perturbation does not hold out exactly {n_holdout} pairs in {s}"
        )
        assert all(p in train_perts and c in train_cells and (p, c) not in train_pairs for p, c in h.to_numpy()), (
            f"{s} pair violates p in P_train, c in C_train, (p, c) not in D_train"
        )
    assert not set(held["valid"]["pert"]) & set(held["test"]["pert"]), "validation and test share perturbations"

    n_train_cells = df.loc[df["split"] == "train"].groupby("pert")["cell"].nunique()
    for s, h in held.items():
        print(
            f"{s}: {len(h)} pairs, {(df['split'] == s).sum()} rows | pairs per cell type {h['cell'].value_counts().sort_index().to_dict()} "
            f"| training cell types per perturbation {n_train_cells.loc[h['pert']].value_counts().sort_index().to_dict()}"
        )
    print(f"train: {len(train_pairs)} pairs, {(df['split'] == 'train').sum()} rows")
