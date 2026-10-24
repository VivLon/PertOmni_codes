"""Drug-gene interaction (DGI) prediction data: labelled (drug, gene) pairs and per-pair PertOmni embeddings.

retrieval tasks (pair folder in paths.DATA):
  drug2gene  drug_target_retrieval  DGIdb, drug -> target genes
  gene2drug  gene_drug_retrieval    DGIdb, gene -> drugs
Pair files: {split}_drug_gene_pairs.pkl, lists of (drug, gene, label).
"""

import itertools
import random
from collections import defaultdict
from itertools import chain

import anndata
import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .. import paths
from ..io import load_obj, run_file, save_obj
from .inputs import expression_path
from .splits import load_target_split

RETRIEVAL_DIRS = {
    "drug2gene": "drug_target_retrieval",
    "gene2drug": "gene_drug_retrieval",
}


class DrugGenePairDataset(Dataset):
    def __init__(self, drug_emb, gene_emb, labels):
        self.drug_emb = drug_emb
        self.gene_emb = gene_emb
        self.labels = labels

    def __len__(self):
        return self.drug_emb.shape[0]

    def __getitem__(self, idx):
        return self.drug_emb[idx], self.gene_emb[idx], self.labels[idx]


def dgidb_ground_truth(interactions, drugs, genes):
    """({drug: [target genes]}, {gene: [drugs]}) of the DGIdb interactions (interactions.tsv) between the given Tahoe
    drugs and CRISPRi target genes. A Tahoe drug is matched by the DGIdb claim name, else by the normalized name."""
    df = pd.read_csv(interactions, sep="\t", usecols=["gene_name", "drug_claim_name", "drug_name"])
    drugs = set(drugs)
    df = df[df["gene_name"].isin(set(genes))]
    df = df.assign(drug=df["drug_claim_name"].where(df["drug_claim_name"].isin(drugs), df["drug_name"]))
    df = df[df["drug"].isin(drugs)]
    return df.groupby("drug")["gene_name"].apply(list).to_dict(), df.groupby("gene_name")["drug"].apply(list).to_dict()


def padded_id_maps(targets, keys, values, control_key, control_value):
    """Id maps of the DGI loss: ({key: ids of its targets, padded with -1 to the longest list}, {value: id}).

    keys are the Tahoe drugs and values the CRISPRi genes for the *drug* losses, the other way round for *gene*; the
    control key holds the control value, keys without targets hold only padding.
    """
    value_id = {value: i for i, value in enumerate(sorted(values))}
    ids = {control_key: [value_id[control_value]]}
    for key in sorted(set(keys) - {control_key}):
        ids[key] = [value_id[v] for v in targets.get(key, []) if v in value_id]
    width = max(map(len, ids.values()))
    return {key: row + [-1] * (width - len(row)) for key, row in ids.items()}, value_id


def load_ground_truth(retrieval):
    """DataFrame with columns drug, target_gene (one side holds lists)."""
    gt = load_obj(paths.DATA / RETRIEVAL_DIRS[retrieval] / "gt_target.pkl")
    if retrieval == "drug2gene":
        return pd.DataFrame(list(gt.items()), columns=["drug", "target_gene"])
    return pd.DataFrame(list(gt.items()), columns=["target_gene", "drug"])


def make_pairs(retrieval="drug2gene", neg_ratio=1, neg_drug=True, seed=0):
    """Write {train,valid,test}_drug_gene_pairs.pkl to the task's folder.

    train / valid: the interacting pairs among that split's drugs and genes, plus neg_ratio negatives per positive
    (a random non-interacting drug for the gene if neg_drug, else a random gene for the drug).
    test: every (test drug, gene) pair (drug2gene) or every (drug, test gene) pair (gene2drug), labelled.
    """
    drug_side = retrieval == "drug2gene"
    out_dir = paths.DATA / RETRIEVAL_DIRS[retrieval]
    gt = load_ground_truth(retrieval)
    L.seed_everything(seed)
    tahoe_targets = dict(zip(("train", "valid", "test"), load_target_split("tahoe")))
    crispr_targets = dict(zip(("train", "valid", "test"), load_target_split("crispr")))
    for split in ("train", "valid"):
        tahoe_target, crispr_target = tahoe_targets[split], crispr_targets[split]
        if drug_side:
            cpd2target = (
                gt[gt["drug"].isin(tahoe_target)]
                .groupby("drug")["target_gene"]
                .apply(lambda col: sorted({g for g in chain.from_iterable(col) if g in crispr_target}))
                .to_dict()
            )
            cpd2target = {k: v for k, v in cpd2target.items() if len(v) > 0}
            target2cpd = defaultdict(list)
            for drug, genes in cpd2target.items():
                for g in genes:
                    target2cpd[g].append(drug)
            target2cpd = dict(target2cpd)
            positives = [
                (drug, gene, 1)
                for drug, genes in cpd2target.items()
                if drug in tahoe_target
                for gene in genes
                if gene in crispr_target
            ]
        else:
            target2cpd = (
                gt[gt["target_gene"].isin(crispr_target)]
                .groupby("target_gene")["drug"]
                .apply(lambda col: sorted({d for d in chain.from_iterable(col) if d in tahoe_target}))
                .to_dict()
            )
            target2cpd = {k: v for k, v in target2cpd.items() if len(v) > 0}
            cpd2target = defaultdict(list)
            for gene, drugs in target2cpd.items():
                for d in drugs:
                    cpd2target[d].append(gene)
            cpd2target = dict(cpd2target)
            positives = [
                (drug, gene, 1)
                for gene, drugs in target2cpd.items()
                if gene in crispr_target
                for drug in drugs
                if drug in tahoe_target
            ]
        negatives = []
        for _ in range(neg_ratio):
            if neg_drug:
                for gene, drugs in target2cpd.items():
                    for _drug in drugs:
                        negatives.append((random.choice([d for d in tahoe_target if d not in drugs]), gene, 0))
            else:
                for drug, genes in cpd2target.items():
                    for _gene in genes:
                        negatives.append((drug, random.choice([g for g in crispr_target if g not in genes]), 0))
        pairs = positives + negatives
        random.shuffle(pairs)
        print(split, len(pairs))
        save_obj(pairs, out_dir / f"{split}_drug_gene_pairs.pkl")

    if drug_side:
        gt_set = {(drug, gene) for drug, genes in zip(gt["drug"], gt["target_gene"]) for gene in _as_list(genes)}
    else:
        gt_set = {(drug, gene) for gene, drugs in zip(gt["target_gene"], gt["drug"]) for drug in _as_list(drugs)}
    if drug_side:
        all_genes = list(set(anndata.read_h5ad(paths.CRISPR_PSEUDOBULK, backed="r").obs["target_gene"]))
        candidates = itertools.product(tahoe_targets["test"], all_genes)
    else:
        all_drugs = list(set(anndata.read_h5ad(paths.TAHOE_PSEUDOBULK, backed="r").obs["drug"]))
        candidates = itertools.product(all_drugs, crispr_targets["test"])
    pairs = [(drug, gene, int((drug, gene) in gt_set)) for drug, gene in candidates]
    print("test", len(pairs))
    save_obj(pairs, out_dir / "test_drug_gene_pairs.pkl")


def _as_list(x):
    return x if isinstance(x, (list, tuple, set)) else [x]


def pair_embeddings(run_dir, retrieval, out_dir):
    """Per-pair inputs of the DGI classifier from a PertOmni run exported with --export_full: pair_features of the
    [expression, text] embedding of every Tahoe / CRISPRi row."""
    features = {}
    for name, key in (("tahoe", "drug"), ("crispr", "target_gene")):
        labels = anndata.read_h5ad(expression_path(name), backed="r").obs[key].to_numpy()
        emb = torch.cat(
            [load_obj(run_file(run_dir, f"full_{name}_{kind}_embeds.pkl")).detach().cpu() for kind in ("gene", "text")],
            dim=1,
        ).numpy()
        features[name] = (labels, emb)
    pair_features(features, retrieval, out_dir)


def pair_features(features, retrieval, out_dir):
    """Per-pair inputs of the DGI classifier from per-row features: {"tahoe": (drug of each row, X), "crispr":
    (target gene of each row, X)}.

    Each pair's drug (gene) gets the mean feature vector of all Tahoe (CRISPRi) rows of that drug (gene). Rows of
    pairs whose drug (gene) is outside the split's perturbations are dropped, except on the side that is ranked
    (genes for drug2gene, drugs for gene2drug). Writes {split}_{tahoe,crispr}_emb.pkl to out_dir.
    """
    folder = RETRIEVAL_DIRS[retrieval]
    for name, column, keep_all in (("tahoe", 0, retrieval == "gene2drug"), ("crispr", 1, retrieval == "drug2gene")):
        labels, emb = features[name]
        splits = dict(zip(("train", "valid", "test"), load_target_split(name)))
        for split, targets in splits.items():
            perts = [pair[column] for pair in load_obj(paths.DATA / folder / f"{split}_drug_gene_pairs.pkl")]
            means = {p: emb[labels == p].mean(axis=0) for p in set(perts)}
            X = np.vstack([means[p] for p in perts])
            if not keep_all:
                X = X[np.isin(perts, list(targets))]
            save_obj(X, out_dir / f"{split}_{name}_emb.pkl")
            print(name, split, X.shape)


def paper_inputs(retrieval, model_name, seed):
    """Input files of the published DGI results: {split: (pairs, drug embeddings, gene embeddings)} paths."""
    folder = paths.DATA / RETRIEVAL_DIRS[retrieval]
    scratch = paths.SCRATCH / RETRIEVAL_DIRS[retrieval]
    test_pairs = "test_drug_gene_pairs.pkl" if retrieval == "drug2gene" else "test_drug_gene_pairs_new.pkl"
    pairs = {
        "train": folder / "train_drug_gene_pairs.pkl",
        "valid": folder / "valid_drug_gene_pairs.pkl",
        "test": folder / test_pairs,
    }
    if model_name == "bothmask_concat_pathway_shared_text_es_acc":
        emb = {
            s: (folder / f"{s}_tahoe_emb_seed{seed}.pkl", folder / f"{s}_arc_emb_seed{seed}.pkl")
            for s in ("train", "valid")
        }
        emb["test"] = (scratch / f"test_tahoe_emb_new_seed{seed}.pkl", scratch / f"test_arc_emb_new_seed{seed}.pkl")
    elif model_name == "vision_scores":
        emb = {
            s: (scratch / f"{s}_tahoe_vision_emb.pkl", scratch / f"{s}_arc_vision_emb.pkl")
            for s in ("train", "valid", "test")
        }
    else:
        emb_dir = folder / model_name / "emb"
        emb = {
            s: (emb_dir / f"{s}_tahoe_emb_seed{seed}.pkl", emb_dir / f"{s}_arc_emb_seed{seed}.pkl")
            for s in ("train", "valid", "test")
        }
    return {s: (pairs[s], *emb[s]) for s in ("train", "valid", "test")}


def run_inputs(pairs_dir, emb_dir):
    """Input files written by make_pairs and pair_embeddings."""
    return {
        s: (pairs_dir / f"{s}_drug_gene_pairs.pkl", emb_dir / f"{s}_tahoe_emb.pkl", emb_dir / f"{s}_crispr_emb.pkl")
        for s in ("train", "valid", "test")
    }
