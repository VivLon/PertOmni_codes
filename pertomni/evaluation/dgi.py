"""Scores of drug-gene interaction (DGI) predictions over labelled (drug, gene, label) pairs.

A prediction is a probability per pair: the classifier's sigmoid output, or the cosine similarity of the mean drug and
gene embeddings of a fine-tuned PertOmni. AUROC / AUPRC are threshold-free; weighted recall / F1 use the cutoff that
maximises the mean validation weighted F1 over seeds (candidates: the 50-95% quantiles of the validation predictions).
Pairs with a control (DMSO_TF, non-targeting) are excluded.
"""

from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score

CONTROLS = ("DMSO_TF", "non-targeting")
QUANTILES = np.linspace(0.5, 0.95, 15)


def without_controls(pairs):
    """(indices of the pairs without a control, their labels)."""
    keep = [i for i, (drug, gene, _) in enumerate(pairs) if drug not in CONTROLS and gene not in CONTROLS]
    return keep, [pairs[i][2] for i in keep]


def tune_cutoff(y_valid, valid_probs):
    """Cutoff with the best mean validation weighted F1; valid_probs: one array per seed."""
    cutoffs = [c for prob in valid_probs for c in np.quantile(prob, QUANTILES)]
    f1s = defaultdict(list)
    for prob in valid_probs:
        for cutoff in cutoffs:
            report = classification_report(y_valid, (np.asarray(prob) > cutoff).astype(int), output_dict=True)
            f1s[cutoff].append(report["weighted avg"]["f1-score"])
    mean_f1 = {c: np.mean(v) for c, v in f1s.items()}
    return max(mean_f1, key=mean_f1.get)


def score(y_true, prob, cutoff):
    prob = np.asarray(prob).ravel()
    report = classification_report(y_true, (prob > cutoff).astype(int), output_dict=True)
    return {
        "auroc": roc_auc_score(y_true, prob),
        "auprc": average_precision_score(y_true, prob),
        "weighted_recall": report["weighted avg"]["recall"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }


def mean_embeddings(emb, perts):
    """{perturbation: mean embedding over its rows}."""
    perts = np.asarray(perts)
    return {p: emb[perts == p].mean(axis=0) for p in np.unique(perts)}


def pair_cosine(drug_means, gene_means, pairs):
    """Cosine similarity of the mean drug and gene embeddings of each pair."""
    d = np.stack([drug_means[p[0]] for p in pairs])
    g = np.stack([gene_means[p[1]] for p in pairs])
    d /= np.clip(np.linalg.norm(d, axis=1, keepdims=True), 1e-12, None)
    g /= np.clip(np.linalg.norm(g, axis=1, keepdims=True), 1e-12, None)
    return np.einsum("ij,ij->i", d, g).astype(np.float32)
