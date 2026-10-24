"""Perturbation prediction: expression of an unseen (perturbation, cell line) from retrieved neighbours or from
perturbation features, and its per-perturbation scores."""

from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity

CTRL = "non-targeting"


def l2_normalize(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def pearson_rows(a, b):
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    return (a * b).sum(axis=1) / np.sqrt((a**2).sum(axis=1) * (b**2).sum(axis=1))


def score(pred, true, ctrl, n_top, deg=None):
    """Pearson correlations per perturbation of the expression and of the change from control, over all genes and
    over the top DEGs. pred/true: (n_pert, G); ctrl: (G,); deg: (n_pert, n_top) gene indices, or None for the
    n_top genes with the largest |true - ctrl|."""
    true_delta, pred_delta = true - ctrl, pred - ctrl
    if deg is None:
        deg = np.argsort(-np.abs(true_delta), axis=1)[:, :n_top]

    def take(m):
        return np.take_along_axis(m, deg, axis=1)

    return {
        "corr": pearson_rows(pred, true),
        "corr_delta": pearson_rows(pred_delta, true_delta),
        "corr_top20": pearson_rows(take(pred), take(true)),
        "corr_delta_top20": pearson_rows(take(pred_delta), take(true_delta)),
    }


def knn_predict(q_emb, ref_emb, ref_X, Ks, temp):
    """For each K: softmax(cosine similarity / temp)-weighted mean expression of the top-K references."""
    sim = l2_normalize(q_emb) @ l2_normalize(ref_emb).T
    idx = np.argsort(-sim, axis=1)[:, : max(Ks)]
    top_sim = np.take_along_axis(sim, idx, axis=1)
    out = {}
    for K in Ks:
        w = np.exp((top_sim[:, :K] - top_sim[:, :1]) / temp)
        w /= w.sum(axis=1, keepdims=True)
        out[K] = np.einsum("qk,qkg->qg", w, ref_X[idx[:, :K]])
    return out


def per_target_mean(X, targets, order):
    return pd.DataFrame(X).groupby(targets).mean().loc[order].to_numpy()


def feature_predict(make_model, params, emb, ref_t, ref_y, q_t, fallback):
    """Fit make_model(p) on the reference perturbations that have a feature vector in `emb` and predict the query
    perturbations; queries without one get `fallback`. Returns ({p: predictions}, fraction of queries covered)."""
    has_ref = np.array([t in emb for t in ref_t])
    has_q = np.array([t in emb for t in q_t])
    Xr = np.stack([emb[t] for t in np.asarray(ref_t)[has_ref]])
    Xq = np.stack([emb[t] for t in np.asarray(q_t)[has_q]]) if has_q.any() else None
    out = {}
    for p in params:
        pred = np.tile(fallback, (len(q_t), 1))
        if Xq is not None:
            pred[has_q] = make_model(p).fit(Xr, ref_y[has_ref]).predict(Xq)
        out[p] = pred
    return out, has_q.mean()


# cross-validation protocol (one fold run per perturbation fold, retrieval within each cell line)


def cv_knn(adata, fold_embeddings, max_K, cell_lines, cl_key="cell_type"):
    """Top-max_K training-side neighbours of every held-out row, within its cell line.

    fold_embeddings: per fold, (train-side AnnData, its embeddings, test AnnData, test text embeddings).
    Returns (neighbour row positions in adata, similarities), (n_obs, max_K), -1 / NaN for rows never held out.
    """
    ranks = np.full((adata.n_obs, max_K), fill_value=-1, dtype=int)
    sims = np.full((adata.n_obs, max_K), fill_value=np.nan, dtype=float)
    for train_adata, train_emb, test_adata, test_emb in fold_embeddings:
        for cl in cell_lines:
            in_train = (train_adata.obs[cl_key] == cl).to_numpy()
            in_test = (test_adata.obs[cl_key] == cl).to_numpy()
            train_pos = adata.obs_names.get_indexer(train_adata.obs_names[in_train])
            test_pos = adata.obs_names.get_indexer(test_adata.obs_names[in_test])
            sim = torch.tensor(cosine_similarity(test_emb[in_test], train_emb[in_train]))
            vals, idx_local = sim.topk(max_K, dim=-1)
            ranks[test_pos] = train_pos[idx_local.numpy()]
            sims[test_pos] = vals.numpy()
    return ranks, sims


def cv_predict(adata, ranks, sims, K, temp, cell_lines, cl_key="cell_type", pert_key="target_gene"):
    """{cell line: {perturbation: softmax(sim/temp)-weighted mean of the top-K neighbours}}, averaged over the
    replicate rows of a perturbation."""
    targets = adata.obs[pert_key].to_numpy()
    cls = adata.obs[cl_key].to_numpy()
    preds = {cl: {} for cl in cell_lines}
    counts = {cl: defaultdict(int) for cl in cell_lines}
    for i in range(adata.n_obs):
        target, cl = targets[i], cls[i]
        if target == CTRL or cl not in preds or ranks[i, 0] < 0:
            continue
        weights = torch.softmax(torch.tensor(sims[i, :K]) / temp, dim=-1).unsqueeze(1).numpy()
        pred = np.dot(weights.T, adata.X[ranks[i, :K]])
        if target in preds[cl]:
            counts[cl][target] += 1
            pred = (preds[cl][target] * counts[cl][target] + pred) / (counts[cl][target] + 1)
        preds[cl][target] = pred
    return preds


def cv_scores(adata_cl, preds, degs=None, pert_key="target_gene"):
    """Per-perturbation MAE / MSE / Pearson correlations (over all genes and the top DEGs) of one cell line."""
    scores = []
    ctrl_expr = adata_cl[adata_cl.obs[pert_key] == CTRL].X.mean(axis=0, keepdims=True)
    for pert in set(adata_cl.obs[pert_key]).difference({CTRL}):
        pred_expr = preds[pert]
        true_expr = adata_cl[adata_cl.obs[pert_key] == pert].X.mean(axis=0, keepdims=True)
        pred_delta, true_delta = pred_expr - ctrl_expr, true_expr - ctrl_expr
        result = {
            "perturbation": pert,
            "mae": float(np.mean(np.abs(pred_expr - true_expr))),
            "mse": float(np.mean((pred_expr - true_expr) ** 2)),
            "corr": float(np.corrcoef(pred_expr, true_expr)[0, 1]),
            "corr_delta": float(np.corrcoef(pred_delta, true_delta)[0, 1]),
        }
        if pred_expr.ndim == 1:
            pred_expr = pred_expr.reshape(1, -1)
        if degs is not None:
            if pert not in degs:
                continue
            deg_idx = adata_cl.var_names.get_indexer(degs[pert])
            result |= {
                "mae_top20": float(np.mean(np.abs(pred_expr[:, deg_idx] - true_expr[:, deg_idx]))),
                "mse_top20": float(np.mean((pred_expr[:, deg_idx] - true_expr[:, deg_idx]) ** 2)),
                "corr_top20": float(np.corrcoef(pred_expr[:, deg_idx], true_expr[:, deg_idx])[0, 1]),
                "corr_delta_top20": float(np.corrcoef(pred_delta[:, deg_idx], true_delta[:, deg_idx])[0, 1]),
            }
        scores.append(result)
    return scores
