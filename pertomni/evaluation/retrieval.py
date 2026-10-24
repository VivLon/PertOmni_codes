"""Cross-modal retrieval metrics: row i of the expression embeddings matches row i of the text embeddings."""

import collections

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity


def get_sim_mtx(gene_embeds, text_embeds, scale=True):
    if not scale:
        return gene_embeds @ text_embeds.T
    return cosine_similarity(np.array(gene_embeds), np.array(text_embeds))


def eval_acc(N, true_indices, ranks, max_k=50):
    """top-k accuracy for k = 1..max_k."""
    return [(ranks[:, :k] == true_indices[:, None]).any(axis=1).sum() / N for k in range(1, max_k + 1)]


def eval_mrr(true_indices, ranks):
    rank_positions = np.where(ranks == true_indices[:, None])[1] + 1  # 1-based rank
    return np.mean(1.0 / rank_positions)


def eval_ndcg(N, true_indices, ranks, max_k=50):
    """nDCG@k for k = 1..max_k with a single relevant item per query."""
    discount = np.log2(np.arange(2, max_k + 2))
    idcg = 1.0 / np.log2(2)
    scores = []
    for k in range(1, max_k + 1):
        rel = (ranks[:, :k] == true_indices[:, None]).astype(int)
        scores.append(np.mean((rel / discount[:k]).sum(axis=1) / idcg))
    return scores


def _ranks(sim, scale):
    if scale:
        return np.argsort(-sim, axis=1)
    return np.array(sim.topk(sim.shape[0], dim=-1).indices)


def benchmark(gene_embeds, text_embeds, scale=False):
    """Retrieval scores in both directions (gene = expression -> text, text = text -> expression).

    Returns (scores at k = 1, 5, 10 and MRR, full curves for k = 1..10).
    """
    N = gene_embeds.shape[0]
    true_indices = np.arange(N)
    result, full_result = {}, {}
    for side, sim in (
        ("gene", get_sim_mtx(gene_embeds, text_embeds, scale)),
        ("text", get_sim_mtx(text_embeds, gene_embeds, scale)),
    ):
        ranks = _ranks(sim, scale)
        acc = eval_acc(N, true_indices, ranks, max_k=10)
        ndcg = eval_ndcg(N, true_indices, ranks, max_k=10)
        mrr = eval_mrr(true_indices, ranks)
        result |= {f"{side}_acc@{k}": acc[k - 1] for k in (1, 5, 10)}
        result |= {f"{side}_ndcg@{k}": ndcg[k - 1] for k in (1, 5, 10)}
        result[f"{side}_mrr"] = mrr
        full_result |= {f"{side}_acc": acc, f"{side}_ndcg": ndcg, f"{side}_mrr": mrr}
    return result, full_result


def benchmark_intra(obs, gene_embeds, text_embeds, group_key="cell_name", side="gene", scale=False, ks=(1, 5, 10)):
    """Retrieval within each cell line (rows grouped by obs[group_key], aligned with the embeddings).

    Returns (scores averaged over cell lines, scores weighted by the number of rows per cell line). If a cell line
    is too small for top-10 retrieval, every cell line is scored at k = 1 only.
    """
    groups = collections.defaultdict(list)
    for i, g in enumerate(obs[group_key]):
        groups[g].append(i)
    try:
        table = {
            grp: _group_scores(gene_embeds, text_embeds, idxs, scale, side, ks, max_k=10)
            for grp, idxs in groups.items()
        }
    except RuntimeError:
        table = {
            grp: _group_scores(gene_embeds, text_embeds, idxs, scale, side, (1,), max_k=1)
            for grp, idxs in groups.items()
        }
    df = pd.DataFrame(table)
    avg = df.mean(axis=1).to_dict()
    weighted_avg = dict.fromkeys(avg, 0)
    for key in avg:
        for grp, idxs in groups.items():
            weighted_avg[key] += len(idxs) * df.loc[key, grp]
    weighted_avg = {key: val / len(obs) for key, val in weighted_avg.items()}
    return avg, weighted_avg


def _group_scores(gene_embeds, text_embeds, idxs, scale, side, ks, max_k):
    sim = get_sim_mtx(gene_embeds[idxs], text_embeds[idxs], scale)
    true_indices = np.arange(len(idxs))
    ranks = sim.topk(sim.shape[0], dim=-1).indices.cpu().numpy()
    acc = eval_acc(len(idxs), true_indices, ranks, max_k=max_k)
    ndcg = eval_ndcg(len(idxs), true_indices, ranks, max_k=max_k)
    res = {}
    for k in ks:
        res[f"{side}_acc@{k}"] = acc[k - 1]
        res[f"{side}_ndcg@{k}"] = ndcg[k - 1]
    res[f"{side}_mrr"] = eval_mrr(true_indices, ranks)
    return res


def get_epoch_acc(all_image_emb, all_text_emb):
    """top-1..10 retrieval accuracy averaged over both directions (training / validation metric)."""
    sim_gene = get_sim_mtx(all_image_emb, all_text_emb, scale=False)
    sim_text = get_sim_mtx(all_text_emb, all_image_emb, scale=False)
    N = sim_gene.shape[0]
    true_indices = np.arange(N)
    top = min(10, all_image_emb.shape[0])
    acc_gene = eval_acc(N, true_indices, sim_gene.topk(top, dim=-1).indices.cpu().numpy(), max_k=top)
    acc_text = eval_acc(N, true_indices, sim_text.topk(top, dim=-1).indices.cpu().numpy(), max_k=top)
    return (np.array(acc_gene) + np.array(acc_text)) / 2


def get_epoch_intra_acc(all_image_emb, all_text_emb, all_cells, max_k=10):
    """get_epoch_acc within each cell line (rows with identical cell embeddings), weighted by group size."""
    _, cell_ids = torch.unique(all_cells, dim=0, return_inverse=True)
    groups = collections.defaultdict(list)
    for i, g in enumerate(cell_ids.tolist()):
        groups[g].append(i)
    table = {}
    for grp, idxs in groups.items():
        acc = get_epoch_acc(all_image_emb[idxs], all_text_emb[idxs])
        table[grp] = np.pad(
            acc, (0, max_k - len(acc)), mode="edge"
        )  # groups smaller than max_k: acc@k = 1 for k >= size
    df = pd.DataFrame(table)
    weighted_acc = sum(len(idxs) * df[grp].values for grp, idxs in groups.items())
    return weighted_acc / all_image_emb.shape[0]
