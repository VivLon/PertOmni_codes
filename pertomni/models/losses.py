"""Contrastive losses between expression and text embeddings.

`logits` is the (n, n) similarity matrix of a batch divided by the temperature; with a second temperature
(diff_temp), `logits2` is the matrix in the other direction.
"""

import torch
import torch.nn.functional as F

LOSSES = ("siglip", "clip", "clip_reg", "cwcl")


def contrastive_loss(
    losstype, expr_features, text_features, expr_emb, text_emb, temperature, mask=None, mask_norm=False
):
    """Loss of one batch of matched (expression, text) samples.

    temperature: (t,), or (t_expr->text, t_text->expr) for two temperatures. mask: (n, n) 0/1, pairs with 0 are
    not used as negatives (siglip); mask_norm: average over the unmasked pairs instead of all pairs.
    """
    if len(temperature) == 2:
        logits = expr_emb @ text_emb.T / temperature[0]
        logits2 = text_emb @ expr_emb.T / temperature[1]
        temperature = (temperature[0] + temperature[1]) / 2
    else:
        logits = text_emb @ expr_emb.T / temperature[0]
        logits2 = None
        temperature = temperature[0]

    if losstype == "siglip":
        labels = 2 * torch.eye(logits.size(0), device=logits.device) - 1
        if mask is not None:
            labels = labels * mask
        return siglip_loss(logits, labels, logits2, valid=mask if mask_norm else None)

    expr_similarity = expr_emb @ expr_emb.T
    text_similarity = text_emb @ text_emb.T
    targets = soft_targets(expr_similarity, text_similarity, temperature)
    if losstype == "clip":
        return clip_loss(logits, targets)
    if losstype == "clip_reg":
        return clip_reg_loss(logits, targets, expr_features, text_features, expr_similarity, text_similarity)
    if losstype == "cwcl":
        return cwcl_loss(logits, targets, expr_features)
    raise ValueError(f"unknown losstype {losstype!r}")


def cross_entropy(preds, targets):
    """Row-wise cross entropy with soft targets, shape (n,)."""
    return (-targets * F.log_softmax(preds, dim=-1)).sum(1)


def soft_targets(expr_similarity, text_similarity, temperature):
    """CLIP targets smoothed by the within-modality similarities (expr_emb @ expr_emb.T, text_emb @ text_emb.T)."""
    return F.softmax((expr_similarity + text_similarity) / 2 * temperature, dim=-1)


def same_row_mask(x):
    """(n, n) float mask, 1 where two rows of x are identical, e.g. two samples of the same cell line (identical cell
    line embeddings) or of the same perturbation (identical text inputs)."""
    return (x.unsqueeze(0) == x.unsqueeze(1)).all(dim=-1).float()


def siglip_loss(logits, labels, logits2=None, valid=None):
    """Pairwise sigmoid loss. labels: +1 for matched pairs, -1 for negatives and 0 for masked-out pairs.

    With `valid` (0/1 mask), masked pairs are dropped and the loss is averaged over the remaining pairs, scaled by n
    to match the unmasked loss (sum over n*n pairs / n).
    """
    n = logits.size(0)
    if valid is not None:
        scale = n / valid.sum()
        if logits2 is None:
            return -torch.sum(F.logsigmoid(labels * logits) * valid) * scale
        return (
            -(torch.sum(F.logsigmoid(labels * logits) * valid) + torch.sum(F.logsigmoid(labels * logits2) * valid))
            * scale
            / 2
        )
    if logits2 is None:
        return -torch.sum(F.logsigmoid(labels * logits)) / n
    return -(torch.sum(F.logsigmoid(labels * logits)) / n + torch.sum(F.logsigmoid(labels * logits2)) / n) / 2


def clip_loss(logits, targets):
    texts_loss = cross_entropy(logits, targets)
    images_loss = cross_entropy(logits.T, targets.T)
    return ((images_loss + texts_loss) / 2.0).mean()


def clip_reg_loss(logits, targets, expr_features, text_features, expr_similarity, text_similarity):
    """CLIP loss + MSE between the embedding similarities and the input-feature similarities of each modality."""
    loss = clip_loss(logits, targets)
    return (
        loss
        + F.mse_loss(expr_similarity, expr_features @ expr_features.T)
        + F.mse_loss(text_similarity, text_features @ text_features.T)
    )


def cwcl_loss(logits, targets, expr_features):
    """Continuously weighted contrastive loss (expression -> text, weights = cosine similarity of the expression
    inputs mapped to [0, 1]) + the text -> expression CLIP term."""
    expr_norm = F.normalize(expr_features, p=2, dim=1)
    weights = (expr_norm @ expr_norm.T / 2 + 0.5).detach()
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
    n = logits.size(0)
    loss_cwcl = -torch.sum(weights * torch.log_softmax(logits.T, dim=1)) / n
    loss_clip = cross_entropy(logits, targets).mean()
    return loss_cwcl + loss_clip
