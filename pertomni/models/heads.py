import torch
from torch import nn


class ProjectionHead(nn.Module):
    """Linear projection followed by a residual GELU-linear block and layer norm."""

    def __init__(self, embedding_dim, projection_dim=128, dropout=0.0):
        super().__init__()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.projection = nn.Linear(embedding_dim, projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        return self.layer_norm(x)


def concat_text(cell, pert, dose=None, ablate=None):
    """Text-encoder input [cell line, perturbation, dose]; ablate="ct" / "pert" / "dosage" drops that part."""
    parts = ([] if ablate == "ct" else [cell]) + ([] if ablate == "pert" else [pert])
    if dose is not None and ablate != "dosage":
        parts.append(dose)
    return torch.cat(parts, axis=1)
