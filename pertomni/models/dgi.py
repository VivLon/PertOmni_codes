"""Siamese classifier of drug-gene interactions from per-perturbation PertOmni embeddings."""

import lightning as L
import torch
from torch import nn


def make_norm(norm, dim):
    if norm == "batch":
        return nn.BatchNorm1d(dim)
    if norm == "layer":
        return nn.LayerNorm(dim)
    return nn.Identity()


class TargetSNN(nn.Module):
    """Shared encoder of the drug and gene embeddings; the classifier scores their element-wise product."""

    def __init__(self, input_dim=256, hidden_dim=64, dropout=0.1, norm="batch"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            make_norm(norm, hidden_dim),
            nn.LeakyReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            make_norm(norm, hidden_dim // 4),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, drug_emb, gene_emb):
        return self.classifier(self.net(drug_emb) * self.net(gene_emb))


def recall(y_pred, y_true):
    # predictions are (N, 1) and labels (N,): flatten so prediction i is compared with label i, not broadcast to (N, N)
    y_pred, y_true = y_pred.reshape(-1), y_true.reshape(-1)
    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    fn = ((y_pred == 0) & (y_true == 1)).sum().item()
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


class TargetModel(L.LightningModule):
    """Binary classifier of (drug, gene, label) batches; "val_acc" is the validation recall."""

    def __init__(self, lr=1e-4, input_dim=256, dropout=0.1, hidden_dim=64, norm="batch"):
        super().__init__()
        self.lr = lr
        self.encoder = TargetSNN(input_dim=input_dim, dropout=dropout, hidden_dim=hidden_dim, norm=norm)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.train_outputs, self.val_outputs = ([], []), ([], [])  # (predictions, labels)

    def forward(self, batch):
        if not torch.isfinite(batch[0]).all() or not torch.isfinite(batch[1]).all():
            raise ValueError("Found NaN/Inf in inputs")
        logits = self.encoder(batch[0], batch[1])
        loss = self.loss_fn(logits, batch[2].unsqueeze(1).float())
        pred = (torch.sigmoid(logits) >= 0.5).float()
        return loss, logits, pred

    def training_step(self, batch):
        loss, _, pred = self(batch)
        self.train_outputs[0].append(pred)
        self.train_outputs[1].append(batch[2])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        loss, _, pred = self(batch)
        self.val_outputs[0].append(pred)
        self.val_outputs[1].append(batch[2])
        self.log("val_loss", loss)
        return loss

    def on_train_epoch_end(self):
        self.log("train_acc", recall(torch.cat(self.train_outputs[0]), torch.cat(self.train_outputs[1])))
        for outputs in self.train_outputs:
            outputs.clear()

    def on_validation_epoch_end(self):
        self.log("val_acc", recall(torch.cat(self.val_outputs[0]), torch.cat(self.val_outputs[1])))
        for outputs in self.val_outputs:
            outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
