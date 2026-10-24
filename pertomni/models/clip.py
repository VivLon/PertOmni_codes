import lightning as L
import torch

from ..evaluation.retrieval import get_epoch_acc, get_epoch_intra_acc
from .heads import ProjectionHead, concat_text
from .losses import contrastive_loss, same_row_mask


class CLIPModel(L.LightningModule):
    """Contrastive model of one dataset: an expression encoder and a text encoder of [cell line, perturbation(, dose)].

    dataset: "tahoe" (text = cell line, drug, dose) or "crispr" (text = cell type, target gene + pathway).
    temperature: [t], or [t_expr->text, t_text->expr] with diff_temp.
    cellline_mask: "hard" = only samples of the same cell line are negatives (siglip); validation accuracy is then
    computed within each cell line.
    ablate: drop the perturbation ("pert"), cell line ("ct") or dose ("dosage") from the text input.
    """

    def __init__(
        self,
        temperature=(1.0,),
        diff_temp=False,
        image_embedding=256,
        text_embedding=1536,
        projection_dim=128,
        lr=1e-4,
        losstype="clip",
        dataset="tahoe",
        cellline_mask=None,
        ablate=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        # attribute names are the checkpoint keys: image_projection = expression encoder, text_projection = text encoder
        self.image_projection = ProjectionHead(embedding_dim=image_embedding, projection_dim=projection_dim)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding, projection_dim=projection_dim)
        self.temperature = tuple(temperature[:2]) if diff_temp else (temperature[0],)
        self.lr = lr
        self.losstype = losstype
        self.dataset = dataset
        self.cellline_mask = cellline_mask
        self.ablate = ablate
        self.train_outputs = ([], [])  # (expression, text) embeddings of the epoch
        self.val_outputs = ([], [], [])  # (expression, text) embeddings and cell line embeddings

    def text_features(self, batch):
        if self.dataset == "tahoe":
            return concat_text(batch["cell"], batch["drug"], batch["exp"], ablate=self.ablate)
        return concat_text(batch["cell"], batch["kd_target"], ablate=self.ablate)

    def embed(self, batch):
        """(expression embeddings, text embeddings) of a batch."""
        return self.image_projection(batch["x"][:, 0, :]), self.text_projection(self.text_features(batch))

    def forward(self, batch):
        expr_features = batch["x"][:, 0, :]
        text_features = self.text_features(batch)
        expr_emb = self.image_projection(expr_features)
        text_emb = self.text_projection(text_features)
        mask = same_row_mask(batch["cell"]) if self.cellline_mask == "hard" else None
        loss = contrastive_loss(self.losstype, expr_features, text_features, expr_emb, text_emb, self.temperature, mask)
        return loss, expr_emb, text_emb

    def training_step(self, batch, batch_idx):
        loss, expr_emb, text_emb = self(batch)
        self.train_outputs[0].append(expr_emb)
        self.train_outputs[1].append(text_emb)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, expr_emb, text_emb = self(batch)
        self.val_outputs[0].append(expr_emb)
        self.val_outputs[1].append(text_emb)
        self.val_outputs[2].append(batch["cell"])
        self.log("val_loss", loss)
        return loss

    def on_train_epoch_end(self):
        acc = get_epoch_acc(torch.cat(self.train_outputs[0]), torch.cat(self.train_outputs[1]))
        self._log_acc("train", acc)
        for outputs in self.train_outputs:
            outputs.clear()

    def on_validation_epoch_end(self):
        expr_emb, text_emb, cells = (torch.cat(outputs) for outputs in self.val_outputs)
        if self.cellline_mask == "hard":  # cross-cell-line pairs are masked, so early-stop on intra-cell-line accuracy
            acc = get_epoch_intra_acc(expr_emb, text_emb, cells)
        else:
            acc = get_epoch_acc(expr_emb, text_emb)
        self._log_acc("val", acc)
        for outputs in self.val_outputs:
            outputs.clear()

    def _log_acc(self, stage, acc):
        for k in (1, 5, 10):
            self.log(f"{stage}_acc@{k}", acc[k - 1])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
