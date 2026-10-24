import lightning as L
import torch
import torch.nn.functional as F
from torch import nn

from ..data.cellpainting import IMG_DIM, N_CHANNELS, aggregate_by_pert
from ..evaluation.retrieval import get_epoch_acc
from .heads import ProjectionHead
from .losses import cross_entropy, same_row_mask

DRUG_TEXT_DIM = 1536
CRISPR_TEXT_DIM = 3072
NAMES = ("drug", "gene")
TEXT_KEYS = ("drug", "kd_target")  # text input of the compound / CRISPR batches


class ChannelWiseProjectionHead(nn.Module):
    """Projects each cell painting channel with shared weights, then mean-pools over channels."""

    def __init__(self, channel_dim=IMG_DIM, n_channels=N_CHANNELS, projection_dim=128):
        super().__init__()
        self.n_channels = n_channels
        self.proj = ProjectionHead(embedding_dim=channel_dim, projection_dim=projection_dim)

    def forward(self, x):
        B, C, D = x.shape  # (batch, channels, channel_dim)
        return self.proj(x.reshape(B * C, D)).reshape(B, C, -1).mean(dim=1)


class RxRx3MultiCLIPModel(L.LightningModule):
    """Contrastive model of RxRx3-core: a shared cell painting encoder (image_projection), a compound text encoder
    (drug_projection) and a CRISPR text encoder (gene_projection, absent with has_crispr=False).

    Training batches are [compound batch, CRISPR batch] (or one compound batch); validation batches come one
    perturbation type at a time and are scored after averaging the wells of each perturbation.
    """

    def __init__(
        self,
        image_dim=IMG_DIM,
        drug_text_dim=DRUG_TEXT_DIM,
        crispr_text_dim=CRISPR_TEXT_DIM,
        projection_dim=128,
        lr=1e-4,
        losstype="siglip",
        temperature=0.07,
        weighted_loss=(1.0, 1.0),
        channel_mode="mean_all",
        has_crispr=True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.losstype = losstype
        self.temperature = temperature
        self.weighted_loss = list(weighted_loss)
        self.channel_mode = channel_mode
        self.has_crispr = has_crispr
        if channel_mode == "per_channel":
            self.image_projection = ChannelWiseProjectionHead(
                channel_dim=image_dim, n_channels=N_CHANNELS, projection_dim=projection_dim
            )
        else:
            self.image_projection = ProjectionHead(embedding_dim=image_dim, projection_dim=projection_dim)
        self.drug_projection = ProjectionHead(embedding_dim=drug_text_dim, projection_dim=projection_dim)
        self.gene_projection = (
            ProjectionHead(embedding_dim=crispr_text_dim, projection_dim=projection_dim) if has_crispr else None
        )
        # epoch outputs per perturbation type (index 0 = compound, 1 = CRISPR)
        self._train_img, self._train_txt = [[], []], [[], []]
        self._val_img, self._val_txt, self._val_pert = [[], []], [[], []], [[], []]

    def encode(self, batch, idx):
        """(image embeddings, text embeddings); idx 0 = compound batch, 1 = CRISPR batch."""
        img_emb = self.image_projection(batch["x"])
        if idx == 0:
            return img_emb, self.drug_projection(batch["drug"])
        assert self.gene_projection is not None, "gene_projection is None (compound-only model)"
        return img_emb, self.gene_projection(batch["kd_target"])

    def _loss(self, img_emb, txt_emb, txt_input, idx):
        """Loss of one batch of wells. txt_input: the text-encoder inputs; the wells of one perturbation share the
        same input, so with siglip every (image, text) pair of the same perturbation is a positive (otherwise two
        wells of one perturbation in a batch would push each image away from its own perturbation's text)."""
        logits = (img_emb @ txt_emb.T) / self.temperature
        w = self.weighted_loss[idx]
        if self.losstype == "siglip":
            labels = 2 * same_row_mask(txt_input) - 1
            return -torch.sum(F.logsigmoid(labels * logits)) / logits.size(0) * w
        targets = F.softmax((img_emb @ img_emb.T + txt_emb @ txt_emb.T) / 2 * self.temperature, dim=-1)
        imgs_loss = cross_entropy(logits.T, targets.T)
        txts_loss = cross_entropy(logits, targets)
        return ((imgs_loss + txts_loss) / 2).mean() * w

    def training_step(self, batch):
        if isinstance(batch, dict):  # compound-only
            batch = [batch]
        total_loss = torch.tensor(0.0, device=self.device)
        for idx, sub in enumerate(batch):
            img_emb, txt_emb = self.encode(sub, idx)
            loss = self._loss(img_emb, txt_emb, sub[TEXT_KEYS[idx]], idx)
            self._train_img[idx].append(img_emb.detach())
            self._train_txt[idx].append(txt_emb.detach())
            total_loss = total_loss + loss
            self.log(f"train_{NAMES[idx]}_loss", loss, on_step=False, on_epoch=True)
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch):
        idx = 0 if "drug" in batch else 1
        img_emb, txt_emb = self.encode(batch, idx)
        loss = self._loss(img_emb, txt_emb, batch[TEXT_KEYS[idx]], idx)
        self._val_img[idx].append(img_emb.detach())
        self._val_txt[idx].append(txt_emb.detach())
        self._val_pert[idx].extend(batch["pert_label"])
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        for idx, name in enumerate(NAMES):
            if self._train_img[idx]:
                acc = get_epoch_acc(torch.cat(self._train_img[idx]).cpu(), torch.cat(self._train_txt[idx]).cpu())
                self.log(f"train_{name}_acc@1", acc[0])
                self.log(f"train_{name}_acc@5", acc[4] if len(acc) > 4 else acc[-1])
        self._train_img, self._train_txt = [[], []], [[], []]

    def on_validation_epoch_end(self):
        val_acc = 0
        for idx, name in enumerate(NAMES):
            if self._val_img[idx]:
                imgs, txts, _ = aggregate_by_pert(
                    torch.cat(self._val_img[idx]).cpu(), torch.cat(self._val_txt[idx]).cpu(), self._val_pert[idx]
                )
                acc = get_epoch_acc(imgs, txts)
                self.log(f"val_{name}_acc@1", acc[0], prog_bar=True)
                self.log(f"val_{name}_acc@5", acc[4] if len(acc) > 4 else acc[-1])
                val_acc += acc[0]
        self._val_img, self._val_txt, self._val_pert = [[], []], [[], []], [[], []]
        self.log("val_acc", val_acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
