import lightning as L
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn

from ..evaluation.retrieval import get_epoch_acc, get_epoch_intra_acc
from .heads import ProjectionHead, concat_text
from .losses import contrastive_loss, same_row_mask, siglip_loss

DATASETS = ("tahoe", "crispr")
DGI_LOSSES = (
    "drug",
    "gene",  # joint training
    "ft_drug",
    "ft_gene",
    "ft_text_drug",
    "ft_text_gene",  # fine-tuning
)


def dgi_labels(drug_target_loss, tahoe_ids, crispr_ids):
    """(n_tahoe, n_crispr) 0/1 matrix of interacting drug-gene pairs.

    *drug* variants: each Tahoe sample carries its drug's target-gene ids (n_tahoe, k), each CRISPRi sample its gene id.
    *gene* variants: each CRISPRi sample carries the ids of the drugs targeting its gene (n_crispr, k).
    """
    if "drug" in drug_target_loss:
        return (tahoe_ids.unsqueeze(1) == crispr_ids.flatten().view(1, -1, 1)).any(dim=-1).int()
    return (crispr_ids.unsqueeze(-1) == tahoe_ids.flatten().view(1, 1, -1)).any(dim=1).int().T


class MultiCLIPModel(L.LightningModule):
    """PertOmni: Tahoe (drug) and CRISPRi (gene knockdown) contrastive models with a shared expression encoder.

    Each dataset has its own text encoder (text_projection1: Tahoe, text_projection2: CRISPRi); the expression encoder
    (image_projection1) and the last linear layer of the two text encoders (shared_fc) are shared.
    Training batches are [Tahoe batch, CRISPRi batch]; validation batches come one dataset at a time.

    hardmask: "both" = only samples of the same cell line are negatives, in both datasets.
    mask_norm: with hardmask, average the loss over the unmasked pairs.
    drug_target_loss: add a sigmoid loss between the Tahoe and CRISPRi embeddings of interacting drug-gene pairs
    (see DGI_LOSSES and dgi_labels); validation then scores this retrieval (AUROC + 1000 AUPRC).
    ft_retrieval: fine-tuning on the DGI loss alone.
    """

    def __init__(
        self,
        temperature_tahoe=(1.0,),
        temperature_crispr=(1.0,),
        diff_temp=False,
        image_embedding=256,
        text_embedding1=1536,
        text_embedding2=1536,
        lr=1e-4,
        losstype="clip",
        drug_target_loss=None,
        hardmask=None,
        mask_norm=False,
        ft_retrieval=False,
        ablate=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.image_projection1 = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection1 = ProjectionHead(embedding_dim=text_embedding1)
        self.text_projection2 = ProjectionHead(embedding_dim=text_embedding2)
        self.shared_fc = nn.Linear(128, 128)
        self.text_projection1.fc = self.shared_fc
        self.text_projection2.fc = self.shared_fc
        self.temperatures = [tuple(t[:2]) if diff_temp else (t[0],) for t in (temperature_tahoe, temperature_crispr)]
        self.lr = lr
        self.losstype = losstype
        self.drug_target_loss = drug_target_loss
        self.hardmask = hardmask
        self.mask_norm = mask_norm
        self.ft_retrieval = ft_retrieval
        self.ablate = ablate
        # per dataset: expression embeddings, text embeddings (+ ids and cell line embeddings for validation)
        self.train_outputs = {name: ([], []) for name in DATASETS}
        self.val_outputs = {name: ([], [], [], []) for name in DATASETS}

    @staticmethod
    def dataset_index(batch):
        return 1 if "kd_target" in batch else 0

    def text_features(self, batch, idx):
        if idx == 0:
            return concat_text(batch["cell"], batch["drug"], batch["exp"], ablate=self.ablate)
        return concat_text(batch["cell"], batch["kd_target"], ablate=self.ablate)

    def embed(self, batch):
        """(expression embeddings, text embeddings) of a batch of one dataset."""
        idx = self.dataset_index(batch)
        text_projection = self.text_projection1 if idx == 0 else self.text_projection2
        return self.image_projection1(batch["x"][:, 0, :]), text_projection(self.text_features(batch, idx))

    def _dataset_loss(self, batch, idx):
        expr_features = batch["x"][:, 0, :]
        text_features = self.text_features(batch, idx)
        expr_emb = self.image_projection1(expr_features)
        text_emb = (self.text_projection1 if idx == 0 else self.text_projection2)(text_features)
        # the DGI loss uses copies, as the embeddings returned for the epoch metrics
        embeddings = (expr_emb.clone(), text_emb.clone())
        mask = same_row_mask(batch["cell"]) if self.hardmask == "both" else None
        loss = contrastive_loss(
            self.losstype,
            expr_features,
            text_features,
            expr_emb,
            text_emb,
            self.temperatures[idx],
            mask,
            self.mask_norm,
        )
        return loss, embeddings

    def forward(self, batches):
        """Training step on [Tahoe batch, CRISPRi batch] -> ([Tahoe loss, CRISPRi loss], per-dataset embeddings)."""
        losses, embeddings = zip(*(self._dataset_loss(batch, idx) for idx, batch in enumerate(batches)))
        losses = list(losses)
        if self.drug_target_loss is not None:
            (tahoe_expr, tahoe_text), (crispr_expr, crispr_text) = embeddings
            labels = dgi_labels(self.drug_target_loss, batches[0]["id"], batches[1]["id"])
            dgi = siglip_loss(tahoe_expr @ crispr_expr.T / 5, labels, tahoe_text @ crispr_text.T / 5) * 0.5
            # fine-tuning back-propagates only the DGI loss; with ft_retrieval="text" the expression encoder is
            # frozen, so only the text term updates the model
            losses = [dgi, 0] if self.ft_retrieval else [losses[0] + dgi, losses[1]]
        return losses, embeddings

    def training_step(self, batch):
        losses, embeddings = self(batch)
        for name, (expr_emb, text_emb) in zip(DATASETS, embeddings):
            self.train_outputs[name][0].append(expr_emb)
            self.train_outputs[name][1].append(text_emb)
        self.log("train_tahoe_loss", losses[0])
        self.log("train_crispr_loss", losses[1])
        return losses[0] + losses[1]

    def validation_step(self, batch):
        idx = self.dataset_index(batch)
        loss, (expr_emb, text_emb) = self._dataset_loss(batch, idx)
        for outputs, value in zip(self.val_outputs[DATASETS[idx]], (expr_emb, text_emb, batch["id"], batch["cell"])):
            outputs.append(value)
        self.log("val_loss", loss)
        return loss

    def on_train_epoch_end(self):
        acc = {}
        for name, outputs in self.train_outputs.items():
            acc[name] = get_epoch_acc(torch.cat(outputs[0]), torch.cat(outputs[1]))
            self._log_acc(f"train_{name}", acc[name])
            for o in outputs:
                o.clear()
        self.log("train_acc@1", acc["tahoe"][0] + acc["crispr"][0])

    def on_validation_epoch_end(self):
        (tahoe_expr, tahoe_text, tahoe_ids, tahoe_cells), (crispr_expr, crispr_text, crispr_ids, crispr_cells) = (
            [torch.cat(o) for o in self.val_outputs[name]] for name in DATASETS
        )
        if self.drug_target_loss is None:
            if self.hardmask is None:
                tahoe_acc = get_epoch_acc(tahoe_expr, tahoe_text)
                crispr_acc = get_epoch_acc(crispr_expr, crispr_text)
            else:  # masked models are early-stopped on intra-cell-line accuracy
                tahoe_acc = get_epoch_intra_acc(tahoe_expr, tahoe_text, tahoe_cells)
                crispr_acc = get_epoch_intra_acc(crispr_expr, crispr_text, crispr_cells)
            self._log_acc("val_tahoe", tahoe_acc)
            self._log_acc("val_crispr", crispr_acc)
            self.log("val_acc@1", tahoe_acc[0] + crispr_acc[0])
        else:  # drug-gene retrieval from both the expression and the text embeddings
            y_true = dgi_labels(self.drug_target_loss, tahoe_ids, crispr_ids).cpu().flatten()
            y_probs = [
                cosine_similarity(a.detach().cpu().numpy(), b.detach().cpu().numpy()).flatten()
                for a, b in ((tahoe_expr, crispr_expr), (tahoe_text, crispr_text))
            ]
            auroc = sum(roc_auc_score(y_true, y_prob) for y_prob in y_probs)
            auprc = sum(average_precision_score(y_true, y_prob) for y_prob in y_probs)
            self.log("val_acc@1", auroc + auprc * 1000)
        for outputs in self.val_outputs.values():
            for o in outputs:
                o.clear()

    def _log_acc(self, stage, acc):
        for k in (1, 5, 10):
            self.log(f"{stage}_acc@{k}", acc[k - 1])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
