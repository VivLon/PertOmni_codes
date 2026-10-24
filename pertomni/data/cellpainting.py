"""RxRx3-core cell painting: well-level image embeddings paired with compound / CRISPR text embeddings."""

from collections import defaultdict

import h5py
import torch
from torch.utils.data import Dataset

IMG_DIM = 1536  # DINOv2-giant embedding per channel
N_CHANNELS = 5  # cell painting channels
CHANNEL_MODES = ("mean_all", "concat", "per_channel")


def load_img_embeddings(h5_path, channel_mode="mean_all"):
    """{well_id: image embedding}, aggregated over the 6 instances x 5 channels of each well (N x 6 x 5 x 1536).

    mean_all: mean over instances and channels -> (1536,); concat: mean over instances, channels concatenated ->
    (7680,); per_channel: mean over instances only -> (5, 1536).
    """
    assert channel_mode in CHANNEL_MODES, f"channel_mode must be one of {CHANNEL_MODES}"
    print(f"Loading image embeddings from {h5_path} (channel_mode={channel_mode})")
    with h5py.File(h5_path, "r") as f:
        well_ids = [i.decode("utf-8") for i in f["well_id"][:]]
        embs = f["embeddings"][:]
    if channel_mode == "mean_all":
        img = embs.mean(axis=(1, 2))
    elif channel_mode == "concat":
        img = embs.mean(axis=1).reshape(len(well_ids), -1)
    else:
        img = embs.mean(axis=1)
    return {wid: img[i] for i, wid in enumerate(well_ids)}


def pert_key(well_label):
    """Perturbation name of a well label: the well-position suffix is dropped ('Phloretin-AD37' -> 'Phloretin')."""
    return "-".join(well_label.split("-")[:-1])


class RxRx3Dataset(Dataset):
    """Wells of one perturbation type: {"x": image embedding, "drug" (COMPOUND) / "kd_target" (CRISPR): text
    embedding, "pert_label": perturbation name}."""

    def __init__(self, well_ids, img_embs, txt_embs, pert_type, well_to_pert):
        assert pert_type in ("COMPOUND", "CRISPR")
        self.pert_type = pert_type
        self.well_ids = [w for w in well_ids if w in img_embs and w in txt_embs]
        self.img_embs = img_embs
        self.txt_embs = txt_embs
        self.well_to_pert = well_to_pert

    def __len__(self):
        return len(self.well_ids)

    def __getitem__(self, idx):
        wid = self.well_ids[idx]
        sample = {"x": torch.from_numpy(self.img_embs[wid]), "pert_label": pert_key(self.well_to_pert[wid])}
        sample["drug" if self.pert_type == "COMPOUND" else "kd_target"] = torch.from_numpy(self.txt_embs[wid])
        return sample


def aggregate_by_pert(imgs, txts, pert_labels):
    """Mean-pool well embeddings (N, D) per perturbation -> (P, D), (P, D) and the sorted perturbation names."""
    pert_to_img, pert_to_txt = defaultdict(list), defaultdict(list)
    for img, txt, p in zip(imgs, txts, pert_labels):
        pert_to_img[p].append(img)
        pert_to_txt[p].append(txt)
    perts = sorted(pert_to_img)
    agg_imgs = torch.stack([torch.stack(pert_to_img[p]).mean(0) for p in perts])
    agg_txts = torch.stack([torch.stack(pert_to_txt[p]).mean(0) for p in perts])
    return agg_imgs, agg_txts, perts
