"""Loading of the model inputs: expression (pseudobulk or cell-embedding) AnnData, LLM text embeddings and id maps."""

import anndata
import pandas as pd

from .. import paths
from ..io import load_obj

DATASETS = ("tahoe", "crispr")
METHODS = ("pseudobulk", "baseline", "scgpt", "pascient")  # baseline = pseudobulk with random text embeddings


def expression_path(dataset, method="pseudobulk"):
    """{dataset}_adata.h5ad: pseudobulk of make_pseudobulk.py (also the input of baseline);
    {dataset}_{method}.h5ad: one scGPT / PaSCient cell embedding per sample. Both in paths.DATA."""
    if method in ("pseudobulk", "baseline"):
        return {"tahoe": paths.TAHOE_PSEUDOBULK, "crispr": paths.CRISPR_PSEUDOBULK}[dataset]
    return paths.DATA / f"{dataset}_{method}.h5ad"


def load_expression(dataset, method="pseudobulk"):
    """Expression input of each (cell line, perturbation[, dose]) sample (expression_path); CRISPRi rows get
    obs["target_pathway"]."""
    adata = anndata.read_h5ad(expression_path(dataset, method))
    if dataset == "tahoe":
        return adata
    target_pathways = pd.read_pickle(paths.CRISPR / "target_pathways_new.pkl")
    adata.obs["target_pathway"] = adata.obs["target_gene"].map(target_pathways)
    return adata


def load_text_embeddings(dataset, method="pseudobulk", seed=0):
    """{"pert": {name: vector}, "cell": {name: vector}}, plus "pathway" for CRISPRi.

    LLM embeddings of the perturbation and cell line descriptions; method="baseline" uses random embeddings of the
    same size, one set per seed.
    """
    if dataset == "tahoe":
        if method == "baseline":
            pert = pd.read_pickle(paths.TAHOE / f"baseline_drug_emb{seed}.pickle")
            cell = pd.read_pickle(paths.TAHOE / f"baseline_cell_name_emb{seed}.pickle")
        else:
            pert = pd.read_pickle(paths.TAHOE / "ensem_emb_druginfo.pickle")
            cell = pd.read_pickle(paths.TAHOE / "ensem_emb_cellline.pickle")
        return {"pert": pert, "cell": cell}
    if method == "baseline":
        pert = pd.read_pickle(paths.CRISPR / f"baseline_target_gene_emb{seed}_new.pickle")
        cell = pd.read_pickle(paths.CRISPR / f"baseline_cell_type_emb{seed}_new.pickle")
    else:
        pert = pd.read_pickle(paths.CRISPR / "gene_emb.pkl")
        cell = pd.read_pickle(paths.CRISPR / "ensem_emb_cellline.pickle")
    return {"pert": pert, "cell": cell, "pathway": pd.read_pickle(paths.CRISPR / "ensem_emb_pathways.pickle")}


def embedding_dim(embeddings):
    return len(next(iter(embeddings.values())))


def text_dim(dataset, embeddings, ablate=None):
    """Size of the text-encoder input: [cell line, perturbation (+ pathway), dose (Tahoe)] minus the ablated part."""
    pert = embedding_dim(embeddings["pert"])
    if dataset == "crispr":
        pert += embedding_dim(embeddings["pathway"])
    size = (0 if ablate == "pert" else pert) + (0 if ablate == "ct" else embedding_dim(embeddings["cell"]))
    if dataset == "tahoe" and ablate != "dosage":
        size += 1
    return size


def load_id_maps(drug_target_loss=None):
    """(drug -> id(s), gene -> id(s)) used to label matching drug-gene pairs in the DGI loss.

    *drug* variants use the maps in which each drug holds its target ids, *gene* variants the maps in which each gene
    holds its drugs' ids (scripts/preprocessing/make_dgi_inputs.py); runs without a DGI loss need no ids (None, None).
    """
    if drug_target_loss is None:
        return None, None
    prefix = "new_" if "drug" in drug_target_loss else "dgi_"
    return load_obj(paths.MULTI / f"{prefix}drug_id.pkl"), load_obj(paths.MULTI / f"{prefix}gene_id.pkl")
