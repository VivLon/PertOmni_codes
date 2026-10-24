"""Drug-gene interaction (DGI) inputs from DGIdb (interactions.tsv of https://dgidb.org/downloads), restricted to the
Tahoe drugs and CRISPRi target genes of the pseudobulk data (pertomni.data.dgi).

  python scripts/preprocessing/make_dgi_inputs.py --interactions data/gene_drug_retrieval/interactions.tsv

drug_target_retrieval/gt_target.pkl  drug -> target genes, ground truth of drug2gene
gene_drug_retrieval/gt_target.pkl    gene -> drugs, ground truth of gene2drug
multi_dataset/new_{drug,gene}_id.pkl id maps of the *drug* DGI losses (each drug holds its target-gene ids)
multi_dataset/dgi_{drug,gene}_id.pkl id maps of the *gene* DGI losses (each gene holds its drugs' ids)
"""

import argparse
from pathlib import Path

import anndata

from pertomni import paths
from pertomni.data.dgi import RETRIEVAL_DIRS, dgidb_ground_truth, padded_id_maps
from pertomni.io import save_obj

DRUG_CONTROL, GENE_CONTROL = "DMSO_TF", "non-targeting"


def obs_values(path, key):
    return sorted(set(anndata.read_h5ad(path, backed="r").obs[key].astype(str)))


def save_new(obj, path, overwrite):
    """save_obj, refusing to replace existing inputs (e.g. the published ones) unless overwrite is set."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    save_obj(obj, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interactions", type=Path, default=paths.DATA / "gene_drug_retrieval" / "interactions.tsv")
    parser.add_argument("--overwrite", action="store_true", help="replace existing files")
    args = parser.parse_args()

    drugs = obs_values(paths.TAHOE_PSEUDOBULK, "drug")
    genes = obs_values(paths.CRISPR_PSEUDOBULK, "target_gene")
    drug2genes, gene2drugs = dgidb_ground_truth(args.interactions, drugs, genes)
    print(len(drug2genes), "drugs and", len(gene2drugs), "genes with an interaction")
    save_new(drug2genes, paths.DATA / RETRIEVAL_DIRS["drug2gene"] / "gt_target.pkl", args.overwrite)
    save_new(gene2drugs, paths.DATA / RETRIEVAL_DIRS["gene2drug"] / "gt_target.pkl", args.overwrite)

    drug_id, gene_id = padded_id_maps(drug2genes, drugs, genes, DRUG_CONTROL, GENE_CONTROL)
    save_new(drug_id, paths.MULTI / "new_drug_id.pkl", args.overwrite)
    save_new(gene_id, paths.MULTI / "new_gene_id.pkl", args.overwrite)
    gene_id, drug_id = padded_id_maps(gene2drugs, genes, drugs, GENE_CONTROL, DRUG_CONTROL)
    save_new(drug_id, paths.MULTI / "dgi_drug_id.pkl", args.overwrite)
    save_new(gene_id, paths.MULTI / "dgi_gene_id.pkl", args.overwrite)


if __name__ == "__main__":
    main()
