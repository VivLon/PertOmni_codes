"""Text-side inputs of PertOmni (pertomni.preprocessing.text_embeddings / pathways). Needs OPENAI_API_KEY.

cell_lines  Tahoe and CRISPRi cell line embeddings  -> ensem_emb_cellline.pickle (+ ensem_describe_cellline.pickle)
drugs       Tahoe drug embeddings                    -> tahoe_100m/ensem_emb_druginfo.pickle
genes       CRISPRi target-gene (GenePT) embeddings  -> arc_challenge/gene_emb.pkl
pathways    target gene -> GO:BP pathway and pathway embeddings
            -> arc_challenge/target_pathways_new.pkl, arc_challenge/ensem_emb_pathways.pickle
"""

import argparse
from pathlib import Path

import anndata

from pertomni import paths
from pertomni.io import save_obj
from pertomni.preprocessing import pathways as pw
from pertomni.preprocessing import text_embeddings as te

TAHOE_PB = paths.TAHOE_PSEUDOBULK
CRISPR_PB = paths.CRISPR_PSEUDOBULK


def obs_values(path, key):
    return sorted(set(anndata.read_h5ad(path, backed="r").obs[key].astype(str)))


def save_new(obj, path, overwrite):
    """save_obj, refusing to replace existing inputs (e.g. the published ones) unless overwrite is set."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    save_obj(obj, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", choices=["cell_lines", "drugs", "genes", "pathways"])
    parser.add_argument("--drug_metadata", type=Path, default=paths.TAHOE / "drug_metadata.parquet")
    parser.add_argument(
        "--genept", type=Path, default=paths.GENEPT / "GenePT_gene_protein_embedding_model_3_text.pickle."
    )
    parser.add_argument("--msigdb_db", type=Path, default=paths.DATA / "database" / "msigdb_v2025.1.Hs.db")
    parser.add_argument("--overwrite", action="store_true", help="replace existing files")
    args = parser.parse_args()

    if "cell_lines" in args.inputs:
        for name, path, key, out_dir in (
            ("tahoe", TAHOE_PB, "cell_name", paths.TAHOE),
            ("crispr", CRISPR_PB, "cell_type", paths.CRISPR),
        ):
            descriptions = te.describe_cell_lines(obs_values(path, key))
            save_new(descriptions, out_dir / "ensem_describe_cellline.pickle", args.overwrite)
            save_new(te.embed_texts(descriptions), out_dir / "ensem_emb_cellline.pickle", args.overwrite)
    if "drugs" in args.inputs:
        drugs = obs_values(TAHOE_PB, "drug")
        embeddings = te.embed_texts(te.drug_texts(args.drug_metadata, drugs))
        save_new(
            te.with_zero_controls(embeddings, ["DMSO_TF"]), paths.TAHOE / "ensem_emb_druginfo.pickle", args.overwrite
        )
    if "genes" in args.inputs:
        genes = te.genept_embeddings(args.genept, obs_values(CRISPR_PB, "target_gene"))
        save_new(genes, paths.CRISPR / "gene_emb.pkl", args.overwrite)
    if "pathways" in args.inputs:
        assignment = pw.assign_pathways(obs_values(CRISPR_PB, "target_gene"), pw.go_bp_gmt())
        save_new(assignment, paths.CRISPR / "target_pathways_new.pkl", args.overwrite)
        descriptions = pw.go_bp_descriptions(args.msigdb_db)
        used = {p: descriptions[p] for p in set(assignment.values()) if p in descriptions}
        embeddings = te.with_zero_controls(te.embed_texts(used), ["non-targeting"])
        save_new(embeddings, paths.CRISPR / "ensem_emb_pathways.pickle", args.overwrite)


if __name__ == "__main__":
    main()
