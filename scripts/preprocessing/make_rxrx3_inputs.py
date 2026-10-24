"""RxRx3-core inputs of scripts/train_rxrx3.py besides the image embeddings (pertomni.preprocessing.rxrx3). Text
embeddings need the openai package and OPENAI_API_KEY.

  python scripts/preprocessing/make_rxrx3_inputs.py

well labels      cellclip_data/rxrx3_results/mapping.pkl  {well_id: "<gene or treatment>-<address>"}
split            scratch cellclip_data/rxrx3_results/split/cellpainting-split-{train,valid,test}-imgpermol.csv;
                 test = the perturbations of --test_perts (drawn with --seed if the file does not exist)
text embeddings  cellclip_data/ensem_emb_drug.pickle      compound descriptions (--drug_descriptions, GPT-4o for
                                                          the missing ones), text-embedding-3-small; controls zero
                 cellclip_data/ensem_emb_gene.pickle      GenePT, text-embedding-3-large of the NCBI summary for the
                                                          genes without one; EMPTY_control zero
                 cellclip_data/ensem_emb_cellline.pickle  GPT-4o description of HUVEC, text-embedding-3-small
"""

import argparse
from pathlib import Path

import pandas as pd

from pertomni import paths
from pertomni.io import load_obj, save_obj
from pertomni.preprocessing import rxrx3
from pertomni.preprocessing import text_embeddings as te
from pertomni.training.rxrx3 import SPLIT_DIR, RxRx3Config

OUT = paths.CELLPAINTING
OUTPUTS = {
    "labels": OUT / "rxrx3_results" / "mapping.pkl",
    **{split: SPLIT_DIR / f"cellpainting-split-{split}-imgpermol.csv" for split in ("train", "valid", "test")},
    "drug": OUT / "ensem_emb_drug.pickle",
    "gene": OUT / "ensem_emb_gene.pickle",
    "cell_description": OUT / "ensem_describe_cellline.pickle",
    "cell": OUT / "ensem_emb_cellline.pickle",
}


def compound_embeddings(meta, descriptions_path):
    """Missing descriptions are written by GPT-4o and added to descriptions_path."""
    compounds = rxrx3.perturbations_of_type(meta, "COMPOUND")
    descriptions = load_obj(descriptions_path) if descriptions_path.exists() else {}
    missing = [c for c in compounds if not isinstance(descriptions.get(c), str)]
    if missing:
        print(len(missing), "compounds described by GPT-4o", flush=True)
        descriptions = descriptions | te.describe(missing, te.COMPOUND_PROMPT)
        save_obj(descriptions, descriptions_path)
    embeddings = te.embed_texts({c: descriptions[c] for c in compounds})
    return te.with_zero_controls(embeddings, rxrx3.CONTROLS)


def gene_embeddings(meta, genept_path):
    genes = rxrx3.perturbations_of_type(meta, "CRISPR")
    embeddings = te.genept_embeddings(genept_path, genes, control="EMPTY_control")
    missing = [g for g in genes if g not in embeddings]
    unknown = sorted(set(missing) - set(rxrx3.MISSING_GENE_SUMMARIES))
    if unknown:
        raise KeyError(f"no GenePT embedding nor summary for {unknown}; add them to MISSING_GENE_SUMMARIES")
    summaries = {g: rxrx3.MISSING_GENE_SUMMARIES[g] for g in missing}
    return embeddings | te.embed_texts(summaries, model="text-embedding-3-large")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--meta", type=Path, default=Path(RxRx3Config.meta_path))
    parser.add_argument("--test_perts", type=Path, default=OUT / "ft_rxrx3_results" / "test_perts.pkl")
    parser.add_argument("--drug_descriptions", type=Path, default=OUT / "drug_descriptions.pkl")
    parser.add_argument(
        "--genept", type=Path, default=paths.GENEPT / "GenePT_gene_protein_embedding_model_3_text.pickle."
    )
    parser.add_argument("--seed", default=0, type=int, help="draw of the validation (and test) perturbations")
    parser.add_argument("--overwrite", action="store_true", help="replace existing files")
    args = parser.parse_args()
    existing = [str(p) for p in OUTPUTS.values() if p.exists()]
    if existing and not args.overwrite:  # checked first, so that nothing is half written
        raise FileExistsError(f"{existing} exist; pass --overwrite to replace them")

    meta = pd.read_csv(args.meta, low_memory=False)
    save_obj(rxrx3.well_labels(meta), OUTPUTS["labels"])

    test = load_obj(args.test_perts) if args.test_perts.exists() else None
    print("test perturbations:", args.test_perts if test is not None else f"drawn (seed {args.seed})")
    for split, wells in rxrx3.perturbation_split(meta, test, seed=args.seed).items():
        OUTPUTS[split].parent.mkdir(parents=True, exist_ok=True)
        wells.to_csv(OUTPUTS[split], index=False)
        print(split, len(wells), "wells")

    save_obj(compound_embeddings(meta, args.drug_descriptions), OUTPUTS["drug"])
    save_obj(gene_embeddings(meta, args.genept), OUTPUTS["gene"])
    huvec = te.describe_cell_lines(["HUVEC"])
    save_obj(huvec, OUTPUTS["cell_description"])
    save_obj(te.embed_texts(huvec), OUTPUTS["cell"])


if __name__ == "__main__":
    main()
