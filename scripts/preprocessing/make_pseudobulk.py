"""Pseudobulk expression inputs of PertOmni (summed counts per sample, then normalize_total + log1p).

Tahoe-100M: one sample per (drug and dose, cell line) of each plate file, plates concatenated.
  python scripts/preprocessing/make_pseudobulk.py --dataset tahoe --inputs plate*_filt_Vevo_Tahoe100M_*.h5ad.gz \
      --out $PERTOMNI_DATA/tahoe_adata.h5ad
CRISPRi: one sample per (target gene, cell type) of each single-cell Perturb-seq file (name=path; the name is the cell
type, except k562_gwps which is k562): Replogle et al. K562/RPE1/HepG2/Jurkat and HCT116 of the Virtual Cell Challenge;
targets without a GenePT embedding are dropped.
  python scripts/preprocessing/make_pseudobulk.py --dataset crispr --inputs k562=K562_essential.h5ad k562_gwps=K562_gwps.h5ad \
      rpe1=rpe1.h5ad hepg2=hepg2.h5ad jurkat=jurkat.h5ad hct116=HCT116.h5ad \
      --out $PERTOMNI_DATA/crispr_adata.h5ad
"""

import argparse
from pathlib import Path

import anndata
import pandas as pd

from pertomni import paths
from pertomni.preprocessing.pseudobulk import (
    concat,
    crispr_pseudobulk,
    keep_embedded_targets,
    normalize,
    tahoe_plate_pseudobulk,
)


def to_gene_symbols(adata):
    """Index genes by var["gene_name"] (Ensembl ids in the Perturb-seq files); duplicated symbols are averaged."""
    if "gene_name" not in adata.var:
        return adata
    names = adata.var["gene_name"].astype(str).to_numpy()
    X = pd.DataFrame(adata.X, columns=names).T.groupby(level=0, sort=False).mean().T
    return anndata.AnnData(X.to_numpy(), obs=adata.obs, var=pd.DataFrame(index=X.columns))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["tahoe", "crispr"])
    parser.add_argument(
        "--inputs", required=True, nargs="+", help="tahoe: plate files; crispr: name=path per Perturb-seq file"
    )
    parser.add_argument(
        "--genept", type=Path, default=paths.GENEPT / "GenePT_gene_protein_embedding_model_3_text.pickle."
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if args.dataset == "tahoe":
        parts = []
        for path in args.inputs:
            part = tahoe_plate_pseudobulk(path)
            print(path, part.shape, flush=True)
            parts.append(part)
        adata = concat(parts)
    else:
        parts = []
        for item in args.inputs:
            name, path = item.split("=", 1)
            part = to_gene_symbols(crispr_pseudobulk(path, cell_type="k562" if name == "k562_gwps" else name))
            print(name, part.shape, flush=True)
            parts.append(part)
        adata = keep_embedded_targets(concat(parts), pd.read_pickle(args.genept))
    adata = normalize(adata)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.out)
    print("saved", adata.shape, args.out)


if __name__ == "__main__":
    main()
