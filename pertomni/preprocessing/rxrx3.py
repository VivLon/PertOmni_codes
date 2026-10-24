"""RxRx3-core inputs of the cell painting model (pertomni.training.rxrx3): well labels and perturbation split.

A well's perturbation is its gene (CRISPR wells) or its treatment (compound wells, including the two controls).
"""

import numpy as np
import pandas as pd

CONTROLS = ("EMPTY_control", "CRISPR_control")
# NCBI summaries of the CRISPR targets that have no GenePT embedding
MISSING_GENE_SUMMARIES = {
    "C7orf26": "Predicted to act upstream of or within brain development; eye development; and mRNA splicing, via "
    "spliceosome. Part of integrator complex. Is active in nucleus.",
    "POLR2B": "This gene encodes the second largest subunit of RNA polymerase II (Pol II), a DNA-dependent RNA "
    "polymerase that catalyzes the transcription of DNA into precursors of mRNA, snRNA and microRNA. This subunit and "
    "the largest subunit form opposite sides of the center cleft of Pol II. Deletion of the flap loop region of this "
    "subunit results in a decrease in the rate of transcriptional elongation. Alternative splicing results in "
    "multiple transcript variants.",
    "INTS2": "INTS2 is a subunit of the Integrator complex, which associates with the C-terminal domain of RNA "
    "polymerase II large subunit (POLR2A; MIM 180660) and mediates 3-prime end processing of small nuclear RNAs U1 "
    "(RNU1; MIM 180680) and U2 (RNU2; MIM 180690)",
}


def perturbations(meta):
    """Perturbation of every well (row of metadata_rxrx3_core.csv)."""
    return meta["gene"].where(meta["gene"].notna(), meta["treatment"]).astype(str)


def well_labels(meta):
    """{well_id: "<perturbation>-<address>"}, e.g. 'Phloretin-AD37'."""
    return dict(zip(meta["well_id"], perturbations(meta) + "-" + meta["address"].astype(str)))


def perturbation_split(meta, test=None, valid_frac=0.1, test_frac=0.2, seed=0):
    """{split: DataFrame with one SAMPLE_KEY (well_id) per well} for train / valid / test (CellCLIP format).

    Split by perturbation: test = the given perturbations (else test_frac of them, drawn), valid = valid_frac of all
    perturbations drawn from the rest, train = the remaining perturbations and the controls.
    """
    pert = perturbations(meta)
    perts = sorted(set(pert) - set(CONTROLS))
    rng = np.random.default_rng(seed)
    test = set(test) if test is not None else set(rng.choice(perts, int(test_frac * len(perts)), replace=False))
    rest = sorted(set(perts) - test)
    valid = set(rng.choice(rest, int(valid_frac * len(perts)), replace=False))
    groups = {"train": (set(rest) - valid) | set(CONTROLS), "valid": valid, "test": test}
    return {s: pd.DataFrame({"SAMPLE_KEY": meta.loc[pert.isin(g), "well_id"].to_numpy()}) for s, g in groups.items()}


def perturbations_of_type(meta, perturbation_type):
    """Sorted non-control perturbations of the COMPOUND or CRISPR wells."""
    pert = perturbations(meta)[meta["perturbation_type"] == perturbation_type]
    return sorted(set(pert) - set(CONTROLS))
