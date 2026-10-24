r"""Train the drug-gene interaction classifiers (one per embedding half and early-stopping mode).

Inputs: per-pair embeddings from make_dgi_embeddings.py (--emb_dir), or the published inputs (--paper_model;
vision_scores only has --use_emb all).
Example:
  python scripts/train_dgi.py --save_name dgi_drug2gene_seed0 --retrieval drug2gene \
      --emb_dir $PERTOMNI_RESULTS/pertomni_seed0/dgi_drug2gene --seed 0
"""

import argparse

from pertomni.config import add_arguments, add_run_arguments, from_args
from pertomni.training.common import prepare_run_dir
from pertomni.training.dgi import DGIConfig, train_snn


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_arguments(parser, eval_only=False)
    add_arguments(parser, DGIConfig)
    args = parser.parse_args()

    config = from_args(DGIConfig, args)
    run_dir = prepare_run_dir(args.results_dir, args.save_name, config, args.overwrite)
    for use_emb in config.use_emb:
        for es_mode in config.es_modes:
            train_snn(
                config.inputs(),
                run_dir,
                seed=config.seed,
                use_emb=use_emb,
                es_mode=es_mode,
                lr=config.lr,
                dropout=config.dropout,
                batch_size=config.batch_size,
                hidden_dim=config.hidden_dim,
                norm=config.norm,
            )


if __name__ == "__main__":
    main()
