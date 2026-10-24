r"""Train PertOmni on Tahoe and CRISPRi together, or fine-tune it on drug-gene interactions.

Examples:
  # PertOmni (masked)
  python scripts/train_multi.py --save_name pertomni_seed0 --split_mode target --losstype siglip \
      --temp_tahoe 5 --temp_crispr 5 --hardmask both --export_full --seed 0
  # drug -> gene DGI fine-tuning of the text encoders
  python scripts/train_multi.py --save_name pertomni_dgi_drug_seed0 --split_mode target --losstype siglip \
      --temp_tahoe 5 --temp_crispr 5 --hardmask both --drug_target_loss ft_text_drug --ft_retrieval text \
      --init_from $PERTOMNI_RESULTS/pertomni_seed0 --seed 0
"""

import argparse

from pertomni.config import add_arguments, add_run_arguments, from_args
from pertomni.training.common import checkpoint_paths, prepare_run_dir
from pertomni.training.multi import MultiConfig, train_multi


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_arguments(parser)
    add_arguments(parser, MultiConfig)
    args = parser.parse_args()

    config = from_args(MultiConfig, args)
    run_dir = prepare_run_dir(args.results_dir, args.save_name, config, args.overwrite, existing_ok=args.eval_only)
    train_multi(config, run_dir, checkpoints=checkpoint_paths(run_dir) if args.eval_only else None)


if __name__ == "__main__":
    main()
