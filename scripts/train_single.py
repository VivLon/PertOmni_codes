r"""Train a contrastive model on one dataset: PertOmni-chem (--dataset tahoe) or PertOmni-cri (--dataset crispr).

Example (PertOmni-cri, masked):
  python scripts/train_single.py --save_name cri_masked_seed0 --dataset crispr --split_mode target \
      --losstype siglip --temp 5 --cellline_mask hard --seed 0
"""

import argparse

from pertomni.config import add_arguments, add_run_arguments, from_args
from pertomni.training.common import checkpoint_paths, prepare_run_dir
from pertomni.training.single import SingleConfig, train_single


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_arguments(parser)
    parser.add_argument("--resume_ckpt", help="continue training the run from this checkpoint")
    add_arguments(parser, SingleConfig)
    args = parser.parse_args()

    config = from_args(SingleConfig, args)
    existing_ok = args.eval_only or args.resume_ckpt is not None
    run_dir = prepare_run_dir(args.results_dir, args.save_name, config, args.overwrite, existing_ok)
    train_single(
        config, run_dir, checkpoints=checkpoint_paths(run_dir) if args.eval_only else None, resume_ckpt=args.resume_ckpt
    )


if __name__ == "__main__":
    main()
