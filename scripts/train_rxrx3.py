"""Train the cell painting contrastive model on RxRx3-core (compound and CRISPR wells).

Example:
  python scripts/train_rxrx3.py --save_name rxrx3_seed0 --losstype clip --seed 0
"""

import argparse

from pertomni.config import add_arguments, add_run_arguments, from_args
from pertomni.training.common import checkpoint_paths, prepare_run_dir
from pertomni.training.rxrx3 import RxRx3Config, train_rxrx3


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_arguments(parser)
    add_arguments(parser, RxRx3Config)
    args = parser.parse_args()

    config = from_args(RxRx3Config, args)
    run_dir = prepare_run_dir(args.results_dir, args.save_name, config, args.overwrite, existing_ok=args.eval_only)
    train_rxrx3(config, run_dir, checkpoint=checkpoint_paths(run_dir)[0] if args.eval_only else None)


if __name__ == "__main__":
    main()
