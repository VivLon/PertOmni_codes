"""Command-line parsing for the run configurations.

Each training entry point describes its options as a dataclass; `add_arguments` turns the fields into `--field`
options (help text and choices come from the field metadata), and the dataclass is saved as the run's config.json.
"""

import dataclasses
import typing
from pathlib import Path

from . import paths


def add_arguments(parser, config_cls):
    """Add one --option per dataclass field: bool -> flag, list[T] -> nargs="+", Optional[T] -> T with default None."""
    hints = typing.get_type_hints(config_cls)
    for f in dataclasses.fields(config_cls):
        kind, meta = hints[f.name], dict(f.metadata)
        required = f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
        default = None if required else (f.default_factory() if f.default is dataclasses.MISSING else f.default)
        if kind is bool:
            parser.add_argument(f"--{f.name}", action="store_true", help=meta.get("help"))
            continue
        if typing.get_origin(kind) is typing.Union:  # Optional[T]
            kind = next(arg for arg in typing.get_args(kind) if arg is not type(None))
        nargs = None
        if typing.get_origin(kind) is list:
            kind, nargs = typing.get_args(kind)[0], "+"
        parser.add_argument(
            f"--{f.name}",
            type=kind,
            nargs=nargs,
            default=default,
            required=required,
            choices=meta.get("choices"),
            help=meta.get("help"),
        )


def from_args(config_cls, args):
    return config_cls(**{f.name: getattr(args, f.name) for f in dataclasses.fields(config_cls)})


def add_run_arguments(parser, eval_only=True):
    """Options shared by every run: where the run folder is and whether an existing one may be replaced."""
    parser.add_argument("--save_name", required=True, help="name of the run folder, {results_dir}/{save_name}/")
    parser.add_argument("--results_dir", type=Path, default=paths.RESULTS, help="parent folder of the run folders")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing run with the same save_name")
    if eval_only:
        parser.add_argument(
            "--eval_only",
            action="store_true",
            help="skip training: export embeddings and scores from the run's checkpoints/best.ckpt and last.ckpt",
        )


def field(default=dataclasses.MISSING, help=None, choices=None):
    """dataclasses.field with CLI metadata; list defaults are copied per instance."""
    metadata = {"help": help, "choices": choices}
    if isinstance(default, list):
        return dataclasses.field(default_factory=lambda: list(default), metadata=metadata)
    return dataclasses.field(default=default, metadata=metadata)
