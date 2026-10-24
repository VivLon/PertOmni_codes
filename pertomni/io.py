import json
import os
import pickle
from pathlib import Path


def save_obj(obj, path):
    os.makedirs(os.path.dirname(os.fspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(os.fspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
        f.write("\n")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def run_file(run_dir, name):
    """Path of a file exported by a run. Runs from before the CRISPRi naming wrote "arc" in its place
    (e.g. full_arc_text_embeds.pkl for full_crispr_text_embeds.pkl); those are found too."""
    path = Path(run_dir) / name
    legacy = Path(run_dir) / name.replace("crispr", "arc")
    return legacy if not path.exists() and legacy.exists() else path
