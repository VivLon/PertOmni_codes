"""LangPert (https://github.com/novonordisk-research/LangPert) with gpt-oss-120b on the CRISPRi target split, one GPU
process with the model loaded once. Inputs: langpert_prep.py. Runs in an environment with LangPert and transformers.

Order (resumable: a pass whose expr_results.pkl exists is loaded, not regenerated):
  1. test: single-pass and refined LangPert                              (pool = train + valid)
  2. val:  PertOmni_text-narrowed single-pass and refined, each --narrow_Ks (pool = train)
  3. select narrow_K = best mean validation correlation of the refined pass over the cell lines
  4. test: narrowed single-pass and refined at the selected narrow_K      (pool = train + valid)
Each pass writes llm_response_results.pkl, expr_results.pkl and run_info.json to {out_dir}/{phase}/{variant}/{cl}/.
Attention: the FlashAttention-3 kernel on Hopper GPUs (as LangPert's GPTOSSBackend), flex_attention elsewhere.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from langpert import LangPert
from langpert.backends.gpt_oss import GPTOSSBackend
from langpert.core.model import PredictionResult
from transformers import AutoModelForCausalLM, AutoTokenizer

from pertomni import paths
from pertomni.io import load_obj, save_json, save_obj


class Backend(GPTOSSBackend):
    def _load_model(self):
        major, _ = torch.cuda.get_device_capability()
        attn = "kernels-community/vllm-flash-attn3" if major == 9 else "flex_attention"
        log(f"loading {self.model_name} on {torch.cuda.get_device_name()}, attn_implementation={attn}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            attn_implementation=attn,
            device_map=self.device_map,
            cache_dir=self.cache_dir,
            torch_dtype="auto",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.generation_config["pad_token_id"] = self.tokenizer.eos_token_id


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(backend, inp, out_dir, phase, cl, variant, prompt, candidates, batch_size, single_pass=None):
    """One LangPert pass. Queries lost to failed batches are retried with half the batch size; queries that still
    fail get the non-control mean."""
    d = out_dir / phase / variant / cl
    if (d / "expr_results.pkl").exists():
        log(f"done before: {d}")
        return load_obj(d / "llm_response_results.pkl")
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = LangPert(backend=backend, observed_effects=inp["observed_effects"], fallback_mean=inp["fallback_mean"])
    queries, results = inp["queries"], {}
    while True:
        todo = [g for g in queries if g not in results]
        if not todo or batch_size < 1:
            break
        log(f"{phase}/{variant}/{cl}: {len(todo)} queries, batch size {batch_size}")
        results.update(
            model.predict_batch(
                target_genes=todo,
                prompt_template=prompt,
                cell_line=cl,
                batch_size=max(batch_size, 2),
                candidate_genes=candidates,
                single_pass_gene_dict=single_pass,
            )
        )
        torch.cuda.empty_cache()
        batch_size //= 2
    missing = [g for g in queries if g not in results]
    for g in missing:  # keep every query, so that refinement and scoring see all of them
        results[g] = PredictionResult(
            gene=g, knn_genes=[], prediction=inp["fallback_mean"].copy(), llm_response="", reasoning=None
        )
    n_fallback = sum(len(r.knn_genes) == 0 for r in results.values())
    save_obj(results, d / "llm_response_results.pkl")
    save_obj(model.get_predictions_matrix(results), d / "expr_results.pkl")
    save_json(
        dict(
            minutes=(time.time() - t0) / 60,
            n_queries=len(queries),
            n_failed_generation=len(missing),
            n_fallback_mean=n_fallback,
        ),
        d / "run_info.json",
    )
    log(f"done {d} in {(time.time() - t0) / 60:.1f} min; {len(missing)} failed, {n_fallback} non-control mean")
    return results


def val_corr(results, inp):
    return float(np.nanmean([np.corrcoef(results[g].prediction, inp["true"][g])[0, 1] for g in inp["queries"]]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", type=Path, default=paths.RESULTS / "langpert" / "inputs.pkl")
    parser.add_argument("--out_dir", type=Path, default=paths.RESULTS / "langpert")
    parser.add_argument("--cell_lines", default=["rpe1", "hepg2", "jurkat"], nargs="+")
    parser.add_argument("--narrow_Ks", default=[25, 50, 100, 200, 500], type=int, nargs="+")
    parser.add_argument("--bs_plain", default=32, type=int, help="batch size with the full candidate pool")
    parser.add_argument("--bs_narrow", default=96, type=int, help="batch size with narrowed candidates")
    parser.add_argument("--cache_dir", help="Hugging Face cache of the model weights")
    args = parser.parse_args()

    inputs = load_obj(args.inputs)
    backend = Backend(model_name="openai/gpt-oss-120b", device_map="auto", cache_dir=args.cache_dir)
    t0 = time.time()  # fail fast if the attention / MXFP4 kernels do not run on this GPU
    out = backend.generate_batch(["Name one human gene.", "Say OK."], system_prompt=None, max_new_tokens=64)
    log(f"smoke test ok in {time.time() - t0:.1f}s: {[o[-80:] for o in out]}")

    def both_passes(phase, cl, variant, candidates, batch_size):
        inp = inputs[(phase, cl)]
        prefix = f"{variant}_" if variant else ""
        common = dict(backend=backend, inp=inp, out_dir=args.out_dir, phase=phase, cl=cl, candidates=candidates)
        sp = run(**common, variant=f"{prefix}single_pass", prompt="cell_line_specific", batch_size=batch_size)
        return run(
            **common, variant=f"{prefix}refined", prompt="cell_line_refine", batch_size=batch_size, single_pass=sp
        )

    def narrowed(phase, cl, K):
        return {g: c[:K] for g, c in inputs[(phase, cl)]["narrow"].items()}

    # 1. test, plain LangPert over the full pool
    for cl in args.cell_lines:
        both_passes("test", cl, "", list(inputs[("test", cl)]["observed_effects"]), args.bs_plain)
    # 2. validation sweep of the narrowing size
    scores = {}
    for K in args.narrow_Ks:
        for cl in args.cell_lines:
            refined = both_passes("val", cl, f"top{K}", narrowed("val", cl, K), args.bs_narrow)
            scores.setdefault(K, {})[cl] = val_corr(refined, inputs[("val", cl)])
        log(f"validation corr, narrow_K={K}: {scores[K]}")
    # 3. selection
    best_K = max(scores, key=lambda K: np.mean(list(scores[K].values())))
    save_json(
        dict(selected_narrow_K=best_K, val_corr={str(k): v for k, v in scores.items()}),
        args.out_dir / "narrow_K_selection.json",
    )
    log(f"selected narrow_K={best_K}")
    # 4. test, PertOmni_text-narrowed LangPert
    for cl in args.cell_lines:
        both_passes("test", cl, f"top{best_K}", narrowed("test", cl, best_K), args.bs_narrow)
    log("all done")


if __name__ == "__main__":
    main()
