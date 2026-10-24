"""LLM text embeddings of the perturbations, cell lines and pathways (OpenAI text-embedding-3-small, 1536 dims).

  cell lines  GPT-4o summary of the cell line (CELL_LINE_PROMPT), then embedded
  drugs       the "gpt-notes-approval" text of Tahoe's drug_metadata.parquet; the vehicle control is a zero vector
  genes       GenePT embeddings (Chen & Zou, 2024; model 3 text); "non-targeting" is a zero vector
  pathways    the MSigDB brief description of each GO:BP pathway; "non-targeting" is a zero vector
Needs the openai package and the OPENAI_API_KEY environment variable.
"""

import time

import numpy as np
import pandas as pd

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
CELL_LINE_PROMPT = "Please summarize the information of cell line: {} in one paragraph. Use academic language and include pathway information."
COMPOUND_PROMPT = "Please summarize the information of compound perturbation: {} in one paragraph. Use academic language and include pathway information."


def _client():
    from openai import OpenAI

    return OpenAI()  # reads OPENAI_API_KEY


def embed_texts(texts, model=EMBEDDING_MODEL, batch_size=128, retries=6):
    """{key: embedding} for {key: text}, in batches with exponential backoff."""
    client, keys, out = _client(), list(texts), {}
    for start in range(0, len(keys), batch_size):
        batch = keys[start : start + batch_size]
        for attempt in range(retries):
            try:
                response = client.embeddings.create(model=model, input=[texts[k].replace("\n", " ") for k in batch])
                break
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2**attempt)
        out.update({k: np.array(d.embedding, dtype=np.float32) for k, d in zip(batch, response.data)})
    return out


def describe(names, prompt, model="gpt-4o"):
    """{name: GPT summary} with prompt.format(name)."""
    client = _client()
    out = {}
    for name in names:
        completion = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt.format(name)}]
        )
        out[name] = completion.choices[0].message.content
    return out


def describe_cell_lines(cell_lines, model="gpt-4o"):
    """{cell line: GPT summary} (CELL_LINE_PROMPT)."""
    return describe(cell_lines, CELL_LINE_PROMPT, model)


def drug_texts(drug_metadata_path, drugs, control="DMSO_TF"):
    """{drug: description} from Tahoe's drug_metadata.parquet (names are matched after stripping whitespace)."""
    meta = pd.read_parquet(drug_metadata_path)
    notes = dict(zip(meta["drug"].str.strip(), meta["gpt-notes-approval"]))
    return {d: notes[d.strip()] for d in drugs if d != control}


def with_zero_controls(embeddings, controls, dim=EMBEDDING_DIM):
    """Add an all-zero embedding for each control (vehicle / non-targeting)."""
    return embeddings | {c: np.zeros(dim) for c in controls}


def genept_embeddings(genept_path, genes, control="non-targeting"):
    """{gene: GenePT embedding} for the genes that have one, plus a zero vector for the control."""
    genept = pd.read_pickle(genept_path)
    subset = {g: genept[g] for g in genes if g in genept}
    return with_zero_controls(subset, [control], dim=len(next(iter(subset.values()))))
