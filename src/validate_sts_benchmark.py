"""
Validates the embedding model against STS-Benchmark before trusting it on
Track 2 (per the project proposal).

STS-Benchmark: 1379 sentence pairs (test split) with a human-annotated
similarity score from 0 (unrelated) to 5 (equivalent meaning). We compute
MiniLM's cosine similarity for each pair and correlate it against the human
scores. Published all-MiniLM-L6-v2 numbers on this test set are typically
Spearman ~0.82-0.85 -- if your run comes in well below that, something is
off (wrong model name, embeddings not normalized, sentence order swapped,
etc.) and it's worth debugging before trusting Track 1/2 results.

Data source: data/sts_benchmark/sts-test.csv and sts-dev.csv, both the
official STS-Benchmark format (tab-separated: genre, source, year, id,
score, sentence1, sentence2), pulled from a GitHub mirror since the
original ixa2.si.ehu.eus host isn't reachable from this project's network
allowlist. Verified all 1379 test rows have the expected 7 fields (no
parsing quirks like the Speed Dating CSV needed).

Usage:
    python validate_sts_benchmark.py            # mock embedder, sanity-checks the pipeline
    python validate_sts_benchmark.py --real      # real MiniLM, needs internet access
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from models import EmbeddingModel, MockEmbeddingModel

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sts_benchmark"
COLUMNS = ["genre", "source", "year", "id", "score", "sentence1", "sentence2"]

# Published all-MiniLM-L6-v2 Spearman correlation on STS-Benchmark test,
# for sanity-checking your own run against.
PUBLISHED_MINILM_SPEARMAN_RANGE = (0.80, 0.86)


def load_split(split: str = "test") -> pd.DataFrame:
    path = DATA_DIR / f"sts-{split}.csv"
    df = pd.read_csv(path, sep="\t", header=None, names=COLUMNS, quoting=3)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["score", "sentence1", "sentence2"])
    dropped = before - len(df)
    if dropped:
        print(f"[validate_sts_benchmark] dropped {dropped} row(s) with missing score/sentence")
    return df


def evaluate(df: pd.DataFrame, embedder) -> dict:
    cosine_sim = embedder.pairwise_similarity(df["sentence1"].tolist(), df["sentence2"].tolist())
    human_score = df["score"].to_numpy()

    pearson_r, pearson_p = pearsonr(cosine_sim, human_score)
    spearman_r, spearman_p = spearmanr(cosine_sim, human_score)

    return {
        "n_pairs": len(df),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
    }


def run(use_real_embedder: bool = False, split: str = "test"):
    df = load_split(split)
    print(f"[validate_sts_benchmark] loaded {len(df)} pairs from the {split} split")

    embedder = EmbeddingModel() if use_real_embedder else MockEmbeddingModel()
    print(f"[validate_sts_benchmark] using {'real MiniLM' if use_real_embedder else 'MOCK'} embedder")

    results = evaluate(df, embedder)
    print(f"\nPearson r:  {results['pearson_r']:.4f}  (p={results['pearson_p']:.2e})")
    print(f"Spearman r: {results['spearman_r']:.4f}  (p={results['spearman_p']:.2e})")

    if use_real_embedder:
        lo, hi = PUBLISHED_MINILM_SPEARMAN_RANGE
        if lo <= results["spearman_r"] <= hi + 0.05:
            print(f"\n[OK] Spearman r is within/near the published range for all-MiniLM-L6-v2 "
                  f"({lo}-{hi}). The embedding model is behaving as expected -- safe to trust its "
                  f"scores in Track 1/2.")
        else:
            print(f"\n[WARNING] Spearman r ({results['spearman_r']:.4f}) is notably outside the "
                  f"published range for all-MiniLM-L6-v2 ({lo}-{hi}). Before trusting Track 1/2 "
                  f"results, check: correct model name, embeddings normalized, sentence1/sentence2 "
                  f"not swapped, no encoding issues in the input text.")
    else:
        print("\n*** NOTE: MockEmbeddingModel produces random similarity scores, so a near-zero "
              "correlation here is EXPECTED and does not indicate a problem -- this run only "
              "validates that the data loading and scoring pipeline executes correctly. Re-run "
              "with --real on a machine with internet access for the actual validation. ***")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true",
                         help="Use the real MiniLM embedder (requires internet access to huggingface.co)")
    parser.add_argument("--split", choices=["test", "dev"], default="test")
    args = parser.parse_args()

    run(use_real_embedder=args.real, split=args.split)
