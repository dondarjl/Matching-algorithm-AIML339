"""
Quick, cheap sanity check on the text proxies -- NOT the full pipeline.
Shows, for a handful of pairs: both texts (plain and rich), their real
MiniLM cosine similarity under each proxy, the actual match outcome, and
int_corr (for eyeballing only -- not a validation, see README).

Runs on a small sample (default 15 pairs) so it's fast even with the real
embedder -- use this to sanity-check the rich text proxy before committing
to a full --real run of train_track1.py / cross_validate_track1.py.

Usage:
    python inspect_text_proxy.py            # mock embedder, offline
    python inspect_text_proxy.py --real      # real MiniLM, needs internet
    python inspect_text_proxy.py --real -n 25 --seed 1
"""
from __future__ import annotations

import argparse

import pandas as pd

from train_track1 import build_dataset
from models import EmbeddingModel, MockEmbeddingModel


def run(use_real_embedder: bool = False, n: int = 15, seed: int = 0):
    plain_df = build_dataset(text_proxy="plain")
    rich_df = build_dataset(text_proxy="rich")
    # both share the same row order/index post-dedupe, so align by position
    sample_idx = plain_df.sample(n=n, random_state=seed).index

    embedder = EmbeddingModel() if use_real_embedder else MockEmbeddingModel()
    print(f"Using {'real MiniLM' if use_real_embedder else 'MOCK'} embedder, n={n} sample pairs\n")

    plain_own = plain_df.loc[sample_idx, "own_text"].tolist()
    plain_partner = plain_df.loc[sample_idx, "partner_text"].tolist()
    rich_own = rich_df.loc[sample_idx, "own_text"].tolist()
    rich_partner = rich_df.loc[sample_idx, "partner_text"].tolist()

    plain_sim = embedder.pairwise_similarity(plain_own, plain_partner)
    rich_sim = embedder.pairwise_similarity(rich_own, rich_partner)

    out = pd.DataFrame({
        "match": plain_df.loc[sample_idx, "match"].to_numpy(),
        "int_corr": plain_df.loc[sample_idx, "int_corr"].round(2).to_numpy(),
        "plain_sim": plain_sim.round(3),
        "rich_sim": rich_sim.round(3),
    })
    pd.set_option("display.width", 120)
    print(out.to_string())

    print("\n--- Example pair (first row of the sample) ---")
    i = sample_idx[0]
    print(f"match={plain_df.loc[i, 'match']}  int_corr={plain_df.loc[i, 'int_corr']}")
    print(f"\n[plain] own:     {plain_df.loc[i, 'own_text']}")
    print(f"[plain] partner: {plain_df.loc[i, 'partner_text']}")
    print(f"[plain] similarity: {plain_sim[list(sample_idx).index(i)]:.3f}")
    print(f"\n[rich]  own:     {rich_df.loc[i, 'own_text']}")
    print(f"[rich]  partner: {rich_df.loc[i, 'partner_text']}")
    print(f"[rich]  similarity: {rich_sim[list(sample_idx).index(i)]:.3f}")

    if not use_real_embedder:
        print("\n*** MOCK embedder -- similarities are meaningless noise, this only checks "
              "the script runs. Re-run with --real. ***")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("-n", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(use_real_embedder=args.real, n=args.n, seed=args.seed)
