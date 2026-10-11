"""
Closes the diagnostic Javier asked for: correlates int_corr (Pearson
correlation of raw interest ratings) against embedding_cosine_similarity,
across the FULL dataset (not just the 15-row sample from
inspect_text_proxy.py), for both the plain and rich text proxies.

Why this matters: if `rich` text embeddings were faithfully encoding the
same interest-similarity signal that int_corr measures, we'd expect a
clearly positive correlation between int_corr and rich_sim. A near-zero
correlation (consistent with the 15-row sample: r=-0.10, not significant)
would support the template-dilution hypothesis -- that the repeated
boilerplate structure ("Loves X. Likes Y...") is drowning out the
fine-grained content differences MiniLM would need to preserve for this
correlation to show up.

This is diagnostic only -- it doesn't touch match/AUC/etc, just checks
whether the embedding is capturing the interest-similarity construct at
all.

Usage:
    python check_int_corr_vs_embedding.py            # mock, offline
    python check_int_corr_vs_embedding.py --real      # real MiniLM
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import pearsonr, spearmanr

from train_track1 import build_dataset
from models import EmbeddingModel, MockEmbeddingModel


def run(use_real_embedder: bool = False):
    plain_df = build_dataset(text_proxy="plain")
    rich_df = build_dataset(text_proxy="rich")

    embedder = EmbeddingModel() if use_real_embedder else MockEmbeddingModel()
    print(f"Using {'real MiniLM' if use_real_embedder else 'MOCK'} embedder, "
          f"n_plain={len(plain_df)}, n_rich={len(rich_df)}")

    plain_sim = embedder.pairwise_similarity(plain_df["own_text"].tolist(), plain_df["partner_text"].tolist())
    rich_sim = embedder.pairwise_similarity(rich_df["own_text"].tolist(), rich_df["partner_text"].tolist())

    # int_corr has some NaNs -- drop pairwise for a clean correlation
    def corr_report(name: str, sim: np.ndarray, int_corr: np.ndarray):
        mask = ~np.isnan(int_corr)
        n = mask.sum()
        r_p, p_p = pearsonr(int_corr[mask], sim[mask])
        r_s, p_s = spearmanr(int_corr[mask], sim[mask])
        print(f"\nint_corr vs {name} (n={n}):")
        print(f"  Pearson  r={r_p:.4f}  p={p_p:.2e}")
        print(f"  Spearman r={r_s:.4f}  p={p_s:.2e}")
        return r_p, p_p

    r_plain, p_plain = corr_report("plain_sim", plain_sim, plain_df["int_corr"].to_numpy())
    r_rich, p_rich = corr_report("rich_sim", rich_sim, rich_df["int_corr"].to_numpy())

    print("\n--- Interpretation ---")
    if abs(r_rich) < 0.1:
        print(f"rich_sim correlation with int_corr is near zero (r={r_rich:.3f}). This SUPPORTS "
              f"the template-dilution hypothesis: the rich embedding is not faithfully capturing "
              f"the same interest-similarity construct int_corr measures, despite being built from "
              f"the same underlying 17 columns.")
    else:
        print(f"rich_sim correlates with int_corr at r={r_rich:.3f} (p={p_rich:.2e}). If this is "
              f"clearly positive and significant, the template-dilution hypothesis is NOT well "
              f"supported -- the embedding IS capturing the interest-similarity signal; the earlier "
              f"AUC results would need a different explanation.")

    if not use_real_embedder:
        print("\n*** MOCK embedder -- these correlations are meaningless noise, this only checks "
              "the script runs correctly. Re-run with --real for the actual answer. ***")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    args = parser.parse_args()
    run(use_real_embedder=args.real)