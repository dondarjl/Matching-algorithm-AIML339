""" 
Track 2 domain-transfer demonstration: run the same feature pipeline used
for Track 1 on the synthetic Conecta profiles, and inspect the top-ranked
pairs by an unsupervised composite score.

There's no ground-truth `match` label for synthetic profiles (unlike Speed
Dating), and two Conecta features (course_overlap_jaccard, same_cohort) have
no Speed Dating equivalent, so Track 1's fitted fusion weights cannot be
transferred here. This track isn't about training/testing a classifier --
it's a qualitative check that the four Conecta pairwise features (including
the embedding similarity) compose into sensible-looking rankings on
genuinely different, Conecta-shaped data.

Usage:
    python demo_track2.py            # mock embedder, fully offline
    python demo_track2.py --real     # real MiniLM, needs internet access
"""
from __future__ import annotations

import argparse

import pandas as pd

from generate_synthetic_profiles import load_profiles
from features import build_pairwise_dataframe
from models import EmbeddingModel, MockEmbeddingModel


def run(use_real_embedder: bool = False):
    profiles = load_profiles()
    print(f"[demo_track2] loaded {len(profiles)} synthetic profiles")

    pairs = build_pairwise_dataframe(profiles)
    print(f"[demo_track2] {len(pairs)} unique pairs")

    embedder = EmbeddingModel() if use_real_embedder else MockEmbeddingModel()
    print(f"[demo_track2] using {'real MiniLM' if use_real_embedder else 'MOCK'} embedder")

    bios = {p.user_id: p.bio for p in profiles}
    texts_a = pairs["user_id_a"].map(bios).tolist()
    texts_b = pairs["user_id_b"].map(bios).tolist()
    pairs["embedding_cosine_similarity"] = embedder.pairwise_similarity(texts_a, texts_b)

    # NOTE: no ground-truth match label exists for synthetic profiles, so we
    # can't fit a supervised fusion model *on Track 2 itself*. Instead we
    # report an unsupervised composite score (simple average of the four
    # normalized features) purely to demonstrate the features compose
    # sensibly on genuinely different (Conecta-shaped) data -- this is a
    # qualitative check, not a trained ranking model.
    pairs["composite_score"] = pairs[
        ["same_cohort", "arrival_date_proximity", "course_overlap_jaccard", "embedding_cosine_similarity"]
    ].mean(axis=1)

    names = {p.user_id: p.display_name for p in profiles}
    pairs["name_a"] = pairs["user_id_a"].map(names)
    pairs["name_b"] = pairs["user_id_b"].map(names)

    top10 = pairs.sort_values("composite_score", ascending=False).head(10)
    cols = ["name_a", "name_b", "same_cohort", "arrival_date_proximity",
            "course_overlap_jaccard", "embedding_cosine_similarity", "composite_score"]
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 10)
    print("\nTop 10 pairs by composite score:")
    print(top10[cols].round(3).to_string(index=False))

    if not use_real_embedder:
        print("\n*** NOTE: embedding_cosine_similarity is from MockEmbeddingModel here, so the "
              "embedding-driven part of the ranking is not meaningful -- same_cohort / "
              "arrival_date_proximity / course_overlap_jaccard ARE real and correctly computed, "
              "since they don't depend on the embedder. Re-run with --real on a machine with "
              "internet access for the actual embedding-based ranking. ***")

    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    args = parser.parse_args()
    run(use_real_embedder=args.real)
