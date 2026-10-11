"""
Track 1 end-to-end pipeline: Speed Dating Experiment data -> features ->
EmbeddingOnlyBaseline vs FusionModel -> statistical comparison.

Text proxy for embeddings: Speed Dating has no free-text bios like Conecta
profiles do, so `field` + `career` free-text columns are concatenated per
person and used as a short text proxy for the embedding model (own text vs.
partner text -> cosine similarity). This is a deliberate substitution, not
an attempt to hide the difference from Track 2's richer bios -- flagged
here and in the README so it doesn't get lost by the time of the write-up.

Split strategy: GroupShuffleSplit grouped by `iid` (participant), NOT a
plain random row split. Speed Dating has many rows per participant (one per
partner they met), so a naive row-level split would leak the same person's
rows across train and test. Splitting by participant avoids that.

Usage:
    python train_track1.py            # mock embedder, runs fully offline
    python train_track1.py --real     # real MiniLM, needs internet access
                                       # to huggingface.co (run on your machine)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from data_loading import load_speed_dating, dedupe_directional_pairs
from features import add_same_field_via_self_join
from interest_bio import add_rich_bio_text
from models import EmbeddingModel, MockEmbeddingModel, EmbeddingOnlyBaseline, FusionModel
from evaluate import compare_models


def add_partner_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds `own_text` and `partner_text` columns (field + career concatenated).

    If `df` has already been through `dedupe_directional_pairs`, it already
    has `field_o`/`career_o` and this just concatenates them directly. Kept
    working on non-deduped frames too (self-join fallback) for flexibility.
    """
    df = df.copy()
    df["field"] = df["field"].fillna("")
    df["career"] = df["career"].fillna("")
    df["own_text"] = (df["field"] + " " + df["career"]).str.strip()

    if "field_o" in df.columns and "career_o" in df.columns:
        df["field_o"] = df["field_o"].fillna("")
        df["career_o"] = df["career_o"].fillna("")
        df["partner_text"] = (df["field_o"] + " " + df["career_o"]).str.strip()
    else:
        partner_text = df.groupby("iid")["own_text"].first()
        df["partner_text"] = df["pid"].map(partner_text)
    return df


def build_dataset(age_decay_rate: float = 0.1, text_proxy: str = "plain") -> pd.DataFrame:
    """
    Load Speed Dating data, dedupe to one row per pair, and assemble
    features + target.

    text_proxy controls what goes into `own_text`/`partner_text` (the input
    to the embedding model):
      - "plain": field + career only (~2-4 words) -- the original proxy.
      - "rich":  field + career + a full sentence describing interests,
                 bucketed from the 17 interest-rating columns (see
                 interest_bio.py). Much closer in richness to an actual
                 Conecta bio; kept as an explicit alternative rather than a
                 silent replacement so both can be compared.
    """
    if text_proxy not in ("plain", "rich"):
        raise ValueError(f"text_proxy must be 'plain' or 'rich', got {text_proxy!r}")

    sd = load_speed_dating()
    sd = dedupe_directional_pairs(sd)
    sd = add_same_field_via_self_join(sd)
    sd = add_partner_text(sd)
    sd = add_rich_bio_text(sd)

    if text_proxy == "rich":
        sd["own_text"] = sd["own_text_rich"]
        sd["partner_text"] = sd["partner_text_rich"]

    sd["age_proximity"] = np.exp(-age_decay_rate * (sd["age"] - sd["age_o"]).abs())

    required = ["match", "same_field", "same_career", "age_proximity", "own_text", "partner_text", "wave"]
    before = len(sd)
    sd = sd.dropna(subset=[c for c in required if c != "own_text" and c != "partner_text"])
    sd = sd[(sd["own_text"] != "") & (sd["partner_text"].notna()) & (sd["partner_text"] != "")]
    after = len(sd)
    print(f"[train_track1] dropped {before - after} rows with missing features "
          f"({after} rows remain, text_proxy={text_proxy!r})")
    return sd


def run(use_real_embedder: bool = False, test_size: float = 0.25, random_state: int = 0,
        text_proxy: str = "plain"):
    df = build_dataset(text_proxy=text_proxy)

    embedder = EmbeddingModel() if use_real_embedder else MockEmbeddingModel()
    print(f"[train_track1] using {'real MiniLM' if use_real_embedder else 'MOCK'} embedder, "
          f"text_proxy={text_proxy!r}")
    df["embedding_cosine_similarity"] = embedder.pairwise_similarity(
        df["own_text"].tolist(), df["partner_text"].tolist()
    )

    structured_features = ["same_field", "same_career", "age_proximity"]
    all_features = ["embedding_cosine_similarity"] + structured_features
    X = df[all_features]
    y = df["match"].astype(int)

    # Group by `wave`, not `iid`: within a wave, participants date each other
    # (a fully-connected component), so a participant's dates never span two
    # waves. Grouping by `iid` alone would NOT prevent leakage post-dedupe --
    # a person can appear as `iid` in some kept rows and as `pid` (partner)
    # in others (since dedupe keeps whichever row has iid < pid), so their
    # dates could be scattered across the group-split unprotected. `wave` has
    # no such ambiguity.
    groups = df["wave"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = df.iloc[train_idx], df.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"[train_track1] train: {len(X_train)} rows / {groups.iloc[train_idx].nunique()} "
          f"waves, test: {len(X_test)} rows / {groups.iloc[test_idx].nunique()} waves")
    print(f"[train_track1] train match rate: {y_train.mean():.3f}, test match rate: {y_test.mean():.3f}")

    baseline = EmbeddingOnlyBaseline().fit(X_train, y_train)
    fusion = FusionModel(structured_features).fit(X_train, y_train)

    proba_baseline = baseline.predict_proba(X_test)
    proba_fusion = fusion.predict_proba(X_test)

    print("\nFusion model coefficients (weight given to each feature):")
    for name, coef in fusion.coefficients.items():
        print(f"  {name}: {coef:.4f}")

    # Fixed 0.5 threshold is a poor choice with ~16% base rate (see
    # evaluate.compute_metrics docstring) -- use the training base rate as a
    # class-balanced cutoff instead, so precision/recall/McNemar are
    # actually informative rather than trivially all-zero/all-agree.
    threshold = float(y_train.mean())
    print(f"\n[train_track1] using class-balanced threshold={threshold:.3f} "
          f"(training base rate) instead of a fixed 0.5")

    print("\n--- Comparison: EmbeddingOnlyBaseline vs FusionModel ---")
    results = compare_models(y_test.to_numpy(), proba_baseline, proba_fusion, threshold=threshold)
    import json
    print(json.dumps(results, indent=2, default=float))

    if not use_real_embedder:
        print("\n*** NOTE: embedding_cosine_similarity above is from MockEmbeddingModel "
              "(hash-seeded random vectors), so these numbers are NOT meaningful research "
              "results -- this run only validates that the pipeline (data -> features -> "
              "models -> stats) executes end to end without errors. Re-run with --real on a "
              "machine with internet access to huggingface.co for actual results. ***")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true",
                         help="Use the real MiniLM embedder (requires internet access to huggingface.co)")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--text-proxy", choices=["plain", "rich"], default="plain",
                         help="'plain'=field+career only (original). 'rich'=field+career+interest-bio sentence.")
    args = parser.parse_args()

    run(use_real_embedder=args.real, test_size=args.test_size, random_state=args.random_state,
        text_proxy=args.text_proxy)
