"""
Robustness check for train_track1.py's single-split result: instead of one
75/25 split (which, with only 21 waves total, risks an unlucky test set --
exactly what we saw, with train match rate 0.188 vs test 0.143), run
GroupKFold cross-validation grouped by `wave`.

Each wave lands in exactly one test fold. For every row, we get an
"out-of-fold" prediction from a model that never saw that row (or that
row's wave) during training. Stitching these together gives predictions
for the ENTIRE dataset, evaluated fairly, without the "did we get a lucky/
unlucky split" question -- every wave contributes to the test evaluation
exactly once.

We also report per-fold AUC to see whether the direction of the
fusion-vs-baseline gap is consistent across folds, or flips around
(which would indicate real instability, not just noise in one split).

Usage:
    python cross_validate_track1.py            # mock embedder, offline
    python cross_validate_track1.py --real      # real MiniLM, needs internet
    python cross_validate_track1.py --real --folds 5
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from train_track1 import build_dataset
from models import EmbeddingModel, MockEmbeddingModel, EmbeddingOnlyBaseline, FusionModel
from evaluate import compare_models


def run(use_real_embedder: bool = False, n_folds: int = 7, random_state: int = 0,
        text_proxy: str = "plain"):
    df = build_dataset(text_proxy=text_proxy)

    embedder = EmbeddingModel() if use_real_embedder else MockEmbeddingModel()
    print(f"[cross_validate_track1] using {'real MiniLM' if use_real_embedder else 'MOCK'} embedder, "
          f"text_proxy={text_proxy!r}")
    df["embedding_cosine_similarity"] = embedder.pairwise_similarity(
        df["own_text"].tolist(), df["partner_text"].tolist()
    )

    structured_features = ["same_field", "same_career", "age_proximity"]
    all_features = ["embedding_cosine_similarity"] + structured_features
    X = df[all_features].reset_index(drop=True)
    y = df["match"].astype(int).reset_index(drop=True)
    waves = df["wave"].reset_index(drop=True)

    n_waves = waves.nunique()
    if n_folds > n_waves:
        raise ValueError(f"n_folds ({n_folds}) can't exceed number of waves ({n_waves})")

    gkf = GroupKFold(n_splits=n_folds)

    oof_baseline = np.full(len(df), np.nan)
    oof_fusion = np.full(len(df), np.nan)
    fold_rows = []

    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=waves), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        baseline = EmbeddingOnlyBaseline().fit(X_train, y_train)
        fusion = FusionModel(structured_features).fit(X_train, y_train)

        proba_baseline = baseline.predict_proba(X_test)
        proba_fusion = fusion.predict_proba(X_test)

        oof_baseline[test_idx] = proba_baseline
        oof_fusion[test_idx] = proba_fusion

        fold_auc_baseline = roc_auc_score(y_test, proba_baseline)
        fold_auc_fusion = roc_auc_score(y_test, proba_fusion)
        fold_rows.append({
            "fold": fold_i,
            "n_test_rows": len(test_idx),
            "n_test_waves": waves.iloc[test_idx].nunique(),
            "test_match_rate": round(float(y_test.mean()), 3),
            "auc_baseline": round(fold_auc_baseline, 4),
            "auc_fusion": round(fold_auc_fusion, 4),
            "auc_diff": round(fold_auc_fusion - fold_auc_baseline, 4),
        })

    fold_df = pd.DataFrame(fold_rows)
    print(f"\n[cross_validate_track1] {n_folds}-fold GroupKFold by wave ({n_waves} waves total)")
    print(fold_df.to_string(index=False))

    n_favoring_fusion = (fold_df["auc_diff"] > 0).sum()
    print(f"\nFusion has higher AUC than baseline in {n_favoring_fusion}/{n_folds} folds.")
    if n_favoring_fusion in (0, n_folds):
        print("-> Direction is CONSISTENT across all folds.")
    else:
        print("-> Direction FLIPS across folds -- the single-split result was not stable; "
              "treat any single-split conclusion with real caution.")

    # Sanity check: every row should have exactly one out-of-fold prediction
    assert not np.isnan(oof_baseline).any(), "some rows never appeared in a test fold"
    assert not np.isnan(oof_fusion).any(), "some rows never appeared in a test fold"

    overall_threshold = float(y.mean())
    print(f"\n--- Aggregate out-of-fold comparison (all {len(df)} rows, each predicted by a "
          f"model that never saw its wave during training) ---")
    print(f"[cross_validate_track1] using overall base rate threshold={overall_threshold:.3f}")
    results = compare_models(y.to_numpy(), oof_baseline, oof_fusion, threshold=overall_threshold,
                              groups=waves.to_numpy())
    import json
    print(json.dumps(results, indent=2, default=float))

    # Row-level vs. cluster bootstrap CI comparison -- the row-level one
    # treats each pair as independent (it isn't: pairs sharing a wave share
    # session-level effects), which can make its CI anti-conservative
    # (narrower than justified). The cluster version resamples whole waves,
    # respecting the real independence structure (~21 waves, not ~4000
    # independent pairs), and should be treated as the more defensible one
    # for the report -- print both side by side to make the difference visible.
    row_ci = results["auc_diff_bootstrap"]
    cluster_ci = results["auc_diff_cluster_bootstrap"]
    print(f"\n[cross_validate_track1] AUC diff CI, row-level bootstrap:     "
          f"[{row_ci['ci_lower']:.4f}, {row_ci['ci_upper']:.4f}] (significant={row_ci['significant']})")
    print(f"[cross_validate_track1] AUC diff CI, cluster (wave) bootstrap: "
          f"[{cluster_ci['ci_lower']:.4f}, {cluster_ci['ci_upper']:.4f}] (significant={cluster_ci['significant']}, "
          f"n_groups={cluster_ci['n_groups']})")
    if row_ci["significant"] and not cluster_ci["significant"]:
        print("[cross_validate_track1] NOTE: the row-level bootstrap was anti-conservative here -- "
              "the cluster-robust CI crosses zero. The wave-clustered result is the one to report.")

    if not use_real_embedder:
        print("\n*** NOTE: embedding_cosine_similarity is from MockEmbeddingModel here, so these "
              "numbers are NOT meaningful research results -- re-run with --real. ***")

    return fold_df, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--folds", type=int, default=7)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--text-proxy", choices=["plain", "rich"], default="plain",
                         help="'plain'=field+career only (original). 'rich'=field+career+interest-bio sentence.")
    args = parser.parse_args()

    run(use_real_embedder=args.real, n_folds=args.folds, random_state=args.random_state,
        text_proxy=args.text_proxy)
