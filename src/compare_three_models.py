"""
Closes the remaining gap in the model comparison: `cross_validate_track1.py`
only compares EmbeddingOnlyBaseline vs. FusionModel. This script adds
StructuredModel (structured features only, no embedding) into the same
wave-grouped cross-validation, so all three models are compared under
identical conditions.

This answers a question the baseline-vs-fusion comparison alone can't:
do the structured features carry any predictive signal on their own, or is
essentially all of fusion's (fragile) edge coming from the embedding
feature, with same_field/same_career/age_proximity contributing little or
even being redundant with it?

Usage:
    python compare_three_models.py            # mock embedder, offline
    python compare_three_models.py --real      # real MiniLM, needs internet
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from train_track1 import build_dataset
from models import EmbeddingModel, MockEmbeddingModel, EmbeddingOnlyBaseline, StructuredModel, FusionModel
from evaluate import (compute_metrics, compare_models, per_participant_ranking_metrics,
                      participant_bootstrap_diff, participant_cluster_bootstrap_diff, holm_adjust)


def run(use_real_embedder: bool = False, n_folds: int = 7, random_state: int = 0,
        text_proxy: str = "plain"):
    df = build_dataset(text_proxy=text_proxy)

    embedder = EmbeddingModel() if use_real_embedder else MockEmbeddingModel()
    print(f"[compare_three_models] using {'real MiniLM' if use_real_embedder else 'MOCK'} embedder, "
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
    gkf = GroupKFold(n_splits=n_folds)

    oof = {
        "baseline": np.full(len(df), np.nan),
        "structured": np.full(len(df), np.nan),
        "fusion": np.full(len(df), np.nan),
    }
    fold_rows = []

    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=waves), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        baseline = EmbeddingOnlyBaseline().fit(X_train, y_train)
        structured = StructuredModel(structured_features).fit(X_train, y_train)
        fusion = FusionModel(structured_features).fit(X_train, y_train)

        p_baseline = baseline.predict_proba(X_test)
        p_structured = structured.predict_proba(X_test)
        p_fusion = fusion.predict_proba(X_test)

        oof["baseline"][test_idx] = p_baseline
        oof["structured"][test_idx] = p_structured
        oof["fusion"][test_idx] = p_fusion

        fold_rows.append({
            "fold": fold_i,
            "n_test_rows": len(test_idx),
            "auc_baseline": round(roc_auc_score(y_test, p_baseline), 4),
            "auc_structured": round(roc_auc_score(y_test, p_structured), 4),
            "auc_fusion": round(roc_auc_score(y_test, p_fusion), 4),
        })

    fold_df = pd.DataFrame(fold_rows)
    print(f"\n[compare_three_models] {n_folds}-fold GroupKFold by wave ({n_waves} waves total)")
    print(fold_df.to_string(index=False))

    for name, arr in oof.items():
        assert not np.isnan(arr).any(), f"some rows never got an out-of-fold prediction for {name}"

    overall_threshold = float(y.mean())
    y_np = y.to_numpy()
    waves_np = waves.to_numpy()

    print(f"\n--- Aggregate out-of-fold metrics (all {len(df)} rows), threshold={overall_threshold:.3f} ---")
    metrics_table = []
    for name, arr in oof.items():
        m = compute_metrics(y_np, arr, threshold=overall_threshold)
        m["model"] = name
        metrics_table.append(m)
    metrics_df = pd.DataFrame(metrics_table).set_index("model")[["auc", "pr_auc", "accuracy", "precision", "recall", "f1"]]
    print(metrics_df.round(6).to_string())

    print("\n--- Per-participant ranking metrics ---")
    for name, arr in oof.items():
        ppm = per_participant_ranking_metrics(df, y_np, arr)
        print(f"  {name}: mean AUC={ppm['mean_auc']:.4f}  median AUC={ppm['median_auc']:.4f}  "
            f"users evaluated={ppm['n_users_evaluated']}/{ppm['n_users_total']}")
        
    ppm_baseline = per_participant_ranking_metrics(df, y_np, oof["baseline"])
    ppm_structured = per_participant_ranking_metrics(df, y_np, oof["structured"])
    ppm_fusion = per_participant_ranking_metrics(df, y_np, oof["fusion"])

    print(f"\n  n_dates distribution (fusion): min={ppm_fusion['per_user']['n_dates'].min()}, "
        f"median={ppm_fusion['per_user']['n_dates'].median()}, max={ppm_fusion['per_user']['n_dates'].max()}")

    for name_a, ppm_a, name_b, ppm_b in [("baseline", ppm_baseline, "fusion", ppm_fusion),
                                        ("structured", ppm_structured, "fusion", ppm_fusion)]:
        r = participant_bootstrap_diff(ppm_a["per_user"], ppm_b["per_user"])
        print(f"  Per-user AUC diff ({name_b}-{name_a}): point={r['point_estimate']:.4f} "
            f"CI=[{r['ci_lower']:.4f},{r['ci_upper']:.4f}] p={r['p_value']:.4f} "
            f"significant={r['significant']} (n={r['n_participants']})")   

    # Wave-cluster version: participants share dates and sit inside ~21 waves, so
    # resampling participants as independent is anti-conservative (see evaluate.py).
    participant_wave = pd.concat([
        pd.Series(df["wave"].to_numpy(), index=df["iid"].to_numpy()),
        pd.Series(df["wave"].to_numpy(), index=df["pid"].to_numpy()),
    ])
    participant_wave = participant_wave[~participant_wave.index.duplicated()].to_dict()
    print("\n  Per-user AUC diffs, WAVE-CLUSTER bootstrap (use these in the report):")
    pp_p = []
    for name_a, ppm_a, name_b, ppm_b in [("baseline", ppm_baseline, "fusion", ppm_fusion),
                                        ("structured", ppm_structured, "fusion", ppm_fusion)]:
        rc = participant_cluster_bootstrap_diff(ppm_a["per_user"], ppm_b["per_user"], participant_wave)
        pp_p.append(rc["p_value"])
        print(f"  ({name_b}-{name_a}): point={rc['point_estimate']:.4f} "
              f"CI=[{rc['ci_lower']:.4f},{rc['ci_upper']:.4f}] p={rc['p_value']:.4f} "
              f"(n={rc['n_participants']}, waves={rc['n_waves']})")
    print(f"  Holm-adjusted over these two: {np.round(holm_adjust(pp_p), 4).tolist()}")
        
    print("\n--- Pairwise comparisons (cluster bootstrap over waves + McNemar) ---")

    print("\n[Structured vs. Baseline] does structured-only beat embedding-only?")
    r1 = compare_models(y_np, oof["baseline"], oof["structured"], threshold=overall_threshold, groups=waves_np)
    _print_pairwise(r1, "baseline", "structured")

    print("\n[Fusion vs. Structured] does adding the embedding on top of structured features help?")
    r2 = compare_models(y_np, oof["structured"], oof["fusion"], threshold=overall_threshold, groups=waves_np)
    _print_pairwise(r2, "structured", "fusion")

    print("\n[Fusion vs. Baseline] the main research question, for reference (same as cross_validate_track1.py)")
    r3 = compare_models(y_np, oof["baseline"], oof["fusion"], threshold=overall_threshold, groups=waves_np)
    _print_pairwise(r3, "baseline", "fusion")
    pr = holm_adjust([r3["auc_diff_cluster_bootstrap"]["p_value"], r3["pr_auc_diff_cluster_bootstrap"]["p_value"]])
    print(f"\n[Primary family: fusion vs baseline] raw p (AUC, PR-AUC) = "
          f"({r3['auc_diff_cluster_bootstrap']['p_value']:.4f}, {r3['pr_auc_diff_cluster_bootstrap']['p_value']:.4f}); "
          f"Holm-adjusted = {np.round(pr, 4).tolist()}")

    if not use_real_embedder:
        print("\n*** NOTE: embedding_cosine_similarity is from MockEmbeddingModel here, so these "
              "numbers are NOT meaningful research results -- re-run with --real. ***")

    return fold_df, metrics_df, (r1, r2, r3)


def _print_pairwise(result: dict, name_a: str, name_b: str):
    auc_c = result["auc_diff_cluster_bootstrap"]
    pr_c = result["pr_auc_diff_cluster_bootstrap"]
    mc = result["mcnemar"]
    print(f"  AUC diff ({name_b} - {name_a}), cluster CI: "
          f"[{auc_c['ci_lower']:.4f}, {auc_c['ci_upper']:.4f}]  "
          f"p={auc_c['p_value']:.4f}  significant={auc_c['significant']}")
    print(f"  PR-AUC diff ({name_b} - {name_a}), cluster CI: "
          f"[{pr_c['ci_lower']:.4f}, {pr_c['ci_upper']:.4f}]  "
          f"p={pr_c['p_value']:.4f}  significant={pr_c['significant']}")
    print(f"  McNemar: {name_b}-right/{name_a}-wrong={mc['b_right_a_wrong']}, "
          f"{name_a}-right/{name_b}-wrong={mc['a_right_b_wrong']}, "
          f"p={mc['p_value']:.2e}  significant={mc['significant']}")


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
