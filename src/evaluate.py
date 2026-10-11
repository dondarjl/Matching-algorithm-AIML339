"""
Evaluation for the fusion-vs-baseline research question.

Two things are needed to claim "fusion beats embedding-only baseline":
  1. A point estimate of the performance gap (AUC, accuracy, F1, ...).
  2. A statistical test that the gap isn't just noise -- this is the part
     flagged as a gap in the peer-reviewed proposal too, so it's treated as
     a first-class piece of this module rather than an afterthought:
       - Paired bootstrap CI on the metric difference (distribution-free,
         works for any metric, e.g. AUC difference).
       - McNemar's test on the paired binary decisions (right tool for
         "did these two classifiers get the same test cases right/wrong",
         since both models are evaluated on the *same* pairs -- a paired
         design, not two independent samples).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score,
    average_precision_score,
)


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    """
    NOTE on threshold: with an imbalanced target (Speed Dating match rate is
    ~16%), a fixed 0.5 threshold on logistic regression output can yield
    precision/recall of exactly 0 even when the model discriminates well --
    predicted probabilities for the minority class rarely cross 0.5. AUC and
    average_precision (PR-AUC) are threshold-independent and should be
    treated as the primary metrics here; accuracy/precision/recall/f1 at a
    fixed threshold are included for completeness but read them alongside
    the class balance, not in isolation.
    """
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def bootstrap_metric_diff(
    y_true: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    metric_fn=roc_auc_score,
    n_boot: int = 2000,
    ci: float = 0.95,
    random_state: int = 0,
) -> dict:
    """
    Paired bootstrap CI for metric(model_b) - metric(model_a), e.g. fusion
    AUC minus baseline AUC. Resamples pair *indices* (not each model
    separately), which preserves the pairing -- important since both models
    are scored on identical test cases.

    Returns the point estimate, the CI bounds, and whether 0 is inside the
    CI (if it's not, and diff > 0, that's evidence fusion beats baseline at
    this confidence level).
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    y_true = np.asarray(y_true)
    proba_a = np.asarray(proba_a)
    proba_b = np.asarray(proba_b)

    point_diff = metric_fn(y_true, proba_b) - metric_fn(y_true, proba_a)

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        # skip degenerate resamples (all one class) where AUC is undefined
        if len(np.unique(y_true[idx])) < 2:
            diffs[i] = np.nan
            continue
        diffs[i] = metric_fn(y_true[idx], proba_b[idx]) - metric_fn(y_true[idx], proba_a[idx])

    diffs = diffs[~np.isnan(diffs)]
    alpha = 1 - ci
    lower, upper = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return {
        "point_estimate": point_diff,
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_level": ci,
        "significant": not (lower <= 0 <= upper),
        "n_boot_used": len(diffs),
    }


def cluster_bootstrap_metric_diff(
    y_true: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    groups: np.ndarray,
    metric_fn=roc_auc_score,
    n_boot: int = 2000,
    ci: float = 0.95,
    random_state: int = 0,
) -> dict:
    """
    Cluster (block) bootstrap CI for metric(model_b) - metric(model_a),
    resampling whole GROUPS (e.g. waves) with replacement, not individual
    rows.

    Why this matters here: `bootstrap_metric_diff` above resamples row
    indices, which implicitly treats each pair as an independent
    observation. But pairs sharing a wave aren't independent -- they share
    session-level effects (who else attended that wave, that day's overall
    dynamics, etc.). With only 21 waves underlying ~4000 pairs, a row-level
    bootstrap can be anti-conservative: it can report a narrower CI than is
    actually justified, because it's implicitly assuming ~4000 independent
    data points when the real independent sample size is closer to 21.

    This resamples GROUP LABELS with replacement (same number as the
    original number of distinct groups), then pools all rows belonging to
    the resampled groups (a group drawn twice contributes its rows twice)
    to compute the metric difference on that resample. This respects the
    actual clustering structure and will generally give an equal or wider
    (more honest) CI than the row-level version.
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    proba_a = np.asarray(proba_a)
    proba_b = np.asarray(proba_b)
    groups = np.asarray(groups)

    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    # precompute row indices per group once, to make resampling fast
    rows_by_group = {g: np.where(groups == g)[0] for g in unique_groups}

    point_diff = metric_fn(y_true, proba_b) - metric_fn(y_true, proba_a)

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        idx = np.concatenate([rows_by_group[g] for g in sampled_groups])
        if len(np.unique(y_true[idx])) < 2:
            diffs[i] = np.nan
            continue
        diffs[i] = metric_fn(y_true[idx], proba_b[idx]) - metric_fn(y_true[idx], proba_a[idx])

    diffs = diffs[~np.isnan(diffs)]
    alpha = 1 - ci
    lower, upper = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_value = min(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()), 1.0)

    return {
        "point_estimate": point_diff,
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_level": ci,
        "p_value": p_value,
        "significant": not (lower <= 0 <= upper),
        "n_boot_used": len(diffs),
        "n_groups": n_groups,
    }


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """
    McNemar's test on paired binary predictions from two models evaluated
    on the same cases. Tests whether the models disagree asymmetrically in
    a way that favors one of them (i.e. whether the *swap* cases -- A right/B
    wrong vs A wrong/B right -- are imbalanced).

    Uses the exact binomial form (appropriate here; the chi-square
    approximation is only needed when both discordant counts are large and
    scipy isn't available -- scipy is already a project dependency).
    """
    from scipy.stats import binomtest

    y_true = np.asarray(y_true)
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)

    # discordant pairs: one model right, the other wrong
    b_right_a_wrong = int(np.sum(correct_b & ~correct_a))
    a_right_b_wrong = int(np.sum(correct_a & ~correct_b))
    n_discordant = b_right_a_wrong + a_right_b_wrong

    if n_discordant == 0:
        return {
            "b_right_a_wrong": 0, "a_right_b_wrong": 0,
            "p_value": 1.0, "significant": False,
            "note": "no discordant pairs -- models agree on every case",
        }

    result = binomtest(b_right_a_wrong, n_discordant, p=0.5)
    return {
        "b_right_a_wrong": b_right_a_wrong,
        "a_right_b_wrong": a_right_b_wrong,
        "p_value": result.pvalue,
        "significant": bool(result.pvalue < 0.05),
    }

def per_participant_ranking_metrics(df_test, y_true, proba, min_dates=2):
    """
    Per-participant AUC: measures whether the model ranks a person's own
    set of dates well, not just whether it separates matches from
    non-matches globally across the whole pool.

    Each pair survives dedup as a single row under either iid or pid
    (whichever satisfies iid < pid). Since same_field, same_career,
    age_proximity, and embedding cosine similarity are symmetric in the
    pair, the model's predicted probability is valid from either
    participant's perspective -- so each row is expanded into both
    directions before grouping, or a participant would silently lose
    half their dates.
    """
    long_df = pd.concat([
        pd.DataFrame({"participant": df_test["iid"].to_numpy(), "y_true": y_true, "proba": proba}),
        pd.DataFrame({"participant": df_test["pid"].to_numpy(), "y_true": y_true, "proba": proba}),
    ], ignore_index=True)

    rows = []
    for participant, group in long_df.groupby("participant"):
        if len(group) < min_dates or group["y_true"].nunique() < 2:
            continue
        rows.append({
            "participant": participant,
            "n_dates": len(group),
            "auc": roc_auc_score(group["y_true"], group["proba"]),
        })

    result_df = pd.DataFrame(rows)
    return {
        "mean_auc": result_df["auc"].mean() if len(result_df) else float("nan"),
        "median_auc": result_df["auc"].median() if len(result_df) else float("nan"),
        "n_users_evaluated": len(result_df),
        "n_users_total": long_df["participant"].nunique(),
        "per_user": result_df,
    }

def compare_models(
    y_true: np.ndarray,
    proba_baseline: np.ndarray,
    proba_fusion: np.ndarray,
    threshold: float = 0.5,
    groups: np.ndarray | None = None,
) -> dict:
    """
    Full comparison bundle: point metrics for both, bootstrap AUC/PR-AUC
    diff, McNemar. If `groups` is provided (e.g. wave labels), also runs
    the cluster bootstrap alongside the row-level one, so both can be
    compared directly -- the cluster version is the more defensible one
    when observations aren't independent within a group.
    """
    metrics_baseline = compute_metrics(y_true, proba_baseline, threshold)
    metrics_fusion = compute_metrics(y_true, proba_fusion, threshold)
    boot_auc = bootstrap_metric_diff(y_true, proba_baseline, proba_fusion, metric_fn=roc_auc_score)
    boot_pr_auc = bootstrap_metric_diff(y_true, proba_baseline, proba_fusion, metric_fn=average_precision_score)
    mcnemar = mcnemar_test(
        y_true,
        (proba_baseline >= threshold).astype(int),
        (proba_fusion >= threshold).astype(int),
    )
    result = {
        "baseline_metrics": metrics_baseline,
        "fusion_metrics": metrics_fusion,
        "auc_diff_bootstrap": boot_auc,
        "pr_auc_diff_bootstrap": boot_pr_auc,
        "mcnemar": mcnemar,
    }

    if groups is not None:
        result["auc_diff_cluster_bootstrap"] = cluster_bootstrap_metric_diff(
            y_true, proba_baseline, proba_fusion, groups, metric_fn=roc_auc_score
        )
        result["pr_auc_diff_cluster_bootstrap"] = cluster_bootstrap_metric_diff(
            y_true, proba_baseline, proba_fusion, groups, metric_fn=average_precision_score
        )

    return result


if __name__ == "__main__":
    # Smoke test with synthetic data: fusion is deliberately made a bit
    # better than baseline so we can check the tests correctly detect it.
    rng = np.random.default_rng(0)
    n = 1000
    y_true = rng.integers(0, 2, n)
    proba_baseline = np.clip(y_true * 0.5 + rng.normal(0.3, 0.25, n), 0, 1)
    proba_fusion = np.clip(y_true * 0.7 + rng.normal(0.25, 0.2, n), 0, 1)
    groups = rng.integers(0, 20, n)  # 20 synthetic "waves"

    result = compare_models(y_true, proba_baseline, proba_fusion, groups=groups)
    import json
    print(json.dumps(result, indent=2, default=float))

def participant_bootstrap_diff(per_user_a, per_user_b, n_boot=2000, ci=0.95, random_state=0):
    """
    Bootstrap CI for mean_auc(b) - mean_auc(a), resampling PARTICIPANTS with
    replacement (not dates), matched by participant id between the two
    per_user dataframes (only participants evaluated in both are used).
    """
    rng = np.random.default_rng(random_state)
    merged = per_user_a.merge(per_user_b, on="participant", suffixes=("_a", "_b"))
    n = len(merged)
    auc_a = merged["auc_a"].to_numpy()
    auc_b = merged["auc_b"].to_numpy()

    point_diff = auc_b.mean() - auc_a.mean()
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = auc_b[idx].mean() - auc_a[idx].mean()

    alpha = 1 - ci
    lower, upper = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_value = min(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()), 1.0)
    return {
        "point_estimate": point_diff, "ci_lower": lower, "ci_upper": upper,
        "p_value": p_value, "significant": not (lower <= 0 <= upper), "n_participants": n,
    }

def holm_adjust(p_values):
    """
    Holm-Bonferroni step-down adjusted p-values (same order as input).
    adj_(i) = max_{j<=i} min(1, (m - j + 1) * p_(j)) over ascending-sorted p.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj_sorted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, min(1.0, (m - rank) * p[idx]))
        adj_sorted[rank] = running_max
    adj = np.empty(m)
    adj[order] = adj_sorted
    return adj


def participant_cluster_bootstrap_diff(per_user_a, per_user_b, participant_wave,
                                       n_boot=2000, ci=0.95, random_state=0):
    """
    Like participant_bootstrap_diff, but resamples WAVES (each carrying its
    participants' per-user AUC differences) instead of participants. Participants
    share dates with each other and live inside 21 waves, so resampling them as
    independent is anti-conservative -- the same argument used for the pooled
    cluster bootstrap. `participant_wave`: dict/Series participant id -> wave.
    """
    rng = np.random.default_rng(random_state)
    merged = per_user_a.merge(per_user_b, on="participant", suffixes=("_a", "_b"))
    merged["wave"] = merged["participant"].map(participant_wave)
    merged = merged.dropna(subset=["wave"])
    diff = (merged["auc_b"] - merged["auc_a"]).to_numpy()
    waves = merged["wave"].to_numpy()
    uw = np.unique(waves)
    idx_by_wave = {w: np.where(waves == w)[0] for w in uw}

    point = diff.mean()
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sampled = rng.choice(uw, size=len(uw), replace=True)
        idx = np.concatenate([idx_by_wave[w] for w in sampled])
        diffs[i] = diff[idx].mean()

    alpha = 1 - ci
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = min(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()), 1.0)
    return {"point_estimate": point, "ci_lower": lo, "ci_upper": hi, "p_value": p,
            "significant": not (lo <= 0 <= hi), "n_participants": len(merged), "n_waves": len(uw)}