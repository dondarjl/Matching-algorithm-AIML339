"""
Null-model check for the 'rich' text proxy: does its AUC edge come from
real interest overlap, or from repeated template structure across
profiles? See permute_interest_categories() in interest_bio.py.

Usage:
    python validate_rich_proxy.py --real
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

from data_loading import load_speed_dating, dedupe_directional_pairs
from features import add_same_field_via_self_join
from interest_bio import add_rich_bio_text, permute_interest_categories, INTEREST_COLUMNS
from train_track1 import add_partner_text
from models import EmbeddingModel, MockEmbeddingModel, EmbeddingOnlyBaseline
from evaluate import cluster_bootstrap_metric_diff

def run(use_real_embedder: bool = False, n_folds: int = 7, random_state: int = 0):
    sd = load_speed_dating()
    sd = dedupe_directional_pairs(sd)
    sd = add_same_field_via_self_join(sd)
    sd = add_partner_text(sd)

    sd_real = add_rich_bio_text(sd.copy())
    sd_perm = permute_interest_categories(sd.copy(), columns=INTEREST_COLUMNS, random_state=random_state)
    sd_perm = add_rich_bio_text(sd_perm)

    required = ["match", "own_text_rich", "partner_text_rich", "wave"]
    sd_real = sd_real.dropna(subset=required)
    sd_perm = sd_perm.dropna(subset=required)
    # keep only rows present in both after dropna, aligned by index
    common_idx = sd_real.index.intersection(sd_perm.index)
    sd_real = sd_real.loc[common_idx]
    sd_perm = sd_perm.loc[common_idx]

    embedder = EmbeddingModel() if use_real_embedder else MockEmbeddingModel()
    print(f"[validate_rich_proxy] using {'real MiniLM' if use_real_embedder else 'MOCK'} embedder")

    y = sd_real["match"].astype(int).to_numpy()
    waves = sd_real["wave"].to_numpy()

    def get_oof_proba(sd_variant, text_col_own, text_col_partner, embedder, waves, y, n_folds):
        sim = embedder.pairwise_similarity(
            sd_variant[text_col_own].tolist(), sd_variant[text_col_partner].tolist()
        )
        X = pd.DataFrame({"embedding_cosine_similarity": np.asarray(sim)})
        gkf = GroupKFold(n_splits=n_folds)
        oof = np.full(len(sd_variant), np.nan)
        for train_idx, test_idx in gkf.split(X, y, groups=waves):
            model = EmbeddingOnlyBaseline().fit(X.iloc[train_idx], y[train_idx])
            oof[test_idx] = model.predict_proba(X.iloc[test_idx])
        return oof

    proba_real = get_oof_proba(sd_real, "own_text_rich", "partner_text_rich", embedder, waves, y, n_folds)
    proba_perm = get_oof_proba(sd_perm, "own_text_rich", "partner_text_rich", embedder, waves, y, n_folds)

    auc_real = roc_auc_score(y, proba_real)
    auc_perm = roc_auc_score(y, proba_perm)
    print(f"AUC real={auc_real:.4f}  AUC permuted={auc_perm:.4f}")

    result = cluster_bootstrap_metric_diff(y, proba_perm, proba_real, waves, metric_fn=roc_auc_score)
    print(f"Diff (real - permuted): point={result['point_estimate']:.4f}  "
          f"CI=[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]  "
          f"p={result['p_value']:.4f}  significant={result['significant']}")

    if not use_real_embedder:
        print("\n*** NOTE: MOCK embedder -- re-run with --real for meaningful results. ***")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--folds", type=int, default=7)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()
    run(use_real_embedder=args.real, n_folds=args.folds, random_state=args.random_state)