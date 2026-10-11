"""
Pairwise compatibility features for the Conecta matching engine.

Four features feed Model B (structured) / the fusion model:
  - same_cohort                  binary
  - arrival_date_proximity       exponential decay, in [0, 1]
  - course_overlap_jaccard       Jaccard index, in [0, 1]
  - embedding_cosine_similarity  MiniLM sentence-embedding cosine sim, in [-1, 1]
                                  (computed separately in models.py -- needs
                                  internet access for the HF model download,
                                  so it can't run in this sandbox)

Note on int_corr (Speed Dating Track 1): not used as a model input, and not
used as an external validation signal either -- an earlier version of this
project's notes framed it as validation for embedding_cosine_similarity,
but that comparison doesn't hold up (int_corr and embedding similarity
measure different constructs with no principled reason to agree). int_corr
is excluded simply because it's a pre-computed statistic Conecta could
never have for real users, so training on it would misrepresent what's
deployable.
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd

from schema import UserProfile


# ---------------------------------------------------------------------------
# Track 2 (Conecta synthetic profiles): pairwise features from UserProfile
# ---------------------------------------------------------------------------

def same_cohort(a: UserProfile, b: UserProfile) -> int:
    """1 if both users are in the same enrollment/exchange cohort, else 0."""
    return int(a.cohort == b.cohort)


def arrival_date_proximity(a: UserProfile, b: UserProfile, decay_rate: float = 0.05) -> float:
    """
    Exponential decay of arrival-date distance, in (0, 1].
    1.0 when arrival dates coincide; decays toward 0 as |days apart| grows.
    decay_rate controls how fast: half-life = ln(2) / decay_rate days
    (default 0.05 -> ~14 day half-life, i.e. arriving 2 weeks apart still
    gives a meaningful similarity signal, a month apart is fairly weak).
    """
    days_apart = abs((a.arrival_date - b.arrival_date).days)
    return math.exp(-decay_rate * days_apart)


def course_overlap_jaccard(a: UserProfile, b: UserProfile) -> float:
    """Jaccard index of course sets. Returns 0.0 if both have no courses."""
    set_a, set_b = set(a.courses), set(b.courses)
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def pairwise_features(a: UserProfile, b: UserProfile) -> dict:
    """All Track-2 structured features for a pair (excludes the embedding feature)."""
    return {
        "same_cohort": same_cohort(a, b),
        "arrival_date_proximity": arrival_date_proximity(a, b),
        "course_overlap_jaccard": course_overlap_jaccard(a, b),
    }


def build_pairwise_dataframe(profiles: list[UserProfile]) -> pd.DataFrame:
    """
    All unordered pairs (i < j) from a list of UserProfiles, with structured
    features computed. embedding_cosine_similarity is added later once
    MiniLM embeddings are available (see models.py).
    """
    rows = []
    for i in range(len(profiles)):
        for j in range(i + 1, len(profiles)):
            a, b = profiles[i], profiles[j]
            row = {"user_id_a": a.user_id, "user_id_b": b.user_id}
            row.update(pairwise_features(a, b))
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Track 1 (Speed Dating): proxy versions of the same feature *concepts*,
# built from the columns available in that dataset, so Model B can be
# trained/evaluated on Track 1 with the same feature semantics as Track 2.
#
#   - same_cohort  -> same `wave`. In the raw data every recorded row is
#     already a within-wave pair by construction (speed dating happens
#     within a session), so this is constant (=1) and carries NO signal on
#     its own -- it's included for schema parity with Track 2, not as a
#     useful Track-1 predictor.
#   - course_overlap_jaccard -> NOT COMPUTABLE from this dataset as a true
#     pairwise feature. The raw columns give `field_cd`/`career_c` for the
#     row's own participant only -- there is no partner-side `field_cd_o`
#     column to compare against. A same-field proxy would need a self-join
#     of the dataset against itself (matching iid <-> pid), which is a
#     separate join step, not something derivable from a single row. Flagging
#     this explicitly rather than faking a placeholder column.
#   - age_proximity -> directly computable (age, age_o are both present in
#     every row already), same exponential-decay functional form as
#     arrival_date_proximity but on age-gap-in-years instead of days.
# ---------------------------------------------------------------------------

def speed_dating_pair_features(df: pd.DataFrame, age_decay_rate: float = 0.1) -> pd.DataFrame:
    """
    Structured features computable directly from the raw per-row Speed
    Dating data (no self-join required). Expects columns: age, age_o.
    """
    out = pd.DataFrame(index=df.index)
    out["age_proximity"] = np.exp(-age_decay_rate * (df["age"] - df["age_o"]).abs())
    return out


def add_same_field_via_self_join(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds true pairwise `same_field` and `same_career` columns.

    NOTE: if `df` has already been through `dedupe_directional_pairs`
    (data_loading.py), it already has `field_cd_o`/`career_c_o` columns and
    this just compares them directly -- no join needed. This self-join path
    is kept for dataframes that HAVEN'T been deduped yet (i.e. still have
    two directional rows per date), where only the row's own field_cd/
    career_c is available and the partner's has to be looked up.
    """
    df = df.copy()
    if "field_cd_o" not in df.columns:
        partner_field = df.groupby("iid")["field_cd"].first()
        df["field_cd_o"] = df["pid"].map(partner_field)
    if "career_c_o" not in df.columns:
        partner_career = df.groupby("iid")["career_c"].first()
        df["career_c_o"] = df["pid"].map(partner_career)
    df["same_field"] = _safe_eq(df["field_cd"], df["field_cd_o"])
    df["same_career"] = _safe_eq(df["career_c"], df["career_c_o"])
    return df


def _safe_eq(s1: pd.Series, s2: pd.Series) -> pd.Series:
    return ((s1 == s2) & s1.notna() & s2.notna()).astype(int)


if __name__ == "__main__":
    # Quick smoke test with two synthetic profiles
    a = UserProfile(
        user_id="u001",
        bio="Loves climbing and design.",
        cohort="2026-T2",
        arrival_date=date(2026, 7, 6),
        courses=["AIML339", "DESN301"],
    )
    b = UserProfile(
        user_id="u002",
        bio="Into film photography and hiking.",
        cohort="2026-T2",
        arrival_date=date(2026, 7, 10),
        courses=["AIML339", "COMP307"],
    )
    print(pairwise_features(a, b))

    import sys
    sys.path.insert(0, ".")
    from data_loading import load_speed_dating
    sd = load_speed_dating()

    feats = speed_dating_pair_features(sd)
    print(feats.describe())

    sd_joined = add_same_field_via_self_join(sd)
    print("\nsame_field via self-join, value counts:")
    print(sd_joined["same_field"].value_counts())
    print("rows where partner iid wasn't resolvable (field_cd_o is NaN):",
          sd_joined["field_cd_o"].isna().sum())
