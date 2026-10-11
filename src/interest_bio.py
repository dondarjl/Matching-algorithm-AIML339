"""
Builds a richer text proxy for the embedding model, combining field/career
with the 17 interest-rating columns (bucketed into descriptive phrases).

This exists alongside the original field+career-only proxy
(train_track1.add_partner_text), not as a silent replacement for it: the
two give genuinely different amounts of semantic content to MiniLM (a
2-word phrase like "Law lawyer" vs. a full sentence describing someone's
interests), and it's worth being able to compare both rather than assuming
richer is better without checking.

Bucketing: interest ratings are on a 0-10 scale (a few out-of-scale values
in the raw data are clipped to 10 in data_loading.load_speed_dating). Cut
points were chosen by inspecting the actual score distribution (skewed
high -- self-reported interest ratings tend to run positive), not picked
arbitrarily:
    8-10  -> "loves"                (very favorable)
    6-7   -> "likes"                (favorable)
    3-5   -> "isn't very interested in"  (unfavorable)
    0-2   -> "dislikes"             (very unfavorable)
English phrasing throughout, to match MiniLM's training distribution and
stay consistent with the field/career text already used.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from data_loading import INTEREST_COLUMNS

# Human-readable display phrase for each column -- a couple (tv, tvsports)
# need disambiguating since the raw column name alone reads oddly as a noun.
DISPLAY_NAME = {
    "sports": "sports",
    "tvsports": "watching sports on TV",
    "exercise": "exercise",
    "dining": "dining out",
    "museums": "museums",
    "art": "art",
    "hiking": "hiking",
    "gaming": "gaming",
    "clubbing": "clubbing",
    "reading": "reading",
    "tv": "watching TV",
    "theater": "theater",
    "movies": "movies",
    "concerts": "concerts",
    "music": "music",
    "shopping": "shopping",
    "yoga": "yoga",
}

BUCKET_PHRASES = [
    (8, 10, "loves"),
    (6, 7, "likes"),
    (3, 5, "isn't very interested in"),
    (0, 2, "dislikes"),
]


def _bucket_label(value: float) -> str | None:
    if pd.isna(value):
        return None
    for lo, hi, label in BUCKET_PHRASES:
        if lo <= value <= hi:
            return label
    return None  # shouldn't happen post-clipping, but fail safe rather than mislabel


def build_interest_bio(row: pd.Series, columns: list[str] = INTEREST_COLUMNS,
                        suffix: str = "") -> str:
    """
    Build one English sentence describing a person's interests, grouped by
    favorability, from a row's interest-rating columns (optionally with a
    suffix, e.g. "_o" to read the partner's own values from a deduped row).
    Returns "" if no interest columns had a usable value.
    """
    groups: dict[str, list[str]] = {label: [] for _, _, label in BUCKET_PHRASES}
    for col in columns:
        colname = f"{col}{suffix}"
        if colname not in row.index:
            continue
        label = _bucket_label(row[colname])
        if label is not None:
            groups[label].append(DISPLAY_NAME[col])

    sentences = []
    for _, _, label in BUCKET_PHRASES:
        items = groups[label]
        if not items:
            continue
        if len(items) == 1:
            joined = items[0]
        elif len(items) == 2:
            joined = f"{items[0]} and {items[1]}"
        else:
            joined = ", ".join(items[:-1]) + f", and {items[-1]}"
        sentences.append(f"{label.capitalize()} {joined}.")

    return " ".join(sentences)

def permute_interest_categories(df: pd.DataFrame, columns: list[str], random_state: int = 0) -> pd.DataFrame:
    """
    Null-model control for the 'rich' text proxy: for each unique
    participant, randomly permutes which interest-intensity rating is
    attached to which category label, while keeping that participant's
    multiset of ratings (and therefore sentence length, word frequency,
    and bucket sizes) identical. This breaks the true category<->intensity
    association -- i.e. whether two people *actually* share interests --
    while preserving every structural/lexical property of the generated
    bio text.

    If AUC survives this permutation, the embedding is reading template
    structure, not real interest overlap.
    """
    rng = np.random.default_rng(random_state)
    df = df.copy()

    # Collect each unique participant's own raw ratings exactly once.
    # A participant's values show up unsuffixed when they are `iid` on a
    # row, and suffixed with `_o` when they are `pid` (the partner) on a
    # row involving someone else.
    own_rows = df[["iid"] + columns].rename(columns={"iid": "pid_key"})
    own_rows.columns = ["pid_key"] + columns
    partner_cols_o = [f"{c}_o" for c in columns]
    partner_rows = df[["pid"] + partner_cols_o].rename(columns={"pid": "pid_key"})
    partner_rows.columns = ["pid_key"] + columns

    all_participants = pd.concat([own_rows, partner_rows], ignore_index=True)
    all_participants = all_participants.drop_duplicates(subset="pid_key").set_index("pid_key")

    # One fixed random permutation of category labels per participant.
    permutation_map = {}
    for pid_key, row in all_participants.iterrows():
        values = row[columns].to_numpy()
        shuffled_cols = rng.permutation(columns)
        permutation_map[pid_key] = dict(zip(shuffled_cols, values))

    def _apply(pid_key, colname):
        return permutation_map[pid_key].get(colname, np.nan)

    for c in columns:
        df[c] = df["iid"].map(lambda p: _apply(p, c))
        df[f"{c}_o"] = df["pid"].map(lambda p: _apply(p, c))

    return df

def add_rich_bio_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds `own_text_rich` and `partner_text_rich` columns: field+career text
    (same as the plain proxy) plus a full interest-bio sentence, for use as
    a richer alternative embedding-model input. Expects a deduped dataframe
    (has `field_o`/`career_o` and the `<interest>_o` columns already).
    """
    df = df.copy()
    df["field"] = df["field"].fillna("")
    df["career"] = df["career"].fillna("")
    df["field_o"] = df["field_o"].fillna("")
    df["career_o"] = df["career_o"].fillna("")

    own_interest_text = df.apply(lambda r: build_interest_bio(r, suffix=""), axis=1)
    partner_interest_text = df.apply(lambda r: build_interest_bio(r, suffix="_o"), axis=1)

    df["own_text_rich"] = (df["field"] + " " + df["career"] + ". " + own_interest_text).str.strip()
    df["partner_text_rich"] = (df["field_o"] + " " + df["career_o"] + ". " + partner_interest_text).str.strip()

    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_loading import load_speed_dating, dedupe_directional_pairs

    sd = load_speed_dating()
    sd = dedupe_directional_pairs(sd)
    sd = add_rich_bio_text(sd)

    print("Example rich bio texts:\n")
    for i in range(3):
        print(f"own:     {sd.loc[i, 'own_text_rich']}")
        print(f"partner: {sd.loc[i, 'partner_text_rich']}")
        print()

    empty = (sd["own_text_rich"].str.strip() == "").sum()
    print(f"rows with empty own_text_rich: {empty} / {len(sd)}")
