"""
Loader for the Speed Dating Experiment dataset (Fisman & Iyengar, Columbia).

The raw CSV as exported has two quirks that break a plain pd.read_csv:

1. Encoding: the file is cp1252 (Windows-1252), not UTF-8. There are accented
   characters (e.g. in `field`/`career` free text) that fail utf-8 decoding.
2. Structure: every data row (not the header) is wrapped as a single quoted
   field, e.g.

       "1,1,0,...,Chicago,""60,521"",""69,487.00"",...,lawyer,...";;

   This looks like the row was serialized as one escaped CSV field and then
   re-wrapped in an outer quoted field with a trailing ";;". To recover the
   real columns we:
     - strip the trailing '";;' and the leading '"'
     - un-escape doubled quotes ('""' -> '"'), which restores the *inner*
       quoting Excel originally used to protect thousand-separated numbers
       like "60,521"
     - parse the result with a normal csv.reader (delimiter=',', quotechar='"')

The header row is untouched plain comma-separated text, so it's read normally.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "speed_dating.csv"

# Columns we actually need for the fusion-vs-baseline research question.
# int_corr is loaded too, but is NOT used anywhere: it's a pre-computed
# statistic from the original study (not something Conecta could ever have
# for real users) and is kept only for reference/inspection, never as a
# model input and never as a "validation" comparison (an earlier version of
# this project's notes framed it as an external validation signal for
# embedding_cosine_similarity, but that comparison doesn't hold up
# logically -- int_corr and embedding_cosine_similarity measure different
# constructs and have no principled reason to agree; see project notes).
INTEREST_COLUMNS = [
    "sports", "tvsports", "exercise", "dining", "museums", "art", "hiking",
    "gaming", "clubbing", "reading", "tv", "theater", "movies", "concerts",
    "music", "shopping", "yoga",
]

RELEVANT_COLUMNS = [
    "iid", "id", "pid", "gender", "wave", "match", "int_corr", "samerace",
    "age", "age_o", "field_cd", "career_c", "field", "career",
] + INTEREST_COLUMNS


def _clean_row(raw_line: str) -> str | None:
    """
    Undo the row-wrapping quirk on a single data line.

    The file mixes two row formats (empirically, ~92% / ~8% of rows):
      - Quote-wrapped: rows that contain a comma-bearing quoted sub-field
        (e.g. a thousands-separated number like "60,521") got the *whole*
        row wrapped in an extra outer pair of quotes with all inner quotes
        doubled: `"...,""60,521"",...";;`
      - Plain: rows with nothing that needed quoting were left as ordinary
        comma-separated text, just with a trailing ";;" sentinel:
        `23,3,0,...,;;`

    Both cases always end with the literal ";;" sentinel before the newline.
    """
    line = raw_line.rstrip("\r\n")
    if not line:
        return None
    if not line.endswith(";;"):
        return None
    line = line[:-2]
    if line.startswith('"') and line.endswith('"'):
        line = line[1:-1]
        line = line.replace('""', '"')
    return line


def load_raw(path: Path = RAW_PATH, encoding: str = "cp1252") -> pd.DataFrame:
    """Parse the full raw Speed Dating CSV into a DataFrame with all 195 columns."""
    with open(path, encoding=encoding) as f:
        header = next(f).strip()
        header_fields = header.split(",")
        n_cols = len(header_fields)

        rows = []
        skipped = 0
        for raw_line in f:
            cleaned = _clean_row(raw_line)
            if cleaned is None:
                skipped += 1
                continue
            parsed = next(csv.reader(io.StringIO(cleaned)))
            if len(parsed) != n_cols:
                # Defensive: don't let a malformed row silently misalign columns
                skipped += 1
                continue
            rows.append(parsed)

    if skipped:
        # A small number of rows (~29 in the known dataset, ~0.3%) are
        # genuinely corrupted at the source: their `career` free-text field
        # contains a literal semicolon (e.g. "Academia; Research; Teaching"),
        # which collides with the ";;" row-end sentinel used by whatever
        # tool originally re-exported this file. These rows cannot be
        # recovered without the original data and are dropped rather than
        # guessed at.
        print(f"[data_loading] skipped {skipped} corrupted/unparseable line(s) "
              f"out of {len(rows) + skipped} total data rows "
              f"(see docstring: literal ';' in free-text fields)")

    df = pd.DataFrame(rows, columns=header_fields)
    return df


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def dedupe_directional_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the two directional rows per date (iid->pid and pid->iid) into
    one row per unordered pair.

    Each real-world date generates two rows in the raw data: one from each
    participant's perspective. `match`, `int_corr`, and `age`/`age_o` are
    already consistent/complementary across both rows (age_o already IS the
    partner's own age -- verified directly against the data), but
    field_cd/career_c/field/career only describe whoever is `iid` in that
    row, with no `_o` counterpart in the raw columns.

    Keeping both directional rows as separate training examples causes two
    problems:
      1. Pseudo-replication: the same date is counted as two independent
         observations, inflating effective sample size and making bootstrap
         CIs look tighter than they should.
      2. Leakage across train/test splits: a GroupShuffleSplit grouped by
         `iid` can put one direction of a date in train and the other
         direction of the SAME date in test, since the two rows have
         different `iid` values.

    This adds field_cd_o/career_c_o/field_o/career_o (partner's own values,
    via self-join on iid -- same pattern as add_same_field_via_self_join),
    plus an `_o` counterpart for each of the 17 interest-rating columns
    (INTEREST_COLUMNS), then keeps exactly one row per unordered {iid, pid}
    pair (iid < pid).
    """
    df = df.copy()

    own_field_cd = df.groupby("iid")["field_cd"].first()
    own_career_c = df.groupby("iid")["career_c"].first()
    own_field = df.groupby("iid")["field"].first()
    own_career = df.groupby("iid")["career"].first()

    df["field_cd_o"] = df["pid"].map(own_field_cd)
    df["career_c_o"] = df["pid"].map(own_career_c)
    df["field_o"] = df["pid"].map(own_field)
    df["career_o"] = df["pid"].map(own_career)

    for col in INTEREST_COLUMNS:
        if col in df.columns:
            own_val = df.groupby("iid")[col].first()
            df[f"{col}_o"] = df["pid"].map(own_val)

    before = len(df)
    deduped = df[df["iid"] < df["pid"]].reset_index(drop=True)
    print(f"[data_loading] dedupe_directional_pairs: {before} directional rows "
          f"-> {len(deduped)} unique pairs")
    return deduped


def load_speed_dating(path: Path = RAW_PATH, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Load and lightly clean the Speed Dating dataset, keeping only the columns
    relevant to the fusion vs. embedding-only research question.

    Note: `samerace` is loaded here for completeness / potential fairness
    auditing, but per the project's fairness constraint it must NOT be used
    as a model input feature.
    """
    df = load_raw(path)
    columns = columns or RELEVANT_COLUMNS
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Expected columns not found in raw data: {missing}")

    df = df[columns].copy()

    numeric_cols = [
        "iid", "id", "pid", "gender", "wave", "match", "int_corr", "samerace",
        "age", "age_o", "field_cd", "career_c",
    ] + INTEREST_COLUMNS
    df = _to_numeric(df, numeric_cols)

    # A handful of interest-rating rows have out-of-scale values (13, 14 on
    # what should be a 0-10 scale) -- almost certainly data entry errors in
    # the original study. Clip rather than drop, since it's a tiny number of
    # values (129 out of ~140k) and only affects the interest-bio text
    # bucketing, not any statistical feature.
    for col in INTEREST_COLUMNS:
        if col in df.columns:
            df[col] = df[col].clip(upper=10)

    # match and samerace are boolean-coded ints (0/1)
    for c in ("match", "samerace"):
        if c in df.columns:
            df[c] = df[c].astype("Int64")

    return df


if __name__ == "__main__":
    df = load_speed_dating()
    print(df.shape)
    print(df.dtypes)
    print(df.head())
    print("\nmatch rate:", df["match"].mean())
    print("rows with missing int_corr:", df["int_corr"].isna().sum())
