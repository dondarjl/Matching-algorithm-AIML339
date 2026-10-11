# Conecta Matching Engine (AIML339)

Semantic matching engine for Conecta, an app that connects exchange students with locals.
**Research question:** does a learned fusion of sentence-embedding similarity with structured
compatibility features outperform an off-the-shelf embedding baseline alone?

**Short answer (see the report for the full analysis):** a small, borderline-significant AUC
advantage for fusion over the embedding-only baseline (+0.027 pooled AUC, cluster-bootstrap
p = 0.022, Holm-adjusted 0.044), but no PR-AUC advantage, no robust advantage over the
structured features alone, and wide variation across folds. Treat the effect as fragile.

## Design

Two tracks, because Conecta has no outcome labels yet.

- **Track 1 (evaluation):** the Speed Dating dataset (Fisman et al., 2006) as a proxy for
  "do two people want to connect?". Label = mutual match. Embedding text = free-text field of study
  and career (`field` + `career`) of each person.
- **Track 2 (demo, no labels):** 20 hand-authored synthetic Conecta profiles. The same feature
  pipeline ranks pairs with an unsupervised composite score. It is a qualitative demonstration
  of domain transfer, not a second trained model.

Models compared: **baseline** (cosine similarity of all-MiniLM-L6-v2 embeddings, logistic
regression on that single feature), **structured** (`same_field`, `same_career`,
`age_proximity`), and **fusion** (embedding similarity + structured features).

## Repository layout

```
src/
  data_loading.py            Custom Speed Dating parser (cp1252, mixed row formats) + pair deduplication
  features.py                Pairwise features (Track 1 proxies and Track 2 features)
  models.py                  EmbeddingModel (MiniLM), MockEmbeddingModel (offline, for tests),
                             StructuredModel, FusionModel, EmbeddingOnlyBaseline
  evaluate.py                Metrics, row/cluster bootstrap, McNemar, Holm, per-participant ranking
                             metrics (participant bootstrap and wave-cluster bootstrap)
  train_track1.py            Single-split pipeline (kept for reference; superseded by CV)
  cross_validate_track1.py   Wave-grouped GroupKFold CV, fusion vs. baseline
  compare_three_models.py    MAIN RESULTS: baseline vs. structured vs. fusion, all pairwise tests
  interest_bio.py            Richer text proxy that adds interest ratings ("rich"), tested and discarded
  validate_rich_proxy.py     Permutation null check for the "rich" proxy
  validate_sts_benchmark.py  Validates the embedder on STS-Benchmark (Spearman 0.82)
  check_int_corr_vs_embedding.py  Diagnostic comparison of int_corr and embedding similarity
  inspect_text_proxy.py      Prints sample text proxies for eyeballing
  generate_synthetic_profiles.py / demo_track2.py / schema.py   Track 2
  make_figures.py            Regenerates the report figures from the final-run numbers
data/
  raw/speed_dating.csv       Speed Dating Experiment (Fisman et al., 2006)
  sts_benchmark/             sts-test.csv, sts-dev.csv (STS Benchmark; Cer et al., 2017)
  synthetic/profiles.json    20 synthetic Conecta profiles
results/                     Logs of the final runs and the report figures
docs/development_log.md      Historical running notes (superseded in places; see its banner)
```

## Setup

Python 3.11. Install the pinned dependencies:

```
pip install -r requirements.txt
```

The first `--real` run downloads `all-MiniLM-L6-v2` from the Hugging Face Hub, so it needs
internet access once; later runs use the local cache. Without `--real`, scripts use
`MockEmbeddingModel` (random vectors seeded by the text): the pipeline runs, but results
involving embeddings are **meaningless**.

## Reproducing the report

Run from `src/`. Each command was run with `--real`; logs are in `results/`.

```
python validate_sts_benchmark.py --real        # embedder sanity check (Spearman r = 0.82)
python compare_three_models.py --real          # MAIN: Tables I-IV, Holm, per-participant tests
python validate_rich_proxy.py --real           # negative result: rich text proxy (Section VIII-E)
python make_figures.py                         # Fig. 1 and Fig. 2 -> results/
```

All randomised steps use seed 0 (`--random-state 0`, the default). Logistic regression and
GroupKFold are deterministic, so reruns reproduce the numbers.

## Key results (final run, 4,061 pairs, 21 waves, 7-fold GroupKFold by wave)

| Model | AUC | PR-AUC | Accuracy |
|---|---|---|---|
| Baseline (embedding only) | 0.541 | 0.203 | 0.619 |
| Structured only | 0.551 | 0.196 | 0.547 |
| Fusion | 0.567 | 0.208 | 0.578 |

- Fusion vs. baseline: ΔAUC +0.027, cluster 95% CI [0.003, 0.052], p = 0.022 (Holm-adjusted over
  AUC and PR-AUC: 0.044); ΔPR-AUC +0.006, p = 0.443.
- Per-participant analysis (exploratory): wave-cluster bootstrap, fusion vs. structured +0.021
  (p = 0.025, Holm 0.050), fusion vs. baseline +0.019 (p = 0.079).
- Fusion beats the baseline in only 4 of 7 folds; fold AUCs for the baseline range 0.43-0.67.
- The "rich" text proxy (adding interest ratings as text) is indistinguishable from chance
  (AUC 0.499) and was discarded.

## Methodological decisions

- **Deduplication.** Each date appears twice (one row per direction). Keeping both is
  pseudo-replication and leaks across splits, so `dedupe_directional_pairs` keeps one row per
  pair: 8,349 directional rows -> 4,169 unique pairs (29 of 8,378 raw rows are unrecoverable
  and logged as skipped).
- **Grouping.** Splits are grouped by `wave`; no participant appears in more than one wave
  (checked: 0 of 549), so grouping by wave isolates each participant's dates.
- **Inference.** Pairs within a wave share session effects, so confidence intervals resample
  whole waves (cluster bootstrap). Two primary tests (AUC, PR-AUC; fusion vs. baseline) are
  Holm-corrected. The per-participant analysis uses a wave-cluster bootstrap; resampling
  participants as independent was found to be anti-conservative and is not used.
- **Excluded signals.** `int_corr` (precomputed interest-rating correlation) is not a model input:
  Conecta would not have it for real users, so training on it would misrepresent what is
  deployable. Individual interest ratings are used only in the discarded "rich" text proxy,
  as text. `samerace` and nationality are excluded on ethical grounds.
- **Features.** `same_field` and `same_career` (category codes, exact match) are correlated at
  r = 0.34 but not redundant; both kept. Free text feeds the embedder; codes feed structured features.
- **Not used.** The OkCupid dataset (non-consensual scraping).

## Limitations

One proxy dataset with 21 waves; a thin text input (field and career, 2-4 words); no
non-linear fusion model, lexical baseline or recent specialised baseline; Track 2 has no labels.
Pooled out-of-fold predictions come from seven separately fitted models. See the report,
Section IX.

## Tests of the evaluation code

`evaluate.py` functions (`holm_adjust`, bootstraps) were checked against known values: for example
`holm_adjust([0.022, 0.443])` returns `[0.044, 0.443]`.