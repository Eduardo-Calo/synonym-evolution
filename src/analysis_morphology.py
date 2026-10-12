from pathlib import Path
import re

import polars as pl

from scipy.stats import chi2_contingency
from nltk.stem import SnowballStemmer


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "../data/sample_pairs"

LD_CATS = {
    "Lost",
    "Split",
}

LPC_CATS = {
    "Persistent with Same Addition",
    "Persistent with Same Addition and Loss",
    "Persistent with Same Loss",
}

TARGET_CATS = LD_CATS | LPC_CATS

stemmer = SnowballStemmer("english")


# ============================================================
# LOAD FILES
# ============================================================

all_dfs = []

pattern = re.compile(
    r"sample_pairs_(ADJ|NOUN|VERB)_thr[\d\.]+_(\d{4})_(\d{4})\.csv"
)

for filepath in Path(DATA_DIR).glob("sample_pairs_*.csv"):

    match = pattern.search(filepath.name)

    if not match:
        continue

    pos = match.group(1)
    t1 = int(match.group(2))
    t2 = int(match.group(3))

    span_length = t2 - t1

    df = pl.read_csv(filepath)

    df = (
        df
        .filter(pl.col("category").is_in(TARGET_CATS))
        .with_columns([
            pl.lit(pos).alias("POS"),
            pl.lit(t1).alias("t1"),
            pl.lit(t2).alias("t2"),
            pl.lit(span_length).alias("span_length"),
        ])
    )

    if df.height == 0:
        continue

    all_dfs.append(df)

df = pl.concat(all_dfs)

print(f"Loaded {df.height:,} rows")


# ============================================================
# LAW LABELS
# ============================================================

df = df.with_columns([
    pl.when(pl.col("category").is_in(LD_CATS))
    .then(pl.lit("LD"))
    .otherwise(pl.lit("LPC"))
    .alias("law")
])


# ============================================================
# STEM FEATURES
# ============================================================

# collect unique lemmas only once
unique_lemmas = set(df["lemma1"].to_list()) | set(df["lemma2"].to_list())

print(f"Unique lemmas: {len(unique_lemmas):,}")

# stem each unique lemma once
stem_map = {
    lemma: stemmer.stem(lemma.lower())
    for lemma in unique_lemmas
}

df = df.with_columns([

    pl.col("lemma1")
    .replace_strict(stem_map)
    .alias("lemma1_stem"),

    pl.col("lemma2")
    .replace_strict(stem_map)
    .alias("lemma2_stem"),

])

df = df.with_columns([
    (
        pl.col("lemma1_stem") ==
        pl.col("lemma2_stem")
    )
    .cast(pl.Int8)
    .alias("same_family")
])


# ============================================================
# BASIC COUNTS
# ============================================================

print("\n=== CATEGORY COUNTS ===")
print(df.group_by("category").len().sort("len", descending=True))

print("\n=== LAW COUNTS ===")
print(df.group_by("law").len())


# ============================================================
# OVERALL SUMMARY
# ============================================================

overall_summary = (
    df
    .group_by("law")
    .agg([

        pl.col("same_family")
        .mean()
        .mul(100)
        .round(2)
        .alias("pct_same_family"),

    ])
)

print("\n=== OVERALL SUMMARY ===")
print(overall_summary)


# ============================================================
# POS SUMMARY
# ============================================================

pos_summary = (
    df
    .group_by(["POS", "law"])
    .agg([

        pl.col("same_family")
        .mean()
        .mul(100)
        .round(2)
        .alias("pct_same_family"),

    ])
    .sort(["POS", "law"])
)

print("\n=== POS SUMMARY ===")
print(pos_summary)


# ============================================================
# POS SUMMARY BY CATEGORY
# ============================================================

pos_category_summary = (
    df
    .group_by(["POS", "category"])
    .agg([

        pl.col("same_family")
        .mean()
        .mul(100)
        .round(2)
        .alias("pct_same_family"),

    ])
    .sort(["POS", "category"])
)

print("\n=== POS SUMMARY BY CATEGORY ===")

pl.Config.set_tbl_rows(df.height)

print(pos_category_summary)


# ============================================================
# OVERALL CHI-SQUARE
# ============================================================

print("\n=== OVERALL CHI-SQUARE TESTS ===")

contingency = (
    df
    .group_by(["law", "same_family"])
    .len()
    .pivot(
        values="len",
        index="law",
        on="same_family",
    )
    .fill_null(0)
)

table = contingency.select(["0", "1"]).to_numpy()

chi2, p, dof, expected = chi2_contingency(table)

print("\nLD vs LPC")
print(f"chi2 = {chi2:.2f}")
print(f"p = {p:.8f}")


# ============================================================
# PER-POS TESTS
# ============================================================

for pos in ["ADJ", "NOUN", "VERB"]:

    subdf = df.filter(pl.col("POS") == pos)

    contingency = (
        subdf
        .group_by(["law", "same_family"])
        .len()
        .pivot(
            values="len",
            index="law",
            on="same_family",
        )
        .fill_null(0)
    )

    table = contingency.select(["0", "1"]).to_numpy()

    chi2, p, dof, expected = chi2_contingency(table)

    print(f"\n=== {pos} ===")
    print(f"chi2 = {chi2:.2f}")
    print(f"p = {p:.8f}")


# ============================================================
# CATEGORY TEST
# ============================================================

print("\n=== CATEGORY CHI-SQUARE ===")

contingency = (
    df
    .group_by(["category", "same_family"])
    .len()
    .pivot(
        values="len",
        index="category",
        on="same_family",
    )
    .fill_null(0)
)

table = contingency.select(["0", "1"]).to_numpy()

chi2, p, dof, expected = chi2_contingency(table)

print(f"chi2 = {chi2:.2f}")
print(f"p = {p:.8f}")