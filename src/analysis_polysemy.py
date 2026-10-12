from pathlib import Path
import re

import polars as pl

from scipy.stats import mannwhitneyu


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
# POLYSEMY FEATURES
# ============================================================

df = df.with_columns([

    pl.col("lemma1_t1_senses_count")
    .alias("lemma1_polysemy"),

    pl.col("lemma2_t1_senses_count")
    .alias("lemma2_polysemy"),

    (
        (
            pl.col("lemma1_t1_senses_count") +
            pl.col("lemma2_t1_senses_count")
        ) / 2
    ).alias("mean_polysemy"),

    pl.max_horizontal([
        pl.col("lemma1_t1_senses_count"),
        pl.col("lemma2_t1_senses_count"),
    ]).alias("max_polysemy"),

    pl.min_horizontal([
        pl.col("lemma1_t1_senses_count"),
        pl.col("lemma2_t1_senses_count"),
    ]).alias("min_polysemy"),

    (
        pl.col("lemma1_t1_senses_count") -
        pl.col("lemma2_t1_senses_count")
    ).abs().alias("polysemy_diff"),

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

        pl.col("mean_polysemy").mean().round(2).alias("mean_poly_mean"),
        pl.col("mean_polysemy").median().round(2).alias("mean_poly_median"),
        pl.col("mean_polysemy").std().round(2).alias("mean_poly_std"),

        pl.col("polysemy_diff").mean().round(2).alias("diff_mean"),
        pl.col("polysemy_diff").median().round(2).alias("diff_median"),
        pl.col("polysemy_diff").std().round(2).alias("diff_std"),

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

        pl.col("mean_polysemy").mean().round(2).alias("mean_poly"),

        pl.col("polysemy_diff").mean().round(2).alias("poly_diff"),

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

        pl.col("mean_polysemy").mean().round(2).alias("mean_poly"),

        pl.col("polysemy_diff").mean().round(2).alias("poly_diff"),

    ])
    .sort(["POS", "category"])
)

print("\n=== POS SUMMARY BY CATEGORY ===")
# print all lines
pl.Config.set_tbl_rows(df.height)
print(pos_category_summary)


# ============================================================
# STATISTICAL TESTS
# ============================================================

print("\n=== OVERALL MANN-WHITNEY TESTS ===")

ld_mean = (
    df
    .filter(pl.col("law") == "LD")
    .get_column("mean_polysemy")
    .to_numpy()
)

lpc_mean = (
    df
    .filter(pl.col("law") == "LPC")
    .get_column("mean_polysemy")
    .to_numpy()
)

u, p = mannwhitneyu(ld_mean, lpc_mean)

print("\nMean polysemy")
print(f"U = {u:.2f}")
print(f"p = {p:.8f}")


ld_diff = (
    df
    .filter(pl.col("law") == "LD")
    .get_column("polysemy_diff")
    .to_numpy()
)

lpc_diff = (
    df
    .filter(pl.col("law") == "LPC")
    .get_column("polysemy_diff")
    .to_numpy()
)

u, p = mannwhitneyu(ld_diff, lpc_diff)

print("\nPolysemy asymmetry")
print(f"U = {u:.2f}")
print(f"p = {p:.8f}")


# ============================================================
# PER-POS TESTS
# ============================================================

for pos in ["ADJ", "NOUN", "VERB"]:

    subdf = df.filter(pl.col("POS") == pos)

    ld_mean = (
        subdf
        .filter(pl.col("law") == "LD")
        .get_column("mean_polysemy")
        .to_numpy()
    )

    lpc_mean = (
        subdf
        .filter(pl.col("law") == "LPC")
        .get_column("mean_polysemy")
        .to_numpy()
    )

    if len(ld_mean) > 0 and len(lpc_mean) > 0:

        u, p = mannwhitneyu(ld_mean, lpc_mean)

        print(f"\n=== {pos} | MEAN POLYSEMY ===")
        print(f"U = {u:.2f}")
        print(f"p = {p:.8f}")

    ld_diff = (
        subdf
        .filter(pl.col("law") == "LD")
        .get_column("polysemy_diff")
        .to_numpy()
    )

    lpc_diff = (
        subdf
        .filter(pl.col("law") == "LPC")
        .get_column("polysemy_diff")
        .to_numpy()
    )

    if len(ld_diff) > 0 and len(lpc_diff) > 0:

        u, p = mannwhitneyu(ld_diff, lpc_diff)

        print(f"\n=== {pos} | POLYSEMY DIFF ===")
        print(f"U = {u:.2f}")
        print(f"p = {p:.8f}")