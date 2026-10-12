from itertools import combinations
from typing import List, Dict, Any, Set, Tuple
import argparse

import numpy as np
from collections import defaultdict
import polars as pl
from tqdm import tqdm
import random
import os
import json
from pathlib import Path

random.seed(42)


def load_qwen_dissimilar_idxs(labels_path: str) -> set[str]:
    """Load idx values with predicted_label == DISSIMILAR from a Qwen JSONL labels file."""
    dissimilar = set()
    with open(labels_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            label = str(row.get("predicted_label", "")).upper().strip()
            idx = str(row.get("idx", "")).strip()
            if label == "DISSIMILAR" and idx:
                dissimilar.add(idx)
    return dissimilar


def remove_qwen_dissimilar_pairs(pairs_df: pl.DataFrame, pairs_csv: str) -> pl.DataFrame:
    """Remove rows that Qwen labeled as DISSIMILAR for the matching POS."""
    if "sense_id1" not in pairs_df.columns or "sense_id2" not in pairs_df.columns:
        print("Warning: sense_id columns not found; skipping Qwen dissimilar filtering")
        return pairs_df

    input_name = Path(pairs_csv).name
    pos = input_name.split("_", 1)[0].upper()
    labels_dir = Path(__file__).resolve().parents[1] / "data" / "llm_output" / "Qwen_Qwen3_5-27B"
    labels_paths = sorted(labels_dir.glob(f"{pos}*_hf_labels_Qwen_Qwen3_5-27B.jsonl"))

    if not labels_paths:
        print(f"Warning: Qwen labels files not found in {labels_dir} for POS={pos}; skipping dissimilar filtering")
        return pairs_df

    dissimilar_idxs = set()
    for labels_path in labels_paths:
        dissimilar_idxs.update(load_qwen_dissimilar_idxs(str(labels_path)))

    if not dissimilar_idxs:
        print(f"No DISSIMILAR labels found in {len(labels_paths)} Qwen label files for POS={pos}; no rows removed")
        return pairs_df

    df = pairs_df.with_columns(
        pl.col("sense_id1").cast(pl.Utf8).fill_null("").alias("_s1"),
        pl.col("sense_id2").cast(pl.Utf8).fill_null("").alias("_s2"),
    ).with_columns(
        (pl.col("_s1") + pl.lit("|||") + pl.col("_s2")).alias("_idx_forward"),
        (pl.col("_s2") + pl.lit("|||") + pl.col("_s1")).alias("_idx_reverse"),
    )

    remove_mask = pl.col("_idx_forward").is_in(list(dissimilar_idxs)) | pl.col("_idx_reverse").is_in(list(dissimilar_idxs))
    n_removed = int(df.filter(remove_mask).height)
    if n_removed == 0:
        print(f"Qwen dissimilar filtering checked {len(pairs_df)} rows; removed 0")
        return pairs_df

    filtered_df = df.filter(~remove_mask).drop(["_s1", "_s2", "_idx_forward", "_idx_reverse"])
    print(f"Removed {n_removed} rows labeled DISSIMILAR by Qwen from {len(labels_paths)} label files for POS={pos}")
    return filtered_df


def analyze_semantic_change(syn_pairs_csv: str, non_syn_pairs_csv: str, syn: bool, lus_df: pl.DataFrame, t1: int, t2: int, threshold: float = 0.8):
    """
    Performs a full, structured analysis of semantic change between two time points using similarity pairs.

    This is single lemma-based.

    LOST: does the lemma die at t2?

    NO CHANGE: does nothing change for the lemma at t2?

    SOME ADDITION: does it add any new LUs at t2 that were not present at t1?
    SOME LOSS: does it lose any LUs at t2 that were present at t1?
    SOME ADDITION + LOSS: SOME ADDITION and SOME LOSS both occur for the same lemma.
    """
    print(f"\n--- Starting Analysis between t1={t1} and t2={t2} with threshold {threshold} ---")

    # Load pairs data
    syn_pairs_df = pl.read_csv(syn_pairs_csv)  # we need this only for PARALLEL ADDITION and PARALLEL LOSS
    syn_pairs_df = remove_qwen_dissimilar_pairs(syn_pairs_df, syn_pairs_csv)  # Remove pairs labeled DISSIMILAR by Qwen

    non_syn_pairs_df = pl.read_csv(non_syn_pairs_csv)
    non_syn_pairs_df = non_syn_pairs_df.filter(pl.col("cosine") < threshold)

    # make syn and non-syn sets disjoint, ie if a lemma appears in syn, remove from non-syn
    syn_lemmas = set(
    pl.concat([
        syn_pairs_df["lu1"],
        syn_pairs_df["lu2"]
    ]).unique().to_list()
    )
    non_syn_lemmas = set(
        pl.concat([
            non_syn_pairs_df["lu1"],
            non_syn_pairs_df["lu2"]
        ]).unique().to_list()
    )

    overlap_lemmas = syn_lemmas & non_syn_lemmas
    if overlap_lemmas:
        print(f"Warning: {len(overlap_lemmas)} lemmas appear in both synonym and non-synonym pairs; removing from non-synonym set")
        non_syn_pairs_df = non_syn_pairs_df.filter(
            ~pl.col("lu1").is_in(list(overlap_lemmas)) &
            ~pl.col("lu2").is_in(list(overlap_lemmas))
        )

    if syn:
        lemmas = set(
        pl.concat([
            syn_pairs_df["lu1"],
            syn_pairs_df["lu2"]
        ]).unique().to_list()
        )
        print(f"Loaded {len(syn_pairs_df)} synonym pairs involving {len(lemmas)} lemmas.")
    else:
        lemmas = set(
        pl.concat([
            non_syn_pairs_df["lu1"],
            non_syn_pairs_df["lu2"]
        ]).unique().to_list()
        )
        print(f"Loaded {len(non_syn_pairs_df)} non-synonym pairs involving {len(lemmas)} lemmas.")

    # Create map from (lemma, definition) to sense_id
    sense_key_to_id = {}
    for row in lus_df.select(["lemma", "definition", "sense_id"]).iter_rows(named=True):
        key = (row["lemma"], row["definition"])
        sense_key_to_id[key] = row["sense_id"]

    # Step 3: Initialize containers for each category of change.
    lost = set()

    # Categories for persistent pairs
    unchanged = set()
    some_addition = set()
    some_loss = set()
    both_addition_and_loss = set()

    # Step 4: Build active categories for t1 and t2
    print("Building active senses maps...")
    active_senses_t1 = build_active_senses_map(lus_df, t1)
    active_senses_t2 = build_active_senses_map(lus_df, t2)

    # Consider only those in pairs_df and active at t1
    considered_lemmas = {w for w in lemmas if w in active_senses_t1}

    # Step 6: Classify every lemma
    for w in tqdm(considered_lemmas, desc="Classifying lemmas"):

        senses_w_t1 = active_senses_t1.get(w, set())
        senses_w_t2 = active_senses_t2.get(w, set())

        if not senses_w_t2:
            lost.add(w)
            continue

        if senses_w_t1 == senses_w_t2:
            unchanged.add(w)
            continue

        new_senses_w = senses_w_t2 - senses_w_t1
        lost_senses_w = senses_w_t1 - senses_w_t2

        has_additions = bool(new_senses_w)
        has_losses = bool(lost_senses_w)

        if has_additions and has_losses:
            both_addition_and_loss.add(w)

        elif has_additions:
            some_addition.add(w)

        elif has_losses:
            some_loss.add(w)

    # --- Summary Table ---
    print("\n--- Summary Table ---")
    print(f"Lost\t{len(lost)}")
    print(f"No Change\t{len(unchanged)}")
    print(f"Addition\t{len(some_addition)}")
    print(f"Loss\t{len(some_loss)}")
    print(f"Addition + Loss\t{len(both_addition_and_loss)}")
    print(f"Total\t{len(considered_lemmas)}")

    return {
        "lost_pairs": lost,
        "persistent_unchanged": unchanged,
        "persistent_with_some_addition": some_addition,
        "persistent_with_some_loss": some_loss,
        "persistent_with_both_addition_and_loss": both_addition_and_loss,
        "active_senses_t1": active_senses_t1,
        "active_senses_t2": active_senses_t2
    }


def build_active_senses_map(lus_df: pl.DataFrame, year: int) -> dict[str, set[str]]:
    df = lus_df.filter(
        # Discard if no start year
        pl.col("start").is_not_null() &
        
        # Must have started by or in this year
        (pl.col("start") <= year) &
        
        # If end is null, it's still active; otherwise, must end after/on this year
        (pl.col("end").is_null() | (pl.col("end") >= year))
    )

    # Group sense_ids by lemma
    agg_df = df.group_by("lemma").agg(pl.col("sense_id"))
    return {row["lemma"]: set(row["sense_id"]) for row in agg_df.iter_rows(named=True)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze semantic change between two time points using similarity pairs.")
    parser.add_argument("pos", help="Part of speech (e.g., NOUN, VERB, ADJ).")
    parser.add_argument("t1", type=int, help="First time point (year).")
    parser.add_argument("t2", type=int, help="Second time point (year).")
    # parser.add_argument("--pos", default="ADJ", help="Part of speech (default: NOUN).")
    # parser.add_argument("--t1", type=int, default=1500, help="First time point (year) (default: 1850).")
    # parser.add_argument("--t2", type=int, default=2025, help="Second time point (year) (default: 1950).")
    parser.add_argument("--threshold", type=float, default=0.8, help="Similarity threshold for synonymy (default: 0.8).")
    parser.add_argument("--syn_pairs_csv", default=None, help="CSV file for synonym pairs. If not specified, defaults to {pos}_pairs_thr{threshold}_1.1_both.csv.")
    parser.add_argument("--non_syn_pairs_csv", default=None, help="CSV file for non-synonym pairs. If not specified, defaults to {pos}_pairs_thr0.0_0.85_both.csv.")
    parser.add_argument("--mode", choices=['syn', 'non-syn'], default='syn', help="Whether to analyze synonym pairs (default) or non-synonym pairs.")

    args = parser.parse_args()

    # Construct file paths
    syn = f"../data/extracted/{args.pos}_pairs_thr{args.threshold}_1.1_both.csv"
    non_syn = f"../data/extracted/{args.pos}_pairs_thr0.0_0.85_both.csv"
    lus_df_path = f"../data/parsed/{args.pos}_wn.csv"

    # Load parsed data to get all lemmas
    lus_df = pl.read_csv(lus_df_path)
    all_lemmas = set(
    lus_df
    .select("lemma")
    .drop_nulls()
    .unique()
    .to_series()
    .to_list())
    print(f"Loaded {len(all_lemmas)} unique lemmas from {lus_df_path}")

    # Find a deterministic example lemma with several senses
    sorted_lemmas = sorted(all_lemmas)
    example_lemma = None
    for lemma in sorted_lemmas:
        lemma_rows = lus_df.filter(pl.col("lemma") == lemma)
        if len(lemma_rows) > 2:  # Has several senses
            example_lemma = lemma
            break
    if example_lemma:
        lemma_rows = lus_df.filter(pl.col("lemma") == example_lemma)
        print(f"\nComposition of lemma '{example_lemma}' (has {len(lemma_rows)} senses):")
        for row in lemma_rows.iter_rows(named=True):
            print(
                f"  Sense: {row['sense_id']}, "
                f"Start: {row['start']}, "
                f"End: {row['end']}, "
                f"Definition: {row['definition'][:150]}..."
            )
    else:
        print("No lemma with several senses found.")

    if args.mode == 'syn':
        print("\nAnalyzing SYNONYM pairs...")
        results = analyze_semantic_change(syn, non_syn, True, lus_df, args.t1, args.t2, args.threshold)
    elif args.mode == 'non-syn':
        print("\nAnalyzing NON-SYNONYM pairs...")
        results = analyze_semantic_change(syn, non_syn, False, lus_df, args.t1, args.t2, args.threshold)