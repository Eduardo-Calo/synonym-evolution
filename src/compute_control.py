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


SEED = 42

random.seed(SEED)


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


def analyze_semantic_change(syn_pairs_csv: str, non_syn_pairs_csv: str, lus_df: pl.DataFrame, t1: int, t2: int, threshold: float = 0.8):
    """
    Performs a full, structured analysis of semantic change between two time points using similarity pairs.

    This is control group vs. compute_change.py to see if synonyms behave differently than random pairs.

    For a random pair of NON-SYNONYM lemmas at t1:

    LOST: does one or both of them die at t2?
    SPLIT: draw a random LU for each lemma in the pair, are either one or both of them lost at t2?

    NO CHANGE: does nothing change for either lemma at t2?

    PARALLEL ADDITION: do they add a shared LU? (i.e., what is the likelihood of two words becoming synonyms?)
    PARALLEL LOSS: randomly fix one LU for each lemma at t1, and check if both lost their chosen LUs at time t2.
    PARALLEL ADDITION + LOSS: PARALLEL ADDITION and PARALLEL LOSS both occur for the same pair.

    SOME ADDITION: do they add any new LUs at t2 that were not present at t1? (but not the same ones)
    SOME LOSS: do they lose any LUs at t2 that were present at t1? (but not the same ones)
    SOME ADDITION + LOSS: SOME ADDITION and SOME LOSS both occur for the same pair.
    """
    print(f"\n--- Starting Analysis between t1={t1} and t2={t2} with threshold {threshold} ---")

    # Load pairs data
    syn_pairs_df = pl.read_csv(syn_pairs_csv)  # we need this only for PARALLEL ADDITION and PARALLEL LOSS
    syn_pairs_df = remove_qwen_dissimilar_pairs(syn_pairs_df, syn_pairs_csv)  # Remove pairs labeled DISSIMILAR by Qwen

    non_syn_pairs_df = pl.read_csv(non_syn_pairs_csv)
    non_syn_pairs_df = non_syn_pairs_df.filter(pl.col("cosine") < threshold)

    # make syn and non-syn sets disjoint, ie if a pair appears in syn, remove all rows with ANY of those lemmas from non-syn
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

    non_syn_lemmas = set(
        pl.concat([
            non_syn_pairs_df["lu1"],
            non_syn_pairs_df["lu2"]
        ]).unique().to_list()
    )
    print(f"Loaded {len(syn_pairs_df)} synonym pairs involving {len(syn_lemmas)} lemmas.")
    print(f"Loaded {len(non_syn_pairs_df)} non-synonym pairs involving {len(non_syn_lemmas)} lemmas.")

    # print number of monosemous vs. polysemous lemmas in the synonym and non-synonym sets, to check if they are balanced
    lus_df = lus_df.filter(pl.col("lemma").is_in(list(syn_lemmas | non_syn_lemmas)))
    lemma_sense_counts = lus_df.group_by("lemma").agg(pl.count("sense_id").alias("sense_count"))
    # Add a 'type' column marking lemmas as monosemous or polysemous
    lemma_sense_counts = lemma_sense_counts.with_columns([
        pl.when(pl.col("sense_count") == 1)
          .then(pl.lit("monosemous"))
          .otherwise(pl.lit("polysemous"))
          .alias("type")
    ])
    syn_lemma_counts = lemma_sense_counts.filter(pl.col("lemma").is_in(list(syn_lemmas))).group_by("type").agg(pl.count("lemma").alias("count"))
    non_syn_lemma_counts = lemma_sense_counts.filter(pl.col("lemma").is_in(list(non_syn_lemmas))).group_by("type").agg(pl.count("lemma").alias("count"))
    print("\nLemma type distribution in SYNONYM set:")
    for row in syn_lemma_counts.iter_rows(named=True):
        print(f"  {row['type']}: {row['count']} lemmas")
    print("\nLemma type distribution in NON-SYNONYM set:")
    for row in non_syn_lemma_counts.iter_rows(named=True):
        print(f"  {row['type']}: {row['count']} lemmas")

    # Create map from (lemma, definition) to sense_id
    sense_key_to_id = {}
    for row in lus_df.select(["lemma", "definition", "sense_id"]).iter_rows(named=True):
        key = (row["lemma"], row["definition"])
        sense_key_to_id[key] = row["sense_id"]

    # Create set of synonymous sense pairs at t1 and t2 for checking parallel changes.
    syn_senses_t1 = build_synonymous_sense_pairs(syn_pairs_df, sense_key_to_id, threshold, t1)
    print(f"Found {len(syn_senses_t1)} synonymous sense pairs at {t1}.")
    syn_senses_t2 = build_synonymous_sense_pairs(syn_pairs_df, sense_key_to_id, threshold, t2)
    print(f"Found {len(syn_senses_t2)} synonymous sense pairs at {t2}.")

    # Step 1: Find non-synonym pairs for each time point.
    non_synonyms_t1 = find_non_synonym_lemmas_at_time(non_syn_pairs_df, t1, threshold)
    non_synonyms_t2 = find_non_synonym_lemmas_at_time(non_syn_pairs_df, t2, threshold)
    print(f"Found {len(non_synonyms_t1)} unique non-synonym lemma pairs at {t1}.")
    print(f"Found {len(non_synonyms_t2)} unique non-synonym lemma pairs at {t2}.")
    # Print some examples of non-synonym pairs for verification
    random.seed(SEED)
    if non_synonyms_t1:
        t1_list = list(non_synonyms_t1)
        examples_t1 = random.sample(t1_list, min(5, len(t1_list)))
        print(f"Examples of non-synonym pairs at {t1}: {examples_t1}")
    if non_synonyms_t2:
        t2_list = list(non_synonyms_t2)
        examples_t2 = random.sample(t2_list, min(5, len(t2_list)))
        print(f"Examples of non-synonym pairs at {t2}: {examples_t2}")

    # Step 3: Initialize containers for each category of change.
    lost_pairs = set()
    split_pairs = set()

    # Categories for persistent pairs
    persistent_non_synonyms = set()
    persistent_unchanged = set()
    persistent_with_some_change = set()
    persistent_with_some_addition = set()
    persistent_with_same_addition = set()
    persistent_with_some_loss = set()
    persistent_with_same_loss = set()
    persistent_with_same_addition_and_loss = set()
    persistent_with_both_addition_and_loss = set()

    # Step 4: Build active categories for t1 and t2
    print("Building active senses maps...")
    active_senses_t1 = build_active_senses_map(lus_df, t1)
    active_senses_t2 = build_active_senses_map(lus_df, t2)

    # Step 5: Choose one random LU for each lemma in the non-synonym pairs at t1
    print("Selecting random LUs per lemma...")
    all_lemmas_in_pairs = list(set(w for pair in non_synonyms_t1 for w in pair))
    
    active_lus_df = lus_df.filter(
        (pl.col("lemma").is_in(all_lemmas_in_pairs)) &
        (pl.col("start").is_not_null()) &
        (pl.col("start") <= t1) &
        (pl.col("end").is_null() | (pl.col("end") >= t1))
    )
    
    random_lu_df = active_lus_df.group_by("lemma").agg(
        pl.col("sense_id").sample(n=1, with_replacement=False, shuffle=True, seed=SEED).first().alias("random_lu")
    )
    
    random_lu_t1 = {row["lemma"]: row["random_lu"] for row in random_lu_df.iter_rows(named=True)}
    
    print(f"Selected random LUs for {len(random_lu_t1)} lemmas at {t1}.")

    # Step 6: Classify every non-synonym pair from t1.
    for w1, w2 in tqdm(non_synonyms_t1, desc="Classifying pairs"):

        senses_w1_t1 = active_senses_t1.get(w1, set())
        senses_w2_t1 = active_senses_t1.get(w2, set())
        senses_w1_t2 = active_senses_t2.get(w1, set())
        senses_w2_t2 = active_senses_t2.get(w2, set())

        # -------------------------
        # Case 1: death at t2
        # -------------------------
        if not senses_w1_t2 or not senses_w2_t2:
            lost_pairs.add((w1, w2))
            continue

        # -------------------------
        # SPLIT
        # -------------------------
        w1_lu = random_lu_t1.get(w1)
        w2_lu = random_lu_t1.get(w2)

        w1_lu_lost = (w1_lu is not None and w1_lu not in senses_w1_t2)
        w2_lu_lost = (w2_lu is not None and w2_lu not in senses_w2_t2)

        has_same_loss = False

        if w1_lu_lost and w2_lu_lost:
            has_same_loss = True

        if (w1_lu_lost or w2_lu_lost) and not has_same_loss:
            split_pairs.add((w1, w2))
            continue

        # -------------------------
        # Case 2: remain non-synonyms at t2
        # -------------------------

        persistent_non_synonyms.add((w1, w2))

        if senses_w1_t1 == senses_w1_t2 and senses_w2_t1 == senses_w2_t2:
            persistent_unchanged.add((w1, w2))
        else:
            persistent_with_some_change.add((w1, w2))

            new_senses_w1 = senses_w1_t2 - senses_w1_t1
            new_senses_w2 = senses_w2_t2 - senses_w2_t1
            lost_senses_w1 = senses_w1_t1 - senses_w1_t2
            lost_senses_w2 = senses_w2_t1 - senses_w2_t2

            has_additions = bool(new_senses_w1 or new_senses_w2)
            has_losses = bool(lost_senses_w1 or lost_senses_w2)

            has_same_addition = any(
                ((s1, s2) in syn_senses_t2 or (s2, s1) in syn_senses_t2) and
                not ((s1, s2) in syn_senses_t1 or (s2, s1) in syn_senses_t1)
                for s1 in new_senses_w1 for s2 in new_senses_w2
            )

            if has_same_addition and has_same_loss:
                persistent_with_same_addition_and_loss.add((w1, w2))

            elif has_same_addition:
                persistent_with_same_addition.add((w1, w2))

            elif has_same_loss:
                persistent_with_same_loss.add((w1, w2))

            elif has_additions and has_losses:
                persistent_with_both_addition_and_loss.add((w1, w2))

            elif has_additions:
                persistent_with_some_addition.add((w1, w2))

            elif has_losses:
                persistent_with_some_loss.add((w1, w2))

    # Step 7: Print the final analysis results.
    print("\n--- Analysis Results ---")

    # --- DIFFERENTIATION ---
    print("  - DIFFERENTIATION:")
    print(f"Found {len(lost_pairs)} pairs where at least one word was lost by {t2}.")
    print(f"Found {len(split_pairs)} pairs that SPLIT.")

    # --- NO CHANGE ---
    print("\n  - NO CHANGE:")
    print(f"    Found {len(persistent_unchanged)} pairs that persisted without any change to their meanings.")

    # --- CHANGE ---
    print("\n  - CHANGE (Persistent pairs where meanings changed):")
    print(f"    Found {len(persistent_with_some_change)} pairs where at least one word changed meanings.")
    print(f"\tFound {len(persistent_with_some_addition)} pairs where ONE or BOTH words acquired new meanings. [SOME ADDITION]")
    print(f"\tFound {len(persistent_with_some_loss)} pairs where ONE or BOTH words lost old meanings. [SOME LOSS]")
    print(f"\tFound {len(persistent_with_both_addition_and_loss)} pairs where ONE or BOTH words acquired new meanings and lost old meanings. [SOME ADDITION + SOME LOSS]")

    # --- PARALLEL CHANGE ---
    print("\n  - PARALLEL CHANGE (Persistent pairs with same changes):")
    print(f"    Found {len(persistent_with_same_addition_and_loss)} pairs where BOTH words acquired synonymous new meanings and lost synonymous old meanings. [SAME ADDITION + SAME LOSS]")
    print(f"    Found {len(persistent_with_same_addition)} pairs where BOTH words acquired synonymous new meanings. [SAME ADDITION]")
    print(f"    Found {len(persistent_with_same_loss)} pairs where BOTH words lost synonymous old meanings. [SAME LOSS]")

    # --- Verification ---
    print("\n--- Verification ---")
    total_accounted_for = (len(lost_pairs) + len(split_pairs) + len(persistent_unchanged) +
                           len(persistent_with_some_addition) + len(persistent_with_same_addition) +
                           len(persistent_with_some_loss) + len(persistent_with_same_loss) +
                           len(persistent_with_both_addition_and_loss) + len(persistent_with_same_addition_and_loss))
    assert len(non_synonyms_t1) == total_accounted_for, \
        "Verification failed: The sum of outcomes does not match the initial number of non-synonyms."
    print("Verification successful: All non-synonym pairs from t1 have been accounted for.")

    # --- Summary Table ---
    print("\n--- Summary Table ---")
    print(f"Lost\t{len(lost_pairs)}")
    print(f"Split\t{len(split_pairs)}")
    print(f"DIFFERENTIATION\t{len(lost_pairs) + len(split_pairs)}")
    print(f"No Change\t{len(persistent_unchanged)}")
    print(f"Parallel Addition\t{len(persistent_with_same_addition)}")
    print(f"Parallel Loss\t{len(persistent_with_same_loss)}")
    print(f"Parallel Addition + Loss\t{len(persistent_with_same_addition_and_loss)}")
    print(f"Addition\t{len(persistent_with_some_addition)}")
    print(f"Loss\t{len(persistent_with_some_loss)}")
    print(f"Addition + Loss\t{len(persistent_with_both_addition_and_loss)}")
    print(f"PERSISTENCE\t{len(persistent_non_synonyms)}")
    print(f"Total\t{len(non_synonyms_t1)}")

    return {
        "lost_pairs": lost_pairs,
        "split_pairs": split_pairs,
        "persistent_non_synonyms": persistent_non_synonyms,
        "persistent_unchanged": persistent_unchanged,
        "persistent_with_some_change": persistent_with_some_change,
        "persistent_with_some_addition": persistent_with_some_addition,
        "persistent_with_same_addition": persistent_with_same_addition,
        "persistent_with_same_addition_and_loss": persistent_with_same_addition_and_loss,
        "persistent_with_some_loss": persistent_with_some_loss,
        "persistent_with_same_loss": persistent_with_same_loss,
        "persistent_with_both_addition_and_loss": persistent_with_both_addition_and_loss,
        "syn_senses_t1": syn_senses_t1,
        "syn_senses_t2": syn_senses_t2,
        "active_senses_t1": active_senses_t1,
        "active_senses_t2": active_senses_t2
    }


def find_synonym_lemmas_at_time(
    pairs_df: pl.DataFrame,
    year: int,
    threshold: float
) -> set[tuple[str, str]]:

    df = pairs_df.filter(
        # 1. Similarity and Identity
        (pl.col("cosine") >= threshold) &
        (pl.col("lu1") != pl.col("lu2")) &
        
        # 2. Strict Start Year Requirement (No Start = Trash)
        (pl.col("start1").is_not_null()) &
        (pl.col("start2").is_not_null()) &
        
        # 3. Temporal Validity
        (pl.col("start1") <= year) &
        (pl.col("start2") <= year) &
        
        # 4. End Year logic (Null = Still Active)
        (pl.col("end1").is_null() | (pl.col("end1") >= year)) &
        (pl.col("end2").is_null() | (pl.col("end2") >= year))
    )

    # Convert to set of canonical tuples (alphabetical) using native Polars expressions
    df = df.with_columns(
        pl.when(pl.col("lu1") <= pl.col("lu2"))
          .then(pl.col("lu1")).otherwise(pl.col("lu2")).alias("first"),
        pl.when(pl.col("lu1") <= pl.col("lu2"))
          .then(pl.col("lu2")).otherwise(pl.col("lu1")).alias("second"),
    )

    return set(df.select("first", "second").unique().iter_rows())


def find_non_synonym_lemmas_at_time(
    pairs_df: pl.DataFrame,
    year: int,
    threshold: float
) -> set[tuple[str, str]]:

    df = pairs_df.filter(
        # 1. Similarity and Identity
        (pl.col("cosine") < threshold) &
        (pl.col("lu1") != pl.col("lu2")) &
        
        # 2. Strict Start Year Requirement (No Start = Trash)
        (pl.col("start1").is_not_null()) &
        (pl.col("start2").is_not_null()) &
        
        # 3. Temporal Validity
        (pl.col("start1") <= year) &
        (pl.col("start2") <= year) &
        
        # 4. End Year logic (Null = Still Active)
        (pl.col("end1").is_null() | (pl.col("end1") >= year)) &
        (pl.col("end2").is_null() | (pl.col("end2") >= year))
    )

    # Convert to set of canonical tuples (alphabetical) using native Polars expressions
    df = df.with_columns(
        pl.when(pl.col("lu1") <= pl.col("lu2"))
          .then(pl.col("lu1")).otherwise(pl.col("lu2")).alias("first"),
        pl.when(pl.col("lu1") <= pl.col("lu2"))
          .then(pl.col("lu2")).otherwise(pl.col("lu1")).alias("second"),
    )

    return set(df.select("first", "second").unique().iter_rows())


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


def build_synonymous_sense_pairs(
    pairs_df: pl.DataFrame,
    sense_key_to_id: dict,
    threshold: float,
    year: int | None = None
) -> set[tuple[str, str]]:

    # Start with identity and start-year check
    condition = (
        (pl.col("cosine") >= threshold) & 
        (pl.col("lu1") != pl.col("lu2")) &
        (pl.col("start1").is_not_null()) &
        (pl.col("start2").is_not_null())
    )

    if year is not None:
        condition &= (
            (pl.col("start1") <= year) & 
            (pl.col("end1").is_null() | (pl.col("end1") >= year)) &
            (pl.col("start2") <= year) & 
            (pl.col("end2").is_null() | (pl.col("end2") >= year))
        )

    df = pairs_df.filter(condition).select(["lu1", "def1", "lu2", "def2"])

    syn_pairs = set()
    for lu1, def1, lu2, def2 in df.iter_rows():
        s1 = sense_key_to_id.get((lu1, def1))
        s2 = sense_key_to_id.get((lu2, def2))
        
        if s1 and s2 and s1 != s2:
            syn_pairs.add(tuple(sorted((s1, s2))))

    return syn_pairs


def build_lemma_to_synonyms(synonym_pairs: set[tuple[str, str]]) -> dict[str, set[str]]:
    """
    Builds a map from each lemma to the set of lemmas it is synonymous with.
    """
    lemma_to_synonyms = defaultdict(set)
    for w1, w2 in synonym_pairs:
        lemma_to_synonyms[w1].add(w2)
        lemma_to_synonyms[w2].add(w1)
    return dict(lemma_to_synonyms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze semantic change between two time points using similarity pairs.")
    parser.add_argument("pos", help="Part of speech (e.g., NOUN, VERB, ADJ).")
    parser.add_argument("t1", type=int, help="First time point (year).")
    parser.add_argument("t2", type=int, help="Second time point (year).")
    # parser.add_argument("--pos", default="NOUN", help="Part of speech (default: NOUN).")
    # parser.add_argument("--t1", type=int, default=1625, help="First time point (year) (default: 1850).")
    # parser.add_argument("--t2", type=int, default=1725, help="Second time point (year) (default: 1950).")
    parser.add_argument("--threshold", type=float, default=0.8, help="Similarity threshold for synonymy (default: 0.8).")
    parser.add_argument("--syn_pairs_csv", default=None, help="CSV file for synonym pairs. If not specified, defaults to {pos}_pairs_thr{threshold}_1.1_both.csv.")
    parser.add_argument("--non_syn_pairs_csv", default=None, help="CSV file for non-synonym pairs. If not specified, defaults to {pos}_pairs_thr0.0_0.85_both.csv.")

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

    results = analyze_semantic_change(syn, non_syn, lus_df, args.t1, args.t2, args.threshold)