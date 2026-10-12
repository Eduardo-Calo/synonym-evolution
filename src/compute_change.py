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


def analyze_semantic_change(pairs_csv: str, lus_df: pl.DataFrame, t1: int, t2: int, threshold: float = 0.8):
    """
    Performs a full, structured analysis of semantic change between two time points using similarity pairs.

    Synonymous lemmas are those with at least one LU pair having cosine >= threshold at the time.

    This function categorizes each synonym pair from t1 into one of three primary outcomes:
    1.  Lost: One or both lemmas in the pair are no longer active at t2.
    2.  Split: Both lemmas are active at t2 but no longer share any high-similarity LU pairs.
    3.  Persistent: The lemmas remain synonyms at t2.

    Persistent pairs are then sub-analyzed for stability or parallel changes.
    """
    print(f"\n--- Starting Analysis between t1={t1} and t2={t2} with threshold {threshold} ---")

    # Load pairs data
    pairs_df = pl.read_csv(pairs_csv)
    pairs_df = remove_qwen_dissimilar_pairs(pairs_df, pairs_csv)

    # Create map from (lemma, definition) to sense_id
    sense_key_to_id = {}
    for row in lus_df.select(["lemma", "definition", "sense_id"]).iter_rows(named=True):
        key = (row["lemma"], row["definition"])
        sense_key_to_id[key] = row["sense_id"]

    # Create set of synonymous sense pairs at t1
    syn_senses_t1 = build_synonymous_sense_pairs(pairs_df, sense_key_to_id, threshold, t1)
    print(f"Found {len(syn_senses_t1)} synonymous sense pairs at {t1}.")
    syn_senses_t2 = build_synonymous_sense_pairs(pairs_df, sense_key_to_id, threshold, t2)
    print(f"Found {len(syn_senses_t2)} synonymous sense pairs at {t2}.")

    # # Create set of synonymous sense pairs
    # synonymous_sense_pairs = build_synonymous_sense_pairs(pairs_df, sense_key_to_id, threshold)
    # print(f"Found {len(synonymous_sense_pairs)} synonymous sense pairs overall.")

    # Step 1: Find synonym pairs for each time point.
    print("Finding synonym lemma pairs...")
    synonyms_t1 = find_synonym_lemmas_at_time(pairs_df, t1, threshold)
    synonyms_t2 = find_synonym_lemmas_at_time(pairs_df, t2, threshold)
    print(f"Found {len(synonyms_t1)} unique synonym lemma pairs at {t1}.")
    print(f"Found {len(synonyms_t2)} unique synonym lemma pairs at {t2}.")
    # Print some examples of synonym pairs for verification
    random.seed(42)
    if synonyms_t1:
        t1_list = list(synonyms_t1)
        examples_t1 = random.sample(t1_list, min(5, len(t1_list)))
        print(f"Examples of synonym pairs at {t1}: {examples_t1}")
    if synonyms_t2:
        t2_list = list(synonyms_t2)
        examples_t2 = random.sample(t2_list, min(5, len(t2_list)))
        print(f"Examples of synonym pairs at {t2}: {examples_t2}")

    # Step 2: Build lemma-to-synonyms maps for detailed change tracking.
    print("Building lemma-to-synonyms maps...")
    lemma_to_synonyms_t1 = build_lemma_to_synonyms(synonyms_t1)
    lemma_to_synonyms_t2 = build_lemma_to_synonyms(synonyms_t2)
    # number of unique lemmas in total
    unique_lemmas = set(lemma_to_synonyms_t1.keys()) | set(lemma_to_synonyms_t2.keys())
    print(f"Unique synonym lemmas in total: {len(unique_lemmas)}")
    # Print some examples of lemma-to-synonyms mappings for verification
    if lemma_to_synonyms_t1:
        lemmas_t1 = list(lemma_to_synonyms_t1.keys())
        example_lemma_t1 = random.choice(lemmas_t1)
        print(f"Example lemma at {t1}: '{example_lemma_t1}' has synonyms: {sorted(lemma_to_synonyms_t1[example_lemma_t1])}")
    if lemma_to_synonyms_t2:
        lemmas_t2 = list(lemma_to_synonyms_t2.keys())
        example_lemma_t2 = random.choice(lemmas_t2)
        print(f"Example lemma at {t2}: '{example_lemma_t2}' has synonyms: {sorted(lemma_to_synonyms_t2[example_lemma_t2])}")

    # Step 3: Initialize containers for each category of change.
    lost_pairs = set()
    split_pairs = set()

    # Categories for persistent pairs
    persistent_synonyms = set()
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

    # Step 5: Classify every synonym pair from t1.
    for w1, w2 in tqdm(synonyms_t1):
        # Check Case 1: Is one or both words lost by t2?
        if not active_senses_t2.get(w1, set()) or not active_senses_t2.get(w2, set()):
            lost_pairs.add((w1, w2))
            continue

        # At this point, both words exist at t2.
        # Check Case 2: Do they remain synonyms?
        if (w1, w2) in synonyms_t2:
            # Case 2a: The pair is a PERSISTENT SYNONYM.
            persistent_synonyms.add((w1, w2))

            # Get the category sets for detailed analysis.
            senses_w1_t1 = active_senses_t1.get(w1, set())
            senses_w2_t1 = active_senses_t1.get(w2, set())
            senses_w1_t2 = active_senses_t2.get(w1, set())
            senses_w2_t2 = active_senses_t2.get(w2, set())

            # Check if the senses for BOTH words are identical between t1 and t2.
            if senses_w1_t1 == senses_w1_t2 and senses_w2_t1 == senses_w2_t2:
                persistent_unchanged.add((w1, w2))
            else:
                # There has been some change in meanings for at least one word.
                persistent_with_some_change.add((w1, w2))

                # Calculate additions and losses
                new_senses_w1 = senses_w1_t2 - senses_w1_t1
                new_senses_w2 = senses_w2_t2 - senses_w2_t1
                lost_senses_w1 = senses_w1_t1 - senses_w1_t2
                lost_senses_w2 = senses_w2_t1 - senses_w2_t2

                # Create boolean flags to make the logic easier to read
                has_additions = bool(new_senses_w1 or new_senses_w2)
                has_losses = bool(lost_senses_w1 or lost_senses_w2)

                # Check for the high-priority "shared change" conditions
                # These are True if both words share AT LEAST one synonymous addition/loss
                has_same_addition = any(tuple(sorted([s1, s2])) in syn_senses_t2 for s1 in new_senses_w1 for s2 in new_senses_w2)
                has_same_loss = any(tuple(sorted([s1, s2])) in syn_senses_t1 for s1 in lost_senses_w1 for s2 in lost_senses_w2)

                # --- Mutually Exclusive Categorization with Priority ---
                # The order of these checks is critical.

                # Priority 1: Pair has BOTH identical additions AND identical losses.
                if has_same_addition and has_same_loss:
                    persistent_with_same_addition_and_loss.add((w1, w2))

                # Priority 2: Pair has identical additions.
                elif has_same_addition:
                    persistent_with_same_addition.add((w1, w2))

                # Priority 3: Pair has identical losses.
                elif has_same_loss:
                    persistent_with_same_loss.add((w1, w2))

                # Priority 4: Pair has BOTH additions and losses (of any kind).
                elif has_additions and has_losses:
                    persistent_with_both_addition_and_loss.add((w1, w2))

                # Priority 5: Pair has ONLY additions, but they are not identical.
                elif has_additions: # This implicitly means "and not has_losses" because of the elif chain
                    persistent_with_some_addition.add((w1, w2))

                # Priority 6: Pair has ONLY losses, but they are not identical.
                elif has_losses: # This implicitly means "and not has_additions"
                    persistent_with_some_loss.add((w1, w2))

        else:
            # Case 2b: Both words exist but are NO LONGER synonyms. This is a SPLIT pair.
            split_pairs.add((w1, w2))

    # Step 6: Print the final analysis results.
    print("\n--- Analysis Results ---")

    # --- DIFFERENTIATION ---
    print("\nDIFFERENTIATION (Synonyms at t1 are no longer synonyms at t2)")
    print(f"Found {len(lost_pairs)} pairs where at least one word was lost by {t2}.")
    print(f"Found {len(split_pairs)} pairs that SPLIT (both words exist at {t2} but no longer share a meaning).")

    # --- PERSISTENCE ---
    print(f"\nPERSISTENCE (Synonyms at t1 are still synonyms at t2): {len(persistent_synonyms)} total pairs")

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
    assert len(synonyms_t1) == total_accounted_for, \
        "Verification failed: The sum of outcomes does not match the initial number of synonyms."
    print("Verification successful: All synonym pairs from t1 have been accounted for.")

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
    print(f"PERSISTENCE\t{len(persistent_synonyms)}")
    print(f"Total\t{len(synonyms_t1)}")

    return {
        "lost_pairs": lost_pairs,
        "split_pairs": split_pairs,
        "persistent_synonyms": persistent_synonyms,
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


def print_sample_pairs(results, lus_df, t1, t2, active_senses_t1, active_senses_t2, sample_size=None, output_csv="sample_pairs.csv"):
    """
    Saves a sample of lemma pairs for each change category to a CSV, showing their definition compositions before and after.
    """
    random.seed(42)
    sense_to_def = {row["sense_id"]: row["definition"] for row in lus_df.select(["sense_id", "definition"]).iter_rows(named=True)}

    categories = [
        ("lost_pairs", "Lost"),
        ("split_pairs", "Split"),
        # ("persistent_unchanged", "Persistent Unchanged"),
        # ("persistent_with_some_addition", "Persistent with Some Addition"),
        ("persistent_with_same_addition", "Persistent with Same Addition"),
        ("persistent_with_same_addition_and_loss", "Persistent with Same Addition and Loss"),
        # ("persistent_with_some_loss", "Persistent with Some Loss"),
        ("persistent_with_same_loss", "Persistent with Same Loss"),
        # ("persistent_with_both_addition_and_loss", "Persistent with Both Addition and Loss"),
    ]

    all_data = []
    for key, name in categories:
        pairs = results.get(key)
        if not pairs:
            continue

        pairs_list = list(pairs)
        if sample_size is not None and len(pairs_list) > sample_size:
            sampled_pairs = random.sample(pairs_list, sample_size)
        else:
            sampled_pairs = pairs_list
        
        if not sampled_pairs:
            continue

        # Convert to DataFrame for efficient processing
        df = pl.DataFrame(sampled_pairs, schema=["lemma1", "lemma2"], orient="row")

        # Get senses and calculate changes in one step
        df = df.with_columns(
            pl.col("lemma1").map_elements(lambda w: active_senses_t2.get(w, set()) - active_senses_t1.get(w, set()), return_dtype=pl.Object).alias("added_w1"),
            pl.col("lemma1").map_elements(lambda w: active_senses_t1.get(w, set()) - active_senses_t2.get(w, set()), return_dtype=pl.Object).alias("lost_w1"),
            pl.col("lemma2").map_elements(lambda w: active_senses_t2.get(w, set()) - active_senses_t1.get(w, set()), return_dtype=pl.Object).alias("added_w2"),
            pl.col("lemma2").map_elements(lambda w: active_senses_t1.get(w, set()) - active_senses_t2.get(w, set()), return_dtype=pl.Object).alias("lost_w2"),
        )

        # Get definitions
        df = df.with_columns(
            pl.col("added_w1").map_elements(lambda s: " ||| ".join(sense_to_def.get(i, "Unknown") for i in sorted(s)), return_dtype=pl.Utf8).alias("lemma1_added_definitions"),
            pl.col("lost_w1").map_elements(lambda s: " ||| ".join(sense_to_def.get(i, "Unknown") for i in sorted(s)), return_dtype=pl.Utf8).alias("lemma1_lost_definitions"),
            pl.col("added_w2").map_elements(lambda s: " ||| ".join(sense_to_def.get(i, "Unknown") for i in sorted(s)), return_dtype=pl.Utf8).alias("lemma2_added_definitions"),
            pl.col("lost_w2").map_elements(lambda s: " ||| ".join(sense_to_def.get(i, "Unknown") for i in sorted(s)), return_dtype=pl.Utf8).alias("lemma2_lost_definitions"),
        )
        
        # Add category and counts
        df = df.with_columns(
            pl.lit(name).alias("category"),
            pl.col("lemma1").map_elements(lambda w: len(active_senses_t1.get(w, set())), return_dtype=pl.Int64).alias("lemma1_t1_senses_count"),
            pl.col("lemma2").map_elements(lambda w: len(active_senses_t1.get(w, set())), return_dtype=pl.Int64).alias("lemma2_t1_senses_count"),
            pl.col("lemma1").map_elements(lambda w: len(active_senses_t2.get(w, set())), return_dtype=pl.Int64).alias("lemma1_t2_senses_count"),
            pl.col("lemma2").map_elements(lambda w: len(active_senses_t2.get(w, set())), return_dtype=pl.Int64).alias("lemma2_t2_senses_count"),
        )

        all_data.append(df.select([
            "category", "lemma1", "lemma2", 
            "lemma1_t1_senses_count", "lemma2_t1_senses_count",
            "lemma1_t2_senses_count", "lemma2_t2_senses_count",
            "lemma1_added_definitions", "lemma1_lost_definitions",
            "lemma2_added_definitions", "lemma2_lost_definitions"
        ]))

    if not all_data:
        print("No sample pairs to save.")
        return

    final_df = pl.concat(all_data)
    final_df.write_csv(output_csv)
    print(f"Saved {len(final_df)} sample pairs to {output_csv}")


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze semantic change between two time points using similarity pairs.")
    parser.add_argument("pos", help="Part of speech (e.g., NOUN, VERB, ADJ).")
    parser.add_argument("t1", type=int, help="First time point (year).")
    parser.add_argument("t2", type=int, help="Second time point (year).")
    # parser.add_argument("--pos", default="ADJ", help="Part of speech (default: NOUN).")
    # parser.add_argument("--t1", type=int, default=1500, help="First time point (year) (default: 1850).")
    # parser.add_argument("--t2", type=int, default=2025, help="Second time point (year) (default: 1950).")
    parser.add_argument("--threshold", type=float, default=0.8, help="Similarity threshold for synonymy (default: 0.8).")
    parser.add_argument("--output_csv", default=None, help="Output CSV file for sample pairs. If not specified, defaults to sample_pairs_{pos}_thr{threshold}_{t1}_{t2}.csv.")

    args = parser.parse_args()

    # Construct file paths
    pairs_csv = f"../data/extracted/{args.pos}_pairs_thr{args.threshold}_1.1_both.csv"
    lus_df_path = f"../data/parsed/{args.pos}_wn.csv"

    if args.output_csv is None:
        args.output_csv = f"../data/sample_pairs/sample_pairs_{args.pos}_thr{args.threshold}_{args.t1}_{args.t2}.csv"

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

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

    results = analyze_semantic_change(pairs_csv, lus_df, args.t1, args.t2, args.threshold)
    active_senses_t1 = results.pop("active_senses_t1")
    active_senses_t2 = results.pop("active_senses_t2")
    print_sample_pairs(results, lus_df, args.t1, args.t2, active_senses_t1, active_senses_t2, output_csv=args.output_csv)