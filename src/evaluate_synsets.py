"""
Evaluate generated synsets against WordNet gold synsets.

For each POS (ADJ, NOUN, VERB):
    1. Load generated synsets from JSONL.
    2. Load WordNet synsets as gold.
    3. Filter generated synsets to lemmas active in a reference year.
    4. Keep singletons; matching thresholds handle singleton behavior.
    5. Compute the intersection of lemmas across gold and generated.
    6. For each generated synset, check if there exists a gold synset with >= *min_overlap* lemma overlap (with special-case rules for singleton vs non-singleton matches).

Usage:
  python evaluate_synsets.py --synsets-dir ../data/synsets --year 1900
"""

import argparse
import json
from pathlib import Path

import nltk
from nltk.corpus import wordnet as wn

nltk.download("wordnet", quiet=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

WN_POS_MAP = {"ADJ": wn.ADJ, "NOUN": wn.NOUN, "VERB": wn.VERB}


def load_generated_synsets(path: Path):
    """Return list of dicts from a JSONL synsets file."""
    synsets = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            synsets.append(json.loads(line))
    return synsets


def is_active(lu: dict, year: int) -> bool:
    """A LU is active at *year* if start <= year and (end is None or end >= year)."""
    start = lu.get("start")
    end = lu.get("end")
    if start is None or start > year:
        return False
    if end is not None and end < year:
        return False
    return True


def filter_synsets_by_year(synsets: list, year: int):
    """
    Keep only LUs active at *year* inside each synset.
    Return a list of sets-of-lemmas (one set per surviving synset).
    """
    filtered = []
    for syn in synsets:
        lemmas = {lu["lemma"].lower() for lu in syn["lus"] if is_active(lu, year)}
        if lemmas:
            filtered.append(lemmas)
    return filtered


def get_wordnet_synsets(pos_tag: str):
    """
    Return a list of sets-of-lemmas for every WordNet synset of the given POS.
    """
    wn_pos = WN_POS_MAP.get(pos_tag)
    if wn_pos is None:
        return []
    result = []
    for syn in wn.all_synsets(wn_pos):
        lemmas = {l.name().replace("_", " ").lower() for l in syn.lemmas()}
        result.append(lemmas)
    return result


def restrict_to_vocab(synset_list, vocab: set):
    """
    Keep only lemmas that belong to *vocab* in each synset.
    Drop synsets that become empty after restriction.
    """
    restricted = []
    for s in synset_list:
        trimmed = s & vocab
        if trimmed:
            restricted.append(trimmed)
    return restricted


# ── Main evaluation ──────────────────────────────────────────────────────────

def evaluate_pos(generated_path: Path, pos: str, year: int, min_overlap: int = 2):
    """Run the full evaluation pipeline for one POS."""

    # 1. Load generated synsets and filter by reference year
    raw_generated = load_generated_synsets(generated_path)
    generated = filter_synsets_by_year(raw_generated, year)
    print(f"\n{'='*60}")
    print(f"POS: {pos}  |  Reference year: {year}")
    print(f"{'='*60}")
    print(f"Generated synsets loaded:            {len(raw_generated)}")
    print(f"Generated synsets after year filter:  {len(generated)}")

    # 2. Load WordNet gold synsets
    gold = get_wordnet_synsets(pos)
    print(f"WordNet gold synsets loaded:          {len(gold)}")

    # 3. Compute lemma vocabularies
    gen_lemmas = set().union(*generated) if generated else set()
    gold_lemmas = set().union(*gold) if gold else set()
    shared_lemmas = gen_lemmas & gold_lemmas
    print(f"\nGenerated lemmas (unique):  {len(gen_lemmas)}")
    print(f"Gold lemmas (unique):       {len(gold_lemmas)}")
    print(f"Shared lemmas (intersection): {len(shared_lemmas)}")

    if not shared_lemmas:
        print("No shared lemmas – skipping evaluation.")
        return None

    # 4. Restrict both sets of synsets to the shared vocabulary and optionally drop singletons
    generated_r = restrict_to_vocab(generated, shared_lemmas)
    gold_r = restrict_to_vocab(gold, shared_lemmas)
    # keep singletons — matching thresholds handle singleton behavior
    print(f"\nGenerated synsets after restricting to shared vocab: {len(generated_r)}")
    print(f"Gold synsets after restricting to shared vocab:      {len(gold_r)}")

    # 5. For each generated synset, find gold synsets with >= min_overlap
    matched_gen = 0
    match_details = []
    # Separate counts for singleton vs non-singleton matches
    matched_gen_singleton = 0
    matched_gen_non_singleton = 0
    matched_gold_singleton_ids = set()
    matched_gold_non_singleton_ids = set()
    matched_gen_ids = set()

    for gi, g_syn in enumerate(generated_r):
        best_overlap = 0
        best_gold_idx = -1
        any_match = False
        for gj, gold_syn in enumerate(gold_r):
            overlap = len(g_syn & gold_syn)
            # Determine required overlap: allow 1 if both are singletons,
            # otherwise require at least max(min_overlap, 2).
            if len(g_syn) == 1 and len(gold_syn) == 1:
                required_overlap = 1
            else:
                required_overlap = max(min_overlap, 2)

            if overlap >= required_overlap:
                any_match = True
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_gold_idx = gj
                # Track matched gold ids separately by singleton status
                if len(gold_syn) == 1:
                    matched_gold_singleton_ids.add(gj)
                else:
                    matched_gold_non_singleton_ids.add(gj)
        if any_match:
            matched_gen += 1
            matched_gen_ids.add(gi)
            if len(g_syn) == 1:
                matched_gen_singleton += 1
            else:
                matched_gen_non_singleton += 1
            match_details.append({
                "generated_synset": sorted(g_syn),
                "best_gold_synset": sorted(gold_r[best_gold_idx]),
                "best_overlap": best_overlap,
            })

    # 6. Summary statistics
    # Counts per group
    n_gen_singleton = sum(1 for s in generated_r if len(s) == 1)
    n_gen_non_singleton = sum(1 for s in generated_r if len(s) > 1)
    n_gold_singleton = sum(1 for s in gold_r if len(s) == 1)
    n_gold_non_singleton = sum(1 for s in gold_r if len(s) > 1)

    # Singleton group metrics
    precision_single = matched_gen_singleton / n_gen_singleton if n_gen_singleton else 0.0
    recall_single = len(matched_gold_singleton_ids) / n_gold_singleton if n_gold_singleton else 0.0
    f1_single = (2 * precision_single * recall_single / (precision_single + recall_single)) if (precision_single + recall_single) > 0 else 0.0

    # Non-singleton group metrics
    precision_non = matched_gen_non_singleton / n_gen_non_singleton if n_gen_non_singleton else 0.0
    recall_non = len(matched_gold_non_singleton_ids) / n_gold_non_singleton if n_gold_non_singleton else 0.0
    f1_non = (2 * precision_non * recall_non / (precision_non + recall_non)) if (precision_non + recall_non) > 0 else 0.0

    # Macro-average across the two groups
    avg_precision = (precision_single + precision_non) / 2.0
    avg_recall = (recall_single + recall_non) / 2.0
    avg_f1 = (f1_single + f1_non) / 2.0

    print(f"\n── Overlap evaluation (min_overlap for non-singletons = {min_overlap}) ──")
    print(f"Singletons:        matched {matched_gen_singleton} / {n_gen_singleton}  (P={precision_single:.3f} R={recall_single:.3f} F1={f1_single:.3f})")
    print(f"Non-singletons:    matched {matched_gen_non_singleton} / {n_gen_non_singleton}  (P={precision_non:.3f} R={recall_non:.3f} F1={f1_non:.3f})")
    print(f"Macro-average:     P={avg_precision:.3f} R={avg_recall:.3f} F1={avg_f1:.3f}")

    # Show some example matches
    n_examples = min(10, len(match_details))
    if n_examples > 0:
        print(f"\n── Top {n_examples} example matches (by overlap size) ──")
        match_details.sort(key=lambda x: x["best_overlap"], reverse=True)
        for i, m in enumerate(match_details[:n_examples], 1):
            gen_str = ", ".join(m["generated_synset"])
            gold_str = ", ".join(m["best_gold_synset"])
            print(f"  {i}. overlap={m['best_overlap']}")
            print(f"     GEN:  {gen_str}")
            print(f"     GOLD: {gold_str}")

    # Print some unmatched generated and gold synsets
    all_gen_ids = set(range(len(generated_r)))
    unmatched_gen_ids = sorted(list(all_gen_ids - matched_gen_ids))
    all_gold_ids = set(range(len(gold_r)))
    matched_gold_ids = matched_gold_singleton_ids | matched_gold_non_singleton_ids
    unmatched_gold_ids = sorted(list(all_gold_ids - matched_gold_ids))

    n_un_examples = min(10, max(len(unmatched_gen_ids), len(unmatched_gold_ids)))
    if unmatched_gen_ids:
        print(f"\n── Unmatched generated synsets (showing up to {n_un_examples}) ──")
        for i, gi in enumerate(unmatched_gen_ids[:n_un_examples], 1):
            print(f"  {i}. {', '.join(sorted(generated_r[gi]))}")

    if unmatched_gold_ids:
        print(f"\n── Unmatched gold synsets (showing up to {n_un_examples}) ──")
        for i, gj in enumerate(unmatched_gold_ids[:n_un_examples], 1):
            print(f"  {i}. {', '.join(sorted(gold_r[gj]))}")

    return {
        "pos": pos,
        "year": year,
        "n_generated": len(generated_r),
        "n_gold": len(gold_r),
        "shared_lemmas": len(shared_lemmas),
        "matched_generated": matched_gen,
        "matched_generated_singleton": matched_gen_singleton,
        "matched_generated_non_singleton": matched_gen_non_singleton,
        "matched_gold": len(matched_gold_singleton_ids | matched_gold_non_singleton_ids),
        "matched_gold_singleton": len(matched_gold_singleton_ids),
        "matched_gold_non_singleton": len(matched_gold_non_singleton_ids),
        "precision_singleton": precision_single,
        "recall_singleton": recall_single,
        "f1_singleton": f1_single,
        "precision_non_singleton": precision_non,
        "recall_non_singleton": recall_non,
        "f1_non_singleton": f1_non,
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate generated synsets against WordNet gold standard."
    )
    parser.add_argument(
        "--synsets-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "synsets",
        help="Directory containing *_synsets_*.jsonl files (default: ../data/synsets)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2000,
        help="Reference year for filtering active lemmas (default: 2000)",
    )
    parser.add_argument(
        "--min-overlap",
        type=int,
        default=2,
        help="Minimum overlap between generated and gold synset to count as a match (default: 2)",
    )
    parser.add_argument(
        "--pos",
        nargs="*",
        default=["ADJ", "NOUN", "VERB"],
        choices=["ADJ", "NOUN", "VERB"],
        help="POS tags to evaluate (default: ADJ NOUN VERB)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = []

    for pos in args.pos:
        # Find the synset file for this POS (pattern: {POS}_synsets_*.jsonl)
        candidates = sorted(args.synsets_dir.glob(f"{pos}_synsets_*.jsonl"))
        if not candidates:
            print(f"WARNING: no synset file found for {pos} in {args.synsets_dir}")
            continue
        # Use all candidates
        for synset_path in candidates:
            filename = synset_path.name
            print(f"\nEvaluating {pos} synsets from file: {filename}")
            result = evaluate_pos(synset_path, pos, args.year, min_overlap=args.min_overlap)
            if result:
                # Extract threshold from filename (pattern: ADJ_synsets_thr0.8_1.1_louvain.jsonl)
                parts = filename.split("_")
                threshold = parts[2]
                result["threshold"] = threshold
                results.append(result)

    # Overall summary
    if results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"{'POS':<6} {'Thresh':>8} {'Gen':>5} {'Gold':>6} {'Shared':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
        print("-" * 60)
        for r in results:
            print(
                f"{r['pos']:<6} {r['threshold']:>8} {r['n_generated']:>5} {r['n_gold']:>6} "
                f"{r['shared_lemmas']:>7} {r['precision']:>7.3f} {r['recall']:>7.3f} {r['f1']:>7.3f}"
            )


if __name__ == "__main__":
    main()
