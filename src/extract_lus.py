import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import numpy as np
from collections import Counter


POS_MAP: Dict[str, str] = {
	# jq -r '.pos_tag' oed_testset/data/raw/OED.jsonl | sort | uniq
	"CC": "CCONJ",
	"IN": "ADP",
	"JJ": "ADJ",
	"NN": "NOUN",
	"PRP": "PRON",
	"RB": "ADV",
	"UH": "INTJ",
	"VB": "VERB",
	"PREFIX": "X",
	"SUFFIX": "X",
	"null": "X",
}


def parse_args() -> argparse.Namespace:
	base_dir = Path(__file__).resolve().parents[1]
	default_input = base_dir / "data" / "raw" / "OED.jsonl"
	default_output = base_dir / "data" / "parsed"
	parser = argparse.ArgumentParser(
		description=(
			"Read JSONL, map POS tags to coarse POS, filter short definitions, "
			"and write CSVs per coarse POS."
		)
	)
	parser.add_argument(
		"input",
		type=Path,
		nargs="?",
		default=default_input,
		help=(
			"Path to JSONL file with fields like lemma, pos_tag, definition, sense_id "
			f"(default: {default_input})."
		),
	)
	parser.add_argument(
		"output_dir",
		type=Path,
		nargs="?",
		default=default_output,
		help=f"Directory to write per-POS CSV files (default: {default_output}).",
	)
	parser.add_argument(
		"--min-percentile",
		type=float,
		default=20.0,
		help="Minimum percentile for word count in definition (default: 20.0).",
	)
	parser.add_argument(
		"--min-shared-lemmas",
		type=int,
		default=4,
		help="Minimum number of lemmas sharing a definition to exclude it (default: 4).",
	)
	return parser.parse_args()


def word_count(text: str) -> int:
	return len([w for w in text.strip().split() if w])


def read_jsonl(path: Path) -> Iterable[dict]:
	with path.open("r", encoding="utf-8") as handle:
		for line_num, line in enumerate(handle, start=1):
			line = line.strip()
			if not line:
				continue
			try:
				yield json.loads(line)
			except json.JSONDecodeError:
				raise ValueError(f"Invalid JSON on line {line_num} in {path}")


def collect_rows(
	records: Iterable[dict],
	min_percentile: float,
	min_shared_lemmas: int,
) -> Tuple[Dict[str, List[dict]], dict, int, int]:
	# First pass: collect all word counts
	all_word_counts = []
	temp_rows = []
	skipped_invalid_lemma = 0
	total_input = 0

	for rec in records:
		total_input += 1
		lemma = rec.get("lemma")
		definition = rec.get("definition")
		start = rec.get("start")
		end = rec.get("end")
		# Skip records with:
		# - lemma (missing, empty, non-alphabetic, non-lowercase, shorter than 2 chars, starting with "a \s"),
		# - definition (missing or empty after stripping),
		# - start year of 0 (which is invalid in OED)
		if not isinstance(lemma, str) or not lemma or not lemma.isalpha() or not lemma.islower() or len(lemma) < 2 or lemma.startswith("a ") or not isinstance(definition, str) or not definition.strip() or start == 0:
			skipped_invalid_lemma += 1
			continue

		if isinstance(definition, str):
			count = word_count(definition)
			all_word_counts.append(count)
			temp_rows.append((rec, count))

	# Identify definitions shared by x or more lemmas
	def_to_lemmas = defaultdict(set)
	for rec, count in temp_rows:
		def_to_lemmas[rec['definition']].add(rec['lemma'])
	bad_defs = {def_text for def_text, lemmas in def_to_lemmas.items() if len(lemmas) >= min_shared_lemmas}

	print(f"Definitions shared by {min_shared_lemmas}+ lemmas: {len(bad_defs)}")

	# Filter out shared definitions for threshold computation
	filtered_temp_rows = [(rec, count) for rec, count in temp_rows if rec['definition'] not in bad_defs]
	filtered_word_counts = [count for rec, count in filtered_temp_rows]

	# Compute threshold
	if filtered_word_counts:
		min_words = np.percentile(filtered_word_counts, min_percentile)
	else:
		min_words = 0

	print(f"Min words threshold (from {min_percentile}th percentile): {min_words}")

	# Second pass: filter
	rows_by_pos: Dict[str, List[dict]] = defaultdict(list)
	seen_keys = set()
	stats = defaultdict(lambda: {
		"total": 0,
		"kept": 0,
		"skipped_no_definition": 0,
		"skipped_short_definition": 0,
		"skipped_duplicate": 0,
		"skipped_shared_definition": 0,
		"skipped_invalid_lemma": 0,
		"def_word_counts_before": [],
		"def_word_counts_after": [],
	})

	for rec, count in temp_rows:
		pos_tag = rec.get("pos_tag")
		coarse_pos = POS_MAP.get(pos_tag, "X")
		stats[coarse_pos]["total"] += 1

		stats[coarse_pos]["def_word_counts_before"].append(count)
		if count <= min_words:
			stats[coarse_pos]["skipped_short_definition"] += 1
			continue

		if rec['definition'] in bad_defs:
			stats[coarse_pos]["skipped_shared_definition"] += 1
			continue

		key = (rec.get("lemma"), coarse_pos, rec.get("definition"))
		if key in seen_keys:
			stats[coarse_pos]["skipped_duplicate"] += 1
			continue
		seen_keys.add(key)

		rows_by_pos[coarse_pos].append(
			{
				"lemma": rec.get("lemma"),
				"pos_tag": pos_tag,
				"coarse_pos": coarse_pos,
				"definition": rec.get("definition"),
				"sense_id": rec.get("sense_id"),
				"start": rec.get("start"),
				"end": rec.get("end"),
				"n_usages": rec.get("n_usages"),
			}
		)
		stats[coarse_pos]["kept"] += 1
		stats[coarse_pos]["def_word_counts_after"].append(count)

	return rows_by_pos, stats, skipped_invalid_lemma, total_input


def write_csvs(rows_by_pos: Dict[str, List[dict]], output_dir: Path) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)
	fieldnames = [
		"lemma",
		"pos_tag",
		"coarse_pos",
		"definition",
		"sense_id",
		"start",
		"end",
		"n_usages",
	]

	for coarse_pos, rows in rows_by_pos.items():
		out_path = output_dir / f"{coarse_pos}.csv"
		with out_path.open("w", encoding="utf-8", newline="") as handle:
			writer = csv.DictWriter(handle, fieldnames=fieldnames)
			writer.writeheader()
			writer.writerows(sorted(rows, key=lambda r: r['lemma']))


def print_top_definitions(rows_by_pos: Dict[str, List[dict]], top_n: int = 5) -> None:
	all_defs = []
	for rows in rows_by_pos.values():
		for row in rows:
			all_defs.append(row['definition'])
	
	freq = Counter(all_defs)
	print(f"\nTop {top_n} most frequent unique definitions:")
	for i, (def_text, count) in enumerate(freq.most_common(top_n), 1):
		truncated = def_text[:100] + "..." if len(def_text) > 100 else def_text
		print(f"{i}. {count} occurrences: {truncated}")
	print()


def format_length_stats(lengths: List[int]) -> str:
	if not lengths:
		return "count=0"
	return (
		f"count={len(lengths)} min={min(lengths)} max={max(lengths)} "
		f"avg={sum(lengths) / len(lengths):.2f}"
	)


def main() -> None:
	global POS_MAP
	args = parse_args()
	records = read_jsonl(args.input)
	rows_by_pos, stats, skipped_invalid_lemma, total_input = collect_rows(records, args.min_percentile, args.min_shared_lemmas)
	write_csvs(rows_by_pos, args.output_dir)
	print_top_definitions(rows_by_pos)

	total_records = sum(s["total"] for s in stats.values())
	total_kept = sum(s["kept"] for s in stats.values())
	total_skipped_no_def = sum(s["skipped_no_definition"] for s in stats.values())
	total_skipped_short = sum(s["skipped_short_definition"] for s in stats.values())
	total_skipped_dup = sum(s["skipped_duplicate"] for s in stats.values())
	total_skipped_shared = sum(s["skipped_shared_definition"] for s in stats.values())

	print("Global stats")
	print(f"total_input={total_input}")
	print(f"lemmas_after_filtering={total_records}")
	print(f"kept={total_kept}")

	print("\nPer-POS stats:")
	for coarse_pos in sorted(stats.keys()):
		s = stats[coarse_pos]
		print(f"\n{coarse_pos}:")
		print(f"  total={s['total']}")
		print(f"  kept={s['kept']}")
		print(f"  skipped_no_definition={s['skipped_no_definition']}")
		print(f"  skipped_short_definition={s['skipped_short_definition']}")
		print(f"  skipped_duplicate={s['skipped_duplicate']}")
		print(f"  skipped_shared_definition={s['skipped_shared_definition']}")
		print(f"  definition_word_lengths_before: {format_length_stats(s['def_word_counts_before'])}")
		print(f"  definition_word_lengths_after: {format_length_stats(s['def_word_counts_after'])}")


if __name__ == "__main__":
	main()
