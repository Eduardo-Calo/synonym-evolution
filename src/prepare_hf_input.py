#!/usr/bin/env python3
"""Prepare hf_annotate_similarity input JSONL from extracted POS pair CSV files.

Input files:
- ../data/extracted/{POS}_pairs_thr{thr_start}_{thr_end}_both.csv

Output files:
- ../data/llm_eval/{POS}_thr{thr_start}_{thr_end}.jsonl
- If --chunk-size is set: ../data/llm_eval/{POS}_thr{thr_start}_{thr_end}/{POS}_thr{thr_start}_{thr_end}_idx{start}-{end}.jsonl

If no sample sentence can be found for a sense_id via sense2usages/usage_history,
the corresponding sample_sentence field is left blank.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set


VALID_POS = ("ADJ", "NOUN", "VERB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build hf_annotate JSONL files from extracted thr0.7 CSVs."
    )
    parser.add_argument(
        "--pos",
        nargs="+",
        default=list(VALID_POS),
        help="POS tags to process (default: ADJ NOUN VERB)",
    )
    parser.add_argument(
        "--input-root",
        default="../data/extracted",
        help="Directory containing {POS}_pairs_thr{thr_start}_{thr_end}_both.csv",
    )
    parser.add_argument(
        "--thr-start",
        default=None,
        required=True,
        help="Start cosine threshold tag used in filenames (default: 0.85)",
    )
    parser.add_argument(
        "--thr-end",
        default=None,
        required=True,
        help="End cosine threshold tag used in filenames (default: 1.1)",
    )
    parser.add_argument(
        "--thr",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output-root",
        default="../data/llm_eval",
        help="Directory where output JSONL files will be written",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help=(
            "Optional max rows per output file. "
            "When set, emits multiple files with _idx{start}-{end} suffixes."
        ),
    )
    parser.add_argument(
        "--sense2usages",
        default="../data/raw/sense2usages.txt",
        help="Path to line-delimited JSON mapping sense_id -> usages",
    )
    parser.add_argument(
        "--usage-history",
        default="../data/raw/usage_history.txt",
        help="Path to usage history JSON stream",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of rows per POS",
    )
    return parser.parse_args()


def iter_json_objects(path: Path, chunk_size: int = 1024 * 1024) -> Iterable[dict]:
    """Yield JSON objects from a file that may contain concatenated JSON records."""
    decoder = json.JSONDecoder()
    buffer = ""

    with path.open("r", encoding="utf-8", errors="replace") as inf:
        while True:
            chunk = inf.read(chunk_size)
            if not chunk and not buffer:
                break
            buffer += chunk

            idx = 0
            while True:
                while idx < len(buffer) and buffer[idx].isspace():
                    idx += 1
                if idx >= len(buffer):
                    buffer = ""
                    break

                try:
                    obj, next_idx = decoder.raw_decode(buffer, idx)
                except json.JSONDecodeError:
                    buffer = buffer[idx:]
                    break

                yield obj
                idx = next_idx

            if not chunk:
                break


def normalize_sentence_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).replace("..", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_pairs(csv_path: Path, limit: int | None = None) -> List[dict]:
    rows: List[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as inf:
        reader = csv.DictReader(inf)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            rows.append(row)
    return rows


def build_sentence_lookup(
    sense_ids: Set[str], sense2usages_path: Path, usage_history_path: Path
) -> Dict[str, str]:
    needed = {sid for sid in sense_ids if sid}
    if not needed:
        return {}

    sense_to_usage_ids: Dict[str, List[str]] = {}
    with sense2usages_path.open("r", encoding="utf-8", errors="replace") as inf:
        for line in inf:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = str(rec.get("sense_id", ""))
            if sid in needed:
                sense_to_usage_ids[sid] = rec.get("usages") or []

    usage_to_senses: Dict[str, List[str]] = {}
    for sid in needed:
        for uid in sense_to_usage_ids.get(sid, []):
            usage_to_senses.setdefault(uid, []).append(sid)

    if not usage_to_senses:
        return {}

    usage_best: Dict[str, tuple[int, str]] = {}
    for rec in iter_json_objects(usage_history_path):
        uid = rec.get("usage")
        if uid not in usage_to_senses:
            continue

        text = rec.get("text") or ""
        if not text:
            continue

        year = rec.get("year")
        try:
            year_val = int(year)
        except Exception:
            year_val = -10**9

        prev = usage_best.get(uid)
        if prev is None or year_val > prev[0]:
            usage_best[uid] = (year_val, text)

    sense_to_sentence: Dict[str, str] = {}
    for sid in needed:
        best_year = -10**9
        best_text = ""
        for uid in sense_to_usage_ids.get(sid, []):
            if uid not in usage_best:
                continue
            year_val, text = usage_best[uid]
            if year_val > best_year:
                best_year = year_val
                best_text = text
        if best_text:
            sense_to_sentence[sid] = normalize_sentence_text(best_text)

    return sense_to_sentence


def convert_rows(rows: List[dict], sentence_lookup: Dict[str, str]) -> List[dict]:
    out: List[dict] = []
    for row in rows:
        s1 = str(row.get("sense_id1", "") or "")
        s2 = str(row.get("sense_id2", "") or "")

        out.append(
            {
                "idx": f"{s1}|||{s2}",
                "word1": row.get("lu1", "") or "",
                "word2": row.get("lu2", "") or "",
                "definition1": row.get("def1", "") or "",
                "definition2": row.get("def2", "") or "",
                # Leave blank if no sentence is found.
                "sample_sentence1": sentence_lookup.get(s1, ""),
                "sample_sentence2": sentence_lookup.get(s2, ""),
            }
        )
    return out


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as outf:
        for row in rows:
            outf.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl_chunks(base_path: Path, rows: List[dict], chunk_size: int) -> List[Path]:
    chunk_dir = base_path.parent / base_path.stem
    chunk_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    if not rows:
        empty_path = chunk_dir / f"{base_path.stem}_idx0-0{base_path.suffix}"
        write_jsonl(empty_path, rows)
        return [empty_path]

    for start in range(0, len(rows), chunk_size):
        end = min(start + chunk_size, len(rows)) - 1
        chunk_path = chunk_dir / f"{base_path.stem}_idx{start}-{end}{base_path.suffix}"
        write_jsonl(chunk_path, rows[start : end + 1])
        written.append(chunk_path)

    return written


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    sense2usages_path = Path(args.sense2usages)
    usage_history_path = Path(args.usage_history)

    pos_list = [p.upper() for p in args.pos]
    invalid = [p for p in pos_list if p not in VALID_POS]
    if invalid:
        raise ValueError(f"Unsupported POS values: {invalid}. Allowed: {VALID_POS}")

    thr_start_raw = args.thr if args.thr is not None else args.thr_start
    try:
        thr_start_str = str(float(thr_start_raw))
    except ValueError as exc:
        raise ValueError(f"Invalid --thr-start value: {thr_start_raw!r}") from exc

    try:
        thr_end_str = str(float(args.thr_end))
    except ValueError as exc:
        raise ValueError(f"Invalid --thr-end value: {args.thr_end!r}") from exc

    if args.chunk_size is not None and args.chunk_size <= 0:
        raise ValueError(f"Invalid --chunk-size value: {args.chunk_size!r}. Must be > 0")

    for pos in pos_list:
        csv_path = input_root / f"{pos}_pairs_thr{thr_start_str}_{thr_end_str}_both.csv"
        if not csv_path.exists():
            available = sorted(input_root.glob(f"{pos}_pairs_thr*_*_both.csv"))
            avail_txt = ", ".join(p.name for p in available) if available else "<none>"
            raise FileNotFoundError(
                f"Missing input CSV: {csv_path}. Available for {pos}: {avail_txt}"
            )

        rows = load_pairs(csv_path, limit=args.limit)
        sense_ids = set()
        for row in rows:
            sense_ids.add(str(row.get("sense_id1", "") or ""))
            sense_ids.add(str(row.get("sense_id2", "") or ""))

        sentence_lookup = build_sentence_lookup(
            sense_ids=sense_ids,
            sense2usages_path=sense2usages_path,
            usage_history_path=usage_history_path,
        )

        output_rows = convert_rows(rows, sentence_lookup)
        base_out_path = output_root / f"{pos}_thr{thr_start_str}_{thr_end_str}.jsonl"

        if args.chunk_size is None:
            write_jsonl(base_out_path, output_rows)
            print(f"[{pos}] wrote {len(output_rows)} rows -> {base_out_path}")
        else:
            chunk_paths = write_jsonl_chunks(base_out_path, output_rows, args.chunk_size)
            chunk_dir = output_root / base_out_path.stem
            print(
                f"[{pos}] wrote {len(output_rows)} rows -> {len(chunk_paths)} chunk files in {chunk_dir}"
            )
            for path in chunk_paths:
                print(f"  - {path.name}")


if __name__ == "__main__":
    main()
