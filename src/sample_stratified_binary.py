import argparse
import csv
import json
import os
import random
import re
import sys
from typing import Dict, List, Tuple

import pandas as pd


def iter_json_objects(path, chunk_size=1024 * 1024):
    """Yield JSON objects from a file that may contain concatenated JSON records."""
    decoder = json.JSONDecoder()
    buffer = ""
    with open(path, "r", encoding="utf-8", errors="replace") as inf:
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
                    # Keep unparsed suffix; it may be an incomplete object.
                    buffer = buffer[idx:]
                    break

                yield obj
                idx = next_idx

            if not chunk:
                break


def build_sentence_lookup(sense_ids, sense2usages_path, usage_history_path):
    """Map sense_id -> newest example sentence text by usage year."""
    needed = {sid for sid in sense_ids if sid}
    if not needed:
        return {}

    sense_to_usage_ids = {}
    with open(sense2usages_path, "r", encoding="utf-8", errors="replace") as inf:
        for line in inf:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec.get("sense_id")
            if sid in needed:
                sense_to_usage_ids[sid] = rec.get("usages") or []

    usage_to_senses = {}
    for sid in needed:
        for uid in sense_to_usage_ids.get(sid, []):
            usage_to_senses.setdefault(uid, []).append(sid)

    if not usage_to_senses:
        return {}

    usage_best = {}
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

    sense_to_sentence = {}
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
            sense_to_sentence[sid] = best_text

    return sense_to_sentence


def normalize_sentence_text(text):
    """Normalize usage sentence text for downstream annotation."""
    if not text:
        return ""
    text = str(text).replace("..", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_idx(idx: str) -> Tuple[str, str]:
    if not idx or "|||" not in idx:
        return "", ""
    left, right = idx.split("|||", 1)
    return left.strip(), right.strip()


def read_qwen_rows(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as inf:
        for line in inf:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def prepare_qwen_df(pos: str, model_dir: str) -> pd.DataFrame:
    file_name = f"{pos}_hf_labels_{os.path.basename(model_dir)}.jsonl"
    path = os.path.join("..", "data", "llm_output", model_dir, file_name)
    print(f"[{pos}] Reading Qwen labels: {path}", file=sys.stderr)
    rows = read_qwen_rows(path)
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if "idx" not in df.columns:
        raise RuntimeError(f"[{pos}] Missing idx in Qwen file: {path}")

    parsed = df["idx"].astype(str).map(parse_idx)
    df["sense_id1"] = parsed.map(lambda t: t[0])
    df["sense_id2"] = parsed.map(lambda t: t[1])

    if "word1" not in df.columns:
        df["word1"] = ""
    if "word2" not in df.columns:
        df["word2"] = ""
    if "definition1" not in df.columns:
        df["definition1"] = ""
    if "definition2" not in df.columns:
        df["definition2"] = ""
    if "sample_sentence1" not in df.columns:
        df["sample_sentence1"] = ""
    if "sample_sentence2" not in df.columns:
        df["sample_sentence2"] = ""

    return df


def prepare_low_cos_df(pos: str, low_cos_csv: str) -> pd.DataFrame:
    path = low_cos_csv.format(pos=pos)
    if not os.path.exists(path):
        raise RuntimeError(f"[{pos}] Low-cosine file not found: {path}")
    print(f"[{pos}] Reading low-cosine pairs: {path}", file=sys.stderr)
    df = pd.read_csv(path)
    if df.empty:
        return df

    required = ["sense_id1", "sense_id2", "lu1", "lu2", "def1", "def2", "cosine"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"[{pos}] Missing required columns in low-cosine file: {missing}")

    df["sense_id1"] = df["sense_id1"].astype(str)
    df["sense_id2"] = df["sense_id2"].astype(str)
    df["idx"] = df["sense_id1"] + "|||" + df["sense_id2"]

    if "sample_sentence1" not in df.columns:
        df["sample_sentence1"] = ""
    if "sample_sentence2" not in df.columns:
        df["sample_sentence2"] = ""

    df["word1"] = df["lu1"]
    df["word2"] = df["lu2"]
    df["definition1"] = df["def1"]
    df["definition2"] = df["def2"]

    return df


def attach_and_filter_sentences(df: pd.DataFrame, sentence_lookup: Dict[str, str], pos: str, tag: str) -> pd.DataFrame:
    if df.empty:
        return df

    def pick_sentence(sid: str, existing: str) -> str:
        looked_up = normalize_sentence_text(sentence_lookup.get(str(sid), ""))
        if looked_up:
            return looked_up
        return normalize_sentence_text(existing)

    df = df.copy()
    df["sample_sentence1"] = [
        pick_sentence(sid, existing)
        for sid, existing in zip(df["sense_id1"].astype(str), df["sample_sentence1"])
    ]
    df["sample_sentence2"] = [
        pick_sentence(sid, existing)
        for sid, existing in zip(df["sense_id2"].astype(str), df["sample_sentence2"])
    ]

    before = len(df)
    df = df[(df["sample_sentence1"] != "") & (df["sample_sentence2"] != "")].copy()
    removed = before - len(df)
    if removed > 0:
        print(f"[{pos}] {tag}: removed {removed} rows missing sample sentences", file=sys.stderr)

    return df


def filter_same_form_pairs(df: pd.DataFrame, pos: str, tag: str) -> pd.DataFrame:
    """Drop pairs where both items have the same written form."""
    if df.empty:
        return df

    work = df.copy()
    left = work["word1"].astype(str).str.strip().str.lower()
    right = work["word2"].astype(str).str.strip().str.lower()

    before = len(work)
    work = work[left != right].copy()
    removed = before - len(work)
    if removed > 0:
        print(f"[{pos}] {tag}: removed {removed} same-form pairs", file=sys.stderr)

    return work


def take_random_rows(df: pd.DataFrame, n: int, seed: int, pos: str, label: str) -> pd.DataFrame:
    if len(df) < n:
        raise RuntimeError(f"[{pos}] Not enough rows for {label}: need {n}, have {len(df)}")
    return df.sample(n=n, random_state=seed)


def stratified_low_cos_sample(df: pd.DataFrame, n: int, seed: int, pos: str) -> pd.DataFrame:
    if n == 0:
        return df.iloc[0:0].copy()

    if df.empty:
        raise RuntimeError(f"[{pos}] No low-cosine rows available after filtering")

    # 0.1 bins: [0.0,0.1), [0.1,0.2), ... [0.7,0.8), [0.8,0.85)
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85]
    labels = list(range(len(bins) - 1))

    work = df.copy()
    work["cos_bin"] = pd.cut(work["cosine"], bins=bins, labels=labels, right=False)
    work = work.dropna(subset=["cos_bin"]).copy()
    work["cos_bin"] = work["cos_bin"].astype(int)

    if work.empty:
        raise RuntimeError(f"[{pos}] No low-cosine rows fell into 0.1 bins")

    counts = work.groupby("cos_bin").size()
    non_empty_bins = list(counts.index)
    capacities = {b: int(counts[b]) for b in non_empty_bins}

    target = min(n, int(counts.sum()))
    if target < n:
        raise RuntimeError(f"[{pos}] Not enough low-cosine rows: need {n}, have {target}")

    allocs = {b: 0 for b in non_empty_bins}
    remaining = target
    active = set(non_empty_bins)

    while remaining > 0 and active:
        base, extra = divmod(remaining, len(active))
        shortfall = 0
        newly_capped = set()

        for i, b in enumerate(sorted(active)):
            give = base + (1 if i < extra else 0)
            space = capacities[b] - allocs[b]
            if give >= space:
                allocs[b] = capacities[b]
                shortfall += give - space
                newly_capped.add(b)
            else:
                allocs[b] += give

        active -= newly_capped
        remaining = shortfall
        if not newly_capped:
            break

    sampled_parts = []
    for b in non_empty_bins:
        k = allocs[b]
        if k == 0:
            continue
        part = work[work["cos_bin"] == b].sample(n=k, random_state=seed + b + 100)
        sampled_parts.append(part)

    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled.drop(columns=["cos_bin"])
    return sampled
def select_for_pos(
    pos: str,
    model_dir: str,
    n_sim: int,
    n_dis: int,
    n_low: int,
    seed: int,
    qwen_enabled: bool,
    low_cos_csv: str,
) -> pd.DataFrame:
    need_qwen = qwen_enabled and (n_sim > 0 or n_dis > 0)

    if not qwen_enabled and (n_sim > 0 or n_dis > 0):
        raise RuntimeError(
            f"[{pos}] Qwen is disabled but n-similar ({n_sim}) or n-dissimilar ({n_dis}) is > 0"
        )

    if need_qwen:
        qwen_df = prepare_qwen_df(pos, model_dir)
    else:
        qwen_df = pd.DataFrame()
        print(f"[{pos}] Skipping Qwen loading", file=sys.stderr)

    low_df = prepare_low_cos_df(pos, low_cos_csv)

    all_sense_ids = set(qwen_df.get("sense_id1", pd.Series(dtype=str)).astype(str))
    all_sense_ids.update(set(qwen_df.get("sense_id2", pd.Series(dtype=str)).astype(str)))
    all_sense_ids.update(set(low_df.get("sense_id1", pd.Series(dtype=str)).astype(str)))
    all_sense_ids.update(set(low_df.get("sense_id2", pd.Series(dtype=str)).astype(str)))

    print(f"[{pos}] Looking up sample sentences for {len(all_sense_ids)} senses...", file=sys.stderr)
    sentence_lookup = build_sentence_lookup(
        all_sense_ids,
        os.path.join("..", "data", "raw", "sense2usages.txt"),
        os.path.join("..", "data", "raw", "usage_history.txt"),
    )

    qwen_df = attach_and_filter_sentences(qwen_df, sentence_lookup, pos, "qwen")
    low_df = attach_and_filter_sentences(low_df, sentence_lookup, pos, "low-cos")

    # Exclude same-form pairs such as x|||x even if senses differ.
    qwen_df = filter_same_form_pairs(qwen_df, pos, "qwen")
    low_df = filter_same_form_pairs(low_df, pos, "low-cos")

    if need_qwen and qwen_df.empty:
        raise RuntimeError(f"[{pos}] No Qwen rows left after sentence filtering")

    picked_sim = pd.DataFrame()
    picked_dis = pd.DataFrame()
    used_idx = set()

    if need_qwen:
        qwen_df["predicted_label"] = qwen_df["predicted_label"].astype(str).str.upper().str.strip()

        sim_df = qwen_df[qwen_df["predicted_label"] == "SIMILAR"].copy()
        dis_df = qwen_df[qwen_df["predicted_label"] == "DISSIMILAR"].copy()

        if n_sim > 0:
            picked_sim = take_random_rows(sim_df, n_sim, seed + 1, pos, "Qwen SIMILAR")
            picked_sim["group"] = "qwen_similar"
            picked_sim["source"] = "qwen"
            used_idx.update(set(picked_sim["idx"].astype(str)))

        if n_dis > 0:
            picked_dis = take_random_rows(dis_df, n_dis, seed + 2, pos, "Qwen DISSIMILAR")
            picked_dis["group"] = "qwen_dissimilar"
            picked_dis["source"] = "qwen"
            used_idx.update(set(picked_dis["idx"].astype(str)))

    low_df = low_df[~low_df["idx"].astype(str).isin(used_idx)].copy()

    picked_low = stratified_low_cos_sample(low_df, n_low, seed + 3, pos)
    picked_low["group"] = "low_cosine"
    picked_low["source"] = "extracted_thr0.0_0.85"
    picked_low["predicted_label"] = ""

    combined = pd.concat([picked_sim, picked_dis, picked_low], ignore_index=True)
    combined = combined.sample(frac=1, random_state=seed + 4).reset_index(drop=True)

    # Keep a compact, consistent output schema.
    keep_cols = [
        "idx",
        "group",
        "source",
        "predicted_label",
        "sense_id1",
        "sense_id2",
        "word1",
        "word2",
        "definition1",
        "definition2",
        "sample_sentence1",
        "sample_sentence2",
        "cosine",
    ]

    for col in keep_cols:
        if col not in combined.columns:
            combined[col] = ""

    combined = combined[keep_cols]

    # Normalize NaNs for writing.
    for col in combined.columns:
        combined[col] = combined[col].apply(lambda x: "" if pd.isna(x) else x)

    return combined


def write_outputs(pos: str, out_dir: str, df: pd.DataFrame):
    os.makedirs(out_dir, exist_ok=True)
    pos_dir = os.path.join(out_dir, pos)
    os.makedirs(pos_dir, exist_ok=True)

    csv_path = os.path.join(pos_dir, f"{pos}_sample_binary.csv")
    jsonl_path = os.path.join(pos_dir, f"{pos}_sample_binary.jsonl")

    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)

    with open(jsonl_path, "w", encoding="utf-8") as outf:
        for _, row in df.iterrows():
            record = {
                "idx": row.get("idx", ""),
                "group": row.get("group", ""),
                "source": row.get("source", ""),
                "predicted_label": row.get("predicted_label", ""),
                "word1": row.get("word1", ""),
                "word2": row.get("word2", ""),
                "definition1": row.get("definition1", ""),
                "definition2": row.get("definition2", ""),
                "sample_sentence1": row.get("sample_sentence1", ""),
                "sample_sentence2": row.get("sample_sentence2", ""),
            }
            outf.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[{pos}] Wrote {len(df)} rows:", file=sys.stderr)
    print(f"[{pos}]   CSV:   {csv_path}", file=sys.stderr)
    print(f"[{pos}]   JSONL: {jsonl_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sample binary evaluation sets per POS: "
            "30 Qwen SIMILAR + 30 Qwen DISSIMILAR + 40 low-cosine stratified by 0.1 bins."
        )
    )
    parser.add_argument("pos", nargs="+", help="Parts of speech (e.g., ADJ NOUN VERB)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-similar", type=int, default=30, help="Qwen SIMILAR samples per POS")
    parser.add_argument("--n-dissimilar", type=int, default=30, help="Qwen DISSIMILAR samples per POS")
    parser.add_argument("--n-low-cos", type=int, default=40, help="Low-cosine stratified samples per POS")
    parser.add_argument(
        "--qwen",
        choices=["on", "off"],
        default="on",
        help="Enable or disable loading Qwen predictions (default: on)",
    )
    parser.add_argument(
        "--model-dir",
        default="Qwen_Qwen3_5-27B",
        help="Model subdirectory under ../data/llm_output",
    )
    parser.add_argument(
        "--low-cos-csv",
        required=True,
        help=(
            "Low-cosine input CSV path (exact path)."
            "You can use {pos} in the path when processing multiple POS values."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join("..", "data", "exp_eval"),
        help="Output root directory",
    )
    args = parser.parse_args()

    if args.n_similar < 0 or args.n_dissimilar < 0 or args.n_low_cos < 0:
        raise RuntimeError("--n-similar, --n-dissimilar, and --n-low-cos must be >= 0")

    random.seed(args.seed)

    for pos in args.pos:
        print(f"\n--- Processing {pos} ---", file=sys.stderr)
        selected = select_for_pos(
            pos=pos,
            model_dir=args.model_dir,
            n_sim=args.n_similar,
            n_dis=args.n_dissimilar,
            n_low=args.n_low_cos,
            seed=args.seed,
            qwen_enabled=(args.qwen == "on"),
            low_cos_csv=args.low_cos_csv,
        )
        expected = args.n_similar + args.n_dissimilar + args.n_low_cos
        if len(selected) != expected:
            raise RuntimeError(f"[{pos}] Expected {expected} rows, got {len(selected)}")
        write_outputs(pos, args.out_dir, selected)


if __name__ == "__main__":
    main()
