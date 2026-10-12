import csv
import json
import os
import random
import argparse
import sys
import re
import pandas as pd


def iter_json_objects(path, chunk_size=1024 * 1024):
    """Yield JSON objects from a file that may contain concatenated JSON records."""
    decoder = json.JSONDecoder()
    buffer = ""
    with open(path, 'r', encoding='utf-8', errors='replace') as inf:
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
    """Map sense_id -> newest example sentence text by usage `year`."""
    needed = {sid for sid in sense_ids if sid}
    if not needed:
        return {}

    sense_to_usage_ids = {}
    with open(sense2usages_path, 'r', encoding='utf-8', errors='replace') as inf:
        for line in inf:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec.get('sense_id')
            if sid in needed:
                sense_to_usage_ids[sid] = rec.get('usages') or []

    usage_to_senses = {}
    for sid in needed:
        usage_ids = sense_to_usage_ids.get(sid, [])
        for uid in usage_ids:
            usage_to_senses.setdefault(uid, []).append(sid)

    sense_to_sentence = {}
    if not usage_to_senses:
        return sense_to_sentence

    # Keep newest (highest year) sentence per usage ID.
    usage_best = {}
    for rec in iter_json_objects(usage_history_path):
        uid = rec.get('usage')
        if uid not in usage_to_senses:
            continue
        text = rec.get('text') or ''
        if not text:
            continue

        year = rec.get('year')
        try:
            year_val = int(year)
        except Exception:
            year_val = -10**9

        prev = usage_best.get(uid)
        if prev is None or year_val > prev[0]:
            usage_best[uid] = (year_val, text)

    # For each sense, choose the sentence from the newest linked usage.
    for sid in needed:
        best_year = -10**9
        best_text = ''
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
    """Normalize usage sentence text for downstream annotation.

    In this data, `..` indicates spacing elision and should be treated as a space.
    """
    if not text:
        return ''
    text = str(text).replace('..', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def sample_pos(pos, num_pairs, cosine_bins, century_bins):
    """Run stratified sampling for a single POS."""
    file_path = f"../data/extracted/{pos}_pairs_thr0.7_1.1_both.csv"

    print(f"[{pos}] Reading {file_path}...", file=sys.stderr)
    df = pd.read_csv(file_path)
    fieldnames = list(df.columns)

    # Filter: different lemma and different definition
    mask = (df['lu1'] != df['lu2']) & (df['def1'] != df['def2'])
    df = df[mask].copy()

    # Remove duplicate lemma pairs (keep first occurrence)
    before = len(df)
    df = df.drop_duplicates(subset=['lu1', 'lu2'], keep='first')
    removed = before - len(df)
    if removed > 0:
        print(f"[{pos}] Removed {removed} duplicate lemma pairs (kept first)", file=sys.stderr)

    if len(df) == 0:
        print(f"[{pos}] No lines match the filters.", file=sys.stderr)
        return [], fieldnames

    # Add one sample sentence per sense and filter rows before stratified sampling.
    sense_ids = set(df['sense_id1'].dropna().astype(str)).union(set(df['sense_id2'].dropna().astype(str)))
    sense2usages_path = '../data/raw/sense2usages.txt'
    usage_history_path = '../data/raw/usage_history.txt'
    print(f"[{pos}] Looking up sample sentences for {len(sense_ids)} senses...", file=sys.stderr)
    sentence_lookup = build_sentence_lookup(sense_ids, sense2usages_path, usage_history_path)

    df['sample_sentence1'] = df['sense_id1'].astype(str).map(lambda sid: normalize_sentence_text(sentence_lookup.get(sid, '')))
    df['sample_sentence2'] = df['sense_id2'].astype(str).map(lambda sid: normalize_sentence_text(sentence_lookup.get(sid, '')))

    before_sentence_filter = len(df)
    df = df[(df['sample_sentence1'] != '') & (df['sample_sentence2'] != '')].copy()
    removed_missing = before_sentence_filter - len(df)
    if removed_missing > 0:
        print(
            f"[{pos}] Removed {removed_missing} pairs missing sample_sentence1/sample_sentence2",
            file=sys.stderr
        )

    if len(df) == 0:
        print(f"[{pos}] No pairs left after sentence filtering.", file=sys.stderr)
        return [], fieldnames

    # Add explicit word fields for downstream annotation/export.
    df['word1'] = df['lu1']
    df['word2'] = df['lu2']

    if 'sample_sentence1' not in fieldnames:
        fieldnames.append('sample_sentence1')
    if 'sample_sentence2' not in fieldnames:
        fieldnames.append('sample_sentence2')
    if 'word1' not in fieldnames:
        fieldnames.append('word1')
    if 'word2' not in fieldnames:
        fieldnames.append('word2')

    # Bin by cosine, century (start1), and whether they share a synset
    cos_labels = list(range(len(cosine_bins) - 1))
    cent_labels = list(range(len(century_bins) - 1))
    df['cos_bin'] = pd.cut(df['cosine'], bins=cosine_bins, right=False, labels=cos_labels)
    df['cent_bin'] = pd.cut(df['start1'].astype(float), bins=century_bins, right=False, labels=cent_labels)

    # Ensure we have a binary indicator for shared synset membership. Prefer explicit
    # `same_synset` column if present; otherwise fall back to `shared_synsets`.
    if 'same_synset' in df.columns:
        try:
            df['syn_bin'] = df['same_synset'].astype(int)
        except Exception:
            df['syn_bin'] = df['same_synset'].apply(lambda x: 1 if str(x) in ('1', 'True', 'true') else 0)
    elif 'shared_synsets' in df.columns:
        # treat non-empty shared_synsets as membership
        df['syn_bin'] = df['shared_synsets'].apply(lambda s: 0 if (pd.isna(s) or str(s).strip() in ('', '[]')) else 1)
    else:
        # If neither column exists, assume no shared synsets
        df['syn_bin'] = 0

    # Drop rows that don't fall into any bin
    df = df.dropna(subset=['cos_bin', 'cent_bin'])
    df['cos_bin'] = df['cos_bin'].astype(int)
    df['cent_bin'] = df['cent_bin'].astype(int)
    df['syn_bin'] = df['syn_bin'].astype(int)

    # Count per stratum
    stratum_counts = df.groupby(['cos_bin', 'cent_bin', 'syn_bin']).size()
    non_empty = list(stratum_counts.index)
    m = len(non_empty)
    total_available = int(stratum_counts.sum())
    print(f"[{pos}] Non-empty strata: {m}, total eligible pairs: {total_available}", file=sys.stderr)

    # Distribute num_pairs across strata, redistributing shortfall iteratively
    target = min(num_pairs, total_available)
    capacities = {key: int(stratum_counts[key]) for key in non_empty}
    allocs = {key: 0 for key in non_empty}
    remaining = target

    # Iterative redistribution: spread evenly, cap, repeat with uncapped strata
    active = set(non_empty)
    while remaining > 0 and active:
        base, extra = divmod(remaining, len(active))
        shortfall = 0
        sorted_active = sorted(active)
        newly_capped = set()
        for idx, key in enumerate(sorted_active):
            give = base + (1 if idx < extra else 0)
            space = capacities[key] - allocs[key]
            if give >= space:
                allocs[key] = capacities[key]
                shortfall += give - space
                newly_capped.add(key)
            else:
                allocs[key] += give
        active -= newly_capped
        remaining = shortfall
        if not newly_capped:
            break  # nothing was capped, we're done

    actual_total = sum(allocs.values())
    print(f"[{pos}] Requested {num_pairs}, allocating {actual_total}", file=sys.stderr)

    # Sample from each stratum
    samples = []
    for key in non_empty:
        n_sample = allocs[key]
        if n_sample == 0:
            continue
        # key is (cos_bin, cent_bin, syn_bin)
        stratum_df = df[(df['cos_bin'] == key[0]) & (df['cent_bin'] == key[1]) & (df['syn_bin'] == key[2])]
        sampled = stratum_df.sample(n=n_sample)
        samples.append(sampled)

    if not samples:
        return [], fieldnames

    result = pd.concat(samples, ignore_index=True)
    # Drop helper columns before returning
    result = result.drop(columns=['cos_bin', 'cent_bin', 'syn_bin'])
    # Shuffle
    result = result.sample(frac=1).reset_index(drop=True)
    # Convert to list of dicts for downstream compatibility
    rows = result.to_dict(orient='records')
    # Convert NaN to empty strings for CSV/JSONL writing
    for row in rows:
        for k, v in row.items():
            if pd.isna(v):
                row[k] = ''
            elif isinstance(v, float) and v == int(v):
                row[k] = int(v)
    return rows, fieldnames


def main():
    parser = argparse.ArgumentParser(description='Randomly sample n lines from CSV file with filters and stratified by cosine.')
    parser.add_argument('pos', nargs='+', help='One or more parts of speech (e.g., ADJ NOUN VERB)')
    parser.add_argument('-n', '--num-pairs', type=int, required=True, help='Exact number of pairs to sample per POS')
    parser.add_argument('--batch-size', '-b', type=int, default=30, help='Number of pairs per output file (default: 30)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    cosine_bins = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.1]
    century_bins = [1500, 1600, 1700, 1800, 1900, 2000, 2026]

    out_dir = '../data/hum_eval'
    os.makedirs(out_dir, exist_ok=True)

    for pos in args.pos:
        print(f"\n--- Processing {pos} ---", file=sys.stderr)
        samples, fieldnames = sample_pos(pos, args.num_pairs, cosine_bins, century_bins)
        if not samples:
            continue

        # Split into batches
        batch_size = args.batch_size
        batches = [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]
        print(f"[{pos}] Splitting {len(samples)} pairs into {len(batches)} files of up to {batch_size} each", file=sys.stderr)

        pos_dir = os.path.join(out_dir, pos)
        os.makedirs(pos_dir, exist_ok=True)

        for batch_idx, batch in enumerate(batches, start=1):
            # Save CSV
            csv_path = os.path.join(pos_dir, f"{pos}_sample_{batch_idx:03d}.csv")
            with open(csv_path, 'w', newline='') as outf:
                writer = csv.DictWriter(outf, fieldnames=fieldnames)
                writer.writeheader()
                for row in batch:
                    writer.writerow(row)

            # Save JSONL
            jsonl_path = os.path.join(pos_dir, f"{pos}_sample_{batch_idx:03d}.jsonl")
            with open(jsonl_path, 'w', encoding='utf-8') as outf:
                for row in batch:
                    record = {
                        "idx": f"{row.get('sense_id1', '')}|||{row.get('sense_id2', '')}",
                        "word1": row.get('word1', row.get('lu1', '')),
                        "word2": row.get('word2', row.get('lu2', '')),
                        "definition1": row.get('def1', ''),
                        "definition2": row.get('def2', ''),
                        "sample_sentence1": row.get('sample_sentence1', ''),
                        "sample_sentence2": row.get('sample_sentence2', '')
                    }
                    outf.write(json.dumps(record, ensure_ascii=False) + '\n')

        print(f"[{pos}] Wrote {len(batches)} file pairs to {pos_dir}/", file=sys.stderr)

if __name__ == '__main__':
    main()
