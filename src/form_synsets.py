import hashlib
import pandas as pd
import networkx as nx
import argparse
from pathlib import Path
import re
import json


def format_start(y):
    if pd.isna(y):
        return 'unknown'
    else:
        return str(int(y))


def format_end(y):
    if pd.isna(y):
        return 'present'
    else:
        return str(int(y))


def parse_args():
    parser = argparse.ArgumentParser(description="Form synsets from similarity pairs.")
    parser.add_argument("--input-csv", required=True, help="Path to pairs CSV from select_pairs.py")
    parser.add_argument("--output", required=True, help="Path to output synsets (will be written as JSONL)")
    parser.add_argument("--min-synset-size", type=int, default=1, help="Minimum number of LUs in a synset (default: 1)")
    parser.add_argument("--algorithm", choices=["connected_components", "label_propagation", "louvain", "k_clique", "spanning_tree"], default="louvain", help="Clustering algorithm (default: louvain)")
    parser.add_argument("--collapse-variants", action=argparse.BooleanOptionalAction, default=True, help="Collapse multiple variants per lemma within each synset (default: on)")
    parser.add_argument("--remove-self-pairs", action=argparse.BooleanOptionalAction, default=True, help="Remove rows where lu1 == lu2 before building synsets (default: on)")
    parser.add_argument("--remove-qwen-dissimilar", action=argparse.BooleanOptionalAction, default=True, help="Remove pairs labeled DISSIMILAR in Qwen output for the same POS (default: on)")
    parser.add_argument("--without-definitions", action=argparse.BooleanOptionalAction, default=False, help="Omit LU definitions in output JSONL while retaining sense_id fields (default: off)")
    return parser.parse_args()


def filter_synset_variants(synsets_list):
    from datetime import datetime
    current_year = datetime.now().year

    def parse_lu(lu):
        # Node key format: "lemma_start-end_defhash"
        parts = lu.split('_')
        if len(parts) >= 3:
            lemma = '_'.join(parts[:-2])
            years_part = parts[-2]
        elif len(parts) == 2:
            lemma, years_part = parts[0], parts[1]
        else:
            lemma, years_part = lu, ''

        if '-' in years_part:
            start_s, end_s = years_part.split('-', 1)
        else:
            start_s, end_s = years_part, 'present'

        def to_start(s):
            if s == 'unknown' or s == '' or pd.isna(s):
                return 0
            try:
                return int(s)
            except Exception:
                return 0

        def to_end(e):
            if e == 'present' or e == '' or pd.isna(e):
                return current_year
            try:
                return int(e)
            except Exception:
                return current_year

        return lemma, start_s, end_s, to_start(start_s), to_end(end_s)

    new_synsets = []
    removed_total = 0
    for syn in synsets_list:
        # group variants by lemma
        groups = {}
        for lu in syn:
            lemma, start_s, end_s, s_val, e_val = parse_lu(lu)
            groups.setdefault(lemma, []).append((lu, s_val, e_val))

        kept = []
        for lemma, variants in sorted(groups.items()):
            if len(variants) == 1:
                kept.append(variants[0][0])
            else:
                # choose variant with largest span (end - start), tiebreak by larger end then larger start
                chosen = max(variants, key=lambda x: (x[2] - x[1], x[2], x[1]))
                kept.append(chosen[0])
                removed_total += (len(variants) - 1)

        if kept:
            new_synsets.append(sorted(kept))

    print(f"Collapsed {removed_total} variant LUs across synsets")
    return new_synsets


def remove_self_pairs(df_in):
    """Remove rows where lu1 == lu2 (self-pairs)."""
    df = df_in.copy()
    same_mask = df['lu1'] == df['lu2']
    n_removed = same_mask.sum()
    if n_removed == 0:
        return df.reset_index(drop=True)

    df = df.loc[~same_mask].copy()
    print(f"Removed {n_removed} self-pair rows from input CSV")
    return df.reset_index(drop=True)


def load_qwen_dissimilar_idxs(labels_path):
    """Load idx values with predicted_label == DISSIMILAR from a Qwen JSONL labels file."""
    dissimilar = set()
    with open(labels_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            label = str(row.get('predicted_label', '')).upper().strip()
            idx = str(row.get('idx', '')).strip()
            if label == 'DISSIMILAR' and idx:
                dissimilar.add(idx)
    return dissimilar


def remove_qwen_dissimilar_pairs(df_in, input_csv_path):
    """Remove rows that Qwen labeled as DISSIMILAR for the matching POS."""
    df = df_in.copy()
    if 'sense_id1' not in df.columns or 'sense_id2' not in df.columns:
        print("Warning: sense_id columns not found; skipping Qwen dissimilar filtering")
        return df.reset_index(drop=True)

    input_name = Path(input_csv_path).name
    pos = input_name.split('_', 1)[0].upper()
    labels_path = Path(__file__).resolve().parents[1] / 'data' / 'llm_output' / 'Qwen_Qwen3_5-27B' / f'{pos}_hf_labels_Qwen_Qwen3_5-27B.jsonl'

    if not labels_path.exists():
        print(f"Warning: Qwen labels file not found at {labels_path}; skipping dissimilar filtering")
        return df.reset_index(drop=True)

    dissimilar_idxs = load_qwen_dissimilar_idxs(labels_path)
    if not dissimilar_idxs:
        print(f"No DISSIMILAR labels found in {labels_path}; no rows removed")
        return df.reset_index(drop=True)

    s1 = df['sense_id1'].fillna('').astype(str)
    s2 = df['sense_id2'].fillna('').astype(str)
    idx_forward = s1 + '|||' + s2
    idx_reverse = s2 + '|||' + s1
    remove_mask = idx_forward.isin(dissimilar_idxs) | idx_reverse.isin(dissimilar_idxs)

    n_removed = int(remove_mask.sum())
    if n_removed == 0:
        print(f"Qwen dissimilar filtering checked {len(df)} rows; removed 0")
        return df.reset_index(drop=True)

    df = df.loc[~remove_mask].copy()
    print(f"Removed {n_removed} rows labeled DISSIMILAR by Qwen from {labels_path.name}")
    return df.reset_index(drop=True)


def main():
    args = parse_args()

    # Extract threshold from input CSV filename
    match = re.search(r'thr(\d+\.\d+)_(\d+\.\d+)', args.input_csv)
    if match:
        threshold_min = float(match.group(1))
        threshold_max = float(match.group(2))
    else:
        threshold_min = None
        threshold_max = None

    # Modify output filename to include threshold and algorithm
    output_path = Path(args.output)
    if threshold_min is not None and threshold_max is not None:
        new_name = f"{output_path.stem}_thr{threshold_min}_{threshold_max}_{args.algorithm}{output_path.suffix or '.jsonl'}"
        output_path = output_path.with_name(new_name)
    # Ensure JSONL extension
    if output_path.suffix != '.jsonl':
        output_path = output_path.with_suffix('.jsonl')
    args.output = str(output_path)

    # Read pairs CSV
    df = pd.read_csv(args.input_csv, engine='python')
    # Optionally remove rows that Qwen judged dissimilar for the same POS
    if args.remove_qwen_dissimilar:
        df = remove_qwen_dissimilar_pairs(df, args.input_csv)
    # Optionally remove self-pair rows from the input CSV
    if args.remove_self_pairs:
        df = remove_self_pairs(df)
    print(f"Loaded {len(df)} pairs from {args.input_csv}")

    # Build graph: nodes are LUs with years + definition hash (unique per sense), edges are pairs
    G = nx.Graph()
    lu_to_def = {}
    lu_to_sense = {}

    def make_node_key(lemma, start, end, definition):
        """Create a unique node key including a short hash of the definition to avoid collisions."""
        def_hash = hashlib.md5(str(definition).encode()).hexdigest()[:8]
        return f"{lemma}_{format_start(start)}-{format_end(end)}_{def_hash}"

    def normalize_sense_id(sense_id):
        if pd.isna(sense_id):
            return None
        return str(sense_id)

    for _, row in df.iterrows():
        lu1 = make_node_key(row['lu1'], row['start1'], row['end1'], row['def1'])
        lu2 = make_node_key(row['lu2'], row['start2'], row['end2'], row['def2'])
        # Add edge with weight as cosine similarity
        G.add_edge(lu1, lu2, weight=row['cosine'])
        lu_to_def[lu1] = row['def1']
        lu_to_def[lu2] = row['def2']
        lu_to_sense[lu1] = normalize_sense_id(row.get('sense_id1'))
        lu_to_sense[lu2] = normalize_sense_id(row.get('sense_id2'))

    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Find communities/clusters using the chosen algorithm
    # Community detection (Louvain/Label Propagation) splits large transitive clusters
    # from connected components into meaningful synsets based on edge density/modularity,
    # avoiding giant over-clustered groups.
    if args.algorithm == "connected_components":
        components = list(nx.connected_components(G))
    elif args.algorithm == "label_propagation":
        components = list(nx.algorithms.community.label_propagation_communities(G))  # No weight support
    elif args.algorithm == "louvain":
        components = list(nx.algorithms.community.louvain_communities(G, weight='weight', resolution=1000.0))  # Higher resolution -> more smaller clusters
    elif args.algorithm == "k_clique":
        components = list(nx.algorithms.community.k_clique_communities(G, k=3))  # Requires cliques of size 3 or more
    elif args.algorithm == "spanning_tree":
        T = nx.maximum_spanning_tree(G, weight='weight')
        components = list(nx.connected_components(T))
    print(f"Found {len(components)} clusters using {args.algorithm}")

    # Filter components by min size
    synsets = [comp for comp in components if len(comp) >= args.min_synset_size]
    print(f"Kept {len(synsets)} synsets with at least {args.min_synset_size} LUs")

    # Optionally collapse multiple variants per lemma within each synset
    if args.collapse_variants:
        synsets = filter_synset_variants(synsets)

    # Prepare output: one row per synset, with synset_id and comma-separated LUs
    output_rows = []
    for synset_id, lus in enumerate(synsets, start=1):
        lus_sorted = sorted(lus)
        lus_list = []
        for lu in lus_sorted:
            # Node key format: "lemma_start-end_defhash"
            # Split off the lemma (first segment) and the rest
            parts = lu.split('_')
            if len(parts) >= 3:
                lemma = '_'.join(parts[:-2])  # handle lemmas with underscores
                years = parts[-2]
            elif len(parts) == 2:
                lemma, years = parts[0], parts[1]
            else:
                lemma, years = lu, ''

            # parse years into start (int or None) and end (int or None)
            start_val = None
            end_val = None
            if '-' in years:
                start_raw, end_raw = years.split('-', 1)
                try:
                    if start_raw != 'unknown' and start_raw != '' and not pd.isna(start_raw):
                        start_val = int(start_raw)
                    else:
                        start_val = None
                except Exception:
                    start_val = None
                if end_raw == 'present' or end_raw == '' or pd.isna(end_raw):
                    end_val = None
                else:
                    try:
                        end_val = int(end_raw)
                    except Exception:
                        end_val = None

            # drop lemmas without a start year
            if start_val is None:
                print(f"Warning: LU '{lu}' in synset {synset_id} has unknown start year, skipping")
                continue

            lu_entry = {
                'lemma': lemma,
                'sense_id': lu_to_sense.get(lu),
                'start': start_val,
                'end': end_val,
            }
            if not args.without_definitions:
                lu_entry['def'] = lu_to_def.get(lu, '')
            lus_list.append(lu_entry)
        # skip synsets that lost all LUs after dropping unknown-start lemmas
        if not lus_list:
            continue

        output_rows.append({
            'synset_id': synset_id,
            'lus': lus_list,
            'size': len(lus_list)
        })

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build DataFrame for statistics and also write JSONL
    out_df = pd.DataFrame(output_rows)

    # Write output as JSONL (one JSON object per line)
    jsonl_path = Path(args.output)
    with open(jsonl_path, 'w', encoding='utf-8') as fh:
        for r in output_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"Saved {len(output_rows)} synsets to {jsonl_path}")

    # Print synset size statistics
    sizes = out_df['size']
    print("\nSynset Size Statistics:")
    print(f"Min size: {sizes.min()}")
    print(f"Max size: {sizes.max()}")
    print(f"Avg size: {sizes.mean():.2f}")
    print(f"Median size: {sizes.median()}")
    print(f"Std dev: {sizes.std():.2f}")
    print(f"Total LUs in synsets: {sizes.sum()}")
    print("\nSize Distribution (top 10):")
    print(sizes.value_counts().head(10))


if __name__ == "__main__":
    main()
