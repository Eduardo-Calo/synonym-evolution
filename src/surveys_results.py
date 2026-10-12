#!/usr/bin/env python3
import json
import argparse
import csv
from pathlib import Path
import krippendorff
import numpy as np


def load_results(path: Path):
    with path.open() as f:
        return json.load(f)


def normalize_annotators(results: dict, source_id: str):
    """Ensure each answer has a non-empty annotator ID.

    When annotator IDs are missing/blank, create deterministic IDs based on
    the source file and answer position, e.g. "SV_xxx__ann1".

    Returns a tuple (normalized_results, n_items_fixed).
    """
    fixed_items = 0

    for item, info in results.items():
        answers = list(info.get('answers', []))
        annots = list(info.get('annotators', []))

        # One annotator ID per answer; fill missing/blank values deterministically.
        normalized_annots = []
        for i in range(len(answers)):
            existing = annots[i] if i < len(annots) else None
            if existing is None or str(existing).strip() == '':
                normalized_annots.append(f"{source_id}__ann{i + 1}")
            else:
                normalized_annots.append(str(existing))

        if normalized_annots != annots:
            fixed_items += 1
        info['annotators'] = normalized_annots

    return results, fixed_items


def merge_results(acc: dict, new: dict):
    """Merge `new` survey results into accumulator `acc`.

    For each item key, append annotators and answers so that multiple
    survey files accumulate annotations for the same item.
    """
    for item, info in new.items():
        if item not in acc:
            # copy shallowly; ensure lists are present
            acc[item] = {
                'annotators': list(info.get('annotators', [])),
                'answers': list(info.get('answers', [])),
                'timing': info.get('timing'),
            }
        else:
            acc[item].setdefault('annotators', [])
            acc[item].setdefault('answers', [])
            acc[item]['annotators'].extend(info.get('annotators', []))
            acc[item]['answers'].extend(info.get('answers', []))
            # ignore merging timing details
    return acc


def load_batch(paths):
    """Load one or more survey JSON files or directories containing them.

    `paths` may be a list of Path objects or strings. Directories are
    searched for files named 'results.json'. Returns a merged results dict
    combining annotators/answers for matching items.
    """
    files = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            # find results.json files under the directory
            files.extend(sorted(pp.rglob('results.json')))
        elif pp.is_file():
            files.append(pp)
        else:
            # try glob expansion
            files.extend(sorted(Path('.').glob(str(p))))

    merged = {}
    for f in files:
        try:
            data = load_results(f)
            source_id = f.parent.name
            data, fixed_items = normalize_annotators(data, source_id=source_id)
            if fixed_items:
                print(f"Info: normalized annotator IDs in {f} ({fixed_items} item(s) fixed)")
        except Exception as e:
            print(f"Warning: failed to load {f}: {e}")
            continue
        merge_results(merged, data)

    return merged, files


def analyze_attention_checks(results: dict):
    """Analyze attention-check items before they are dropped.

    Returns (total_att_checks, per_annotator_pass_count, failed_both_list, failed_all_list)
    """
    att_items = {k: v for k, v in results.items() if 'attention_check' in k}
    total = len(att_items)

    # collect annotators present in attention checks
    annotator_set = set()
    for info in att_items.values():
        for a in info.get('annotators', []):
            annotator_set.add(a)
    annotators = sorted(annotator_set)

    per_annot = {a: 0 for a in annotators}
    for info in att_items.values():
        anns = info.get('annotators', [])
        answers = info.get('answers', [])
        for a, ans in zip(anns, answers):
            try:
                val = float(ans)
            except Exception:
                continue
            if val == 1:
                per_annot[a] = per_annot.get(a, 0) + 1

    # flagging
    failed_all = [a for a, c in per_annot.items() if c == 0]
    failed_both = []
    if total == 2:
        failed_both = failed_all.copy()

    return total, per_annot, failed_both, failed_all


def filter_attention_checks(results: dict):
    # Drop any item keys containing 'attention_check'
    return {k: v for k, v in results.items() if 'attention_check' not in k}


def build_matrix(results: dict):
    # Collect unique annotator ids
    annotator_set = set()
    for item, info in results.items():
        for a in info.get('annotators', []):
            annotator_set.add(a)
    annotators = sorted(annotator_set)

    items = sorted(results.keys())

    # Build matrix as list of lists: one row per annotator, columns per item
    matrix = []
    for ann in annotators:
        row = []
        for it in items:
            info = results[it]
            # find annotator index if present
            try:
                idx = info.get('annotators', []).index(ann)
            except ValueError:
                idx = None
            if idx is None:
                row.append(None)
            else:
                ans = info.get('answers', [])[idx]
                # convert to number or None
                row.append(ans if ans is not None else None)
        matrix.append(row)

    return annotators, items, matrix


def encode_categorical_matrix(matrix):
    """Convert matrix entries to floats and set missing to NaN.

    Assumes entries are numeric (1/2/3) or numeric strings; returns a
    numpy float array with NaN for missing/invalid values. Krippendorff's
    nominal measurement treats these as symbolic categories, so using the
    original numeric labels is sufficient.
    """
    emat = []
    for row in matrix:
        erow = []
        for v in row:
            if v is None:
                erow.append(np.nan)
                continue
            try:
                erow.append(float(int(v)))
            except Exception:
                erow.append(np.nan)
        emat.append(erow)
    return np.array(emat, dtype=float)


def per_item_stats(results: dict):
    """Return per-item categorical counts and majority agreement.

    Returns list of tuples: (item, counts_dict, n_answers, majority_label, majority_prop)
    where counts_dict maps label->count and majority_prop is fraction of annotators
    who chose the majority label.
    """
    rows = []
    for item, info in sorted(results.items()):
        answers = [a for a in info.get('answers', []) if a is not None]
        n = len(answers)
        counts = {'similar': 0, 'dissimilar': 0, 'cannot decide': 0}
        for a in answers:
            try:
                ai = int(a)
            except Exception:
                continue
            if ai == 1:
                counts['similar'] += 1
            elif ai == 2:
                counts['dissimilar'] += 1
            elif ai == 3:
                counts['cannot decide'] += 1
            else:
                counts['cannot decide'] += 1

        if n == 0:
            maj_label = None
            maj_prop = 0.0
        else:
            # keep the three categories separate again (similar, dissimilar, cannot decide)
            max_count = max(counts.values())
            tied = [lbl for lbl, c in counts.items() if c == max_count]
            if len(tied) == 1:
                maj_label = tied[0]
                maj_prop = counts[maj_label] / n
            else:
                # ambiguous majority (tie across two or more categories)
                maj_label = None
                maj_prop = 0.0

        rows.append((item, counts, n, maj_label, maj_prop))
    return rows


def normalize_binary_label(label):
    """Map labels to {'similar', 'dissimilar'} or return None."""
    if label is None:
        return None
    s = str(label).strip().lower()
    if s in {'similar', '1'}:
        return 'similar'
    if s in {'dissimilar', '2'}:
        return 'dissimilar'
    return None


def load_qwen_predictions(jsonl_paths):
    """Load qwen predictions from one or more JSONL files.

    Expected fields per row: idx, source, predicted_label.
    Only rows with binary predicted labels are used.
    Returns dict: item_id -> predicted_label ('similar'/'dissimilar').
    """
    preds = {}
    n_rows = 0
    n_kept = 0

    files = []
    for p in jsonl_paths:
        pp = Path(p)
        if pp.is_dir():
            files.extend(sorted(pp.rglob('*.jsonl')))
        elif pp.is_file():
            files.append(pp)
        else:
            files.extend(sorted(Path('.').glob(str(p))))

    for fp in files:
        try:
            with fp.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    n_rows += 1
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue

                    item_id = row.get('idx')
                    pred = normalize_binary_label(row.get('predicted_label'))
                    if not item_id or pred is None:
                        continue

                    preds[str(item_id)] = pred
                    n_kept += 1
        except Exception as e:
            print(f"Warning: failed to read JSONL {fp}: {e}")

    return preds, n_rows, n_kept


def compute_precision_recall_binary(gold_by_item, pred_by_item):
    """Compute precision/recall/f1 for similar and dissimilar classes.

    gold_by_item and pred_by_item map item_id -> {'similar','dissimilar'}.
    Returns a report dict with per-class stats and overall accuracy.
    """
    labels = ['similar', 'dissimilar']

    common_items = sorted(set(gold_by_item.keys()) & set(pred_by_item.keys()))
    if not common_items:
        return {
            'n_common': 0,
            'n_gold': len(gold_by_item),
            'n_pred': len(pred_by_item),
            'accuracy': None,
            'per_label': {lbl: {'tp': 0, 'fp': 0, 'fn': 0, 'precision': None, 'recall': None, 'f1': None} for lbl in labels},
        }

    y_true = [gold_by_item[it] for it in common_items]
    y_pred = [pred_by_item[it] for it in common_items]

    per_label = {}
    for lbl in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p == lbl)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lbl and p == lbl)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p != lbl)

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else None
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else None
        f1 = None
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)

        per_label[lbl] = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(common_items)

    return {
        'n_common': len(common_items),
        'n_gold': len(gold_by_item),
        'n_pred': len(pred_by_item),
        'accuracy': accuracy,
        'per_label': per_label,
    }


def pairwise_alpha(annotators, matrix, level='nominal'):
    """Compute Krippendorff's alpha for every annotator pair.

    Returns a dict mapping (a_i, a_j) -> alpha (or None on failure).
    """
    res = {}
    n = len(annotators)
    for i in range(n):
        for j in range(i + 1, n):
            sub = [matrix[i], matrix[j]]
            try:
                a = krippendorff.alpha(reliability_data=sub, level_of_measurement=level)
            except Exception:
                a = None
            res[(annotators[i], annotators[j])] = a
    return res


def triplewise_alpha(annotators, matrix, level='nominal'):
    """Compute Krippendorff's alpha for every annotator triple.

    Returns a dict mapping (a_i, a_j, a_k) -> alpha (or None on failure).
    """
    res = {}
    n = len(annotators)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                sub = [matrix[i], matrix[j], matrix[k]]
                try:
                    a = krippendorff.alpha(reliability_data=sub, level_of_measurement=level)
                except Exception:
                    a = None
                res[(annotators[i], annotators[j], annotators[k])] = a
    return res


def recurrent_low_pairs(pw_dict, threshold=0.2):
    """Given pairwise-alpha dict {(a,b): alpha}, count for each annotator
    how many of their pairs fall below `threshold` (exclude None values).

    Returns (per_annot_dict, low_pairs_list) where per_annot_dict maps
    annotator -> (low_count, total_count, ratio) and low_pairs_list is
    a list of ((a,b), alpha) for pairs below threshold.
    """
    per = {}
    low_pairs = []
    for (a, b), val in pw_dict.items():
        per.setdefault(a, [0, 0])
        per.setdefault(b, [0, 0])
        if val is None:
            # treat missing as not counted for the ratio
            continue
        per[a][1] += 1
        per[b][1] += 1
        if val < threshold:
            per[a][0] += 1
            per[b][0] += 1
            low_pairs.append(((a, b), val))

    # convert to tuples with ratio
    out = {}
    for a, (low, tot) in per.items():
        ratio = (low / tot) if tot > 0 else 0.0
        out[a] = (low, tot, ratio)
    return out, low_pairs


def items_with_label(results: dict, label='cannot decide'):
    """Return list of item keys where at least one annotator gave `label`.

    If `label` is 'cannot decide' we check for numeric code 3.
    """
    found = []
    want_code = None
    if isinstance(label, str) and label.strip().lower() == 'cannot decide':
        want_code = 3
    for item, info in sorted(results.items()):
        for a in info.get('answers', []):
            try:
                if want_code is not None:
                    if int(a) == want_code:
                        found.append(item)
                        break
                else:
                    # fallback: string match
                    if str(a).strip().lower() == str(label).strip().lower():
                        found.append(item)
                        break
            except Exception:
                continue
    return found


def remove_annotators_and_items(results_dict, remove_annots, remove_items):
    """Recompute alphas after removals in three scenarios:
    remove bad annotators, remove difficult items, remove both"""
    out = {}
    for it, info in results_dict.items():
        if it in remove_items:
            continue
        anns = []
        answers = []
        for a, ans in zip(info.get('annotators', []), info.get('answers', [])):
            if a in remove_annots:
                continue
            anns.append(a)
            answers.append(ans)
        if len(anns) > 0:
            out[it] = {'annotators': anns, 'answers': answers}
    return out


def run_analysis(merged, files, args):
    """Perform the analysis and printing."""
    print(f'Loaded {len(files)} file(s).')

    # Analyze attention checks BEFORE they are dropped
    total_att, per_annot_pass, failed_both, failed_all = analyze_attention_checks(merged)
    print(f"Attention-check items found: {total_att}")
    if total_att:
        if failed_both:
            print('\nAnnotators who failed both attention checks:')
            for a in failed_both:
                print(f"- {a}")
        else:
            print('\nNo annotators failed both attention checks.')

    filtered = filter_attention_checks(merged)

    # Build annotator/item matrix directly from filtered results
    p_annotators, p_items, p_matrix = build_matrix(filtered)
    print(f"\nItems (after removing attention checks): {len(p_items)}")
    print(f"Annotators found: {len(p_annotators)}")

    # CATEGORICAL encoding: encode categories to nominal integers and report nominal alpha
    emat = encode_categorical_matrix(p_matrix)
    try:
        alpha_nom = krippendorff.alpha(reliability_data=emat, level_of_measurement='nominal')
    except Exception:
        alpha_nom = None
    print('\n=== Nominal results (categorical) ===')
    print(f"Krippendorff's alpha (nominal, categories encoded): {alpha_nom}\n")

    pwb = pairwise_alpha(p_annotators, emat, level='nominal')
    if args.verbose:
        print("Pairwise Krippendorff's alpha (nominal, categories) — sorted by alpha desc:")
        # sort by alpha value (descending), put None at the end
        items_pw_sorted = sorted(pwb.items(), key=lambda kv: (kv[1] is None, -(kv[1] if kv[1] is not None else 0)))
        for (a, b), val in items_pw_sorted:
            vstr = f"{val:.6f}" if val is not None else 'None'
            print(f"- {a} vs {b}: {vstr}")

    per_annot_bin, low_pairs_bin = recurrent_low_pairs(pwb, threshold=args.pw_threshold)
    if args.verbose:
        print(f"\nAnnotator low-pair summary (nominal) with threshold {args.pw_threshold}:")
        for a, (low, tot, ratio) in sorted(per_annot_bin.items(), key=lambda x: x[1][2], reverse=True):
            print(f"- {a}: {low}/{tot} low pairs (ratio={ratio:.2f})")

    # Triple-wise (three annotators) Krippendorff's alpha
    twb = triplewise_alpha(p_annotators, emat, level='nominal')
    if args.verbose:
        print("\nTriplewise Krippendorff's alpha (nominal, categories) — sorted by alpha desc:")
        # sort by alpha value (descending), put None at the end
        items_sorted = sorted(twb.items(), key=lambda kv: (kv[1] is None, -(kv[1] if kv[1] is not None else 0)))
        for (a, b, c), val in items_sorted:
            vstr = f"{val:.6f}" if val is not None else 'None'
            print(f"- {a} vs {b} vs {c}: {vstr}")

    # Pair-label consensus summary: how many pairs are dominated by label 1 or 2
    # Item-based majority summary (ignore annotator pairs)
    # compute per-item categorical stats used below
    stats = per_item_stats(filtered)

    items_major_sim = sum(1 for _it, counts, n, maj, majp in stats if maj == 'similar')
    items_major_dis = sum(1 for _it, counts, n, maj, majp in stats if maj == 'dissimilar')
    print(f"\nItems majority similar: {items_major_sim}")
    print(f"Items majority dissimilar: {items_major_dis}")

    # overall category totals
    overall = {'similar': 0, 'dissimilar': 0, 'cannot decide': 0}
    for _it, counts, n, maj, majp in stats:
        for k, v in counts.items():
            overall[k] = overall.get(k, 0) + v
    print('\nOverall category totals:')
    for k, v in overall.items():
        print(f"- {k}: {v}")

    # items labeled 'cannot decide' by at least one annotator
    items_cd = items_with_label(filtered, label='cannot decide')
    if args.verbose:
        print(f"\nItems with 'cannot decide' by at least one annotator: {len(items_cd)}")
        for it in items_cd:
            print(f"- {it}")

    # identify bad annotators (ratio > 0.5) from nominal pairwise summary
    bad_annotators = [a for a, (_low, _tot, ratio) in per_annot_bin.items() if ratio > 0.5]
    print(f"\nAnnotators with >0.5 low-pair ratio (flagged as bad): {len(bad_annotators)}")
    for a in bad_annotators:
        low, tot, ratio = per_annot_bin[a]
        print(f"- {a}: {low}/{tot} low pairs (ratio={ratio:.2f})")

    # recompute nominal alpha removing bad annotators only
    new_results = remove_annotators_and_items(filtered, bad_annotators, [])
    new_ann, new_items, new_mat = build_matrix(new_results)
    # encode categorical
    new_emat = encode_categorical_matrix(new_mat)
    if bad_annotators:
        print('\nRecomputed nominal alphas after removing bad annotators:')
        try:
            new_alpha = krippendorff.alpha(reliability_data=new_emat, level_of_measurement='nominal')
        except Exception:
            new_alpha = None
        print(f"- remove_bad_annotators: items={len(new_items)}, annotators={len(new_ann)}, Krippendorff's alpha (nominal)={new_alpha}")

    if args.verbose:
        print('\nPer-item categorical stats (item, counts, n_answers, majority_label, majority_prop):')
        for item, counts, n, maj, majp in stats:
            print(f"- {item}: counts={counts}, n={n}, majority={maj} ({majp:.2f})")
    else:
        flagged = [(item, counts, n, maj, majp) for item, counts, n, maj, majp in stats if maj is None or maj == 'cannot decide']
        print("\nItems with no majority or 'cannot decide' majority:")
        if flagged:
            for item, counts, n, maj, majp in flagged:
                print(f"- {item}: counts={counts}, n={n}, majority={maj}, majority_prop={majp:.2f}")
        else:
            print('- None')

    if args.out_csv:
        outp = Path(args.out_csv)
        with outp.open('w', newline='') as csvf:
            w = csv.writer(csvf)
            # export per-item categorical stats (counts and majority)
            w.writerow(['item', 'similar_count', 'dissimilar_count', 'cannot_decide_count', 'n_answers', 'majority_label', 'majority_prop'])
            for item, counts, n, maj, majp in stats:
                w.writerow([item, counts.get('similar', 0), counts.get('dissimilar', 0), counts.get('cannot decide', 0), n, maj if maj is not None else '', f"{majp:.3f}"])

    if args.qwen_jsonl:
        gold_by_item = {}
        for item, _counts, _n, maj, _majp in stats:
            gold = normalize_binary_label(maj)
            if gold is not None:
                gold_by_item[item] = gold

        qwen_preds, n_rows, n_kept = load_qwen_predictions(args.qwen_jsonl)
        report = compute_precision_recall_binary(gold_by_item, qwen_preds)

        print('\n=== Qwen precision/recall vs majority-vote gold ===')
        print(f"Qwen JSONL rows read: {n_rows}")
        print(f"Qwen predictions kept (source=qwen, binary label): {n_kept}")
        print(f"Gold items (majority label available): {report['n_gold']}")
        print(f"Overlapping items used for evaluation: {report['n_common']}")

        acc = report['accuracy']
        print(f"Accuracy: {acc:.4f}" if acc is not None else 'Accuracy: None')

        for lbl in ['similar', 'dissimilar']:
            m = report['per_label'][lbl]
            p = m['precision']
            r = m['recall']
            f1 = m['f1']
            p_str = f"{p:.4f}" if p is not None else 'None'
            r_str = f"{r:.4f}" if r is not None else 'None'
            f1_str = f"{f1:.4f}" if f1 is not None else 'None'
            print(f"- {lbl}: precision={p_str}, recall={r_str}, f1={f1_str} (tp={m['tp']}, fp={m['fp']}, fn={m['fn']})")


def main():
    p = argparse.ArgumentParser(description='Compute IAA and per-item averages from survey results JSON')
    p.add_argument('paths', nargs='+', help='path(s) to results.json or directory containing surveys')
    p.add_argument('--out-csv', help='write per-item categorical stats to CSV', default=None)
    p.add_argument('--qwen-jsonl', nargs='+', default=None, help='path(s) to Qwen sample/prediction JSONL files used for precision/recall evaluation')
    p.add_argument('--pw-threshold', type=float, default=0.2, help='threshold for low pairwise alpha')
    p.add_argument('--verbose', '-v', action='store_true', default=False, help='print debug matrices and intermediate arrays')
    args = p.parse_args()

    merged, files = load_batch(args.paths)
    if not files:
        print('No survey files found for the given paths.')
        return

    run_analysis(merged, files, args)


if __name__ == '__main__':
    main()
