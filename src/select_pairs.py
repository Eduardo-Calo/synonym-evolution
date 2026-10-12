import numpy as np
import os
import argparse
import pandas as pd
import math
from tqdm import tqdm
import random
import tempfile
import shutil
from multiprocessing import Pool, cpu_count


WORKER_CSV = None
WORKER_SIMS = {}


def init_worker(csv_data):
    global WORKER_CSV, WORKER_SIMS
    WORKER_CSV = csv_data
    WORKER_SIMS = {}


def get_sims_mmap(sims_path):
    sims = WORKER_SIMS.get(sims_path)
    if sims is None:
        sims = np.load(sims_path, mmap_mode="r")
        WORKER_SIMS[sims_path] = sims
    return sims


def iter_qualifying_pairs(
    sims,
    csv,
    i_start,
    j_start,
    j_end,
    threshold_min,
    threshold_max,
    membership,
    row_start,
    row_end,
):
    syn_j = csv["synsets"][j_start:j_end]
    syn_j_sets = [set(s.split(',')) if s else set() for s in syn_j]
    lemma_valid = csv["lemma_valid"]

    row_end = min(row_end, sims.shape[0])

    for local_i in range(row_start, row_end):
        gi = i_start + local_i
        if not lemma_valid[gi]:
            continue

        row = sims[local_i]

        mask = (row >= threshold_min) & (row <= threshold_max)
        if not mask.any():
            continue

        syn_i_str = csv["synsets"][gi]
        syn_i_set = set(syn_i_str.split(',')) if syn_i_str else set()

        for local_j in np.where(mask)[0]:
            gj = j_start + local_j

            if gi == gj or gi > gj:
                continue

            if not lemma_valid[gj]:
                continue

            syn_j_set = syn_j_sets[local_j]
            same_synset = not syn_i_set.isdisjoint(syn_j_set)

            # Filter based on membership option
            if membership == "yes" and not same_synset:
                continue
            elif membership == "no" and same_synset:
                continue

            yield gi, gj, float(row[local_j]), same_synset, syn_i_set, syn_j_set


def format_pair_line(csv, gi, gj, cosine, same_synset, syn_i_set, syn_j_set):
    lemma1 = csv['lemma'][gi]
    lemma2 = csv['lemma'][gj]

    if same_synset:
        shared_synsets = ','.join(sorted(syn_i_set & syn_j_set))
    else:
        shared_synsets = ''

    start1 = int(csv['start'][gi]) if not pd.isna(csv['start'][gi]) else ''
    end1 = int(csv['end'][gi]) if not pd.isna(csv['end'][gi]) else ''
    start2 = int(csv['start'][gj]) if not pd.isna(csv['start'][gj]) else ''
    end2 = int(csv['end'][gj]) if not pd.isna(csv['end'][gj]) else ''

    # Escape quotes in strings for CSV
    def1 = csv['definition'][gi].replace('\n', ' ').replace('\r', ' ').replace('"', '""')
    def2 = csv['definition'][gj].replace('\n', ' ').replace('\r', ' ').replace('"', '""')
    shared_esc = shared_synsets.replace('\n', ' ').replace('\r', ' ').replace('"', '""')

    senses1 = str(csv['sense_id'][gi])
    senses2 = str(csv['sense_id'][gj])

    return (
        f"\"{lemma1}\",\"{lemma2}\","
        f"\"{def1}\","
        f"\"{def2}\","
        f"{cosine:.6f},"
        f"{int(same_synset)},"
        f"\"{shared_esc}\","
        f"{start1},{end1},"
        f"{start2},{end2},"
        f"{senses1},{senses2}\n"
    )


def get_bin_count(threshold_min, threshold_max, bin_size):
    if threshold_max <= threshold_min:
        return 1
    return max(1, int(math.ceil((threshold_max - threshold_min) / bin_size)))


def get_bin_index(cosine, threshold_min, threshold_max, bin_size, n_bins):
    if cosine < threshold_min or cosine > threshold_max:
        return None

    if threshold_max == threshold_min:
        return 0

    rel = (cosine - threshold_min) / bin_size
    idx = int(rel)
    if idx < 0:
        idx = 0
    if idx >= n_bins:
        idx = n_bins - 1
    return idx


def process_block_dump(args):
    """
    Process a single similarity block in parallel.
    Writes qualifying pair lines to a temporary file and returns its path and count.
    """
    (
        sims_path,
        i_start,
        j_start,
        j_end,
        threshold_min,
        threshold_max,
        membership,
        tmp_dir,
        row_start,
        row_end,
    ) = args

    csv = WORKER_CSV

    if i_start > j_start:
        return None, 0

    sims = get_sims_mmap(sims_path)

    fd, tmp_path = tempfile.mkstemp(prefix="pairs_", suffix=".csvpart", dir=tmp_dir)
    os.close(fd)
    pair_count = 0

    with open(tmp_path, "w", encoding="utf-8", newline="") as out:
        for gi, gj, cosine, same_synset, syn_i_set, syn_j_set in iter_qualifying_pairs(
            sims,
            csv,
            i_start,
            j_start,
            j_end,
            threshold_min,
            threshold_max,
            membership,
            row_start,
            row_end,
        ):
            out.write(format_pair_line(csv, gi, gj, cosine, same_synset, syn_i_set, syn_j_set))
            pair_count += 1

    return tmp_path, pair_count


def count_block(args):
    sims_path, i_start, j_start, j_end, threshold_min, threshold_max, membership, row_start, row_end = args
    csv = WORKER_CSV

    if i_start > j_start:
        return 0

    sims = get_sims_mmap(sims_path)

    if membership == "both":
        lemma_valid = csv["lemma_valid"]
        row_end = min(row_end, sims.shape[0])
        count = 0
        for local_i in range(row_start, row_end):
            gi = i_start + local_i
            if not lemma_valid[gi]:
                continue

            row = sims[local_i]
            mask = (row >= threshold_min) & (row <= threshold_max)
            if not mask.any():
                continue

            local_js = np.where(mask)[0]
            if local_js.size == 0:
                continue

            gjs = j_start + local_js
            valid = (gjs > gi) & lemma_valid[gjs]
            count += int(np.count_nonzero(valid))
        return count

    count = 0
    for _ in iter_qualifying_pairs(
        sims,
        csv,
        i_start,
        j_start,
        j_end,
        threshold_min,
        threshold_max,
        membership,
        row_start,
        row_end,
    ):
        count += 1
    return count


def sample_block(args):
    (
        sims_path,
        i_start,
        j_start,
        j_end,
        threshold_min,
        threshold_max,
        membership,
        row_start,
        row_end,
        block_count,
        quota,
        seed,
    ) = args

    if quota <= 0 or block_count <= 0:
        return []

    csv = WORKER_CSV

    if i_start > j_start:
        return []

    sims = get_sims_mmap(sims_path)
    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(block_count), quota))
    sel_ptr = 0
    seen = 0
    sampled_lines = []

    for gi, gj, cosine, same_synset, syn_i_set, syn_j_set in iter_qualifying_pairs(
        sims,
        csv,
        i_start,
        j_start,
        j_end,
        threshold_min,
        threshold_max,
        membership,
        row_start,
        row_end,
    ):
        if sel_ptr >= quota:
            break

        if seen == selected_indices[sel_ptr]:
            sampled_lines.append(
                format_pair_line(csv, gi, gj, cosine, same_synset, syn_i_set, syn_j_set)
            )
            sel_ptr += 1

        seen += 1

    return sampled_lines


def count_block_bins(args):
    (
        sims_path,
        i_start,
        j_start,
        j_end,
        threshold_min,
        threshold_max,
        membership,
        row_start,
        row_end,
        bin_size,
        n_bins,
    ) = args

    if i_start > j_start:
        return [0] * n_bins

    csv = WORKER_CSV
    sims = get_sims_mmap(sims_path)
    counts = [0] * n_bins

    for _, _, cosine, _, _, _ in iter_qualifying_pairs(
        sims,
        csv,
        i_start,
        j_start,
        j_end,
        threshold_min,
        threshold_max,
        membership,
        row_start,
        row_end,
    ):
        b = get_bin_index(cosine, threshold_min, threshold_max, bin_size, n_bins)
        if b is not None:
            counts[b] += 1

    return counts


def sample_block_bins(args):
    (
        sims_path,
        i_start,
        j_start,
        j_end,
        threshold_min,
        threshold_max,
        membership,
        row_start,
        row_end,
        block_bin_counts,
        quotas_by_bin,
        bin_size,
        n_bins,
        seed,
    ) = args

    if i_start > j_start:
        return []

    csv = WORKER_CSV
    sims = get_sims_mmap(sims_path)

    if not any(quotas_by_bin):
        return []

    selected_indices_by_bin = {}
    for b in range(n_bins):
        quota = int(quotas_by_bin[b])
        block_count = int(block_bin_counts[b])
        if quota <= 0 or block_count <= 0:
            continue
        rng = random.Random((seed + (b + 1) * 1000003) % (2 ** 32))
        selected_indices_by_bin[b] = sorted(rng.sample(range(block_count), quota))

    if not selected_indices_by_bin:
        return []

    seen_by_bin = [0] * n_bins
    ptr_by_bin = [0] * n_bins
    sampled_lines = []

    for gi, gj, cosine, same_synset, syn_i_set, syn_j_set in iter_qualifying_pairs(
        sims,
        csv,
        i_start,
        j_start,
        j_end,
        threshold_min,
        threshold_max,
        membership,
        row_start,
        row_end,
    ):
        b = get_bin_index(cosine, threshold_min, threshold_max, bin_size, n_bins)
        if b is None or b not in selected_indices_by_bin:
            continue

        ptr = ptr_by_bin[b]
        picks = selected_indices_by_bin[b]
        if ptr < len(picks) and seen_by_bin[b] == picks[ptr]:
            sampled_lines.append(
                format_pair_line(csv, gi, gj, cosine, same_synset, syn_i_set, syn_j_set)
            )
            ptr_by_bin[b] += 1

        seen_by_bin[b] += 1

    return sampled_lines


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="../data/similarities/{pos}")
    parser.add_argument("--pos", required=True)
    parser.add_argument("--input-csv", default="../data/parsed/{pos}_wn.csv")
    parser.add_argument("--threshold-min", type=float, required=True, help="Minimum cosine similarity threshold")
    parser.add_argument("--threshold-max", type=float, default=1.1, help="Maximum cosine similarity threshold (default: 1.1; some sims might be slightly above 1 due to numerical rounding)")
    parser.add_argument("--membership", choices=["yes", "no", "both"], default="both",  # outdated. always keep "both".
                        help="Filter by synset membership: 'yes' for shared synsets, 'no' for no shared, 'both' for all above threshold")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of pairs to sample (default: all)")
    parser.add_argument("--bin-size", type=float, default=None,
                        help="Cosine bin width for stratified sampling (e.g., 0.1)")
    parser.add_argument("--samples-per-bin", type=int, default=None,
                        help="Sample up to N pairs from each cosine bin")
    parser.add_argument("--random-seed", type=int, default=None, help="Random seed for sampling (default: None)")
    parser.add_argument("--rows-per-task", type=int, default=1024,
                        help="Rows per task chunk (default: 1024). Lower values give smoother progress updates.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes (default: cpu_count()-1)")
    parser.add_argument("--output-csv", default="../data/extracted/{pos}_pairs_thr{min}_{max}_{mem}.csv")
    return parser.parse_args()


NULL_STRINGS = {"nan", "null", "none", ""}


def load_csv(input_csv):
    df = pd.read_csv(input_csv)
    lemmas = df["lemma"].fillna('').astype(str).str.strip().tolist()
    return {
        "lemma": lemmas,
        "lemma_valid": np.array([lemma.lower() not in NULL_STRINGS for lemma in lemmas], dtype=bool),
        "definition": df["definition"].tolist(),
        "synsets": [s if s != 'nan' else '' for s in df["synsets"].fillna('').astype(str).str.strip().tolist()],
        "start": df["start"].tolist(),
        "end": df["end"].tolist(),
        "sense_id": df["sense_id"].tolist()
    }


def with_suffix_before_ext(path, suffix):
    base, ext = os.path.splitext(path)
    return f"{base}{suffix}{ext}"


def main():
    args = parse_args()
    input_dir = args.input_dir.format(pos=args.pos)
    input_csv = args.input_csv.format(pos=args.pos)
    output_csv = args.output_csv.format(pos=args.pos, min=args.threshold_min, max=args.threshold_max, mem=args.membership)

    csv = load_csv(input_csv)

    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be >= 1")
    if args.samples_per_bin is not None and args.samples_per_bin < 1:
        raise ValueError("--samples-per-bin must be >= 1")
    if args.rows_per_task < 1:
        raise ValueError("--rows-per-task must be >= 1")
    if args.bin_size is not None and args.bin_size <= 0:
        raise ValueError("--bin-size must be > 0")
    if (args.bin_size is None) != (args.samples_per_bin is None):
        raise ValueError("--bin-size and --samples-per-bin must be provided together")
    if args.samples_per_bin is not None and args.max_samples is not None:
        raise ValueError("Use either --max-samples or --samples-per-bin, not both")

    if args.samples_per_bin is not None:
        suffix = f"_binsz{args.bin_size:g}_n{args.samples_per_bin}"
        output_csv = with_suffix_before_ext(output_csv, suffix)

    sim_files = sorted(
        f for f in os.listdir(input_dir)
        if f.startswith("sims_") and f.endswith(".npy")
    )

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    tmp_dir = tempfile.mkdtemp(prefix=f"{args.pos}_pairs_")

    # Prepare chunked tasks for parallel processing
    task_specs = []
    for sim_file in sim_files:
        sims_path = os.path.join(input_dir, sim_file)
        meta_path = sim_file.replace("sims_", "meta_").replace(".npy", ".txt")
        meta_path = os.path.join(input_dir, meta_path)
        meta = np.loadtxt(meta_path, dtype=int).reshape(2, 2)
        i_start, i_end = meta[0]
        j_start, j_end = meta[1]

        if i_start > j_start:
            continue

        n_rows = i_end - i_start
        for row_start in range(0, n_rows, args.rows_per_task):
            row_end = min(n_rows, row_start + args.rows_per_task)
            task_specs.append((sims_path, i_start, j_start, j_end, row_start, row_end))

    # Use multiprocessing to process blocks in parallel
    n_workers = args.workers if args.workers is not None else max(1, cpu_count() - 1)
    n_workers = max(1, min(n_workers, len(task_specs))) if task_specs else 1
    print(f"Using {n_workers} CPU cores")

    header = "lu1,lu2,def1,def2,cosine,same_synset,shared_synsets,start1,end1,start2,end2,sense_id1,sense_id2\n"
    total_pairs = 0

    try:
        if args.samples_per_bin is not None:
            n_bins = get_bin_count(args.threshold_min, args.threshold_max, args.bin_size)

            count_bin_tasks = [
                (
                    sims_path,
                    i_start,
                    j_start,
                    j_end,
                    args.threshold_min,
                    args.threshold_max,
                    args.membership,
                    row_start,
                    row_end,
                    args.bin_size,
                    n_bins,
                )
                for sims_path, i_start, j_start, j_end, row_start, row_end in task_specs
            ]

            block_bin_counts = []
            with Pool(processes=n_workers, initializer=init_worker, initargs=(csv,)) as pool:
                for counts in tqdm(
                    pool.imap(count_block_bins, count_bin_tasks),
                    total=len(count_bin_tasks),
                    desc="Counting per-bin qualifying pairs"
                ):
                    block_bin_counts.append(counts)

            bin_totals = [0] * n_bins
            for counts in block_bin_counts:
                for b in range(n_bins):
                    bin_totals[b] += int(counts[b])

            bin_targets = [min(args.samples_per_bin, int(total)) for total in bin_totals]
            total_pairs = int(sum(bin_totals))

            sample_positions_by_bin = []
            rng = random.Random(args.random_seed)
            for b in range(n_bins):
                total_b = int(bin_totals[b])
                target_b = int(bin_targets[b])
                if target_b <= 0:
                    sample_positions_by_bin.append([])
                else:
                    sample_positions_by_bin.append(sorted(rng.sample(range(total_b), target_b)))

            quotas_per_task = [[0] * n_bins for _ in range(len(task_specs))]
            for b in range(n_bins):
                selected_positions = sample_positions_by_bin[b]
                if not selected_positions:
                    continue

                sel_idx = 0
                cumulative = 0
                for task_idx, counts in enumerate(block_bin_counts):
                    if sel_idx >= len(selected_positions):
                        break
                    block_count = int(counts[b])
                    block_end = cumulative + block_count
                    while sel_idx < len(selected_positions) and selected_positions[sel_idx] < block_end:
                        quotas_per_task[task_idx][b] += 1
                        sel_idx += 1
                    cumulative = block_end

            seed_base = args.random_seed if args.random_seed is not None else int.from_bytes(os.urandom(8), "little")
            sample_tasks = []
            for idx, (task_spec, counts, quotas) in enumerate(zip(task_specs, block_bin_counts, quotas_per_task)):
                if not any(quotas):
                    continue
                sims_path, i_start, j_start, j_end, row_start, row_end = task_spec
                task_seed = (seed_base + (idx + 1) * 1000003) % (2 ** 32)
                sample_tasks.append(
                    (
                        sims_path,
                        i_start,
                        j_start,
                        j_end,
                        args.threshold_min,
                        args.threshold_max,
                        args.membership,
                        row_start,
                        row_end,
                        counts,
                        quotas,
                        args.bin_size,
                        n_bins,
                        task_seed,
                    )
                )

            with open(output_csv, "w", encoding="utf-8", newline="") as out:
                out.write(header)
                with Pool(processes=n_workers, initializer=init_worker, initargs=(csv,)) as pool:
                    for sampled_lines in tqdm(
                        pool.imap_unordered(sample_block_bins, sample_tasks),
                        total=len(sample_tasks),
                        desc="Sampling stratified pairs"
                    ):
                        out.writelines(sampled_lines)

            selected_total = int(sum(bin_targets))
            print(f"Stratified sampling bins={n_bins} bin_size={args.bin_size} samples_per_bin={args.samples_per_bin}")
            print(f"Selected pairs written: {selected_total}")

        elif args.max_samples is None:
            dump_tasks = [
                (
                    sims_path,
                    i_start,
                    j_start,
                    j_end,
                    args.threshold_min,
                    args.threshold_max,
                    args.membership,
                    tmp_dir,
                    row_start,
                    row_end,
                )
                for sims_path, i_start, j_start, j_end, row_start, row_end in task_specs
            ]

            with open(output_csv, "w", encoding="utf-8", newline="") as out:
                out.write(header)
                with Pool(processes=n_workers, initializer=init_worker, initargs=(csv,)) as pool:
                    for tmp_path, pair_count in tqdm(
                        pool.imap_unordered(process_block_dump, dump_tasks),
                        total=len(dump_tasks),
                        desc="Processing blocks"
                    ):
                        total_pairs += pair_count
                        if tmp_path is None:
                            continue
                        with open(tmp_path, "r", encoding="utf-8") as tmp_in:
                            shutil.copyfileobj(tmp_in, out)
                        os.remove(tmp_path)
        else:
            # Pass 1: count qualifying pairs per task chunk
            count_tasks = [
                (
                    sims_path,
                    i_start,
                    j_start,
                    j_end,
                    args.threshold_min,
                    args.threshold_max,
                    args.membership,
                    row_start,
                    row_end,
                )
                for sims_path, i_start, j_start, j_end, row_start, row_end in task_specs
            ]
            counts = []
            with Pool(processes=n_workers, initializer=init_worker, initargs=(csv,)) as pool:
                for count in tqdm(
                    pool.imap(count_block, count_tasks),
                    total=len(count_tasks),
                    desc="Counting qualifying pairs"
                ):
                    counts.append(count)

            total_pairs = int(sum(counts))
            target = min(args.max_samples, total_pairs)

            if target == 0:
                with open(output_csv, "w", encoding="utf-8", newline="") as out:
                    out.write(header)
            elif target == total_pairs:
                # Degenerate case: requested sample covers all qualifying pairs.
                dump_tasks = [
                    (
                        sims_path,
                        i_start,
                        j_start,
                        j_end,
                        args.threshold_min,
                        args.threshold_max,
                        args.membership,
                        tmp_dir,
                        row_start,
                        row_end,
                    )
                    for sims_path, i_start, j_start, j_end, row_start, row_end in task_specs
                ]
                with open(output_csv, "w", encoding="utf-8", newline="") as out:
                    out.write(header)
                    with Pool(processes=n_workers, initializer=init_worker, initargs=(csv,)) as pool:
                        for tmp_path, _ in tqdm(
                            pool.imap_unordered(process_block_dump, dump_tasks),
                            total=len(dump_tasks),
                            desc="Writing all qualifying pairs"
                        ):
                            if tmp_path is None:
                                continue
                            with open(tmp_path, "r", encoding="utf-8") as tmp_in:
                                shutil.copyfileobj(tmp_in, out)
                            os.remove(tmp_path)
            else:
                # Pass 2 quota allocation: exact without-replacement by sampling
                # global pair indices and mapping them to task chunks.
                rng = random.Random(args.random_seed)
                selected_positions = sorted(rng.sample(range(total_pairs), target))

                quotas = [0] * len(counts)
                sel_idx = 0
                cumulative = 0
                for idx, count in enumerate(counts):
                    if sel_idx >= target:
                        break
                    block_end = cumulative + count
                    while sel_idx < target and selected_positions[sel_idx] < block_end:
                        quotas[idx] += 1
                        sel_idx += 1
                    cumulative = block_end

                seed_base = args.random_seed if args.random_seed is not None else int.from_bytes(os.urandom(8), "little")
                sample_tasks = []
                for idx, ((sims_path, i_start, j_start, j_end, row_start, row_end), quota) in enumerate(zip(task_specs, quotas)):
                    if quota <= 0:
                        continue
                    task_seed = (seed_base + (idx + 1) * 1000003) % (2 ** 32)
                    sample_tasks.append(
                        (
                            sims_path,
                            i_start,
                            j_start,
                            j_end,
                            args.threshold_min,
                            args.threshold_max,
                            args.membership,
                            row_start,
                            row_end,
                            counts[idx],
                            quota,
                            task_seed,
                        )
                    )

                with open(output_csv, "w", encoding="utf-8", newline="") as out:
                    out.write(header)
                    with Pool(processes=n_workers, initializer=init_worker, initargs=(csv,)) as pool:
                        for sampled_lines in tqdm(
                            pool.imap_unordered(sample_block, sample_tasks),
                            total=len(sample_tasks),
                            desc="Sampling pairs"
                        ):
                            out.writelines(sampled_lines)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("Done. Extracted pairs written to")
    print(output_csv)
    print(f"Total qualifying pairs scanned: {total_pairs}")


if __name__ == "__main__":
    main()
