import numpy as np
import os
import argparse
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="../data/similarities/{pos}")
    parser.add_argument("--pos", required=True)
    parser.add_argument("--wn-csv", default="../data/parsed/{pos}_wn.csv")
    parser.add_argument("--output", default="../data/histograms/{pos}/results.npz")
    parser.add_argument("--bins", type=int, default=1000)
    return parser.parse_args()


def load_synsets(wn_csv):
    df = pd.read_csv(wn_csv)
    synsets = [s if s != 'nan' else '' for s in df["synsets"].fillna("").astype(str).tolist()]
    return synsets


def process_block(args):
    """
    Process a single similarity block in parallel.
    Computes histograms for same and different synset pairs within the block.
    """
    (
        sims_path,
        meta_path,
        synsets,
        bins
    ) = args

    # Load block metadata: i_start, i_end, j_start, j_end
    meta = np.loadtxt(meta_path, dtype=int)
    i_start, i_end, j_start, j_end = meta

    # Load similarity matrix for the block (memory-mapped for efficiency)
    sims = np.load(sims_path, mmap_mode="r")

    # Get synset sets for the i and j ranges in this block
    syn_i_block = synsets[i_start:i_end]
    syn_j_block = synsets[j_start:j_end]

    # Initialize histograms for this block
    same_hist = np.zeros(bins, dtype=np.int64)
    diff_hist = np.zeros(bins, dtype=np.int64)

    # Precompute overlap matrix: True if synsets intersect (at least one shared synset)
    # This avoids recomputing intersections for each pair in the inner loop
    overlap = np.zeros((len(syn_i_block), len(syn_j_block)), dtype=bool)
    for ii, si in enumerate(syn_i_block):
        if not si:  # Skip if no synsets
            continue
        for jj, sj in enumerate(syn_j_block):
            if si & sj:  # Set intersection
                overlap[ii, jj] = True

    # Indices for j in global space
    gj_indices = np.arange(j_start, j_end)

    # Process each row (i) in the block
    for local_i in range(sims.shape[0]):
        gi = i_start + local_i  # Global i index
        row = sims[local_i]  # Similarity scores for this i vs all j in block

        # Valid pairs: gj > gi (upper triangle, excluding diagonal)
        valid_mask = gj_indices > gi
        if not valid_mask.any():
            continue  # No valid pairs for this i

        row_valid = row[valid_mask]  # Valid similarities
        same_mask_valid = overlap[local_i, valid_mask]  # Valid overlaps

        syn_i_set = syn_i_block[local_i]

        if not syn_i_set:  # i has no synsets
            # All valid pairs are "different"
            diff_hist += np.histogram(
                row_valid, bins=bins, range=(0.0, 1.0)
            )[0]
        else:
            # Split into same and different based on overlap
            if same_mask_valid.any():
                same_hist += np.histogram(
                    row_valid[same_mask_valid],
                    bins=bins,
                    range=(0.0, 1.0)
                )[0]

            if (~same_mask_valid).any():
                diff_hist += np.histogram(
                    row_valid[~same_mask_valid],
                    bins=bins,
                    range=(0.0, 1.0)
                )[0]

    return same_hist, diff_hist


def main():
    args = parse_args()
    input_dir = args.input_dir.format(pos=args.pos)
    output_path = args.output.format(pos=args.pos)
    wn_csv = args.wn_csv.format(pos=args.pos)

    print("Loading WordNet synsets")
    synsets = load_synsets(wn_csv)
    synsets = np.array(synsets, dtype=object)

    bins = args.bins
    # Global histograms to accumulate results from all blocks
    same_hist = np.zeros(bins, dtype=np.int64)
    diff_hist = np.zeros(bins, dtype=np.int64)

    # Find all similarity block files
    sim_files = sorted(
        f for f in os.listdir(input_dir)
        if f.startswith("sims_") and f.endswith(".npy")
    )

    print(f"Found {len(sim_files)} similarity blocks")

    # Preprocess synsets into sets for fast intersection
    synsets = [
        set(s.split(',')) if s else set()
        for s in synsets
    ]
    synsets = np.array(synsets, dtype=object)

    # Prepare tasks for parallel processing
    tasks = []
    for sim_file in sim_files:
        sims_path = os.path.join(input_dir, sim_file)
        meta_path = os.path.join(
            input_dir,
            sim_file.replace("sims_", "meta_").replace(".npy", ".txt")
        )
        tasks.append(
            (sims_path, meta_path, synsets, bins)
        )

    n_workers = max(1, cpu_count() - 1)
    print(f"Using {n_workers} CPU cores")

    with Pool(processes=n_workers) as pool:
        # Accumulate histograms from each block
        for same_h, diff_h in tqdm(
            pool.imap_unordered(process_block, tasks),
            total=len(tasks),
            desc="Processing blocks"
        ):
            same_hist += same_h
            diff_hist += diff_h

    # Compute bin centers and CDFs
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    same_total = same_hist.sum()
    diff_total = diff_hist.sum()

    same_cdf = np.cumsum(same_hist) / same_total if same_total > 0 else np.zeros(bins)
    diff_cdf = np.cumsum(diff_hist) / diff_total if diff_total > 0 else np.zeros(bins)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save aggregated results
    np.savez(
        output_path,
        bins=bins,
        bin_centers=bin_centers,
        same_hist=same_hist,
        diff_hist=diff_hist,
        same_cdf=same_cdf,
        diff_cdf=diff_cdf,
        same_total=same_total,
        diff_total=diff_total,
    )

    print("Saved results to", output_path)


if __name__ == "__main__":
    main()
