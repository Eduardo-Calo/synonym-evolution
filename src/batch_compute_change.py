import subprocess
import multiprocessing as mp
from typing import List, Tuple, Dict
import argparse
import os


def generate_default_year_ranges(start: int = 1500, end: int = 2025, step: int = 10, max_window: int = 500) -> List[str]:
    """
    Generates default ranges for all window lengths (step, 2*step, ... max_window)
    over decade-aligned starts, and includes the full span start-end.
    """
    ranges = []
    for window in range(step, max_window + 1, step):
        for t1 in range(start, end - step + 1, step):
            t2 = t1 + window
            if t2 <= end:
                ranges.append(f"{t1}-{t2}")

    full_span = f"{start}-{end}"
    if full_span not in ranges:
        ranges.append(full_span)

    return ranges


def run_compute_change(pos: str, t1: int, t2: int, threshold: float, mode: str, lemma_mode: str = 'syn') -> str:
    """
    Runs the appropriate compute_*.py script for given params and captures the output.
    Mode may be: 'change' (default), 'control', or 'lemma'. When 'lemma' is selected,
    `lemma_mode` is forwarded to `compute_per_lemma.py` as its `--mode` (choices: 'syn','non-syn').
    """
    print(f"Processing {pos} {t1}-{t2} threshold={threshold} mode={mode} lemma_mode={lemma_mode if mode == 'lemma' else 'N/A'}")
    # Map mode to script name
    if mode == 'control':
        script = 'compute_control.py'
    elif mode == 'lemma':
        script = 'compute_per_lemma.py'
    else:
        script = 'compute_change.py'

    cmd = ['python', script, pos, str(t1), str(t2), '--threshold', str(threshold)]
    # If running per-lemma, forward the per-lemma internal mode
    if mode == 'lemma' and lemma_mode:
        cmd += ['--mode', lemma_mode]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
    if result.returncode != 0:
        print(f"Error for {pos} {t1}-{t2} {threshold}: stderr: {result.stderr}")
        print(f"stdout: {result.stdout}")
        return ""
    return result.stdout


def parse_summary_table(output: str) -> Dict[str, int]:
    """
    Parses the summary table from the output.
    """
    lines = output.split('\n')
    in_table = False
    table = {}
    for line in lines:
        if line.strip() == '--- Summary Table ---':
            in_table = True
            continue
        if in_table and line.strip() and '\t' in line:
            key, value = line.split('\t', 1)
            try:
                table[key] = int(value)
            except ValueError:
                pass  # Skip non-numeric
        if in_table and not line.strip():
            break
    return table


def main():
    parser = argparse.ArgumentParser(description="Batch compute semantic change for multiple POS, thresholds, and year ranges in parallel.")
    parser.add_argument("--pos", nargs='+', default=['ADJ', 'NOUN', 'VERB'], help="List of POS (default: ADJ NOUN VERB).")
    parser.add_argument("--thresholds", nargs='+', type=float, default=[0.8], help="List of thresholds (default: 0.8).")
    parser.add_argument("--year_ranges", nargs='+', help="List of year ranges as t1-t2 (default: auto-generate all year aligned ranges; see --step).")
    parser.add_argument("--step", type=int, default=25, help="Step size in years for auto-generated ranges (default: 25).")
    parser.add_argument("--output", default="../data/plots/batch_summary_tables.txt", help="Output file for aggregated tables.")
    parser.add_argument("--cores", type=int, default=int(mp.cpu_count() / 4), help="Number of cores to use.")
    parser.add_argument("--mode", choices=['change', 'control', 'lemma'], default='change',
                        help="Mode to run: 'change' (default), 'control', or 'lemma'.")
    parser.add_argument("--lemma_mode", choices=['syn', 'non-syn'], default='syn',
                        help="When --mode=lemma, which internal mode to pass to compute_per_lemma.py (default: syn).")

    args = parser.parse_args()

    if args.step <= 0:
        parser.error("--step must be a positive integer")

    if args.mode != 'change':
        if args.output == "../data/plots/batch_summary_tables.txt":
            if args.mode == 'lemma':
                args.output = f"../data/plots/batch_summary_tables_{args.mode}_{args.lemma_mode}.txt"
            else:
                args.output = f"../data/plots/batch_summary_tables_{args.mode}.txt"

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if not args.year_ranges:
        args.year_ranges = generate_default_year_ranges(start=1500, end=2025, step=args.step, max_window=500)

    # Parse year ranges
    year_ranges = []
    for yr in args.year_ranges:
        t1, t2 = map(int, yr.split('-'))
        year_ranges.append((t1, t2))

    # Generate all combinations
    tasks = []
    for pos in args.pos:
        for thresh in args.thresholds:
            for t1, t2 in year_ranges:
                tasks.append((pos, t1, t2, thresh, args.mode, args.lemma_mode))

    print(f"Running {len(tasks)} tasks in parallel using {args.cores} cores...")

    # Run in parallel
    with mp.Pool(args.cores) as pool:
        results = pool.starmap(run_compute_change, tasks)

    # Process results
    output_lines = []
    for i, (pos, t1, t2, thresh, _, _) in enumerate(tasks):
        output = results[i]
        table = parse_summary_table(output)
        if table:
            output_lines.append(f"# {pos} {thresh} {t1} {t2}")
            for key, value in table.items():
                output_lines.append(f"{key}\t{value}")
            output_lines.append("")  # Blank line

    # Write to file
    with open(args.output, 'w') as f:
        f.write('\n'.join(output_lines))

    print(f"Aggregated tables saved to {args.output}")

if __name__ == "__main__":
    main()
