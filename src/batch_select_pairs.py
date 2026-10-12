import subprocess
import argparse
import multiprocessing as mp
import os

def run_select_pairs(params):
    """
    Run select_pairs.py with the given parameters.
    params: dict with keys: pos, threshold_min, membership
    """
    pos, threshold_min, membership = params['pos'], params['threshold_min'], params['membership']
    
    cmd = [
        'python', 'select_pairs.py',
        '--pos', pos,
        '--threshold-min', str(threshold_min),
        '--membership', membership
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
    
    if result.returncode != 0:
        print(f"Error in {pos} {threshold_min} {membership}: {result.stderr}")
    else:
        print(f"Completed: {pos} {threshold_min} {membership}")
    
    return result.returncode

def main():
    parser = argparse.ArgumentParser(description="Batch run select_pairs.py for multiple POS, thresholds, and memberships.")
    parser.add_argument("--pos", nargs='+', required=True, help="List of POS (e.g., NOUN VERB ADJ)")
    parser.add_argument("--threshold_mins", nargs='+', type=float, required=True, help="List of minimum thresholds (e.g., 0.7 0.8 0.9)")
    parser.add_argument("--memberships", nargs='+', choices=['yes', 'no', 'both'], default=['both'], help="List of memberships (e.g., yes no both)")
    parser.add_argument("--cores", type=int, default=int(mp.cpu_count() / 3), help="Number of cores to use (default: 1/3 of available cores)")  # needs to reduce further if thresholds are lower
    args = parser.parse_args()
    
    # Generate all combinations
    tasks = []
    for pos in args.pos:
        for threshold_min in args.threshold_mins:
            for membership in args.memberships:
                tasks.append({
                    'pos': pos,
                    'threshold_min': threshold_min,
                    'membership': membership
                })
    
    print(f"Total tasks: {len(tasks)}")
    
    # Run in parallel
    with mp.Pool(processes=args.cores) as pool:
        results = pool.map(run_select_pairs, tasks)
    
    # Check results
    failed = sum(1 for r in results if r != 0)
    if failed > 0:
        print(f"Failed tasks: {failed}")
    else:
        print("All tasks completed successfully.")

if __name__ == "__main__":
    main()
