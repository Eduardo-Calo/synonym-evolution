#!/usr/bin/env python3
"""
Scan Prolific/Qualtrics run logs for survey IDs and export them via
`qualtrics_export_survey.py`.

Usage:
  python export_surveys_from_logs.py [--log-dir LOG_DIR] [--workers N] [--dry-run]

Defaults:
  LOG_DIR: logs/prolific
  workers: 1 (serial). Set >1 to run exports in parallel.
"""
import os
import re
import sys
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed


SURVEY_ID_RE = re.compile(r"Survey ID\s*:\s*(\S+)", re.IGNORECASE)


def find_survey_ids(log_dir):
    ids = []
    if not os.path.isdir(log_dir):
        raise FileNotFoundError(f"Log directory not found: {log_dir}")
    for fname in os.listdir(log_dir):
        if not fname.endswith('.txt'):
            continue
        path = os.path.join(log_dir, fname)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception:
            continue
        for m in SURVEY_ID_RE.finditer(text):
            ids.append(m.group(1).strip())
    # preserve order but remove duplicates
    seen = set()
    unique = []
    for s in ids:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def export_survey(survey_id, dry_run=False):
    script_path = os.path.join(os.path.dirname(__file__), 'qualtrics_export_survey.py')
    cmd = [sys.executable, script_path, survey_id]
    if dry_run:
        print(f"DRY-RUN: would run: {' '.join(cmd)}")
        return True
    try:
        print(f"Exporting survey {survey_id}...")
        subprocess.run(cmd, check=True)
        print(f"Export completed: {survey_id}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Export failed for {survey_id}: {e}")
        return False


def main():
    p = argparse.ArgumentParser(description="Export Qualtrics surveys found in log files.")
    p.add_argument('--log-dir', default='logs/prolific', help='Directory containing .txt logs')
    p.add_argument('--workers', type=int, default=1, help='Number of parallel workers (default: 1)')
    p.add_argument('--dry-run', action='store_true', help="Don't actually run exports, just show commands")
    args = p.parse_args()

    try:
        survey_ids = find_survey_ids(args.log_dir)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    if not survey_ids:
        print(f"No survey IDs found in {args.log_dir}")
        return

    print(f"Found {len(survey_ids)} unique survey IDs.")

    successes = []
    if args.workers and args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(export_survey, sid, args.dry_run): sid for sid in survey_ids}
            for fut in as_completed(futures):
                sid = futures[fut]
                try:
                    ok = fut.result()
                except Exception as e:
                    print(f"Exception exporting {sid}: {e}")
                    ok = False
                successes.append((sid, ok))
    else:
        for sid in survey_ids:
            ok = export_survey(sid, dry_run=args.dry_run)
            successes.append((sid, ok))

    failed = [s for s, ok in successes if not ok]
    print("\nSummary:")
    print(f"  Total: {len(successes)}")
    print(f"  Successful: {len(successes) - len(failed)}")
    print(f"  Failed: {len(failed)}")
    if failed:
        print("Failed survey IDs:")
        for f in failed:
            print(" -", f)


if __name__ == '__main__':
    main()
