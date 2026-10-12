from pathlib import Path
import re
from typing import Dict, List
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests
import numpy as np

##############################################################################
# CONFIG
##############################################################################

PLOTS_FOLDER = "../data/plots"
MAIN_PATH = f"{PLOTS_FOLDER}/batch_summary_tables.txt"
CONTROL_PATH = f"{PLOTS_FOLDER}/batch_summary_tables_control.txt"

MIN_EXPECTED = 5    # fallback threshold for Fisher

CATEGORY_GROUPS: Dict[str, str] = {
    "Lost": "Differentiation",
    "Split": "Differentiation",
    "No Change": "No Change",
    "Parallel Addition": "Parallel Change",
    "Parallel Loss": "Parallel Change",
    "Parallel Addition + Loss": "Parallel Change",
    "Addition": "Non-Parallel Change",
    "Loss": "Non-Parallel Change",
    "Addition + Loss": "Non-Parallel Change",
}

##############################################################################
# PARSER
##############################################################################

def parse_file(input_path: str) -> pd.DataFrame:
    with open(input_path, "r") as f:
        lines = f.readlines()

    records: List[Dict] = []
    i = 0

    while i < len(lines):
        header = lines[i].strip()
        if not header.startswith("#"):
            i += 1
            continue

        match = re.match(r"#\s+(\w+)\s+(\d+\.?\d*)\s+(\d+)\s+(\d+)", header)
        if not match:
            i += 1
            continue

        pos, threshold_str, t1_str, t2_str = match.groups()
        threshold = float(threshold_str)
        t1 = int(t1_str)
        t2 = int(t2_str)
        period = f"{t1}-{t2}"
        span = t2 - t1

        i += 1
        counts: Dict[str, int] = {}
        total = 0

        while i < len(lines) and not lines[i].strip().startswith("#"):
            row = lines[i].strip()
            if "\t" in row:
                cat, value_str = row.split("\t", 1)
                try:
                    value = int(value_str)
                    counts[cat] = value
                    if cat == "Total":
                        total = value
                except ValueError:
                    pass
            i += 1

        for cat, value in counts.items():
            percentage = (value / total * 100.0) if total > 0 else 0.0
            records.append(
                {
                    "POS": pos,
                    "Threshold": threshold,
                    "T1": t1,
                    "T2": t2,
                    "Period": period,
                    "Span": span,
                    "Category": cat,
                    "Count": value,
                    "Total": total,
                    "Percentage": percentage,
                }
            )

    df = pd.DataFrame(records)
    print(f"Parsed {len(df)} records from {input_path}")
    return df

##############################################################################
# LOAD
##############################################################################

main_df = parse_file(MAIN_PATH)
control_df = parse_file(CONTROL_PATH)

main_df["condition"] = "main"
control_df["condition"] = "control"

df = pd.concat([main_df, control_df], ignore_index=True)
print(f"Combined DataFrame with {len(df)} records")

##############################################################################
# TESTS
##############################################################################

results = []

for keys, group in df.groupby(["POS", "Span", "Category"]):

    if keys[2] not in CATEGORY_GROUPS:
        continue

    pos, span, law = keys
    # eg = ('ADJ', 100, 'Lost')

    main_row = group[group["condition"] == "main"]
    control_row = group[group["condition"] == "control"]

    if len(main_row) == 0 or len(control_row) == 0:
        continue

    main_count = int(main_row["Count"].sum())
    main_total = int(main_row["Total"].sum())

    control_count = int(control_row["Count"].sum())
    control_total = int(control_row["Total"].sum())

    # 2x2 table
    #
    #              law     not law
    # main         a       b
    # control      c       d

    a = main_count
    b = main_total - main_count

    c = control_count
    d = control_total - control_count

    table = np.array([
        [a, b],
        [c, d]
    ])

    fisher = np.nan
    chi2 = np.nan

    # If an entire row or column is zero, chi-square is invalid
    if (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():

        fisher, p_value = fisher_exact(table)

        test_used = "fisher"

    else:
            
        chi2, p_value, dof, expected = chi2_contingency(
            table,
            correction=False
        )

        if (expected < MIN_EXPECTED).any():

            fisher, p_value = fisher_exact(table)
            chi2 = np.nan  # Not valid

            test_used = "fisher"

        else:

            test_used = "chi2"

    # 1. Calculate local proportions (percentages within the group baseline)
    prop_main = a / main_total if main_total > 0 else 0.0
    prop_control = c / control_total if control_total > 0 else 0.0

    # 2. Determine who is stronger based on local proportions
    if prop_main > prop_control:
        stronger_in = "Main"
    elif prop_control > prop_main:
        stronger_in = "Control"
    else:
        stronger_in = "Equal"

    # 3. Calculate reliable Odds Ratio using Haldane-Anscombe correction
    # This prevents division-by-zero or infinity results if a cell contains 0.
    A_corr, B_corr, C_corr, D_corr = a, b, c, d
    if 0 in (a, b, c, d):
        A_corr += 0.5
        B_corr += 0.5
        C_corr += 0.5
        D_corr += 0.5

    odds_ratio = (A_corr * D_corr) / (B_corr * C_corr) if (B_corr * C_corr) > 0 else np.nan

    results.append({
        "POS": pos,
        "Span": span,
        "Category": law,
        "main_count": a,
        "control_count": c,
        "main_total": main_total,
        "control_total": control_total,
        "prop_main": prop_main,
        "prop_control": prop_control,
        "odds_ratio": odds_ratio,
        "stronger_in": stronger_in,
        "fisher": fisher,
        "chi2": chi2,
        "p_value": p_value,
        "test": test_used,
    })

##############################################################################
# RESULTS DF
##############################################################################

results_df = pd.DataFrame(results)

##############################################################################
# MULTIPLE TESTING CORRECTION
##############################################################################

results_df["p_corrected"] = multipletests(
    results_df["p_value"],
    method="fdr_bh"
)[1]

results_df["significant"] = (
    results_df["p_corrected"] < 0.05
)

##############################################################################
# SORT
##############################################################################

results_df = results_df.sort_values(
    ["POS", "Span", "Category", "significant", "odds_ratio"],
    ascending=[True, True, True, False, False]
)

##############################################################################
# SAVE
##############################################################################

output_path = f"{PLOTS_FOLDER}/law_significance_results.xlsx"
results_df.to_excel(output_path, index=False)
print(f"Saved results to {output_path}")