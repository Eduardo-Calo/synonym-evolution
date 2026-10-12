import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


LABEL_RE = re.compile(r"LABEL:\s*(SIMILAR|DISSIMILAR)", re.IGNORECASE)


def bin_midpoint(bin_label: str) -> float:
	# bin label format from pandas cut: "[0.85, 0.9)"
	clean = bin_label.strip().lstrip("[").rstrip(")")
	left, right = clean.split(",")
	return (float(left.strip()) + float(right.strip())) / 2.0


def bin_left_edge(bin_label: str) -> float:
	clean = bin_label.strip().lstrip("[").rstrip(")")
	left, _right = clean.split(",")
	return float(left.strip())


def load_full_distribution(repo_root: Path, pos: str) -> tuple[np.ndarray, np.ndarray] | None:
	npz_path = repo_root / "data" / "histograms" / pos / "results.npz"
	if not npz_path.exists():
		return None

	data = np.load(npz_path)
	bin_centers = data["bin_centers"].astype(float)
	total_hist = data["same_hist"].astype(float) + data["diff_hist"].astype(float)
	if total_hist.sum() > 0:
		total_hist /= total_hist.sum()
	return bin_centers, total_hist


def load_full_distribution_counts(repo_root: Path, pos: str) -> tuple[np.ndarray, np.ndarray] | None:
	npz_path = repo_root / "data" / "histograms" / pos / "results.npz"
	if not npz_path.exists():
		return None

	data = np.load(npz_path)
	bin_centers = data["bin_centers"].astype(float)
	total_hist_counts = data["same_hist"].astype(float) + data["diff_hist"].astype(float)
	return bin_centers, total_hist_counts


def save_plotly_figure(fig: go.Figure, output_path: Path) -> tuple[Path, str | None]:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	try:
		fmt = output_path.suffix.lower().lstrip(".") or "svg"
		fig.write_image(str(output_path), format=fmt)
		return output_path, None
	except (ValueError, RuntimeError):
		fallback_path = output_path.with_suffix(".html")
		fig.write_html(str(fallback_path), include_plotlyjs="cdn") #, config={"displayModeBar": False})
		return (
			fallback_path,
			"Static export failed. Falling back to HTML plot. "
			"If you need SVG/PDF on this server, install Chrome for Kaleido.",
		)


def format_n(n):
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def plot_similarity_matrix(
	matrix_numeric: pd.DataFrame, output_path: Path, repo_root: Path
) -> tuple[Path, str | None]:
	# Kept for backward compatibility; now delegates to standalone full-distribution plot.
	return plot_full_distributions(output_path, repo_root, list(matrix_numeric.index.astype(str)))


def plot_full_distributions(
    output_path: Path, repo_root: Path, pos_list: list[str]
) -> tuple[Path, str | None]:
    line_colors = {
        "ADJ": "#ff7f0e",  # yellow
        "NOUN": "#9467bd",  # purple
        "VERB": "#8c564b",  # brown
    }

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=[], y=[], xaxis="x2", showlegend=False))

    all_centers: dict[str, np.ndarray] = {}
    all_counts: dict[str, np.ndarray] = {}
    
    for pos in pos_list:
        pos_upper = pos.upper()
        dist_counts = load_full_distribution_counts(repo_root, pos_upper)
        if dist_counts is None:
            continue
        centers, counts = dist_counts
        all_centers[pos_upper] = centers
        all_counts[pos_upper] = counts

        fig.add_trace(
            go.Bar(
                x=centers,
                y=counts,
                name=pos_upper,
                opacity=1.0,
                marker_color=line_colors.get(pos_upper, None),
                showlegend=True,
            )
        )

    count_edges = np.round(np.arange(0.0, 1.0001, 0.05), 4)
    total_counts_by_bin = np.zeros(len(count_edges) - 1, dtype=float)
    for pos in all_counts.keys():
        centers = all_centers[pos]
        counts = all_counts[pos]
        idx = np.digitize(centers, count_edges, right=False) - 1
        valid = (idx >= 0) & (idx < len(total_counts_by_bin))
        for i, c in zip(idx[valid], counts[valid]):
            total_counts_by_bin[i] += float(c)

    count_map: dict[float, int] = {}
    for i, left in enumerate(count_edges[:-1]):
        key = round(float(left), 2)
        count_map[key] = int(round(total_counts_by_bin[i]))

    for left in count_edges[:-1]:
        left_r = float(left)
        fig.add_vline(x=left_r, line_width=1, line_dash="dot", line_color="rgba(120,120,120,0.18)")
        key = round(float(left), 2)
        txt = format_n(count_map.get(key, 0))
        fig.add_annotation(
            x=left_r + 0.025,
            y=1.06,
            xref="x",
            yref="paper",
            text=txt,
            showarrow=False,
            font=dict(size=10, color="rgba(60,60,60,1)"),
        )

    fig.update_layout(
        xaxis_title="Cosine similarity",
        xaxis=dict(
            range=[0.0, 1.0], 
            tick0=0.0, 
            dtick=0.05, 
            tickformat=".2f",
            tickfont=dict(size=10),
            title_font=dict(size=22)
        ),
        xaxis2=dict(
            overlaying="x",
            side="top",
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            showline=False,
            title=dict(
                text="Total number of pairs per similarity bin",
                font=dict(size=22),
                standoff=25
            ),
        ),
        yaxis=dict(
            visible=False,
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        template="plotly_white",
        width=1100,
        height=750,
        barmode="overlay",
        margin=dict(t=160, b=150, l=90, r=70),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.6,
            entrywidthmode="fraction",
            entrywidth=0.32,
            font=dict(size=17),
        ),
    )

    return save_plotly_figure(fig, output_path)


def plot_threshold_tradeoff(
	result_num: pd.DataFrame, output_path: Path, repo_root: Path
) -> tuple[Path, str | None]:
	work = result_num.copy()
	work["bin_left"] = work["bin"].map(bin_left_edge)

	thresholds = np.round(np.arange(0.5, 0.951, 0.05), 2)
	pos_list = sorted(work["pos"].astype(str).unique())
	precision_by_pos: dict[str, list[float]] = {}
	for pos in pos_list:
		pos_work = work[work["pos"] == pos]
		# Use per-bin similar_rate at each 0.05 threshold point (not cumulative >= threshold).
		bin_rate_map = {
			round(float(left), 2): float(rate)
			for left, rate in zip(pos_work["bin_left"], pos_work["similar_rate"])
		}
		precision_vals: list[float] = []
		for thr in thresholds:
			precision_vals.append(bin_rate_map.get(round(float(thr), 2), np.nan))
		precision_by_pos[pos] = precision_vals

	all_centers: dict[str, np.ndarray] = {}
	all_counts: dict[str, np.ndarray] = {}
	for pos in pos_list:
		dist_counts = load_full_distribution_counts(repo_root, pos)
		if dist_counts is None:
			continue
		centers, counts = dist_counts
		all_centers[pos] = centers
		all_counts[pos] = counts

	# Aggregate true pair counts into 0.05 bins for top-of-plot annotations.
	count_thresholds = np.round(np.arange(0.5, 0.951, 0.05), 2)
	count_edges = np.round(np.arange(0.5, 1.0001, 0.05), 4)
	total_counts_by_bin = np.zeros(len(count_edges) - 1, dtype=float)
	for pos in pos_list:
		if pos not in all_counts:
			continue
		centers = all_centers[pos]
		counts = all_counts[pos]
		idx = np.digitize(centers, count_edges, right=False) - 1
		valid = (idx >= 0) & (idx < len(total_counts_by_bin))
		for i, c in zip(idx[valid], counts[valid]):
			total_counts_by_bin[i] += float(c)

	count_map: dict[float, int] = {}
	for i, left in enumerate(count_edges[:-1]):
		key = round(float(left), 2)
		count_map[key] = int(round(total_counts_by_bin[i]))

	fig = go.Figure()
	bar_colors = {
        "ADJ": "#ff7f0e",  # yellow
        "NOUN": "#9467bd",  # purple
        "VERB": "#8c564b",  # brown
	}

	right_axis_max = 0.0
	# Add histogram bars first (will be in the background).
	for pos in sorted(all_counts.keys()):
		centers = all_centers[pos]
		counts = all_counts[pos]
		mask = centers >= 0.5
		if mask.any():
			right_axis_max = max(right_axis_max, float(np.max(counts[mask])))
		fig.add_trace(
			go.Bar(
				x=centers,
				y=counts,
				name=f"{pos} histogram",
				opacity=0.5,
				marker_color=bar_colors.get(pos, None),
				yaxis="y2",
			)
		)

	# Add scatter lines on top of the histogram.
	for pos in pos_list:
		fig.add_trace(
			go.Scatter(
				x=thresholds,
				y=np.array(precision_by_pos[pos]) * 100.0,
				mode="lines+markers",
				name=f"{pos} inflection",
				line=dict(color=bar_colors.get(pos, None), width=4),
				marker=dict(size=8),
			)
		)
	fig.add_vline(x=0.8, line_width=2, line_dash="dash", line_color="black")

	for thr in count_thresholds:
		if thr in count_map:
			fig.add_vline(x=float(thr), line_width=1, line_dash="dot", line_color="rgba(120,120,120,0.35)")
			fig.add_annotation(
				x=float(thr) + 0.025,
				y=1.06,
				xref="x",
				yref="paper",
				text=format_n(count_map[thr]),
				showarrow=False,
				font=dict(size=17, color="rgba(60,60,60,1)"),
			)

	# Auto-focus the left axis on the precision band so inflection shape is visible.
	all_precision = np.concatenate([
		np.array(vals, dtype=float) * 100.0 for vals in precision_by_pos.values()
	])
	valid_precision = all_precision[~np.isnan(all_precision)]
	if valid_precision.size > 0:
		p_min = float(valid_precision.min())
		p_max = float(valid_precision.max())
		pad = max(1.0, (p_max - p_min) * 0.1)
		y1_min = max(0.0, p_min - pad)
		y1_max = min(100.0, p_max + pad)
		if y1_max - y1_min < 2.0:
			center = (y1_max + y1_min) / 2.0
			y1_min = max(0.0, center - 1.0)
			y1_max = min(100.0, center + 1.0)
	else:
		y1_min, y1_max = 0.0, 100.0

	if right_axis_max <= 0:
		right_axis_max = 1.0

	fig.add_trace(
		go.Scatter(
			x=[], 
			y=[], 
			xaxis="x2", 
			showlegend=False
		)
	)

	fig.update_layout(
		xaxis_title="<b>Cosine Similarity Threshold</b>",
		yaxis_title="<b>Qwen3.5 Similarity Rate (%)</b>",
		xaxis=dict(
			range=[0.5, 0.95],
			tick0=0.5,
			dtick=0.05,
			tickformat=".2f",
			title_font=dict(size=26),
			tickfont=dict(size=17),
			side="bottom",
		),
		xaxis2=dict(
			overlaying="x",
			side="top",
			showticklabels=False,
			showgrid=False,
			zeroline=False,
			showline=False,
			title=dict(text="<b>Total Number of Pairs per Similarity Bin</b>", font=dict(size=26), standoff=25),
		),
		yaxis=dict(range=[y1_min, y1_max], title_font=dict(size=26), tickfont=dict(size=17)),
		yaxis2=dict(
			overlaying="y",
			side="right",
			range=[0, right_axis_max * 0.001],
			showgrid=False,
			showticklabels=False,
			ticks="",
			visible=False,
		),
		template="plotly_white",
		width=1100,
		height=750,
		barmode="overlay",
		margin=dict(t=160, b=150, l=90, r=70),
		legend=dict(
			orientation="h",
			yanchor="top",
			y=-0.22,           
			xanchor="center",  
			x=0.55,             
			entrywidthmode="fraction",
			entrywidth=0.30,   
			font=dict(size=17),
		),
	)

	return save_plotly_figure(fig, output_path)


def infer_label(row: pd.Series) -> str | None:
	predicted = row.get("predicted_label")
	if isinstance(predicted, str):
		clean = predicted.strip().upper()
		if clean in {"SIMILAR", "DISSIMILAR"}:
			return clean

	model_output = row.get("model_output")
	if isinstance(model_output, str):
		match = LABEL_RE.search(model_output)
		if match:
			return match.group(1).upper()

	return None


def load_pos_file(path: Path) -> pd.DataFrame:
	if path.suffix.lower() == ".jsonl":
		return pd.read_json(path, lines=True)
	if path.suffix.lower() == ".csv":
		return pd.read_csv(path)
	raise ValueError(f"Unsupported file format: {path}")


def build_cosine_lookup(repo_root: Path, pos: str) -> pd.DataFrame:
	parts: list[pd.DataFrame] = []

	# Low/mid-cosine sampled set.
	llm_eval_csv = repo_root / "data" / "llm_eval" / pos / f"{pos}_sample_binary_binned.csv"
	if llm_eval_csv.exists():
		parts.append(pd.read_csv(llm_eval_csv, usecols=["idx", "cosine"]))

	# High-cosine set (0.85-1.1) from extracted pairs.
	extracted_csv = repo_root / "data" / "extracted" / f"{pos}_pairs_thr0.85_1.1_both.csv"
	if extracted_csv.exists():
		high_df = pd.read_csv(extracted_csv, usecols=["sense_id1", "sense_id2", "cosine"])
		high_df["idx"] = high_df["sense_id1"].astype(str) + "|||" + high_df["sense_id2"].astype(str)
		parts.append(high_df[["idx", "cosine"]])

	if not parts:
		return pd.DataFrame(columns=["idx", "cosine"])

	lookup = pd.concat(parts, ignore_index=True).dropna(subset=["idx", "cosine"])
	lookup = lookup.drop_duplicates(subset=["idx"], keep="first")
	return lookup


def attach_cosine_if_missing(repo_root: Path, pos: str, df: pd.DataFrame) -> pd.DataFrame:
	if "idx" not in df.columns:
		return df

	lookup = build_cosine_lookup(repo_root, pos)
	if lookup.empty:
		return df

	if "cosine" in df.columns:
		missing_mask = df["cosine"].isna()
		if not missing_mask.any():
			return df
		fill_df = df.loc[missing_mask].drop(columns=["cosine"]).merge(lookup, on="idx", how="left")
		out = df.copy()
		out.loc[missing_mask, "cosine"] = fill_df["cosine"].values
		return out

	return df.merge(lookup, on="idx", how="left")


def find_input_files(repo_root: Path, pos: str, model: str | None) -> list[Path]:
	if model:
		model_dir = repo_root / "data" / "llm_output" / model
		patterns = [
			f"{pos}_sample_binary_binned*{model}*.jsonl",
			f"{pos}_hf_labels_{model}.jsonl",
		]
		matches: list[Path] = []
		for pattern in patterns:
			matches.extend(sorted(model_dir.glob(pattern)))
		if matches:
			# Preserve order and drop duplicates by string path.
			seen = set()
			unique_matches = []
			for p in matches:
				sp = str(p)
				if sp in seen:
					continue
				seen.add(sp)
				unique_matches.append(p)
			return unique_matches

	fallback = repo_root / "data" / "llm_eval" / pos / f"{pos}_sample_binary_binned.csv"
	if fallback.exists():
		return [fallback]

	raise FileNotFoundError(f"No input file found for POS={pos}")


def summarize_pos(df: pd.DataFrame, pos: str, bin_size: float) -> pd.DataFrame:
	required = {"cosine"}
	missing = required - set(df.columns)
	if missing:
		raise ValueError(f"{pos}: missing required columns: {sorted(missing)}")

	work = df.copy()
	work["label"] = work.apply(infer_label, axis=1)
	work = work[work["label"].isin(["SIMILAR", "DISSIMILAR"])].copy()
	work["cosine"] = pd.to_numeric(work["cosine"], errors="coerce")
	work = work.dropna(subset=["cosine"])

	if work.empty:
		return pd.DataFrame(
			columns=["pos", "bin", "total_labeled", "similar", "similar_rate"]
		)

	edges = [0.0]
	current = 0.0
	while current < 1.0:
		current = round(current + bin_size, 10)
		edges.append(min(current, 1.0))
		if edges[-1] == 1.0:
			break

	work["bin"] = pd.cut(work["cosine"], bins=edges, include_lowest=True, right=False)

	grouped = (
		work.groupby("bin", observed=False)
		.agg(
			total_labeled=("label", "size"),
			similar=("label", lambda s: (s == "SIMILAR").sum()),
		)
		.reset_index()
	)
	grouped["similar_rate"] = grouped["similar"] / grouped["total_labeled"].replace(0, pd.NA)
	grouped["similar_rate"] = grouped["similar_rate"].fillna(0.0)
	grouped["pos"] = pos
	grouped["bin"] = grouped["bin"].astype(str)

	return grouped[["pos", "bin", "total_labeled", "similar", "similar_rate"]]


def print_full_distribution_counts(repo_root: Path, pos_list: list[str]):
	"""Prints a table of pair counts from the full distribution, binned by 0.05."""
	all_pos_counts = []
	bin_edges = np.round(np.arange(0.0, 1.001, 0.05), 10)
	bin_labels = [f"[{left},{right})" for left, right in zip(bin_edges[:-1], bin_edges[1:])]
	bin_labels[-1] = bin_labels[-1].replace(")", "]") # Make last bin inclusive

	for pos in pos_list:
		dist_data = load_full_distribution_counts(repo_root, pos)
		if dist_data is None:
			continue
		
		centers, counts = dist_data
		
		# Digitize the 0.01-binned centers into our 0.05-width bins
		binned_indices = np.digitize(centers, bin_edges, right=False) - 1
		
		pos_total_counts = pd.Series(np.zeros(len(bin_labels)), index=bin_labels)

		for i, count in zip(binned_indices, counts):
			if 0 <= i < len(bin_labels):
				pos_total_counts[bin_labels[i]] += count
		
		temp_df = pos_total_counts.reset_index()
		temp_df.columns = ["bin", "count"]
		temp_df["pos"] = pos
		all_pos_counts.append(temp_df)

	if not all_pos_counts:
		print("\nNo full distribution data found.")
		return

	full_dist_df = pd.concat(all_pos_counts, ignore_index=True)
	
	# Pivot to create the matrix
	dist_matrix = full_dist_df.pivot(index="pos", columns="bin", values="count").fillna(0).astype(int)
	
	# Ensure columns are sorted correctly
	dist_matrix = dist_matrix[bin_labels]

	print("\nPairs per bin matrix (from full distribution)")
	print(dist_matrix.to_string())


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Print SIMILAR counts per POS and cosine bin."
	)
	parser.add_argument(
		"--repo-root",
		type=Path,
		default=Path(__file__).resolve().parents[1],
		help="Repository root (default: inferred from this script).",
	)
	parser.add_argument(
		"--model",
		type=str,
		default=None,
		help="Model folder name under data/llm_output (optional).",
	)
	parser.add_argument(
		"--pos",
		nargs="+",
		default=["ADJ", "NOUN", "VERB"],
		help="POS tags to include (default: ADJ NOUN VERB).",
	)
	parser.add_argument(
		"--bin-size",
		type=float,
		default=0.05,
		help="Cosine bin width (default: 0.05).",
	)
	parser.add_argument(
		"--plot-path",
		type=Path,
		default=None,
		help="Optional output path for line plot image (default: data/plots/similar_rate_lineplot_<model_or_default>.svg).",
	)
	args = parser.parse_args()

	if not (0 < args.bin_size <= 1.0):
		raise ValueError("--bin-size must be in (0, 1].")

	all_rows = []
	for pos in args.pos:
		pos_upper = pos.upper()
		input_paths = find_input_files(args.repo_root, pos_upper, args.model)
		df_parts = [load_pos_file(path) for path in input_paths]
		df = pd.concat(df_parts, ignore_index=True) if len(df_parts) > 1 else df_parts[0]
		if "idx" in df.columns:
			df = df.drop_duplicates(subset=["idx"], keep="first")
		df = attach_cosine_if_missing(args.repo_root, pos_upper, df)
		summary = summarize_pos(df, pos_upper, args.bin_size)
		all_rows.append(summary)

	result = pd.concat(all_rows, ignore_index=True)
	if result.empty:
		print("No labeled rows found (SIMILAR/DISSIMILAR) in selected inputs.")
		return

	result_num = result.copy()
	result["similar_rate"] = result["similar_rate"].map(lambda v: f"{v:.2%}")

	print("Per-bin counts")
	print(result.to_string(index=False))

	overall = result.groupby("pos", as_index=False).agg(
		total_labeled=("total_labeled", "sum"), similar=("similar", "sum")
	)
	overall["similar_rate"] = (overall["similar"] / overall["total_labeled"]).map(
		lambda v: f"{v:.2%}"
	)

	print("\nOverall by POS")
	print(overall.to_string(index=False))

	# Final matrix: rows=POS, cols=bins (only bins with non-zero samples), cells=percentages.
	non_zero_bins = (
		result_num.groupby("bin", as_index=False)["total_labeled"].sum().query("total_labeled > 0")["bin"]
	)
	matrix_source = result_num[result_num["bin"].isin(non_zero_bins)]
	matrix_numeric = matrix_source.pivot(index="pos", columns="bin", values="similar_rate").fillna(0.0)
	matrix_numeric = matrix_numeric[sorted(matrix_numeric.columns)]
	matrix = matrix_numeric.applymap(lambda v: f"{v:.2%}")

	print("\nSimilar rate matrix (%)")
	print(matrix.to_string())

	# Pairs per bin matrix
	pairs_matrix = matrix_source.pivot(index="pos", columns="bin", values="total_labeled").fillna(0)
	pairs_matrix = pairs_matrix[sorted(pairs_matrix.columns)].astype(int)
	print("\nPairs per bin matrix (from labeled data)")
	print(pairs_matrix.to_string())

	print_full_distribution_counts(args.repo_root, [p.upper() for p in args.pos])

	if args.plot_path is None:
		model_tag = args.model if args.model else "default"
		base_plot_path = args.repo_root / "data" / "plots" / f"similar_rate_lineplot_{model_tag}.svg"
	else:
		base_plot_path = args.plot_path

	full_dist_path = base_plot_path.with_name(f"{base_plot_path.stem}_full_dist{base_plot_path.suffix}")
	full_saved_path, full_warning = plot_full_distributions(
		full_dist_path, args.repo_root, [p.upper() for p in args.pos]
	)
	if full_warning:
		print(f"\n{full_warning}")
	print(f"Saved full-distribution plot: {full_saved_path}")

	inflection_plot_path = base_plot_path.with_name(
		f"{base_plot_path.stem}_inflection_with_hist{base_plot_path.suffix}"
	)
	inflection_saved_path, inflection_warning = plot_threshold_tradeoff(
		result_num, inflection_plot_path, args.repo_root
	)
	if inflection_warning:
		print(f"\n{inflection_warning}")
	print(f"Saved inflection + histogram plot: {inflection_saved_path}")


if __name__ == "__main__":
	main()
