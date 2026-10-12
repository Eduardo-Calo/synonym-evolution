#!/usr/bin/env python3
"""analysis_synsets.py

Compute synset-size statistics across time spans from a synsets JSONL file.

Usage examples:
  python analysis_synsets.py \
    --input ../data/synsets/ADJ_synsets_thr0.8_1.1_louvain.jsonl \
    --start 1500 --end 2000 --step 50 --max-year 2026 --out stats.csv \
    --include-zero --summary-out stats_summary.json --print

The script treats a lemma as active in a span if its date-range
overlaps the span. If a lemma's `end` is null, it is treated as active
up to `--max-year` (default: current year).
"""

from __future__ import annotations

import argparse
import json
import csv
import sys
import random
from typing import Iterable, List, Tuple, Dict, Any, Optional
import statistics
import os
import plotly.graph_objects as go
import plotly.offline as pyo


def parse_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def make_spans(start: int, end: int, step: int) -> List[Tuple[int, int]]:
    spans = []
    cur = start
    while cur <= end:
        span_end = min(cur + step - 1, end)
        spans.append((cur, span_end))
        cur += step
    return spans


def lemma_active_in_span(lemma: Dict[str, Any], span_start: int, span_end: int, max_year: int) -> bool:
    s = lemma.get("start")
    e = lemma.get("end")
    if s is None:
        return False
    if e is None:
        e = max_year
    return not (e < span_start or s > span_end)


def synset_size_in_span(synset: Dict[str, Any], span_start: int, span_end: int, max_year: int) -> int:
    count = 0
    for lemma in synset.get("lus", []):
        if lemma_active_in_span(lemma, span_start, span_end, max_year):
            count += 1
    return count


def synset_first_active_year(synset: Dict[str, Any]) -> Optional[int]:
    starts = [l.get("start") for l in synset.get("lus", []) if l.get("start") is not None]
    if not starts:
        return None
    return min(starts)


def synset_last_active_year(synset: Dict[str, Any], max_year: int) -> Optional[int]:
    # If any lemma has open-ended activity (end is None), treat synset as still active
    ends = [l.get("end") for l in synset.get("lus", [])]
    if any(e is None for e in ends):
        return None
    ends_clean = [e for e in ends if e is not None]
    if not ends_clean:
        return None
    return max(ends_clean)


def stats_from_list(values: List[int]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "stdev": None,
        }
    out = {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }
    return out


def aggregate_stats_per_span(
    synsets: Iterable[Dict[str, Any]],
    spans: List[Tuple[int, int]],
    max_year: int,
    sample_n: int = 5,
    include_zero: bool = False,
) -> List[Dict[str, Any]]:
    synsets_list = list(synsets)
    results: List[Dict[str, Any]] = []
    for span_start, span_end in spans:
        sizes_info: List[Tuple[Any, int, Dict[str, Any]]] = []
        for s in synsets_list:
            sid = s.get("synset_id")
            size = synset_size_in_span(s, span_start, span_end, max_year)
            sizes_info.append((sid, size, s))

        sizes = [info[1] for info in sizes_info]
        nonzero = [v for v in sizes if v > 0]
        stat_all = stats_from_list(sizes)
        stat_nonzero = stats_from_list(nonzero)

        # appeared/disappeared synsets (store full synset objects for JSON)
        appeared_synsets: List[Dict[str, Any]] = []
        disappeared_synsets: List[Dict[str, Any]] = []
        for sid, _, s in sizes_info:
            first = synset_first_active_year(s)
            last = synset_last_active_year(s, max_year)
            if first is not None and span_start <= first <= span_end:
                appeared_synsets.append(s)
            if last is not None and span_start <= last <= span_end:
                disappeared_synsets.append(s)

        # top N largest
        top_sorted = sorted(sizes_info, key=lambda x: x[1], reverse=True)
        top_list = [{"synset_id": t[0], "size": t[1]} for t in top_sorted[:sample_n]]

        # bottom N smallest (optionally excluding zeros)
        bottom_sorted = sorted(sizes_info, key=lambda x: x[1])
        if not include_zero:
            bottom_sorted = [b for b in bottom_sorted if b[1] > 0]
        bottom_list = [{"synset_id": b[0], "size": b[1]} for b in bottom_sorted[:sample_n]]

        results.append({
            "span_start": span_start,
            "span_end": span_end,
            "synsets_total": len(synsets_list),
            "synsets_with_any": stat_nonzero["count"],
            "synsets_appeared": len(appeared_synsets),
            "synsets_disappeared": len(disappeared_synsets),
            "appeared_synsets": appeared_synsets,
            "disappeared_synsets": disappeared_synsets,
            "top_synsets": top_list,
            "bottom_synsets": bottom_list,
            "min_all": stat_all["min"],
            "max_all": stat_all["max"],
            "mean_all": stat_all["mean"],
            "median_all": stat_all["median"],
            "stdev_all": stat_all["stdev"],
            "min_nonzero": stat_nonzero["min"],
            "max_nonzero": stat_nonzero["max"],
            "mean_nonzero": stat_nonzero["mean"],
            "median_nonzero": stat_nonzero["median"],
            "stdev_nonzero": stat_nonzero["stdev"],
        })
    return results


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("No rows to write.")
        return
    # Exclude list/dict fields from CSV (we keep scalar summary fields only)
    keys = [k for k, v in rows[0].items() if not isinstance(v, (list, dict))]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            row = {k: r.get(k) for k in keys}
            writer.writerow(row)


def write_json(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)


def plot_synset_timeline(
    synsets_path: str,
    spans: List[Tuple[int, int]],
    out_html: str,
    top_k: int = 50,
    marker_min: int = 4,
    marker_max: int = 30,
    select_mode: str = "max",
    pos: Optional[str] = None,
    seed: Optional[int] = None,
):
    """Create a Plotly HTML where each line is a synset and marker size reflects synset size in each span.

    - Selects top_k synsets by maximum size across spans to avoid clutter.
    - Saves an interactive HTML file to `out_html`.
    """

    synsets_list = list(parse_jsonl(synsets_path))
    centers = [ (s + e) / 2.0 for s,e in spans ]
    max_year = spans[-1][1] if spans else 2026

    # compute sizes per synset
    synset_sizes: List[Tuple[Any, List[int], Dict[str, Any]]] = []
    for s in synsets_list:
        sid = s.get("synset_id")
        sizes = [synset_size_in_span(s, span_start, span_end, max_year=max_year) for span_start, span_end in spans]
        synset_sizes.append((sid, sizes, s))

    # select top_k according to select_mode
    if seed is not None:
        random.seed(seed)

    def score_item(item: Tuple[Any, List[int], Dict[str, Any]]) -> float:
        _, sizes, _ = item
        # max: choose synsets by their largest observed size across spans
        if select_mode == "max":
            return max(sizes) if sizes else 0
        # shrinking: score by decline = first_span_size - last_span_size (positive => shrinking)
        if select_mode == "shrinking":
            return (sizes[0] if sizes else 0) - (sizes[-1] if sizes else 0)
        # variance: prefer synsets whose sizes vary the most across spans
        if select_mode == "variance":
            try:
                return statistics.pvariance(sizes) if len(sizes) > 1 else 0
            except Exception:
                return 0
        # growth: prefer synsets that increased (last - first)
        if select_mode == "growth":
            return (sizes[-1] if sizes else 0) - (sizes[0] if sizes else 0)
        # random: random score for stochastic selection
        if select_mode == "random":
            return random.random()
        # fallback: treat like 'max'
        return max(sizes) if sizes else 0

    if select_mode == "random":
        random.shuffle(synset_sizes)
        selected = synset_sizes[:top_k]
    else:
        synset_sizes_sorted = sorted(synset_sizes, key=score_item, reverse=True)
        selected = synset_sizes_sorted[:top_k]

    # determine scaling for marker sizes
    all_sizes = [sz for _, sizes, _ in selected for sz in sizes]
    max_size = max(all_sizes) if all_sizes else 0

    def scale(sz: int) -> float:
        if max_size <= 0:
            return marker_min
        return marker_min + (float(sz) / max_size) * (marker_max - marker_min)

    traces = []
    y_labels = []
    for idx, (sid, sizes, syn) in enumerate(reversed(selected)):
        # reversed so largest is at top
        marker_sizes = [scale(int(v)) for v in sizes]
        # hover text: lemmas active in each span
        hover = []
        for (span_start, span_end), v in zip(spans, sizes):
            # collect active lemmas and their definitions for this span
            active_entries = []
            for l in syn.get("lus", []):
                if lemma_active_in_span(l, span_start, span_end, max_year):
                    lemma = l.get("lemma", "")
                    ddef = l.get("def", "") or l.get("definition", "")
                    ddef_snip = " ".join(str(ddef).split())[:140]
                    active_entries.append(f"{lemma}: {ddef_snip}")
            # show up to 6 active lemma+definition entries, each on its own line in hover
            lemmas_text = "<br>".join(active_entries[:6]) if active_entries else "(none)"
            hover.append(f"id: {sid}<br>size: {int(v)}<br>lemmas:<br>{lemmas_text}")

        # create traces only for contiguous non-zero segments so lines start when synset appears
        n = len(centers)
        i = 0
        while i < n:
            # skip zeros
            if sizes[i] <= 0:
                i += 1
                continue
            j = i
            while j + 1 < n and sizes[j + 1] > 0:
                j += 1
            # extend the segment so the line starts at the span start and ends at span end
            seg_centers = centers[i : j + 1]
            seg_start = spans[i][0]
            seg_end = spans[j][1]
            # x points: [start, center_i, center_{i+1}, ..., center_j, end]
            x_points = [seg_start] + seg_centers + [seg_end]
            seg_y = [idx] * len(x_points)
            # marker sizes: no marker at boundaries
            seg_marker_sizes = [0] + marker_sizes[i : j + 1] + [0]
            # hover: empty for boundary points
            seg_hover = [""] + hover[i : j + 1] + [""]
            traces.append(
                go.Scatter(
                    x=x_points,
                    y=seg_y,
                    mode="lines+markers",
                    marker=dict(size=seg_marker_sizes, sizemode="area", opacity=0.9),
                    line=dict(width=1),
                    hoverinfo="text",
                    text=seg_hover,
                    name=str(sid),
                    showlegend=False,
                )
            )
            i = j + 1

        lemmas_summary = ", ".join([l.get("lemma", "") for l in syn.get("lus", [])[:6]])
        label = f"{sid}: {lemmas_summary[:60]}"
        y_labels.append(label)

    height = max(300, 20 * len(selected))
    # label ticks at span starts (dashed vertical lines) to avoid clutter
    span_starts = [s for s, _ in spans]
    x_ticktext = [str(int(s)) for s in span_starts]
    shapes = []
    for x in span_starts:
        shapes.append(
            dict(
                type="line",
                x0=x,
                x1=x,
                y0=-1,
                y1=len(selected),
                line=dict(color="gray", width=1, dash="dash"),
            )
        )

    layout = go.Layout(
        title="Synset size timeline",
        xaxis=dict(title="Year", tickmode="array", tickvals=span_starts, ticktext=x_ticktext),
        yaxis=dict(title="Synset", tickmode="array", tickvals=list(range(len(selected))), ticktext=y_labels),
        height=height,
        hovermode="closest",
        margin=dict(l=200),
        paper_bgcolor="white",
        plot_bgcolor="white",
        shapes=shapes,
    )

    title_prefix = f"{pos} " if pos else ""
    fig = go.Figure(data=traces, layout=layout)
    fig.update_layout(title=f"{title_prefix}Synset size timeline (method={select_mode})")
    # write HTML
    pyo.plot(fig, filename=out_html, auto_open=False)



def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Compute synset-size statistics across time spans")
    p.add_argument("--input", required=True, help="Input synsets JSONL file")
    p.add_argument("--start", type=int, required=True, help="Start year (inclusive)")
    p.add_argument("--end", type=int, required=True, help="End year (inclusive)")
    p.add_argument("--step", type=int, default=50, help="Span width in years (default 50)")
    p.add_argument("--max-year", type=int, default=2026, help="Year to substitute for null `end` (default: 2026)")
    p.add_argument("--out", help="CSV output path (optional). If omitted prints CSV to stdout.")
    p.add_argument("--summary-out", help="Write JSON summary with lists (top/bottom/appeared/disappeared)")
    p.add_argument("--sample-n", type=int, default=5, help="Number of top/bottom samples per span (default 5)")
    p.add_argument("--include-zero", action="store_true", help="Include zero-sized synsets when selecting smallest samples")
    p.add_argument("--print", dest="print_details", action="store_true", help="Print human-readable per-span details to stdout")
    p.add_argument("--plot-out", help="Write interactive Plotly HTML timeline")
    p.add_argument("--plot-top-k", type=int, default=60, help="Top-K synsets to plot (default 60)")
    p.add_argument("--plot-marker-min", type=int, default=5, help="Minimum marker size for plotting")
    p.add_argument("--plot-marker-max", type=int, default=100, help="Maximum marker size for plotting")
    p.add_argument(
        "--plot-select",
        choices=["max", "random", "shrinking", "variance", "growth"],
        default="max",
        help="Selection mode for synsets to plot: max (largest), random, shrinking (first-last), variance, growth (last-first)",
    )
    p.add_argument("--plot-seed", type=int, default=42, help="Random seed (deterministic random selection)")
    p.add_argument("--plot-all", action="store_true", help="Generate plots for all selection modes and save them together")
    args = p.parse_args(argv)

    spans = make_spans(args.start, args.end, args.step)
    synsets = parse_jsonl(args.input)
    results = aggregate_stats_per_span(synsets, spans, args.max_year, sample_n=args.sample_n, include_zero=args.include_zero)

    # Output: create ../data/syns_evolution and save everything there
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.abspath(os.path.join(script_dir, "..", "data", "syns_evolution"))
    os.makedirs(out_dir, exist_ok=True)

    # CSV: always write into out_dir (use basename if user provided a name)
    csv_out = os.path.join(out_dir, os.path.basename(args.out)) if args.out else os.path.join(out_dir, "stats.csv")
    write_csv(csv_out, results)
    print(f"Wrote CSV to {csv_out}")

    # JSON summary
    summary_out = os.path.join(out_dir, os.path.basename(args.summary_out)) if args.summary_out else os.path.join(out_dir, "summary.json")
    write_json(summary_out, results)
    print(f"Wrote JSON summary to {summary_out}")

    # print human readable details if requested
    if args.print_details:
        for r in results:
            print(f"Span {r['span_start']}-{r['span_end']}: total={r['synsets_total']} with_any={r['synsets_with_any']}")
            print("  Top:")
            for t in r.get("top_synsets", []):
                print(f"    {t['synset_id']} (size={t['size']})")
            print("  Bottom:")
            for b in r.get("bottom_synsets", []):
                print(f"    {b['synset_id']} (size={b['size']})")
            print(f"  Appeared: {r['synsets_appeared']}")
            print(f"  Disappeared: {r['synsets_disappeared']}")

    # plotting: always write into out_dir; infer POS from input filename
    basename = os.path.basename(args.input)
    pos = basename.split("_")[0] if "_" in basename else None

    modes = [args.plot_select]
    if args.plot_all:
        modes = ["max", "random", "shrinking", "variance", "growth"]

    for mode in modes:
        if args.plot_out:
            out_base, out_ext = os.path.splitext(os.path.basename(args.plot_out))
            suffix = f"_{mode}"
            if suffix not in out_base:
                plot_name = f"{out_base}{suffix}{out_ext}"
            else:
                plot_name = f"{out_base}{out_ext}"
        else:
            plot_name = f"synset_timeline_{mode}.html"

        plot_path = os.path.join(out_dir, plot_name)
        plot_synset_timeline(
            args.input,
            spans,
            plot_path,
            top_k=args.plot_top_k,
            marker_min=args.plot_marker_min,
            marker_max=args.plot_marker_max,
            select_mode=mode,
            pos=pos,
            seed=args.plot_seed,
        )
        print(f"Wrote plot to {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
