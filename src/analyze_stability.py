import os
import re
import glob
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PLOTS_FOLDER = "../data/plots"


# We keep only change categories for the trend plot, grouped for color families.
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


GROUP_COLORS_FINE: Dict[str, Dict[str, str]] = {
    "Differentiation": {
        "Lost": "#DE73E4",
        "Split": "#ed3c3c",
    },
    "No Change": {
        "No Change": "#5e5c5c",
    },
    "Parallel Change": {
        "Parallel Addition": "#155533",
        "Parallel Loss": "#3a9561",
        "Parallel Addition + Loss": "#74f2a6",
    },
    "Non-Parallel Change": {
        "Addition": "#D6690F",
        "Loss": "#d7d71b",
        "Addition + Loss": "#a04f04",
    },
}

GROUP_COLORS_HIGHER: Dict[str, Dict[str, str]] = {
    "Differentiation": "#700606",
    "No Change": "#5e5c5c",
    "Parallel Change": "#155533",
    "Non-Parallel Change": "#143B5D"
}



def parse_batch_summary(input_path: str) -> pd.DataFrame:
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

    return pd.DataFrame(records)


def save_xlsx(df: pd.DataFrame, output_path: str) -> None:
    # Keep one full sheet plus a compact span-level summary sheet.
    trend_df = df[df["Category"].isin(CATEGORY_GROUPS.keys())].copy()
    trend_summary = (
        trend_df.groupby(["POS", "Span", "Category"], as_index=False)["Percentage"]
        .mean()
        .sort_values(["POS", "Span", "Category"])
    )

    with pd.ExcelWriter(output_path) as writer:
        df.sort_values(["POS", "Threshold", "T1", "Category"]).to_excel(
            writer, sheet_name="all_time_ranges", index=False
        )
        trend_summary.to_excel(writer, sheet_name="trend_summary", index=False)


def save_trend_plot(df: pd.DataFrame, output_path: str) -> None:
    trend_df = df[df["Category"].isin(CATEGORY_GROUPS.keys())].copy()
    summary = (
        trend_df.groupby(["Threshold", "POS", "Span", "Category"])["Percentage"]
        .agg(['mean', 'std'])
        .reset_index()
        .sort_values(["Threshold", "POS", "Span", "Category"])
    )

    def hex_to_rgba(hex_color: str, alpha: float) -> str:
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            return f"rgba({int(hex_color[0:2], 16)}, {int(hex_color[2:4], 16)}, {int(hex_color[4:6], 16)}, {alpha})"
        return hex_color

    preferred_pos_order = ["ADJ", "NOUN", "VERB"]
    present_pos = summary["POS"].unique().tolist()
    pos_list = [p for p in preferred_pos_order if p in present_pos] + [p for p in present_pos if p not in preferred_pos_order]

    thresholds = sorted(summary["Threshold"].unique().tolist())

    subplot_titles: List[str] = []
    for threshold in thresholds:
        for pos in pos_list:
            subplot_titles.append(f"<b>{pos}</b>")

    fig = make_subplots(
        rows=max(len(thresholds), 1),
        cols=max(len(pos_list), 1),
        subplot_titles=subplot_titles if subplot_titles else ["No data"],
        # shared_yaxes=True,
        # shared_xaxes=True,
    )

    for row_idx, threshold in enumerate(thresholds, start=1):
        thr_data = summary[summary["Threshold"] == threshold]
        for col_idx, pos in enumerate(pos_list, start=1):
            pos_data = thr_data[thr_data["POS"] == pos]

            for category in CATEGORY_GROUPS.keys():
                cat_data = pos_data[pos_data["Category"] == category]
                group = CATEGORY_GROUPS[category]
                color = GROUP_COLORS_FINE[group][category]
                
                x_vals = list(cat_data["Span"])
                y_mean = list(cat_data["mean"])
                y_std = list(cat_data["std"].fillna(0))
                
                y_upper = [m + s for m, s in zip(y_mean, y_std)]
                y_lower = [max(0, m - s) for m, s in zip(y_mean, y_std)]

                fig.add_trace(
                    go.Scatter(
                        x=x_vals + x_vals[::-1],
                        y=y_upper + y_lower[::-1],
                        fill='toself',
                        fillcolor=hex_to_rgba(color, 0.2),
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip",
                        showlegend=False,
                        legendgroup=category,
                    ),
                    row=row_idx,
                    col=col_idx,
                )

                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_mean,
                        mode="lines",
                        name=f"{category} ({group})",
                        legendgroup=category,
                        line=dict(color=color, width=3),
                        showlegend=(row_idx == 1 and col_idx == 1),
                    ),
                    row=row_idx,
                    col=col_idx,
                )

    fig.for_each_annotation(
        lambda a: a.update(
            font=dict(size=26)
        )
    )

    fig.update_xaxes(
        tickfont=dict(size=18),
        title_font=dict(size=22),
    )

    fig.update_yaxes(
        tickfont=dict(size=18),
        title_font=dict(size=22),
    )

    fig.update_layout(
        yaxis_title_font=dict(size=26),
        yaxis_title="<b>Percentage (%)</b>",
        template="plotly_white",
        # legend_title="Category",
        autosize=False,
        height=max(650 * len(thresholds), 800),
        width=max(700 * len(pos_list), 1500),
        showlegend=True,
        margin=dict(
            l=180,
            r=80,
            t=120,
            b=300
        ),
        legend=dict(
            font=dict(size=18),
            orientation="h",
            yanchor="top",
            y=-0.30,
            xanchor="center",
            x=0.5
        ),
        font=dict(size=20),
    )

    # ===== GLOBAL X AXIS TITLE =====
    fig.add_annotation(
        text="<b>Time Span Length (Years)</b>",
        x=0.5,
        y=-0.18,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=26),
        xanchor="center",
    )

    fig.write_html(output_path)


def save_stacked_trend_plot(df: pd.DataFrame, output_path: str) -> None:
    trend_df = df[df["Category"].isin(CATEGORY_GROUPS.keys())].copy()
    summary = (
        trend_df.groupby(["Threshold", "POS", "Span", "Category"], as_index=False)["Percentage"]
        .mean()
        .sort_values(["Threshold", "POS", "Span", "Category"])
    )

    preferred_pos_order = ["ADJ", "NOUN", "VERB"]
    present_pos = summary["POS"].unique().tolist()
    pos_list = [p for p in preferred_pos_order if p in present_pos] + [p for p in present_pos if p not in preferred_pos_order]

    thresholds = sorted(summary["Threshold"].unique().tolist())

    subplot_titles: List[str] = []
    for threshold in thresholds:
        for pos in pos_list:
            subplot_titles.append(f"<b>{pos}</b>")

    fig = make_subplots(
        rows=max(len(thresholds), 1),
        cols=max(len(pos_list), 1),
        subplot_titles=subplot_titles if subplot_titles else ["No data"],
        shared_yaxes=True
    )

    for row_idx, threshold in enumerate(thresholds, start=1):
        thr_data = summary[summary["Threshold"] == threshold]
        for col_idx, pos in enumerate(pos_list, start=1):
            pos_data = thr_data[thr_data["POS"] == pos]

            for category in CATEGORY_GROUPS.keys():
                cat_data = pos_data[pos_data["Category"] == category]
                group = CATEGORY_GROUPS[category]
                color = GROUP_COLORS_FINE[group][category]

                fig.add_trace(
                    go.Scatter(
                        x=cat_data["Span"],
                        y=cat_data["Percentage"],
                        mode='lines',
                        line=dict(width=0.5, color='white'),
                        stackgroup='one', # this is the key for stacking
                        name=category,
                        fillcolor=color,
                        showlegend=False,
                    ),
                    row=row_idx,
                    col=col_idx,
                )

    fig.update_layout(
        yaxis_title_font=dict(size=26),
        template="plotly_white",
        height=max(450 * len(thresholds), 600),
        width=1500,
        yaxis=dict(range=[0, 100]),
    )

    # ===== GLOBAL X AXIS TITLE =====
    fig.add_annotation(
        text="<b>Time Span Length (Years)</b>",
        x=0.5,
        y=-0.15,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=26),
        xanchor="center",
    )

    fig.write_html(output_path)


def save_stacked_trend_plot_higher(
    df: pd.DataFrame,
    output_path: str
) -> None:

    # =========================================================
    # FILTER + AGGREGATE
    # =========================================================

    trend_df = df[df["Category"].isin(CATEGORY_GROUPS.keys())].copy()

    summary = (
        trend_df
        .groupby(
            ["Threshold", "POS", "Span", "Category"],
            as_index=False
        )["Percentage"]
        .mean()
    )

    summary["Group"] = summary["Category"].map(CATEGORY_GROUPS)

    group_summary = (
        summary
        .groupby(
            ["Threshold", "POS", "Span", "Group"]
        )["Percentage"]
        .sum()
        .reset_index()
    )

    # =========================================================
    # ORDERING
    # =========================================================

    preferred_pos_order = ["ADJ", "NOUN", "VERB"]

    present_pos = group_summary["POS"].unique().tolist()

    pos_list = (
        [p for p in preferred_pos_order if p in present_pos]
        + [p for p in present_pos if p not in preferred_pos_order]
    )

    thresholds = sorted(
        group_summary["Threshold"].unique().tolist()
    )

    # =========================================================
    # SUBPLOT TITLES
    # =========================================================

    subplot_titles: List[str] = []

    for threshold in thresholds:
        for pos in pos_list:
            subplot_titles.append(f"<b>{pos}</b>")

    # =========================================================
    # DYNAMIC FIGURE SIZE
    # IMPORTANT FOR HTML -> PDF PIPELINES
    # =========================================================

    rows = max(len(thresholds), 1)
    cols = max(len(pos_list), 1)

    width = 700 * cols
    height = 500 * rows

    # =========================================================
    # CREATE SUBPLOTS
    # =========================================================

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles if subplot_titles else ["No data"],
        shared_yaxes=True,

        horizontal_spacing=0.08,
        vertical_spacing=0.12
    )

    # =========================================================
    # ADD STACKED TRACES
    # =========================================================

    for row_idx, threshold in enumerate(thresholds, start=1):

        thr_data = group_summary[
            group_summary["Threshold"] == threshold
        ]

        for col_idx, pos in enumerate(pos_list, start=1):

            pos_data = thr_data[
                thr_data["POS"] == pos
            ]

            for group, color in GROUP_COLORS_HIGHER.items():

                group_data = pos_data[
                    pos_data["Group"] == group
                ]

                fig.add_trace(
                    go.Scatter(
                        x=group_data["Span"],
                        y=group_data["Percentage"],

                        mode="lines",

                        line=dict(
                            width=0.5,
                            color="white"
                        ),

                        stackgroup="one",

                        fillcolor=color,

                        name=group,

                        showlegend=(
                            row_idx == 1 and col_idx == 1
                        ),
                    ),

                    row=row_idx,
                    col=col_idx,
                )

    # =========================================================
    # FONT SIZES
    # =========================================================

    fig.for_each_annotation(
        lambda a: a.update(
            font=dict(size=26)
        )
    )

    fig.update_xaxes(
        tickfont=dict(size=18),
        title_font=dict(size=22),
        automargin=True
    )

    fig.update_yaxes(
        tickfont=dict(size=18),
        title_font=dict(size=22),

        range=[0, 100],

        automargin=True
    )

    # =========================================================
    # LAYOUT
    # =========================================================

    fig.update_layout(

        # IMPORTANT
        autosize=False,

        # EXPLICIT SIZE
        width=width,
        height=height,

        # BACKGROUNDS
        paper_bgcolor="white",
        plot_bgcolor="white",

        # MARGINS
        margin=dict(
            l=180,
            r=80,
            t=120,
            b=140
        ),

        # LEGEND
        legend=dict(
            font=dict(size=18),
            orientation="h",
            yanchor="bottom",
            y=-0.45,
            xanchor="center",
            x=0.5
        ),

        # GLOBAL FONT
        font=dict(
            size=20
        )
    )

    # =========================================================
    # GLOBAL X LABEL
    # IMPORTANT:
    # keep annotations INSIDE paper
    # =========================================================

    fig.add_annotation(
        text="<b>Time Span Length (Years)</b>",

        x=0.5,
        y=-0.30,

        xref="paper",
        yref="paper",

        showarrow=False,

        font=dict(size=28),

        xanchor="center",
    )

    # =========================================================
    # GLOBAL Y LABEL
    # =========================================================

    fig.add_annotation(
        text="<b>Percentage (%)</b>",

        x=-0.06,
        y=0.5,

        xref="paper",
        yref="paper",

        showarrow=False,

        textangle=-90,

        font=dict(size=28),

        xanchor="center",
        yanchor="middle"
    )

    # =========================================================
    # EXPORT HTML
    # IMPORTANT FOR PDF CONVERSION
    # =========================================================

    fig.write_html(
        output_path,

        full_html=True,

        include_plotlyjs="cdn",

        config={
            "responsive": False
        }
    )


def save_combined_trend_plot(
    df: pd.DataFrame,
    sig_df: pd.DataFrame,
    output_path: str
) -> None:

    main_df = df[df["condition"] == "main"].copy()
    control_df = df[df["condition"] == "control"].copy()

    # =========================================================
    # SUMMARIES
    # =========================================================

    main_summary = (
        main_df[main_df["Category"].isin(CATEGORY_GROUPS.keys())]
        .groupby(
            ["Threshold", "POS", "Span", "Category"],
            as_index=False
        )["Percentage"]
        .mean()
    )

    control_summary = (
        control_df[control_df["Category"].isin(CATEGORY_GROUPS.keys())]
        .groupby(
            ["Threshold", "POS", "Span", "Category"],
            as_index=False
        )["Percentage"]
        .mean()
    )

    # =========================================================
    # SIGNIFICANCE
    # =========================================================

    sig_cols = [
        "POS",
        "Span",
        "Category",
        "significant",
        "stronger_in"
    ]

    main_summary = pd.merge(
        main_summary,
        sig_df[sig_cols],
        on=["POS", "Span", "Category"],
        how="left"
    )

    control_summary = pd.merge(
        control_summary,
        sig_df[sig_cols],
        on=["POS", "Span", "Category"],
        how="left"
    )

    main_summary["significant"] = (
        main_summary["significant"].fillna(False)
    )

    control_summary["significant"] = (
        control_summary["significant"].fillna(False)
    )

    # =========================================================
    # POS ORDER
    # =========================================================

    preferred_pos_order = ["ADJ", "NOUN", "VERB"]

    present_pos = (
        main_summary["POS"]
        .unique()
        .tolist()
    )

    pos_list = (
        [p for p in preferred_pos_order if p in present_pos]
        + [p for p in present_pos if p not in preferred_pos_order]
    )

    thresholds = sorted(
        main_summary["Threshold"]
        .unique()
        .tolist()
    )

    # =========================================================
    # GROUPS
    # =========================================================

    group_order = [
        "Differentiation",
        "No Change",
        "Parallel Change",
        "Non-Parallel Change"
    ]

    group_to_col = {
        group: i + 1
        for i, group in enumerate(group_order)
    }

    row_titles = [
        f"{pos}"
        for thr in thresholds
        for pos in pos_list
    ]

    categories = list(CATEGORY_GROUPS.keys())

    # =========================================================
    # FIGURE
    # =========================================================

    fig = make_subplots(
        rows=len(row_titles),
        cols=len(group_order),
        shared_xaxes=True,
        shared_yaxes=False,
        vertical_spacing=0.02,
    )

    # =========================================================
    # PLOTS
    # =========================================================

    row_idx = 1

    for threshold in thresholds:

        for pos in pos_list:

            for category in categories:

                group = CATEGORY_GROUPS[category]

                color = GROUP_COLORS_FINE[group][category]

                col_idx = group_to_col[group]

                # =====================================================
                # MAIN DATA
                # =====================================================

                main_cat_data = main_summary[
                    (main_summary["Threshold"] == threshold)
                    & (main_summary["POS"] == pos)
                    & (main_summary["Category"] == category)
                ]

                fig.add_trace(
                    go.Scatter(
                        x=main_cat_data["Span"],
                        y=main_cat_data["Percentage"],

                        mode="lines",

                        name=category,

                        line=dict(
                            color=color,
                            width=2
                        ),

                        # only show once in legend
                        showlegend=(row_idx == 1),
                    ),
                    row=row_idx,
                    col=col_idx,
                )

                sig_main_data = main_cat_data[
                    main_cat_data["significant"]
                    & (
                        main_cat_data["stronger_in"]
                        == "Main"
                    )
                ]

                fig.add_trace(
                    go.Scatter(
                        x=sig_main_data["Span"],
                        y=sig_main_data["Percentage"],

                        mode="markers",

                        marker=dict(
                            size=6,
                            color=color,
                            symbol="star"
                        ),

                        showlegend=False,
                    ),
                    row=row_idx,
                    col=col_idx,
                )

                # =====================================================
                # CONTROL DATA
                # =====================================================

                control_cat_data = control_summary[
                    (control_summary["Threshold"] == threshold)
                    & (control_summary["POS"] == pos)
                    & (control_summary["Category"] == category)
                ]

                fig.add_trace(
                    go.Scatter(
                        x=control_cat_data["Span"],
                        y=control_cat_data["Percentage"],

                        mode="lines",

                        line=dict(
                            color=color,
                            width=2,
                            dash="dash"
                        ),

                        opacity=0.7,

                        # no legend for control traces
                        showlegend=False,
                    ),
                    row=row_idx,
                    col=col_idx,
                )

                sig_control_data = control_cat_data[
                    control_cat_data["significant"]
                    & (
                        control_cat_data["stronger_in"]
                        == "Control"
                    )
                ]

                fig.add_trace(
                    go.Scatter(
                        x=sig_control_data["Span"],
                        y=sig_control_data["Percentage"],

                        mode="markers",

                        marker=dict(
                            size=6,
                            color=color,
                            symbol="circle"
                        ),

                        showlegend=False,
                    ),
                    row=row_idx,
                    col=col_idx,
                )

            row_idx += 1

    # =========================================================
    # AXES STYLE
    # =========================================================

    fig.update_xaxes(
        tickfont=dict(size=18)
    )

    fig.update_yaxes(
        tickfont=dict(size=18)
    )

    # =========================================================
    # LAYOUT
    # =========================================================

    fig.update_layout(
        template="plotly_white",

        width=1600,

        height=max(
            300 * len(row_titles),
            600
        ),

        margin=dict(
            l=220,
            r=80,
            t=220,
            b=220
        ),

        font=dict(size=18),

        showlegend=True,

        legend=dict(
            orientation="h",

            x=0.5,
            xanchor="center",

            y=-0.21,
            yanchor="top",

            font=dict(size=18),
        ),
    )

    # =========================================================
    # GLOBAL X TITLE
    # =========================================================

    fig.add_annotation(
        text="<b>Time Span Length (Years)</b>",

        x=0.5,
        y=-0.15,

        xref="paper",
        yref="paper",

        showarrow=False,

        font=dict(size=22),

        xanchor="center",
    )

    # =========================================================
    # GLOBAL Y TITLE
    # =========================================================

    fig.add_annotation(
        text="<b>Percentage (%)</b>",

        x=-0.10,
        y=0.5,

        xref="paper",
        yref="paper",

        showarrow=False,

        textangle=-90,

        font=dict(size=22),

        yanchor="middle",
    )

    # =========================================================
    # COLUMN TITLES
    # =========================================================

    for col_idx, title in enumerate(group_order, start=1):

        xaxis_name = (
            "xaxis"
            if col_idx == 1
            else f"xaxis{col_idx}"
        )

        domain = fig.layout[xaxis_name].domain

        x_center = (
            domain[0] + domain[1]
        ) / 2

        fig.add_annotation(
            x=x_center,
            y=1.08,

            xref="paper",
            yref="paper",

            text=(
                f"<b>"
                f"<span style='color:{GROUP_COLORS_HIGHER[title]}'>"
                f"{title}"
                f"</span>"
                f"</b>"
            ),

            showarrow=False,

            font=dict(size=22),

            xanchor="center",
            yanchor="bottom",
        )

    # =========================================================
    # ROW TITLES
    # =========================================================

    n_rows = len(row_titles)

    for i, title in enumerate(row_titles):

        y_pos = 1 - (
            (i + 0.5) / n_rows
        )

        fig.add_annotation(
            x=-0.05,
            y=y_pos,

            xref="paper",
            yref="paper",

            text=f"<b>{title}</b>",

            showarrow=False,

            textangle=-90,

            font=dict(size=22),

            xanchor="center",
            yanchor="middle",
        )

    # =========================================================
    # SAVE
    # =========================================================

    fig.write_html(output_path)


def save_pos_stacked_comparison_plot(df: pd.DataFrame, output_folder: str) -> None:
    """
    Generates a 2x2 stacked area plot for each POS.
    - col 1: synonym group (main)
    - col 2: control group
    - row 1: higher-level categories
    - row 2: fine-grained categories
    """
    main_df = df[df["condition"] == "main"].copy()
    control_df = df[df["condition"] == "control"].copy()

    main_summary = (
        main_df[main_df["Category"].isin(CATEGORY_GROUPS.keys())]
        .groupby(["POS", "Span", "Category"], as_index=False)["Percentage"]
        .mean()
    )
    control_summary = (
        control_df[control_df["Category"].isin(CATEGORY_GROUPS.keys())]
        .groupby(["POS", "Span", "Category"], as_index=False)["Percentage"]
        .mean()
    )

    summaries = {"main": main_summary, "control": control_summary}

    preferred_pos_order = ["ADJ", "NOUN", "VERB"]
    present_pos = main_df["POS"].unique().tolist()
    pos_list = [p for p in preferred_pos_order if p in present_pos] + [p for p in present_pos if p not in preferred_pos_order]

    for pos in pos_list:
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "<b>Synonym Group (Combined)</b>",
                "<b>Control Group (Combined)</b>",
                "<b>Synonym Group (Fine-Grained)</b>",
                "<b>Control Group (Fine-Grained)</b>",
            ],
            shared_xaxes=True,
            shared_yaxes=False,
            vertical_spacing=0.1,
        )

        # Plot higher-level (row 1)
        for col_idx, condition in enumerate(["main", "control"], start=1):
            summary = summaries[condition]
            pos_data = summary[summary["POS"] == pos].copy()
            pos_data["Group"] = pos_data["Category"].map(CATEGORY_GROUPS)
            
            group_summary = pos_data.groupby(["Span", "Group"])["Percentage"].sum().reset_index()

            for group, color in GROUP_COLORS_HIGHER.items():
                group_data = group_summary[group_summary["Group"] == group]
                fig.add_trace(
                    go.Scatter(
                        x=group_data["Span"],
                        y=group_data["Percentage"],
                        mode='lines',
                        line=dict(width=0.5, color='white'),
                        stackgroup=f'one-{condition}-higher',
                        name=group,
                        fillcolor=color,
                        showlegend=(col_idx==1),
                    ),
                    row=1,
                    col=col_idx,
                )

        # Plot fine-grained (row 2)
        for col_idx, condition in enumerate(["main", "control"], start=1):
            summary = summaries[condition]
            pos_data = summary[summary["POS"] == pos]

            for category in CATEGORY_GROUPS.keys():
                cat_data = pos_data[pos_data["Category"] == category]
                group = CATEGORY_GROUPS[category]
                color = GROUP_COLORS_FINE[group][category]

                fig.add_trace(
                    go.Scatter(
                        x=cat_data["Span"],
                        y=cat_data["Percentage"],
                        mode='lines',
                        line=dict(width=0.5, color='white'),
                        stackgroup=f'one-{condition}-fine',
                        name=category,
                        fillcolor=color,
                        showlegend=(col_idx==1),
                    ),
                    row=2,
                    col=col_idx,
                )

        fig.update_layout(
            template="plotly_white",
            height=800,
            width=1200,
            legend_title="Category",
        )
        fig.update_yaxes(title="<b>Percentage (%)</b>", range=[0, 100])
        fig.update_xaxes(title="<b>Time Span Length (Years)</b>", row=2)

        output_path = os.path.join(output_folder, f"stacked_comparison_{pos}.html")
        fig.write_html(output_path)
        print(f"Saved POS stacked comparison plot: {output_path}")


def save_comparison_summary(output_path: str) -> None:
    """
    Generates an Excel file comparing lemma-wise and pairwise results
    for both main and control conditions.
    """
    plots_folder = PLOTS_FOLDER
    file_mapping = {
        "pairwise_main": os.path.join(plots_folder, "batch_summary_tables.txt"),
        "pairwise_control": os.path.join(plots_folder, "batch_summary_tables_control.txt"),
        "lemmawise_main": os.path.join(plots_folder, "batch_summary_tables_lemma_syn.txt"),
        "lemmawise_control": os.path.join(plots_folder, "batch_summary_tables_lemma_non-syn.txt"),
    }

    all_dfs = []
    for name, path in file_mapping.items():
        if not os.path.exists(path):
            print(f"Warning: File not found, skipping: {path}")
            continue

        analysis_type, condition = name.split("_")
        df = parse_batch_summary(path)
        df["analysis_type"] = analysis_type.replace("_", " ")
        df["condition"] = condition
        all_dfs.append(df)

    if not all_dfs:
        print("Error: No summary files found to generate comparison. Aborting.")
        return

    full_df = pd.concat(all_dfs, ignore_index=True)

    # We only care about the shared laws
    full_df = full_df[full_df["Category"].isin(CATEGORY_GROUPS.keys())]

    # Average percentages over different time periods, grouping by span
    summary = (
        full_df.groupby(["POS", "Span", "Category", "analysis_type", "condition"], as_index=False)[
            "Percentage"
        ]
        .mean()
    )

    # Pivot to get the desired structure
    pivot_df = summary.pivot_table(
        index=["POS", "Span", "Category"],
        columns=["condition", "analysis_type"],
        values="Percentage",
    )

    # Reorder columns to match request: main/control then lemma/pairwise
    pivot_df = pivot_df.reorder_levels([1, 0], axis=1).sort_index(axis=1)
    
    # Flatten multi-index columns for better readability in Excel
    pivot_df.columns = [f"{c1}_{c2}".replace(" ", "_") for c1, c2 in pivot_df.columns]


    pivot_df.to_excel(output_path)
    print(f"Comparison summary saved to: {output_path}")


def save_latex_summary(df: pd.DataFrame, output_path: str) -> None:
    """
    Generates a single LaTeX file with one table summarizing mean and std dev of percentages
    by POS, Span, and Category using longtable.
    """
    trend_df = df[df["Category"].isin(CATEGORY_GROUPS.keys())].copy()

    category_to_macro = {
        "Lost": r"\binlost",
        "Split": r"\binsplit",
        "No Change": r"\binnochg",
        "Parallel Addition": r"\binpadd",
        "Parallel Loss": r"\binploss",
        "Parallel Addition + Loss": r"\binpaddloss",
        "Addition": r"\binadd",
        "Loss": r"\binloss",
        "Addition + Loss": r"\binaddloss",
    }

    # Calculate mean and std dev
    summary = trend_df.groupby(["POS", "Span", "Category"])["Percentage"].agg(['mean', 'std']).reset_index()
    summary['mean'] = summary['mean'].round(2)
    summary['std'] = summary['std'].round(2)

    # Sort by POS, then Span, then custom Category order
    category_order = list(category_to_macro.keys())
    summary['Category'] = pd.Categorical(summary['Category'], categories=category_order, ordered=True)
    summary = summary.sort_values(["POS", "Span", "Category"])

    latex_string = ""

    # Use @{\extracolsep{\fill}} to make the table take up the full text width
    latex_string += r"\begin{longtable}{@{\extracolsep{\fill}}l l l r r@{}}" + "\n"

    # --- Table Header ---
    latex_string += r"\toprule" + "\n"
    latex_string += r"\textbf{POS} & \textbf{Time Span Length} & \textbf{Category} & \textbf{Mean (\%)} & \textbf{Std Dev (\%)} \\" + "\n"
    latex_string += r"\midrule" + "\n"
    latex_string += r"\endfirsthead" + "\n"
    
    latex_string += r"\toprule" + "\n"
    latex_string += r"\textbf{POS} & \textbf{Time Span Length} & \textbf{Category} & \textbf{Mean (\%)} & \textbf{Std Dev (\%)} \\" + "\n"
    latex_string += r"\midrule" + "\n"
    latex_string += r"\endhead" + "\n"
    latex_string += r"\endfoot" + "\n"
    
    # --- Table Body ---
    last_pos = None
    last_span = None
    for _, row in summary.iterrows():
        pos_label = row['POS'] if row['POS'] != last_pos else ""
        span_label = str(row['Span']) if row['Span'] != last_span or row['POS'] != last_pos else ""
        
        category_macro = category_to_macro.get(row['Category'], row['Category'])
        
        latex_string += f"{pos_label} & {span_label} & {category_macro} & {row['mean']} & {row['std']} \\\\\n"
        
        last_pos = row['POS']
        last_span = row['Span']

        # Add a midrule after each group of spans, but not if the POS is also changing
        current_index = summary.index.get_loc(_)
        is_last_row = current_index == len(summary) - 1

        if not is_last_row:
            next_row = summary.iloc[current_index + 1]
            is_changing_pos = next_row['POS'] != row['POS']
            is_changing_span = next_row['Span'] != row['Span']

            if is_changing_pos:
                latex_string += r"\bottomrule" + "\n"
                latex_string += r"\addlinespace" + "\n"
            elif is_changing_span:
                latex_string += r"\cmidrule{2-5}" + "\n"

    latex_string += r"\bottomrule" + "\n"
    latex_string += r"\caption{Full breakdown of mean and standard deviation of change percentages by POS, time span length, and taxonomy category, accompanying Figure~\ref{fig:main-results-pairwise}. Mean is the average percentage across all bins of the same time span length for each category.}" + "\n"
    latex_string += r"\label{tab:stability_summary}\\" + "\n"

    latex_string += r"\end{longtable}" + "\n\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_string)
    print(f"LaTeX summary saved to: {output_path}")


def main() -> None:
    os.makedirs(PLOTS_FOLDER, exist_ok=True)

    # --- 1. Generate the specific Main vs. Control Combined Plot ---
    main_input_path = os.path.join(PLOTS_FOLDER, "batch_summary_tables.txt")
    control_input_path = os.path.join(PLOTS_FOLDER, "batch_summary_tables_control.txt")
    sig_path = os.path.join(PLOTS_FOLDER, "law_significance_results.xlsx")

    required_files_exist = all(
        os.path.exists(p) for p in [main_input_path, control_input_path, sig_path]
    )

    if required_files_exist:
        print("--- Running analysis for Main vs. Control (Combined Plot) ---")
        main_df = parse_batch_summary(main_input_path)
        main_df["condition"] = "main"
        control_df = parse_batch_summary(control_input_path)
        control_df["condition"] = "control"
        df = pd.concat([main_df, control_df], ignore_index=True)
        sig_df = pd.read_excel(sig_path)

        if not df.empty and not sig_df.empty:
            combined_plot_path = os.path.join(PLOTS_FOLDER, "stability_trends_combined.html")
            save_combined_trend_plot(df, sig_df, combined_plot_path)
            save_pos_stacked_comparison_plot(df, PLOTS_FOLDER)
            print(f"Combined trend plot saved: {combined_plot_path}")
        else:
            print("Warning: Data or significance file is empty, skipping combined plot.")
    else:
        print("Warning: Missing required files for combined plot, skipping.")

    # --- 2. Generate Standard Plots for backward compatibility ---
    print("\n--- Running analysis for Standard Plots (Backward Compatibility) ---")
    pattern = os.path.join(PLOTS_FOLDER, "batch_summary_tables*.txt")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No batch summary files found matching: {pattern}")
        return

    for input_path in files:
        basename = os.path.basename(input_path)

        suffix = basename.replace("batch_summary_tables", "").replace(".txt", "")
        print(f"Processing for standard plots: {basename} (suffix='{suffix}')")

        df = parse_batch_summary(input_path)
        if df.empty:
            print(f"Warning: No valid data parsed, skipping. Path: {input_path}")
            continue

        # Standard plots (XLSX and simple trend)
        xlsx_path = os.path.join(PLOTS_FOLDER, f"stability_all_time_ranges{suffix}.xlsx")
        plot_path = os.path.join(PLOTS_FOLDER, f"stability_trends_by_span{suffix}.html")
        stacked_plot_path = os.path.join(PLOTS_FOLDER, f"stability_trends_stacked{suffix}.html")
        stacked_plot_higher_path = os.path.join(PLOTS_FOLDER, f"stability_trends_stacked_higher{suffix}.html")
        save_xlsx(df, xlsx_path)
        save_trend_plot(df, plot_path)
        save_stacked_trend_plot(df, stacked_plot_path)
        save_stacked_trend_plot_higher(df, stacked_plot_higher_path)
        print(f"Standard XLSX saved: {xlsx_path}")
        print(f"Standard trend plot saved: {plot_path}")
        print(f"Stacked trend plot saved: {stacked_plot_path}")
        print(f"Stacked trend plot (higher) saved: {stacked_plot_higher_path}")

        latex_path = os.path.join(PLOTS_FOLDER, f"stability_summary{suffix}.tex")
        save_latex_summary(df, latex_path)
        print(f"LaTeX summary saved: {latex_path}")


    # --- 3. Generate the Comparison Summary Excel ---
    print("\n--- Generating Comparison Summary ---")
    comparison_output_path = os.path.join(PLOTS_FOLDER, "comparison_summary.xlsx")
    save_comparison_summary(comparison_output_path)


if __name__ == "__main__":
    main()