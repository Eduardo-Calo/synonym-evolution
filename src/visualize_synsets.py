"""
Interactive Dash app for visualizing synsets.
Run: python visualize_synsets.py --synsets <synsets.jsonl> --embeddings <embeddings.npz>
Then open http://localhost:8050 in browser.
"""

import argparse
import json
import re
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional
from dash import Dash, dcc, html, Input, Output, callback
import plotly.graph_objects as go
from sklearn.decomposition import TruncatedSVD


@dataclass
class AppState:
    """Holds all precomputed data for the Dash callbacks."""
    embeddings: np.ndarray = None
    reduced_emb: np.ndarray = None
    synsets: List[Dict] = field(default_factory=list)
    lus: List[Dict] = field(default_factory=list)

    # Per-LU arrays (aligned with `lus`)
    lu_to_synset_id: np.ndarray = None   # actual synset_id (-1 if unmapped)
    hover_text: np.ndarray = None
    lu_sy_size: np.ndarray = None
    lu_is_singleton: np.ndarray = None
    lu_start_year: np.ndarray = None
    lu_end_year: np.ndarray = None
    lu_words: np.ndarray = None

    # Per-synset (indexed by position in synsets list)
    synset_id_list: List[int] = field(default_factory=list)
    synset_sizes: List[int] = field(default_factory=list)
    synset_metadata: List[Dict] = field(default_factory=list)

    # Lookup: synset_id -> position index
    synset_id_to_idx: Dict[int, int] = field(default_factory=dict)

    dropdown_options: List[Dict] = field(default_factory=list)
    pos: Optional[str] = None
    year_min: int = 0
    year_max: int = 2026


state = AppState()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reduce_embeddings(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Reduce embeddings to 2D using TruncatedSVD."""
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    return svd.fit_transform(embeddings)


def load_npz_embeddings(path: Path) -> Dict[str, np.ndarray]:
    """Load embeddings from a .npz (preferred) or .npy file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    if path.suffix == '.npz':
        loaded = np.load(path, allow_pickle=False)
        return {k: loaded[k] for k in loaded.files}
    elif path.suffix == '.npy':
        return {'default': np.load(path)}
    else:
        try:
            arr = np.load(path)
            return {'default': arr}
        except Exception as e:
            raise ValueError(f"Unsupported embeddings file: {path} ({e})")


def load_synsets_json(path: Path) -> List[Dict]:
    """Load synsets from JSON or JSONL file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Synsets file not found: {path}")

    synsets = []
    if path.suffix in ('.jsonl', '.ndjson'):
        with path.open('r', encoding='utf8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                synsets.append(json.loads(line))
    else:
        with path.open('r', encoding='utf8') as fh:
            data = json.load(fh)
            if isinstance(data, list):
                synsets = data
            elif isinstance(data, dict) and 'synsets' in data:
                synsets = data['synsets']
            else:
                raise ValueError(f"Unexpected JSON structure in {path}")
    return synsets


def build_size_marks(max_size: int) -> dict:
    max_size = int(max_size)
    anchors = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    keys = [int(v) for v in anchors if v <= max_size]
    if not keys or keys[-1] != max_size:
        keys.append(max_size)
    if len(keys) >= 2 and (max_size - keys[-2]) <= 5:
        keys = keys[:-2] + [max_size]
    keys = list(dict.fromkeys(keys))
    return {int(v): str(int(v)) for v in keys}


def _parse_year(x):
    if x is None or x == '':
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Layout — returned as a callable so Dash re-evaluates on each page load
# ---------------------------------------------------------------------------

def build_layout():
    """Build Dash app layout from current state."""
    max_synset_size = int(max(state.synset_sizes)) if state.synset_sizes else 1
    slider_marks = build_size_marks(max_synset_size)
    initial_upper = min(max_synset_size, 50)

    header_text = "Synsets Visualization"
    if state.pos:
        header_text = f"Synsets Visualization ({state.pos})"

    return html.Div([
        html.H1(header_text, style={"textAlign": "center", "marginBottom": 20}),

        html.Div([
            html.Div([
                html.Label("Synset Size Range:", style={"fontWeight": "bold"}),
                dcc.RangeSlider(
                    id='size-slider',
                    min=1,
                    max=max_synset_size,
                    step=1,
                    marks=slider_marks,
                    value=[1, initial_upper],
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ], style={"width": "45%", "display": "inline-block", "marginRight": "5%"}),

            html.Div([
                html.Label("Filter Options:", style={"fontWeight": "bold"}),
                dcc.Checklist(
                    id='filter-options',
                    options=[
                        {"label": " Exclude singletons", "value": "exclude_singletons"},
                        {"label": " Only singleton LUs", "value": "only_singleton_lus"},
                    ],
                    value=[],
                    style={"marginTop": 10}
                ),
            ], style={"width": "45%", "display": "inline-block"}),
        ], style={"marginBottom": 20}),

        html.Div([
            dcc.Checklist(
                id='show-year-toggle',
                options=[{"label": " Show Year Slider", "value": "show_year"}],
                value=["show_year"],
                style={"marginBottom": 6}
            ),
            html.Div([
                html.Label("Year (select a single year):", style={"fontWeight": "bold"}),
                dcc.Slider(
                    id='year-slider',
                    min=state.year_min,
                    max=state.year_max,
                    step=1,
                    value=state.year_min,
                    marks={state.year_min: str(state.year_min), state.year_max: str(state.year_max)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], id='year-slider-container', style={"marginBottom": 20})
        ]),

        html.Div([
            html.Div([
                html.Label("Search by Word (comma-separated):", style={"fontWeight": "bold"}),
                dcc.Input(
                    id='word-search',
                    type='text',
                    placeholder='Enter word...',
                    style={"width": "100%"}
                ),
            ], style={"width": "45%", "display": "inline-block", "marginRight": "5%"}),
        ], style={"marginBottom": 20}),

        html.Div([
            html.Label("Select Synsets:", style={"fontWeight": "bold"}),
            dcc.Dropdown(
                id='synset-dropdown',
                options=[],
                value=[],
                multi=True,
                placeholder="Type synset ID to search...",
                style={"width": "100%"},
                searchable=True,
                clearable=True
            ),
        ], style={"marginBottom": 20}),

        html.Div([
            html.Div(dcc.Graph(id='synset-graph'), style={"width": "66%", "display": "inline-block", "verticalAlign": "top"}),
            html.Div(dcc.Graph(id='timeline-graph'), style={"width": "33%", "display": "inline-block", "verticalAlign": "top", "paddingLeft": 10}),
        ], style={"display": "flex", "flexDirection": "row"}),

        html.Div(id='stats-display', style={"marginTop": 20, "padding": 10,
                                            "backgroundColor": "#f0f0f0", "borderRadius": 5})
    ])


app = Dash(__name__)
app.layout = build_layout


# ---------------------------------------------------------------------------
# Scatter figure builder
# ---------------------------------------------------------------------------

def create_scatter_figure(visible_mask):
    coords = state.reduced_emb
    colors_to_plot = state.lu_to_synset_id[visible_mask]
    color_min = int(np.min(colors_to_plot)) if len(colors_to_plot) > 0 else 0
    color_max = int(np.max(colors_to_plot)) if len(colors_to_plot) > 0 else 1

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=coords[visible_mask, 0],
        y=coords[visible_mask, 1],
        mode='markers',
        marker=dict(
            size=6,
            color=colors_to_plot,
            colorscale='Viridis',
            cmin=color_min,
            cmax=color_max,
            showscale=True,
            colorbar=dict(title="Synset ID", thickness=15, len=0.7),
            line=dict(width=0.5, color='white'),
            opacity=0.8
        ),
        text=state.hover_text[visible_mask],
        hovertemplate='%{text}<extra></extra>',
        name="LUs",
        hoverlabel=dict(bgcolor='black'),
        showlegend=False
    ))

    title = "Synsets Visualization"
    if state.pos:
        title = f"Synsets Visualization ({state.pos})"

    fig.update_layout(
        title=title,
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        hovermode='closest',
        width=1100,
        height=700,
        template='plotly_white',
        showlegend=True,
        legend=dict(x=1.02, y=1, xanchor='left', yanchor='top')
    )
    return fig


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

@callback(
    [Output('synset-graph', 'figure'),
     Output('timeline-graph', 'figure'),
     Output('stats-display', 'children'),
     Output('synset-dropdown', 'options'),
     Output('year-slider-container', 'style')],
    [Input('size-slider', 'value'),
     Input('filter-options', 'value'),
     Input('synset-dropdown', 'value'),
     Input('word-search', 'value'),
     Input('show-year-toggle', 'value'),
     Input('year-slider', 'value')]
)
def update_figure(size_range, filter_options, selected_synsets, word_search, show_year_toggle, year):
    # Normalize inputs
    if not selected_synsets:
        selected_synsets = []
    if not filter_options:
        filter_options = []

    n_lus = len(state.lus)
    max_size_available = max(state.synset_sizes) if state.synset_sizes else 1

    if isinstance(size_range, (int, float, np.integer, np.floating)):
        size_range = [1, int(size_range)]
    if not isinstance(size_range, (list, tuple)) or len(size_range) != 2:
        size_range = [1, max_size_available]

    size_min_raw, size_max_raw = size_range
    min_size = max(1, min(size_min_raw, size_max_raw))
    max_size = min(max_size_available, max(size_min_raw, size_max_raw))

    # Convert selected synsets — dropdown values are synset_id strings
    selected_syn_ids = set()
    if selected_synsets:
        try:
            selected_syn_ids = set(int(s) for s in selected_synsets)
        except (ValueError, TypeError):
            selected_syn_ids = set()

    apply_year = bool(
        show_year_toggle
        and isinstance(show_year_toggle, (list, tuple))
        and 'show_year' in show_year_toggle
    )

    # Build visibility mask
    visible_mask = np.ones(n_lus, dtype=bool)
    has_synset = (state.lu_to_synset_id >= 0)

    if 'exclude_singletons' in filter_options:
        exclude_sing = has_synset & (state.lu_is_singleton | (state.lu_sy_size == 1))
        visible_mask &= ~exclude_sing

    if 'only_singleton_lus' in filter_options:
        only_sing = has_synset & (state.lu_is_singleton | (state.lu_sy_size == 1))
        visible_mask &= only_sing

    # Word search (exact match)
    if word_search:
        w_list = [w.strip().lower() for w in word_search.split(',') if w.strip()]
        if w_list:
            visible_mask &= np.isin(state.lu_words, w_list)

    # Year filter
    if apply_year and year is not None:
        try:
            y_val = float(year)
        except Exception:
            y_val = None
        if y_val is not None:
            e_arr = np.nan_to_num(state.lu_end_year, nan=9999.0)
            valid_s = ~np.isnan(state.lu_start_year)
            in_range = (state.lu_start_year <= y_val) & (y_val <= e_arr)
            visible_mask &= (valid_s & in_range)

    # Explicit synset selection — uses synset_id, not index
    if selected_syn_ids:
        visible_mask &= np.isin(state.lu_to_synset_id, list(selected_syn_ids))

    # Size range (only when no explicit selection / word search)
    if not selected_syn_ids and not word_search:
        size_out = (state.lu_sy_size < min_size) | (state.lu_sy_size > max_size)
        visible_mask &= ~(has_synset & size_out)

    # Scatter figure
    fig = create_scatter_figure(visible_mask)

    # Timeline figure — keep all items in memory, no truncation
    idxs = np.where(visible_mask)[0]

    if len(idxs) > 0:
        s_sort = np.nan_to_num(state.lu_start_year[idxs], nan=999999.0)
        syn_sort = state.lu_to_synset_id[idxs]
        sort_order = np.lexsort((syn_sort, s_sort))
        ordered_idxs = idxs[sort_order]

        starts = state.lu_start_year[ordered_idxs].copy()
        ends = state.lu_end_year[ordered_idxs].copy()

        # Drop LUs without a valid start year
        valid_start_mask = ~np.isnan(starts)
        if not np.all(valid_start_mask):
            ordered_idxs = ordered_idxs[valid_start_mask]
            starts = starts[valid_start_mask]
            ends = ends[valid_start_mask]

        N = len(ordered_idxs)
        x_max_limit = float(state.year_max)
        starts = np.maximum(starts, 0.0)
        ends = np.nan_to_num(ends, nan=9999.0)
        ends = np.minimum(ends, x_max_limit)

        # Line segments: [s0, e0, nan, s1, e1, nan, ...]
        x_mat = np.empty((N, 3))
        x_mat[:, 0] = starts
        x_mat[:, 1] = ends
        x_mat[:, 2] = np.nan
        xs = x_mat.ravel()

        y_range = np.arange(N, dtype=float)
        y_mat = np.empty((N, 3))
        y_mat[:, 0] = y_range
        y_mat[:, 1] = y_range
        y_mat[:, 2] = np.nan
        ys = y_mat.ravel()

        # Marker midpoints
        tm_x = (starts + ends) * 0.5
        tm_y = y_range
        tm_text = state.hover_text[ordered_idxs]

        c_vals = state.lu_to_synset_id[ordered_idxs]
        cmin = int(np.min(c_vals)) if len(c_vals) > 0 else 0
        cmax = int(np.max(c_vals)) if len(c_vals) > 0 else 1

        timeline_fig = go.Figure()
        timeline_fig.add_trace(go.Scattergl(
            x=xs, y=ys, mode='lines',
            line=dict(color='lightgray', width=2),
            hoverinfo='skip', showlegend=False
        ))
        timeline_fig.add_trace(go.Scattergl(
            x=tm_x, y=tm_y, mode='markers',
            marker=dict(
                size=8, color=c_vals, colorscale='Viridis',
                cmin=cmin, cmax=cmax, showscale=False,
                line=dict(width=0)
            ),
            text=tm_text,
            hovertemplate='%{text}<extra></extra>',
            hoverlabel=dict(bgcolor='black'),
            showlegend=False
        ))
    else:
        timeline_fig = go.Figure()

    timeline_fig.update_layout(
        title='LU Activity Timeline',
        xaxis=dict(title='Year', range=[0, int(state.year_max)]),
        yaxis_title='LU (ordered)',
        height=700, width=400,
        template='plotly_white'
    )
    timeline_fig.update_yaxes(autorange='reversed', showticklabels=False)

    # Stats
    num_visible = int(np.sum(visible_mask))
    if num_visible > 0:
        vis_syns = state.lu_to_synset_id[visible_mask]
        num_visible_synsets = len(np.unique(vis_syns[vis_syns >= 0]))
    else:
        num_visible_synsets = 0

    stats_text = f"Showing {num_visible} LUs in {num_visible_synsets} synsets"
    if selected_syn_ids:
        stats_text += f" | Selected: {len(selected_syn_ids)}"
    elif not word_search:
        stats_text += f" | Size: [{min_size}, {max_size}]"
    if word_search:
        stats_text += f" | Word: '{word_search}'"
    if 'only_singleton_lus' in filter_options:
        stats_text += " | Singleton LUs only"

    slider_style = {"display": "block"} if apply_year else {"display": "none"}
    return fig, timeline_fig, html.P(stats_text, style={"fontSize": 14}), state.dropdown_options, slider_style


# ---------------------------------------------------------------------------
# Main — populates AppState, then runs Dash
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize synsets with Dash (JSON/NPZ pipeline)")
    parser.add_argument("--synsets", required=True, help="Path to synsets JSON or JSONL file")
    parser.add_argument("--embeddings", required=True, help="Path to embeddings .npz (preferred) or .npy file")
    parser.add_argument("--port", type=int, default=8050, help="Port to run app on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to run app on")
    args = parser.parse_args()

    # Detect POS from filename
    try:
        fname = Path(args.synsets).stem
        m = re.match(r'^([A-Za-z]+)', fname)
        state.pos = m.group(1).upper() if m else None
    except Exception:
        state.pos = None

    # Load synsets
    state.synsets = load_synsets_json(Path(args.synsets))

    # Build synset metadata & bidirectional ID mapping
    for syn_pos, syn in enumerate(state.synsets):
        sid = syn.get('synset_id', syn_pos)
        orig = syn.get('lus', [])
        state.synset_id_list.append(sid)
        state.synset_sizes.append(len(orig))
        state.synset_id_to_idx[sid] = syn_pos
        state.synset_metadata.append({
            'synset_id': sid,
            'original_lus': [l.get('lemma') if isinstance(l, dict) else l for l in orig],
            'is_singleton': syn.get('is_singleton_from_lu', False),
            'pos': syn.get('pos', ''),
        })

    # Load embeddings
    emb_dict = load_npz_embeddings(Path(args.embeddings))
    hist_keys = list(emb_dict.keys())
    if 'default' in emb_dict:
        selected_embeddings = emb_dict['default']
    elif len(hist_keys) == 1:
        selected_embeddings = emb_dict[hist_keys[0]]
    elif len(hist_keys) > 1:
        selected_embeddings = np.vstack([emb_dict[k] for k in sorted(hist_keys)])
    else:
        selected_embeddings = None
    if selected_embeddings is None:
        raise ValueError("No embeddings found in --embeddings")

    # Build flat LU list from synsets
    lus_flat = []
    for syn_pos, syn in enumerate(state.synsets):
        sid = syn.get('synset_id', syn_pos)
        for idx_lu, lu in enumerate(syn.get('lus', [])):
            lus_flat.append({
                'lu_id': f"lu_{sid}_{idx_lu}",
                'lemma': (lu.get('lemma') or lu.get('word') or '').strip(),
                'word': (lu.get('lemma') or lu.get('word') or '').strip(),
                'meaning': (lu.get('def') or lu.get('meaning') or '').strip(),
                'start': lu.get('start'),
                'end': lu.get('end'),
                'marker_type': lu.get('marker_type', 0),
                'synset_id': sid,
            })

    # Locate parsed CSV for embedding alignment
    pos_for_parsed = (state.pos or Path(args.synsets).stem.split('_')[0]).upper()
    parsed_csv = Path(args.synsets).parents[1] / 'parsed' / f"{pos_for_parsed}.csv"
    if not parsed_csv.exists():
        raise ValueError(f"Parsed CSV not found at {parsed_csv}")

    # Build key -> parsed-row-index map using pandas
    print(f"Loading parsed CSV: {parsed_csv}")
    parsed_df = pd.read_csv(parsed_csv, dtype=str).fillna('')
    parsed_df['lemma'] = parsed_df['lemma'].str.strip()
    parsed_df['definition'] = parsed_df['definition'].str.split().str.join(' ').str.strip()
    parsed_df['start'] = parsed_df['start'].str.strip()
    parsed_df['end'] = parsed_df['end'].str.strip()

    parsed_index_map = {}
    for idx_row in range(len(parsed_df)):
        row = parsed_df.iloc[idx_row]
        key = (row['lemma'], row['definition'], row['start'], row['end'])
        if key not in parsed_index_map:
            parsed_index_map[key] = idx_row
    parsed_rows_count = len(parsed_df)

    # Match synset LUs to parsed CSV rows -> embedding indices
    emb_indices = []
    kept_lus = []
    unmatched_lus = []

    for lu in lus_flat:
        lemma_key = (lu.get('lemma') or '').strip()
        def_key = ' '.join((lu.get('meaning') or '').split()).strip()
        start_val = lu.get('start')
        end_val = lu.get('end')
        start_key = str(start_val).strip() if start_val is not None and start_val != '' else ''
        end_key = str(end_val).strip() if end_val is not None and end_val != '' else ''
        key = (lemma_key, def_key, start_key, end_key)

        if key in parsed_index_map:
            pidx = parsed_index_map[key]
            emb_indices.append(pidx)
            kept_lus.append(lu)
        else:
            unmatched_lus.append(lu)

    emb_indices = np.array(emb_indices, dtype=int)
    num_kept = len(emb_indices)
    print(f"Parsed rows: {parsed_rows_count}; Embeddings: {selected_embeddings.shape[0]}; Kept: {num_kept}; Dropped: {len(unmatched_lus)}")

    if unmatched_lus:
        for lu in unmatched_lus[:5]:
            print(f"  Dropped: synset_id={lu.get('synset_id')} lemma={lu.get('lemma')} start={lu.get('start')} end={lu.get('end')}")
    if num_kept == 0:
        raise ValueError("No synset LUs matched the parsed CSV; aborting.")

    # Select & reorder embeddings
    try:
        state.embeddings = selected_embeddings[emb_indices]
    except Exception:
        state.embeddings = np.take(selected_embeddings, emb_indices, axis=0)

    state.lus = kept_lus

    # Build lu_to_synset_id using actual synset_id, not index
    state.lu_to_synset_id = np.array([lu.get('synset_id', -1) for lu in state.lus], dtype=int)

    # Reduce embeddings
    print("Reducing embeddings with TruncatedSVD...")
    state.reduced_emb = reduce_embeddings(state.embeddings)

    # Precompute hover text
    hover_list = []
    for i, lu in enumerate(state.lus):
        lu_id = lu.get("lu_id", f"idx_{i}")
        word = lu.get("word", "N/A")
        meaning = lu.get("meaning", "")[:50]
        sid = state.lu_to_synset_id[i]

        if sid >= 0 and sid in state.synset_id_to_idx:
            syn_pos = state.synset_id_to_idx[sid]
            meta = state.synset_metadata[syn_pos]
            size = state.synset_sizes[syn_pos]
            example_lus = ", ".join(meta.get("original_lus", [])[:3])
            if len(meta.get("original_lus", [])) > 3:
                example_lus += "..."
            hover_list.append(
                f"<b>{lu_id}</b><br>"
                f"Word: {word}<br>"
                f"Meaning: {meaning}<br>"
                f"Start: {lu.get('start')}<br>"
                f"End: {lu.get('end')}<br>"
                f"<b>Synset {sid}</b> (size {size})<br>"
                f"Example LUs: {example_lus}"
            )
        else:
            hover_list.append(
                f"<b>{lu_id}</b><br>"
                f"Word: {word}<br>"
                f"Meaning: {meaning}<br>"
                f"Start: {lu.get('start')}<br>"
                f"End: {lu.get('end')}<br>"
                "Not in any synset"
            )
    state.hover_text = np.array(hover_list)

    # Precompute per-LU filter arrays
    _sizes = np.array(state.synset_sizes, dtype=int)
    _is_single = np.array([m.get('is_singleton', False) for m in state.synset_metadata], dtype=bool)

    # Map each LU's synset_id -> position index for array lookups
    _lu_syn_pos = np.array(
        [state.synset_id_to_idx.get(sid, -1) for sid in state.lu_to_synset_id],
        dtype=int
    )
    valid_mask = (_lu_syn_pos >= 0) & (_lu_syn_pos < len(_sizes))

    state.lu_sy_size = np.zeros(len(state.lus), dtype=int)
    state.lu_sy_size[valid_mask] = _sizes[_lu_syn_pos[valid_mask]]

    state.lu_is_singleton = np.zeros(len(state.lus), dtype=bool)
    state.lu_is_singleton[valid_mask] = _is_single[_lu_syn_pos[valid_mask]]

    state.lu_start_year = np.array([_parse_year(lu.get('start')) for lu in state.lus], dtype=np.float32)
    state.lu_end_year = np.array([_parse_year(lu.get('end')) for lu in state.lus], dtype=np.float32)
    state.lu_words = np.array([(lu.get('word') or '').lower().strip() for lu in state.lus], dtype=object)

    # Dropdown options — value is synset_id (not index)
    state.dropdown_options = [
        {"label": f"{state.synset_id_list[i]} (size {state.synset_sizes[i]})",
         "value": str(state.synset_id_list[i])}
        for i in range(len(state.synsets))
    ]

    # Year bounds — safe against LUs with both start and end None
    valid_starts = [lu.get('start') for lu in state.lus if lu.get('start') is not None]
    valid_ends = []
    for lu in state.lus:
        if lu.get('end') is not None:
            valid_ends.append(lu['end'])
        elif lu.get('start') is not None:
            valid_ends.append(lu['start'])
    state.year_min = int(min(valid_starts)) if valid_starts else 0
    state.year_max = int(max(valid_ends)) if valid_ends else 2026

    print(f"Starting Dash app at http://{args.host}:{args.port}")
    app.run(debug=True, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
