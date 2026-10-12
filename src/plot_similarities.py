import numpy as np
import argparse
import plotly.graph_objects as go


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos", required=True, choices=["NOUN", "VERB", "ADJ"])
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = f"../data/histograms/{args.pos}/results.npz"
    data = np.load(data_path)

    bin_centers = data["bin_centers"]
    same_hist = data["same_hist"].astype(float)
    diff_hist = data["diff_hist"].astype(float)

    # normalize to probability mass
    if same_hist.sum() > 0:
        same_hist /= same_hist.sum()
    if diff_hist.sum() > 0:
        diff_hist /= diff_hist.sum()

    # infer bin width
    bin_width = bin_centers[1] - bin_centers[0]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bin_centers,
        y=same_hist,
        name="Same synset",
        opacity=0.7,
        marker_color='blue',
        width=bin_width
    ))
    fig.add_trace(go.Bar(
        x=bin_centers,
        y=diff_hist,
        name="Different synset",
        opacity=0.7,
        marker_color='red',
        width=bin_width
    ))

    fig.update_layout(
        title=f"Similarity distributions for {args.pos}",
        xaxis_title="Cosine similarity",
        yaxis_title="Probability mass",
        barmode='overlay',  # to overlay bars
        template="plotly_white"
    )

    out = f"../data/histograms/{args.pos}/similarity_histogram_{args.pos}.html"
    fig.write_html(out)
    print("Saved", out)


if __name__ == "__main__":
    main()
