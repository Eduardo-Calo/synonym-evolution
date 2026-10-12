import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


def embed_texts(texts: List[str], model_name: str, batch_size: int, prompt: str = "") -> np.ndarray:
    """Embed texts using SentenceTransformer model."""
    model = SentenceTransformer(
        model_name,
        model_kwargs={"device_map": "auto"},
        tokenizer_kwargs={"padding_side": "left"},
    )
    embeddings = model.encode(
        texts,
        prompt=prompt if prompt else None,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    return embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed definitions from per-POS CSV files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory with per-POS CSV files (e.g., NOUN.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save embeddings as .npy files.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-Embedding-8B",  # https://huggingface.co/spaces/mteb/leaderboard
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for embedding (default: 1024).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Embed the following definition:",
        help="Prompt for the model (default: 'Embed the following definition:').",
    )
    parser.add_argument(
        "--pos",
        nargs="*",
        default=[],
        help="List of POS to process (default: all).",
    )
    return parser.parse_args()


def read_definitions(csv_path: Path) -> List[str]:
    definitions = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            definitions.append(row["definition"])
    return definitions


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise ValueError(f"Input directory {args.input_dir} does not exist.")
    if not args.input_dir.is_dir():
        raise ValueError(f"Input path {args.input_dir} is not a directory.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.pos:
        csv_files = []
        for p in args.pos:
            csv_file = args.input_dir / f"{p}.csv"
            if not csv_file.exists():
                print(f"Warning: {csv_file} does not exist, skipping {p}.")
            else:
                csv_files.append(csv_file)
        if not csv_files:
            raise ValueError(f"No valid CSV files found for specified POS in {args.input_dir}.")
    else:
        csv_files = list(args.input_dir.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in {args.input_dir}.")

    for csv_file in sorted(csv_files):
        pos = csv_file.stem
        print(f"Processing {pos}...")
        definitions = read_definitions(csv_file)
        if not definitions:
            print(f"No definitions in {csv_file}, skipping.")
            continue
        embeddings = embed_texts(definitions, args.model_name, args.batch_size, args.prompt)
        out_path = args.output_dir / f"{pos}_embeddings.npy"
        np.save(out_path, embeddings)
        print(f"Saved {len(embeddings)} embeddings to {out_path}")


if __name__ == "__main__":
    main()