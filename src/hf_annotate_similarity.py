#!/usr/bin/env python3
"""Run Hugging Face LLM annotation for definition similarity JSONL files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


LABELS = [
    "SIMILAR",
    "DISSIMILAR",
    "CANNOT DECIDE",
]

DEFAULT_HF_HUB_ROOT = "/mimer/NOBACKUP/Datasets/LLM/huggingface/hub"
DEFAULT_LLM_OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "data" / "llm_output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prompt a Hugging Face LLM to label definition relatedness."
    )
    parser.add_argument(
        "--input",
        default="../data/hum_eval/ADJ/ADJ_sample_001.jsonl",
        help="Input JSONL with keys: idx, definition1, definition2",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (auto-generated with model name if omitted)",
    )
    parser.add_argument(
        "--config",
        default="qualtrics_config.json",
        help="Path to qualtrics config JSON containing INSTRUCTION_TEXT",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help=(
            "Model identifier. Can be a repo id (e.g. Qwen/Qwen2-0.5B), a "
            "model directory name (models--Qwen--Qwen2-0.5B), or a full path"
        ),
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional full path to a local model snapshot directory",
    )
    parser.add_argument(
        "--hub-root",
        default=DEFAULT_HF_HUB_ROOT,
        help="Root of local Hugging Face cache containing models--*/snapshots/*",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=48,
        help="Maximum generated tokens per item",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 for greedy)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p for sampling",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of records to process",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to tokenizer/model loading",
    )
    parser.add_argument(
        "--fsync-each-row",
        action="store_true",
        help="Call fsync after each written row (slower, but crash-safe)",
    )
    return parser.parse_args()


def clean_instruction_text(text: str) -> str:
    text = text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_prompt(
    instruction_text: str,
    definition1: str,
    definition2: str,
    word1: str = "",
    word2: str = "",
    sample_sentence1: str = "",
    sample_sentence2: str = "",
) -> str:
    label_list = ", ".join(LABELS)

    details = []
    if word1:
        details.append(f"Word 1: {word1}")
    details.append(f"Definition 1: {definition1}")
    if sample_sentence1:
        details.append(f"Sample sentence 1: {sample_sentence1}")
    if word2:
        details.append(f"Word 2: {word2}")
    details.append(f"Definition 2: {definition2}")
    if sample_sentence2:
        details.append(f"Sample sentence 2: {sample_sentence2}")

    pair_text = "\n".join(details)

    return (
        f"{instruction_text}\n\n"
        "Now label the following pair.\n"
        f"{pair_text}\n\n"
        f"Return exactly one label from: {label_list}.\n"
        "Output format: LABEL: <one label>\n"
        "Do not include explanations, steps, or any other text."
    )


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def normalize_label(text: str) -> str:
    normalized_text = re.sub(r"[_-]", " ", text.upper())

    allowed_ordered = [
        "SIMILAR",
        "DISSIMILAR",
        "CANNOT DECIDE",
    ]
    allowed_set = set(allowed_ordered)

    # 1) Preferred: explicit tagged label anywhere in output.
    tagged_pattern = (
        r"\b(?:LABEL|ANSWER)\b\s*[:=\-]\s*"
        r"(DISSIMILAR|SIMILAR|CANNOT\s+DECIDE)\b"
    )
    tagged_match = re.search(tagged_pattern, normalized_text)
    if tagged_match:
        return re.sub(r"\s+", " ", tagged_match.group(1)).strip()

    # 2) Accept lines that are exactly a label (or LABEL:/ANSWER: + label).
    for raw_line in normalized_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^(?:LABEL|ANSWER)\s*[:=\-]\s*", "", line)
        line = re.sub(r"[\s\.;:,!?]+$", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line in allowed_set:
            return line

    return "CANNOT EXTRACT"


def generate_label(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are an experienced lexicographer, expert in semantic relationships. Follow the instruction exactly.",
        },
        {"role": "user", "content": prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            # Qwen3-family models may emit long "thinking" traces unless disabled.
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    else:
        input_text = prompt

    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature > 0.0
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0, inputs["input_ids"].shape[1] :]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return decoded


def pick_input_path(input_arg: str) -> Path:
    requested = Path(input_arg)
    if requested.exists():
        return requested

    parent = requested.parent
    name = requested.name

    # Allow fallback from POS_sample.jsonl -> POS_sample_*.jsonl for ADJ/NOUN/VERB.
    m = re.match(r"^(ADJ|NOUN|VERB)_sample\.jsonl$", name)
    if m:
        pos = m.group(1)
        candidates = sorted(parent.glob(f"{pos}_sample_*.jsonl"))
        if candidates:
            return candidates[0]

    # Backward-compatible convenience: if no POS is specified, prefer ADJ first.
    if name == "sample.jsonl":
        for pos in ("ADJ", "NOUN", "VERB"):
            candidates = sorted(parent.glob(f"{pos}_sample_*.jsonl"))
            if candidates:
                return candidates[0]

    raise FileNotFoundError(f"Input file not found: {requested}")


def model_id_to_dirname(model_id: str) -> str:
    return "models--" + model_id.strip("/").replace("/", "--")


def latest_snapshot_dir(model_dir: Path) -> Path:
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists() or not snapshots_dir.is_dir():
        raise FileNotFoundError(f"No snapshots directory found: {snapshots_dir}")

    snapshots = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    if not snapshots:
        raise FileNotFoundError(f"No snapshot folders found in: {snapshots_dir}")

    return max(snapshots, key=lambda p: p.stat().st_mtime)


def resolve_model_path(model: str, model_path: Optional[str], hub_root: str) -> Path:
    if model_path:
        explicit = Path(model_path)
        if explicit.exists():
            if (explicit / "config.json").exists():
                return explicit
            if (explicit / "snapshots").is_dir():
                return latest_snapshot_dir(explicit)
        raise FileNotFoundError(f"Provided --model-path does not exist: {explicit}")

    model_arg = Path(model)
    if model_arg.exists():
        if (model_arg / "config.json").exists():
            return model_arg
        if (model_arg / "snapshots").is_dir():
            return latest_snapshot_dir(model_arg)

    hub = Path(hub_root)
    if model.startswith("models--"):
        model_dir = hub / model
    else:
        model_dir = hub / model_id_to_dirname(model)

    if model_dir.is_dir():
        return latest_snapshot_dir(model_dir)

    raise FileNotFoundError(
        "Could not resolve local model path. Tried model arg/path and hub lookup at "
        f"{model_dir}. You can pass --model-path with a full snapshots/<hash> path."
    )


def model_tag(model: str, resolved_model_path: Path) -> str:
    # Prefer model directory name from the resolved local hub path.
    model_dir_name = resolved_model_path.parent.parent.name
    if model_dir_name.startswith("models--"):
        raw = model_dir_name[len("models--") :].replace("--", "_")
    else:
        raw = model.replace("/", "_")

    raw = raw.replace(".", "_")
    return re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_") or "model"


def default_output_path(input_path: Path, model_name_tag: str) -> Path:
    output_dir = DEFAULT_LLM_OUTPUT_ROOT / model_name_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{input_path.stem}_hf_labels_{model_name_tag}.jsonl"


def is_binary_sample_input(input_path: Path) -> bool:
    return re.match(r"^(ADJ|NOUN|VERB)_sample_binary\.jsonl$", input_path.name) is not None


def has_qwen_label(row: Dict[str, Any]) -> bool:
    source = str(row.get("source", "")).strip().lower()
    label = str(row.get("predicted_label", "")).strip().upper()
    return source == "qwen" and label in LABELS


def main() -> None:
    args = parse_args()

    input_path = pick_input_path(args.input)
    config_path = Path(args.config)

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    instruction_text = clean_instruction_text(
        config.get("INSTRUCTION_TEXT_BINARY", config["INSTRUCTION_TEXT"])
    )
    resolved_model_path = resolve_model_path(
        model=args.model,
        model_path=args.model_path,
        hub_root=args.hub_root,
    )
    model_name_tag = model_tag(args.model, resolved_model_path)
    output_path = (
        Path(args.output)
        if args.output is not None
        else default_output_path(input_path, model_name_tag)
    )

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map: Optional[str] = "auto" if torch.cuda.is_available() else None

    print(f"Loading model from: {resolved_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(resolved_model_path),
        trust_remote_code=args.trust_remote_code,
    )

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "dtype": dtype,
        "device_map": device_map,
    }

    model = AutoModelForCausalLM.from_pretrained(
        str(resolved_model_path),
        **model_kwargs,
    )

    records = list(load_jsonl(input_path))
    if args.limit is not None:
        records = records[: args.limit]

    total_rows = len(records)

    binary_sample_mode = is_binary_sample_input(input_path)
    annotate_indices = []
    for i, row in enumerate(records):
        if binary_sample_mode and has_qwen_label(row):
            continue
        annotate_indices.append(i)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    annotated_count = 0
    skipped_qwen_count = 0

    annotate_set = set(annotate_indices)

    with output_path.open("w", encoding="utf-8") as out:
        for i, row in enumerate(tqdm(
            records,
            desc="Annotating",
            unit="pair",
            total=total_rows,
            dynamic_ncols=True,
            bar_format="{desc}: {n_fmt}/{total_fmt} | {elapsed}<{remaining} | {rate_fmt}",
        )):
            if i not in annotate_set:
                result = dict(row)
                skipped_qwen_count += 1
            else:
                d1 = str(row.get("definition1", "")).strip()
                d2 = str(row.get("definition2", "")).strip()
                w1 = str(row.get("word1", "")).strip()
                w2 = str(row.get("word2", "")).strip()
                s1 = str(row.get("sample_sentence1", "")).strip()
                s2 = str(row.get("sample_sentence2", "")).strip()
                idx = row.get("idx")

                if not d1 or not d2:
                    result = {
                        **row,
                        "predicted_label": "CANNOT DECIDE",
                        "model_output": "",
                        "error": "Missing definition1/definition2",
                    }
                else:
                    prompt = build_prompt(
                        instruction_text,
                        d1,
                        d2,
                        word1=w1,
                        word2=w2,
                        sample_sentence1=s1,
                        sample_sentence2=s2,
                    )
                    model_output = generate_label(
                        tokenizer=tokenizer,
                        model=model,
                        prompt=prompt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                    predicted_label = normalize_label(model_output)
                    canonical_output = (
                        f"LABEL: {predicted_label}"
                        if predicted_label in LABELS
                        else model_output
                    )

                    result = {
                        **row,
                        "idx": idx,
                        "predicted_label": predicted_label,
                        "model_output": canonical_output,
                    }

                annotated_count += 1

            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            if args.fsync_each_row:
                os.fsync(out.fileno())
            count += 1

    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Processed rows: {count}")
    if binary_sample_mode:
        print(f"Mode: binary sample (skip existing qwen labels)")
        print(f"Rows sent to LLM: {annotated_count}")
        print(f"Rows kept from qwen: {skipped_qwen_count}")
    else:
        print(f"Rows sent to LLM: {annotated_count}")


if __name__ == "__main__":
    main()
