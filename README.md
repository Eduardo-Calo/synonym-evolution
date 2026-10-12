# Investigating the Laws of Synonymy Evolution

This repository contains the official code accompanying the paper _Investigating the Laws of Synonymy Evolution through a Fine-Grained Lemma-Based Framework_ (EMNLP 2026).

It contains the pipeline for extracting lexical units, computing embedding-based similarity, selecting candidate pairs, and running semantic-change analyses.

The code is published mainly for reproducibility purposes.

## Data note

The pipeline uses the Oxford English Dictionary (OED) as its source data. We cannot redistribute the OED-derived inputs or outputs, so reproducing the pipeline requires obtaining the appropriate OED license.

## Layout

- `src/` contains the scripts.
- `data/raw/` holds the private input JSONL and other source material.
- `data/parsed/` stores per-POS CSVs produced by the first stages.
- `data/embedding/` stores definition embeddings as `.npy` files.
- `data/similarities/` stores blockwise cosine similarity matrices and metadata.
- `data/histograms/` stores aggregated similarity summaries.
- `data/extracted/` stores selected pair CSVs.
- `data/sample_pairs/` stores sampled pair reports.
- `data/plots/` stores summary tables and downstream analysis outputs.
- `data/llm_eval/` stores JSONL files prepared from extracted pairs for LLM labeling.
- `data/llm_output/` stores LLM prediction outputs.
- `data/exp_eval/` stores sampled JSONL files used to build human evaluation surveys.
- `data/hum_eval/` stores human-evaluation JSONL inputs.
- `data/surveys/` stores exported survey files and parsed survey results.

## Setup

Run the commands from `src/` so the relative paths that point into `../data/...` resolve correctly.

Create and activate a virtual environment before installing dependencies:

```bash
cd oed_testset/
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cd src/
```

Create a local `src/qualtrics_config.json` for Qualtrics/Prolific scripts. This file is private and intentionally not tracked in git.

## Core pipeline

The shortest end-to-end path is below. Replace the private raw input path with the local source file you are allowed to use.

```bash
cd oed_testset/src/
```

1. Extract lexical units into per-POS CSV files.

```bash
python3 extract_lus.py ../data/raw/<licensed-oed-jsonl> ../data/parsed
```

2. Add WordNet synset information for the supported coarse POS values.

```bash
python3 extract_synset_info.py --parsed-dir ../data/parsed
```

3. Create definition embeddings.

The cluster wrapper is configured for a single POS value and submits a SLURM job.

```bash
sbatch run_embeddings.sh
```

To run the Python stage directly for a specific POS:

```bash
python3 embed_definitions.py --input-dir ../data/parsed --output-dir ../data/embedding --pos <POS>
```

4. Compute blockwise cosine similarities.

The cluster wrapper is also configured for a single POS value.

```bash
sbatch run_similarity.sh
```

Direct execution looks like this:

```bash
python3 compute_similarities.py --pos <POS> --block-size 16384
```

5. Merge similarity blocks into a histogram summary.

```bash
python3 merge_similarities.py --pos <POS>
```

Optional plotting:

```bash
python3 plot_similarities.py --pos <POS>
```

6. Select candidate lemma pairs from the similarity blocks.

```bash
python3 select_pairs.py --pos <POS> --threshold-min 0.8 --membership both
```

Batch selection across multiple POS or thresholds:

```bash
python3 batch_select_pairs.py --pos <POS1> <POS2> <POS3> --threshold_mins 0.7 0.8 0.9 --memberships both
```

## LLM and human evaluation workflow

Prepare LLM input files from extracted pairs (`data/extracted/` -> `data/llm_eval/`):

```bash
python prepare_hf_input.py --thr-start <THR_START> --thr-end <THR_END> [--chunk-size CHUNK_SIZE]
```

Run LLM annotation on one JSONL file:

```bash
sbatch run_llm.sh ../data/llm_eval/{pos}.jsonl
```

`run_llm.sh` executes:

```bash
python3 hf_annotate_similarity.py \
	--input ../data/llm_eval/{pos}.jsonl \
	--model models--Qwen--Qwen3.5-27B \
	--trust-remote-code
```

Alternative input example:

```bash
python3 hf_annotate_similarity.py \
	--input ../data/exp_eval/{pos}/{pos}_sample_binary.jsonl \
	--model models--Qwen--Qwen3.5-27B \
	--trust-remote-code
```

For folders with batched JSONL files:

```bash
chmod +x run_llm_all.sh
./run_llm_all.sh ../data/llm_eval/{pos}_thr0.8_0.85
```

Select pairs to show annotators for human evaluation:

```bash
python sample_stratified_binary.py <POS1> <POS2> <POS3> --low-cos-csv ../data/extracted/{pos}_pairs_thr0.0_0.85_both_binsz0.1_n10000.csv
```

Create a Qualtrics survey from sampled JSONL:

```bash
python qualtrics.py ../data/exp_eval/{pos}/{pos}_sample_binary.jsonl
```

Export one survey:

```bash
python qualtrics_export_survey.py <survey_id>
```

Aggregate survey results and compare against Qwen labels:

```bash
python surveys_results.py ../data/surveys/<survey_id> --qwen-jsonl ../data/llm_output/Qwen_Qwen3_5-27B/ADJ_sample_binary_hf_labels_Qwen_Qwen3_5-27B.jsonl
```

## Validation workflow (threshold methodology)

```bash
python select_pairs.py --pos ADJ --threshold-min 0.0 --threshold-max 0.85 --bin-size 0.1 --samples-per-bin 10000 --random-seed 42 --workers 8 --rows-per-task 1024
```

```bash
python sample_stratified_binary.py ADJ --low-cos-csv ../data/extracted/ADJ_pairs_thr0.0_0.85_both_binsz0.1_n10000.csv --qwen off --n-similar 0 --n-dissimilar 0 --n-low-cos 4000 --out-dir ../data/llm_eval/
```

```bash
python count_per_bin.py --model Qwen_Qwen3_5-27B
```

`count_per_bin.py` reports binned percentages (`similar/total`) with a default bin size of 0.05 and writes plots under `data/plots/`.

Publish to Prolific:

```bash
python prolific_publish.py <jsonl_file> --total_available_places <N> --estimated_completion_time <M> --prolific-reward <P>
```

## Semantic change scripts

These scripts perform full semasiological semantic change analysis using the project taxonomy between two time frames:

- `compute_change.py`
- `compute_control.py`
- `compute_per_lemma.py`

1. Compute semantic-change summaries for the selected pairs.

```bash
sbatch run_change.sh --mode change
```

To submit the full four-way batch of change, control, and lemma analyses:

```bash
sbatch run_change_all.sh
```

The batch driver that `run_change.sh` calls is `batch_compute_change.py`, which can also be run directly for custom POS, thresholds, or year ranges.

Batch execution examples:

```bash
python batch_compute_change.py --mode lemma --lemma_mode syn
```

Downstream analysis scripts:

```bash
python compute_stats.py
python analyze_stability.py
```

## Quick verification checklist

- `data/parsed/` contains per-POS CSV files after extraction.
- `data/parsed/*_wn.csv` exists after synset enrichment.
- `data/embedding/` contains one `.npy` file per processed POS.
- `data/similarities/<POS>/` contains `sims_*.npy` and `meta_*.txt` block outputs.
- `data/histograms/<POS>/results.npz` exists after merging similarities.
- `data/extracted/` contains the pair CSVs selected from the similarity blocks.
- `data/sample_pairs/` and `data/plots/` contain the downstream change-analysis artifacts.

## Citation

```TBD```