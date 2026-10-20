# Scripts

Run every script from the repository root. Each script adds `src/` to the import path itself, and `--help` lists all options. Experiment commands are in [`docs/experiments.md`](../docs/experiments.md).

## Running experiments

| Script | Purpose |
|---|---|
| [`run_experiment_variants.sh`](run_experiment_variants.sh) | **Recommended entry point.** Runs one meeting (`single`) or a corpus (`batch`), with shared preparation across variants |
| [`run_meeting_pipeline.py`](run_meeting_pipeline.py) | One meeting, one variant: Stage 1 prep → Stage 2 simulation → Stage 3 evaluation. Resumable |
| [`run_batch_meetings.py`](run_batch_meetings.py) | Calls `run_meeting_pipeline.py` for every meeting in one or more input directories |
| [`check_llm_setup.py`](check_llm_setup.py) | Loads the configuration and makes a live test call to the active model |

## Evaluation and aggregation

| Script | Purpose |
|---|---|
| [`capa_eval.py`](capa_eval.py) | Stage 3: per-meeting metrics from the Stage 2 artifacts (no LLM calls) |
| [`aggregate_corpus.py`](aggregate_corpus.py) | Corpus statistics over all meetings under a root: mean, SD, and 95% bootstrap CI (median/IQR for latency) |
| [`aggregate_combined.py`](aggregate_combined.py) | Combines several `aggregate_corpus.py` outputs (e.g. train + val + test) into one corpus |
| [`summarize_llm_telemetry.py`](summarize_llm_telemetry.py) | Per-module LLM calls, latency, tokens, and cost per decision |

## Robustness studies

| Script | Purpose |
|---|---|
| [`k_window_study/`](k_window_study/) | Evaluation-window sensitivity, k ∈ {3, 5, 7, 9}. See its [README](k_window_study/README.md) |
| [`k_window_study.py`](k_window_study.py) | Helper for the k-study shell scripts: shard planning, batch commands, prep validation, summary |
| [`context_window_study/freeze_anchors_from_episodes.py`](context_window_study/freeze_anchors_from_episodes.py) | Freezes final episodes into anchor inventories, so the N ∈ {10, 20, 30} sweep scores the same targets |
| [`regenerate_capa_eim_from_prep.py`](regenerate_capa_eim_from_prep.py) | Copies a finished prep and regenerates only the Perceiver states with another backbone |

## Data utilities

| Script | Purpose |
|---|---|
| [`convert_csv_to_json.py`](convert_csv_to_json.py) | Converts speaker-labelled CSV meetings to the transcript JSON format |
| [`analyze_ami_stats.py`](analyze_ami_stats.py) | Utterance statistics for the AMI splits in `data/ami/` |
