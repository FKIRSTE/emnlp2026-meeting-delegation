# Experiments

Commands for the main comparison, the ablations, the sensitivity studies, other backbones, and the cost analysis. Run everything from the repository root after the [installation](../README.md#1-install).

## Before you start

- **Backbone.** All main experiments use GPT-4o (`--model gpt-4o-openai`, the wrapper default). The paper used snapshot `2024-11-20`. To pin it, set `"model": "gpt-4o-2024-11-20"` in `src/config_extended.json` (see *Configuring backbones* in the [README](../README.md#1-install)).
- **Defaults.** All experiments share these settings, which are the wrapper defaults, so you do not need to pass them:
  - N = 20 preceding utterances
  - k = 5 evaluation window
  - at most 20 episodes per meeting and 5 per participant
  - a 50-turn raw window for the baselines
  - bootstrap with B = 10,000
- **Matched subset.** The ablations and sensitivity studies use the 20 test meetings, `data/ami/test` (`Meeting15test` … `Meeting34test`).
- **Shared preparation.** All variants under one `--output-root` reuse the same Stage 1 artifacts in `<root>/_prep/`. Systems are therefore compared on identical profiles, briefings, and episodes.
- **Resumability.** Batch runs resume by default. Interrupted or failed meetings are retried, and complete meetings are skipped.
- **Dry runs.** Use `--dry-run` to inspect the planned commands, and `--limit` to try a few meetings first.
- **Outputs.** `scripts/aggregate_corpus.py --root <root>/<variant>` writes these files to `<root>/<variant>/aggregated_evals/`:
  - `paper_metrics.csv`: mean and 95% CI per metric
  - `corpus_summary.json`: the same statistics as JSON
  - `aggregated_meeting_metrics.csv`: one row per meeting

  The README maps the [output keys to the paper's metric names](../README.md#evaluation).

```bash
ALL_AMI="data/ami/test,data/ami/val,data/ami/train"   # 137 meetings
SUBSET_DIR="data/ami/test"                            # matched 20-meeting subset
```

## 1. Main comparison

CAPA against the transcript-only and Reflexion-style baselines on all 137 meetings:

```bash
scripts/run_experiment_variants.sh batch \
  --input-dirs "$ALL_AMI" \
  --output-root results/ami \
  --variants capa,mono-raw,mono-raw-reflection

for v in capa mono-raw mono-raw-reflection; do
  python scripts/aggregate_corpus.py --root results/ami/$v
done
```

| Output | Location |
|---|---|
| Outcome rates, Decision F1, safety rates | `results/ami/<variant>/aggregated_evals/paper_metrics.csv`: `idea_strict_recall_rate`, `idea_loose_only_rate`, `idea_bad_attempt_rate`, `idea_ignored_rate`, `gt_idea_action_f1_micro`, `hallucination_rate_speaks` |
| Quality scores | the `delegate_*_mean` rows of the same file |
| Per-unit timing | `results/ami/<variant>/<meeting>/evals/<meeting>/idea_unit_metrics.csv` |
| Per-participant metrics | `.../evals/<meeting>/participant_metrics.csv` |
| Delegate Judge verdicts per step | `.../artifacts/capa_artifacts_*.json` |

If the three splits were run into separate output roots, combine their per-meeting metrics:

```bash
python scripts/aggregate_combined.py \
  --roots results/ami_test/capa results/ami_val/capa results/ami_train/capa \
  --out results/ami_combined/capa
```

## 2. Meeting-state ablation

`raw-modular` keeps the modular Controller/helper pipeline but replaces *m*<sub>t</sub> and recalibration with a 50-turn raw window:

```bash
scripts/run_experiment_variants.sh batch \
  --input-dir "$SUBSET_DIR" \
  --output-root results/subset \
  --variants capa,raw-modular

python scripts/aggregate_corpus.py --root results/subset/capa
python scripts/aggregate_corpus.py --root results/subset/raw-modular
```

## 3. Recalibration ablation

The variant without recalibration keeps *m*<sub>t</sub>, the Predictor, and both judges. Write it to its own output root, but reuse the preparation of the matched subset:

```bash
scripts/run_experiment_variants.sh batch \
  --input-dir "$SUBSET_DIR" \
  --output-root results/subset_no_recal \
  --shared-prep-root results/subset/_prep \
  --variants capa --no-recalibration

python scripts/aggregate_corpus.py --root results/subset_no_recal/capa
```

## 4. Raw context window N

Hold the episode targets fixed by freezing the anchors of the N = 20 run from §2, then rerun CAPA with N ∈ {10, 30}:

```bash
python scripts/context_window_study/freeze_anchors_from_episodes.py \
  --input-dir "$SUBSET_DIR" \
  --source-prep-root results/subset/_prep \
  --out-root results/n_study/anchors \
  --inventory-reference-k 5

for N in 10 30; do
  scripts/run_experiment_variants.sh batch \
    --input-dir "$SUBSET_DIR" \
    --output-root results/n_study/N$N \
    --variants capa --capa-N $N \
    --anchor-inventory-root results/n_study/anchors
  python scripts/aggregate_corpus.py --root results/n_study/N$N/capa
done
```

## 5. Evaluation window k

The k-study fixes a reference horizon k<sub>ref</sub> = 9, freezes the anchor inventory once, and rebuilds per-k episodes for k ∈ {3, 5, 7, 9} on the same targets. It runs on the test split by default. The scripts must run in order; see [`scripts/k_window_study/README.md`](../scripts/k_window_study/README.md).

```bash
bash scripts/k_window_study/01_reduced_canonical_freeze.sh   # optional smoke test (1 truncated meeting)
bash scripts/k_window_study/02_reduced_sweep.sh              # optional smoke test
bash scripts/k_window_study/03_single_meeting_full.sh        # optional: one full meeting
bash scripts/k_window_study/04_batch_canonical_freeze.sh     # canonical prep for the 20 test meetings
bash scripts/k_window_study/05_batch_run.sh                  # validate prep, run all k
bash scripts/k_window_study/06_batch_aggregate.sh            # -> $KSTUDY_BASE/batch/summary/k_window_summary.{csv,json}
```

## 6. Other backbones

**Gemini-2.5-Pro on the matched subset.** Keep participant briefings and episodes from the GPT-4o run in §2 fixed, and regenerate only the model-visible Perceiver states with Gemini. Then run CAPA on that prep with Stage 1 skipped. Stage 1's reuse check includes the model name, so a normal run would rebuild the briefings with the new backbone.

```bash
for f in $SUBSET_DIR/*_parsed.json; do
  M=$(basename "$f" _parsed.json)
  python scripts/regenerate_capa_eim_from_prep.py \
    --input "$f" \
    --source-prep-dir results/subset/_prep/$M \
    --output-prep-dir results/backbone_gemini/_prep/$M \
    --model gemini
  scripts/run_experiment_variants.sh single \
    --input "$f" \
    --output-root results/backbone_gemini \
    --variants capa --model gemini \
    --skip-generation
done

python scripts/aggregate_corpus.py --root results/backbone_gemini/capa
```

**Llama-3.3-70B and Qwen3.6-27B via Groq:**

```bash
for M in llama-3.3-groq qwen-3.6-groq; do
  scripts/run_experiment_variants.sh batch \
    --input-dir "$SUBSET_DIR" \
    --output-root results/backbone_$M \
    --variants capa --model $M
  python scripts/aggregate_corpus.py --root results/backbone_$M/capa
done
```

## 7. Latency, tokens, and cost

Every LLM call writes a telemetry record under `llm_telemetry/` in the prep and result directories. To summarize per-module calls, latency, tokens, and cost per decision:

```bash
# Pass the llm_telemetry/ folders only (directories are searched recursively for *.jsonl)
python scripts/summarize_llm_telemetry.py results/subset/capa/*/llm_telemetry results/subset/_prep/*/llm_telemetry \
  --output-json results/telemetry_gpt4o.json --output-csv results/telemetry_gpt4o.csv

# Open-weight pricing from Appendix B.4 (Qwen shown; Llama: 0.59 / 0.79)
python scripts/summarize_llm_telemetry.py results/backbone_qwen-3.6-groq/capa/*/llm_telemetry \
  --input-usd-per-million 0.60 --cached-input-usd-per-million 0.60 --output-usd-per-million 3.00
```

The defaults are GPT-4o rates: $2.50 per million uncached input tokens, $1.25 per million cached input tokens, and $10 per million output tokens. For reference, the paper reports about $0.21 per decision for the full architecture with GPT-4o (Table 5).
