# CLI reference

All commands run from the repository root. The recommended entry point is the shell wrapper. The Python entry points underneath it are listed for single-variant or single-stage use.

| Layer | Entry point | Use it for |
|---|---|---|
| Wrapper | `scripts/run_experiment_variants.sh` | One meeting or a corpus, one or more variants, shared prep |
| Pipeline | `scripts/run_meeting_pipeline.py` | One meeting, one variant (prep → simulation → evaluation) |
| Pipeline | `scripts/run_batch_meetings.py` | Many meetings, one variant |
| Stage 1 | `src/data_pipeline.py` | Preparation only |
| Stage 2 | `src/participant_simulation.py` | Simulation for one participant |
| Stage 3 | `scripts/capa_eval.py` | Metrics for one meeting |
| Corpus | `scripts/aggregate_corpus.py`, `scripts/aggregate_combined.py` | Corpus statistics with bootstrap CIs |

Every Python entry point supports `--help`.

## `run_experiment_variants.sh`

```text
scripts/run_experiment_variants.sh single --input <meeting.json> --output-root <root> [options]
scripts/run_experiment_variants.sh batch  --input-dirs <dir1,dir2> --output-root <root> [options]
```

Output: `<root>/_prep/<meeting>/` holds the shared Stage 1 artifacts. `<root>/<variant>/<meeting>/` holds the Stage 2–3 outputs for each variant.

### Core options

| Option | Default | Description |
|---|---|---|
| `--input PATH` | – | Single mode: one transcript JSON |
| `--input-dir DIR` (repeatable), `--input-dirs CSV` | – | Batch mode: directories containing `*_parsed.json` |
| `--output-root DIR` | – (required) | Root for prep and results |
| `--shared-prep-root DIR` | `<output-root>/_prep` | Reuse an existing preparation, e.g. for ablations |
| `--variants CSV` | `capa` | `capa`; baselines `mono-raw`, `mono-raw-reflection`, `raw-modular` |
| `--model NAME` | `gpt-4o-openai` | Model alias from `src/config_extended.json` |
| `--participants CSV` | `all` | Restrict Stage 2–3 to some participants (Stage 1 still prepares all of them) |
| `--python BIN` | `$PYTHON`, else `python` | Interpreter for the Python entry points |

### Experiment knobs

| Option | Default | Description |
|---|---|---|
| `--capa-N INT` | `20` | Preceding utterances per episode (raw context window N) |
| `--capa-k INT` | `5` | Evaluation window k (also the runtime window) |
| `--cap-meeting INT` | `20` | Maximum episodes per meeting |
| `--cap-per-participant INT` | `5` | Maximum episodes per participant |
| `--min-gap INT` | `0` | Minimum utterance gap between selected cues per participant |
| `--raw-context-N INT` | `50` | Raw transcript window for the baselines |
| `--no-recalibration` | off | CAPA only: disable the state patch and guidance |

### Run control

| Option | Mode | Description |
|---|---|---|
| `--dry-run` | both | Print the commands without running them |
| `--force` | both | Recompute selected meetings |
| `--resume` / `--no-resume` | batch | Skip complete meetings (default: resume). Single mode always resumes |
| `--meetings CSV` | batch | Run only these meeting IDs |
| `--skip CSV` | batch | Exclude exact IDs, inclusive ranges (`Meeting90train..Meeting160train`), or globs (`Meeting1*train`) |
| `--limit INT` | batch | Only the first N meetings |
| `--skip-generation`, `--skip-simulation`, `--skip-evaluation` | single | Skip a stage and reuse its outputs |
| `--debug-max-utterances INT` | single | Truncate the transcript to its first N utterances (for smoke tests) |

### Advanced preparation

| Option | Description |
|---|---|
| `--freeze-anchor-inventory` | Freeze the anchor inventory before rebuilding k-specific episodes (k-study) |
| `--anchor-inventory-root DIR` | Use frozen anchor inventories (`<DIR>/<meeting>/capa_anchor_inventory/`) |
| `--anchor-seed-k INT` | Reference k used when freezing anchors (default `5`) |
| `--static-prep-root DIR` | Reuse profiles and briefings from another prep root |
| `--eim-cache-root DIR` | Share the Perceiver cache across runs |

**Multi-directory batch rules:**

- Directory order is preserved, and meetings are sorted within each directory.
- `--meetings` and `--skip` apply across all directories.
- A meeting ID that appears in more than one directory aborts the run.

## Direct Python entry points

One meeting, one variant:

```bash
python scripts/run_meeting_pipeline.py \
  --input data/ami/test/Meeting32test_parsed.json \
  --output-dir results/capa/Meeting32test \
  --shared-prep-dir results/_prep/Meeting32test \
  --delegate-variant capa \
  --model gpt-4o-openai
```

Many meetings, one variant:

```bash
python scripts/run_batch_meetings.py \
  --input-dirs data/ami/test,data/ami/val \
  --output-dir results \
  --shared-prep-root results/_prep \
  --delegate-variant capa \
  --model gpt-4o-openai \
  --resume
```

> Called directly, `run_batch_meetings.py` only skips completed meetings with `--resume`. The shell wrapper passes this by default.

### Stage 1: preparation

```bash
python src/data_pipeline.py \
  --input data/ami/test/Meeting32test_parsed.json \
  --output results/_prep/Meeting32test \
  --meeting-id Meeting32test \
  --mode capa --selector llm-idea \
  --capa-N 20 --capa-k 5 \
  --debug-capa-episodes --jit-enable-sliding-windows --jit-enable-llm-cleanup \
  --model gpt-4o-openai --resume
```

These are the flags `run_meeting_pipeline.py` passes. `--participants "Project Manager,Marketing"` prepares a subset.

### Stage 2: simulation (one participant)

```bash
python src/participant_simulation.py \
  --participant "Project Manager" --meeting Meeting32test \
  --agent-type two --mode capa --delegate-variant capa \
  --capa-episodes-dir results/_prep/Meeting32test/capa_episodes \
  --capa-eim-dir results/_prep/Meeting32test/capa_eim_states \
  --briefings-dir results/_prep/Meeting32test/jit_briefings \
  --user-profiles-dir results/_prep/Meeting32test/user_profiles \
  --output-dir results/capa/Meeting32test \
  --enable-artifact-tracking --model gpt-4o-openai --resume
```

For the raw baselines (`--delegate-variant mono-raw|mono-raw-reflection|raw-modular`), also pass `--raw-context-N 50 --transcript-path data/ami/test/Meeting32test_parsed.json`.

### Stage 3: evaluation

```bash
python scripts/capa_eval.py \
  --meeting Meeting32test \
  --results-dir results/capa/Meeting32test \
  --episodes-dir results/_prep/Meeting32test/capa_episodes \
  --mode idea
```

This stage makes no LLM calls. It reads the per-step artifacts of Stage 2 and writes `evals/<meeting>/`:

- `step_metrics.csv`
- `episode_metrics.csv`
- `idea_unit_metrics.csv`
- `participant_metrics.csv`
- `participants.jsonl`
- `meeting_metrics.{json,csv}`

`--participants` restricts evaluation to some participants. `--strict-missing-artifacts` fails if any artifact is missing; the pipeline sets it.

### Aggregation

```bash
python scripts/aggregate_corpus.py --root results/capa                  # -> results/capa/aggregated_evals/
python scripts/aggregate_combined.py --roots <root1> <root2> --out <dir> # combine separately-run roots
```

Both scripts compute macro-averages over meetings with 95% percentile-bootstrap CIs (`--bootstrap 10000`, `--seed 37`). Latency metrics are reported as median and IQR.

## Resume semantics

- Each stage records a manifest: `prep_manifest.json`, `simulation_manifest.json`, `eval_manifest.json`. A stage is reused only if its manifest matches the current configuration.
- A meeting counts as complete only after all three stages succeed. Partial or failed meetings are retried.
- Use `--force` to recompute.

## Operational notes

- Raw baselines reuse CAPA's Stage 1 preparation (profiles, briefings, episodes) and see only the raw transcript at runtime.
- To parallelize, run disjoint `--meetings` subsets into the same output root.
- The wrapper works with bash ≥ 3.2, including the macOS system bash.
