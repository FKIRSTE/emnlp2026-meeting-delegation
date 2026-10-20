# k-window study

These scripts run the evaluation-window sensitivity study (paper Appendix I.4). Each script changes to the repository root itself, so you can run it from anywhere.

Protocol:

- fix `k_ref = 9` as the benchmark-construction horizon
- freeze the canonical anchor inventory once
- rebuild per-`k` episodes from that fixed inventory
- compare `k ∈ {3,5,7,9}` on the same canonical target set

Run order:

1. `scripts/k_window_study/01_reduced_canonical_freeze.sh`
2. `scripts/k_window_study/02_reduced_sweep.sh`
3. `scripts/k_window_study/03_single_meeting_full.sh`
4. `scripts/k_window_study/04_batch_canonical_freeze.sh`
5. `scripts/k_window_study/05_batch_run.sh`
6. `scripts/k_window_study/06_batch_aggregate.sh`

Steps 1–3 are optional smoke tests on a single meeting. Steps 4–6 produce the table.

Defaults are defined in `scripts/k_window_study/common.sh`. Override them per run through environment variables:

```bash
KSTUDY_BASE=results/my_k_study \
MODEL=gpt-4o-openai \
REDUCED_MEETING=Meeting32test \
bash scripts/k_window_study/01_reduced_canonical_freeze.sh
```

| Variable | Default | Meaning |
|---|---|---|
| `KSTUDY_BASE` | `results/k_window_study` | Output root |
| `MODEL` | `gpt-4o-openai` | Model alias |
| `INPUT_DIR` | `data/ami/test` | Meetings to use |
| `BATCH_LIMIT` | `20` | Number of meetings in the batch stages |
| `N` | `20` | Preceding utterances per episode |
| `K_REF` | `9` | Reference horizon for anchor freezing |
| `K_VALUES_CSV` | `3,5,7,9` | Tested evaluation windows |
| `REDUCED_MEETING`, `REDUCED_DEBUG_UTTS` | `Meeting33test`, `120` | Smoke-test meeting and truncation |
| `FULL_MEETING` | `Meeting33test` | Single full-meeting check |
| `PY` | active venv, `.venv/`, else `python3` | Python interpreter |

Script roles:

- `01_reduced_canonical_freeze.sh`
  - Builds a reduced canonical prep root for one debug-truncated meeting using `k_ref`.
  - Skips simulation and evaluation.
- `02_reduced_sweep.sh`
  - Rebuilds reduced per-`k` episodes from the reduced canonical inventory and runs the reduced smoke sweep.
- `03_single_meeting_full.sh`
  - Runs the full one-meeting check: canonical freeze plus per-`k` sweep and summary.
- `04_batch_canonical_freeze.sh`
  - Freezes canonical prep for the full meeting set up to `BATCH_LIMIT`.
- `05_batch_run.sh`
  - Validates canonical prep, then emits and executes the shard-wise batch commands for all tested `k`.
- `06_batch_aggregate.sh`
  - Aggregates all batch conditions and writes the final summary table and figure.

Study-only reuse behavior is automatic in the later scripts:

- canonical seed prep provides frozen anchors and static prep reuse
- final per-`k` prep roots remain separate
- only the internal EIM cache is shared across `k` for the batch/sweep stages

Batch preflight policy:

- missing canonical `capa_anchor_inventory` for any selected meeting is a hard blocker
- missing canonical `user_profiles` or `jit_briefings` only generates warnings
- canonical `capa_eim_states` are not required for the later batch sweep
- the default validation report path is `results/k_window_study/batch/canonical_prep_validation.json`

Outputs: `06_batch_aggregate.sh` writes `k_window_summary.{csv,json}`, `k_window_table.md`, and `k_window_tradeoff.png` to `$KSTUDY_BASE/batch/summary/`.
