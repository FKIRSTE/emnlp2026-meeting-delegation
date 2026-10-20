# Architecture: paper → code

This page maps each part of CAPA as described in the paper (§3.3, Figure 1, Appendix J) to its implementation, and describes how one decision is executed.

## Naming

| Paper | Code |
|---|---|
| Perceiver | Environment Interpretation Module, **EIM** (`EnvironmentInterpretationModule`) |
| Meeting state *m*<sub>t</sub> | `EIMMasterState` / `EIMMeetingStateVector` (`src/schemas.py`) |
| Session briefing *b*<sub>u</sub> | **JIT briefing** (`jit_briefings/`) |
| Participant profile *p*<sub>u</sub> | User profile (`user_profiles/`) |
| Controller + Generator | Two-tier delegate (`TwoTierUserModule`) |
| Environment Judge | `CapaJudge` (prediction judge) |
| Delegate Judge | `CapaDelegateJudge` |
| Guidance | `ControllerGuidance` / `GuidanceBundle` (`src/capa/types.py`) |
| Idea unit / episode | `IdeaUnit` / `CAPAEpisode` (`src/schemas.py`) |
| Transcript-only delegate | `mono-raw` variant (`MonoRawAgent`) |
| Reflexion-style baseline | `mono-raw-reflection` variant (`MonoRawReflectionAgent`) |
| State-ablated CAPA | `raw-modular` variant (`RawModularAgent`) |

## Components

### Shared Memory: `src/shared_memory.py`

`SharedMemoryModule` holds the four stores from the paper's 2 × 2 partition:

| Store | Paper | Contents |
|---|---|---|
| Long-term memory | profile *p*<sub>u</sub> | Role, expertise, goals, communication style |
| Session memory | briefing *b*<sub>u</sub> | Candidate talking points for this meeting |
| Working memory | meeting state *m*<sub>t</sub> | Active topic, decisions, open questions, participant stances, coverage, floor |
| Episodic memory | action trace | The delegate's prior actions; recalibration guidance is stored alongside |

The state fields are schema-validated Pydantic models in `src/schemas.py`. Examples include `EIMMeetingStateVector`, `TopicLogEntry` (decisions, open questions, key points), `EIMParticipantModel` (stances), and `SnapshotAnalysis` (floor status). Per-candidate coverage is tracked as `CandidateCoverageStatus`.

### Offline preparation (Stage 1): `src/data_pipeline.py` → `src/prep_resume.py:run_capa_prep`

| Artifact | Code | Output |
|---|---|---|
| Profile *p*<sub>u</sub>, built from the first 60% of the meeting | `UserProfileGenerator.generate_rich_user_profile` (`src/user_profile_generator.py`) | `user_profiles/` |
| Briefing *b*<sub>u</sub> | `JITBriefingGenerator.generate_briefings_for_meeting` (`src/jit_briefing_generator.py`), with optional sliding windows and an LLM clean-up pass | `jit_briefings/` |
| Idea units | `select_idea_units` (`src/capa/idea_unit_selector_llm.py`) | inside `capa_episodes/` |
| Episodes (cue, anchor, k-window targets) | `src/capa/episode_utils.py` (`build_official_idea_targets`, …), `src/capa/anchor_inventory.py` | `capa_episodes/`, `capa_anchor_inventory/` |
| Perceiver state at each cue | `EnvironmentInterpretationModule.process_history` + `analyze_snapshot` | `capa_eim_states/`, `_eim_cache/` |

### Online loop (Stage 2): `src/participant_simulation.py` → `src/two_tier_agent.py`

`ParticipantSimulation.run_capa_simulation` replays every episode of one participant. For each episode, `_run_capa_episode_with_tracking` advances through the *k* steps after the cue. At every step it calls `TwoTierUserModule.run_turn` with only the causally observed prefix.

```text
for each step t in the k-window after the cue:
  PERCEIVE     m_t ← state at the cue (Stage 1), patched by the Recalibrator after each prior step
  ACT          helpers:    candidate curator → coverage evaluator → speaker scorer
               Controller: SILENT | SPEAK + proposition z_t   (reads m_t, p_u, b_u, guidance)
               Generator:  z_t → utterance y_t in the participant's style (on SPEAK only)
  PREDICT      Predictor:  intent-level forecast of the next turn from the pre-action state
               (hidden from the action policy; used only for diagnosis)
  observe the ground-truth continuation
  RECALIBRATE  Environment Judge: forecast vs. observed     → perception error
               Delegate Judge:    action vs. observed       → action error, credit, safety flags
               Recalibrator:      state patch on m_t (anchored to the observed turn) + guidance for t+1
```

| Step | Code |
|---|---|
| Helpers | `TwoTierUserModule._run_helper_sequence`: `candidate_curator`, `coverage_evaluator`, `speaker_scorer` |
| Controller | `TwoTierUserModule.controller_phase`, plus rule-based guards such as `_apply_participant_stance_consistency_guard` and `_apply_safe_controller_fallback` |
| Generator | `TwoTierUserModule.generator_phase` |
| Predictor | `CapaPredictor.predict` (`src/capa/predictor.py`) |
| Environment Judge | `CapaJudge.judge_prediction` (`src/capa/judge.py`) |
| Delegate Judge | `CapaDelegateJudge.judge_delegate` (`src/capa/delegate_judge.py`), with a ground-truth match verifier and a redundancy verifier |
| Recalibrator | `CapaRecalibrator.build_fusion` (state update) and `_generate_guidance` (feedback coach) (`src/capa/recalibrator.py`) |
| Loop orchestration | `CapaDriver.step` (`src/capa/driver.py`) |

**Causality.** The helpers, Controller, and Generator receive only the observed dialogue prefix, the participant context, and the shared memory. The future evaluation window and the episode's idea units are passed only to the two judges and the Recalibrator, after the action. Their verdicts reach later decisions only through the Recalibrator's state patch and guidance (§3.3). Briefing points reach the delegate without the transcript turn IDs they were derived from.

With the default configuration (`recalibrator_drives_state: true`), the Recalibrator's state patch is the per-turn update of *m*<sub>t</sub>. With `--no-recalibration`, the Perceiver refreshes *m*<sub>t</sub> from the observed turn instead. The Predictor and Environment Judge still run, but no guidance is produced (Table 9).

### Baselines: `src/raw_agents.py`, `src/raw_context.py`, `src/raw_prompt_bundle.py`

The raw baselines see a raw transcript window (`--raw-context-N`, default 50) instead of *m*<sub>t</sub>. They are scored by `ShadowCapaEvaluator` (`src/capa/shadow_eval.py`), which runs a hidden Perceiver and the same Delegate Judge. Credit and safety judgments are therefore identical across systems, but the state is never shown to the baseline. Which modules each variant enables is defined in `build_delegate_capabilities` (`src/raw_baseline_support.py`).

### Evaluation (Stage 3): `scripts/capa_eval.py` → `src/evals/`

`src/evals/loader.py` reads the per-step artifacts (`artifacts/capa_artifacts_*.json`). `src/evals/idea_metrics.py` (`idea_strict_loose_recall_for_unit`) assigns each idea unit one of four outcomes: `hit_strict`, `hit_loose_only`, `attempted_but_bad`, `never_attempted`. `src/evals/metrics.py` computes Decision F1 and the quality and safety rates. This stage makes no LLM calls; all LLM judgments are made during Stage 2 and stored in the artifacts.

## Prompts (Appendix J)

All prompts are inline in the module that uses them.

| Appendix | Prompt | Location |
|---|---|---|
| J.1 | User Profile Generator | `src/user_profile_generator.py` (`generate_rich_user_profile`) |
| J.2 | JIT Briefing Generator | `src/jit_briefing_generator.py` (`generate_jit_briefing_for_participant` and the clean-up pass) |
| J.3 | Perceiver | `src/environment_interpreter.py` (`process_history`, `analyze_snapshot`) |
| J.4 | Controller | `src/two_tier_agent.py` (`controller_system_prompt`, `controller_user_prompt_template`) |
| J.5 | Helpers (curator, coverage, speaker scorer) | `src/two_tier_agent.py` (`curator_system_prompt`, `coverage_system_prompt`, `speaker_system_prompt`, per-helper instructions in `_run_helper_sequence`) |
| J.6 | Generator | `src/two_tier_agent.py` (`generator_system_prompt`, `generator_user_prompt_template`) |
| J.7 | Predictor | `src/capa/predictor.py` (`CapaPredictor.predict`) |
| J.8 | Environment Judge | `src/capa/judge.py` (`CapaJudge._build_system_prompt`) |
| J.9 | Delegate Judge | `src/capa/delegate_judge.py` (`_build_main_judge_prompts`, `_build_gt_match_verifier_prompts`, `_build_redundancy_verifier_prompts`) |
| J.10 | Recalibrator | `src/capa/recalibrator.py` (`build_fusion`, `_feedback_coach_prompt`) |
| J.11 | Idea Unit Selector | `src/capa/idea_unit_selector_llm.py` (`render_prompt`) |
| – | Transcript-only / state-ablated baselines | `src/raw_prompt_bundle.py` (`build_mono_raw_system_prompt`, `build_raw_controller_system_prompt`) |
| – | Reflexion-style critic | `src/raw_agents.py` (`MonoRawReflectionAgent`) |

## Configuration

`src/config.py` loads `src/config_extended.json`. Its `models` section holds the provider aliases, and its `capa` section holds runtime switches:

| Key | Shipped value | Effect |
|---|---|---|
| `capa.enabled` | `true` | Enables Predictor, judges, and Recalibrator. If `false`, all four are off |
| `capa.late_window_k` | `5` | Runtime window; the wrappers override it with `--capa-k` |
| `capa.recalibrator_drives_state` | `true` | The Recalibrator's patch is the per-turn state update |
| `capa.guidance_enabled` | `true` | Recalibrator guidance is injected into the next Controller prompt |

Credentials are resolved from the environment variables named in each alias (`api_key_env`, `endpoint_env`, …). `.env` is loaded automatically from the working directory, `src/`, or the repository root, and never overrides variables that are already exported. All LLM calls go through `src/llm_client.py`, which requests schema-constrained JSON from every provider and falls back from `json_schema` to `json_object` where a provider requires it. `src/llm_telemetry.py` records per-call latency and token use under `llm_telemetry/`.
