"""
CLI: Compute CAPA evaluation metrics per eval_plan.md.

Usage:
  python scripts/capa_eval.py --meeting Meeting31 --results-dir results --export both
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to sys.path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from evals.loader import (
    discover_artifact_files,
    load_artifact_file,
    iter_steps_from_artifacts,
    group_steps_by_episode,
)
from evals.metrics import (
    gt_action_following,
    cue_action_congruence,
    cue_expected_speak,
    delegate_gt_hit_strict,
    delegate_gt_hit_loose,
    delegate_unit_hit_strict,
    delegate_unit_hit_loose,
    delegate_hit_strict,
    delegate_hit_loose,
    inventory_compliance,
    recall_na_reason,
)
from evals.idea_metrics import gt_idea_action_following, idea_strict_loose_recall_for_unit
from evals.aggregate import aggregate_confusion, _confusion_to_metrics
from evals.aggregate import compute_episode_metrics, aggregate_meeting
from raw_baseline_support import DELEGATE_JUDGE_REGIME


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _delegate_utterance_present(delegate_utterance: Optional[str]) -> bool:
    return bool(str(delegate_utterance or "").strip())


def _episode_meta_key(participant_id: Optional[str], test_case_id: Optional[str]) -> Optional[str]:
    if not test_case_id:
        return None
    return f"{str(participant_id or '').strip()}\t{str(test_case_id).strip()}"


def _step_episode_key(row: Dict[str, Any]) -> Optional[str]:
    return _episode_meta_key(row.get("participant_id"), row.get("test_case_id"))


def _legacy_duplicate_step_ok(row: Dict[str, Any]) -> bool:
    return bool(row.get("legacy_duplicate_ok") or row.get("legacy_duplicate_step_ok"))


def _assert_unique_step_rows(step_rows: List[Dict[str, Any]]) -> None:
    seen: set[tuple[str, str, int]] = set()
    duplicates: List[str] = []
    for row in step_rows:
        if _legacy_duplicate_step_ok(row):
            continue
        participant_id = str(row.get("participant_id") or "").strip()
        test_case_id = str(row.get("test_case_id") or "").strip()
        delta = row.get("delta")
        if not participant_id or not test_case_id or delta is None:
            continue
        try:
            delta_int = int(delta)
        except Exception:
            continue
        key = (participant_id, test_case_id, delta_int)
        if key in seen:
            duplicates.append(f"{participant_id}:{test_case_id}:d{delta_int}")
        seen.add(key)
    if duplicates:
        sample = ", ".join(duplicates[:12])
        raise ValueError(
            "Duplicate CAPA eval step keys detected. Regenerate prep with unique episode ids before aggregation: "
            f"{sample}"
        )


def _parse_participant_filter(raw_value: Optional[str]) -> List[str]:
    if not raw_value:
        return []
    seen = set()
    participants: List[str] = []
    for item in str(raw_value).split(','):
        participant = item.strip()
        if not participant:
            continue
        key = participant.casefold()
        if key in seen:
            continue
        seen.add(key)
        participants.append(participant)
    return participants


def _slugify_participant_id(participant_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", participant_id.strip()).strip("_")
    return slug or "unknown_participant"


def _participant_scope_slug(participants: List[str], max_len: int = 120) -> str:
    slugs = sorted(_slugify_participant_id(pid) for pid in participants if pid)
    if not slugs:
        return "empty_subset"
    if len(slugs) == 1:
        return slugs[0]
    joined = "__".join(slugs)
    if len(joined) <= max_len:
        return joined
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:10]
    return f"subset_{digest}"


def _load_episode_meta(results_dir: Path, meeting: str, episodes_dir_override: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Load optional episode provenance (cue, indices, tags, idea_units) from capa_episodes.

    Returns a mapping (participant_id, test_case_id) -> meta fields to enrich
    episode_metrics rows.
    Safe to call even if files are missing or diverse in shape.
    Search order (first hit wins):
      1) episodes_dir_override if provided
      2) results_dir / "capa_episodes"
      3) results_dir.parent / "capa_episodes"  (common when artifacts live in results_dir but episodes live alongside)
    """
    out: Dict[str, Dict[str, Any]] = {}
    candidates: List[Path] = []
    if episodes_dir_override:
        candidates.append(episodes_dir_override)
    candidates.append(results_dir / "capa_episodes")
    candidates.append(results_dir.parent / "capa_episodes")
    epi_dir = next((p for p in candidates if p and p.exists()), None)
    if epi_dir is None:
        return out
    try:
        patterns = [f"*from_{meeting}.json", f"*_{meeting}.json"]
        files: List[Path] = []
        for pat in patterns:
            files.extend(list(epi_dir.glob(pat)))
        files = list({str(p.resolve()): p for p in files}.values())
        for p in files:
            try:
                with p.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            if isinstance(data, list):
                eps = data
            elif isinstance(data, dict):
                eps = data.get("episodes") or data.get("capa_test_episodes") or [data]
            else:
                eps = []
            for e in eps or []:
                if not isinstance(e, dict):
                    continue
                tcid = e.get("test_case_id") or e.get("episode_id")
                if not tcid:
                    continue
                represented_user = e.get("represented_user") or e.get("participant") or e.get("participant_id")
                meta = {
                    "test_case_id": tcid,
                    "cue_test_case_id": e.get("cue_test_case_id"),
                    "cue_anchor_id": e.get("cue_anchor_id") or e.get("cue_test_case_id"),
                    "episode_id_suffix": e.get("episode_id_suffix"),
                    "represented_user": represented_user,
                    "cue_utt_id": e.get("cue_utt_id") or e.get("cue_utterance_id") or e.get("cue_utt_id"),
                    "cue_index": e.get("cue_index") or e.get("cue_idx"),
                    "N": e.get("N") or e.get("n"),
                    "k": e.get("k"),
                    "selection_tags": e.get("selection_tags"),
                    "selection_confidence": e.get("selection_confidence"),
                    "selection_reason": e.get("selection_reason"),
                    "rebuilt_from_anchor_inventory": e.get("rebuilt_from_anchor_inventory"),
                    "anchor_inventory_id": e.get("anchor_inventory_id"),
                    "anchor_inventory_reference_k": e.get("anchor_inventory_reference_k"),
                    "preferred_cue_utt_id": e.get("preferred_cue_utt_id"),
                    "preferred_cue_index": e.get("preferred_cue_index"),
                    "preferred_cue_type": e.get("preferred_cue_type"),
                    "preferred_cue_gap": e.get("preferred_cue_gap"),
                    "preferred_cue_retained": e.get("preferred_cue_retained"),
                    "self_initiation_fallback": e.get("self_initiation_fallback"),
                    "cue_mode": e.get("cue_mode"),
                    "effective_cue_gap": e.get("effective_cue_gap"),
                }
                if e.get("idea_units"):
                    meta["idea_units"] = e.get("idea_units")
                key = _episode_meta_key(represented_user, tcid)
                if key in out:
                    raise ValueError(f"Duplicate CAPA episode metadata key for {meeting}: {represented_user}:{tcid}")
                if key:
                    out[key] = meta
    except ValueError:
        raise
    except Exception:
        return out
    return out


def _load_meeting_k_diagnostics_summary(
    results_dir: Path,
    meeting: str,
    episode_meta: Dict[str, Dict[str, Any]],
    episodes_dir_override: Optional[Path] = None,
) -> Dict[str, Any]:
    candidates: List[Path] = []
    if episodes_dir_override:
        candidates.append(episodes_dir_override.parent / "capa_k_diagnostics")
    candidates.append(results_dir / "capa_k_diagnostics")
    candidates.append(results_dir.parent / "capa_k_diagnostics")
    diag_dir = next((path for path in candidates if path.exists()), None)
    if diag_dir is None:
        return {}

    unique_k: List[int] = []
    for meta in (episode_meta or {}).values():
        raw_value = meta.get("k")
        if raw_value is None:
            continue
        try:
            parsed_value = int(raw_value)
        except Exception:
            continue
        if parsed_value not in unique_k:
            unique_k.append(parsed_value)
    unique_k.sort()
    candidate_files: List[Path] = []
    if len(unique_k) == 1:
        candidate_files.append(diag_dir / f"k_diagnostics_summary_for_{meeting}_k{unique_k[0]}.json")
    candidate_files.extend(sorted(diag_dir.glob(f"k_diagnostics_summary_for_{meeting}_k*.json")))

    for path in candidate_files:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                return payload
        except Exception:
            continue
    return {}


def _compute_episode_row(ep_steps, participant: str, latency_bucket, latency_signed) -> Dict[str, Any]:
    """Compute a full episode row from StepRecords, including deltas, accuracies, timing.

    This mirrors the logic previously inlined in main(), but is safe to call
    inside the episode loop so every episode contributes one row.
    """
    # Deltas from environment/delegate judges
    base = compute_episode_metrics(ep_steps, participant)

    # Confusion-derived accuracies by recomputing per-step (cheap at episode scale)
    gt_tp = gt_fp = gt_tn = gt_fn = 0
    cue_tp = cue_fp = cue_tn = cue_fn = 0
    for s in ep_steps:
        g = gt_action_following(s.__dict__, participant)
        c = cue_action_congruence(s.__dict__, participant)
        gt_tp += g["tp"]; gt_fp += g["fp"]; gt_tn += g["tn"]; gt_fn += g["fn"]
        cue_tp += c["tp"]; cue_fp += c["fp"]; cue_tn += c["tn"]; cue_fn += c["fn"]
    def _acc(tp,fp,tn,fn):
        tot = tp+fp+tn+fn
        return ((tp+tn)/tot) if tot else None
    base.update({
        "gt_action_accuracy": _acc(gt_tp, gt_fp, gt_tn, gt_fn),
        "cue_action_accuracy": _acc(cue_tp, cue_fp, cue_tn, cue_fn),
    })

    # Episode latency and timing coverage
    base.update({
        "latency_bucket": latency_bucket,
        "latency_signed": latency_signed,
        "latency_bucket_on_time": 1.0 if latency_bucket == "on_time" else 0.0,
        "latency_bucket_early": 1.0 if latency_bucket == "early" else 0.0,
        "latency_bucket_late": 1.0 if latency_bucket == "late" else 0.0,
        "missed_target_rate": 1.0 if latency_bucket == "missed_target" else 0.0,
        "no_target_spoke_rate": 1.0 if latency_bucket == "no_target_spoke" else 0.0,
        "no_target_silent_rate": 1.0 if latency_bucket == "no_target_silent" else 0.0,
    })
    speaking_steps = 0
    timing_values: List[int] = []
    strict_applicable = 0
    loose_applicable = 0
    recall_nomatch_count = 0
    for t in ep_steps:
        spoke = _delegate_utterance_present(t.delegate_utterance)
        if spoke:
            speaking_steps += 1
            dj = t.delegate_judge or {}
            # Derive timing_delta consistent with step-level policy (delegate-only):
            # - if matched_utt_id == anchor -> timing_delta := 0
            # - else use delegate_judge.timing_delta when provided (do not use env timing)
            td = dj.get("timing_delta") if isinstance(dj, dict) else None
            matched = dj.get("matched_utt_id") if isinstance(dj, dict) else None
            if matched is None:
                recall_nomatch_count += 1
            else:
                if t.anchor_utt_id is not None:
                    strict_applicable += 1
                if matched == t.anchor_utt_id:
                    td = 0
            if isinstance(td, int):
                timing_values.append(td)
                if td >= 0:
                    loose_applicable += 1
    if timing_values:
        mean_td = sum(timing_values) / len(timing_values)
        med_sorted = sorted(timing_values)
        mid = len(med_sorted) // 2
        if len(med_sorted) % 2 == 1:
            median_td = float(med_sorted[mid])
        else:
            median_td = (med_sorted[mid - 1] + med_sorted[mid]) / 2.0
        pct_zero = sum(1 for v in timing_values if v == 0) / len(timing_values)
    else:
        mean_td = None; median_td = None; pct_zero = None
    base.update({
        "speaking_steps": speaking_steps,
        "mean_delegate_timing_delta_speaks": mean_td,
        "median_delegate_timing_delta_speaks": median_td,
        "pct_delegate_timing_delta_eq_0": pct_zero,
        "recall_strict_coverage": (strict_applicable / speaking_steps) if speaking_steps else None,
        "recall_loose_coverage": (loose_applicable / speaking_steps) if speaking_steps else None,
        "recall_nomatch_count": recall_nomatch_count,
    })
    return base


def main():
    ap = argparse.ArgumentParser(description="CAPA eval runner")
    ap.add_argument("--meeting", required=True)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--export", choices=["csv", "json", "both"], default="both")
    ap.add_argument("--mode", choices=["step", "idea", "hybrid"], default="step", help="Eval mode: legacy step metrics, idea-unit metrics, or both.")
    ap.add_argument("--ctx-strict", type=float, default=0.80, help="Strict context_appropriateness threshold for idea recall.")
    ap.add_argument("--rel-strict", type=float, default=0.70, help="Strict relevance threshold for idea recall.")
    ap.add_argument("--ctx-loose", type=float, default=0.70, help="Loose context_appropriateness threshold for idea recall.")
    ap.add_argument("--rel-loose", type=float, default=0.70, help="Loose relevance threshold for idea recall.")
    ap.add_argument("--episodes-dir", help="Optional explicit path to capa_episodes (used to load idea_units metadata).", default=None)
    ap.add_argument("--participants", default=None, help="Optional comma-separated participant filter for partial sanity evaluation.")
    ap.add_argument(
        "--strict-missing-artifacts",
        action="store_true",
        help="Exit nonzero when no artifact files are found for the requested meeting or participant filter.",
    )
    args = ap.parse_args()

    meeting = args.meeting
    results_dir = Path(args.results_dir)
    participant_filter = _parse_participant_filter(args.participants)
    participant_filter_lookup = {participant.casefold() for participant in participant_filter}

    artifact_paths = discover_artifact_files(results_dir, meeting)
    if not artifact_paths:
        print(f"No artifact files found for meeting {meeting} under {results_dir}/artifacts", file=sys.stderr)
        raise SystemExit(1 if args.strict_missing_artifacts else 0)

    artifact_payloads = []
    for artifact_path in artifact_paths:
        data = load_artifact_file(artifact_path)
        participant = data.get("participant_id") or "unknown"
        if participant_filter_lookup and participant.casefold() not in participant_filter_lookup:
            continue
        artifact_payloads.append((artifact_path, data, participant))

    if participant_filter_lookup and not artifact_payloads:
        requested = ", ".join(participant_filter)
        print(
            f"No artifact files matched participants [{requested}] for meeting {meeting} under {results_dir}/artifacts",
            file=sys.stderr,
        )
        raise SystemExit(1 if args.strict_missing_artifacts else 0)

    participants_evaluated = sorted({participant for _, _, participant in artifact_payloads})
    out_dir = (
        results_dir / "evals_partial" / meeting / _participant_scope_slug(participants_evaluated)
        if participant_filter_lookup
        else results_dir / "evals" / meeting
    )
    _ensure_dir(out_dir)

    # Accumulators for CSVs
    step_rows: List[Dict[str, Any]] = []
    episode_rows: List[Dict[str, Any]] = []
    participant_rows: List[Dict[str, Any]] = []
    idea_unit_rows: List[Dict[str, Any]] = []

    episodes_dir_override = Path(args.episodes_dir) if args.episodes_dir else None
    epi_meta_all = _load_episode_meta(results_dir, meeting, episodes_dir_override)
    for _, data, participant in artifact_payloads:
        steps = list(iter_steps_from_artifacts(data))
        episodes = group_steps_by_episode(steps)
        for ep_id, ep_steps in episodes.items():
            # Compute latency targets from the episode anchor (delta==0) window
            anchor_step = None
            for t in ep_steps:
                if t.delta == 0:
                    anchor_step = t
                    break
            if anchor_step is None:
                anchor_step = ep_steps[0]
            # target_exists/j from anchor window: search indices 0..k-1 (include anchor)
            gt_speakers = (anchor_step.gt_window_speakers or [])
            target_idx = None
            for idx in range(0, len(gt_speakers)):
                if gt_speakers[idx] == participant:
                    target_idx = idx
                    break
            target_exists = target_idx is not None
            # observed_speak earliest i across episode
            observed_idx = None
            for t in ep_steps:
                if _delegate_utterance_present(t.delegate_utterance):
                    observed_idx = t.delta
                    break
            # Derive latency bucket and signed latency
            if target_exists:
                if observed_idx is None:
                    latency_bucket = "missed_target"
                    latency_signed = None
                else:
                    if observed_idx == target_idx:
                        latency_bucket = "on_time"
                        latency_signed = 0
                    elif observed_idx < target_idx:
                        latency_bucket = "early"
                        latency_signed = observed_idx - target_idx
                    else:
                        latency_bucket = "late"
                        latency_signed = observed_idx - target_idx
            else:
                if observed_idx is None:
                    latency_bucket = "no_target_silent"
                    latency_signed = None
                else:
                    latency_bucket = "no_target_spoke"
                    latency_signed = None

            # Build episode-level row early (ensures every episode contributes one row).
            try:
                tcid__ = ep_steps[0].test_case_id if ep_steps else None
            except Exception:
                tcid__ = None
            if tcid__ is not None:
                ep_row = _compute_episode_row(ep_steps, participant, latency_bucket, latency_signed)
                # De-duplicate in case of legacy block path
                if not any((r.get("test_case_id") == tcid__ and r.get("participant_id") == participant) for r in episode_rows):
                    episode_rows.append(ep_row)

            # Note: env recall strict/loose are computed per step below. Do not precompute
            # anchor speaker from the episode anchor; use each step's own t+1 speaker.

            # Build step rows for this episode

            for s in ep_steps:
                step_dict = {
                    "meeting_id": s.meeting_id,
                    "participant_id": s.participant_id,
                    "test_case_id": s.test_case_id,
                    "delta": s.delta,
                    "anchor_utt_id": s.anchor_utt_id,
                    "gt_window_ids": s.gt_window_ids,
                    "gt_is_banal": s.gt_is_banal,
                    "matched_unit_id": s.matched_unit_id,
                    "unit_timing_delta": s.unit_timing_delta,
                    "delegate_utterance": s.delegate_utterance,
                    # Include controller_output so idea-action uses real action_name
                    "controller_output": s.controller_output or {},
                }
                # Preserve full delegate judge payload for idea-unit recall (strict/loose)
                step_dict["delegate_judge"] = s.delegate_judge or {}
                step_dict["delegate_judge_output"] = s.delegate_judge or {}
                # Explicit flags for expected/observed speak (strict per-step: use THIS step's gt anchor)
                step_gt_anchor = (s.gt_window_speakers[0] if s.gt_window_speakers else None)
                step_dict["gt_expected_speak"] = (1 if step_gt_anchor == participant else 0) if step_gt_anchor is not None else None
                step_dict["observed_speak"] = 1 if _delegate_utterance_present(s.delegate_utterance) else 0

                # Action congruence (GT strict) — confusion counts
                gt_counts = gt_action_following(s.__dict__, participant)
                counts = {
                    "gt_tp": gt_counts["tp"],
                    "gt_fp": gt_counts["fp"],
                    "gt_tn": gt_counts["tn"],
                    "gt_fn": gt_counts["fn"],
                }
                step_dict.update(counts)
                # Per-episode accumulators removed; counts are persisted per-step only
                # Cue congruence — confusion counts plus expected value
                cue_counts = cue_action_congruence(s.__dict__, participant)
                step_dict["cue_expected_speak"] = cue_expected_speak(s.__dict__, participant)
                step_dict.update({
                    "cue_tp": cue_counts["tp"],
                    "cue_fp": cue_counts["fp"],
                    "cue_tn": cue_counts["tn"],
                    "cue_fn": cue_counts["fn"],
                })
                # Per-episode accumulators removed; counts are persisted per-step only
                # Hit flags
                dgt_strict = delegate_gt_hit_strict(s.__dict__)
                dgt_loose = delegate_gt_hit_loose(s.__dict__)
                dut_strict = delegate_unit_hit_strict(s.__dict__)
                dut_loose = delegate_unit_hit_loose(s.__dict__)
                drs = delegate_hit_strict(s.__dict__)
                drl = delegate_hit_loose(s.__dict__)
                step_dict.update({
                    "delegate_gt_hit_strict": dgt_strict,
                    "delegate_gt_hit_loose": dgt_loose,
                    "delegate_unit_hit_strict": dut_strict,
                    "delegate_unit_hit_loose": dut_loose,
                    "delegate_hit_strict": drs,
                    "delegate_hit_loose": drl,
                })
                # Delegate judge presence and diagnostics
                dj = s.delegate_judge or {}
                step_dict["delegate_judge_present"] = 1 if dj else 0
                step_dict["judge_contract_violation"] = 1 if dj.get("judge_contract_violation") else 0
                step_dict["judge_contract_failure_reasons"] = ";".join(dj.get("judge_contract_failure_reasons") or []) if dj else None
                delegate_judge_failure = getattr(s, "delegate_judge_failure", None) or {}
                step_dict["delegate_judge_failure"] = 1 if delegate_judge_failure else 0
                step_dict["delegate_judge_failure_reasons"] = (
                    ";".join(key for key, value in dict(delegate_judge_failure or {}).items() if value)
                    if delegate_judge_failure
                    else None
                )
                step_dict["delegate_matched_utt_id"] = dj.get("matched_utt_id") if dj else None
                step_dict["match_window_status"] = dj.get("match_window_status") if dj else None
                step_dict["matched_response_utt_id"] = dj.get("matched_response_utt_id") if dj else None
                step_dict["matched_response_delta"] = dj.get("matched_response_delta") if dj else None
                step_dict["first_response_delta"] = dj.get("first_response_delta") if dj else None
                # Delegate timing delta at step (speaks only; NA otherwise)
                step_observed = step_dict.get("observed_speak")
                if step_observed == 1:
                    dj = s.delegate_judge or {}
                    # If matched_utt_id equals anchor, timing_delta must be 0 (by definition: t+1 -> 0)
                    if (dj.get("matched_utt_id") is not None) and (s.anchor_utt_id is not None) and (dj.get("matched_utt_id") == s.anchor_utt_id):
                        step_dict["delegate_timing_delta"] = 0
                    else:
                        step_dict["delegate_timing_delta"] = dj.get("timing_delta")
                else:
                    step_dict["delegate_timing_delta"] = None
                # NA reasons for recall (explicit coverage tracking)
                strict_reason = "none" if drs is not None else recall_na_reason(s.__dict__, variant="strict")
                loose_reason = "none" if drl is not None else recall_na_reason(s.__dict__, variant="loose")
                step_dict["recall_strict_na_reason"] = strict_reason
                step_dict["recall_loose_na_reason"] = loose_reason
                # Stewardship
                inv = inventory_compliance(s.__dict__)
                step_dict.update({
                    "treatment_violation": 1 if inv["violations"] > 0 else 0,
                    "violation_reason": ";".join(inv["reasons"]) if inv["reasons"] else None,
                    "spoke_against_global_suggestion": 1 if ("spoke_against_global_suggestion" in (inv["reasons"] or [])) else 0,
                })
                # Timing, quality fields
                env = s.env_judge or {}
                step_dict.update({
                    "env_timing_delta": env.get("timing_delta"),
                    "env_category": env.get("category"),
                    # Decision appropriateness applies to both speak and silent decisions
                    "delegate_decision_appropriateness": (s.delegate_judge or {}).get("decision_appropriateness"),
                    # Content quality metrics apply to speaking steps only
                    "delegate_context_appropriateness": (s.delegate_judge or {}).get("context_appropriateness") if step_observed == 1 else None,
                    "delegate_relevance": (s.delegate_judge or {}).get("relevance") if step_observed == 1 else None,
                    "delegate_grounding": (s.delegate_judge or {}).get("grounding") if step_observed == 1 else None,
                    "redundancy": (s.delegate_judge or {}).get("redundancy") if step_observed == 1 else None,
                    "delegate_redundancy_kind": (s.delegate_judge or {}).get("redundancy_kind") if step_observed == 1 else None,
                    "delegate_redundancy_reason": (s.delegate_judge or {}).get("redundancy_reason") if step_observed == 1 else None,
                    "hallucinated_claim": (s.delegate_judge or {}).get("hallucinated_claim") if step_observed == 1 else None,
                    "off_topic": (s.delegate_judge or {}).get("off_topic") if step_observed == 1 else None,
                })
                # Coverage treatment (from controller payload stored as judge_input)
                cov_rt = None
                try:
                    dji = s.delegate_judge_input or {}
                    ev = ((dji.get("evaluation_context") or {}).get("coverage_assessment") or {}).get("coverage_evaluator")
                    if isinstance(ev, dict):
                        cov_rt = ev.get("recommended_treatment")
                except Exception:
                    cov_rt = None
                step_dict["coverage_recommended_treatment"] = cov_rt
                # Env recall flags
                pred_spk = ((s.env_pred or {}).get("speaker_id") if isinstance(s.env_pred, dict) else None)
                env_cat = env.get("category")
                # Strict recall as a standard binary flag: 1 for Aligned-Immediate,
                # 0 for any other category when env_judge is present; None if category missing.
                if env_cat is None:
                    strict_hit = None
                else:
                    strict_hit = 1 if env_cat == "Aligned-Immediate" else 0
                step_dict["env_recall_strict"] = strict_hit
                step_dict["env_recall_loose"] = 1 if env_cat in ("Aligned-Immediate", "Aligned-Late") else (0 if env_cat else None)
                # Latency fields (replicated on each step row in the episode)
                step_dict.update({
                    "target_exists": target_exists,
                    "target_idx": target_idx,
                    "observed_speak_idx": observed_idx,
                    "latency_signed": latency_signed,
                    "latency_bucket": latency_bucket,
                })
                step_rows.append(step_dict)

        # Episode-level metrics are computed once via _compute_episode_row above

    # end for ep_id
    # end for p

    # Participant aggregates (micro via steps)
    participants = sorted(list({r["participant_id"] for r in step_rows}))
    _assert_unique_step_rows(step_rows)
    relevant_episode_keys = {_step_episode_key(r) for r in step_rows if _step_episode_key(r)}
    relevant_episode_meta = {
        key: meta
        for key, meta in (epi_meta_all or {}).items()
        if key in relevant_episode_keys
    }

    # IdeaUnit metrics (only if mode != step)
    if args.mode in ("idea", "hybrid"):
        # Annotate idea-anchor flags per step (flex: anchor anywhere in window)
        for r in step_rows:
            tcid = r.get("test_case_id")
            meta = relevant_episode_meta.get(_step_episode_key(r)) if relevant_episode_meta else None
            idea_units = (meta or {}).get("idea_units") or []
            anchor_ids = {iu.get("anchor_utt_id") or (iu.get("response_utterance_ids") or [None])[0] for iu in idea_units}
            gw_ids = r.get("gt_window_ids") or []
            # Flag anchor if ANY turn in the window is an idea anchor
            r["is_idea_anchor"] = 1 if (gw_ids and any(uid in anchor_ids for uid in gw_ids)) else 0
        # Per-participant idea action confusion
        idea_confusion_by_part: Dict[str, Dict[str, int]] = {pid: {"tp":0,"fp":0,"tn":0,"fn":0} for pid in participants}
        for r in step_rows:
            if r.get("is_idea_anchor") is None:
                continue
            c = gt_idea_action_following(r)
            pid = r.get("participant_id")
            acc = idea_confusion_by_part.get(pid)
            if acc:
                for k,v in c.items():
                    acc[k] += v
        # Per-idea-unit recall/status rows
        idea_status_counts_meeting = {"hit_strict":0,"hit_loose_only":0,"attempted_but_bad":0,"never_attempted":0}
        latencies: List[int] = []
        for _, meta in relevant_episode_meta.items():
            tcid = meta.get("test_case_id")
            represented_user = meta.get("represented_user")
            idea_units = meta.get("idea_units") or []
            if not idea_units:
                continue
            tc_steps = [
                r for r in step_rows
                if r.get("test_case_id") == tcid and r.get("participant_id") == represented_user
            ]
            for iu in idea_units:
                strict_hit, loose_hit, status, earliest_delta = idea_strict_loose_recall_for_unit(
                    tc_steps,
                    iu.get("unit_id"),
                    ctx_strict=args.ctx_strict,
                    rel_strict=args.rel_strict,
                    ctx_loose=args.ctx_loose,
                    rel_loose=args.rel_loose,
                )
                idea_unit_rows.append({
                    "meeting_id": meeting,
                    "participant_id": represented_user,
                    "test_case_id": tcid,
                    "unit_id": iu.get("unit_id"),
                    "strict_hit": strict_hit,
                    "loose_hit": loose_hit,
                    "status": status,
                    "earliest_timing_delta": earliest_delta,
                    "anchor_utt_id": iu.get("anchor_utt_id"),
                    "anchor_idx": iu.get("first_response_idx"),
                    "topic_summary": iu.get("topic_summary"),
                    "cue_type": iu.get("cue_type"),
                })
                idea_status_counts_meeting[status] += 1
                if earliest_delta is not None:
                    latencies.append(earliest_delta)
        # Attach idea metrics to participant_rows
        for row in participant_rows:
            pid = row["participant_id"]
            conf = idea_confusion_by_part.get(pid) or {"tp":0,"fp":0,"tn":0,"fn":0}
            metrics = _confusion_to_metrics(conf["tp"], conf["fp"], conf["tn"], conf["fn"])
            row.update({
                "gt_idea_action_accuracy_micro": metrics["accuracy"],
                "gt_idea_action_precision_micro": metrics["precision"],
                "gt_idea_action_recall_micro": metrics["recall"],
                "gt_idea_action_f1_micro": metrics["f1"],
            })
        # Meeting-level idea aggregates
        meeting_idea_conf = aggregate_confusion(list(idea_confusion_by_part.values()))
        meeting_idea_metrics = _confusion_to_metrics(meeting_idea_conf["tp"], meeting_idea_conf["fp"], meeting_idea_conf["tn"], meeting_idea_conf["fn"])
    else:
        meeting_idea_metrics = {}
        idea_status_counts_meeting = {}
        latencies = []
    for pid in participants:
        part_steps = [r for r in step_rows if r["participant_id"] == pid]
        # Seed row before optional idea metrics
        row = {
            "meeting_id": meeting,
            "participant_id": pid,
        }
        if args.mode in ("idea", "hybrid"):
            conf = idea_confusion_by_part.get(pid) or {"tp":0,"fp":0,"tn":0,"fn":0}
            midea = _confusion_to_metrics(conf["tp"], conf["fp"], conf["tn"], conf["fn"])
            row.update({
                "gt_idea_action_accuracy_micro": midea["accuracy"],
                "gt_idea_action_precision_micro": midea["precision"],
                "gt_idea_action_recall_micro": midea["recall"],
                "gt_idea_action_f1_micro": midea["f1"],
            })
        # Micro GT
        gt_tp = sum(int(r.get("gt_tp", 0) or 0) for r in part_steps)
        gt_fp = sum(int(r.get("gt_fp", 0) or 0) for r in part_steps)
        gt_tn = sum(int(r.get("gt_tn", 0) or 0) for r in part_steps)
        gt_fn = sum(int(r.get("gt_fn", 0) or 0) for r in part_steps)
        def _acc(tp,fp,tn,fn):
            tot = tp+fp+tn+fn
            return ((tp+tn)/tot) if tot else None
        def _f1(tp,fp,tn,fn):
            denom = 2*tp + fp + fn
            return (2*tp/denom) if denom else None
        row.update({
            "gt_action_accuracy_micro": _acc(gt_tp,gt_fp,gt_tn,gt_fn),
            "gt_action_f1_micro": _f1(gt_tp,gt_fp,gt_tn,gt_fn),
        })
        cue_tp = sum(int(r.get("cue_tp", 0) or 0) for r in part_steps)
        cue_fp = sum(int(r.get("cue_fp", 0) or 0) for r in part_steps)
        cue_tn = sum(int(r.get("cue_tn", 0) or 0) for r in part_steps)
        cue_fn = sum(int(r.get("cue_fn", 0) or 0) for r in part_steps)
        row.update({
            "cue_action_accuracy_micro": _acc(cue_tp,cue_fp,cue_tn,cue_fn),
            "cue_action_f1_micro": _f1(cue_tp,cue_fp,cue_tn,cue_fn),
        })
        participant_rows.append(row)

    # Episode rows are constructed per-episode above; no step-derived fallback is needed

    # Meeting-level outputs (computed once at end)
    meeting_metrics = aggregate_meeting(step_rows, episode_rows, participant_rows, meeting)
    if not participant_filter_lookup:
        k_diagnostics_summary = _load_meeting_k_diagnostics_summary(
            results_dir,
            meeting,
            relevant_episode_meta,
            episodes_dir_override,
        )
        if k_diagnostics_summary:
            planned_episode_count = k_diagnostics_summary.get("kept_episode_count")
            planned_step_count = k_diagnostics_summary.get("total_evaluated_steps")
            meeting_metrics.update({
                "capa_k": k_diagnostics_summary.get("k"),
                "inventory_reference_k": k_diagnostics_summary.get("inventory_reference_k"),
                "frozen_anchor_count": k_diagnostics_summary.get("frozen_anchor_count"),
                "kept_episode_count": k_diagnostics_summary.get("kept_episode_count"),
                "dropped_anchor_count": k_diagnostics_summary.get("dropped_anchor_count"),
                "preferred_cue_retained_count": k_diagnostics_summary.get("preferred_cue_retained_count"),
                "preferred_cue_retained_rate": k_diagnostics_summary.get("preferred_cue_retained_rate"),
                "selector_reanchor_count": k_diagnostics_summary.get("selector_reanchor_count"),
                "selector_reanchor_rate": k_diagnostics_summary.get("selector_reanchor_rate"),
                "deterministic_reanchor_count": k_diagnostics_summary.get("deterministic_reanchor_count"),
                "deterministic_reanchor_rate": k_diagnostics_summary.get("deterministic_reanchor_rate"),
                "self_initiation_fallback_count": k_diagnostics_summary.get("self_initiation_fallback_count"),
                "self_initiation_fallback_rate": k_diagnostics_summary.get("self_initiation_fallback_rate"),
                "cap_dropped_count": k_diagnostics_summary.get("cap_dropped_count"),
                "cap_dropped_rate": k_diagnostics_summary.get("cap_dropped_rate"),
                "anchor_drop_rate": k_diagnostics_summary.get("anchor_drop_rate"),
                "mean_preferred_cue_gap": k_diagnostics_summary.get("mean_preferred_cue_gap"),
                "mean_effective_cue_gap": k_diagnostics_summary.get("mean_effective_cue_gap"),
                "total_evaluated_steps": k_diagnostics_summary.get("total_evaluated_steps"),
                "cue_mode_counts": k_diagnostics_summary.get("cue_mode_counts"),
                "planned_episode_count": planned_episode_count,
                "planned_step_count": planned_step_count,
                "emitted_episode_count": len(episode_rows),
                "emitted_step_count": len(step_rows),
                "terminal_shortfall_step_count": (
                    int(planned_step_count) - len(step_rows)
                    if planned_step_count is not None
                    else None
                ),
            })
    if args.mode in ("idea", "hybrid"):
        meeting_metrics.update({
            "gt_idea_action_accuracy_micro": meeting_idea_metrics.get("accuracy"),
            "gt_idea_action_precision_micro": meeting_idea_metrics.get("precision"),
            "gt_idea_action_recall_micro": meeting_idea_metrics.get("recall"),
            "gt_idea_action_f1_micro": meeting_idea_metrics.get("f1"),
            "idea_strict_recall_rate": sum(1 for r in idea_unit_rows if r.get("strict_hit") == 1) / len(idea_unit_rows) if idea_unit_rows else None,
            "idea_loose_recall_rate": sum(1 for r in idea_unit_rows if r.get("loose_hit") == 1) / len(idea_unit_rows) if idea_unit_rows else None,
            "idea_latency_mean": (sum(latencies)/len(latencies)) if latencies else None,
            "idea_latency_median": (sorted(latencies)[len(latencies)//2] if latencies else None),
        })
        meeting_metrics.update({f"idea_status_{k}": v for k,v in idea_status_counts_meeting.items()})

    # ------------------------------------------------------------------
    # Prune metrics in idea-only mode to the curated set we report
    # ------------------------------------------------------------------
    def _filter_dict(d: Dict[str, Any], allowed: List[str]) -> Dict[str, Any]:
        return {k: d.get(k) for k in allowed if k in d}

    if args.mode == "idea":
        part_allowed = {
            "meeting_id",
            "participant_id",
            "gt_idea_action_accuracy_micro",
            "gt_idea_action_precision_micro",
            "gt_idea_action_recall_micro",
            "gt_idea_action_f1_micro",
        }
        participant_rows = [_filter_dict(r, list(part_allowed)) for r in participant_rows]

        meeting_allowed = [
            "meeting_id",
            "gt_idea_action_accuracy_micro",
            "gt_idea_action_precision_micro",
            "gt_idea_action_recall_micro",
            "gt_idea_action_f1_micro",
            "idea_strict_recall_rate",
            "idea_loose_recall_rate",
            "idea_latency_mean",
            "idea_latency_median",
            "matched_unit_rate",
            "idea_match_synchronized_rate",
            "idea_match_proactive_rate",
            "idea_status_hit_strict",
            "idea_status_hit_loose_only",
            "idea_status_attempted_but_bad",
            "idea_status_never_attempted",
            "delegate_decision_appropriateness_mean",
            "spoken_step_count",
            "context_scored_step_count",
            "relevance_scored_step_count",
            "grounding_scored_step_count",
            "judge_contract_violation_count",
            "delegate_judge_failure_count",
            "delegate_context_appropriateness_mean",
            "delegate_relevance_mean",
            "delegate_grounding_mean",
            "hallucination_rate_speaks",
            "redundancy_rate_speaks",
            "off_topic_rate_speaks",
            "delegate_judge_missing_rate_speaks",
            "capa_k",
            "inventory_reference_k",
            "frozen_anchor_count",
            "kept_episode_count",
            "dropped_anchor_count",
            "preferred_cue_retained_count",
            "preferred_cue_retained_rate",
            "selector_reanchor_count",
            "selector_reanchor_rate",
            "deterministic_reanchor_count",
            "deterministic_reanchor_rate",
            "self_initiation_fallback_count",
            "self_initiation_fallback_rate",
            "cap_dropped_count",
            "cap_dropped_rate",
            "anchor_drop_rate",
            "mean_preferred_cue_gap",
            "mean_effective_cue_gap",
            "total_evaluated_steps",
            "cue_mode_counts",
            "planned_episode_count",
            "planned_step_count",
            "emitted_episode_count",
            "emitted_step_count",
            "terminal_shortfall_step_count",
        ]
        meeting_metrics = _filter_dict(meeting_metrics, meeting_allowed)
        # note: step and episode outputs remain full; idea_unit rows remain full.

    if participant_filter_lookup:
        meeting_metrics.update({
            "evaluation_scope": "participant_subset",
            "participants_evaluated": participants_evaluated,
            "participant_filter_applied": True,
            "is_partial_eval": True,
        })

    meeting_metrics["delegate_judge_regime"] = DELEGATE_JUDGE_REGIME

    # Episode/participant/step/idea CSVs (write after pruning)
    if args.export in ("csv", "both"):
        # Enrich episode rows with provenance from capa_episodes when available
        epi_meta = _load_episode_meta(results_dir, meeting, episodes_dir_override)
        for r in episode_rows:
            tcid = r.get("test_case_id")
            if not tcid:
                continue
            meta = epi_meta.get(_episode_meta_key(r.get("participant_id"), tcid))
            if meta:
                for k,v in meta.items():
                    if k not in r or r.get(k) is None:
                        r[k] = v
        # Episode metrics CSV
        ep_csv = out_dir / "episode_metrics.csv"
        fnames: List[str] = []
        for r in episode_rows:
            for k in r.keys():
                if k not in fnames:
                    fnames.append(k)
        with ep_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fnames)
            w.writeheader()
            w.writerows(episode_rows)

        # Participant metrics CSV
        part_csv = out_dir / "participant_metrics.csv"
        pfields: List[str] = []
        for r in participant_rows:
            for k in r.keys():
                if k not in pfields:
                    pfields.append(k)
        with part_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=pfields)
            w.writeheader()
            w.writerows(participant_rows)

        # Step metrics CSV
        step_csv = out_dir / "step_metrics.csv"
        sfields: List[str] = []
        for r in step_rows:
            for k in r.keys():
                if k not in sfields:
                    sfields.append(k)
        with step_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sfields)
            w.writeheader()
            w.writerows(step_rows)

        # Idea unit metrics CSV
        if args.mode in ("idea", "hybrid") and idea_unit_rows:
            idea_csv = out_dir / "idea_unit_metrics.csv"
            fn_idea: List[str] = []
            for r in idea_unit_rows:
                for k in r.keys():
                    if k not in fn_idea:
                        fn_idea.append(k)
            with idea_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fn_idea)
                w.writeheader()
                w.writerows(idea_unit_rows)

        # Optional JSONL per participant
        ppl_jsonl = out_dir / "participants.jsonl"
        with ppl_jsonl.open("w", encoding="utf-8") as f:
            for r in participant_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # JSON
    mm_json = out_dir / "meeting_metrics.json"
    with mm_json.open("w", encoding="utf-8") as f:
        json.dump(meeting_metrics, f, indent=2, ensure_ascii=False)
    # CSV (single row)
    mm_csv = out_dir / "meeting_metrics.csv"
    with mm_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(meeting_metrics.keys()))
        w.writeheader()
        w.writerow(meeting_metrics)


if __name__ == "__main__":
    main()
