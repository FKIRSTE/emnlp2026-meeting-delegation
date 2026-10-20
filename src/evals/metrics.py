from __future__ import annotations

"""
Metric functions for CAPA evals.

All functions are pure and operate on normalized step records (dict-like or dataclass).
See eval_plan.md Sections 5, 6, 20 for exact semantics.
"""

from typing import Any, Dict, List, Optional


def _observed_speak(controller_output: Optional[dict], ack_counts_as_speak: bool = True) -> Optional[int]:
    if not controller_output:
        return None
    action = ((controller_output.get("action_details") or {}).get("action_name") or "").strip()
    if not action:
        return None
    if action == "Remain_Silent":
        return 0
    return 1


def _gt_expected_speak(gt_window_speakers: Optional[List[str]], represented_user: Optional[str]) -> Optional[int]:
    if not gt_window_speakers or represented_user is None:
        return None
    anchor_speaker = gt_window_speakers[0] if gt_window_speakers else None
    if anchor_speaker is None:
        return None
    return 1 if anchor_speaker == represented_user else 0


def _delegate_judge_payload(step: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    payload = step.get("delegate_judge_output") or step.get("delegate_judge") or {}
    return payload or None


def _applicable_speaking_step(step: Dict[str, Any]) -> bool:
    ctrl = step.get("controller_output") or {}
    action = ((ctrl.get("action_details") or {}).get("action_name") or "").strip()
    return bool(action and action != "Remain_Silent")


def _not_creditable(dj: Dict[str, Any]) -> bool:
    return bool(dj.get("redundancy")) or bool(dj.get("hallucinated_claim"))


def gt_action_following(step: Dict[str, Any], represented_user: str) -> Dict[str, int]:
    """Strict per-delta action congruence vs GT anchor speaker."""
    obs = _observed_speak(step.get("controller_output"))
    exp = _gt_expected_speak(step.get("gt_window_speakers"), represented_user)
    out = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    if obs is None or exp is None:
        return out
    if exp == 1 and obs == 1:
        out["tp"] += 1
    elif exp == 1 and obs == 0:
        out["fn"] += 1
    elif exp == 0 and obs == 1:
        out["fp"] += 1
    elif exp == 0 and obs == 0:
        out["tn"] += 1
    return out


def cue_expected_speak(step: Dict[str, Any], represented_user: str) -> Optional[int]:
    """Derive cue-based expectation only when an authoritative cue/floor source exists."""
    payload = step.get("delegate_judge_input") or {}
    ctx = payload.get("evaluation_context") or {}
    explicit = ctx.get("explicit_cue")
    target = ctx.get("explicit_cue_target") or ctx.get("explicit_cue_target_id")
    floor_status = ctx.get("floor_status")
    ctrl = step.get("controller_output") or {}
    speaker_rec = ctrl.get("speaker_recommendation")

    if explicit is True:
        if target is None:
            return None
        return 1 if str(target).strip().casefold() == str(represented_user).strip().casefold() else 0
    if floor_status is None:
        return None
    if speaker_rec is None:
        return None
    return 1 if str(speaker_rec).strip().casefold() == str(represented_user).strip().casefold() else 0


def cue_action_congruence(step: Dict[str, Any], represented_user: str) -> Dict[str, int]:
    obs = _observed_speak(step.get("controller_output"))
    exp = cue_expected_speak(step, represented_user)
    out = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    if obs is None or exp is None:
        return out
    if exp == 1 and obs == 1:
        out["tp"] += 1
    elif exp == 1 and obs == 0:
        out["fn"] += 1
    elif exp == 0 and obs == 1:
        out["fp"] += 1
    elif exp == 0 and obs == 0:
        out["tn"] += 1
    return out


def delegate_gt_hit_strict(step: Dict[str, Any]) -> Optional[int]:
    if not _applicable_speaking_step(step):
        return None
    dj = _delegate_judge_payload(step)
    if not dj:
        return None
    status = dj.get("match_window_status")
    if status is not None:
        if status != "accepted_current_window":
            return 0
        if dj.get("matched_response_utt_id") is None:
            return 0
        if _not_creditable(dj):
            return 0
        strict_delta = dj.get("first_response_delta")
        return 1 if strict_delta == 0 else 0
    matched = dj.get("matched_utt_id")
    if matched is None:
        return 0
    anchor_id = step.get("anchor_utt_id")
    if anchor_id is None:
        return None
    if _not_creditable(dj):
        return 0
    return 1 if matched == anchor_id else 0


def delegate_gt_hit_loose(step: Dict[str, Any]) -> Optional[int]:
    if not _applicable_speaking_step(step):
        return None
    dj = _delegate_judge_payload(step)
    if not dj:
        return None
    status = dj.get("match_window_status")
    if status is not None:
        if status != "accepted_current_window":
            return 0
        if dj.get("matched_response_utt_id") is None:
            return 0
        if _not_creditable(dj):
            return 0
        loose_delta = dj.get("matched_response_delta")
        return 1 if isinstance(loose_delta, int) and loose_delta >= 0 else 0
    if dj.get("matched_utt_id") is None:
        return 0
    timing_delta = dj.get("timing_delta")
    if timing_delta is None:
        return 0
    if _not_creditable(dj):
        return 0
    return 1 if isinstance(timing_delta, int) and timing_delta >= 0 else 0


def delegate_unit_hit_strict(step: Dict[str, Any]) -> Optional[int]:
    if not _applicable_speaking_step(step):
        return None
    dj = _delegate_judge_payload(step)
    if not dj:
        return None
    if dj.get("matched_unit_id") is None:
        return 0
    if dj.get("match_window_status") not in (None, "accepted_current_window"):
        return 0
    if _not_creditable(dj):
        return 0
    strict_delta = dj.get("first_response_delta")
    if strict_delta is None:
        strict_delta = dj.get("unit_timing_delta")
    return 1 if strict_delta == 0 else 0


def delegate_unit_hit_loose(step: Dict[str, Any]) -> Optional[int]:
    if not _applicable_speaking_step(step):
        return None
    dj = _delegate_judge_payload(step)
    if not dj:
        return None
    if dj.get("matched_unit_id") is None:
        return 0
    if dj.get("match_window_status") not in (None, "accepted_current_window"):
        return 0
    loose_delta = dj.get("matched_response_delta")
    if loose_delta is None:
        loose_delta = dj.get("unit_timing_delta")
    if loose_delta is None:
        return 0
    if _not_creditable(dj):
        return 0
    return 1 if isinstance(loose_delta, int) and loose_delta >= 0 else 0


def delegate_hit_strict(step: Dict[str, Any]) -> Optional[int]:
    return delegate_gt_hit_strict(step)


def delegate_hit_loose(step: Dict[str, Any]) -> Optional[int]:
    return delegate_gt_hit_loose(step)


def recall_na_reason(step: Dict[str, Any], variant: str = "strict") -> Optional[str]:
    """Return NA reason enum for delegate recall at a step."""
    ctrl = step.get("controller_output") or {}
    action = ((ctrl.get("action_details") or {}).get("action_name") or "").strip()
    if not action:
        return None
    observed = 0 if action == "Remain_Silent" else 1
    if observed == 0:
        return "silent"
    dj = _delegate_judge_payload(step)
    if not dj:
        return "judge_missing"
    return "none"


def inventory_compliance(step: Dict[str, Any]) -> Dict[str, Any]:
    ctrl = step.get("controller_output") or {}
    digest = ctrl.get("helper_candidates_digest") or []
    active_id = ctrl.get("active_candidate_id")
    global_sugg = ctrl.get("helper_global_suggestion")
    violations = []
    if active_id and isinstance(digest, list):
        for item in digest:
            if (item or {}).get("candidate_id") == active_id:
                if str((item or {}).get("recommended_treatment", "")).lower() == "skip":
                    violations.append("selected_skip_candidate")
                break
    obs = _observed_speak(ctrl)
    if obs == 1 and global_sugg == "Remain_Silent":
        violations.append("spoke_against_global_suggestion")
    return {"violations": len(violations), "reasons": violations}


def latency_bucket(step: Dict[str, Any], represented_user: str) -> Dict[str, Any]:
    """Compute latency bucket and signed latency using eval_plan Section 6.1.3."""
    tgt_exists = step.get("target_exists")
    target_idx = step.get("target_idx")
    observed_idx = step.get("observed_speak_idx")
    bucket = None
    signed = None
    if tgt_exists:
        if observed_idx is None:
            bucket = "missed_target"
        else:
            if observed_idx == target_idx:
                bucket = "on_time"
                signed = 0
            elif observed_idx < target_idx:
                bucket = "early"
                signed = observed_idx - target_idx
            else:
                bucket = "late"
                signed = observed_idx - target_idx
    else:
        if observed_idx is None:
            bucket = "no_target_silent"
        else:
            bucket = "no_target_spoke"
    return {"latency_bucket": bucket, "latency_signed": signed}
