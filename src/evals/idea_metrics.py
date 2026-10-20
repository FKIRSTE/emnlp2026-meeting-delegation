"""
Idea-unit level evaluation metrics.

Implements:
- gt_idea_action_following: confusion counts for action at idea anchors.
- idea_strict_loose_recall_for_unit: strict/loose recall, status buckets.
"""

from typing import Dict, Any, List, Tuple, Optional


def gt_idea_action_following(step: Dict[str, Any]) -> Dict[str, int]:
    """Compute tp/fp/tn/fn for GT idea-anchored action at a step."""
    is_anchor = bool(step.get("is_idea_anchor"))
    if step.get("observed_speak") is not None:
        observed_speak = 1 if step.get("observed_speak") else 0
    elif str(step.get("delegate_utterance") or "").strip():
        observed_speak = 1
    else:
        ctrl = step.get("controller_output") or {}
        act_name = ((ctrl.get("action_details") or {}).get("action_name") or "").strip()
        observed_speak = 0 if (not act_name or act_name == "Remain_Silent") else 1
    expected = 1 if is_anchor else 0
    tp = 1 if expected == 1 and observed_speak == 1 else 0
    fn = 1 if expected == 1 and observed_speak == 0 else 0
    fp = 1 if expected == 0 and observed_speak == 1 else 0
    tn = 1 if expected == 0 and observed_speak == 0 else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def idea_strict_loose_recall_for_unit(
    steps: List[Dict[str, Any]],
    unit_id: str,
    ctx_strict: float = 0.80,
    rel_strict: float = 0.70,
    ctx_loose: float = 0.70,
    rel_loose: float = 0.70,
) -> Tuple[int, int, str, Optional[int]]:
    """
    Returns (strict_hit, loose_hit, status, earliest_delta_if_hit).
    status in {'hit_strict', 'hit_loose_only', 'attempted_but_bad', 'never_attempted'}.

    Soft thresholds are retained in the signature for compatibility but are not used
    for official idea recall. Official recall depends on verifier-backed unit matching,
    timing, redundancy, and hallucination flags.
    """
    strict_hit = 0
    loose_hit = 0
    earliest_delta = None
    any_attempt = False
    best_loose_delta = None
    for step in steps:
        if step.get("observed_speak") == 1 or str(step.get("delegate_utterance") or "").strip():
            any_attempt = True
        dj = step.get("delegate_judge") or {}
        if not dj:
            continue
        if dj.get("matched_unit_id") != unit_id:
            continue
        if dj.get("match_window_status") not in (None, "accepted_current_window"):
            continue
        unit_timing_delta = dj.get("unit_timing_delta")
        first_response_delta = dj.get("first_response_delta")
        matched_response_delta = dj.get("matched_response_delta")
        redundancy = bool(dj.get("redundancy"))
        hallucinated = bool(dj.get("hallucinated_claim"))
        strict_delta = first_response_delta if isinstance(first_response_delta, int) else unit_timing_delta
        loose_delta = matched_response_delta if isinstance(matched_response_delta, int) else unit_timing_delta
        if strict_delta == 0 and not redundancy and not hallucinated:
            strict_hit = 1
            earliest_delta = 0
        if isinstance(loose_delta, int) and loose_delta >= 0 and not redundancy and not hallucinated:
            loose_hit = 1
            if best_loose_delta is None or loose_delta < best_loose_delta:
                best_loose_delta = loose_delta
                earliest_delta = loose_delta
    status = "never_attempted"
    if strict_hit:
        status = "hit_strict"
    elif loose_hit:
        status = "hit_loose_only"
    elif any_attempt:
        status = "attempted_but_bad"
    return strict_hit, loose_hit, status, earliest_delta


__all__ = ["gt_idea_action_following", "idea_strict_loose_recall_for_unit"]
