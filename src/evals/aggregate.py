from __future__ import annotations

"""
Aggregation utilities for CAPA evals.

Implements bottom-up rollups per eval_plan.md Section 20.
"""

from typing import Any, Dict, List, Optional, Tuple
import math

from .loader import StepRecord


def _safe_mean(values: List[Optional[float]]) -> Optional[float]:
    xs = [v for v in values if isinstance(v, (int, float))]
    if not xs:
        return None
    return sum(xs) / len(xs)


def _confusion_to_metrics(tp: int, fp: int, tn: int, fn: int) -> Dict[str, Optional[float]]:
    tot = tp + fp + tn + fn
    acc = (tp + tn) / tot if tot else None
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if prec is not None and rec is not None and (prec + rec) else None
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "n": tot}


def aggregate_confusion(confusions: List[Dict[str, int]]) -> Dict[str, int]:
    tp = fp = tn = fn = 0
    for c in confusions:
        tp += int(c.get("tp", 0) or 0)
        fp += int(c.get("fp", 0) or 0)
        tn += int(c.get("tn", 0) or 0)
        fn += int(c.get("fn", 0) or 0)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def compute_episode_metrics(ep_steps: List[StepRecord], represented_user: str) -> Dict[str, Any]:
    # Confusions
    gt_tp = gt_fp = gt_tn = gt_fn = 0
    cue_tp = cue_fp = cue_tn = cue_fn = 0
    # Latency (replicated per step; take from first step that has bucket)
    latency_bucket = None
    latency_signed = None
    for s in ep_steps:
        # step rows will include confusion counts later, but for safety recompute from StepRecord would be heavy.
        # Here we rely on CLI-supplied step_rows; thus populate minimal placeholders.
        # Episode latency
        if latency_bucket is None:
            # CLI sets these on step rows; StepRecord doesn't. We'll recompute from controller/env when StepRecord gets extended.
            pass
    # We'll return latency derived from attached dict in CLI; they will override as needed.
    # Patch deltas
    def _first_last(series: List[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
        first = None
        last = None
        for v in series:
            if v is None:
                continue
            if first is None:
                first = v
            last = v
        return first, last

    # Environment deltas
    env_intent = [ (s.env_judge or {}).get("intent_score") for s in ep_steps ]
    env_topic  = [ (s.env_judge or {}).get("topic_score")  for s in ep_steps ]
    env_floor  = [ (s.env_judge or {}).get("floor_score")  for s in ep_steps ]
    f,l = _first_last(env_intent); delta_intent = (l - f) if f is not None and l is not None else None
    f,l = _first_last(env_topic);  delta_topic  = (l - f) if f is not None and l is not None else None
    f,l = _first_last(env_floor);  delta_floor  = (l - f) if f is not None and l is not None else None

    # Delegate deltas
    del_dec = [ (s.delegate_judge or {}).get("decision_appropriateness") for s in ep_steps ]
    del_rel = [ (s.delegate_judge or {}).get("relevance") for s in ep_steps ]
    del_grd = [ (s.delegate_judge or {}).get("grounding") for s in ep_steps ]
    f,l = _first_last(del_dec); delta_decision = (l - f) if f is not None and l is not None else None
    f,l = _first_last(del_rel); delta_relevance = (l - f) if f is not None and l is not None else None
    f,l = _first_last(del_grd); delta_grounding = (l - f) if f is not None and l is not None else None

    # Return minimal episode container; confusion/latency will be filled by CLI which has step-level counters
    return {
        "meeting_id": ep_steps[0].meeting_id if ep_steps else None,
        "participant_id": ep_steps[0].participant_id if ep_steps else None,
        "test_case_id": ep_steps[0].test_case_id if ep_steps else None,
        "delta_intent": delta_intent,
        "delta_topic": delta_topic,
        "delta_floor": delta_floor,
        "delta_decision_appropriateness": delta_decision,
        "delta_relevance": delta_relevance,
        "delta_grounding": delta_grounding,
    }


def aggregate_meeting(step_rows: List[Dict[str, Any]], episode_rows: List[Dict[str, Any]], participant_rows: List[Dict[str, Any]], meeting_id: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"meeting_id": meeting_id}
    # Micro GT & Cue action
    gt_tp = sum(int(r.get("gt_tp", 0) or 0) for r in step_rows)
    gt_fp = sum(int(r.get("gt_fp", 0) or 0) for r in step_rows)
    gt_tn = sum(int(r.get("gt_tn", 0) or 0) for r in step_rows)
    gt_fn = sum(int(r.get("gt_fn", 0) or 0) for r in step_rows)
    m = _confusion_to_metrics(gt_tp, gt_fp, gt_tn, gt_fn)
    out.update({
        "gt_action_accuracy_micro": m["accuracy"],
        "gt_action_precision_micro": m["precision"],
        "gt_action_recall_micro": m["recall"],
        "gt_action_f1_micro": m["f1"],
    })
    cue_tp = sum(int(r.get("cue_tp", 0) or 0) for r in step_rows)
    cue_fp = sum(int(r.get("cue_fp", 0) or 0) for r in step_rows)
    cue_tn = sum(int(r.get("cue_tn", 0) or 0) for r in step_rows)
    cue_fn = sum(int(r.get("cue_fn", 0) or 0) for r in step_rows)
    cm = _confusion_to_metrics(cue_tp, cue_fp, cue_tn, cue_fn)
    out.update({
        "cue_action_accuracy_micro": cm["accuracy"],
        "cue_action_f1_micro": cm["f1"],
    })
    # Macro as mean over participants (F1 and accuracy)
    part_accs = [r.get("gt_action_accuracy_micro") for r in participant_rows if r.get("gt_action_accuracy_micro") is not None]
    part_f1s  = [r.get("gt_action_f1_micro") for r in participant_rows if r.get("gt_action_f1_micro") is not None]
    out.update({
        "gt_action_accuracy_macro": _safe_mean(part_accs),
        "gt_action_f1_macro": _safe_mean(part_f1s),
    })
    cue_accs = [r.get("cue_action_accuracy_micro") for r in participant_rows if r.get("cue_action_accuracy_micro") is not None]
    cue_f1s  = [r.get("cue_action_f1_micro") for r in participant_rows if r.get("cue_action_f1_micro") is not None]
    out.update({
        "cue_action_accuracy_macro": _safe_mean(cue_accs),
        "cue_action_f1_macro": _safe_mean(cue_f1s),
    })
    # Latency buckets (proportions over episodes)
    buckets = {"on_time":0, "early":0, "late":0, "missed_target":0, "no_target_spoke":0, "no_target_silent":0}
    signed_values: List[float] = []
    for ep in episode_rows:
        b = ep.get("latency_bucket")
        if b in buckets:
            buckets[b] += 1
        s = ep.get("latency_signed")
        if isinstance(s, (int, float)):
            signed_values.append(float(s))
    n_ep = len(episode_rows) or 1
    out.update({
        "latency_bucket_on_time": buckets["on_time"]/n_ep,
        "latency_bucket_early": buckets["early"]/n_ep,
        "latency_bucket_late": buckets["late"]/n_ep,
        "missed_target_rate": buckets["missed_target"]/n_ep,
        "no_target_spoke_rate": buckets["no_target_spoke"]/n_ep,
        "no_target_silent_rate": buckets["no_target_silent"]/n_ep,
        "latency_mean_signed": (sum(signed_values)/len(signed_values) if signed_values else None),
        "latency_median_signed": (sorted(signed_values)[len(signed_values)//2] if signed_values else None),
        "n_signed_latency": len(signed_values),
    })
    # Delegate timing delta (macro over episodes) per eval_plan_2.md
    ep_means = [ep.get("mean_delegate_timing_delta_speaks") for ep in episode_rows if isinstance(ep.get("mean_delegate_timing_delta_speaks"), (int, float))]
    ep_medians = [ep.get("median_delegate_timing_delta_speaks") for ep in episode_rows if isinstance(ep.get("median_delegate_timing_delta_speaks"), (int, float))]
    ep_pct_zero = [ep.get("pct_delegate_timing_delta_eq_0") for ep in episode_rows if isinstance(ep.get("pct_delegate_timing_delta_eq_0"), (int, float))]
    def _median(vals: List[float]) -> Optional[float]:
        if not vals:
            return None
        s = sorted(vals)
        m = len(s)//2
        if len(s) % 2 == 1:
            return float(s[m])
        return (s[m-1] + s[m]) / 2.0
    out.update({
        "mean_delegate_timing_delta_speaks_macro": (sum(ep_means)/len(ep_means) if ep_means else None),
        "median_delegate_timing_delta_speaks_macro": _median(ep_medians),
        "pct_delegate_timing_delta_eq_0_macro": (sum(ep_pct_zero)/len(ep_pct_zero) if ep_pct_zero else None),
    })
    # Speaking steps total and speaking rate
    speaking_steps_total = sum(int(ep.get("speaking_steps") or 0) for ep in episode_rows)
    out["speaking_steps_total"] = speaking_steps_total
    denom_steps = len(step_rows) or 1
    out["speaking_rate"] = (speaking_steps_total / denom_steps) if denom_steps else None
    # Recall coverage and NA breakdowns (conditioned on speaking steps)
    def _coverage_and_na(prefix: str) -> Dict[str, Any]:
        # Under new policy, speaking+judge_present steps are applicable; NA arises from judge missing
        den = 0
        na = 0
        breakdown: Dict[str, int] = {"judge_missing": 0}
        for r in step_rows:
            if r.get("observed_speak") != 1:
                continue
            if r.get("delegate_judge_present") == 1:
                den += 1
            else:
                na += 1
                breakdown["judge_missing"] += 1
        cov = (den / speaking_steps_total) if speaking_steps_total else None
        return {
            f"recall_{prefix}_den": den,
            f"recall_{prefix}_na": na,
            f"recall_{prefix}_na_breakdown": breakdown,
            f"recall_{prefix}_coverage": cov,
        }
    out.update(_coverage_and_na("strict"))
    out.update(_coverage_and_na("loose"))
    # Total no_match across meeting (strict variant)
    out["recall_nomatch_count_total"] = sum(
        1 for r in step_rows
        if r.get("observed_speak") == 1 and r.get("delegate_judge_present") == 1 and not r.get("delegate_matched_utt_id")
    )
    # Timing & floor
    env_timing = [r.get("env_timing_delta") for r in step_rows if r.get("env_timing_delta") is not None]
    out["env_timing_delta_mean"] = _safe_mean(env_timing)
    cats = [r.get("env_category") for r in step_rows if r.get("env_category")]
    denom_steps = len(step_rows) or 1
    def rate(label: str) -> float:
        return sum(1 for c in cats if c == label) / denom_steps
    out.update({
        "env_aligned_immediate_rate": rate("Aligned-Immediate"),
        "env_aligned_late_rate": rate("Aligned-Late"),
        "env_topic_only_rate": rate("Topic-Only"),
        "env_missed_cue_rate": sum(rate(x) for x in ["Missed-Cue", "Missed-Cue (Novel)"]),
    })
    # Delegate content
    out["delegate_decision_appropriateness_mean"] = _safe_mean([r.get("delegate_decision_appropriateness") for r in step_rows])
    out["delegate_relevance_mean"] = _safe_mean([r.get("delegate_relevance") for r in step_rows])
    out["delegate_grounding_mean"] = _safe_mean([r.get("delegate_grounding") for r in step_rows])
    out["redundancy_rate"] = sum(1 for r in step_rows if r.get("redundancy") is True) / denom_steps
    out["hallucination_rate"] = sum(1 for r in step_rows if r.get("hallucinated_claim") is True) / denom_steps
    out["off_topic_rate"] = sum(1 for r in step_rows if r.get("off_topic") is True) / denom_steps
    # Redundancy kinds (when provided)
    kinds = [str(r.get("delegate_redundancy_kind")).lower() for r in step_rows if r.get("delegate_redundancy_kind")]
    denom_kinds = len(kinds) or 1
    out.update({
        "redundancy_duplicate_rate": sum(1 for k in kinds if k == "duplicate")/denom_kinds,
        "redundancy_ack_rate": sum(1 for k in kinds if k == "ack")/denom_kinds,
        "redundancy_synthesis_rate": sum(1 for k in kinds if k == "synthesis")/denom_kinds,
        "redundancy_fresh_rate": sum(1 for k in kinds if k == "fresh")/denom_kinds,
    })
    # Coverage treatment agreement (when both coverage recommendation and redundancy_kind exist)
    def _map_treatment(rt: Optional[str]) -> Optional[str]:
        if not rt:
            return None
        t = str(rt).strip().lower()
        if t in {"skip"}:
            return "duplicate"
        if t in {"acknowledge", "reinforce"}:
            return "ack"
        if t in {"fresh"}:
            return "fresh"
        if t in {"synthesis", "synthesise", "synth"}:
            return "synthesis"
        return None
    agree = 0
    den_agree = 0
    for r in step_rows:
        kind = r.get("delegate_redundancy_kind")
        rt = r.get("coverage_recommended_treatment")
        mk = str(kind).lower() if kind else None
        mt = _map_treatment(rt)
        if mk and mt:
            den_agree += 1
            if mk == mt or (mk == "synthesis" and mt == "ack"):
                agree += 1
    out["coverage_treatment_agreement_rate"] = (agree/den_agree) if den_agree else None
    # Recall
    def rate_field(name: str) -> Optional[float]:
        vals = [r.get(name) for r in step_rows if r.get(name) is not None]
        if not vals:
            return None
        return sum(1 for v in vals if v == 1)/len(vals)
    out["delegate_hit_strict_rate"] = rate_field("delegate_hit_strict")
    out["delegate_hit_loose_rate"]  = rate_field("delegate_hit_loose")
    out["env_recall_strict_rate"]      = rate_field("env_recall_strict")
    out["env_recall_loose_rate"]       = rate_field("env_recall_loose")
    # Stewardship
    out["treatment_violation_rate"] = sum(1 for r in step_rows if r.get("treatment_violation") == 1)/denom_steps
    out["spoke_against_global_suggestion_rate"] = sum(1 for r in step_rows if r.get("spoke_against_global_suggestion") == 1)/denom_steps
    # Patch deltas (means; fractions positive)
    def mean_delta(name: str) -> Optional[float]:
        vals = [ep.get(name) for ep in episode_rows if ep.get(name) is not None]
        return (sum(vals)/len(vals)) if vals else None
    out["delta_intent_mean"] = mean_delta("delta_intent")
    out["delta_topic_mean"] = mean_delta("delta_topic")
    out["delta_floor_mean"] = mean_delta("delta_floor")
    out["delta_decision_appropriateness_mean"] = mean_delta("delta_decision_appropriateness")
    out["delta_relevance_mean"] = mean_delta("delta_relevance")
    out["delta_grounding_mean"] = mean_delta("delta_grounding")
    def frac_pos(name: str) -> Optional[float]:
        vals = [ep.get(name) for ep in episode_rows if isinstance(ep.get(name), (int, float))]
        if not vals:
            return None
        return sum(1 for v in vals if v > 0)/len(vals)
    out["fraction_positive_delta_intent"] = frac_pos("delta_intent")
    out["fraction_positive_delta_decision"] = frac_pos("delta_decision_appropriateness")
    # Totals
    out.update({
        "n_steps": len(step_rows),
        "n_episodes": len(episode_rows),
        "n_participants": len({r.get("participant_id") for r in step_rows}),
        "emitted_step_count": len(step_rows),
        "emitted_episode_count": len(episode_rows),
    })

    # -----------------------------
    # Additional idea-aligned aggregates
    # -----------------------------
    speaks = [r for r in step_rows if r.get("observed_speak") == 1]
    out["spoken_step_count"] = len(speaks)
    out["context_scored_step_count"] = sum(
        1 for r in speaks if isinstance(r.get("delegate_context_appropriateness"), (int, float))
    )
    out["relevance_scored_step_count"] = sum(
        1 for r in speaks if isinstance(r.get("delegate_relevance"), (int, float))
    )
    out["grounding_scored_step_count"] = sum(
        1 for r in speaks if isinstance(r.get("delegate_grounding"), (int, float))
    )
    out["judge_contract_violation_count"] = sum(
        1 for r in step_rows if r.get("judge_contract_violation") in (1, True)
    )
    out["delegate_judge_failure_count"] = sum(
        1 for r in step_rows if r.get("delegate_judge_failure") in (1, True)
    )

    def _mean(field: str, rows: List[Dict[str, Any]]) -> Optional[float]:
        vals = [r.get(field) for r in rows if isinstance(r.get(field), (int, float))]
        return (sum(vals) / len(vals)) if vals else None

    # Content-quality means (speaking steps only)
    out["delegate_context_appropriateness_mean"] = _mean("delegate_context_appropriateness", speaks)
    out["delegate_relevance_mean"] = _mean("delegate_relevance", speaks)
    out["delegate_grounding_mean"] = _mean("delegate_grounding", speaks)

    # Idea matching quality (speaking steps)
    speaks_with_match = [r for r in speaks if r.get("matched_unit_id") is not None]
    out["matched_unit_rate"] = (len(speaks_with_match) / len(speaks)) if speaks else None
    synchronized = [r for r in speaks_with_match if r.get("unit_timing_delta") == 0]
    proactive = [r for r in speaks_with_match if isinstance(r.get("unit_timing_delta"), (int, float)) and r.get("unit_timing_delta") > 0]
    out["idea_match_synchronized_rate"] = (len(synchronized) / len(speaks)) if speaks else None
    out["idea_match_proactive_rate"] = (len(proactive) / len(speaks)) if speaks else None

    # Anchor-aware attempt rates
    anchor_steps = [r for r in step_rows if r.get("is_idea_anchor") == 1]
    if anchor_steps:
        anchor_attempts = [r for r in anchor_steps if r.get("matched_unit_id") is not None]
        out["idea_attempt_when_anchor_rate"] = len(anchor_attempts) / len(anchor_steps)
    else:
        out["idea_attempt_when_anchor_rate"] = None

    # Decision quality split by anchor presence
    def _rate_dec_ok(rows: List[Dict[str, Any]], threshold: float = 0.8) -> Optional[float]:
        vals = [r.get("delegate_decision_appropriateness") for r in rows if isinstance(r.get("delegate_decision_appropriateness"), (int, float))]
        if not vals:
            return None
        return sum(1 for v in vals if v >= threshold) / len(vals)

    out["decision_ok_on_anchor_rate"] = _rate_dec_ok(anchor_steps)
    non_anchor_steps = [r for r in step_rows if r.get("is_idea_anchor") != 1]
    out["decision_ok_off_anchor_rate"] = _rate_dec_ok(non_anchor_steps)

    # Safety flags on speaking steps
    def _flag_rate(field: str, rows: List[Dict[str, Any]]) -> Optional[float]:
        scored_rows = [r for r in rows if isinstance(r.get(field), bool)]
        if not scored_rows:
            return None
        return sum(1 for r in scored_rows if r.get(field) is True) / len(scored_rows)

    out["redundancy_rate_speaks"] = _flag_rate("redundancy", speaks)
    out["hallucination_rate_speaks"] = _flag_rate("hallucinated_claim", speaks)
    out["off_topic_rate_speaks"] = _flag_rate("off_topic", speaks)

    # Judge availability on speaking steps
    if speaks:
        missing = sum(1 for r in speaks if r.get("delegate_judge_present") == 0)
        out["delegate_judge_missing_rate_speaks"] = missing / len(speaks)
    else:
        out["delegate_judge_missing_rate_speaks"] = None

    return out
