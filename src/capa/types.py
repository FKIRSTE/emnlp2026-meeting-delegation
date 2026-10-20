"""
CAPA core types: prediction, error reports, pending items, episodic events.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Intent-level forecast of the next conversational turn."""
    speaker_id: Optional[str] = Field(default=None, description="Hypothesized next speaker.")
    intent_summary: str = Field(description="Concise description of the intended content.")
    micro_topic: Optional[str] = Field(default=None, description="Fine-grained topic focus.")
    floor_effect: Optional[str] = Field(default=None, description="Predicted floor dynamics label.")
    confidence: Optional[float] = Field(default=None, description="Optional 0–1 confidence of prediction.")
    intent_signature: Optional[str] = Field(default=None, description="Stable signature for pending matching.")


class ErrorReport(BaseModel):
    """Judge output describing alignment and error category."""
    intent_score: float = Field(ge=0.0, le=1.0)
    topic_score: float = Field(ge=0.0, le=1.0)
    floor_score: float = Field(ge=0.0, le=1.0)
    matched_utt_id: Optional[str] = Field(default=None)
    timing_delta: Optional[int] = Field(default=None, description="0 for t+1 (anchor), 1..k-1 for t+2..t+k (late); null if no match")
    category: str = Field(description="One of the CAPA error categories.")
    details: Optional[str] = Field(default=None)
    novelty: Optional[bool] = Field(default=None, description="True if observed turn introduces a new key point/topic not present in WM or last_utterances.")
    gt_is_banal: Optional[bool] = Field(default=None, description="True if the current GT turn is phatic/banal/backchannel.")


class PendingItem(BaseModel):
    intent_signature: Optional[str] = None
    topic_label: Optional[str] = None
    inserted_at_turn: int
    relevance: float = Field(ge=0.0, le=1.0)
    ttl_left: Optional[int] = None
    matched_utt_id: Optional[str] = None


class EpisodicEvent(BaseModel):
    cue_utt_id: str
    prediction: PredictionResponse
    gt_utt_id: Optional[str] = None
    error_report: Optional[ErrorReport] = None
    patch_summary: Optional[str] = None
    pending_snapshot: Optional[List[PendingItem]] = None


class DelegateErrorReport(BaseModel):
    """Judge output for delegate (Controller/Generator) behavior per turn."""
    # Identity (set programmatically by the pipeline; the LLM need not emit it)
    represented_user: Optional[str] = Field(default=None, description="Participant id/role the delegate represents for this judgment.")
    # Scores (0..1)
    decision_appropriateness: float = Field(ge=0.0, le=1.0, description="Action-choice quality for speaking vs remaining silent given floor, cue, and opportunity structure. Always scored; content defects should mainly surface in the other fields.")
    context_appropriateness: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Local semantic and temporal fit to an accepted same-proposition GT utterance. Null when silent or when no accepted GT match exists.")
    relevance: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Directness to the active local discussion need, not just broad topical relatedness. Null ONLY for Yield/Silent with no utterance content.")
    grounding: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Support for the asserted content, including added specifics, from current context, EIM, and payload-provided evidence. Null ONLY for Yield/Silent with no utterance content.")

    # Flags
    floor_violation: bool = Field(description="True if delegate mishandled the floor (spoke when shouldn't OR silent when should speak).")
    floor_violation_kind: Optional[str] = Field(default=None, description="One of: intrusion | omission. Intrusion=spoke on other-directed/held floor; omission=silent despite explicit cue to self. Null when floor_violation=false.")
    off_topic: bool = Field(description="True if content diverged from snapshot_major_topic/micro-topic.")
    redundancy: bool = Field(description="True if repeated covered point with low novelty.")
    hallucinated_claim: bool = Field(description="True if unsupported or contradictory specifics are asserted as fact relative to current context, EIM, or payload-provided evidence.")
    # Optional refined redundancy classification (if available from judge)
    redundancy_kind: Optional[str] = Field(default=None, description="One of: duplicate | ack | synthesis | fresh. Optional; default inferred as 'duplicate' when redundancy=true and kind missing.")
    redundancy_reason: Optional[str] = Field(default=None, description="Short rationale for redundancy_kind assignment.")

    # Summary
    category: str = Field(description="One of: Good | FloorViolation | OffTopic | Redundant | Hallucination | Banal")
    matched_utt_id: Optional[str] = Field(default=None)
    timing_delta: Optional[int] = Field(default=None, description="0..k-1 index in current trimmed gt_window (0=anchor t+1, 1=t+2, ...); null if no match or silent.")
    details: Optional[str] = Field(default=None)
    gt_is_banal: Optional[bool] = Field(default=None, description="True if the current GT turn is phatic/banal/backchannel.")
    matched_unit_id: Optional[str] = Field(default=None, description="IdeaUnit ID the delegate utterance attempts to express (episode-local).")
    unit_timing_delta: Optional[int] = Field(default=None, description="Number of utterances between delegate's contribution step (GT window anchor) and idea unit anchor. 0=at anchor, 1+=early. Formula: unit_anchor_idx - step_anchor_idx.")
    match_window_status: Optional[str] = Field(default=None, description="accepted_current_window | same_past_outside_window | partial_near_miss | none.")
    matched_response_utt_id: Optional[str] = Field(default=None, description="Official in-window response utterance id credited for the idea-unit match.")
    matched_response_delta: Optional[int] = Field(default=None, description="Delta from the current step anchor to the credited official response utterance.")
    first_response_delta: Optional[int] = Field(default=None, description="Delta from the current step anchor to the first official response utterance for the matched idea unit.")
    matched_response_text: Optional[str] = Field(default=None, description="Exact official target utterance text credited for the match, when available.")
    matched_target_summary: Optional[str] = Field(default=None, description="Official target summary for the matched idea unit, when available.")
    matched_target_context: Optional[str] = Field(default=None, description="Compact local context used to bind the matched target proposition, when available.")
    match_decision_rationale: Optional[str] = Field(default=None, description="Verifier/reconciliation rationale for the final match decision.")
    judge_contract_violation: bool = Field(default=False, description="True when the final reconciled report still violates the delegate-judge field contract after one repair retry.")
    judge_contract_failure_reasons: Optional[List[str]] = Field(default=None, description="Short machine-readable reasons describing any unresolved delegate-judge contract failure.")
    judge_contract_repair_attempted: bool = Field(default=False, description="True when the pipeline requested one repair retry because the draft report violated the delegate-judge contract.")


class GTMatchVerifierReport(BaseModel):
    """Narrow verifier output for GT-window semantic matching."""

    matched_utt_id: Optional[str] = Field(default=None, description="Accepted GT utterance id only when proposition_relation=same and the match is strong enough to count; null otherwise.")
    proposition_relation: Literal["same", "partial", "different"] = Field(description="same means the same core proposition, partial means directionally aligned but not enough to count, and different means no substantive proposition match.")
    same_proposition: bool = Field(description="Whether delegate and GT express the same core proposition. True only when proposition_relation=same.")
    match_strength: Literal["exact", "strong", "weak", "none"] = Field(description="Discrete semantic-alignment strength. Official acceptance requires proposition_relation=same and strength in {exact,strong}.")
    delegate_core_proposition: Optional[str] = Field(default=None, description="Short normalized proposition extracted from the delegate utterance.")
    gt_core_proposition: Optional[str] = Field(default=None, description="Short normalized proposition extracted from the matched GT utterance, if any.")
    rationale: Optional[str] = Field(default=None, description="Short explanation for the verification outcome.")
    matched_response_text: Optional[str] = Field(default=None, description="Exact official response text used for the match, if any.")
    matched_target_summary: Optional[str] = Field(default=None, description="Official target summary used for the match, if any.")
    match_decision_rationale: Optional[str] = Field(default=None, description="Short rationale focused on why the match was accepted or rejected.")


class RedundancyVerifierReport(BaseModel):
    """Narrow verifier output for redundancy classification."""

    redundant: bool = Field(description="Whether the delegate utterance repeats an already-made substantive point with negligible added value.")
    redundancy_kind: Literal["duplicate", "ack", "synthesis", "fresh"] = Field(description="Redundancy subtype or fresh-value classification.")
    prior_reference_id: Optional[str] = Field(default=None, description="Recent delegate-action reference that the current utterance repeats or builds on, if any.")
    adds_new_value: bool = Field(description="Whether the utterance adds substantive new value relative to prior delegate content.")
    rationale: Optional[str] = Field(default=None, description="Short explanation for the redundancy decision.")


__all__ = [
    "PredictionResponse",
    "ErrorReport",
    "DelegateErrorReport",
    "GTMatchVerifierReport",
    "RedundancyVerifierReport",
    "PendingItem",
    "EpisodicEvent",
]


# ==========================
# Recalibrator Enhancements
# ==========================

class ControllerGuidance(BaseModel):
    """Advisory recalibration signals for Controller decision-making.
    
    These are SOFT SIGNALS that help the Controller make better decisions.
    They never override floor rules, explicit cues, or social appropriateness.
    """
    
    # Contribution pressure based on idea unit coverage + recency
    contribution_pressure: Optional[Literal["dial-up", "dial-down", "neutral"]] = Field(
        default=None,
        description="dial-up: opportunities remain, be alert. dial-down: lean toward silence. neutral: follow normal signals."
    )
    
    # Rich error correction from prior step
    error_correction: Optional[str] = Field(
        default=None,
        description="Course-correction if prior turn had issues. E.g., 'Prior contribution off-topic. Focus was X.'"
    )
    
    # Meta-guidance for helper signal weighting
    strategy_adjustment: Optional[str] = Field(
        default=None,
        description="Adjust approach based on error patterns. E.g., 'Weight Coverage recommendations higher.'"
    )
    
    # Backward-looking feedback on last action
    last_action_outcome: Optional[str] = Field(
        default=None,
        description="What happened with prior action. E.g., 'matched (timing=2)' | 'no_match' | 'appropriate_silence'"
    )
    
    # Cooldown after recent contribution
    post_contribution_cooldown: Optional[bool] = Field(
        default=None,
        description="True if recently contributed. Consider yielding."
    )
    
    # Episode-level contribution balance
    contribution_balance: Optional[Literal["under", "over", "balanced"]] = Field(
        default=None,
        description="Episode-level: 'under' if few contributions, 'over' if many, 'balanced' otherwise."
    )
    
    # One-line summary
    guidance_summary: Optional[str] = Field(
        default=None,
        description="One sentence summarizing the key guidance signal."
    )


# Backward compatibility alias
GuidanceBundle = ControllerGuidance


class RecalibrationOutput(BaseModel):
    """Container for recalibrator outputs in 'recalibrator_drives_state' mode."""
    # These will be validated externally against schemas before being set
    diff: Optional[object] = Field(default=None, description="EIMMasterState diff object")
    snapshot_analysis: Optional[object] = Field(default=None, description="SnapshotAnalysis object for (t-N..t+1)")
    guidance: Optional[GuidanceBundle] = Field(default=None)
    anchor_utterance_id: Optional[str] = Field(default=None)


__all__ += [
    "ControllerGuidance",
    "GuidanceBundle",  # Backward compat alias
    "RecalibrationOutput",
]
