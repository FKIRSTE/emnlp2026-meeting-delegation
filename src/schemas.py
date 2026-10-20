"""
Pydantic Schemas for Meeting Delegate System

All schema definitions extracted from the original notebook into a single file
for better organization and reusability across modules.
"""

from pydantic import BaseModel, Field, ValidationError, model_validator
from typing import List, Dict, Optional, Any

# ==============================================================================
#  Idea Unit Schema (Idea-Units regime)
# ==============================================================================

class IdeaUnit(BaseModel):
    """Ground-truth Idea Unit (one coherent idea by the represented participant)."""
    unit_id: str = Field(description="Stable identifier for the Idea Unit (e.g., U1).")
    topic_summary: str = Field(description="One-line normalized idea summary.")
    response_utterance_ids: List[str] = Field(description="Utterance IDs (represented user only) that express this idea.")
    first_response_idx: int = Field(description="Global transcript idx of the first substantive idea utterance.")
    cue_utterance_id: str = Field(description="Utterance ID of the triggering cue.")
    cue_idx: int = Field(description="Global transcript idx of the triggering cue.")
    cue_type: str = Field(description="Cue type: explicit | implicit-handoff | open_floor | self-initiation.")
    semantic_cue_utterance_id: Optional[str] = Field(default=None, description="Optional discourse-origin cue utterance id when the emitted cue is a fixed-k local re-anchor.")
    semantic_cue_idx: Optional[int] = Field(default=None, description="Optional discourse-origin cue transcript idx.")
    semantic_cue_type: Optional[str] = Field(default=None, description="Optional discourse-origin cue type.")
    anchor_reason: Optional[str] = Field(default=None, description="Rationale for linking cue to this idea.")
    importance_score: Optional[float] = Field(default=None, description="Selector-provided importance (0..1).")
    confidence: Optional[float] = Field(default=None, description="Selector confidence (0..1).")
    anchor_utt_id: Optional[str] = Field(default=None, description="(Convenience) Anchor utterance id, typically response_utterance_ids[0].")
    selector_response_utterance_ids: Optional[List[str]] = Field(default=None, description="Original selector-emitted response utterance ids before fixed-k official-window trimming.")
    selector_topic_summary: Optional[str] = Field(default=None, description="Original selector topic summary before any conservative fixed-window summary repair.")
    trimmed_response_utterance_ids: Optional[List[str]] = Field(default=None, description="Response utterance ids retained after fixed-k trimming and before same-proposition constituent expansion.")
    expanded_response_utterance_ids: Optional[List[str]] = Field(default=None, description="Additional same-speaker in-window constituent utterance ids added to the official target.")
    official_constituent_expansion_reason: Optional[str] = Field(default=None, description="Short reason describing why official response constituents were expanded.")
    official_target_text: Optional[str] = Field(default=None, description="Official in-window represented-user target text used for recall.")
    canonical_target_proposition: Optional[str] = Field(default=None, description="Conservative judge-facing proposition built from official target evidence, local context, and participant-local inventory support.")
    tested_facet_text: Optional[str] = Field(default=None, description="For testable_partial targets, the concrete supported facet used as the official recall target.")
    summary_rewritten: Optional[bool] = Field(default=None, description="Whether prep rewrote the official topic summary/canonical target after fixed-k trimming or availability verification.")
    summary_rewrite_reason: Optional[str] = Field(default=None, description="Machine-readable reason for summary/canonical target rewriting.")
    official_response_id_count: Optional[int] = Field(default=None, description="Number of represented-user response ids retained inside the fixed-k GT window.")
    response_window_trimmed: Optional[bool] = Field(default=None, description="Whether response_utterance_ids were trimmed to keep the official target inside gt_next_window.")
    target_availability_verdict: Optional[str] = Field(default=None, description="Prep verifier verdict: supported | testable_partial | unsupported_generic | unsupported_missing.")
    target_availability_method: Optional[str] = Field(default=None, description="Verifier method used for participant-local target availability.")
    target_available_to_delegate: Optional[bool] = Field(default=None, description="Whether prep deemed this official target fairly available to the represented delegate.")
    target_availability_reason: Optional[str] = Field(default=None, description="Machine-readable target availability reason.")
    target_availability_evidence_ids: Optional[List[str]] = Field(default=None, description="Participant inventory evidence ids supporting the official target.")
    target_availability_missing_facets: Optional[List[str]] = Field(default=None, description="Important target facets not supported by participant inventory.")
    target_availability_rationale: Optional[str] = Field(default=None, description="Short rationale for target availability decision.")
    quality_status: Optional[str] = Field(default=None, description="Prep quality gate status for this official target.")
    quality_drop_reason: Optional[str] = Field(default=None, description="Prep quality drop reason when not kept.")
    response_substance_score: Optional[float] = Field(default=None, description="Deterministic response-substance score used by prep diagnostics.")
    context_already_covered: Optional[bool] = Field(default=None, description="Whether visible context already covered the official target.")


class CoverageSummary(BaseModel):
    """Coverage counts by cue type for Idea Unit selection.
    
    Uses Field aliases to support hyphenated JSON keys from LLM output
    (e.g., 'implicit-handoff' in JSON maps to 'implicit_handoff' in Python).
    """
    explicit: int = Field(default=0, description="Count of explicit cue idea units.")
    implicit_handoff: int = Field(default=0, alias="implicit-handoff", description="Count of implicit handoff idea units.")
    open_floor: int = Field(default=0, description="Count of open floor idea units.")
    self_initiation: int = Field(default=0, alias="self-initiation", description="Count of self-initiation idea units.")
    
    model_config = {"populate_by_name": True}  # Allow both snake_case and hyphen-case


class IdeaUnitSelectorResponse(BaseModel):
    """Response schema for the Idea Unit Selector LLM call.
    
    Used as response_model for structured output with GPT-5-mini and Gemini.
    """
    idea_units: List[IdeaUnit] = Field(default_factory=list, description="List of extracted idea units.")
    coverage_summary: CoverageSummary = Field(default_factory=CoverageSummary, description="Coverage counts by cue type.")


# ==============================================================================
#  User Profile Schemas (Long-Term Memory)
# ==============================================================================

class InferredRoleAndGoals(BaseModel):
    primary_role: str = Field(description="A high-level summary of the participant's primary function in the meeting (e.g., 'Meeting Facilitator', 'Technical Expert', 'Marketing Stakeholder').")
    high_level_goals: List[str] = Field(description="A list of the participant's overarching objectives, more abstract than specific intents.")


class SpecificKnowledgePoint(BaseModel):
    Context: str = Field(description="The generalized context in which this knowledge was shared.")
    Information: str = Field(description="The concrete fact, opinion, or proposal.")


class CoreInfo(BaseModel):
    knowledge_areas: List[str] = Field(description="A list of general topics or domains this participant has expertise in.")
    specific_knowledge_points: List[SpecificKnowledgePoint] = Field(description="Concrete knowledge shared by the participant.")


class CommunicationStyle(BaseModel):
    formality: str = Field(description="Level of formality (e.g., 'Formal', 'Semi-formal', 'Informal').")
    verbosity: str = Field(description="Speaking pattern (e.g., 'Concise', 'Moderate', 'Verbose').")
    typical_tone: str = Field(description="Overall tone (e.g., 'Analytical', 'Collaborative', 'Questioning').")
    jargon_usage: str = Field(description="Description of their use of technical language.")
    questioning_style: str = Field(description="Description of how and how often they ask questions.")
    common_fillers_catchphrases: Optional[List[str]] = Field(default=[])
    interaction_patterns: List[str] = Field(description="Observed patterns of interaction (e.g., 'Builds on others ideas', 'Summarizes discussions').")


class ConfidenceOnTopic(BaseModel):
    Topic_Category: str
    Apparent_Confidence: str = Field(description="e.g., High, Medium, Low")


class ExpressedStance(BaseModel):
    Topic: str
    Stance: str


class KnowledgeStance(BaseModel):
    confidence_on_topics: Optional[List[ConfidenceOnTopic]] = Field(default=[])
    expressed_stances: Optional[List[ExpressedStance]] = Field(default=[])


class UserProfile(BaseModel):
    inferred_role_and_goals: InferredRoleAndGoals
    core_info: CoreInfo
    communication_style: CommunicationStyle
    knowledge_stance: Optional[KnowledgeStance] = None
    identity_aliases: Optional[List[str]] = Field(
        default=None,
        description="Known name/alias strings that refer to this participant inside the meeting (e.g., nicknames, first names, initials)."
    )


class RichProfileGenerationResponse(BaseModel):
    Thoughts: str = Field(description="Step-by-step reasoning for the profile generation, showing synthesis from observations to abstractions.")
    UserProfile: UserProfile


# ==============================================================================
#  EIM (Environment Interpretation Module) Schemas
# ==============================================================================

class StanceEntry(BaseModel):
    stance_summary: Optional[str] = Field(default=None, description="Concise summary of stance (present in EIM data).")
    source_utterance_id: Optional[str] = Field(default=None, description="Utterance ID where stance was expressed (present in EIM data).")
    confidence: Optional[float] = Field(default=None, description="Optional 0–1 confidence for this stance.")


class KeyPoint(BaseModel):
    point: str = Field(description="A key point or conclusion from the discussion.")
    speaker: str = Field(description="The speaker who made this point.")
    utterance_id: str = Field(description="The utterance ID where this point was made.")
    kind: Optional[str] = Field(default=None, description="Optional kind: 'fact'|'proposal'|'decision'|'question'.")


class OpenQuestion(BaseModel):
    question_summary: str = Field(description="Concise statement of the open question.")
    raised_by: str = Field(description="Speaker ID who raised the question.")
    source_utterance_id: str = Field(description="Utterance ID where the question was raised.")
    resolved: bool = Field(default=False, description="Whether the question is resolved.")
    resolved_at_utterance_id: Optional[str] = Field(default=None, description="Utterance ID where resolution occurred, if any.")


class DecisionEntry(BaseModel):
    decision_summary: str = Field(description="Concise statement of the decision.")
    decided_by: Optional[str] = Field(default=None, description="Speaker ID or role who decided.")
    source_utterance_id: str = Field(description="Utterance ID where the decision was made.")
    status: Optional[str] = Field(default="Tentative", description="'Tentative' | 'Final'.")
    confidence: Optional[float] = Field(default=None, description="Optional 0–1 confidence for this decision entry.")


class TopicLogEntry(BaseModel):
    label: str = Field(description="A concise label for the topic.")
    last_updated_at_utterance_id: Optional[str] = Field(
        default=None,
        description="The ID of the last utterance that contributed to this topic."
    )
    status: str = Field(description="The current status: 'Ongoing', 'Resolved', 'Deferred', etc.")
    key_points: List[KeyPoint] = Field(description="Key points and conclusions for this topic.")
    open_questions: Optional[List[OpenQuestion]] = Field(default_factory=list, description="Optional list of open questions under this topic.")
    decisions: Optional[List[DecisionEntry]] = Field(default_factory=list, description="Optional list of decisions under this topic.")


class EIMParticipantModel(BaseModel):
    """A dynamic, in-meeting profile for a single participant."""
    inferred_expertise_keywords: List[str] = Field(
        default=[],
        description="A running list of keywords suggesting this person's expertise demonstrated *in this meeting* (e.g., 'database', 'API')."
    )
    
    expressed_stance_on_topics: Dict[str, List[StanceEntry]] = Field(
        default={},
        description="Maps topic labels to a summary of their expressed stance)."
    )
    
    recent_dialogue_act: Optional[str] = Field(
        default=None,
        description="The dialogue act of their most recent utterance (e.g., 'Questioning', 'Proposing', 'Agreeing')."
    )
    engagement_score: Optional[float] = Field(default=None, description="Optional 0–1 short-term activity salience for this participant.")

class LastUtteranceContext(BaseModel):
    """Context information for the very last utterance in a snapshot."""
    speaker: str = Field(description="The speaker of the last utterance")
    content: str = Field(description="The exact text content of the last utterance")
    utterance_id: str = Field(description="The unique identifier of the last utterance")
    intent_type: str = Field(description="e.g., 'Direct Question', 'Statement', 'Request'")
    target_addressed: Optional[str] = Field(default=None, description="Who is being addressed, if specific")

class MicroTopicAnalysis(BaseModel):
    """Enhanced analysis of conversational micro-topics with hierarchical granularity (MESO + MICRO levels)."""
    broader_context: str = Field(description="MESO-level: SPECIFIC thematic focus of the last 4-5 substantive utterances. Must be MORE SPECIFIC than snapshot_major_topic (not near-duplicate). E.g., 'Evaluating OAuth 2.0 vs API key authentication trade-offs' not just 'API security'")
    immediate_focus: str = Field(description="MICRO-level: active focus of the last 1-2 relevant substantive utterances after filtering phatic, banal, backchannel, fragmentary, and disfluent noise.")
    topic_continuity: str = Field(description="'CONTINUATION' if immediate_focus continues broader_context, 'SHIFT' if immediate_focus introduces new focus within or diverging from broader_context")
    last_utterance_verbatim: Optional[LastUtteranceContext] = Field(default=None, description="Verbatim content and metadata of the very last utterance; this is exact final-turn evidence, not the same thing as immediate_focus.")
    immediate_focus_sanity_flag: Optional[str] = Field(default=None, description="Optional postprocess flag when immediate_focus had to be repaired or checked against noisy final-turn content.")

class SnapshotAnalysis(BaseModel):
    """The detailed analysis of the *immediate* local context of a snapshot."""
    # Core micro-topic structure (required)
    micro_topic_analysis: Optional[MicroTopicAnalysis] = Field(default=None, description="Hierarchical micro-topic analysis with MESO (broader_context) and MICRO (immediate_focus) levels. Each level must be progressively more specific.")
    
    snapshot_major_topic: Optional[str] = Field(default=None, description="MACRO-level: Broad theme/agenda of the ENTIRE snapshot window. E.g., 'Q4 API development planning meeting'. The MESO/MICRO levels in micro_topic_analysis must be MORE SPECIFIC than this.")
    conversational_state: str = Field(description="The current phase of the conversation (e.g., 'Brainstorming Solutions', 'Decision Making').")
    conversational_floor_status: str = Field(description="e.g., 'Open_Floor', 'Directed_Turn (to: SpeakerX)', 'Dyadic_Exchange (between: SpeakerX, SpeakerY)'.")
    floor_rationale: Optional[str] = Field(default=None, description="One-line rationale describing why the current floor status was selected.")
    key_points_made_in_snapshot: List[KeyPoint] = Field(description="A summary of key points made *only within this snapshot*.")
    
    @property
    def effective_micro_topic(self) -> str:
        """Get the most appropriate micro-topic based on micro_topic_analysis.
        
        Prioritizes immediate_focus when there's a SHIFT, otherwise uses broader_context.
        """
        if self.micro_topic_analysis:
            # If there's a topic shift, prioritize immediate focus
            if self.micro_topic_analysis.topic_continuity == "SHIFT":
                return self.micro_topic_analysis.immediate_focus
            else:
                return self.micro_topic_analysis.broader_context
        else:
            return self.snapshot_major_topic or "Unknown"


class EIMMeetingStateVector(BaseModel):
    """
    The final, comprehensive output object for any given turn (Mt).
    This is the "world model" that the EIM provides to the SMM.
    """
    last_processed_utterance_id: str = Field(description="The final utterance ID processed to generate this state.")

    # Global, historically-aware context
    topic_summary_log: Dict[str, TopicLogEntry] = Field(description="The full, historically-aware log of all topics discussed so far.")
    participant_models: Dict[str, EIMParticipantModel] = Field(description="The full, historically-aware models of all participants.")
    identity_aliases: Optional[Dict[str, str]] = Field(default_factory=dict, description="Optional map of personal names to participant IDs or roles (discovered in meeting).")
   
    # Local, immediate context
    snapshot_analysis: SnapshotAnalysis = Field(description="The deep analysis of the immediate snapshot.")

class EIMMasterState(BaseModel):
    """
    Defines the structure of the EIM's persistent, evolving memory of the meeting.
    This is what gets updated chunk by chunk.
    """
    topic_summary_log: Dict[str, TopicLogEntry] = Field(default={})
    participant_models: Dict[str, EIMParticipantModel] = Field(default={})
    identity_aliases: Optional[Dict[str, str]] = Field(default_factory=dict, description="Optional map of personal names to participant IDs or roles (discovered in meeting).")


# ==============================================================================
#  Test Case Generation Schemas
# ==============================================================================

class TestCaseInstance(BaseModel):
    Explanation: str = Field(description="Reasoning for selecting and categorizing this instance.")
    Type: str = Field(description="The category: 'Chime In', 'Explicit Cue', or 'Implicit Cue'.")
    Response_IDs: List[str] = Field(description="List of utterance_ids for the participant's response.")
    ID: str = Field(description="The single utterance_id of the cue or preceding turn.")
    Speaker: str = Field(description="The speaker_id for the utterance with 'ID'.")


class TestCaseExtractionResponse(BaseModel):
    extracted_data: List[TestCaseInstance]


# ==============================================================================
#  Episodic Memory & Coverage Tracking Schemas
# ==============================================================================

class CandidateCoverageStatus(BaseModel):
    """
    O(1) lookup structure for tracking candidate coverage across episodes.
    Replaces O(N) linear scanning through episodic history for coverage checks.
    """
    candidate_id: str = Field(description="Stable identifier for the candidate (e.g., PM_pt_001, UX_pt_042)")
    text: str = Field(description="Canonical text of the talking point")
    source: Optional[str] = Field(default=None, description="Origin: JIT | Profile | Synthesized")
    
    # Coverage flags for O(1) lookup
    covered_in_current_episode: bool = Field(
        default=False, 
        description="True if delegate already said this point in THIS episode"
    )
    covered_in_previous_episodes: bool = Field(
        default=False,
        description="True if delegate said this point in PREVIOUS episodes"
    )
    
    # Provenance tracking
    last_covered_utterance_id: Optional[str] = Field(
        default=None,
        description="Most recent utterance_id where this candidate was covered"
    )
    last_covered_episode_id: Optional[str] = Field(
        default=None,
        description="Episode identifier where this candidate was last covered"
    )
    coverage_count: int = Field(
        default=0,
        description="Number of times this candidate has been used across all episodes"
    )
    
    # Metadata for analysis
    first_covered_at: Optional[str] = Field(
        default=None,
        description="Utterance_id where this candidate was first introduced"
    )


# ==============================================================================
#  Two-Tier User Module Schemas
# ==============================================================================

class ActionDetails(BaseModel):
    action_name: str = Field(description="The final chosen action from the Action Space.")
    scope: str = Field(description="The intended scope, either 'Concise' or 'Comprehensive'.")
    justification: str = Field(description="The reasoning behind the chosen action.")

class ContentPlan(BaseModel):
    semantic_intent: str = Field(description="A high-level instruction for the Generator.")
    knowledge_to_convey: Optional[str] = Field(default=None)
    concepts_to_frame: Optional[List[str]] = Field(default=None)
    
class ControllerOutput(BaseModel):
    content_plan: Optional[ContentPlan] = None
    action_details: ActionDetails

    # Helper-aligned digest so downstream analysis can trace decisions without re-running elimination
    class CandidateDigestItem(BaseModel):
        candidate_id: str = Field(description="Stable identifier for the candidate (e.g., UX_pt_001)")
        text: str = Field(description="Canonical wording of the talking point")
        source: Optional[str] = Field(default=None, description="'JIT', 'Profile', or 'Synthesized'")
        suitability: Optional[str] = Field(default=None, description="Helper-labelled suitability for this turn (High|Medium|Low)")
        recommended_treatment: Optional[str] = Field(default=None, description="Coverage recommendation (Fresh|Reinforce|Acknowledge|Skip)")
        suggested_usage: Optional[str] = Field(default=None, description="Helper usage recommendation (CorePoint|Reinforce|Acknowledge|Skip)")
        stance_function: Optional[str] = Field(default=None, description="Semantic stance role for the candidate: support|proposal|constraint|caution|question|facilitation|neutral.")
        relevance_score: Optional[int] = Field(default=None, description="1-10 helper relevance score")
        novelty_score: Optional[int] = Field(default=None, description="1-10 helper novelty score")
        helper_notes: Optional[str] = Field(default=None, description="Additional commentary from helper tools")
        support_utt_ids: Optional[List[str]] = Field(default=None, description="JIT/profile evidence utterance ids supporting this candidate, when available.")
        anchor_phrases: Optional[List[str]] = Field(default=None, description="Short phrases from source evidence supporting this candidate.")
        residual_contribution_value: Optional[str] = Field(default=None, description="Coverage residual value used for controller consistency checks.")
        viability_status: Optional[str] = Field(default=None, description="Programmatic viability label for this turn: viable | blocked.")
        viability_reason: Optional[str] = Field(default=None, description="Why the candidate is considered viable.")
        blocking_reason: Optional[str] = Field(default=None, description="Why the candidate is blocked, if blocked.")

    helper_candidates_digest: Optional[List[CandidateDigestItem]] = Field(default=None, description="Ordered list of curator / coverage annotated candidates passed to the decision-maker")
    active_candidate_id: Optional[str] = Field(default=None, description="Identifier of the candidate the controller ultimately acted upon (if any)")
    candidate_decision_trace: Optional[str] = Field(default=None, description="Short narrative tying the choice to helper guidance and meeting context")
    helper_global_suggestion: Optional[str] = Field(default=None, description="Derived from Controller's final action (Speak|Remain_Silent|Acknowledge) for evaluation purposes")
    speaker_recommendation: Optional[str] = Field(default=None, description="Participant ID recommended by speaker scorer, if provided")
    speaker_recommendation_confidence: Optional[float] = Field(default=None, description="Confidence for the speaker scorer recommendation (0-1)")
    override_applied: Optional[bool] = Field(default=None, description="Whether a programmatic post-controller guard changed the controller output.")
    pre_override_action_name: Optional[str] = Field(default=None, description="Controller action before the most recent programmatic override.")
    post_override_action_name: Optional[str] = Field(default=None, description="Controller action after the most recent programmatic override.")
    override_reason: Optional[str] = Field(default=None, description="Machine-readable reason for the most recent programmatic override.")
    stance_consistency_status: Optional[str] = Field(default=None, description="Participant-stance consistency guard status.")
    stance_consistency_reason: Optional[str] = Field(default=None, description="Participant-stance consistency guard rationale.")
    stance_consistency_alternative_candidate_id: Optional[str] = Field(default=None, description="Candidate selected as the stance-safe alternative, if any.")

    selected_point: Optional[str] = Field(default=None, description="The single, final selected point for Concise scope")
    selection_reason: Optional[str] = Field(default=None, description="Brief reason why selected_point beats others")

    class ResponseShape(BaseModel):
        sentences: Optional[int] = Field(default=None, description="Desired sentence count for the utterance (1-3)")
        speech_act: Optional[str] = Field(default=None, description="'assert' or 'question'")
        link_to_context: Optional[bool] = Field(default=None, description="If true, begin with a subtle context-anchoring clause")

    response_shape: Optional[ResponseShape] = Field(default=None, description="Optional guidance for the Generator to shape the utterance.")


class ControllerLLMOutput(BaseModel):
    """Reduced schema for Controller LLM structured output.
    
    Contains ONLY the fields that the prompt asks the LLM to output:
    - action_details (required)
    - content_plan (nullable)
    - active_candidate_id (nullable)
    - candidate_decision_trace (string)
    
    Other fields in ControllerOutput (helper_candidates_digest, speaker_recommendation,
    selected_point, selection_reason, etc.) are populated programmatically AFTER the LLM returns.
    """
    action_details: ActionDetails = Field(description="The final chosen action details")
    content_plan: Optional[ContentPlan] = Field(default=None, description="High-level instruction for the Generator (null if silent)")
    active_candidate_id: Optional[str] = Field(default=None, description="The candidate_id selected (or null)")
    candidate_decision_trace: Optional[str] = Field(default=None, description="1-3 sentences explaining helper integration")


class MonoRawLLMOutput(BaseModel):
    """Structured output for the mono-raw delegate."""
    action_details: ActionDetails = Field(description="The final chosen action details")
    content_plan: Optional[ContentPlan] = Field(default=None, description="High-level instruction for the utterance (null if silent)")
    active_candidate_id: Optional[str] = Field(default=None, description="The candidate_id selected (or null)")
    candidate_decision_trace: Optional[str] = Field(default=None, description="1-3 sentences explaining why this candidate or silence was chosen")
    response: Optional[str] = Field(default=None, description="The utterance to say if speaking")


class MonoRawReflectionCriticOutput(BaseModel):
    """Structured output for the transcript-only reflection critic."""
    verdict: str = Field(description="'update', 'no_update', or 'curate_memory'")
    feedback_type: Optional[str] = Field(default=None, description="Coarse failure or maintenance type, e.g. missed_opportunity, redundancy, unsupported_agreement, curation")
    observed_turn_type: Optional[str] = Field(default=None, description="substantive, phatic, backchannel, fragment, disfluent, or ambiguous")
    evidence_quality: Optional[str] = Field(default=None, description="high, medium, low, or insufficient")
    rationale: str = Field(description="Brief rationale for the verdict")
    lesson: Optional[str] = Field(default=None, description="Single procedural lesson when verdict=update")
    updated_memory: List[str] = Field(default_factory=list, description="Full revised reflection memory after applying the verdict")
    drop_reason: Optional[str] = Field(default=None, description="Reason no update was retained, if applicable")


class GeneratorOutput(BaseModel):
    utterance: str = Field(description="The actual response to be delivered.")


class DelegateResponse(BaseModel):
    """Final output from the delegate agent."""
    thoughts: str = Field(description="Brief reasoning for the decision")
    response: Optional[str] = Field(default=None, description="The response if speaking")


# ==============================================================================
#  Shared Memory Module Schemas
# ==============================================================================

class DelegateActionLogEntry(BaseModel):
    """A record of a single action taken by our delegate agent."""
    utterance_id_of_action: str = Field(description="The cue utterance ID that the agent acted upon.")
    action_taken: str = Field(description="The high-level action from the Action Space (e.g., 'Chime_In').")
    point_made_summary: Optional[str] = Field(default=None, description="A brief summary of the 'speak' content or the content itself, if any.")


# ==============================================================================
#  Single-Tier Agent Schemas
# ==============================================================================

class SingleTierDelegateResponse(BaseModel):
    """Response schema for single-tier (monolithic) delegate agent."""
    Reasoning: str = Field(description="The comprehensive reasoning behind the response.")
    Decision: str = Field(description="'Speak' or 'Stay Silent'")
    Response: Optional[str] = Field(default=None, description="The response content if speaking.")


# ==============================================================================
#  JIT Briefing Schemas
# ==============================================================================

# class JITBriefingPoint(BaseModel):
#     """Individual briefing point for just-in-time briefings."""
#     point: str = Field(description="The briefing point content.")
#     context: str = Field(description="Context for when this point is relevant.")
#     priority: str = Field(description="Priority level: 'High', 'Medium', 'Low'.")


class JITBriefingPointEvidence(BaseModel):
    """Optional, per-point grounding artifacts for JIT briefings (downstream-safe)."""
    text: str = Field(description="Exact bullet text as emitted in jit_briefing_points.")
    support_utt_ids: List[str] = Field(description="Participant utterance IDs in the testing segment supporting this bullet.")
    intent_type: Optional[str] = Field(default=None, description="One of: question|proposal|decision|metric|stance|risk|request|clarification.")
    topical_tags: Optional[List[str]] = Field(default=None, description="Short topical tags.")
    anchor_phrases: Optional[List[str]] = Field(default=None, description="1–2 short phrases (2–4 words) quoted from the participant's lines.")
    framing: Optional[str] = Field(default=None, description="Contribution framing: one of introduces|builds_on|supports|questions|counters|summarizes.")
    context_utt_ids: Optional[List[str]] = Field(default=None, description="Optional non-participant utterance IDs that establish the idea being built on/refined.")


class JITBriefingResponse(BaseModel):
    """Response from JIT briefing generation."""
    jit_briefing_points: List[str] = Field(description="Generated briefing points for the session.")
    thoughts: str = Field(description="Reasoning behind the generated briefing points.")
    # Optional enrichment (preferred but not required; downstream-safe)
    points_evidence: Optional[List[JITBriefingPointEvidence]] = Field(default=None, description="Optional grounding evidence per bullet.")


# ==============================================================================
#  CAPA Episode Schemas (Unified per-user episodes)
# ==============================================================================

class CAPAEpisode(BaseModel):
    """Unified CAPA episode used for both environment and delegate judging."""
    meeting_id: str = Field(description="Identifier for the meeting.")
    represented_user: str = Field(description="Participant id/role this episode is for.")
    cue_utt_id: str = Field(description="Utterance id at cue t (last item in context_last_N).")
    cue_index: Optional[int] = Field(default=None, description="Optional numeric index for cue t in the transcript.")
    N: int = Field(description="History window size (context_last_N length).")
    k: int = Field(description="Future peek window size (gt_next_window length).")
    context_last_N: List[Dict[str, Any]] = Field(description="Utterances [t−N..t] used for perception and decision.")
    gt_next_window: List[Dict[str, Any]] = Field(description="Ground-truth utterances [t+1..t+k] used for judging.")
    test_case_id: Optional[str] = Field(default=None, description="Optional id for traceability.")
    cue_test_case_id: Optional[str] = Field(default=None, description="Cue-anchor id without selector disambiguation, useful for human-readable diagnostics.")
    episode_id_suffix: Optional[str] = Field(default=None, description="Stable selector-derived suffix that disambiguates multiple episodes sharing the same cue.")
    # Optional selection metadata (when episodes are chosen via LLM selector)
    selection_reason: Optional[str] = Field(default=None, description="Why this cue was selected by the selector.")
    selection_tags: Optional[List[str]] = Field(default=None, description="Tags describing the opportunity type (explicit, decision, open_floor, topic_shift, conflict, implicit).")
    selection_confidence: Optional[float] = Field(default=None, description="Selector-reported confidence 0..1 (if provided).")
    selection_min_gap: Optional[int] = Field(default=None, description="Minimum spacing (utterances) enforced between selected cues for this participant.")
    # Idea Units (Idea-Units regime; optional for legacy compatibility)
    idea_units: Optional[List[IdeaUnit]] = Field(default=None, description="Episode-local Idea Units whose anchors fall in the k-window.")
    # Frozen-anchor inventory metadata (optional; used for deterministic k-window rebuilds)
    rebuilt_from_anchor_inventory: Optional[bool] = Field(default=None, description="Whether this episode was rebuilt from a frozen anchor inventory.")
    anchor_inventory_id: Optional[str] = Field(default=None, description="Stable anchor inventory identifier, typically keyed by participant and anchor utterance id.")
    anchor_inventory_reference_k: Optional[int] = Field(default=None, description="Reference k used when freezing the anchor inventory.")
    preferred_cue_utt_id: Optional[str] = Field(default=None, description="Cue utterance id preferred by the frozen anchor inventory before k-specific rebuilding.")
    preferred_cue_index: Optional[int] = Field(default=None, description="Transcript index of the preferred cue from the frozen anchor inventory.")
    preferred_cue_type: Optional[str] = Field(default=None, description="Preferred cue type from the frozen anchor inventory.")
    preferred_cue_gap: Optional[int] = Field(default=None, description="Gap between the preferred cue and anchor utterance in transcript turns.")
    preferred_cue_retained: Optional[bool] = Field(default=None, description="Whether the preferred cue was retained for this target k.")
    self_initiation_fallback: Optional[bool] = Field(default=None, description="Whether this episode used the deterministic self-initiation fallback cue.")
    cue_mode: Optional[str] = Field(default=None, description="Normalization outcome for the selected cue: retained | selector_reanchor | deterministic_reanchor | self_initiation.")
    effective_cue_gap: Optional[int] = Field(default=None, description="Gap between the effective cue used for simulation and the anchor utterance in transcript turns.")


# ==============================================================================
#  CAPA Artifact Tracking Schemas
# ==============================================================================

# Import CAPA types
try:
    from capa.types import PredictionResponse, ErrorReport, DelegateErrorReport
except ImportError:
    # Define minimal stubs if CAPA not available
    class PredictionResponse(BaseModel):
        pass
    class ErrorReport(BaseModel):
        pass  
    class DelegateErrorReport(BaseModel):
        pass


class CapaStepArtifacts(BaseModel):
    """CAPA artifacts for a single step with proper identification"""
    
    # Identification (matches simulation structure exactly)
    test_case_id: str = Field(description="Episode identifier (e.g., 'Meet_UX_t31')")
    delta: int = Field(description="Step number within episode (0, 1, 2)")
    anchor_utt_id: str = Field(description="Anchor utterance ID (e.g., 'Meet_033')")
    participant_id: str = Field(description="Participant being simulated (UX, DEV, PM)")
    meeting_id: str = Field(description="Meeting identifier (Q4_roadmap_meeting)")
    
    # Core artifacts
    controller_output: Optional[ControllerOutput] = Field(default=None)
    predictor_output: Optional[PredictionResponse] = Field(default=None)
    environment_judge_output: Optional[ErrorReport] = Field(default=None)
    delegate_judge_output: Optional[DelegateErrorReport] = Field(default=None)
    recalibrator_output: Optional[EIMMasterState] = Field(default=None)
    
    # Meta-info
    patch_applied: bool = Field(default=False)
    capa_step_successful: bool = Field(default=False)


class CapaSimulationArtifacts(BaseModel):
    """Complete artifact collection for a simulation run"""
    
    # Simulation metadata (matches current structure)
    participant_id: str
    meeting_id: str
    total_episodes: int
    total_steps: int = Field(description="Total steps across all episodes")
    
    # Artifacts organized for easy lookup
    artifacts: List[CapaStepArtifacts] = Field(description="All step artifacts in execution order")
    
    # Lookup indices for fast access
    artifacts_by_episode: Dict[str, List[CapaStepArtifacts]] = Field(description="Artifacts grouped by test_case_id")
    artifacts_by_anchor: Dict[str, CapaStepArtifacts] = Field(description="Artifacts keyed by anchor_utt_id")


# Export commonly used exceptions
__all__ = [
    'ValidationError',
    'UserProfile', 'RichProfileGenerationResponse',
    'EIMMeetingStateVector', 'EIMMasterState',
    'TestCaseInstance', 'TestCaseExtractionResponse', 
    'DelegateResponse', 'SingleTierDelegateResponse',
    'ControllerOutput', 'ControllerLLMOutput', 'MonoRawLLMOutput', 'GeneratorOutput',
    'DelegateActionLogEntry',
    'JITBriefingResponse',
    'CAPAEpisode', 'IdeaUnit',
    'CapaStepArtifacts', 'CapaSimulationArtifacts'
]
