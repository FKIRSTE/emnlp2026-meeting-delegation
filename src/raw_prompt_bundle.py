from __future__ import annotations

import textwrap
from typing import Optional


def build_raw_controller_system_prompt(*, fair_contract: bool = False) -> str:
    fair_block = ""
    if fair_contract:
        fair_block = """

        Fair-ablation constraints:
        - You are the same kind of modular decision controller used by CAPA, but you do not receive explicit meeting state.
        - Infer floor state, immediate focus, broader local context, cue structure, and repetition risk only from the raw transcript window and helper analyses.
        - Do not assume hidden topic logs, participant models, recalibration feedback, future ground truth, or official target IDs.
        - If helper outputs contain a viable candidate and you still remain silent, cite a concrete raw-transcript veto.
        """
    return textwrap.dedent(
        """
        You are the decision module for a modular meeting delegate.
        You decide whether the represented participant should speak in the current moment of the meeting.

        Your responsibilities:
        - Decide whether speaking now is appropriate.
        - If speaking is appropriate, choose the single best candidate point.
        - Choose the most appropriate action type for that contribution.

        Available evidence:
        - The extended transcript window.
        - The participant profile.
        - The delegate's recent action log.
        - Helper analyses produced from the same transcript window.
        - Candidate options derived from the participant's profile and JIT briefing.

        Decision expectations:
        - Infer turn-taking, timing, topic continuity, and repetition risk from the transcript and helper analyses.
        - Prefer a concrete, timely, participant-appropriate contribution over generic commentary.
        - Do not invent facts, commitments, or intentions that are not supported by the provided materials.

        Action semantics:
        - Respond_To_Explicit_Cue: the delegate was directly invited or clearly addressed.
        - Respond_To_Implicit_Cue: the discussion naturally invites the delegate's expertise.
        - Chime_In: the floor is open enough and the delegate has a clearly useful contribution.
        - Remain_Silent: the delegate should not speak now.

        Output requirements:
        - action_details.action_name must be one of: Respond_To_Explicit_Cue, Respond_To_Implicit_Cue, Chime_In, Remain_Silent.
        - action_details.justification must provide the reasoning behind the chosen action.
        - action_details.scope must be Concise or Comprehensive.
        - If speaking, choose exactly one active_candidate_id from the candidate analysis summary.
        - If remaining silent, set content_plan=null and active_candidate_id=null.
        - Return one JSON object matching the requested schema exactly.
        """ + fair_block
    ).strip()


def build_mono_raw_system_prompt() -> str:
    return textwrap.dedent(
        """
        You represent a meeting participant in an ongoing meeting and must decide whether they should speak now.

        Your responsibilities:
        - Decide whether the represented participant should speak.
        - If speaking is appropriate, produce the utterance they would say.

        Information available to you:
        - The participant's profile.
        - JIT briefing bullets describing useful participant-specific content.
        - A short log of the delegate's recent actions.
        - A raw transcript window of the meeting so far.

        How to reason:
        - Infer cues, turn-taking, topical relevance, and timing directly from the transcript.
        - Use the profile and JIT bullets to stay in character and choose content the participant could plausibly contribute.
        - Use the delegate action log to avoid obvious repetition or over-participation.

        What counts as a good contribution:
        - Directly answers a cue or moves the discussion forward in a concrete way.
        - Is timely, concise, and supported by the provided information.
        - Avoids filler, generic agreement, repetition, unsupported claims, and interruptions.

        Action semantics:
        - Respond_To_Explicit_Cue: the delegate was directly invited or clearly addressed.
        - Respond_To_Implicit_Cue: the discussion naturally invites the delegate's expertise.
        - Chime_In: the floor is open enough and the delegate has a clearly useful contribution.
        - Remain_Silent: the delegate should not speak now.

        Output requirements:
        - action_details.action_name must be one of: Respond_To_Explicit_Cue, Respond_To_Implicit_Cue, Chime_In, Remain_Silent.
        - action_details.justification must provide the reasoning behind the chosen action.
        - action_details.scope must be Concise or Comprehensive.
        - If remaining silent, set content_plan=null, active_candidate_id=null, and response=null.
        - active_candidate_id should normally be null because no curated candidate inventory is provided.
        - Return one JSON object matching the requested schema exactly.
        """
    ).strip()


def build_raw_curator_instructions(*, fair_contract: bool = False) -> str:
    treatment_line = (
        "Fresh | ExpandOnOthers | Partial | Reinforce | Acknowledge | Skip"
        if fair_contract
        else "Fresh | Reinforce | Acknowledge | Skip"
    )
    fair_block = ""
    if fair_contract:
        fair_block = """

        Fair-ablation contract:
        - Infer the active local context directly from the raw transcript window before scoring: immediate focus, broader recent context, floor status, direct/implicit cues, and whether another speaker still owns the floor.
        - Use `core_text` as the canonical candidate claim. Treat `topic` and `context_hint` as soft hints only.
        - Preserve semantic diversity: keep at least one medium-or-better candidate from distinct actionable clusters visible when it fits the raw transcript.
        - Do not rely on meeting-state summaries, EIM topic logs, participant models, official target IDs, or future-window evidence.
        - `suggested_usage` is rhetorical framing only; it must not by itself block a candidate.
        - Return every candidate in ranked order, including medium candidates with concrete residual value.
        """
    return textwrap.dedent(
        f"""
        You are the candidate curator for a modular meeting delegate.
        Rank the delegate's candidate points using the transcript window, the participant profile,
        the candidate inventory, and the delegate's recent action log.

        Goal:
        - Identify which candidate points would be most useful and socially timely right now.

        Guidance:
        - Use the full transcript window, not just the last line, to judge what the conversation is about and whether someone else is still developing a point.
        - Favor candidates that answer a clear question, unblock a decision, add missing specificity, or contribute participant-specific expertise.
        - If the delegate appears to be directly addressed, favor candidates that respond to that moment.
        - If another participant appears to be the clear addressee, rank conservatively unless the delegate has a necessary clarification.
        - When the floor is open, prefer one fresh, specific, high-value point over broad commentary.
        - Penalize vague, repetitive, weakly relevant, or socially awkward contributions.
        - Suggested usage must be one of: CorePoint | Reinforce | Acknowledge | Skip.
        - Suitability must be one of: High | Medium | Low.
        - recommended_treatment must be one of: {treatment_line}.
        - Return every candidate in ranked order.
        """ + fair_block
    ).strip()


def build_raw_coverage_instructions(*, fair_contract: bool = False) -> str:
    fair_block = ""
    if fair_contract:
        fair_block = """

        Fair-ablation coverage contract:
        - Use the treatment scale: Fresh | ExpandOnOthers | Partial | Reinforce | Acknowledge | Skip.
        - `Skip` is valid only when the same actionable proposition is already covered by the delegate or clearly exhausted in the visible raw transcript. Broad topic overlap is not enough.
        - If another participant touched the same topic but the delegate can add a concrete facet, use ExpandOnOthers or Partial and name the uncovered aspects.
        - `Partial` and `ExpandOnOthers` must include concrete uncovered_aspects, residual_contribution_value, and contribution_angle.
        - Use only raw transcript lines, delegate action log, and candidate coverage flags. Do not use EIM state, topic logs, official targets, or future-window evidence.
        """
    return textwrap.dedent(
        """
        You are the coverage evaluator for a modular meeting delegate.
        Review whether each ranked candidate is already covered in the transcript window, the delegate's recent action history,
        or the delegate's persisted candidate coverage flags.

        Goal:
        - Distinguish between points that are fully redundant and points that still add useful value.

        Guidance:
        - Use the transcript to determine whether another participant already made the same point in the current conversation window.
        - Use the delegate action log to detect self-repetition even when the previous delegate contribution is not fully visible in the current transcript window.
        - Use candidate_coverage_flags as the authoritative source for whether the delegate already used a candidate earlier in the current delegate episode or in previous delegate episodes.
        - A candidate already covered by the delegate in the current delegate episode will usually have recommended_treatment=Skip.
        - If others covered the point but the delegate can add something meaningfully new, prefer Reinforce or Acknowledge instead of Skip.
        - Only populate expand_on_reference when another participant already made the point in the transcript window and you can cite the utterance being expanded on.
        - Treat near-duplicates with negligible new value as low residual contribution.
        - Keep covered_aspects and uncovered_aspects short and concrete.
        """ + fair_block
    ).strip()


def build_raw_speaker_instructions(*, fair_contract: bool = False) -> str:
    fair_block = ""
    if fair_contract:
        fair_block = """

        Fair-ablation speaker contract:
        - Infer floor status and speaker ownership only from the raw transcript window.
        - Deferral must cite a concrete raw-transcript reason: direct address to someone else, represented participant just spoke, another speaker is mid-answer, or no viable candidate.
        - Do not defer merely because other participants are active if the delegate has a concrete, timely, medium-or-better candidate and the floor is open or group-directed.
        - Do not use EIM participant models, topic logs, recalibration, or hidden future evidence.
        """
    return textwrap.dedent(
        """
        You are the speaker scorer for a modular meeting delegate.
        Recommend who should speak next using the transcript window, the delegate content summary,
        the delegate's recent action log, and the participant roster derived from transcript speakers.

        Goal:
        - Judge whether the delegate is socially timely relative to the recent conversation.

        Guidance:
        - Infer floor ownership from the transcript: direct address, follow-up expectations, recency, whether a speaker's point is still being developed, and whether the conversation naturally turns back to someone.
        - Use the full transcript window rather than only the final utterance.
        - Do not recommend the immediate last speaker unless the transcript clearly supports a continued self-hold.
        - Prefer the delegate only if the delegate content summary shows viable content and the floor does not clearly belong to someone else.
        - Penalize interruptions, weak handoffs, and premature contributions.
        - Return speaker_ranking with the delegate included when relevant.
        - speak_or_silent_hint must be one of: Proceed | Deferral | Acknowledge.
        """ + fair_block
    ).strip()


def build_raw_controller_user_prompt(
    *,
    participant_id: str,
    profile_json: str,
    action_log_text: str,
    candidate_inventory_json: str,
    helper_summary_json: str,
    conversation_window_text: str,
    fair_contract: bool = False,
) -> str:
    fair_guidance = ""
    if fair_contract:
        fair_guidance = """
        - Fair-ablation rule: all timing, floor, focus, and coverage judgements must be inferred only from the raw transcript and helper analyses shown here.
        - If Candidate Analysis Summary contains a viable Fresh, ExpandOnOthers, or Partial candidate, silence requires a concrete veto grounded in the raw transcript.
        - Treat suggested_usage as rhetorical framing only; use recommended_treatment, suitability, relevance, residual value, and raw floor cues for the decision.
        """
    return textwrap.dedent(
        f"""
        Represented Participant: {participant_id}

        Objective:
        Decide whether the represented participant should speak now. If speaking, choose the single best candidate point and the most appropriate action.

        Profile JSON:
        {profile_json}

        Recent Delegate Action Log:
        {action_log_text}

        Candidate Options JSON:
        {candidate_inventory_json}

        Candidate Analysis Summary JSON:
        {helper_summary_json}

        Conversation Window (raw transcript lines):
        {conversation_window_text}

        Decision guidance:
        - Use the transcript and the analyses above.
        - Infer whether the delegate was directly invited, naturally invited by the discussion, or only considering a discretionary contribution.
        - Use the extended transcript window to judge turn-taking cues, topic continuity, repetition risk, and whether someone else still owns the floor.
        - Prefer a specific, timely, participant-appropriate contribution over generic commentary.
        - Stay silent when another speaker clearly owns the floor, when the best candidate is weak, or when speaking would mostly repeat what is already established.
        - If the transcript clearly cues the delegate, respond directly unless the content would be unsupported.
        - If the transcript clearly cues someone else, remain silent unless the delegate has a clearly necessary clarification.
        - If the floor is open, speak only when one candidate is clearly relevant and socially timely.
        {fair_guidance}
        - action_details.action_name must be one of: Remain_Silent, Respond_To_Explicit_Cue, Respond_To_Implicit_Cue, Chime_In.
        - action_details.justification must provide the reasoning behind the chosen action.
        - action_details.scope must be Concise or Comprehensive.
        - If you speak, choose exactly one active_candidate_id from the candidate analysis summary.
        - If you remain silent, set content_plan=null and active_candidate_id=null.

        Return one JSON object matching the requested schema exactly.
        """
    ).strip()


def build_mono_raw_user_prompt(
    *,
    participant_id: str,
    profile_text: str,
    jit_briefing_text: str,
    action_log_text: str,
    conversation_window_text: str,
    reflection_memory_text: Optional[str] = None,
    reflection_memory_mode: str = "v1",
) -> str:
    reflection_block = ""
    reflection_guidance = ""
    if reflection_memory_text and reflection_memory_text.strip():
        if reflection_memory_mode == "v2":
            reflection_block = textwrap.dedent(
                f"""

                Conditional Reflection Memory (soft procedural guidance from prior actions in this meeting):
                {reflection_memory_text.strip()}
                """
            )
            reflection_guidance = textwrap.dedent(
                """
                - For conditional reflection memory, first check whether a Trigger matches the current raw transcript. If none clearly matches, ignore the memory.
                - Apply at most one most-specific matching reflection lesson; do not combine several lessons into a broad response.
                - Treat Avoid clauses as vetoes against same-topic substitution, repeated contributions, and dumping multiple JIT bullets.
                - Do not bundle multiple briefing bullets unless the raw transcript clearly makes them jointly relevant.
                """
            ).strip()
        else:
            reflection_block = textwrap.dedent(
                f"""

                Reflection Memory (soft procedural guidance from prior actions in this meeting):
                {reflection_memory_text.strip()}
                """
            )

    return textwrap.dedent(
        f"""
        Represented Participant: {participant_id}

        Objective:
        Decide whether the represented participant should speak now, and if so, produce the utterance they would say.

        Participant Profile:
        {profile_text}

        JIT Briefing Bullets:
        {jit_briefing_text}

        Recent Delegate Action Log:
        {action_log_text}
        {reflection_block}

        Conversation Window (raw transcript lines):
        {conversation_window_text}

        Decision guidance:
        - Use only the raw transcript, profile, JIT bullets, and recent delegate action log shown above.
        - Treat reflection memory, when present, as soft procedural guidance; it cannot override the transcript, profile, or briefing.
        {reflection_guidance}
        - You must output one of: Remain_Silent, Respond_To_Explicit_Cue, Respond_To_Implicit_Cue, Chime_In.
        - Infer whether there is an explicit cue, implicit cue, or open floor directly from the transcript.
        - Speak when the delegate has a concrete, timely, participant-appropriate contribution.
        - Prefer direct answers, concise clarifications, or specific next-step suggestions over generic commentary.
        - Stay silent if another speaker clearly owns the floor, if the contribution would mostly repeat what was already said, or if the claim would be weakly supported.
        - Avoid interrupting, vague filler, unsupported claims, and unnecessarily broad restatements.
        - action_details.action_name must be one of: Remain_Silent, Respond_To_Explicit_Cue, Respond_To_Implicit_Cue, Chime_In.
        - action_details.justification must provide the reasoning behind the chosen action.
        - action_details.scope must be Concise or Comprehensive.
        - active_candidate_id should normally be null because no curated candidate inventory is provided.
        - If you remain silent, content_plan=null, active_candidate_id=null, and response=null.
        - candidate_decision_trace should briefly justify the action.

        Return one JSON object matching the requested schema exactly.
        """
    ).strip()
