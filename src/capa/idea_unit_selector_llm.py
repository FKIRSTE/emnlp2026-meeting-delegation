"""
Idea-Unit CAPA Selector (LLM)

Single-pass prompt to:
  1) identify Idea Units (one coherent idea) by the represented participant,
  2) rank them by importance, and
  3) link each to a triggering cue.

Output JSON matches the schema described in idea_units_episodes_and_evals.md.
This module is self-contained to keep legacy contribution selector intact.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json

from llm_client import LLMClient

# Import structured output schemas
try:
    from ..schemas import IdeaUnitSelectorResponse
except ImportError:
    from schemas import IdeaUnitSelectorResponse


@dataclass
class TableRow:
    index: int
    utterance_id: str
    speaker_id: str
    text_trim: str


def _render_system_prompt() -> str:
    return (
        "You are an NLP expert agent specializing in meeting conversation analysis.\n"
        "Goal: extract IDEA UNITS for ONE represented participant from a trimmed transcript table and emit STRICT JSON.\n"
        "\n"
        "Clarifications (must follow all):\n"
        "- IDEA UNIT = exactly one coherent idea: proposal, decision, key fact/metric, stance/opinion, non-trivial question/request...\n"
        "- Size: usually 1-2 utterances. Split whenever the overall idea changes, even if topic words repeat.\n"
        "- Speaker constraint: every response_utterance_id MUST be by the represented participant. Non-contiguous is allowed if the turns are close in idx AND clearly the SAME idea (e.g., answer + short clarification).\n"
        "- Anchor choice: pick the EARLIEST substantive idea-bearing response as the anchor (first_response_idx and first response_utterance_id). Drop weak preambles if a later represented-user utterance is the first substantive start of the idea.\n"
        "- Banality: exclude pure phatic/backchannel turns (ok, okay, yeah, yup, thanks, right, got it, sure, mm-hmm, uh-huh, {{vocalsound}}, {{disfmarker}}) unless inseparable from the idea.\n"
        "- Do not fill a quota. It is valid to return fewer than the budget, or zero, when the participant has few genuinely useful idea units.\n"
        "- topic_summary must be directly entailed by the selected response utterances themselves; do not make a weak acknowledgment sound like a substantive contribution.\n"
        "- Drop thin repetitions where the participant already made the same substantive point earlier and the future turn only acknowledges, restates, or weakly elaborates it.\n"
        "- cue_* fields describe the ACTUAL fixed-k episode cue that downstream simulation will use. It MUST satisfy cue_idx < first_response_idx and fit within k.\n"
        "- semantic_cue_* fields are optional and describe the discourse-origin cue when the actual cue is a closer local re-anchor.\n"
        "- Cue proximity (CRITICAL): when multiple cues are plausible, pick the CLOSEST preceding cue. STRONGLY prefer cues where gap = (first_response_idx - cue_idx) is at most k (typically 5).\n"
        "- Window integrity: every official response_utterance_id you emit must fall within the next k turns after the actual cue (idx in (cue_idx, cue_idx+k]). If later clarifications fall outside this fixed window, omit them from response_utterance_ids rather than expanding k.\n"
        "- importance_score must be COMPARATIVE across this participant's candidates in the full testing segment, not local salience alone.\n"
        "- Higher importance_score: decisions, concrete proposals, blockers/risks, quantified facts/constraints, commitments/ownership, or reframings with clear downstream consequences.\n"
        "- Lower importance_score: repetitions, weak preferences, minor clarifications, or local elaborations with little effect on subsequent discussion or decisions.\n"
        "- Use the full 0..1 range when justified; avoid bunching most items into the same narrow score band.\n"
        "- Ordering: use the idx column only. Do NOT invent content. Output strict JSON; no prose."
    )


def _render_user_prompt(
    meeting_id: str,
    participant_id: str,
    role_summary: str,
    table_rows: List[TableRow],
    N: int,
    k: int,
    max_items: Optional[int] = None,
) -> str:
    header = "idx | utterance_id | speaker | text_trim"
    lines = [header]
    for r in table_rows:
        lines.append(f"{r.index} | {r.utterance_id} | {r.speaker_id} | {r.text_trim}")
    selection_budget = (
        f"- Selection budget: return AT MOST {max_items} highest-yield Idea Units for this participant. This is a maximum, not a quota; return fewer when fewer genuinely high-yield units exist.\n\n"
        if max_items is not None
        else ""
    )
    cap_instruction = (
        f"Return AT MOST {max_items} Idea Units total. If more than {max_items} plausible candidates exist, keep only the {max_items} strongest by meeting impact. Do not include filler items to reach {max_items}.\n\n"
        if max_items is not None
        else ""
    )
    cap_self_check = (
        f"- Ensure the final JSON contains at most {max_items} Idea Units, sorted by importance_score descending.\n\n"
        if max_items is not None
        else ""
    )

    schema_block = (
        "{\n"
        '  "idea_units": [\n'
        "    {\n"
        '      "unit_id": "U1",\n'
        '      "importance_score": 0.0,\n'
        '      "topic_summary": "<1-2 line accurate and precide idea summary, reflecting properly what the constituent utt-ids represented>",\n'
        '      "response_utterance_ids": ["UTT_ID_1","UTT_ID_2"],\n'
        '      "first_response_idx": 0,\n'
        '      "cue_utterance_id": "UTT_ID_CUE",\n'
        '      "cue_idx": 0,\n'
        '      "cue_type": "explicit|implicit-handoff|open_floor|self-initiation",\n'
        '      "semantic_cue_utterance_id": "UTT_ID_ORIGIN",\n'
        '      "semantic_cue_idx": 0,\n'
        '      "semantic_cue_type": "explicit|implicit-handoff|open_floor|self-initiation",\n'
        '      "anchor_reason": "Why this cue triggered the idea",\n'
        '      "confidence": 0.0\n'
        "    }\n"
        "  ],\n"
        '  "coverage_summary": {"explicit":0, "implicit-handoff":0, "open_floor":0, "self-initiation":0}\n'
        "}"
    )

    return (
        f"## Context\n"
        f"- Meeting: {meeting_id}\n"
        f"- Participant (represented user): {participant_id}\n"
        f"- Role summary (1-2 lines): {role_summary}\n"
        f"- Window parameters (downstream only): N={N}, k={k}\n\n"
        f"{selection_budget}"
        "## Definitions\n"
        "- Idea Unit: one coherent idea (question, proposal, decision, key fact/metric, stance). Keep it small (often 1-2 utterances). All utterances must be by the represented user, and the idea must be visible in those utterances rather than inferred from surrounding context alone.\n"
        "- Banality lexicon (exclude unless inseparable): ok, okay, yeah, yup, thanks, right, got it, sure, mm-hmm, uh-huh, {vocalsound}, {disfmarker}, ...\n"
        "- Cue types: explicit, implicit-handoff, open_floor, self-initiation.\n"
        "- Non-contiguous allowed: you may pick separated turns by the same speaker if close and clearly the SAME idea (e.g., answer + brief clarification).\n"
        "- Anchor policy: set first_response_idx (and make it the first response_utterance_id) to the EARLIEST substantive utterance that starts the idea; later clarifications can remain in response_utterance_ids if they are the same idea.\n"
        "- Importance rubric: score higher when the idea changes decisions, proposes a concrete next step, introduces a blocker/risk, adds quantified evidence, imposes a constraint, or clearly redirects the discussion. Score lower for minor repetition or low-consequence elaboration.\n"
        "- Splitting rule: if two related but standalone ideas appear (each makes sense alone), split into separate Idea Units instead of bundling them.\n\n"
        "## Task\n"
        f"Identify {'up to ' + str(max_items) + ' highest-yield ' if max_items is not None else 'every '}Idea Unit{'s' if max_items is not None else ''} by {participant_id} in the testing segment table. For each Idea Unit:\n"
        "1) response_utterance_ids: only the represented user's utterances that express this one idea (may be non-contiguous but close).\n"
        "2) first_response_idx: idx of the EARLIEST substantive response_utterance_id (anchor). This must be the first element of response_utterance_ids.\n"
        "3) cue_utterance_id with cue_idx and cue_type:\n"
        "   - Must satisfy cue_idx < first_response_idx.\n"
        "   - Prefer explicit > implicit-handoff > open_floor > self-initiation.\n"
        "   - cue_* MUST be the ACTUAL fixed-k episode cue that fits within k of the earliest substantive response.\n"
        "   - If the discourse-origin cue is farther away, keep that origin in semantic_cue_* and emit the NEAREST plausible local cue in cue_*.\n"
        "   - If no plausible local cue within k exists, use the immediate previous utterance as cue and set cue_type=\"self-initiation\".\n"
        "4) semantic_cue_*: optional. Only include when the discourse-origin cue is different from cue_*.\n"
        "5) topic_summary, importance_score (0..1), anchor_reason, confidence.\n\n"
        f"{cap_instruction}"
        "## Hard Constraints (must pass all; otherwise OMIT the Idea Unit)\n"
        f"- response_utterance_ids MUST be utterances by {participant_id} only. Verify speaker_id in the table for each ID.\n"
        "- first_response_idx MUST equal the idx of the EARLIEST substantive response_utterance_id you keep as anchor. Make it the first element of response_utterance_ids; order the rest idx-ascending and dedup.\n"
        "- Exclude banal/phatic responses unless inseparable from the idea.\n"
        "- Exclude fragment-only, thin-acknowledgment, and low-yield repetition responses. If the represented user already made the substantive point earlier and the future turn only repeats or acknowledges it, omit the unit.\n"
        "- topic_summary must faithfully summarize the selected response_utterance_ids; do not upgrade a vague or fragmentary response into a stronger idea.\n"
        "- Cue integrity: cue_idx < first_response_idx must hold.\n"
        f"- Cue proximity (CRITICAL): if several cues qualify, STRONGLY prefer the closest one (smallest gap). Gaps of 1-3 are ideal; 4-5 acceptable.\n"
        f"- Window integrity: every response_utterance_id must be within the next k={k} turns after cue_* (idx in (cue_idx, cue_idx+{k}]). If the discourse-origin cue is farther away, keep it in semantic_cue_* and emit a nearer cue_*.\n"
        "- Self-initiation fallback: if no plausible local cue exists within k, pick the turn at (first_response_idx - 1) as cue_* and set cue_type=\"self-initiation\".\n\n"
        "## Internal Self-Check (do NOT output, but apply before emitting JSON)\n"
        f"- Verify every response_utterance_id speaker == {participant_id}; drop mismatches; if none remain, drop the Idea Unit.\n"
        "- Verify first_response_idx == idx of the earliest substantive response_utterance_id you chose as anchor; ensure it is the first element in response_utterance_ids.\n"
        "- If two distinct ideas would both stand alone, SPLIT into two Idea Units instead of bundling.\n"
        "- If any response_utterance_id is not within k turns after cue_*, drop that response id or re-anchor cue_* locally; do not leave official target ids out of range.\n"
        "- If multiple cues fit, choose the closest one that passes window integrity.\n"
        "- If all responses are banal/phatic, drop the Idea Unit.\n\n"
        "- When several candidates compete for limited slots, rank the ones with clearer downstream consequence above similar but lower-consequence clarifications.\n\n"
        f"{cap_self_check}"
        "## Examples\n"
        "### Example 1 (Explicit cue; single-utterance idea; phatic nearby)\n"
        "idx | utterance_id | speaker | text_trim\n"
        "40 | M31_040 | PM | UX, can you confirm latency budget?\n"
        "41 | M31_041 | UX | Our budget is 120ms p95; current build is 105ms.\n"
        "42 | M31_042 | PM | Okay, thanks.\n"
        "Output JSON (partial):\n"
        "{ \"idea_units\": [ { \"unit_id\": \"U1\", \"importance_score\": 0.9, \"topic_summary\": \"States latency budget 120ms and current 105ms\", \"response_utterance_ids\": [\"M31_041\"], \"first_response_idx\": 41, \"cue_utterance_id\": \"M31_040\", \"cue_idx\": 40, \"cue_type\": \"explicit\", \"anchor_reason\": \"Direct question to UX about latency\", \"confidence\": 0.9 } ], \"coverage_summary\": {\"explicit\":1,\"implicit-handoff\":0,\"open_floor\":0,\"self-initiation\":0} }\n\n"
        "### Example 2 (Implicit handoff; two nearby turns; banality filtered)\n"
        "idx | utterance_id | speaker | text_trim\n"
        "60 | M31_060 | PM | Need risks before sign-off.\n"
        "61 | M31_061 | QA | Load harness still flaky.\n"
        "62 | M31_062 | QA | We need two more days to stabilize.\n"
        "63 | M31_063 | PM | ok.\n"
        "Output JSON (partial):\n"
        "{ \"idea_units\": [ { \"unit_id\": \"U1\", \"importance_score\": 0.82, \"topic_summary\": \"Flags load harness flakiness and asks for 2 more days\", \"response_utterance_ids\": [\"M31_061\",\"M31_062\"], \"first_response_idx\": 61, \"cue_utterance_id\": \"M31_060\", \"cue_idx\": 60, \"cue_type\": \"implicit-handoff\", \"anchor_reason\": \"Risk request led QA to raise blockers\", \"confidence\": 0.82 } ], \"coverage_summary\": {\"explicit\":0,\"implicit-handoff\":1,\"open_floor\":0,\"self-initiation\":0} }\n\n"
        "### Example 3 (Self-initiation; non-contiguous but close)\n"
        "idx | utterance_id | speaker | text_trim\n"
        "80 | M31_080 | Design | We locked layout.\n"
        "81 | M31_081 | Marketing | Launch overlaps competitor blitz; we should shift to week 38.\n"
        "82 | M31_082 | Dev | Week 38 is ok.\n"
        "83 | M31_083 | Marketing | Last year shifting late-Q3 gave +18%.\n"
        "84 | M31_084 | Design | yeah right.\n"
        "Output JSON (partial):\n"
        "{ \"idea_units\": [ { \"unit_id\": \"U1\", \"importance_score\": 0.8, \"topic_summary\": \"Advocates shifting launch to week 38 citing past uplift\", \"response_utterance_ids\": [\"M31_081\",\"M31_083\"], \"first_response_idx\": 81, \"cue_utterance_id\": \"M31_080\", \"cue_idx\": 80, \"cue_type\": \"self-initiation\", \"anchor_reason\": \"No direct ask; marketing volunteers timing change after layout lock\", \"confidence\": 0.8 } ], \"coverage_summary\": {\"explicit\":0,\"implicit-handoff\":0,\"open_floor\":0,\"self-initiation\":1} }\n\n"
        "### Example 4 (Multiple Idea Units in one snippet)\n"
        "idx | utterance_id | speaker | text_trim\n"
        "120 | M31_120 | PM | Open floor for quick risks.\n"
        "121 | M31_121 | Backend | We need feature flags before rollout.\n"
        "122 | M31_122 | PM | yeah got it.\n"
        "123 | M31_123 | Backend | Also DB backups need rehearsal; last run failed.\n"
        "124 | M31_124 | UX | Noted.\n"
        "Output JSON (partial):\n"
        "{ \"idea_units\": [\n"
        "  { \"unit_id\": \"U1\", \"importance_score\": 0.86, \"topic_summary\": \"Requests feature flags before rollout\", \"response_utterance_ids\": [\"M31_121\"], \"first_response_idx\": 121, \"cue_utterance_id\": \"M31_120\", \"cue_idx\": 120, \"cue_type\": \"open_floor\", \"anchor_reason\": \"PM opened floor for risks; backend raised feature flags\", \"confidence\": 0.86 },\n"
        "  { \"unit_id\": \"U2\", \"importance_score\": 0.80, \"topic_summary\": \"Calls for DB backup rehearsal after failed run\", \"response_utterance_ids\": [\"M31_123\"], \"first_response_idx\": 123, \"cue_utterance_id\": \"M31_120\", \"cue_idx\": 120, \"cue_type\": \"open_floor\", \"anchor_reason\": \"Same open floor prompt elicited second idea\", \"confidence\": 0.80 }\n"
        "], \"coverage_summary\": {\"explicit\":0,\"implicit-handoff\":0,\"open_floor\":2,\"self-initiation\":0} }\n\n"
        "## Input: Testing Segment (trimmed table; idx is authoritative)\n"
        + "\n".join(lines)
        + "\n\n## Output (strict JSON only; exact shape)\n"
        + schema_block
        + "\n"
    )


def render_prompt(
    meeting_id: str,
    participant_id: str,
    role_summary: str,
    table_rows: List[TableRow],
    N: int,
    k: int,
    max_items: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Return messages = [{role, content}, ...] for the LLM call."""
    system = _render_system_prompt()
    user = _render_user_prompt(meeting_id, participant_id, role_summary, table_rows, N, k, max_items=max_items)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def select_idea_units(
    client: LLMClient,
    meeting_id: str,
    participant_id: str,
    role_summary: str,
    transcript_table: List[TableRow],
    N: int,
    k: int,
    max_items: Optional[int] = None,
) -> Dict[str, Any]:
    """Call the LLM and parse its strict-JSON response.

    Returns a dict with keys: idea_units (list) and coverage_summary (dict).
    On parse failure, returns {"idea_units": [], "coverage_summary": {}}.
    """
    messages = render_prompt(
        meeting_id=meeting_id,
        participant_id=participant_id,
        role_summary=role_summary,
        table_rows=transcript_table,
        N=N,
        k=k,
        max_items=max_items,
    )
    # Use response_model for structured output (critical for GPT-5-mini)
    # Note: GPT-5-mini (o-series) uses internal reasoning tokens that count against
    # the output budget, so we need a higher max_tokens than the expected JSON size
    resp = client.call_with_retry(
        model=client.model,
        messages=messages,
        max_tokens=8000,  # Increased for GPT-5-mini reasoning overhead
        temperature=0.0,
        json_required=True,  # Fallback for models without structured output
        response_model=IdeaUnitSelectorResponse,  # GPT-5-mini/Gemini use this
        use_strict_json=False, # Disable rigid schema constraint validation
        telemetry_label="prep.idea_unit_selector",
        telemetry_phase="prep",
    )

    # DEBUGGING
    print("===" * 20)
    print(f"IDEA UNIT SELECTOR O/P:\n{resp}")
    print("===" * 20, end="\n\n")
    
    if not resp:
        return {"idea_units": [], "coverage_summary": {}}
    
    # Handle both Pydantic model (structured output) and raw JSON (fallback)
    try:
        if hasattr(resp, 'model_dump'):
            # Pydantic model - convert to dict
            data = resp.model_dump(by_alias=True)  # Use aliases for hyphenated keys
        elif isinstance(resp, str):
            data = json.loads(resp)
        else:
            data = resp
        
        if not isinstance(data, dict):
            return {"idea_units": [], "coverage_summary": {}}
        data.setdefault("idea_units", [])
        data.setdefault("coverage_summary", {})
        return data
    except Exception as e:
        print(f"[WARN] Idea Unit Selector parse error: {e}")
        return {"idea_units": [], "coverage_summary": {}}


__all__ = ["TableRow", "select_idea_units", "render_prompt"]
