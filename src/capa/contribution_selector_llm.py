"""
Contribution-First CAPA Selector (LLM)

Single-pass prompt to:
  1) identify the represented participant's substantive contribution spans,
  2) rank them by importance, and
  3) link each span to one upstream cue.

Returns a JSON object matching the documented schema in update_test_gen.md.
This module only renders prompts and parses the LLM response; deterministic
validation and episode building live in episode_utils and the driver script.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json

from llm_client import LLMClient


@dataclass
class TableRow:
    index: int
    utterance_id: str
    speaker_id: str
    text_trim: str


def _render_system_prompt() -> str:
    return (
        "You are an NLP expert agent specializing in meeting conversation analysis. You analyze meeting transcripts to extract a represented participant's "
        "substantive contributions, rank them by meeting impact, and link each to its "
        "triggering cue. A contribution can span multiple adjacent utterances by the same "
        "participant when they advance one idea. Exclude pure acknowledgments/backchannels/phatic content. "
        "Do not fill a quota: return fewer items or an empty list when the participant lacks genuinely useful contributions. "
        "topic_summary must be entailed by the selected response utterances themselves, not inflated from surrounding context. "
        "Drop low-yield repetitions where the future turn merely acknowledges or weakly restates an idea already made earlier. "
        "Use the idx column as the ONLY source of order. cue_* must always be the actual fixed-k episode cue. "
        "If the discourse-origin cue is farther away, keep it in semantic_cue_* and emit a closer cue_* that fits within k. "
        "Every official response_utterance_id you emit must also fit inside the fixed k window after cue_*. "
        "importance_score must be comparative across this participant's candidates: decisions, concrete proposals, blockers/risks, quantified constraints, and commitments rank above repetitions or low-consequence elaborations. "
        "Return strict JSON only."
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
        f"- Selection budget: return AT MOST {max_items} highest-yield contributions for this participant. This is a maximum, not a quota; return fewer when fewer genuinely high-yield contributions exist.\n\n"
        if max_items is not None
        else ""
    )
    cap_instruction = (
        f"Return AT MOST {max_items} contributions total. If more than {max_items} plausible candidates exist, keep only the {max_items} strongest by meeting impact. Do not include filler items to reach {max_items}.\n\n"
        if max_items is not None
        else ""
    )
    cap_self_check = (
        f"- Ensure the final JSON contains at most {max_items} contributions, sorted by importance_score descending.\n"
        if max_items is not None
        else ""
    )

    schema_block = (
        "{\n"
        "  \"contributions\": [\n"
        "    {\n"
        "      \"contribution_id\": \"C1\",\n"
        "      \"importance_score\": 0.0,\n"
        "      \"topic_summary\": \"<One-line idea summary>\",\n"
        "      \"response_utterance_ids\": [\"Meeting31_219\",\"Meeting31_223\"],\n"
        "      \"first_response_idx\": 218,\n"
        "      \"cue_utterance_id\": \"Meeting31_217\",\n"
        "      \"cue_idx\": 216,\n"
        "      \"cue_type\": \"explicit|implicit-handoff|open_floor|self-initiation\",\n"
        "      \"semantic_cue_utterance_id\": \"Meeting31_210\",\n"
        "      \"semantic_cue_idx\": 209,\n"
        "      \"semantic_cue_type\": \"explicit|implicit-handoff|open_floor|self-initiation\",\n"
        "      \"anchor_reason\": \"Why this cue plausibly triggered the contribution\",\n"
        "      \"confidence\": 0.0\n"
        "    }\n"
        "  ],\n"
        "  \"coverage_summary\": {\"explicit\":0, \"implicit-handoff\":0, \"open_floor\":0, \"self-initiation\":0}\n"
        "}"
    )

    examples_block = (
        "## Examples\n"
        "### Example 1 (Explicit; represented participant responds with KBU within k)\n"
        "Context: Meeting=MeetingX, Participant=Industrial Designer, N=15, k=5\n"
        "Input snippet (trimmed table):\n"
        "idx | utterance_id | speaker | text_trim\n"
        "40 | MeetingX_040 | Project Manager | Industrial Designer, can you estimate the battery impact of always-on backlight?\n"
        "41 | MeetingX_041 | User Interface | We could only light applicable buttons.\n"
        "42 | MeetingX_042 | Industrial Designer | At 200mA LED draw, five minutes per hour yields under 2% daily drain with 2500mAh.\n"
        "43 | MeetingX_043 | Project Manager | That sounds acceptable.\n"
        "44 | MeetingX_044 | User Interface | We can tie it to ambient sensor.\n"
        "Output JSON:\n"
        "{\n"
        "  \"contributions\": [\n"
        "    {\n"
        "      \"contribution_id\": \"C1\",\n"
        "      \"importance_score\": 0.91,\n"
        "      \"topic_summary\": \"Estimated backlight impact and feasibility.\",\n"
        "      \"response_utterance_ids\": [\"MeetingX_042\"],\n"
        "      \"first_response_idx\": 42,\n"
        "      \"cue_utterance_id\": \"MeetingX_040\",\n"
        "      \"cue_idx\": 40,\n"
        "      \"cue_type\": \"explicit\",\n"
        "      \"anchor_reason\": \"PM explicitly asked; the response provided a quantified estimate.\",\n"
        "      \"confidence\": 0.9\n"
        "    }\n"
        "  ],\n"
        "  \"coverage_summary\": {\"explicit\":1, \"implicit-handoff\":0, \"open_floor\":0, \"self-initiation\":0}\n"
        "}\n\n"
        "### Example 2 (Implicit-handoff; represented participant responds with KBUs within k)\n"
        "Context: Meeting=MeetingY, Participant=User Interface, N=15, k=5\n"
        "Input snippet (trimmed table):\n"
        "idx | utterance_id | speaker | text_trim\n"
        "20 | MeetingY_020 | Project Manager | Before we lock hardware, someone should outline UI constraints for single-hand use.\n"
        "21 | MeetingY_021 | Industrial Designer | Grip width is 58-62 mm on average.\n"
        "22 | MeetingY_022 | User Interface | Single-hand reach maps suggest max thumb arc ~70 mm; primary actions should sit within that arc.\n"
        "23 | MeetingY_023 | Project Manager | Good.\n"
        "24 | MeetingY_024 | User Interface | We'll add a sticky overflow action to avoid edge stretch.\n"
        "Output JSON:\n"
        "{\n"
        "  \"contributions\": [\n"
        "    {\n"
        "      \"contribution_id\": \"C1\",\n"
        "      \"importance_score\": 0.88,\n"
        "      \"topic_summary\": \"Outlined UI constraints and mitigation for single-hand use.\",\n"
        "      \"response_utterance_ids\": [\"MeetingY_022\", \"MeetingY_024\"],\n"
        "      \"first_response_idx\": 22,\n"
        "      \"cue_utterance_id\": \"MeetingY_020\",\n"
        "      \"cue_idx\": 20,\n"
        "      \"cue_type\": \"implicit-handoff\",\n"
        "      \"anchor_reason\": \"PM raised a role-relevant need for UI constraints; UI contributed reach constraint and mitigation.\",\n"
        "      \"confidence\": 0.88\n"
        "    }\n"
        "  ],\n"
        "  \"coverage_summary\": {\"explicit\":0, \"implicit-handoff\":1, \"open_floor\":0, \"self-initiation\":0}\n"
        "}\n\n"
        "### Example 3 (Self-initiation; chime-in; cue defaults to previous)\n"
        "Context: Meeting=MeetingZ, Participant=Marketing, N=15, k=5\n"
        "Input snippet (trimmed table):\n"
        "idx | utterance_id | speaker | text_trim\n"
        "55 | MeetingZ_055 | Project Manager | Tentative launch is Q3.\n"
        "56 | MeetingZ_056 | Industrial Designer | Tooling needs 10 weeks from final CAD.\n"
        "57 | MeetingZ_057 | Marketing | Q3 overlaps a competitor promo; last year a late-Q3 shift yielded +18% — we should target week 37–38.\n"
        "58 | MeetingZ_058 | Project Manager | Let's pencil in week 37.\n"
        "59 | MeetingZ_059 | User Interface | That leaves time for polish.\n"
        "Output JSON:\n"
        "{\n"
        "  \"contributions\": [\n"
        "    {\n"
        "      \"contribution_id\": \"C1\",\n"
        "      \"importance_score\": 0.82,\n"
        "      \"topic_summary\": \"Recommended shifting launch timing based on market events.\",\n"
        "      \"response_utterance_ids\": [\"MeetingZ_057\"],\n"
        "      \"first_response_idx\": 57,\n"
        "      \"cue_utterance_id\": \"MeetingZ_056\",\n"
        "      \"cue_idx\": 56,\n"
        "      \"cue_type\": \"self-initiation\",\n"
        "      \"anchor_reason\": \"No explicit/implicit ask; participant volunteered a high-impact recommendation. Using immediate predecessor as cue.\",\n"
        "      \"confidence\": 0.82\n"
        "    }\n"
        "  ],\n"
        "  \"coverage_summary\": {\"explicit\":0, \"implicit-handoff\":0, \"open_floor\":0, \"self-initiation\":1}\n"
        "}\n\n"
        "### Example 4 (No valid contributions; only banal replies)\n"
        "Context: Meeting=MeetingN, Participant=Project Manager, N=15, k=3\n"
        "Input snippet (trimmed table):\n"
        "idx | utterance_id | speaker | text_trim\n"
        "10 | MeetingN_010 | User Interface | We're almost done.\n"
        "11 | MeetingN_011 | Project Manager | Okay.\n"
        "12 | MeetingN_012 | Marketing | Yep.\n"
        "13 | MeetingN_013 | Industrial Designer | {vocalsound}\n"
        "Output JSON:\n"
        "{\n"
        "  \"contributions\": [],\n"
        "  \"coverage_summary\": {\"explicit\":0, \"implicit-handoff\":0, \"open_floor\":0, \"self-initiation\":0}\n"
        "}\n"
    )

    return (
        f"## Context\n"
        f"- Meeting: {meeting_id}\n"
        f"- Participant (represented user): {participant_id}\n"
        f"- Role summary (1-2 lines): {role_summary}\n"
        f"- Window parameters (downstream only): N={N}, k={k}\n\n"
        f"{selection_budget}"
        "## Definitions\n"
        "- Knowledge-Bearing Unit (KBU): fact, proposal, decision, plan, quantified detail, constraint, or non-trivial question that is visible in the represented participant's selected utterances.\n"
        "- Banality: pure acknowledgments/backchannels or very short replies (≤5 tokens), e.g., ok, okay, yeah, yup, mm-hmm, uh-huh, right, got it, thanks, sure, hmm, {vocalsound}, {disfmarker}.\n"
        "- Importance rubric: score higher when the contribution changes decisions, proposes a concrete next step, introduces a blocker/risk, adds quantified evidence, imposes a constraint, or clearly redirects the discussion. Score lower for repetition or low-consequence elaboration.\n"
        "- Cue types: explicit, implicit-handoff, open_floor, self-initiation.\n\n"
        "## Task\n"
        f"Identify {'up to ' + str(max_items) + ' highest-yield ' if max_items is not None else 'every '}substantive contribution by {participant_id}. Group adjacent turns when they reprsent the same idea/point. For each contribution:\n"
        "1) response_utterance_ids (represented user only; exclude banal).\n"
        "2) first_response_idx (idx of the earliest substantive response in the span).\n"
        "3) cue_utterance_id with cue_idx and cue_type:\n"
        "   - Must satisfy cue_idx < first_response_idx.\n"
        "   - Prefer explicit > implicit-handoff > open_floor > self-initiation.\n"
        "   - cue_* MUST be the ACTUAL fixed-k episode cue used downstream.\n"
        "   - If the discourse-origin cue is farther away, keep that origin in semantic_cue_* and emit the nearest plausible local cue in cue_*.\n"
        "   - If none plausible, set cue_type=\"self-initiation\" and pick the immediate previous utterance as cue.\n"
        "4) semantic_cue_*: optional. Only include when the discourse-origin cue differs from cue_*.\n"
        "5) topic_summary, importance_score (0..1), anchor_reason, confidence.\n\n"
        f"{cap_instruction}"
        "Order the final list by importance_score (desc). If nothing meets the bar, return an empty list.\n\n"
        "## Hard Constraints (must pass all; otherwise omit the contribution)\n"
        f"- response_utterance_ids MUST be utterances by {participant_id} only. Verify speaker_id in the table for each ID. If any ID is not by the participant, exclude it; if none remain, omit the contribution.\n"
        "- first_response_idx MUST equal the idx of the earliest substantive response_utterance_id (use the idx column).\n"
        "- Order response_utterance_ids by idx ascending and remove duplicates.\n"
        "- Exclude banal/phatic responses unless they are part of a multi-turn substantive span.\n"
        "- Exclude fragment-only, thin-acknowledgment, and low-yield repetition responses. If the represented user already made the substantive point earlier and the future turn only repeats or acknowledges it, omit the contribution.\n"
        "- topic_summary must faithfully summarize the selected response_utterance_ids; do not upgrade a vague or fragmentary response into a stronger contribution.\n"
        "- Cue integrity: cue_idx < first_response_idx must hold.\n"
        "- Window integrity: every response_utterance_id must be within the next k turns after cue_* (idx in (cue_idx, cue_idx+k]).\n"
        "- If the discourse-origin cue is farther away, store it in semantic_cue_* and emit a nearer cue_* that satisfies window integrity.\n"
        "- Self-initiation: use the immediate previous line as cue_utterance_id and set cue_type=\"self-initiation\"; still satisfy window integrity.\n\n"

        "## Internal Self-Check (apply before emitting JSON; do not include in output)\n"
        f"- For each response_utterance_id, map to idx and speaker in the table and confirm speaker == {participant_id}. If any mismatch, drop that ID; if none remain, omit the contribution.\n"
        "- Confirm first_response_idx equals the earliest substantive idx(response_utterance_ids).\n"
        "- If after filtering responses there is no substantive content within k, omit the contribution. Do not keep later out-of-window clarifications in official response_utterance_ids.\n"
        "- When several candidates compete for limited slots, keep the ones with clearer downstream consequence over similar low-consequence clarifications.\n"
        f"{cap_self_check}\n"
        "## Input: Testing Segment (trimmed table; idx is authoritative)\n"
        + "\n".join(lines)
        + "\n\n## Output (strict JSON only; exact shape)\n"
        + schema_block
        + "\n\n"
        + examples_block
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


def select_contributions(
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

    Returns a dict with keys: contributions (list) and coverage_summary (dict).
    On parse failure, returns {"contributions": [], "coverage_summary": {}}.
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
    resp = client.call_with_retry(
        model=client.model,
        messages=messages,
        max_tokens=3000,
        temperature=0.0,
        json_required=True,
        telemetry_label="prep.contribution_selector",
        telemetry_phase="prep",
    )

    # DEBUGGING
    print("==="*20)
    print(f"CONTRIBUTION SELECTOR O/P:\n{resp}")
    print("==="*20, end='\n\n')
    if not resp:
        return {"contributions": [], "coverage_summary": {}}
    try:
        data = json.loads(resp) if isinstance(resp, str) else resp
        # Basic shape normalization
        if not isinstance(data, dict):
            return {"contributions": [], "coverage_summary": {}}
        data.setdefault("contributions", [])
        data.setdefault("coverage_summary", {})
        return data
    except Exception:
        return {"contributions": [], "coverage_summary": {}}
