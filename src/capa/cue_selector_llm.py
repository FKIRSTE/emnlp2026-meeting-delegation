"""
LLM-based Cue Selector for CAPA Episodes

Selects a small set of cue utterances per participant from the testing segment.
The selector returns only utterance_ids (plus short reasons/tags); N,k windows are
built locally by the episode generator.

Inputs are intentionally simple and deterministic:
- Compact transcript table: idx | utterance_id | speaker | text_trim
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import json

from llm_client import LLMClient


@dataclass
class TableRow:
    index: int
    utterance_id: str
    speaker_id: str
    text_trim: str


@dataclass
class SelectedCue:
    utterance_id: str
    reason: Optional[str]
    tags: List[str]
    confidence: Optional[float]


def _trim_text(s: str, max_len: int = 150) -> str:
    s = (s or "").strip()
    return (s[: max_len - 3] + "...") if len(s) > max_len else s


def build_transcript_table(
    transcript: List[Dict[str, Any]],
    idx_map: Dict[str, int],
    start_uid: str,
    end_uid: str,
) -> List[TableRow]:
    """Build compact, global-indexed table for the testing segment (inclusive bounds)."""
    if start_uid not in idx_map or end_uid not in idx_map:
        return []
    start_idx = idx_map[start_uid]
    end_idx = idx_map[end_uid]
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx
    rows: List[TableRow] = []
    for i in range(start_idx, end_idx + 1):
        utt = transcript[i]
        uid = utt.get("utterance_id") or f"utt_{i}"
        spk = utt.get("speaker_id") or ""
        txt = utt.get("utterance_content") or utt.get("content") or ""
        rows.append(TableRow(i, uid, spk, _trim_text(txt)))
    return rows


def _render_prompt(
    meeting_id: str,
    participant_id: str,
    role_summary: str,
    table_rows: List[TableRow],
    budget: int,
    min_gap: int,
    N: int,
    k: int,
) -> List[Dict[str, str]]:
    # Render compact table text
    header = "idx | utterance_id | speaker | text_trim"
    lines = [header]
    for r in table_rows:
        lines.append(f"{r.index} | {r.utterance_id} | {r.speaker_id} | {r.text_trim}")

    system = (
        "You are an NLP expert agent specializing in meeting conversation analysis. Your task is to analyze a meeting transcript segment and select high-information 'cue utterances' that will be used for creating test instances for evaluating a meeting delegate that represents "
        "a particular user within that meeting. A cue is a utterance (line/dialogue) that, when observed by the delegate for the represented participant, creates a substantive opportunity for that participant to contribute within the next few (k) turns(e.g., explicit question to them, role-relevant decision point, topic shift, open floor, conflict/clarification). " 
        "A valid anchor is a cue after which, within the next k turns, the represented participant actually produces a knowledge-bearing contribution (KBU) (verifiable from the given transcript).\n "
        "Knowledge-Bearing Unit (KBU): fact, proposal, decision, plan, quantified detail, constraint, non-trivial question. Try to avoid: pure acknowledgments/backchannels/phatic utterances."
        "Return strict JSON only."
    )
    # user = (
    #     f"## Context\n- Meeting: {meeting_id}\n- Participant: {participant_id}\n- Role summary (1-2 lines): {role_summary}\n"
    #     f"- Selector budget: Select at most {budget} cue utterances, spaced ≥ {min_gap} utterances apart.\n"
    #     f"- Windows are built downstream: we will construct episodes using fixed N={N} and k={k}; focus on picking anchor utterances.\n\n"
    #     "## What counts as a high-information cue (choose diverse ones if present)\n"
    #     "- explicit: direct question/request addressed by name/role.\n"
    #     "- decision: commitments/assignments/approvals.\n"
    #     "- open_floor: safe opportunity to add value on open floor.\n"
    #     "- topic_shift: boundary/transition where framing matters.\n"
    #     "- conflict: disagreement, correction, clarification need.\n"
    #     "- implicit: strongly role-relevant ask without naming the person.\n\n"
    #     "## Input: Testing Segment (trimmed table)\n"
    #     + "\n".join(lines)
    #     + "\n\n##Your task\n"
    #     f"- Select ≤ {budget} utterance_ids that represent substantive contribution opportunities for {participant_id}. Verify actually if the represented user then follows with substantive responses within the following k utterances in the transcript for each potential cue, and only then proceed with that cue.\n"
    #     f"- Keep picks ≥ {min_gap} utterances apart (use idx distance).\n"
    #     "- Prefer diversity across tags (explicit, decision, open_floor, topic_shift, conflict, implicit).\n"
    #     "- Prefer cues where the participant's contribution would be specific and valuable, not generic acknowledgments.\n"
    #     "##Guardrails\n"
    #     "- Prefer anchors that precede confirmed speaking turn(s) by the participant within the next k turns when such anchors exist. Further, these speaking turns (after the selected cue/anchor) by the represented participant should be informative and substantial, rather than banal acknowledgments or phatic utterances, so that we can really test the delegate.\n"
    #     "- Do not use the participant's own utterance as the cue unless that line explicitly commits to a follow-up by them (e.g., 'I\'ll propose options next'); otherwise prefer an external open-floor/topic-shift/handoff anchor.\n"
    #     "- Never invent ids. If you cannot find enough, return as many as you can (0..budget).\n\n"
    #     "- Prefer anchors immediately preceding a confirmed, information-bearing contribution by the represented participant within k.\n"
    #     "- Avoid anchors near the segment end unless a clear participant contribution appears within k."
    #     "## Output (strict JSON only)\n"

    #     "## Examples:"
    # )

    user = (
  f"## Context\n- Meeting: {meeting_id}\n- Participant: {participant_id}\n- Role summary (1-2 lines): {role_summary}\n"
#   f"- Selector budget: Select at most {budget} cue utterances, spaced >= {min_gap} utterances apart.\n"
  f"- Windows are built downstream: we will construct test episodes/cases for the meeting delegate of the participant using fixed N={N} (given context window) and k={k}(future window to predict); focus on picking anchor utterances.\n\n"
  "## What counts as a high-information cue (choose diverse ones if present)\n"
  "- explicit: direct question/request addressed by name/role.\n"
  "- decision: commitments/assignments/approvals.\n"
  "- open_floor: safe opportunity to add value on open floor.\n"
  "- topic_shift: boundary/transition where framing matters.\n"
  "- conflict: disagreement, correction, clarification need.\n"
  "- implicit: strongly role-relevant ask without naming the person.\n\n"
  "## Banality guidance (filter out)\n"
  "- Treat as banal/backchannel if the represented participant's within-k line is very short (<= 5 tokens) or mainly: ok, okay, yeah, yup, mm-hmm, uh-huh, "
  "right, got it, thanks, sure, hmm, {vocalsound}, {disfmarker}, ...\n\n"
  "## Ordering — IMPORTANT\n"
  "- Use the 'idx' column as the ONLY source of order and distance.\n"
  "- Do NOT infer order or distance from utterance_id strings; they are identifiers, not necessarily ordering.\n\n"
  "## Input: Testing Segment (trimmed table)\n"
  + "\n".join(lines)
  + "\n\n## Your task\n"
#   f"- Select <= {budget} utterance_ids that represent substantive contribution opportunities for {participant_id}. Verify against the transcript that the "
f"- Select utterance_ids that represent substantive contribution opportunities for {participant_id}. Verify against the transcript that the "
  f"represented participant actually responds with knowledge-bearing content **within the next k turns of the selected cue utterance**;\n"
#   f"- Keep picks >= {min_gap} utterances apart (use idx distance).\n"
  "- Only keep anchors that pass ALL checks below:\n"
  "  1) Within-k presence: The represented participant MUST speak within idx range (i+1..i+k).\n"
  "  2) KBU: At least one of those (could be multiple) within-k turns MUST contain a knowledge-bearing unit (fact, proposal, decision, plan, quantified detail, constraint, or non-trivial question) by the represented participant — not just acknowledgments/backchannels/phatic content.\n"
  "  3) Directionality: All cited responses MUST have idx > cue idx (future-only). If any cited response has idx <= cue idx, discard the cue.\n"
  "- When multiple anchors precede the same valid response, choose the EARLIEST valid anchor (smallest idx) .\n\n" # that also respects min_gap
  "- Prefer diversity across tags (explicit, decision, open_floor, topic_shift, conflict, implicit).\n"
  "- Prefer cues where the participant's contribution would be specific and valuable, not generic acknowledgments.\n\n"
  "## Guardrails\n"
  "- A valid cue must be followed (within k) by at least one response turn from the represented participant containing a knowledge-bearing unit (KBU).\n"
  "- Do not use the participant's own utterance as the cue unless that line explicitly commits to a follow-up by them (e.g., 'I'll propose options next', ...; also inferable/verifiable from whether they expand on it later (within k turns) in the given transcript); "
  "otherwise prefer an external open-floor/topic-shift/handoff anchor.\n"
#   "- Never invent ids. If you cannot find enough, return as many as you can (0..budget).\n"
  "- Avoid cues near the segment end unless a clear participant contribution appears within k.\n\n"
  "## Output (strict JSON only)\n"
  "{\n"
  '  "selected_cues": [\n'
  '    {"utterance_id": "<id>", "reason": "<brief transcript-grounded justification>", "tags": '
  '["explicit"|"decision"|"open_floor"|"topic_shift"|"conflict"|"implicit"], "confidence": 0.0}\n'
  "  ],\n"
  '  "coverage_summary": {"explicit":0, "decision":0, "open_floor":0, "topic_shift":0, "conflict":0, "implicit":0}\n'
  "}\n\n"
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
  '  "selected_cues": [\n'
  '    {"utterance_id": "MeetingX_040", "reason": "Explicit ask to Industrial Designer; within k=5 they respond with a quantified KBU at MeetingX_042.", '
  '"tags": ["explicit"], "confidence": 0.91}\n'
  "  ],\n"
  '  "coverage_summary": {"explicit":1, "decision":0, "open_floor":0, "topic_shift":0, "conflict":0, "implicit":0}\n'
  "}\n\n"
  "### Example 2 (Implicit/open_floor; represented participant responds with KBUs within k)\n"
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
  '  "selected_cues": [\n'
  '    {"utterance_id": "MeetingY_020", "reason": "Implicit handoff relevant to UI; within k=5, UI contributes KBUs at MeetingY_022 and MeetingY_024.", '
  '"tags": ["implicit","open_floor"], "confidence": 0.88}\n'
  "  ],\n"
  '  "coverage_summary": {"explicit":0, "decision":0, "open_floor":1, "topic_shift":0, "conflict":0, "implicit":1}\n'
  "}\n\n"
  "### Example 3 (Chime In; represented participant contributes within k without being addressed)\n"
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
  '  "selected_cues": [\n'
  '    {"utterance_id": "MeetingZ_056", "reason": "Marketing chimes in with a quantified market-timing KBU at MeetingZ_057 within k=5 after the tooling '
  'constraint.", "tags": ["implicit","topic_shift"], "confidence": 0.82}\n'
  "  ],\n"
  '  "coverage_summary": {"explicit":0, "decision":0, "open_floor":0, "topic_shift":1, "conflict":0, "implicit":1}\n'
  "}\n\n"
  "### Example 4 (No valid cues; only phatic/banal responses within k)\n"
  "Context: Meeting=MeetingN, Participant=Project Manager, N=15, k=3\n"
  "Input snippet (trimmed table):\n"
  "idx | utterance_id | speaker | text_trim\n"
  "10 | MeetingN_010 | User Interface | We're almost done.\n"
  "11 | MeetingN_011 | Project Manager | Okay.\n"
  "12 | MeetingN_012 | Marketing | Yep.\n"
  "13 | MeetingN_013 | Industrial Designer | {vocalsound}\n"
  "Output JSON:\n"
  "{\n"
  '  "selected_cues": [],\n'
  '  "coverage_summary": {"explicit":0, "decision":0, "open_floor":0, "topic_shift":0, "conflict":0, "implicit":0}\n'
  "}\n"
  "### Targeted example\n"
  "Bad (reject): cue=Meet_32 (idx=32) with reason citing UI responses at Meet_19 and Meet_23.\n"
  "- Invalid because 19 and 23 have idx < 32 (they are in the past, not within (32+1..32+k)).\n"
  "Good (keep): choose an earlier anchor (e.g., idx around 218-220) that truly precedes UI responses at 19/23.\n\n" #  and satisfies min_gap
  )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def select_cues_for_participant(
    client: LLMClient,
    meeting_id: str,
    participant_id: str,
    role_summary: str,
    transcript_table: List[TableRow],
    budget: int,
    min_gap: int,
    N: int,
    k: int,
) -> List[SelectedCue]:
    messages = _render_prompt(
        meeting_id=meeting_id,
        participant_id=participant_id,
        role_summary=role_summary,
        table_rows=transcript_table,
        budget=budget,
        min_gap=min_gap,
        N=N,
        k=k,
    )
    resp = client.call_with_retry(
        model=client.model,
        messages=messages,
        max_tokens=2500,
        temperature=0.0,
        json_required=True,
        telemetry_label="prep.cue_selector",
        telemetry_phase="prep",
    )  # Increased for GPT-5-mini (was 1200)
    if not resp:
        return []
    try:
        data = json.loads(resp) if isinstance(resp, str) else resp
    except Exception:
        return []
    cues: List[SelectedCue] = []
    for item in (data.get("selected_cues") or []):
        uid = str(item.get("utterance_id")) if item.get("utterance_id") is not None else None
        if not uid:
            continue
        reason = item.get("reason")
        tags = [str(t) for t in (item.get("tags") or []) if isinstance(t, str)]
        conf = item.get("confidence")
        try:
            conf_f = float(conf) if conf is not None else None
        except Exception:
            conf_f = None
        cues.append(SelectedCue(uid, reason, tags, conf_f))
    return cues


def enforce_min_gap(selected: List[SelectedCue], id_to_index: Dict[str, int], min_gap: int) -> List[SelectedCue]:
    """Enforce spacing on LLM-selected cues deterministically.

    Sort by global index asc, then by confidence desc to prefer earlier and higher-confidence picks.
    """
    def _key(c: SelectedCue) -> Tuple[int, float]:
        idx = id_to_index.get(c.utterance_id, 10**9)
        conf = float(c.confidence) if c.confidence is not None else 0.0
        return (idx, -conf)

    ordered = sorted(selected, key=_key)
    kept: List[SelectedCue] = []
    last_idx = -10**9
    for c in ordered:
        idx = id_to_index.get(c.utterance_id, 10**9)
        if idx - last_idx >= min_gap:
            kept.append(c)
            last_idx = idx
    return kept
