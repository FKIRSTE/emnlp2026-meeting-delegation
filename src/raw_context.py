from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from schemas import CandidateCoverageStatus


@dataclass
class RawModularPromptContext:
    participant_id: str
    participant_code: str
    conversation_window: List[Dict[str, Any]]
    candidate_inventory: List[Dict[str, Any]]
    available_candidates: List[Dict[str, Any]]
    covered_candidates_readonly: List[Dict[str, Any]]
    candidate_coverage_flags: Dict[str, Dict[str, Any]]
    delegate_action_log: List[Dict[str, Any]]


@dataclass
class MonoRawPromptContext:
    participant_id: str
    profile_text: str
    jit_briefing_points: List[str]
    delegate_action_log: List[Dict[str, Any]]
    conversation_window: List[Dict[str, Any]]


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _normalize_lookup_key(text: str) -> str:
    text = _normalize_whitespace(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return _normalize_whitespace(text)


def _trim_text(text: str, limit: int = 180) -> str:
    text = _normalize_whitespace(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _utterance_content(utterance: Optional[Dict[str, Any]]) -> str:
    if not isinstance(utterance, dict):
        return ""
    return str(utterance.get("utterance_content") or utterance.get("content") or "")


def raw_briefing_point_text(point: Any) -> str:
    """Project structured JIT/profile-like records to raw text only."""
    if isinstance(point, dict):
        for key in ("text", "Information", "information", "point", "summary"):
            value = point.get(key)
            if value:
                return str(value)
        return ""
    return str(point or "")


def _parse_candidate_point(text: str) -> Tuple[str, Optional[str], Optional[str]]:
    if not isinstance(text, str):
        return str(text or ""), None, None
    s = text.strip()
    if not s:
        return "", None, None
    topic = None
    context_hint = None
    core_text = s
    try:
        match = re.search(
            r"\(topic:\s*([^;()]+?)(?:;\s*context:\s*([^()]+?))?\)\s*$",
            s,
            flags=re.IGNORECASE,
        )
        if match:
            topic = (match.group(1) or "").strip() or None
            if match.group(2):
                context_hint = (match.group(2) or "").strip() or None
            core_text = s[: match.start()].rstrip()
    except Exception:
        topic = None
        context_hint = None
        core_text = s
    return core_text or s, topic, context_hint


def infer_candidate_stance_function(text: Optional[str]) -> str:
    """Infer a coarse, text-only stance label without using meeting state."""
    normalized = _normalize_lookup_key(str(text or ""))
    if not normalized:
        return "neutral"
    caution_terms = {
        "risk", "risks", "cost", "costs", "expensive", "budget", "constraint", "constraints",
        "limit", "limits", "limited", "avoid", "concern", "concerns", "feasible", "feasibility",
        "tradeoff", "tradeoffs", "requirement", "requirements", "must", "need", "needs",
        "should not", "cannot", "can't", "instead", "rather than",
    }
    question_terms = {"ask", "question", "clarify", "check", "whether", "if"}
    facilitation_terms = {"summarize", "assign", "coordinate", "schedule", "decide", "next step", "action item"}
    support_terms = {"support", "agree", "advocate", "propose", "recommend", "suggest", "highlight", "emphasize"}

    def has_phrase(terms: Set[str]) -> bool:
        return any(term in normalized for term in terms)

    if has_phrase(caution_terms):
        return "constraint"
    if has_phrase(question_terms):
        return "question"
    if has_phrase(facilitation_terms):
        return "facilitation"
    if has_phrase(support_terms):
        return "proposal"
    return "neutral"


def normalize_participant_code(participant_id: Optional[str], profile: Optional[Any]) -> str:
    base = participant_id or ""
    if not base and profile and getattr(profile, "inferred_role_and_goals", None):
        base = getattr(profile.inferred_role_and_goals, "primary_role", "") or ""
    base = base or "AGENT"
    code = re.sub(r"[^A-Za-z0-9]", "", base.upper())
    return code or "AGENT"


def _iter_profile_points(profile: Optional[Any]) -> Iterable[Tuple[str, Optional[str]]]:
    if not profile or not getattr(profile, "core_info", None):
        return []
    points = getattr(profile.core_info, "specific_knowledge_points", []) or []
    rows: List[Tuple[str, Optional[str]]] = []
    for point in points:
        if isinstance(point, dict):
            info = point.get("information") or point.get("Information") or point.get("text") or point.get("Text")
            context = point.get("context") or point.get("Context")
        else:
            info = getattr(point, "Information", None) or getattr(point, "information", None) or getattr(point, "text", None)
            context = getattr(point, "Context", None) or getattr(point, "context", None)
        if info:
            rows.append((str(info), context))
    return rows


def _coverage_flag_value(flag: Any, key: str, default: Any = False) -> Any:
    if isinstance(flag, CandidateCoverageStatus):
        return getattr(flag, key, default)
    if isinstance(flag, dict):
        return flag.get(key, default)
    return default


def build_conversation_window(snapshot_utterances: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "utterance_id": utt.get("utterance_id"),
            "speaker_id": utt.get("speaker_id"),
            "content": _utterance_content(utt),
        }
        for utt in snapshot_utterances or []
        if isinstance(utt, dict)
    ]


def build_transcript_participant_roster(snapshot_utterances: Sequence[Dict[str, Any]]) -> List[str]:
    seen: Set[str] = set()
    roster: List[str] = []
    for utt in snapshot_utterances or []:
        if not isinstance(utt, dict):
            continue
        speaker = utt.get("speaker_id")
        if speaker and speaker not in seen:
            seen.add(speaker)
            roster.append(speaker)
    return roster


def render_profile_as_plain_text(profile: Optional[Any], participant_id: str) -> str:
    if profile is None:
        return f"Participant role: {participant_id}"

    lines: List[str] = [f"Participant role: {participant_id}"]

    inferred = getattr(profile, "inferred_role_and_goals", None)
    primary_role = getattr(inferred, "primary_role", None) if inferred else None
    if primary_role:
        lines.append(f"Primary role: {primary_role}")

    goals = list(getattr(inferred, "high_level_goals", []) or []) if inferred else []
    if goals:
        lines.append("Goals:")
        lines.extend(f"- {goal}" for goal in goals if goal)

    core_info = getattr(profile, "core_info", None)
    knowledge_areas = list(getattr(core_info, "knowledge_areas", []) or []) if core_info else []
    if knowledge_areas:
        lines.append("Knowledge areas:")
        lines.extend(f"- {area}" for area in knowledge_areas if area)

    specific_points = list(getattr(core_info, "specific_knowledge_points", []) or []) if core_info else []
    if specific_points:
        lines.append("Specific knowledge points:")
        for point in specific_points:
            context = getattr(point, "Context", None) or getattr(point, "context", None)
            info = getattr(point, "Information", None) or getattr(point, "information", None) or getattr(point, "text", None)
            if not info and isinstance(point, dict):
                context = point.get("Context") or point.get("context")
                info = point.get("Information") or point.get("information") or point.get("text")
            info_text = _normalize_whitespace(str(info or ""))
            context_text = _normalize_whitespace(str(context or ""))
            if not info_text:
                continue
            if context_text:
                lines.append(f"- [{context_text}] {info_text}")
            else:
                lines.append(f"- {info_text}")

    style = getattr(profile, "communication_style", None)
    style_bits: List[str] = []
    for label, value in (
        ("Tone", getattr(style, "typical_tone", None) if style else None),
        ("Verbosity", getattr(style, "verbosity", None) if style else None),
        ("Formality", getattr(style, "formality", None) if style else None),
    ):
        if value:
            style_bits.append(f"{label}: {value}")
    if style_bits:
        lines.append("Communication style: " + "; ".join(style_bits))

    aliases = list(getattr(profile, "identity_aliases", []) or [])
    if aliases:
        alias_text = ", ".join(_normalize_whitespace(str(alias)) for alias in aliases if _normalize_whitespace(str(alias)))
        if alias_text:
            lines.append(f"Known names or aliases: {alias_text}")

    return "\n".join(lines)


def build_raw_action_log(smm: Any, limit: int = 5) -> List[Dict[str, Any]]:
    episodic = list(smm.get_episodic_memory() or []) if smm is not None else []
    recent_actions: List[Dict[str, Any]] = []
    for entry in episodic[-limit:]:
        recent_actions.append(
            {
                "cue_utterance_id": getattr(entry, "utterance_id_of_action", None),
                "action": getattr(entry, "action_taken", None),
                "spoken_content": getattr(entry, "point_made_summary", None),
            }
        )
    return recent_actions


def build_mono_raw_prompt_context(
    smm: Any,
    snapshot_utterances: Sequence[Dict[str, Any]],
    participant_id: str,
    action_log_limit: int = 5,
) -> MonoRawPromptContext:
    profile = smm.get_user_profile() if smm is not None else None
    briefing = list(smm.get_jit_briefing() or []) if smm is not None else []
    return MonoRawPromptContext(
        participant_id=participant_id,
        profile_text=render_profile_as_plain_text(profile, participant_id),
        jit_briefing_points=[
            raw_briefing_point_text(point)
            for point in briefing
            if raw_briefing_point_text(point).strip()
        ],
        delegate_action_log=build_raw_action_log(smm, limit=action_log_limit),
        conversation_window=build_conversation_window(snapshot_utterances),
    )


def build_raw_modular_prompt_context(
    smm: Any,
    snapshot_utterances: Sequence[Dict[str, Any]],
    participant_id: str,
    coverage_flags: Optional[Dict[str, Any]] = None,
    action_log_limit: int = 5,
    fair_projection: bool = False,
) -> RawModularPromptContext:
    profile = smm.get_user_profile() if smm is not None else None
    briefing = list(smm.get_jit_briefing() or []) if smm is not None else []
    participant_code, candidate_inventory, available_candidates, covered_candidates_readonly = build_candidate_inventory(
        participant_id=participant_id,
        profile=profile,
        briefing=briefing,
        coverage_flags=coverage_flags or {},
        fair_projection=fair_projection,
    )
    candidate_coverage_flags = {
        entry.get("candidate_id"): dict(entry.get("coverage_status") or {})
        for entry in candidate_inventory
        if entry.get("candidate_id")
    }
    return RawModularPromptContext(
        participant_id=participant_id,
        participant_code=participant_code,
        conversation_window=build_conversation_window(snapshot_utterances),
        candidate_inventory=candidate_inventory,
        available_candidates=available_candidates,
        covered_candidates_readonly=covered_candidates_readonly,
        candidate_coverage_flags=candidate_coverage_flags,
        delegate_action_log=build_raw_action_log(smm, limit=action_log_limit),
    )


def mono_raw_prompt_context_to_payload(prompt_context: MonoRawPromptContext) -> Dict[str, Any]:
    return asdict(prompt_context)


def raw_modular_prompt_context_to_payload(prompt_context: RawModularPromptContext) -> Dict[str, Any]:
    return asdict(prompt_context)


def build_eval_only_context(
    smm: Any,
    snapshot_utterances: Sequence[Dict[str, Any]],
    participant_id: str,
) -> Dict[str, Any]:
    recent_delegate_actions = build_raw_action_log(smm, limit=3)
    payload: Dict[str, Any] = {}
    if recent_delegate_actions:
        payload["recent_delegate_actions"] = recent_delegate_actions
    return payload


def slice_transcript_window(
    transcript_utterances: Sequence[Dict[str, Any]],
    end_index: int,
    window_size: int,
) -> List[Dict[str, Any]]:
    if not transcript_utterances:
        return []
    safe_end = max(0, min(int(end_index), len(transcript_utterances) - 1))
    size = max(1, int(window_size))
    start = max(0, safe_end - size + 1)
    return list(transcript_utterances[start:safe_end + 1])


def build_candidate_inventory(
    participant_id: str,
    profile: Optional[Any],
    briefing: Sequence[str],
    coverage_flags: Optional[Dict[str, Any]] = None,
    fair_projection: bool = False,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    participant_code = normalize_participant_code(participant_id, profile)
    coverage_flags = coverage_flags or {}

    deduped: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    def safe_text(value: Any) -> str:
        if isinstance(value, dict):
            return _normalize_whitespace(str(
                value.get("text")
                or value.get("point")
                or value.get("Information")
                or value.get("information")
                or value.get("info")
                or ""
            ))
        return _normalize_whitespace(str(value or ""))

    def add_item(text: Optional[Any], source: str, origin_index: int, context_note: Optional[str] = None) -> None:
        normalized = safe_text(text) if fair_projection else _normalize_whitespace(text or "")
        if not normalized:
            return
        key = _normalize_lookup_key(normalized)
        if not key or key in seen:
            return
        seen.add(key)
        core_text, topic, context_hint = _parse_candidate_point(normalized)
        entry: Dict[str, Any] = {
            "source": source,
            "text": normalized,
            "core_text": core_text,
            "origin_index": origin_index,
            "inventory_tier": "primary" if source == "JIT" else "supplementary",
            "stance_function": infer_candidate_stance_function(core_text),
        }
        if topic:
            entry["topic"] = topic
        if context_hint:
            entry["context_hint"] = context_hint
        if context_note:
            entry["context"] = context_note
        deduped.append(entry)

    for idx, point in enumerate(briefing or []):
        add_item(point if fair_projection else (point if isinstance(point, str) else str(point or "")), "JIT", idx)

    for idx, (point, context_note) in enumerate(_iter_profile_points(profile)):
        add_item(point, "Profile", idx, context_note)

    inventory: List[Dict[str, Any]] = []
    available: List[Dict[str, Any]] = []
    readonly: List[Dict[str, Any]] = []
    for idx, entry in enumerate(deduped, start=1):
        candidate = dict(entry)
        candidate_id = f"{participant_code}_pt_{idx:03d}"
        candidate["candidate_id"] = candidate_id
        flag = coverage_flags.get(candidate_id)
        candidate["coverage_status"] = {
            "covered_in_current_episode": bool(_coverage_flag_value(flag, "covered_in_current_episode", False)),
            "covered_in_previous_episodes": bool(_coverage_flag_value(flag, "covered_in_previous_episodes", False)),
            "coverage_count": int(_coverage_flag_value(flag, "coverage_count", 0) or 0),
            "last_covered_utterance_id": _coverage_flag_value(flag, "last_covered_utterance_id", None),
            "last_covered_episode_id": _coverage_flag_value(flag, "last_covered_episode_id", None),
        }
        inventory.append(candidate)
        if candidate["coverage_status"]["covered_in_current_episode"]:
            readonly.append(candidate)
        else:
            available.append(candidate)

    return participant_code, inventory, available, readonly
