"""
Shared helpers for CAPA episode construction and validation.

Focused, minimal utilities used by selector-driven CAPA prep to standardize
episode acceptance checks and fixed-k cue normalization.
"""

import hashlib
import re
from typing import List, Dict, Any, Tuple, Optional, Sequence, Iterable

from schemas import CAPAEpisode


BANAL_LEXICON = {
    "ok",
    "okay",
    "yeah",
    "yup",
    "yep",
    "yes",
    "no",
    "mm",
    "mmm",
    "mmhm",
    "mm-hmm",
    "uh-huh",
    "uhuh",
    "right",
    "got it",
    "thanks",
    "sure",
    "hmm",
    "uh",
    "um",
    "er",
    "erm",
    "eh",
    "ah",
    "oh",
    "{vocalsound}",
    "{disfmarker}",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "but",
    "by",
    "for",
    "from",
    "have",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "just",
    "like",
    "of",
    "on",
    "or",
    "so",
    "that",
    "the",
    "then",
    "there",
    "this",
    "to",
    "uh",
    "um",
    "we",
    "you",
    "your",
}

NUMBER_WORD_TO_VALUE = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
    "hundred": "100",
    "thousand": "1000",
    "million": "1000000",
}
NUMBER_WORDS = set(NUMBER_WORD_TO_VALUE)

CONSTRAINT_CUE_PATTERNS = (
    r"\b(has|have|had)\s+to\b",
    r"\bneed(s|ed)?\s+to\b",
    r"\bmust\b",
    r"\bshould\b",
    r"\bcannot\b",
    r"\bcan't\b",
    r"\bwithout\b",
    r"\bno\s+\w+",
    r"\btarget\b",
    r"\bbudget\b",
    r"\blimit\b",
    r"\bdeadline\b",
    r"\brequire(s|d|ment)?\b",
    r"\bconstraint\b",
)


def build_id_index(transcript: List[Dict[str, Any]]) -> Dict[str, int]:
    return {u.get("utterance_id"): i for i, u in enumerate(transcript) if u.get("utterance_id")}


def _safe_id_token(value: Any, *, max_len: int = 24) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    token = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_")
    if not token:
        return ""
    if len(token) <= max_len:
        return token
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"{token[:max_len - 9]}_{digest}"


def make_cue_test_case_id(meeting_id: str, represented_user: str, cue_idx: int) -> str:
    return f"{meeting_id}_{represented_user}_t{int(cue_idx)}"


def selector_episode_suffix(selector_item: Optional[Dict[str, Any]]) -> str:
    item = selector_item or {}
    for field_name, prefix in (
        ("unit_id", "u"),
        ("contribution_id", "c"),
        ("selector_unit_id", "u"),
        ("anchor_inventory_id", "a"),
    ):
        token = _safe_id_token(item.get(field_name))
        if token:
            return f"{prefix}{token}"

    raw_rank = item.get("episode_rank") or item.get("_episode_rank") or item.get("cap_rank")
    if raw_rank is not None:
        try:
            return f"r{int(raw_rank):02d}"
        except Exception:
            token = _safe_id_token(raw_rank)
            if token:
                return f"r{token}"

    hash_payload = {
        "response_utterance_ids": list(item.get("response_utterance_ids") or []),
        "topic_summary": item.get("topic_summary"),
        "anchor_reason": item.get("anchor_reason"),
        "cue_utterance_id": item.get("cue_utterance_id"),
    }
    digest = hashlib.sha1(str(hash_payload).encode("utf-8")).hexdigest()[:8]
    return f"h{digest}"


def make_episode_test_case_id(
    meeting_id: str,
    represented_user: str,
    cue_idx: int,
    selector_item: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, str]:
    cue_test_case_id = make_cue_test_case_id(meeting_id, represented_user, cue_idx)
    suffix = selector_episode_suffix(selector_item)
    return f"{cue_test_case_id}_{suffix}", cue_test_case_id, suffix


def refresh_episode_identity(episode: CAPAEpisode) -> CAPAEpisode:
    cue_idx = int(episode.cue_index or 0)
    cue_test_case_id = make_cue_test_case_id(episode.meeting_id, episode.represented_user, cue_idx)
    suffix = episode.episode_id_suffix
    if not suffix:
        try:
            idea_units = list(episode.idea_units or [])
            first_unit = idea_units[0]
            if hasattr(first_unit, "model_dump"):
                suffix = selector_episode_suffix(first_unit.model_dump())
            elif isinstance(first_unit, dict):
                suffix = selector_episode_suffix(first_unit)
        except Exception:
            suffix = None
    suffix = suffix or selector_episode_suffix({})
    episode.cue_test_case_id = cue_test_case_id
    episode.episode_id_suffix = suffix
    episode.test_case_id = f"{cue_test_case_id}_{suffix}"
    return episode


def assert_unique_episode_ids(
    episodes_by_participant: Dict[str, Sequence[CAPAEpisode]],
    *,
    meeting_id: str,
) -> None:
    seen: set[Tuple[str, str]] = set()
    duplicates: List[str] = []
    for participant_id, episodes in (episodes_by_participant or {}).items():
        for episode in episodes or []:
            tcid = str(episode.test_case_id or "").strip()
            if not tcid:
                duplicates.append(f"{participant_id}:<missing>")
                continue
            key = (participant_id, tcid)
            if key in seen:
                duplicates.append(f"{participant_id}:{tcid}")
            seen.add(key)
    if duplicates:
        sample = ", ".join(duplicates[:10])
        raise ValueError(f"Duplicate CAPA episode ids for {meeting_id}: {sample}")


def _token_count(text: str) -> int:
    if not text:
        return 0
    return len(str(text).split())


def _clean_for_tokens(text: str) -> str:
    cleaned = re.sub(r"\{[^}]+\}", " ", str(text or "").lower())
    cleaned = re.sub(r"[^a-z0-9€$]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _tokens(text: str) -> List[str]:
    cleaned = _clean_for_tokens(text)
    return cleaned.split() if cleaned else []


def _content_tokens(text: str) -> List[str]:
    return [tok for tok in _tokens(text) if tok not in STOPWORDS and len(tok) >= 2]


def _quantity_tokens(text: str) -> set[str]:
    tokens = _tokens(text)
    quantities: set[str] = set()
    for tok in tokens:
        if re.fullmatch(r"\d+([.,]\d+)?", tok):
            quantities.add(f"num:{tok.replace(',', '.')}")
        elif tok in NUMBER_WORD_TO_VALUE:
            quantities.add(f"num:{NUMBER_WORD_TO_VALUE[tok]}")
        elif tok in {"euro", "euros"}:
            quantities.add("currency:euro")
        elif tok in {"dollar", "dollars"}:
            quantities.add("currency:dollar")
        elif tok in {"pound", "pounds"}:
            quantities.add("currency:pound")
        elif tok in {"percent", "percentage"}:
            quantities.add("unit:percent")
        elif tok in {"ms", "millisecond", "milliseconds"}:
            quantities.add("unit:millisecond")
        elif tok in {"second", "seconds"}:
            quantities.add("unit:second")
        elif tok in {"minute", "minutes"}:
            quantities.add("unit:minute")
        elif tok in {"hour", "hours"}:
            quantities.add("unit:hour")
        elif tok in {"day", "days"}:
            quantities.add("unit:day")
        elif tok in {"week", "weeks"}:
            quantities.add("unit:week")
        elif tok in {"month", "months"}:
            quantities.add("unit:month")
    return quantities


def is_fragmentary(text: str) -> bool:
    """Return true for incomplete fragments/disfluency-only turns."""
    tokens = _tokens(text)
    if not tokens:
        return True
    if all(tok in BANAL_LEXICON or tok in {"disfmarker", "vocalsound"} for tok in tokens):
        return True
    if len(tokens) <= 3 and tokens[-1] not in {"i", "a"} and len(tokens[-1]) == 1:
        return True
    if len(tokens) <= 4 and tokens[0] in {"and", "also", "so", "but"} and len(_content_tokens(text)) <= 1:
        return True
    return False


def _has_concrete_short_constraint(text: str) -> bool:
    t = _clean_for_tokens(text)
    tokens = set(t.split())
    content = _content_tokens(text)
    has_quantity = bool(re.search(r"\b\d+([.,]\d+)?\b", t)) or bool(tokens & NUMBER_WORDS)
    has_constraint_cue = any(re.search(pattern, t) for pattern in CONSTRAINT_CUE_PATTERNS)
    # Generalizable short-turn rescue: keep compact utterances that contain a
    # quantity or an explicit constraint cue plus non-banal content. This avoids
    # meeting-specific term whitelists while preserving real constraints.
    return bool(content) and (has_quantity or has_constraint_cue)


def is_banal(text: str, token_threshold: int = 5, lexicon: Optional[set] = None) -> bool:
    """Heuristic banality detector: short or lexicon-dominated utterances."""
    if not text:
        return True
    lx = lexicon or BANAL_LEXICON
    t = _clean_for_tokens(text)
    if not t:
        return True
    tokens = t.split()
    if _has_concrete_short_constraint(text):
        return False
    if is_fragmentary(text):
        return True
    if len(tokens) <= token_threshold:
        content = [tok for tok in tokens if tok not in lx and tok not in STOPWORDS]
        if not content:
            return True
        if tokens[0] in lx and len(content) <= 1:
            return True
    for kw in lx:
        if kw in t and _token_count(t) <= token_threshold:
            return True
    return False


def response_substance_score(text: str) -> float:
    """Small deterministic score used only for prep gating/diagnostics."""
    if is_banal(text) or is_fragmentary(text):
        return 0.0
    tokens = _content_tokens(text)
    if not tokens:
        return 0.0
    t = _clean_for_tokens(text)
    if _has_concrete_short_constraint(text):
        return 0.75
    score = 0.18 + min(len(set(tokens)), 12) * 0.045
    if len(set(tokens)) >= 3:
        # A compact utterance with three or more non-stopword terms is usually
        # a real proposition, even without a numeric or explicit constraint cue.
        score = max(score, 0.45)
    if any(re.search(pattern, t) for pattern in CONSTRAINT_CUE_PATTERNS) or any(phrase in t for phrase in ("i propose", "could", "would")):
        score += 0.15
    if bool(re.search(r"\b\d+([.,]\d+)?\b", t)) or bool(set(_tokens(t)) & NUMBER_WORDS):
        score += 0.2
    if len(tokens) <= 2 and not _has_concrete_short_constraint(text):
        score -= 0.2
    return max(0.0, min(1.0, score))


def is_low_substance_response(text: str, *, min_score: float = 0.45) -> bool:
    return response_substance_score(text) < min_score


def _flatten_text_values(payload: Any) -> List[str]:
    if payload is None:
        return []
    if isinstance(payload, str):
        return [payload]
    if isinstance(payload, (int, float, bool)):
        return [str(payload)]
    if isinstance(payload, list):
        values: List[str] = []
        for item in payload:
            values.extend(_flatten_text_values(item))
        return values
    if isinstance(payload, dict):
        values = []
        for value in payload.values():
            values.extend(_flatten_text_values(value))
        return values
    return [str(payload)]


def build_delegate_inventory_text(*payloads: Any) -> str:
    """Flatten participant-local JIT/profile payloads into verifier evidence text."""
    parts: List[str] = []
    for payload in payloads:
        parts.extend(_flatten_text_values(payload))
    return "\n".join(part for part in parts if str(part).strip())


def _strip_metadata_suffix(text: str) -> str:
    return re.sub(r"\s*\(topic:.*?\)\s*$", "", str(text or ""), flags=re.IGNORECASE).strip()


def build_delegate_inventory_evidence(*payloads: Any) -> List[Dict[str, Any]]:
    """Extract auditable participant-local evidence items from profile/JIT payloads."""
    evidence: List[Dict[str, Any]] = []
    seen: Dict[str, Dict[str, Any]] = {}

    def _add(item: Dict[str, Any]) -> None:
        text = str(item.get("text") or item.get("core_text") or "").strip()
        if not text:
            return
        key = f"{item.get('source')}::{text}"
        if key in seen:
            existing = seen[key]
            for list_key in ("support_utt_ids", "anchor_phrases", "context_utt_ids", "topical_tags"):
                merged = list(existing.get(list_key) or [])
                for value in item.get(list_key) or []:
                    if value not in merged:
                        merged.append(value)
                if merged:
                    existing[list_key] = merged
            for scalar_key in ("framing", "intent_type"):
                if not existing.get(scalar_key) and item.get(scalar_key):
                    existing[scalar_key] = item.get(scalar_key)
            return
        idx = len(evidence) + 1
        item.setdefault("evidence_id", f"{str(item.get('source') or 'inventory').lower()}_{idx:03d}")
        item.setdefault("core_text", _strip_metadata_suffix(text))
        evidence.append(item)
        seen[key] = item

    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for idx, point in enumerate(payload.get("jit_briefing_points") or [], start=1):
            _add(
                {
                    "evidence_id": f"jit_{idx:03d}",
                    "source": "jit",
                    "text": str(point),
                    "core_text": _strip_metadata_suffix(str(point)),
                    "support_utt_ids": [],
                    "anchor_phrases": [],
                    "context_utt_ids": [],
                }
            )
        for idx, ev in enumerate(payload.get("points_evidence") or [], start=1):
            if not isinstance(ev, dict):
                continue
            _add(
                {
                    "evidence_id": f"jit_evidence_{idx:03d}",
                    "source": "jit",
                    "text": str(ev.get("text") or ""),
                    "core_text": _strip_metadata_suffix(str(ev.get("text") or "")),
                    "support_utt_ids": list(ev.get("support_utt_ids") or []),
                    "anchor_phrases": list(ev.get("anchor_phrases") or []),
                    "context_utt_ids": list(ev.get("context_utt_ids") or []),
                    "framing": ev.get("framing"),
                    "intent_type": ev.get("intent_type"),
                    "topical_tags": list(ev.get("topical_tags") or []),
                }
            )
        core = payload.get("core_info") or {}
        for idx, point in enumerate(core.get("specific_knowledge_points") or [], start=1):
            if not isinstance(point, dict):
                continue
            context = str(point.get("Context") or point.get("context") or "").strip()
            info = str(point.get("Information") or point.get("information") or "").strip()
            text = f"{context} {info}".strip()
            _add(
                {
                    "evidence_id": f"profile_knowledge_{idx:03d}",
                    "source": "profile",
                    "text": text,
                    "core_text": info or text,
                    "support_utt_ids": [],
                    "anchor_phrases": [],
                    "context_utt_ids": [],
                }
            )
    return evidence


def _selector_target_text(selector_item: Dict[str, Any], episode: Optional[CAPAEpisode]) -> str:
    parts = [
        str(selector_item.get("topic_summary") or ""),
        str(selector_item.get("anchor_reason") or ""),
    ]
    response_ids = {str(rid) for rid in selector_item.get("response_utterance_ids") or []}
    if episode is not None:
        for utt in episode.gt_next_window or []:
            if not isinstance(utt, dict):
                continue
            if response_ids and str(utt.get("utterance_id")) not in response_ids:
                continue
            parts.append(_utterance_text(utt))
    return " ".join(part for part in parts if part)


def _official_target_response_utterances(
    selector_item: Dict[str, Any],
    episode: Optional[CAPAEpisode],
) -> List[Dict[str, Any]]:
    if episode is None:
        return []
    response_ids = {str(rid) for rid in selector_item.get("response_utterance_ids") or []}
    rows: List[Dict[str, Any]] = []
    for utt in episode.gt_next_window or []:
        if not isinstance(utt, dict):
            continue
        utt_id = str(utt.get("utterance_id") or "")
        if response_ids and utt_id not in response_ids:
            continue
        rows.append(
            {
                "utterance_id": utt_id,
                "speaker_id": utt.get("speaker_id"),
                "utterance_content": _utterance_text(utt),
            }
        )
    return rows


def _official_target_text(selector_item: Dict[str, Any], episode: Optional[CAPAEpisode]) -> str:
    explicit = str(selector_item.get("official_target_text") or "").strip()
    if explicit:
        return explicit
    utterance_text = " ".join(_utterance_text(utt) for utt in _official_target_response_utterances(selector_item, episode)).strip()
    if utterance_text:
        return utterance_text
    return str(selector_item.get("topic_summary") or "").strip()


def _extractive_target_summary(text: str, *, max_chars: int = 180) -> str:
    """Conservative summary fallback: official target text, lightly shortened."""
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return ""
    if len(cleaned) <= max_chars:
        return cleaned
    cut = cleaned[:max_chars].rsplit(" ", 1)[0].strip()
    return f"{cut}..."


def _window_ordered_rows_by_ids(
    gt_window: Sequence[Dict[str, Any]],
    response_ids: Sequence[str],
) -> List[Tuple[int, Dict[str, Any]]]:
    wanted = {str(rid) for rid in response_ids if str(rid or "").strip()}
    rows: List[Tuple[int, Dict[str, Any]]] = []
    for offset, utt in enumerate(gt_window or []):
        if not isinstance(utt, dict):
            continue
        utt_id = str(utt.get("utterance_id") or "")
        if utt_id and utt_id in wanted:
            rows.append((offset, utt))
    return rows


def _is_near_same_proposition_constituent(
    *,
    candidate_text: str,
    basis_text: str,
    min_overlap: float = 0.34,
) -> bool:
    if is_low_substance_response(candidate_text):
        return False
    candidate_tokens = set(_content_tokens(candidate_text))
    basis_tokens = set(_content_tokens(basis_text))
    if len(candidate_tokens) < 3 or len(basis_tokens) < 3:
        return False
    overlap = _token_overlap_score(candidate_text, basis_text)
    shared = candidate_tokens & basis_tokens
    if overlap >= min_overlap and len(shared) >= 2:
        return True
    # Allow slightly lower overlap when the shared terms are numerous. This
    # catches multi-utterance continuations without whitelisting domain terms.
    return overlap >= 0.25 and len(shared) >= 4


def _expand_same_speaker_constituents(
    *,
    gt_window: Sequence[Dict[str, Any]],
    retained_ids: Sequence[str],
    retained_texts: Sequence[str],
    represented_user: str,
    original_summary: str,
    max_gap: int = 2,
) -> Tuple[List[str], List[str], List[int], List[str], Optional[str]]:
    """Expand official response IDs to same-speaker in-window constituents.

    This stays deliberately conservative: only same represented-user turns
    inside the fixed GT window, near an already retained constituent, and with
    clear content overlap against the retained target/summary can be added.
    """
    retained_rows = _window_ordered_rows_by_ids(gt_window, retained_ids)
    if not retained_rows:
        return list(retained_ids), list(retained_texts), [], [], None
    retained_offsets = [offset for offset, _ in retained_rows]
    retained_id_set = {str(rid) for rid in retained_ids}
    basis_text = " ".join([str(original_summary or ""), *[str(text or "") for text in retained_texts]]).strip()
    expanded_offsets = set(retained_offsets)
    added_ids: List[str] = []
    for offset, utt in enumerate(gt_window or []):
        if not isinstance(utt, dict) or utt.get("speaker_id") != represented_user:
            continue
        utt_id = str(utt.get("utterance_id") or "")
        if not utt_id or utt_id in retained_id_set:
            continue
        nearest_gap = min(abs(offset - retained_offset) for retained_offset in retained_offsets)
        if nearest_gap > max_gap:
            continue
        text = _utterance_text(utt)
        if not _is_near_same_proposition_constituent(candidate_text=text, basis_text=basis_text):
            continue
        expanded_offsets.add(offset)
        added_ids.append(utt_id)

    ordered_ids: List[str] = []
    ordered_texts: List[str] = []
    ordered_indices: List[int] = []
    for offset in sorted(expanded_offsets):
        utt = gt_window[offset]
        ordered_ids.append(str(utt.get("utterance_id") or ""))
        ordered_texts.append(_utterance_text(utt))
        ordered_indices.append(offset)
    reason = None
    if added_ids:
        reason = f"same_speaker_in_window_constituent_expansion:{','.join(added_ids)}"
    return ordered_ids, ordered_texts, ordered_indices, added_ids, reason


def _is_elliptical_target_text(text: str) -> bool:
    tokens = _tokens(text)
    content = _content_tokens(text)
    if not content:
        return True
    deictic = {"this", "that", "these", "those", "it", "something", "anything", "one", "ones"}
    if len(content) <= 4:
        return True
    if set(tokens) & deictic and len(content) <= 8:
        return True
    return False


def _best_target_evidence(
    *,
    selector_item: Dict[str, Any],
    target_text: str,
    inventory_evidence: Optional[List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    evidence_items = [item for item in (inventory_evidence or []) if isinstance(item, dict)]
    if not evidence_items:
        return None
    evidence_ids = {str(eid) for eid in selector_item.get("target_availability_evidence_ids") or []}
    response_ids = {str(rid) for rid in selector_item.get("response_utterance_ids") or []}

    def _score(item: Dict[str, Any]) -> float:
        text = " ".join(
            str(part or "")
            for part in (
                item.get("core_text"),
                item.get("text"),
                " ".join(str(x) for x in item.get("anchor_phrases") or []),
            )
        )
        score = max(_token_overlap_score(target_text, text), _target_coverage_score(target_text, text))
        if str(item.get("evidence_id") or "") in evidence_ids:
            score += 0.50
        support_ids = {str(uid) for uid in item.get("support_utt_ids") or []}
        if response_ids and support_ids & response_ids:
            score += 0.65
        return score

    best = max(evidence_items, key=_score)
    return best if _score(best) > 0.25 else None


def canonicalize_official_target(
    *,
    selector_item: Dict[str, Any],
    episode: CAPAEpisode,
    inventory_evidence: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Attach a conservative canonical proposition for judge/eval evidence."""
    updated = dict(selector_item or {})
    target_text = _official_target_text(updated, episode)
    raw_summary = str(updated.get("topic_summary") or "").strip()
    original_summary = str(updated.get("selector_topic_summary") or raw_summary or "").strip()
    verdict = str(updated.get("target_availability_verdict") or "").strip()
    evidence = _best_target_evidence(
        selector_item=updated,
        target_text=" ".join(part for part in (target_text, raw_summary) if part),
        inventory_evidence=inventory_evidence,
    )
    evidence_text = _strip_metadata_suffix(str((evidence or {}).get("core_text") or (evidence or {}).get("text") or "")).strip()

    canonical = target_text or raw_summary
    tested_facet = None
    rewrite_reason = None
    should_use_evidence = bool(evidence_text) and (
        verdict == "testable_partial"
        or _is_elliptical_target_text(target_text)
        or (raw_summary and target_text and _token_overlap_score(raw_summary, target_text) < 0.25)
    )
    if should_use_evidence:
        canonical = evidence_text
        rewrite_reason = (
            "testable_partial_supported_facet"
            if verdict == "testable_partial"
            else "inventory_supported_elliptical_target"
        )
        if verdict == "testable_partial":
            tested_facet = evidence_text
    elif raw_summary and target_text and _token_overlap_score(raw_summary, target_text) >= 0.25:
        canonical = raw_summary

    canonical = _extractive_target_summary(canonical, max_chars=260)
    summary_rewritten = bool(rewrite_reason and canonical)
    updated["canonical_target_proposition"] = canonical
    updated["tested_facet_text"] = tested_facet
    updated["summary_rewritten"] = summary_rewritten
    updated["summary_rewrite_reason"] = rewrite_reason
    if summary_rewritten:
        if original_summary and not updated.get("selector_topic_summary"):
            updated["selector_topic_summary"] = original_summary
        updated["topic_summary"] = canonical
    return updated


def _episode_idea_unit_dicts(episode: CAPAEpisode) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    for unit in episode.idea_units or []:
        if isinstance(unit, dict):
            units.append(dict(unit))
        elif hasattr(unit, "model_dump"):
            units.append(unit.model_dump(exclude_none=True))
        else:
            try:
                units.append(dict(unit))
            except Exception:
                continue
    return units


def enforce_official_response_window(
    *,
    selector_item: Dict[str, Any],
    episode: CAPAEpisode,
    represented_user: str,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Ensure official recall targets are contained in the fixed-k gt_next_window.

    Selector responses may describe a broader discourse unit that extends beyond
    the runtime k-window. Official CAPA episodes must instead evaluate only the
    represented-user target content actually visible inside `episode.gt_next_window`.
    This sanitizer trims response ids to the fixed window, repairs anchor fields,
    and preserves the selector-origin fields as provenance.
    """
    gated = dict(selector_item or {})
    original_ids = [str(rid) for rid in (gated.get("response_utterance_ids") or [])]
    original_summary = str(gated.get("topic_summary") or "")
    if original_ids and not gated.get("selector_response_utterance_ids"):
        gated["selector_response_utterance_ids"] = list(original_ids)
    if original_summary and not gated.get("selector_topic_summary"):
        gated["selector_topic_summary"] = original_summary

    gt_window = list(episode.gt_next_window or [])
    gt_ids = [str(utt.get("utterance_id")) for utt in gt_window if isinstance(utt, dict) and utt.get("utterance_id")]
    gt_id_set = set(gt_ids)
    retained_ids: List[str] = []
    retained_texts: List[str] = []
    retained_indices: List[int] = []
    for offset, utt in enumerate(gt_window):
        if not isinstance(utt, dict):
            continue
        utt_id = str(utt.get("utterance_id") or "")
        if not utt_id:
            continue
        if original_ids and utt_id not in original_ids:
            continue
        if utt.get("speaker_id") != represented_user:
            continue
        text = _utterance_text(utt)
        if is_low_substance_response(text):
            continue
        retained_ids.append(utt_id)
        retained_texts.append(text)
        try:
            retained_indices.append(int(episode.cue_index or 0) + offset + 1)
        except Exception:
            retained_indices.append(offset)

    if not retained_ids:
        gated.update(
            {
                "quality_status": "dropped",
                "quality_drop_reason": "response_outside_window",
                "dropped_response_outside_window": 1,
                "trimmed_response_outside_window": 0,
                "official_response_id_count": 0,
                "official_target_text": "",
                "response_ids_outside_window": [rid for rid in original_ids if rid not in gt_id_set],
            }
        )
        return False, "response_outside_window", gated

    trimmed_response_ids = list(retained_ids)
    (
        retained_ids,
        retained_texts,
        retained_offsets,
        expanded_ids,
        expansion_reason,
    ) = _expand_same_speaker_constituents(
        gt_window=gt_window,
        retained_ids=retained_ids,
        retained_texts=retained_texts,
        represented_user=represented_user,
        original_summary=original_summary,
    )
    retained_indices = []
    for offset in retained_offsets:
        try:
            retained_indices.append(int(episode.cue_index or 0) + int(offset) + 1)
        except Exception:
            retained_indices.append(int(offset))

    trimmed = bool(original_ids and trimmed_response_ids != original_ids)
    official_target_text = " ".join(text for text in retained_texts if text).strip()
    first_idx = min(retained_indices) if retained_indices else int(episode.cue_index or 0) + 1
    first_id = retained_ids[0]

    repaired_summary = original_summary
    if not repaired_summary or _token_overlap_score(repaired_summary, official_target_text) < 0.25:
        repaired_summary = _extractive_target_summary(official_target_text)

    gated.update(
        {
            "unit_id": gated.get("unit_id") or gated.get("contribution_id") or selector_episode_suffix(gated),
            "response_utterance_ids": retained_ids,
            "trimmed_response_utterance_ids": trimmed_response_ids,
            "expanded_response_utterance_ids": expanded_ids,
            "official_constituent_expansion_reason": expansion_reason,
            "first_response_idx": int(first_idx),
            "anchor_utt_id": first_id,
            "topic_summary": repaired_summary,
            "official_target_text": official_target_text,
            "canonical_target_proposition": _extractive_target_summary(official_target_text, max_chars=260),
            "tested_facet_text": None,
            "summary_rewritten": bool(repaired_summary != original_summary),
            "summary_rewrite_reason": "fixed_window_extractive_summary" if repaired_summary != original_summary else None,
            "official_response_id_count": len(retained_ids),
            "response_window_trimmed": trimmed,
            "trimmed_response_outside_window": 1 if trimmed else 0,
            "dropped_response_outside_window": 0,
            "response_ids_outside_window": [rid for rid in original_ids if rid not in gt_id_set],
        }
    )

    # Keep episode-local IdeaUnit metadata in sync with the official target.
    repaired_units = _episode_idea_unit_dicts(episode)
    if repaired_units:
        repaired_units[0].update(gated)
    else:
        repaired_units = [dict(gated)]
    try:
        from schemas import IdeaUnit

        episode.idea_units = [IdeaUnit(**unit) for unit in repaired_units]
    except Exception:
        episode.idea_units = repaired_units  # type: ignore[assignment]
    return True, "", gated


def has_future_rep(
    gt_window: List[Dict[str, Any]],
    represented_user: str,
    token_threshold: int = 5,
    lexicon: Optional[set] = None,
) -> Tuple[bool, bool, float]:
    """Return (present, contentful_present, avg_tokens) for represented user in gt."""
    present = False
    contentful = False
    tok_counts: List[int] = []
    for utt in gt_window or []:
        if utt.get("speaker_id") == represented_user:
            present = True
            text = utt.get("utterance_content") or utt.get("content") or ""
            tok = _token_count(text)
            tok_counts.append(tok)
            if not is_banal(text, token_threshold=token_threshold, lexicon=lexicon):
                contentful = True
    avg = sum(tok_counts) / len(tok_counts) if tok_counts else 0.0
    return present, contentful, avg


def _substantive_rep_texts(utterances: Iterable[Dict[str, Any]], represented_user: str) -> List[str]:
    texts: List[str] = []
    for utt in utterances or []:
        if not isinstance(utt, dict) or utt.get("speaker_id") != represented_user:
            continue
        text = _utterance_text(utt)
        if not is_low_substance_response(text):
            texts.append(text)
    return texts


def _token_overlap_score(a_text: str, b_text: str) -> float:
    a = set(_content_tokens(a_text))
    b = set(_content_tokens(b_text))
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, min(len(a), len(b)))


def _target_coverage_score(target_text: str, evidence_text: str) -> float:
    target = set(_content_tokens(target_text))
    evidence = set(_content_tokens(evidence_text))
    if not target or not evidence:
        return 0.0
    return len(target & evidence) / max(1, len(target))


def _meaningful_missing_facets(target_text: str, evidence_text: str) -> List[str]:
    target_tokens = set(_content_tokens(target_text))
    evidence_tokens = set(_content_tokens(evidence_text))
    quantity_tokens = _quantity_tokens(target_text)
    missing = [tok for tok in quantity_tokens if tok not in _quantity_tokens(evidence_text)]
    specific_tokens = sorted(
        tok
        for tok in target_tokens
        if len(tok) >= 7 and tok not in evidence_tokens and tok not in {"participant", "represented"}
    )
    missing.extend(tok for tok in specific_tokens[:5] if tok not in missing)
    return missing[:8]


def _has_negation_marker(text: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9']+", str(text or "").lower()))
    return bool(tokens & {"no", "not", "never", "without", "avoid", "exclude", "excluded", "scrap", "scrapped", "drop", "dropped", "cannot", "can't", "wont", "won't"})


def assess_target_availability_stage1(
    *,
    selector_item: Dict[str, Any],
    episode: Optional[CAPAEpisode],
    inventory_text: str,
    inventory_evidence: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Deterministic evidence prefilter for participant-local target availability."""
    evidence_text = str(inventory_text or "")
    target_text = _official_target_text(selector_item, episode)
    summary_text = str(selector_item.get("topic_summary") or "").strip()
    response_ids = [str(rid) for rid in selector_item.get("response_utterance_ids") or []]
    target_tokens = set(_content_tokens(target_text))
    evidence_tokens = set(_content_tokens(evidence_text))
    base: Dict[str, Any] = {
        "target_available_to_delegate": False,
        "target_availability_verdict": "unsupported_missing",
        "target_availability_method": "deterministic_stage1",
        "target_availability_reason": "unsupported_missing",
        "target_availability_score": 0.0,
        "target_availability_evidence_ids": [],
        "target_availability_missing_facets": [],
        "target_availability_rationale": "",
        "target_availability_requires_semantic_review": False,
    }
    if not target_tokens:
        base.update(
            target_availability_verdict="unsupported_generic",
            target_availability_reason="target_has_no_substantive_tokens",
            target_availability_rationale="Official target text has no substantive content.",
        )
        return base
    if not evidence_tokens:
        base.update(
            target_availability_reason="empty_delegate_inventory",
            target_availability_missing_facets=sorted(target_tokens)[:8],
            target_availability_rationale="Participant inventory is empty.",
        )
        return base

    target_quantities = _quantity_tokens(target_text)
    evidence_quantities = _quantity_tokens(evidence_text)
    missing_numeric = sorted(tok for tok in target_quantities if tok not in evidence_quantities)
    if missing_numeric:
        base.update(
            target_availability_reason=f"missing_concrete_inventory_token:{','.join(missing_numeric[:3])}",
            target_availability_missing_facets=missing_numeric[:8],
            target_availability_rationale="Official target contains concrete numeric/quantity facets absent from participant inventory.",
        )
        return base

    evidence_items = inventory_evidence or [{"evidence_id": "inventory_text", "text": evidence_text, "core_text": evidence_text}]
    response_id_set = set(response_ids)
    scored: List[Tuple[float, Dict[str, Any], List[str]]] = []
    direct_id_hits: List[str] = []
    for item in evidence_items:
        if not isinstance(item, dict):
            continue
        item_text = " ".join(
            str(part or "")
            for part in (
                item.get("core_text"),
                item.get("text"),
                " ".join(str(x) for x in item.get("anchor_phrases") or []),
            )
        )
        score = max(_target_coverage_score(target_text, item_text), _token_overlap_score(target_text, item_text))
        item_quantities = _quantity_tokens(item_text)
        if target_quantities and target_quantities.issubset(item_quantities):
            score = max(score, 0.72)
        missing = _meaningful_missing_facets(target_text, item_text)
        support_ids = {str(uid) for uid in item.get("support_utt_ids") or []}
        if response_id_set and support_ids & response_id_set:
            score = max(score, 0.95)
            direct_id_hits.append(str(item.get("evidence_id") or "inventory_evidence"))
        scored.append((score, item, missing))

    scored.sort(key=lambda row: row[0], reverse=True)
    best_score, best_item, missing_facets = scored[0] if scored else (0.0, {}, [])
    best_id = str(best_item.get("evidence_id") or "inventory_text")
    best_text = str(best_item.get("core_text") or best_item.get("text") or "")
    summary_overlap = _token_overlap_score(summary_text, str(best_item.get("core_text") or best_item.get("text") or ""))
    evidence_ids = direct_id_hits or ([best_id] if best_item else [])

    if best_score >= 0.45 and _has_negation_marker(target_text) != _has_negation_marker(best_text):
        base.update(
            target_availability_reason="contradictory_delegate_inventory_evidence",
            target_availability_score=round(float(best_score), 3),
            target_availability_evidence_ids=evidence_ids[:5],
            target_availability_missing_facets=["contradictory_polarity"],
            target_availability_rationale="Best participant-local inventory evidence appears to contradict the target polarity.",
            target_availability_requires_semantic_review=True,
        )
        return base

    if direct_id_hits or (best_score >= 0.68 and not missing_facets):
        verdict = "supported"
        available = True
        reason = "supported_by_delegate_inventory"
        rationale = "Participant inventory directly supports the official target."
        requires_review = False
    elif best_score >= 0.40 and len(missing_facets) <= 2:
        verdict = "testable_partial"
        available = True
        reason = "testable_partial_delegate_inventory"
        rationale = "Participant inventory supports a concrete, directly usable target facet."
        requires_review = best_score < 0.55 and not direct_id_hits
    elif best_score >= 0.22 or summary_overlap >= 0.30:
        verdict = "unsupported_missing"
        available = False
        reason = "semantic_review_required"
        rationale = "Lexical evidence is ambiguous; semantic verifier should decide whether support is concrete enough."
        requires_review = True
    else:
        verdict = "unsupported_missing"
        available = False
        reason = "insufficient_delegate_inventory_overlap"
        rationale = "No participant-local inventory evidence supports the official target."
        requires_review = False

    base.update(
        target_available_to_delegate=available,
        target_availability_verdict=verdict,
        target_availability_reason=reason,
        target_availability_score=round(float(best_score), 3),
        target_availability_evidence_ids=evidence_ids[:5],
        target_availability_missing_facets=missing_facets[:8],
        target_availability_rationale=rationale,
        target_availability_requires_semantic_review=requires_review,
    )
    return base


def context_already_covers_response(
    *,
    context_last_N: List[Dict[str, Any]],
    gt_next_window: List[Dict[str, Any]],
    represented_user: str,
    response_utterance_ids: Optional[Sequence[str]] = None,
) -> Tuple[bool, float]:
    """Detect cases where the official target was already said in the visible context."""
    response_ids = {str(rid) for rid in (response_utterance_ids or [])}
    target_texts: List[str] = []
    for utt in gt_next_window or []:
        if not isinstance(utt, dict) or utt.get("speaker_id") != represented_user:
            continue
        if response_ids and str(utt.get("utterance_id")) not in response_ids:
            continue
        text = _utterance_text(utt)
        if text and not is_banal(text):
            target_texts.append(text)
    if not target_texts:
        return False, 0.0

    prior_texts = _substantive_rep_texts(context_last_N or [], represented_user)
    if not prior_texts:
        return False, 0.0

    target_text = " ".join(target_texts)
    prior_text = " ".join(prior_texts)
    overlap = _token_overlap_score(target_text, prior_text)
    target_score = max(response_substance_score(text) for text in target_texts)
    prior_token_count = len(set(_content_tokens(prior_text)))
    covered = prior_token_count >= 4 and overlap >= 0.58 and target_score < 0.72
    return covered, overlap


def assess_episode_quality(
    *,
    selector_item: Dict[str, Any],
    episode: CAPAEpisode,
    represented_user: str,
) -> Dict[str, Any]:
    """Final deterministic gate for official CAPA episodes before EIM/eval."""
    response_ids = [str(rid) for rid in selector_item.get("response_utterance_ids") or []]
    response_texts: List[str] = []
    for utt in episode.gt_next_window or []:
        if not isinstance(utt, dict) or utt.get("speaker_id") != represented_user:
            continue
        if response_ids and str(utt.get("utterance_id")) not in set(response_ids):
            continue
        response_texts.append(_utterance_text(utt))

    scores = [response_substance_score(text) for text in response_texts]
    max_score = max(scores) if scores else 0.0
    context_covered, coverage_score = context_already_covers_response(
        context_last_N=episode.context_last_N or [],
        gt_next_window=episode.gt_next_window or [],
        represented_user=represented_user,
        response_utterance_ids=response_ids,
    )

    cue_text = ""
    cue_speaker = None
    if episode.context_last_N:
        cue_utt = episode.context_last_N[-1]
        cue_text = _utterance_text(cue_utt)
        cue_speaker = cue_utt.get("speaker_id")
    target_text = " ".join(response_texts)
    self_init_anchor_invalid = (
        cue_speaker == represented_user
        and not is_banal(cue_text)
        and _token_overlap_score(cue_text, target_text) >= 0.45
    )

    reason = None
    if not response_texts or max_score < 0.45:
        reason = "low_substance"
    elif context_covered:
        reason = "context_already_covered"
    elif self_init_anchor_invalid:
        reason = "self_init_anchor_invalid"

    return {
        "quality_status": "dropped" if reason else "kept",
        "quality_drop_reason": reason,
        "response_substance_score": round(float(max_score), 3),
        "context_already_covered": bool(context_covered),
        "context_coverage_overlap": round(float(coverage_score), 3),
        "self_init_anchor_invalid": bool(self_init_anchor_invalid),
    }


def target_available_to_delegate(
    *,
    selector_item: Dict[str, Any],
    episode: Optional[CAPAEpisode],
    inventory_text: str,
) -> Tuple[bool, str, float]:
    """Backward-compatible wrapper around the richer deterministic availability gate."""
    verdict = assess_target_availability_stage1(
        selector_item=selector_item,
        episode=episode,
        inventory_text=inventory_text,
    )
    return (
        bool(verdict.get("target_available_to_delegate")),
        str(verdict.get("target_availability_reason") or ""),
        float(verdict.get("target_availability_score") or 0.0),
    )


def build_official_idea_targets(
    episode: Optional[CAPAEpisode],
    *,
    context_utterances: Optional[List[Dict[str, Any]]] = None,
    gt_window: Optional[List[Dict[str, Any]]] = None,
    context_tail: int = 8,
) -> Optional[List[Dict[str, Any]]]:
    """Build judge-facing idea targets with exact raw utterance evidence and local context."""
    if not episode or not getattr(episode, "idea_units", None):
        return None
    utterance_map: Dict[str, Dict[str, Any]] = {}
    for utt in list(context_utterances or []) + list(gt_window or []) + list(getattr(episode, "gt_next_window", None) or []):
        if not isinstance(utt, dict):
            continue
        uid = str(utt.get("utterance_id") or "")
        if uid:
            utterance_map[uid] = utt
    local_context = [
        {
            "utterance_id": str(utt.get("utterance_id") or ""),
            "speaker_id": utt.get("speaker_id"),
            "utterance_content": _utterance_text(utt),
        }
        for utt in list(context_utterances or [])[-int(context_tail):]
        if isinstance(utt, dict)
    ]
    targets: List[Dict[str, Any]] = []
    for unit in getattr(episode, "idea_units", []) or []:
        if hasattr(unit, "model_dump"):
            unit_data = unit.model_dump(exclude_none=True)
        elif isinstance(unit, dict):
            unit_data = dict(unit)
        else:
            continue
        response_ids = [str(rid) for rid in unit_data.get("response_utterance_ids") or []]
        response_rows = []
        for rid in response_ids:
            utt = utterance_map.get(rid) or {}
            response_rows.append(
                {
                    "utterance_id": rid,
                    "speaker_id": utt.get("speaker_id"),
                    "utterance_content": _utterance_text(utt),
                }
            )
        anchor_uid = unit_data.get("anchor_utt_id") or (response_ids[0] if response_ids else None)
        anchor_utt = utterance_map.get(str(anchor_uid)) if anchor_uid else {}
        targets.append(
            {
                "unit_id": unit_data.get("unit_id"),
                "anchor_utt_id": anchor_uid,
                "anchor_index": unit_data.get("first_response_idx"),
                "anchor_text": _utterance_text(anchor_utt or {}) if anchor_utt else None,
                "response_utterance_ids": response_ids,
                "response_texts": response_rows,
                "raw_response_utterances": response_rows,
                "topic_summary": unit_data.get("topic_summary"),
                "official_target_text": unit_data.get("official_target_text") or " ".join(row["utterance_content"] for row in response_rows if row.get("utterance_content")),
                "canonical_target_proposition": unit_data.get("canonical_target_proposition"),
                "tested_facet_text": unit_data.get("tested_facet_text"),
                "summary_rewritten": unit_data.get("summary_rewritten"),
                "summary_rewrite_reason": unit_data.get("summary_rewrite_reason"),
                "target_availability_verdict": unit_data.get("target_availability_verdict"),
                "target_availability_evidence_ids": unit_data.get("target_availability_evidence_ids"),
                "local_context": local_context,
            }
        )
    return targets or None


def _utterance_text(utt: Dict[str, Any]) -> str:
    return str(utt.get("utterance_content") or utt.get("content") or "")


def _cue_gap(anchor_idx: int, cue_idx: Optional[int]) -> Optional[int]:
    if cue_idx is None:
        return None
    return anchor_idx - cue_idx


def _resolve_idx(
    cue_uid: Optional[str],
    raw_idx: Optional[Any],
    id_to_idx: Dict[str, int],
) -> Optional[int]:
    if cue_uid and cue_uid in id_to_idx:
        return id_to_idx[cue_uid]
    if raw_idx is None:
        return None
    try:
        return int(raw_idx)
    except Exception:
        return None


def _is_valid_cue_idx(anchor_idx: int, cue_idx: Optional[int], k: int) -> bool:
    if cue_idx is None:
        return False
    gap = anchor_idx - cue_idx
    return 0 < gap <= int(k)


def _sanitize_response_ids(
    response_ids: List[Any],
    id_to_idx: Dict[str, int],
    transcript: List[Dict[str, Any]],
    represented_user: str,
) -> List[str]:
    sanitized: List[str] = []
    seen: set[str] = set()
    for rid in response_ids or []:
        rid_str = str(rid)
        if rid_str in seen or rid_str not in id_to_idx:
            continue
        idx = id_to_idx[rid_str]
        if idx < 0 or idx >= len(transcript):
            continue
        if transcript[idx].get("speaker_id") != represented_user:
            continue
        seen.add(rid_str)
        sanitized.append(rid_str)
    return sorted(sanitized, key=lambda rid: id_to_idx.get(rid, 10**9))


def _trim_leading_banal_response_ids(
    response_ids: List[str],
    id_to_idx: Dict[str, int],
    transcript: List[Dict[str, Any]],
) -> List[str]:
    trimmed = list(response_ids or [])
    while len(trimmed) > 1:
        idx = id_to_idx.get(trimmed[0])
        if idx is None:
            trimmed.pop(0)
            continue
        if not is_banal(_utterance_text(transcript[idx])):
            break
        trimmed.pop(0)
    return trimmed


def _find_nearest_non_banal_cue(
    anchor_idx: int,
    transcript: List[Dict[str, Any]],
    k: int,
) -> Optional[int]:
    start = max(0, anchor_idx - int(k))
    for idx in range(anchor_idx - 1, start - 1, -1):
        if not is_banal(_utterance_text(transcript[idx])):
            return idx
    return None


def _find_nearest_external_cue(
    anchor_idx: int,
    transcript: List[Dict[str, Any]],
    represented_user: str,
    k: int,
) -> Optional[int]:
    start = max(0, anchor_idx - int(k))
    for idx in range(anchor_idx - 1, start - 1, -1):
        utt = transcript[idx]
        if utt.get("speaker_id") == represented_user:
            continue
        if not is_banal(_utterance_text(utt)):
            return idx
    return None


def _cue_type_for_idx(
    cue_idx: int,
    transcript: List[Dict[str, Any]],
    represented_user: str,
    fallback_type: Optional[str] = None,
) -> str:
    speaker_id = str((transcript[cue_idx] or {}).get("speaker_id") or "")
    if speaker_id == represented_user:
        return "self-initiation"
    if fallback_type in {"explicit", "implicit-handoff", "open_floor"}:
        return fallback_type
    return "implicit-handoff"


def normalize_selector_item(
    item: Dict[str, Any],
    *,
    id_to_idx: Dict[str, int],
    transcript: List[Dict[str, Any]],
    represented_user: str,
    k: int,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Normalize selector output to a fixed-k CAPA episode contract.

    The returned item preserves the represented participant's substantive later
    response while resolving a cue that is valid for the downstream fixed-k
    runtime.
    """
    normalized = dict(item or {})
    response_ids = _sanitize_response_ids(
        list(normalized.get("response_utterance_ids") or []),
        id_to_idx=id_to_idx,
        transcript=transcript,
        represented_user=represented_user,
    )
    if not response_ids:
        return False, "no_valid_response_ids", {}

    response_ids = _trim_leading_banal_response_ids(
        response_ids=response_ids,
        id_to_idx=id_to_idx,
        transcript=transcript,
    )
    if not response_ids:
        return False, "all_responses_banal", {}

    anchor_utt_id = response_ids[0]
    anchor_idx = id_to_idx.get(anchor_utt_id)
    if anchor_idx is None:
        return False, "anchor_id_not_found", {}
    if all(is_banal(_utterance_text(transcript[id_to_idx[rid]])) for rid in response_ids):
        return False, "all_responses_banal", {}

    normalized["response_utterance_ids"] = response_ids
    normalized["first_response_idx"] = anchor_idx
    normalized["anchor_utt_id"] = anchor_utt_id

    actual_cue_uid = normalized.get("cue_utterance_id")
    actual_cue_idx = _resolve_idx(actual_cue_uid, normalized.get("cue_idx"), id_to_idx)
    actual_cue_type = str(normalized.get("cue_type") or "").strip() or None
    if actual_cue_idx is not None and 0 <= actual_cue_idx < len(transcript):
        actual_cue_uid = str(transcript[actual_cue_idx].get("utterance_id") or actual_cue_uid or "")

    semantic_cue_uid = normalized.get("semantic_cue_utterance_id")
    semantic_cue_idx = _resolve_idx(semantic_cue_uid, normalized.get("semantic_cue_idx"), id_to_idx)
    semantic_cue_type = str(normalized.get("semantic_cue_type") or "").strip() or None
    if semantic_cue_idx is not None and 0 <= semantic_cue_idx < len(transcript):
        semantic_cue_uid = str(transcript[semantic_cue_idx].get("utterance_id") or semantic_cue_uid or "")

    preferred_cue_uid = semantic_cue_uid or actual_cue_uid
    preferred_cue_idx = semantic_cue_idx if semantic_cue_idx is not None else actual_cue_idx
    preferred_cue_type = semantic_cue_type or actual_cue_type

    chosen_cue_uid: Optional[str] = None
    chosen_cue_idx: Optional[int] = None
    chosen_cue_type: Optional[str] = None
    cue_mode = "retained"

    if _is_valid_cue_idx(anchor_idx, actual_cue_idx, k):
        chosen_cue_idx = int(actual_cue_idx)
        chosen_cue_uid = str(actual_cue_uid or transcript[chosen_cue_idx].get("utterance_id") or "")
        chosen_cue_type = _cue_type_for_idx(
            chosen_cue_idx,
            transcript,
            represented_user,
            actual_cue_type,
        )
        if semantic_cue_uid and chosen_cue_uid and semantic_cue_uid != chosen_cue_uid:
            cue_mode = "selector_reanchor"
    elif _is_valid_cue_idx(anchor_idx, semantic_cue_idx, k):
        chosen_cue_idx = int(semantic_cue_idx)
        chosen_cue_uid = str(semantic_cue_uid or transcript[chosen_cue_idx].get("utterance_id") or "")
        chosen_cue_type = _cue_type_for_idx(
            chosen_cue_idx,
            transcript,
            represented_user,
            semantic_cue_type,
        )
        cue_mode = "retained"
    else:
        deterministic_idx = _find_nearest_non_banal_cue(anchor_idx, transcript, k)
        if deterministic_idx is not None:
            chosen_cue_idx = deterministic_idx
            chosen_cue_uid = str(transcript[chosen_cue_idx].get("utterance_id") or "")
            chosen_cue_type = _cue_type_for_idx(chosen_cue_idx, transcript, represented_user)
            cue_mode = "deterministic_reanchor"
        elif anchor_idx > 0:
            chosen_cue_idx = anchor_idx - 1
            chosen_cue_uid = str(transcript[chosen_cue_idx].get("utterance_id") or "")
            chosen_cue_type = "self-initiation"
            cue_mode = "self_initiation"

    if chosen_cue_idx is None or chosen_cue_uid is None or chosen_cue_idx >= anchor_idx:
        return False, "no_valid_cue", {}

    chosen_utt = transcript[chosen_cue_idx]
    if chosen_utt.get("speaker_id") == represented_user and not is_banal(_utterance_text(chosen_utt)):
        external_idx = _find_nearest_external_cue(anchor_idx, transcript, represented_user, k)
        if external_idx is None:
            return False, "self_init_anchor_invalid", {}
        chosen_cue_idx = external_idx
        chosen_cue_uid = str(transcript[chosen_cue_idx].get("utterance_id") or "")
        chosen_cue_type = _cue_type_for_idx(chosen_cue_idx, transcript, represented_user)
        cue_mode = "deterministic_reanchor"

    normalized["cue_utterance_id"] = chosen_cue_uid
    normalized["cue_idx"] = chosen_cue_idx
    normalized["cue_type"] = chosen_cue_type or "self-initiation"

    if semantic_cue_uid:
        normalized["semantic_cue_utterance_id"] = semantic_cue_uid
    if semantic_cue_idx is not None:
        normalized["semantic_cue_idx"] = int(semantic_cue_idx)
    if semantic_cue_type:
        normalized["semantic_cue_type"] = semantic_cue_type

    normalized["preferred_cue_utterance_id"] = preferred_cue_uid
    normalized["preferred_cue_idx"] = preferred_cue_idx
    normalized["preferred_cue_type"] = preferred_cue_type
    normalized["preferred_cue_gap"] = _cue_gap(anchor_idx, preferred_cue_idx)
    normalized["preferred_cue_retained"] = (
        preferred_cue_uid is None
        or (chosen_cue_uid == preferred_cue_uid and chosen_cue_idx == preferred_cue_idx)
    )
    normalized["self_initiation_fallback"] = cue_mode == "self_initiation"
    normalized["cue_mode"] = cue_mode
    normalized["effective_cue_gap"] = _cue_gap(anchor_idx, chosen_cue_idx)

    return True, "", normalized


def validate_contribution(
    contrib: Dict[str, Any],
    id_to_idx: Dict[str, int],
    transcript: List[Dict[str, Any]],
    represented_user: str,
    N: int,
    k: int,
    k_max: int = 10,
) -> Tuple[bool, str, int]:
    """
    Validate LLM-proposed contribution and compute k_used to ensure first response fits.

    Returns (ok, reason_if_not, k_used).
    """
    cue_idx = contrib.get("cue_idx")
    first_idx = contrib.get("first_response_idx")
    resp_ids = contrib.get("response_utterance_ids") or []
    cue_uid = contrib.get("cue_utterance_id")

    if cue_idx is None or first_idx is None or cue_uid is None:
        return False, "missing required indices", k

    try:
        cue_idx = int(cue_idx)
        first_idx = int(first_idx)
    except Exception:
        return False, "non-integer indices", k

    if cue_idx >= first_idx:
        return False, "directionality violated", k

    # Check response ids exist and belong to represented_user
    for rid in resp_ids:
        if rid not in id_to_idx:
            return False, f"unknown response id {rid}", k
        idx = id_to_idx[rid]
        if idx < 0 or idx >= len(transcript):
            return False, f"index out of bounds for {rid}", k
        sp = transcript[idx].get("speaker_id")
        if sp != represented_user:
            return False, f"response id {rid} not by represented user", k

    # Ensure first response is within the gt window using possibly increased k
    k_needed = (first_idx - cue_idx) + 1  # include the turn that contains first response
    k_used = min(max(k, k_needed), k_max)
    return True, "", k_used


def validate_idea_unit(
    unit: Dict[str, Any],
    id_to_idx: Dict[str, int],
    transcript: List[Dict[str, Any]],
    represented_user: str,
    N: int,
    k: int,
    k_max: int = 10,
) -> Tuple[bool, str, int]:
    """
    Validate LLM-proposed Idea Unit and compute k_used to ensure anchor fits.

    Returns (ok, reason_if_not, k_used).
    """
    cue_idx = unit.get("cue_idx")
    first_idx = unit.get("first_response_idx")
    resp_ids = unit.get("response_utterance_ids") or []
    cue_uid = unit.get("cue_utterance_id")

    if cue_idx is None or first_idx is None or cue_uid is None:
        return False, "missing required indices", k

    try:
        cue_idx = int(cue_idx)
        first_idx = int(first_idx)
    except Exception:
        return False, "non-integer indices", k

    # Remap cue_idx to authoritative index if available
    if cue_uid not in id_to_idx:
        return False, f"unknown cue id {cue_uid}", k
    cue_idx = id_to_idx[cue_uid]
    unit["cue_idx"] = cue_idx

    # Check response ids exist and belong to represented_user (non-contiguous allowed)
    response_indices: List[int] = []
    for rid in resp_ids:
        if rid not in id_to_idx:
            return False, f"unknown response id {rid}", k
        idx = id_to_idx[rid]
        if idx < 0 or idx >= len(transcript):
            return False, f"index out of bounds for {rid}", k
        sp = transcript[idx].get("speaker_id")
        if sp != represented_user:
            return False, f"response id {rid} not by represented user", k
        response_indices.append(idx)

    # Re-derive first_response_idx from response ids to avoid off-by-one drift
    if not response_indices:
        return False, "no response ids provided", k
    earliest_idx = min(response_indices)
    unit["first_response_idx"] = earliest_idx
    first_idx = earliest_idx
    # Sort response ids by actual index for consistency
    try:
        resp_ids_sorted = sorted(resp_ids, key=lambda rid: id_to_idx.get(rid, 10**9))
        unit["response_utterance_ids"] = resp_ids_sorted
    except Exception:
        pass
    # Convenience anchor id
    try:
        earliest_rid = min(resp_ids, key=lambda rid: id_to_idx.get(rid, 10**9))
        unit["anchor_utt_id"] = earliest_rid
    except Exception:
        pass

    if cue_idx >= first_idx:
        return False, "directionality violated", k

    # Ensure first response is within the gt window using possibly increased k
    k_needed = (first_idx - cue_idx) + 1  # include the turn that contains first response
    if k_needed > k_max:
        return False, "k_needed exceeds k_max", k
    k_used = min(max(k, k_needed), k_max)
    return True, "", k_used


def _derive_meeting_id(utt_id: str) -> str:
    try:
        return str(utt_id).split("_")[0]
    except Exception:
        return "Meeting"


def map_cue_type_to_tags(cue_type: Optional[str]) -> List[str]:
    ct = (cue_type or "").strip().lower()
    if ct == "explicit":
        return ["explicit"]
    if ct == "implicit-handoff":
        return ["implicit"]
    if ct == "open_floor":
        return ["open_floor"]
    if ct == "self-initiation":
        return ["implicit"]
    return []


def build_episode_from_contrib(
    contrib: Dict[str, Any],
    transcript: List[Dict[str, Any]],
    represented_user: str,
    N: int,
    k_used: int,
) -> CAPAEpisode:
    cue_idx = int(contrib.get("cue_idx"))
    cue_uid = str(contrib.get("cue_utterance_id"))
    topic_summary = str(contrib.get("topic_summary") or "")
    anchor_reason = str(contrib.get("anchor_reason") or "")
    selection_reason = (topic_summary + ("; " if topic_summary and anchor_reason else "") + anchor_reason).strip()
    importance = contrib.get("importance_score")
    cue_type = contrib.get("cue_type")
    tags = map_cue_type_to_tags(cue_type)

    left = max(0, cue_idx - (N - 1))
    right = cue_idx + 1
    context = transcript[left:right]
    gt = transcript[cue_idx + 1 : cue_idx + 1 + k_used]

    meeting_id = _derive_meeting_id(cue_uid)
    test_case_id, cue_test_case_id, episode_id_suffix = make_episode_test_case_id(
        meeting_id,
        represented_user,
        cue_idx,
        contrib,
    )
    episode = CAPAEpisode(
        meeting_id=meeting_id,
        represented_user=represented_user,
        cue_utt_id=cue_uid,
        cue_index=cue_idx,
        N=N,
        k=k_used,
        context_last_N=context,
        gt_next_window=gt,
        test_case_id=test_case_id,
        cue_test_case_id=cue_test_case_id,
        episode_id_suffix=episode_id_suffix,
        selection_reason=selection_reason or None,
        selection_tags=tags or None,
        selection_confidence=importance if isinstance(importance, (int, float)) else None,
        preferred_cue_utt_id=contrib.get("preferred_cue_utterance_id") or contrib.get("semantic_cue_utterance_id"),
        preferred_cue_index=contrib.get("preferred_cue_idx") if contrib.get("preferred_cue_idx") is not None else contrib.get("semantic_cue_idx"),
        preferred_cue_type=contrib.get("preferred_cue_type") or contrib.get("semantic_cue_type"),
        preferred_cue_gap=contrib.get("preferred_cue_gap"),
        preferred_cue_retained=contrib.get("preferred_cue_retained"),
        self_initiation_fallback=contrib.get("self_initiation_fallback"),
        cue_mode=contrib.get("cue_mode"),
        effective_cue_gap=contrib.get("effective_cue_gap"),
    )
    return episode


def compute_episode_metrics(episodes: List[CAPAEpisode], represented_user: str) -> Dict[str, Any]:
    present = 0
    contentful = 0
    tok_total = 0.0
    tok_ct = 0
    for ep in episodes:
        p, c, avg = has_future_rep(ep.gt_next_window or [], represented_user)
        if p:
            present += 1
        if c:
            contentful += 1
        if avg > 0:
            tok_total += avg
            tok_ct += 1
    total = len(episodes)
    return {
        "total": total,
        "presence_rate": round(100.0 * present / total, 1) if total else 0.0,
        "contentful_rate": round(100.0 * contentful / total, 1) if total else 0.0,
        "avg_tokens_in_rep_gt": round(tok_total / tok_ct, 1) if tok_ct else 0.0,
    }
