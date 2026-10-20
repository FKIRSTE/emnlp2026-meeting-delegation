from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from schemas import IdeaUnit

from capa.episode_utils import build_episode_from_contrib, has_future_rep


AnchorRecord = Dict[str, Any]


def stable_anchor_inventory_id(
    meeting_id: str,
    participant_id: str,
    anchor_utt_id: str,
    *,
    source_unit_id: Optional[str] = None,
    response_utterance_ids: Optional[Sequence[str]] = None,
    topic_summary: Optional[str] = None,
) -> str:
    base = f"{meeting_id}::{participant_id}::{anchor_utt_id}"
    source = str(source_unit_id or "").strip()
    if source:
        return f"{base}::{source}"
    signature_payload = {
        "response_utterance_ids": list(response_utterance_ids or []),
        "topic_summary": str(topic_summary or "").strip(),
    }
    if signature_payload["response_utterance_ids"] or signature_payload["topic_summary"]:
        digest = hashlib.sha1(
            json.dumps(signature_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:10]
        return f"{base}::h{digest}"
    return base


def anchor_inventory_filename(participant_id: str, meeting_id: str) -> str:
    slug = participant_id.replace(" ", "_")
    return f"anchor_inventory_for_{slug}_from_{meeting_id}.json"


def k_diagnostics_filename(participant_id: str, meeting_id: str, k: int) -> str:
    slug = participant_id.replace(" ", "_")
    return f"k_diagnostics_for_{slug}_from_{meeting_id}_k{k}.json"


def freeze_anchor_inventory_records(
    *,
    meeting_id: str,
    participant_id: str,
    units: Sequence[Dict[str, Any]],
    id_to_idx: Dict[str, int],
    inventory_reference_k: Optional[int] = None,
    selector: str = "llm-idea",
) -> List[AnchorRecord]:
    records: List[AnchorRecord] = []
    seen_keys: set[Tuple[str, str, str]] = set()

    for unit in units or []:
        response_ids = [rid for rid in list(unit.get("response_utterance_ids") or []) if rid in id_to_idx]
        if not response_ids:
            continue
        response_ids = sorted(dict.fromkeys(response_ids), key=lambda rid: id_to_idx.get(rid, 10**9))
        anchor_utt_id = response_ids[0]
        anchor_idx = id_to_idx.get(anchor_utt_id)
        if anchor_idx is None:
            continue

        preferred_cue_utt_id = unit.get("semantic_cue_utterance_id") or unit.get("cue_utterance_id")
        preferred_cue_idx = id_to_idx.get(preferred_cue_utt_id) if preferred_cue_utt_id else None
        if preferred_cue_idx is None:
            raw_idx = unit.get("semantic_cue_idx")
            if raw_idx is None:
                raw_idx = unit.get("cue_idx")
            try:
                preferred_cue_idx = int(raw_idx) if raw_idx is not None else None
            except Exception:
                preferred_cue_idx = None

        record: AnchorRecord = {
            "meeting_id": meeting_id,
            "participant_id": participant_id,
            "unit_id": stable_anchor_inventory_id(
                meeting_id,
                participant_id,
                anchor_utt_id,
                source_unit_id=unit.get("unit_id"),
                response_utterance_ids=response_ids,
                topic_summary=unit.get("topic_summary"),
            ),
            "selector_unit_id": unit.get("unit_id"),
            "topic_summary": unit.get("topic_summary"),
            "anchor_reason": unit.get("anchor_reason"),
            "importance_score": unit.get("importance_score"),
            "confidence": unit.get("confidence"),
            "response_utterance_ids": response_ids,
            "first_response_idx": anchor_idx,
            "anchor_utt_id": anchor_utt_id,
            "preferred_cue_utterance_id": preferred_cue_utt_id,
            "preferred_cue_idx": preferred_cue_idx,
            "preferred_cue_type": unit.get("semantic_cue_type") or unit.get("cue_type"),
            "preferred_cue_gap": (anchor_idx - preferred_cue_idx) if preferred_cue_idx is not None else None,
            "inventory_reference_k": int(inventory_reference_k) if inventory_reference_k is not None else None,
            "selector": selector,
        }
        key = anchor_record_key(record)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        records.append(record)

    records.sort(key=lambda record: (record.get("first_response_idx", 10**9), record.get("anchor_utt_id") or ""))
    return records


def write_anchor_inventory_records(path: Path, records: Sequence[AnchorRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(list(records), handle, indent=2, ensure_ascii=False)


def load_anchor_inventory_records(path: Path) -> List[AnchorRecord]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Anchor inventory must be a JSON list: {path}")
    return [dict(item) for item in payload if isinstance(item, dict)]


def _anchor_record_unit_signature(record: Dict[str, Any]) -> str:
    for field_name in ("selector_unit_id", "unit_id", "contribution_id", "test_case_id"):
        value = str(record.get(field_name) or "").strip()
        if value:
            return value
    signature_payload = {
        "response_utterance_ids": list(record.get("response_utterance_ids") or []),
        "topic_summary": str(record.get("topic_summary") or "").strip(),
        "anchor_reason": str(record.get("anchor_reason") or "").strip(),
    }
    digest = hashlib.sha1(
        json.dumps(signature_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:10]
    return f"h{digest}"


def anchor_record_key(record: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(record.get("participant_id") or "").strip(),
        str(record.get("anchor_utt_id") or "").strip(),
        _anchor_record_unit_signature(record),
    )


def anchor_keyset(records: Iterable[Dict[str, Any]]) -> set[Tuple[str, str, str]]:
    return {anchor_record_key(record) for record in records if all(anchor_record_key(record))}


def jaccard_overlap(
    left: Iterable[Dict[str, Any]],
    right: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    left_keys = anchor_keyset(left)
    right_keys = anchor_keyset(right)
    union = left_keys | right_keys
    intersection = left_keys & right_keys
    return {
        "left_count": len(left_keys),
        "right_count": len(right_keys),
        "intersection_count": len(intersection),
        "union_count": len(union),
        "jaccard": (len(intersection) / len(union)) if union else 1.0,
    }


def rebuild_episode_from_anchor_record(
    *,
    record: AnchorRecord,
    transcript: List[Dict[str, Any]],
    represented_user: str,
    N: int,
    k: int,
) -> Dict[str, Any]:
    anchor_idx = int(record.get("first_response_idx"))
    preferred_cue_idx = record.get("preferred_cue_idx")
    preferred_cue_gap = record.get("preferred_cue_gap")
    cue_mode = "retained"

    if preferred_cue_idx is None:
        cue_mode = "self_initiation"
    else:
        try:
            preferred_cue_idx = int(preferred_cue_idx)
        except Exception:
            preferred_cue_idx = None
            cue_mode = "self_initiation"

    if preferred_cue_idx is None or preferred_cue_idx < 0 or preferred_cue_idx >= anchor_idx:
        cue_mode = "self_initiation"

    if cue_mode == "retained":
        try:
            preferred_cue_gap = int(preferred_cue_gap) if preferred_cue_gap is not None else (anchor_idx - preferred_cue_idx)
        except Exception:
            preferred_cue_gap = anchor_idx - preferred_cue_idx
        if preferred_cue_gap > int(k):
            cue_mode = "self_initiation"

    if cue_mode == "retained":
        cue_idx = preferred_cue_idx
        cue_uid = record.get("preferred_cue_utterance_id")
        cue_type = record.get("preferred_cue_type") or "self-initiation"
    else:
        cue_idx = anchor_idx - 1
        if cue_idx < 0 or cue_idx >= len(transcript):
            return {
                "status": "dropped",
                "drop_reason": "no_valid_fallback_cue",
                "cue_mode": "dropped",
                "preferred_cue_gap": preferred_cue_gap,
                "effective_cue_gap": None,
                "episode": None,
                "idea_unit": None,
            }
        cue_uid = transcript[cue_idx].get("utterance_id")
        cue_type = "self-initiation"

    contrib = {
        "unit_id": record.get("selector_unit_id") or record.get("unit_id"),
        "anchor_inventory_id": record.get("unit_id"),
        "_episode_rank": record.get("_episode_rank"),
        "topic_summary": record.get("topic_summary"),
        "anchor_reason": record.get("anchor_reason"),
        "importance_score": record.get("importance_score"),
        "confidence": record.get("confidence"),
        "response_utterance_ids": list(record.get("response_utterance_ids") or []),
        "first_response_idx": anchor_idx,
        "anchor_utt_id": record.get("anchor_utt_id"),
        "cue_utterance_id": cue_uid,
        "cue_idx": cue_idx,
        "cue_type": cue_type,
    }
    episode = build_episode_from_contrib(
        contrib=contrib,
        transcript=transcript,
        represented_user=represented_user,
        N=N,
        k_used=int(k),
    )
    try:
        idea_unit = IdeaUnit(
            unit_id=str(record.get("unit_id") or record.get("anchor_utt_id")),
            topic_summary=str(record.get("topic_summary") or ""),
            response_utterance_ids=list(record.get("response_utterance_ids") or []),
            first_response_idx=anchor_idx,
            cue_utterance_id=str(cue_uid),
            cue_idx=int(cue_idx),
            cue_type=str(cue_type),
            anchor_reason=record.get("anchor_reason"),
            importance_score=record.get("importance_score"),
            confidence=record.get("confidence"),
            anchor_utt_id=str(record.get("anchor_utt_id") or ""),
        )
        episode.idea_units = [idea_unit]
    except Exception:
        idea_unit = None

    episode.rebuilt_from_anchor_inventory = True
    episode.anchor_inventory_id = str(record.get("unit_id") or record.get("anchor_utt_id") or "")
    episode.anchor_inventory_reference_k = (
        int(record.get("inventory_reference_k"))
        if record.get("inventory_reference_k") is not None
        else None
    )
    episode.preferred_cue_utt_id = record.get("preferred_cue_utterance_id")
    episode.preferred_cue_index = preferred_cue_idx
    episode.preferred_cue_type = record.get("preferred_cue_type")
    episode.preferred_cue_gap = preferred_cue_gap
    episode.preferred_cue_retained = cue_mode == "retained"
    episode.self_initiation_fallback = cue_mode == "self_initiation"
    episode.cue_mode = cue_mode

    present, contentful, _ = has_future_rep(episode.gt_next_window or [], represented_user)
    if not (present and contentful):
        return {
            "status": "dropped",
            "drop_reason": "not_present_or_not_contentful",
            "cue_mode": cue_mode,
            "preferred_cue_gap": preferred_cue_gap,
            "effective_cue_gap": anchor_idx - cue_idx,
            "episode": None,
            "idea_unit": idea_unit.model_dump() if idea_unit else None,
        }

    return {
        "status": "kept",
        "drop_reason": None,
        "cue_mode": cue_mode,
        "preferred_cue_gap": preferred_cue_gap,
        "effective_cue_gap": anchor_idx - cue_idx,
        "episode": episode,
        "idea_unit": idea_unit.model_dump() if idea_unit else None,
    }


def summarize_k_diagnostics(
    *,
    participant_id: str,
    meeting_id: str,
    k: int,
    frozen_anchor_count: int,
    diagnostics: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    kept = [diag for diag in diagnostics if diag.get("status") == "kept"]
    dropped = [diag for diag in diagnostics if diag.get("status") != "kept"]
    retained = [diag for diag in kept if diag.get("cue_mode") == "retained"]
    selector_reanchored = [diag for diag in kept if diag.get("cue_mode") == "selector_reanchor"]
    deterministic_reanchored = [diag for diag in kept if diag.get("cue_mode") == "deterministic_reanchor"]
    fallback = [diag for diag in kept if diag.get("cue_mode") == "self_initiation"]
    cap_dropped = [diag for diag in dropped if diag.get("drop_reason") == "cap_per_participant"]
    low_substance = [diag for diag in dropped if diag.get("drop_reason") == "low_substance"]
    context_covered = [diag for diag in dropped if diag.get("drop_reason") == "context_already_covered"]
    self_init_invalid = [diag for diag in dropped if diag.get("drop_reason") == "self_init_anchor_invalid"]
    unavailable = [diag for diag in dropped if diag.get("drop_reason") == "target_unavailable_to_delegate"]
    response_outside_window = [diag for diag in dropped if diag.get("drop_reason") == "response_outside_window"]
    trimmed_response = [diag for diag in diagnostics if diag.get("trimmed_response_outside_window")]
    expanded_response = [diag for diag in diagnostics if diag.get("expanded_response_utterance_ids")]
    summary_rewritten = [diag for diag in diagnostics if diag.get("summary_rewritten")]
    canonicalized_target = [diag for diag in diagnostics if diag.get("canonical_target_proposition")]
    preferred_gaps = [float(diag["preferred_cue_gap"]) for diag in diagnostics if diag.get("preferred_cue_gap") is not None]
    effective_gaps = [float(diag["effective_cue_gap"]) for diag in kept if diag.get("effective_cue_gap") is not None]
    total_steps = 0
    for diag in kept:
        episode = diag.get("episode")
        if episode is not None:
            total_steps += int(episode.k)
            continue
        if diag.get("episode_k") is not None:
            total_steps += int(diag.get("episode_k") or 0)

    def _rate(count: int) -> float:
        return (count / frozen_anchor_count) if frozen_anchor_count else 0.0

    return {
        "meeting_id": meeting_id,
        "participant_id": participant_id,
        "k": int(k),
        "frozen_anchor_count": int(frozen_anchor_count),
        "kept_episode_count": len(kept),
        "dropped_anchor_count": len(dropped),
        "preferred_cue_retained_count": len(retained),
        "preferred_cue_retained_rate": _rate(len(retained)),
        "selector_reanchor_count": len(selector_reanchored),
        "selector_reanchor_rate": _rate(len(selector_reanchored)),
        "deterministic_reanchor_count": len(deterministic_reanchored),
        "deterministic_reanchor_rate": _rate(len(deterministic_reanchored)),
        "self_initiation_fallback_count": len(fallback),
        "self_initiation_fallback_rate": _rate(len(fallback)),
        "cap_dropped_count": len(cap_dropped),
        "cap_dropped_rate": _rate(len(cap_dropped)),
        "dropped_low_substance": len(low_substance),
        "dropped_context_already_covered": len(context_covered),
        "dropped_self_init_anchor_invalid": len(self_init_invalid),
        "dropped_target_unavailable_to_delegate": len(unavailable),
        "dropped_response_outside_window": len(response_outside_window),
        "trimmed_response_outside_window": len(trimmed_response),
        "response_constituent_expanded_count": len(expanded_response),
        "summary_rewritten_count": len(summary_rewritten),
        "canonicalized_target_count": len(canonicalized_target),
        "anchor_drop_rate": _rate(len(dropped)),
        "mean_preferred_cue_gap": (sum(preferred_gaps) / len(preferred_gaps)) if preferred_gaps else None,
        "mean_effective_cue_gap": (sum(effective_gaps) / len(effective_gaps)) if effective_gaps else None,
        "total_evaluated_steps": int(total_steps),
        "cue_mode_counts": _count_by_value(diag.get("cue_mode") for diag in kept),
        "drop_reasons": _count_by_value(diag.get("drop_reason") for diag in dropped),
    }


def summarize_meeting_k_diagnostics(
    *,
    meeting_id: str,
    k: int,
    inventory_reference_k: Optional[int],
    participant_summaries: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    frozen = sum(int(summary.get("frozen_anchor_count") or 0) for summary in participant_summaries)
    kept = sum(int(summary.get("kept_episode_count") or 0) for summary in participant_summaries)
    retained = sum(int(summary.get("preferred_cue_retained_count") or 0) for summary in participant_summaries)
    selector_reanchored = sum(int(summary.get("selector_reanchor_count") or 0) for summary in participant_summaries)
    deterministic_reanchored = sum(int(summary.get("deterministic_reanchor_count") or 0) for summary in participant_summaries)
    fallback = sum(int(summary.get("self_initiation_fallback_count") or 0) for summary in participant_summaries)
    cap_dropped = sum(int(summary.get("cap_dropped_count") or 0) for summary in participant_summaries)
    low_substance = sum(int(summary.get("dropped_low_substance") or 0) for summary in participant_summaries)
    context_covered = sum(int(summary.get("dropped_context_already_covered") or 0) for summary in participant_summaries)
    self_init_invalid = sum(int(summary.get("dropped_self_init_anchor_invalid") or 0) for summary in participant_summaries)
    unavailable = sum(int(summary.get("dropped_target_unavailable_to_delegate") or 0) for summary in participant_summaries)
    response_outside_window = sum(int(summary.get("dropped_response_outside_window") or 0) for summary in participant_summaries)
    trimmed_response = sum(int(summary.get("trimmed_response_outside_window") or 0) for summary in participant_summaries)
    expanded_response = sum(int(summary.get("response_constituent_expanded_count") or 0) for summary in participant_summaries)
    summary_rewritten = sum(int(summary.get("summary_rewritten_count") or 0) for summary in participant_summaries)
    canonicalized_target = sum(int(summary.get("canonicalized_target_count") or 0) for summary in participant_summaries)
    dropped = sum(int(summary.get("dropped_anchor_count") or 0) for summary in participant_summaries)
    total_steps = sum(int(summary.get("total_evaluated_steps") or 0) for summary in participant_summaries)
    preferred_gap_values = [float(summary["mean_preferred_cue_gap"]) for summary in participant_summaries if summary.get("mean_preferred_cue_gap") is not None]
    effective_gap_values = [float(summary["mean_effective_cue_gap"]) for summary in participant_summaries if summary.get("mean_effective_cue_gap") is not None]

    return {
        "meeting_id": meeting_id,
        "k": int(k),
        "inventory_reference_k": int(inventory_reference_k) if inventory_reference_k is not None else None,
        "participant_count": len(participant_summaries),
        "frozen_anchor_count": frozen,
        "kept_episode_count": kept,
        "dropped_anchor_count": dropped,
        "preferred_cue_retained_count": retained,
        "preferred_cue_retained_rate": (retained / frozen) if frozen else 0.0,
        "selector_reanchor_count": selector_reanchored,
        "selector_reanchor_rate": (selector_reanchored / frozen) if frozen else 0.0,
        "deterministic_reanchor_count": deterministic_reanchored,
        "deterministic_reanchor_rate": (deterministic_reanchored / frozen) if frozen else 0.0,
        "self_initiation_fallback_count": fallback,
        "self_initiation_fallback_rate": (fallback / frozen) if frozen else 0.0,
        "cap_dropped_count": cap_dropped,
        "cap_dropped_rate": (cap_dropped / frozen) if frozen else 0.0,
        "dropped_low_substance": low_substance,
        "dropped_context_already_covered": context_covered,
        "dropped_self_init_anchor_invalid": self_init_invalid,
        "dropped_target_unavailable_to_delegate": unavailable,
        "dropped_response_outside_window": response_outside_window,
        "trimmed_response_outside_window": trimmed_response,
        "response_constituent_expanded_count": expanded_response,
        "summary_rewritten_count": summary_rewritten,
        "canonicalized_target_count": canonicalized_target,
        "anchor_drop_rate": (dropped / frozen) if frozen else 0.0,
        "mean_preferred_cue_gap": (sum(preferred_gap_values) / len(preferred_gap_values)) if preferred_gap_values else None,
        "mean_effective_cue_gap": (sum(effective_gap_values) / len(effective_gap_values)) if effective_gap_values else None,
        "total_evaluated_steps": total_steps,
        "cue_mode_counts": _count_by_value(
            key
            for summary in participant_summaries
            for key, count in dict(summary.get("cue_mode_counts") or {}).items()
            for _ in range(int(count or 0))
        ),
        "participants": list(participant_summaries),
    }


def _count_by_value(values: Iterable[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


__all__ = [
    "anchor_inventory_filename",
    "freeze_anchor_inventory_records",
    "jaccard_overlap",
    "k_diagnostics_filename",
    "load_anchor_inventory_records",
    "rebuild_episode_from_anchor_record",
    "stable_anchor_inventory_id",
    "summarize_k_diagnostics",
    "summarize_meeting_k_diagnostics",
    "write_anchor_inventory_records",
]
