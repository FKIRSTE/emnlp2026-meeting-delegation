#!/usr/bin/env python
"""Freeze finalized CAPA episodes into anchor inventories for N-window studies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from capa.anchor_inventory import (  # noqa: E402
    anchor_inventory_filename,
    anchor_record_key,
    stable_anchor_inventory_id,
    write_anchor_inventory_records,
)


DEFAULT_SELECTOR = "frozen_from_final_official_episodes"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _discover_meeting_ids(input_dir: Path, meetings: Optional[Sequence[str]] = None) -> List[str]:
    selected = {item.strip() for item in meetings or [] if item.strip()}
    meeting_ids = sorted(
        path.stem[:-7] if path.stem.endswith("_parsed") else path.stem
        for path in input_dir.glob("*_parsed.json")
    )
    if selected:
        meeting_ids = [meeting_id for meeting_id in meeting_ids if meeting_id in selected]
    return meeting_ids


def _participant_from_episode_path(path: Path, episodes: Sequence[Dict[str, Any]]) -> str:
    for episode in episodes:
        participant = str(episode.get("represented_user") or "").strip()
        if participant:
            return participant
    name = path.name
    if name.startswith("capa_episodes_for_") and "_from_" in name:
        return name.split("capa_episodes_for_", 1)[1].split("_from_", 1)[0].replace("_", " ")
    raise ValueError(f"Cannot infer participant from episode file: {path}")


def _episode_list(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("episodes", "capa_episodes"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value if isinstance(item, dict)]
    raise ValueError("Episode artifact must be a JSON list or an object containing an episodes list")


def _first_idea_unit(episode: Dict[str, Any]) -> Dict[str, Any]:
    idea_units = episode.get("idea_units") or episode.get("idea_unit") or []
    if isinstance(idea_units, dict):
        return dict(idea_units)
    if isinstance(idea_units, list) and idea_units:
        first = idea_units[0]
        if isinstance(first, dict):
            return dict(first)
    return {}


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _derive_first_response_idx(
    *,
    episode: Dict[str, Any],
    idea_unit: Dict[str, Any],
    response_ids: Sequence[str],
) -> Optional[int]:
    raw_idx = _coerce_int(idea_unit.get("first_response_idx"))
    if raw_idx is not None:
        return raw_idx
    cue_idx = _coerce_int(episode.get("cue_index"))
    if cue_idx is None:
        return None
    gt_window = episode.get("gt_next_window") or []
    for offset, utterance in enumerate(gt_window, start=1):
        if not isinstance(utterance, dict):
            continue
        if str(utterance.get("utterance_id") or "") in set(response_ids):
            return cue_idx + offset
    return None


def _canonical_response_ids(episode: Dict[str, Any], idea_unit: Dict[str, Any]) -> List[str]:
    raw_ids = (
        idea_unit.get("response_utterance_ids")
        or episode.get("response_utterance_ids")
        or []
    )
    ids = [str(item).strip() for item in raw_ids if str(item or "").strip()]
    return list(dict.fromkeys(ids))


def episode_to_anchor_record(
    *,
    episode: Dict[str, Any],
    participant_id: str,
    source_meeting_id: str,
    inventory_reference_k: int,
    selector: str = DEFAULT_SELECTOR,
) -> Dict[str, Any]:
    idea_unit = _first_idea_unit(episode)
    response_ids = _canonical_response_ids(episode, idea_unit)
    if not response_ids:
        raise ValueError(f"Episode has no response_utterance_ids: {episode.get('test_case_id')}")

    first_response_idx = _derive_first_response_idx(
        episode=episode,
        idea_unit=idea_unit,
        response_ids=response_ids,
    )
    if first_response_idx is None:
        raise ValueError(f"Cannot derive first_response_idx: {episode.get('test_case_id')}")

    anchor_utt_id = (
        idea_unit.get("anchor_utt_id")
        or episode.get("anchor_utt_id")
        or response_ids[0]
    )
    cue_idx = _coerce_int(episode.get("cue_index"))
    if cue_idx is None:
        cue_idx = _coerce_int(idea_unit.get("cue_idx"))
    preferred_cue_utt_id = episode.get("cue_utt_id") or idea_unit.get("cue_utterance_id")
    if cue_idx is None or not preferred_cue_utt_id:
        raise ValueError(f"Cannot derive cue boundary: {episode.get('test_case_id')}")

    selector_unit_id = (
        idea_unit.get("unit_id")
        or episode.get("episode_id_suffix")
        or episode.get("test_case_id")
    )
    record_unit_id = stable_anchor_inventory_id(
        source_meeting_id,
        participant_id,
        str(anchor_utt_id),
        source_unit_id=str(selector_unit_id or ""),
        response_utterance_ids=response_ids,
        topic_summary=idea_unit.get("topic_summary") or episode.get("selection_reason"),
    )
    preferred_gap = first_response_idx - cue_idx

    return {
        "meeting_id": source_meeting_id,
        "participant_id": participant_id,
        "unit_id": record_unit_id,
        "selector_unit_id": selector_unit_id,
        "source_test_case_id": episode.get("test_case_id"),
        "source_cue_test_case_id": episode.get("cue_test_case_id"),
        "source_episode_id_suffix": episode.get("episode_id_suffix"),
        "topic_summary": idea_unit.get("topic_summary") or episode.get("selection_reason"),
        "anchor_reason": idea_unit.get("anchor_reason") or episode.get("selection_reason"),
        "importance_score": idea_unit.get("importance_score") or episode.get("selection_confidence"),
        "confidence": idea_unit.get("confidence"),
        "response_utterance_ids": response_ids,
        "first_response_idx": first_response_idx,
        "anchor_utt_id": str(anchor_utt_id),
        "preferred_cue_utterance_id": preferred_cue_utt_id,
        "preferred_cue_idx": cue_idx,
        "preferred_cue_type": idea_unit.get("cue_type") or episode.get("cue_mode"),
        "preferred_cue_gap": preferred_gap,
        "inventory_reference_k": int(inventory_reference_k),
        "selector": selector,
        "official_target_text": idea_unit.get("official_target_text"),
        "canonical_target_proposition": idea_unit.get("canonical_target_proposition"),
        "tested_facet_text": idea_unit.get("tested_facet_text"),
        "selector_response_utterance_ids": idea_unit.get("selector_response_utterance_ids"),
        "trimmed_response_utterance_ids": idea_unit.get("trimmed_response_utterance_ids"),
        "expanded_response_utterance_ids": idea_unit.get("expanded_response_utterance_ids"),
        "summary_rewritten": idea_unit.get("summary_rewritten"),
        "summary_rewrite_reason": idea_unit.get("summary_rewrite_reason"),
        "official_constituent_expansion_reason": idea_unit.get("official_constituent_expansion_reason"),
    }


def _episode_files_for_meeting(source_prep_root: Path, meeting_id: str) -> List[Path]:
    ep_dir = source_prep_root / meeting_id / "capa_episodes"
    if not ep_dir.exists():
        raise FileNotFoundError(f"Missing CAPA episode directory: {ep_dir}")
    return sorted(ep_dir.glob("capa_episodes_for_*_from_*.json"))


def _freeze_meeting(
    *,
    source_prep_root: Path,
    out_root: Path,
    meeting_id: str,
    inventory_reference_k: int,
    selector: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    meeting_out = out_root / meeting_id
    inventory_dir = meeting_out / "capa_anchor_inventory"
    participant_summaries: List[Dict[str, Any]] = []
    meeting_records = 0
    duplicate_keys: List[Tuple[str, str, str]] = []

    for episode_path in _episode_files_for_meeting(source_prep_root, meeting_id):
        episodes = _episode_list(_read_json(episode_path))
        participant_id = _participant_from_episode_path(episode_path, episodes)
        records: List[Dict[str, Any]] = []
        seen = set()
        for episode in episodes:
            record = episode_to_anchor_record(
                episode=episode,
                participant_id=participant_id,
                source_meeting_id=meeting_id,
                inventory_reference_k=inventory_reference_k,
                selector=selector,
            )
            key = anchor_record_key(record)
            if key in seen:
                duplicate_keys.append(key)
            seen.add(key)
            records.append(record)
        records.sort(
            key=lambda item: (
                int(item.get("first_response_idx") or 10**9),
                str(item.get("anchor_utt_id") or ""),
                str(item.get("selector_unit_id") or item.get("unit_id") or ""),
            )
        )
        meeting_records += len(records)
        inventory_path = inventory_dir / anchor_inventory_filename(participant_id, meeting_id)
        if not dry_run:
            write_anchor_inventory_records(inventory_path, records)
        participant_summaries.append({
            "participant_id": participant_id,
            "episode_file": str(episode_path),
            "anchor_inventory_file": str(inventory_path),
            "source_episode_count": len(episodes),
            "frozen_anchor_count": len(records),
        })

    summary = {
        "meeting_id": meeting_id,
        "source_prep_root": str(source_prep_root),
        "out_root": str(out_root),
        "inventory_reference_k": int(inventory_reference_k),
        "selector": selector,
        "source_episode_count": sum(item["source_episode_count"] for item in participant_summaries),
        "frozen_anchor_count": meeting_records,
        "duplicate_anchor_key_count": len(duplicate_keys),
        "participants": participant_summaries,
    }
    if not dry_run:
        _write_json(meeting_out / "freeze_manifest.json", summary)
    return summary


def freeze_anchor_inventories(
    *,
    input_dir: Path,
    source_prep_root: Path,
    out_root: Path,
    meetings: Optional[Sequence[str]] = None,
    inventory_reference_k: int = 5,
    selector: str = DEFAULT_SELECTOR,
    dry_run: bool = False,
) -> Dict[str, Any]:
    meeting_ids = _discover_meeting_ids(input_dir, meetings)
    meeting_summaries = [
        _freeze_meeting(
            source_prep_root=source_prep_root,
            out_root=out_root,
            meeting_id=meeting_id,
            inventory_reference_k=inventory_reference_k,
            selector=selector,
            dry_run=dry_run,
        )
        for meeting_id in meeting_ids
    ]
    summary = {
        "input_dir": str(input_dir),
        "source_prep_root": str(source_prep_root),
        "out_root": str(out_root),
        "meeting_count": len(meeting_summaries),
        "source_episode_count": sum(item["source_episode_count"] for item in meeting_summaries),
        "frozen_anchor_count": sum(item["frozen_anchor_count"] for item in meeting_summaries),
        "inventory_reference_k": int(inventory_reference_k),
        "selector": selector,
        "meetings": meeting_summaries,
    }
    if not dry_run:
        _write_json(out_root / "freeze_summary.json", summary)
    return summary


def _parse_meetings(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path, help="AMI parsed input directory")
    parser.add_argument("--source-prep-root", required=True, type=Path, help="Final shared prep root to freeze from")
    parser.add_argument("--out-root", required=True, type=Path, help="Output root for frozen anchor inventories")
    parser.add_argument("--meetings", default=None, help="Optional comma-separated meeting IDs")
    parser.add_argument("--inventory-reference-k", type=int, default=5)
    parser.add_argument("--selector", default=DEFAULT_SELECTOR)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = freeze_anchor_inventories(
        input_dir=args.input_dir.resolve(),
        source_prep_root=args.source_prep_root.resolve(),
        out_root=args.out_root.resolve(),
        meetings=_parse_meetings(args.meetings),
        inventory_reference_k=args.inventory_reference_k,
        selector=args.selector,
        dry_run=args.dry_run,
    )
    print(json.dumps({
        "meeting_count": summary["meeting_count"],
        "source_episode_count": summary["source_episode_count"],
        "frozen_anchor_count": summary["frozen_anchor_count"],
        "out_root": summary["out_root"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
