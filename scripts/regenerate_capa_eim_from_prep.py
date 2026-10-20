#!/usr/bin/env python
"""Copy finalized CAPA prep and regenerate only CAPA EIM states with a model.

This is useful for backbone checks where official episodes, profiles, and JIT
briefings should stay fixed, but model-visible EIM snapshots should be produced
by the backbone under evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from capa.episode_precompute import precompute_eim_for_episodes  # noqa: E402
from config import Config  # noqa: E402
from environment_interpreter import EnvironmentInterpretationModule  # noqa: E402
from llm_client import LLMClient  # noqa: E402
from llm_telemetry import make_telemetry_path  # noqa: E402
from schemas import CAPAEpisode  # noqa: E402


PREP_SUBDIRS_TO_COPY = (
    "user_profiles",
    "jit_briefings",
    "capa_anchor_inventory",
    "capa_k_diagnostics",
    "capa_episodes",
)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _transcript_to_list(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("utterances", "transcript"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _participant_slug(participant: str) -> str:
    return participant.replace(" ", "_")


def _discover_participants(source_prep_dir: Path, requested: Optional[Sequence[str]]) -> List[str]:
    if requested:
        return [item.strip() for item in requested if item.strip()]
    episode_dir = source_prep_dir / "capa_episodes"
    participants: List[str] = []
    for path in sorted(episode_dir.glob("capa_episodes_for_*_from_*.json")):
        payload = _read_json(path)
        episodes = payload if isinstance(payload, list) else payload.get("episodes", [])
        participant = None
        for episode in episodes or []:
            if isinstance(episode, dict) and episode.get("represented_user"):
                participant = str(episode["represented_user"])
                break
        if participant is None:
            participant = path.name.split("capa_episodes_for_", 1)[1].split("_from_", 1)[0].replace("_", " ")
        if participant not in participants:
            participants.append(participant)
    return participants


def _copy_fixed_prep(source_prep_dir: Path, output_prep_dir: Path) -> None:
    if source_prep_dir.resolve() == output_prep_dir.resolve():
        raise ValueError("source prep and output prep must be different directories")
    output_prep_dir.mkdir(parents=True, exist_ok=True)
    for subdir in PREP_SUBDIRS_TO_COPY:
        src = source_prep_dir / subdir
        dst = output_prep_dir / subdir
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)


def _safe_clear_eim_dir(output_prep_dir: Path) -> Path:
    eim_dir = output_prep_dir / "capa_eim_states"
    resolved_output = output_prep_dir.resolve()
    resolved_eim = eim_dir.resolve() if eim_dir.exists() else eim_dir.absolute()
    if resolved_output not in resolved_eim.parents and resolved_eim != resolved_output:
        raise ValueError(f"Refusing to clear EIM dir outside output prep: {resolved_eim}")
    if eim_dir.exists():
        shutil.rmtree(eim_dir)
    eim_dir.mkdir(parents=True, exist_ok=True)
    return eim_dir


def _load_participant_episodes(output_prep_dir: Path, meeting_id: str, participant: str) -> List[CAPAEpisode]:
    path = (
        output_prep_dir
        / "capa_episodes"
        / f"capa_episodes_for_{_participant_slug(participant)}_from_{meeting_id}.json"
    )
    payload = _read_json(path)
    raw_episodes = payload if isinstance(payload, list) else payload.get("episodes", [])
    return [CAPAEpisode.model_validate(item) for item in raw_episodes or []]


def _context_hash(episode: CAPAEpisode) -> str:
    context_ids = [
        str(utterance.get("utterance_id") or "")
        for utterance in (episode.context_last_N or [])
        if isinstance(utterance, dict)
    ]
    return hashlib.sha256("|".join(context_ids).encode("utf-8")).hexdigest()[:16]


def regenerate_eim_states(
    *,
    transcript: List[Dict[str, Any]],
    output_prep_dir: Path,
    meeting_id: str,
    participants: Sequence[str],
    client: LLMClient,
) -> Dict[str, Any]:
    eim_dir = _safe_clear_eim_dir(output_prep_dir)
    episodes_by_participant: Dict[str, List[CAPAEpisode]] = {}
    unique_episodes: Dict[str, CAPAEpisode] = {}
    consumers: Dict[str, List[Tuple[str, str]]] = {}

    for participant in participants:
        episodes = _load_participant_episodes(output_prep_dir, meeting_id, participant)
        episodes_by_participant[participant] = episodes
        for episode in episodes:
            key = f"{episode.cue_utt_id}__{_context_hash(episode)}"
            if key not in unique_episodes:
                unique_episodes[key] = CAPAEpisode(
                    meeting_id=episode.meeting_id,
                    represented_user=episode.represented_user,
                    cue_utt_id=episode.cue_utt_id,
                    cue_index=episode.cue_index,
                    N=episode.N,
                    k=episode.k,
                    context_last_N=episode.context_last_N,
                    gt_next_window=episode.gt_next_window,
                    test_case_id=f"UKEY:{key}",
                )
                consumers[key] = []
            consumers[key].append((participant, episode.test_case_id))

    eim = EnvironmentInterpretationModule(client)
    precomputed = precompute_eim_for_episodes(
        transcript=transcript,
        episodes=list(unique_episodes.values()),
        eim=eim,
    )
    ukey_to_output: Dict[str, Dict[str, Any]] = {}
    for entry in precomputed:
        test_case_id = str(entry.get("test_case_id") or "")
        if test_case_id.startswith("UKEY:") and entry.get("eim_output"):
            ukey_to_output[test_case_id.split("UKEY:", 1)[1]] = entry["eim_output"]

    counts: Dict[str, int] = {}
    missing: List[Dict[str, str]] = []
    for participant in participants:
        entries: List[Dict[str, Any]] = []
        for episode in episodes_by_participant.get(participant, []):
            key = f"{episode.cue_utt_id}__{_context_hash(episode)}"
            eim_output = ukey_to_output.get(key)
            if eim_output is None:
                missing.append({"participant": participant, "test_case_id": str(episode.test_case_id)})
                continue
            entries.append({"test_case_id": episode.test_case_id, "eim_output": eim_output})
        out_path = eim_dir / f"capa_eim_states_for_{_participant_slug(participant)}_in_{meeting_id}.json"
        _write_json(out_path, entries)
        counts[participant] = len(entries)

    return {
        "participant_state_counts": counts,
        "unique_window_count": len(unique_episodes),
        "missing_state_count": len(missing),
        "missing_states": missing[:20],
    }


def _parse_participants(raw: str) -> Optional[List[str]]:
    if not raw or raw.lower() == "all":
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Parsed meeting transcript JSON")
    parser.add_argument("--source-prep-dir", required=True, type=Path, help="Final prep dir to copy fixed material from")
    parser.add_argument("--output-prep-dir", required=True, type=Path, help="Prep dir to write copied fixed prep plus regenerated EIM")
    parser.add_argument("--meeting-id", default=None)
    parser.add_argument("--participants", default="all")
    parser.add_argument("--model", default="gemini", help="Model alias from config_extended.json")
    parser.add_argument("--disable-llm-telemetry", action="store_true")
    args = parser.parse_args(argv)

    os.environ["MD_ACTIVE_MODEL"] = args.model
    meeting_id = args.meeting_id or args.input.stem.replace("_parsed", "")
    source_prep_dir = args.source_prep_dir.resolve()
    output_prep_dir = args.output_prep_dir.resolve()
    _copy_fixed_prep(source_prep_dir, output_prep_dir)

    participants = _discover_participants(source_prep_dir, _parse_participants(args.participants))
    if not participants:
        raise SystemExit("No participants found in source prep")

    transcript = _transcript_to_list(_read_json(args.input.resolve()))
    if not transcript:
        raise SystemExit(f"Unsupported or empty transcript: {args.input}")

    config = Config()
    client = LLMClient(config)
    if not args.disable_llm_telemetry:
        telemetry_path = make_telemetry_path(output_prep_dir, stage="eim_regeneration")
        client.configure_telemetry(
            telemetry_path,
            base_context={
                "stage": "eim_regeneration",
                "meeting_id": meeting_id,
                "source_prep_dir": str(source_prep_dir),
            },
        )
        print(f"[LLM TELEMETRY] Writing {telemetry_path}")
    result = regenerate_eim_states(
        transcript=transcript,
        output_prep_dir=output_prep_dir,
        meeting_id=meeting_id,
        participants=participants,
        client=client,
    )
    manifest = {
        "created_at": datetime.now().isoformat(),
        "meeting_id": meeting_id,
        "source_prep_dir": str(source_prep_dir),
        "output_prep_dir": str(output_prep_dir),
        "fixed_material": list(PREP_SUBDIRS_TO_COPY),
        "regenerated_material": ["capa_eim_states"],
        "model_descriptor": {
            "active_model_name": config.active_model_name,
            "provider": config.provider,
            "model": config.model,
        },
        "participants": list(participants),
        **result,
    }
    _write_json(output_prep_dir / "model_specific_eim_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0 if result["missing_state_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
