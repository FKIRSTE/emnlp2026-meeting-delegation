from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field
from capa.episode_generator import generate_dense_episodes_multi_participant, write_episodes_to_file
from capa.episode_utils import (
    assert_unique_episode_ids,
    assess_target_availability_stage1,
    assess_episode_quality,
    build_delegate_inventory_text,
    build_delegate_inventory_evidence,
    canonicalize_official_target,
    enforce_official_response_window,
    is_banal,
    make_cue_test_case_id,
    normalize_selector_item,
    refresh_episode_identity,
)
from environment_interpreter import EnvironmentInterpretationModule
from pipeline_resume import (
    PREP_MANIFEST_VERSION,
    PIPELINE_RUNTIME_VERSION,
    PROFILE_STAGE_VERSION,
    NORMALIZED_PROFILE_STAGE_VERSION,
    ANCHOR_INVENTORY_STAGE_VERSION,
    EPISODE_STAGE_VERSION,
    CAPA_EIM_STAGE_VERSION,
    JIT_STAGE_VERSION,
    atomic_write_json,
    file_lock,
    fingerprint_contract,
    fingerprint_file,
    fingerprint_payload,
    fingerprint_transcript_file,
    load_manifest,
    mark_unit_complete,
    mark_unit_failed,
    mark_unit_in_progress,
    save_manifest,
    summarize_manifest_state,
    unit_is_reusable,
)
from schemas import CAPAEpisode, IdeaUnit
from capa.anchor_inventory import (
    anchor_inventory_filename,
    freeze_anchor_inventory_records,
    k_diagnostics_filename,
    load_anchor_inventory_records,
    rebuild_episode_from_anchor_record,
    summarize_k_diagnostics,
    summarize_meeting_k_diagnostics,
    write_anchor_inventory_records,
)

EIM_CACHE_MANIFEST_VERSION = "eim_cache_manifest_v1"
EIM_CHECKPOINT_STRIDE = 50
EPISODE_NORMALIZATION_VERSION = "fixed_k_episode_normalization_v9_canonical_targets"
EIM_SNAPSHOT_POSTPROCESS_VERSION = "eim_snapshot_last_utterance_v2"


class TargetAvailabilityVerifierReport(BaseModel):
    verdict: Literal["supported", "testable_partial", "unsupported_generic", "unsupported_missing"] = Field(
        description="Whether the participant inventory supports the official target."
    )
    evidence_ids: List[str] = Field(default_factory=list, description="Inventory evidence IDs supporting the verdict.")
    missing_facets: List[str] = Field(default_factory=list, description="Concrete target facets absent from inventory.")
    rationale: str = Field(description="Short explanation for the verdict.")


def selector_shortlist_size(cap_per_participant: int) -> Optional[int]:
    cap = int(cap_per_participant or 0)
    if cap <= 0:
        return None
    return max(cap, min(cap + 5, 10))


def participant_slug(participant_id: str) -> str:
    return participant_id.replace(" ", "_")


def participant_unit_key(stage_name: str, participant_id: str) -> str:
    return f"{stage_name}::{participant_id}"


def safe_json_load(path: Path, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def stable_segment_fingerprint(segment: List[Dict[str, Any]]) -> str:
    utterance_ids = [utt.get("utterance_id") for utt in segment if isinstance(utt, dict)]
    return fingerprint_payload(utterance_ids)


def transcript_to_list(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("utterances", "transcript"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


def build_selector_diag_row(
    *,
    participant_id: str,
    meeting_id: str,
    selector_item: Dict[str, Any],
    status: str,
    drop_reason: Optional[str],
    episode: Optional[CAPAEpisode],
) -> Dict[str, Any]:
    anchor_idx = selector_item.get("first_response_idx")
    cue_idx = selector_item.get("cue_idx")
    effective_cue_gap = None
    if anchor_idx is not None and cue_idx is not None:
        try:
            effective_cue_gap = int(anchor_idx) - int(cue_idx)
        except Exception:
            effective_cue_gap = None

    cue_anchor_id = (
        episode.cue_test_case_id
        if episode is not None
        else (
            make_cue_test_case_id(meeting_id, participant_id, int(cue_idx))
            if cue_idx is not None
            else None
        )
    )
    return {
        "participant_id": participant_id,
        "meeting_id": meeting_id,
        "anchor_utt_id": selector_item.get("anchor_utt_id") or (selector_item.get("response_utterance_ids") or [None])[0],
        "unit_id": selector_item.get("unit_id") or selector_item.get("contribution_id"),
        "status": status,
        "drop_reason": drop_reason,
        "cue_mode": selector_item.get("cue_mode"),
        "importance_score": selector_item.get("importance_score"),
        "preferred_cue_gap": selector_item.get("preferred_cue_gap"),
        "effective_cue_gap": effective_cue_gap,
        "quality_status": selector_item.get("quality_status"),
        "quality_drop_reason": selector_item.get("quality_drop_reason"),
        "target_available_to_delegate": selector_item.get("target_available_to_delegate"),
        "target_availability_reason": selector_item.get("target_availability_reason"),
        "target_availability_score": selector_item.get("target_availability_score"),
        "target_availability_verdict": selector_item.get("target_availability_verdict"),
        "target_availability_method": selector_item.get("target_availability_method"),
        "target_availability_evidence_ids": selector_item.get("target_availability_evidence_ids"),
        "target_availability_missing_facets": selector_item.get("target_availability_missing_facets"),
        "target_availability_rationale": selector_item.get("target_availability_rationale"),
        "context_already_covered": selector_item.get("context_already_covered"),
        "context_coverage_overlap": selector_item.get("context_coverage_overlap"),
        "response_substance_score": selector_item.get("response_substance_score"),
        "selector_response_utterance_ids": selector_item.get("selector_response_utterance_ids"),
        "selector_topic_summary": selector_item.get("selector_topic_summary"),
        "trimmed_response_utterance_ids": selector_item.get("trimmed_response_utterance_ids"),
        "expanded_response_utterance_ids": selector_item.get("expanded_response_utterance_ids"),
        "official_constituent_expansion_reason": selector_item.get("official_constituent_expansion_reason"),
        "official_response_id_count": selector_item.get("official_response_id_count"),
        "official_target_text": selector_item.get("official_target_text"),
        "canonical_target_proposition": selector_item.get("canonical_target_proposition"),
        "tested_facet_text": selector_item.get("tested_facet_text"),
        "summary_rewritten": selector_item.get("summary_rewritten"),
        "summary_rewrite_reason": selector_item.get("summary_rewrite_reason"),
        "response_window_trimmed": selector_item.get("response_window_trimmed"),
        "trimmed_response_outside_window": selector_item.get("trimmed_response_outside_window"),
        "dropped_response_outside_window": selector_item.get("dropped_response_outside_window"),
        "response_ids_outside_window": selector_item.get("response_ids_outside_window"),
        "cap_rank": None,
        "test_case_id": episode.test_case_id if episode is not None else None,
        "cue_test_case_id": cue_anchor_id,
        "cue_anchor_id": cue_anchor_id,
        "episode_k": int(episode.k) if episode is not None else None,
    }


def enforce_episode_cap(
    *,
    participant_id: str,
    participant_episodes: List[CAPAEpisode],
    kept_diagnostics: List[Dict[str, Any]],
    cap_per_participant: int,
    debug_episodes: bool,
) -> List[CAPAEpisode]:
    cap = int(cap_per_participant or 0)
    if cap <= 0:
        return participant_episodes
    for rank, diag in enumerate(kept_diagnostics, start=1):
        diag["cap_rank"] = rank
    if len(participant_episodes) <= cap:
        return participant_episodes
    overflow_count = len(participant_episodes) - cap
    if debug_episodes:
        print(
            f"     [DBG] {participant_id}: dropping {overflow_count} kept episode(s) due to cap_per_participant={cap}",
            flush=True,
        )
    for diag in kept_diagnostics[cap:]:
        diag["status"] = "dropped"
        diag["drop_reason"] = "cap_per_participant"
    return participant_episodes[:cap]


def prep_paths_for_participant(output_root: Path, meeting_id: str, participant_id: str, k: int) -> Dict[str, Path]:
    slug = participant_slug(participant_id)
    return {
        "profile": output_root / "user_profiles" / f"capa_profile_for_{slug}_in_{meeting_id}.json",
        "briefing": output_root / "jit_briefings" / f"{slug}_briefing_from_{meeting_id}_partB.json",
        "anchor_inventory": output_root / "capa_anchor_inventory" / anchor_inventory_filename(participant_id, meeting_id),
        "episodes": output_root / "capa_episodes" / f"capa_episodes_for_{slug}_from_{meeting_id}.json",
        "k_diagnostics": output_root / "capa_k_diagnostics" / k_diagnostics_filename(participant_id, meeting_id, k),
        "eim": output_root / "capa_eim_states" / f"capa_eim_states_for_{slug}_in_{meeting_id}.json",
    }


def copy_artifact_if_needed(source_path: Path, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() == destination_path.resolve():
        return
    source_fingerprint = fingerprint_file(source_path)
    destination_fingerprint = fingerprint_file(destination_path) if destination_path.exists() else None
    if source_fingerprint is not None and source_fingerprint == destination_fingerprint:
        return
    shutil.copy2(source_path, destination_path)


def import_reusable_stage_output(
    *,
    source_manifest: Optional[Dict[str, Any]],
    unit_key: str,
    contract_fingerprint: str,
    source_path: Path,
    destination_path: Path,
) -> bool:
    source_unit = ((source_manifest or {}).get("units") or {}).get(unit_key)
    if not unit_is_reusable(source_unit, contract_fingerprint, [source_path]):
        return False
    copy_artifact_if_needed(source_path, destination_path)
    return True


def run_capa_prep(
    pipeline: Any,
    *,
    transcript_path: str,
    output_dir: str,
    meeting_id: str,
    profile_split_ratio: float,
    target_participant: Optional[str],
    participant_subset: Optional[List[str]],
    capa_N: int,
    capa_k: int,
    freeze_anchor_inventory: bool,
    anchor_inventory_dir: Optional[str],
    anchor_seed_k: Optional[int],
    static_prep_source_dir: Optional[str],
    eim_cache_dir: Optional[str],
    selector: str,
    cap_meeting: int,
    cap_per_participant: int,
    min_gap: int,
    debug_episodes: bool,
    jit_enable_sliding_windows: bool,
    jit_window_size: int,
    jit_window_stride: int,
    jit_emit_evidence_sidecar: bool,
    jit_enable_llm_cleanup: bool,
    resume: bool,
    force: bool,
) -> Dict[str, Any]:
    print("=" * 60, flush=True)
    print(f"PROCESSING TRANSCRIPT: {meeting_id}", flush=True)
    print(f"Source: {transcript_path}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    output_root = Path(output_dir)
    profiles_dir = output_root / "user_profiles"
    briefings_dir = output_root / "jit_briefings"
    anchor_inventory_output_dir = output_root / "capa_anchor_inventory"
    k_diagnostics_output_dir = output_root / "capa_k_diagnostics"
    capa_episodes_dir = output_root / "capa_episodes"
    capa_eim_dir = output_root / "capa_eim_states"
    eim_cache_root = Path(eim_cache_dir).resolve() if eim_cache_dir else (output_root / "_eim_cache")
    eim_window_dir = eim_cache_root / "window_vectors"
    eim_checkpoint_dir = eim_cache_root / "history_checkpoints"
    eim_cache_manifest_path = eim_cache_root / "cache_manifest.json"
    eim_cache_lock_path = eim_cache_root / ".cache.lock"
    static_prep_source_root = Path(static_prep_source_dir).resolve() if static_prep_source_dir else None

    for path in (
        profiles_dir,
        briefings_dir,
        anchor_inventory_output_dir,
        k_diagnostics_output_dir,
        capa_episodes_dir,
        capa_eim_dir,
        eim_window_dir,
        eim_checkpoint_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "meeting_id": meeting_id,
        "transcript_path": transcript_path,
        "output_directories": {
            "profiles": str(profiles_dir),
            "briefings": str(briefings_dir),
            "capa_anchor_inventory": str(anchor_inventory_output_dir),
            "capa_k_diagnostics": str(k_diagnostics_output_dir),
            "capa_episodes": str(capa_episodes_dir),
            "capa_eim_states": str(capa_eim_dir),
            "eim_cache_root": str(eim_cache_root),
        },
        "generated_files": {},
        "participants": [],
        "processing_stats": {},
    }

    def read_json_required(path: Path) -> Any:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def requested_participants_from_transcript(full_transcript: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        all_participants = list(dict.fromkeys(
            utt.get("speaker_id") for utt in full_transcript if isinstance(utt, dict) and utt.get("speaker_id")
        ))
        requested: List[str] = []
        missing: List[str] = []
        for participant in participant_subset or []:
            if participant in all_participants and participant not in requested:
                requested.append(participant)
            elif participant not in missing:
                missing.append(participant)
        if target_participant:
            if target_participant in all_participants and target_participant not in requested:
                requested.append(target_participant)
            elif target_participant not in missing:
                missing.append(target_participant)
        if missing:
            print(f"   - [WARN] Requested participants not found in transcript: {', '.join(missing)}")
        return all_participants, requested or all_participants

    def resolve_anchor_inventory_source_path(participant_id: str) -> Path:
        if not anchor_inventory_source_root:
            return requested_paths[participant_id]["anchor_inventory"]
        if anchor_inventory_source_root.is_file():
            return anchor_inventory_source_root
        filename = anchor_inventory_filename(participant_id, meeting_id)
        candidates = [
            anchor_inventory_source_root / filename,
            anchor_inventory_source_root / "capa_anchor_inventory" / filename,
            anchor_inventory_source_root / "anchor_inventory" / filename,
            anchor_inventory_source_root / meeting_id / "capa_anchor_inventory" / filename,
            anchor_inventory_source_root / meeting_id / "anchor_inventory" / filename,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def build_model_descriptor() -> Dict[str, Any]:
        return {
            "active_model_name": getattr(pipeline.config, "active_model_name", None),
            "provider": getattr(pipeline.config, "provider", None),
            "model": getattr(pipeline.config, "model", None),
        }

    def build_profile_contract(participant_id: str, transcript_fingerprint: str, model_descriptor: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        payload = {
            "stage": "profile",
            "stage_version": PROFILE_STAGE_VERSION,
            "pipeline_runtime_version": PIPELINE_RUNTIME_VERSION,
            "meeting_id": meeting_id,
            "participant_id": participant_id,
            "transcript_fingerprint": transcript_fingerprint,
            "profile_split_ratio": profile_split_ratio,
            "model_descriptor": model_descriptor,
        }
        return payload, fingerprint_contract(**payload)

    def build_normalized_profile_contract(participant_id: str, profile_file_fingerprint: Optional[str]) -> Tuple[Dict[str, Any], str]:
        payload = {
            "stage": "normalized_profile",
            "stage_version": NORMALIZED_PROFILE_STAGE_VERSION,
            "meeting_id": meeting_id,
            "participant_id": participant_id,
            "profile_file_fingerprint": profile_file_fingerprint,
        }
        return payload, fingerprint_contract(**payload)

    def build_anchor_inventory_contract(
        participant_id: str,
        testing_segment_fingerprint: str,
        normalized_profile_fingerprint: Optional[str],
        model_descriptor: Dict[str, Any],
        source_fingerprint: Optional[str],
    ) -> Tuple[Dict[str, Any], str]:
        payload = {
            "stage": "anchor_inventory",
            "stage_version": ANCHOR_INVENTORY_STAGE_VERSION,
            "meeting_id": meeting_id,
            "participant_id": participant_id,
            "transcript_fingerprint": transcript_fingerprint,
            "testing_segment_fingerprint": testing_segment_fingerprint,
            "profile_split_ratio": profile_split_ratio,
            "selector": selector,
            "capa_N": capa_N,
            "anchor_seed_k": int(anchor_seed_k or capa_k),
            "freeze_anchor_inventory": bool(freeze_anchor_inventory),
            "anchor_inventory_source": "external" if anchor_inventory_dir else "generated",
            "anchor_inventory_source_fingerprint": source_fingerprint,
            "normalized_profile_fingerprint": normalized_profile_fingerprint,
            "episode_normalization_version": EPISODE_NORMALIZATION_VERSION,
            "model_descriptor": model_descriptor,
        }
        return payload, fingerprint_contract(**payload)

    def build_episode_contract(
        participant_id: str,
        testing_segment_fingerprint: str,
        normalized_profile_fingerprint: Optional[str],
        briefing_fingerprint: Optional[str],
        model_descriptor: Dict[str, Any],
        anchor_inventory_fingerprint: Optional[str],
    ) -> Tuple[Dict[str, Any], str]:
        payload = {
            "stage": "episodes",
            "stage_version": EPISODE_STAGE_VERSION,
            "meeting_id": meeting_id,
            "participant_id": participant_id,
            "transcript_fingerprint": transcript_fingerprint,
            "testing_segment_fingerprint": testing_segment_fingerprint,
            "profile_split_ratio": profile_split_ratio,
            "normalized_profile_fingerprint": normalized_profile_fingerprint,
            "briefing_fingerprint": briefing_fingerprint,
            "selector": selector,
            "capa_N": capa_N,
            "capa_k": capa_k,
            "cap_meeting": cap_meeting,
            "cap_per_participant": cap_per_participant,
            "min_gap": min_gap,
            "freeze_anchor_inventory": bool(freeze_anchor_inventory),
            "anchor_inventory_source": "external" if anchor_inventory_dir else "generated",
            "anchor_seed_k": int(anchor_seed_k or capa_k),
            "anchor_inventory_fingerprint": anchor_inventory_fingerprint,
            "episode_normalization_version": EPISODE_NORMALIZATION_VERSION,
            "model_descriptor": model_descriptor,
        }
        return payload, fingerprint_contract(**payload)

    def build_jit_contract(participant_id: str, testing_segment_fingerprint: str, model_descriptor: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        payload = {
            "stage": "jit",
            "stage_version": JIT_STAGE_VERSION,
            "meeting_id": meeting_id,
            "participant_id": participant_id,
            "transcript_fingerprint": transcript_fingerprint,
            "testing_segment_fingerprint": testing_segment_fingerprint,
            "profile_split_ratio": profile_split_ratio,
            "jit_enable_sliding_windows": jit_enable_sliding_windows,
            "jit_window_size": jit_window_size,
            "jit_window_stride": jit_window_stride,
            "jit_emit_evidence_sidecar": jit_emit_evidence_sidecar,
            "jit_enable_llm_cleanup": jit_enable_llm_cleanup,
            "model_descriptor": model_descriptor,
        }
        return payload, fingerprint_contract(**payload)

    def build_eim_contract(participant_id: str, episode_file_fingerprint: Optional[str], model_descriptor: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        payload = {
            "stage": "eim",
            "stage_version": CAPA_EIM_STAGE_VERSION,
            "meeting_id": meeting_id,
            "participant_id": participant_id,
            "transcript_fingerprint": transcript_fingerprint,
            "episode_file_fingerprint": episode_file_fingerprint,
            "snapshot_postprocess_version": EIM_SNAPSHOT_POSTPROCESS_VERSION,
            "model_descriptor": model_descriptor,
        }
        return payload, fingerprint_contract(**payload)

    def old_manifest_contract_matches(manifest: Dict[str, Any], model_descriptor: Dict[str, Any]) -> bool:
        meeting_unit = (manifest.get("units") or {}).get("meeting") or {}
        metadata = meeting_unit.get("metadata") or {}
        contract = metadata.get("contract") or {}
        if not contract:
            return manifest.get("state") == "corrupt"
        comparable_keys = (
            "meeting_id",
            "transcript_fingerprint",
            "profile_split_ratio",
            "selector",
            "capa_N",
            "capa_k",
            "freeze_anchor_inventory",
            "anchor_seed_k",
            "cap_meeting",
            "cap_per_participant",
            "min_gap",
            "jit_enable_sliding_windows",
            "jit_window_size",
            "jit_window_stride",
            "jit_emit_evidence_sidecar",
            "jit_enable_llm_cleanup",
        )
        for key in comparable_keys:
            if contract.get(key) != current_prep_base_contract.get(key):
                return False
        return contract.get("model_descriptor") == model_descriptor

    def bootstrap_manifest_if_needed(manifest: Dict[str, Any], model_descriptor: Dict[str, Any], requested_paths: Dict[str, Dict[str, Path]]) -> Dict[str, Any]:
        if manifest.get("manifest_version") == PREP_MANIFEST_VERSION and any("::" in key for key in (manifest.get("units") or {})):
            return manifest
        bootstrapped = {
            "manifest_version": PREP_MANIFEST_VERSION,
            "state": "not_started",
            "units": {},
            "updated_at": manifest.get("updated_at"),
        }
        allow_backfill = old_manifest_contract_matches(manifest, model_descriptor)
        if not allow_backfill:
            return bootstrapped
        for participant_id, paths in requested_paths.items():
            profile_path = paths["profile"]
            if profile_path.exists():
                profile_payload, profile_fp = build_profile_contract(participant_id, transcript_fingerprint, model_descriptor)
                mark_unit_complete(
                    bootstrapped,
                    participant_unit_key("profile", participant_id),
                    profile_fp,
                    [profile_path],
                    metadata={"participant_id": participant_id, "bootstrapped": True, "normalized_output_only": True, "contract": profile_payload},
                )
                normalized_payload, normalized_fp = build_normalized_profile_contract(participant_id, fingerprint_file(profile_path))
                mark_unit_complete(
                    bootstrapped,
                    participant_unit_key("normalized_profile", participant_id),
                    normalized_fp,
                    [profile_path],
                    metadata={"participant_id": participant_id, "bootstrapped": True, "contract": normalized_payload},
                )
            episode_path = paths["episodes"]
            if episode_path.exists() and profile_path.exists():
                episode_payload, episode_fp = build_episode_contract(
                    participant_id,
                    testing_segment_fingerprint,
                    fingerprint_file(profile_path),
                    fingerprint_file(paths.get("briefing")) if paths.get("briefing") else None,
                    model_descriptor,
                    fingerprint_file(paths.get("anchor_inventory")) if paths.get("anchor_inventory") else None,
                )
                mark_unit_complete(
                    bootstrapped,
                    participant_unit_key("episodes", participant_id),
                    episode_fp,
                    [episode_path],
                    metadata={"participant_id": participant_id, "bootstrapped": True, "contract": episode_payload},
                )
            briefing_path = paths["briefing"]
            if briefing_path.exists():
                briefing_data = safe_json_load(briefing_path, default={}) or {}
                semantic_empty = not (briefing_data.get("jit_briefing_points") or [])
                jit_payload, jit_fp = build_jit_contract(participant_id, testing_segment_fingerprint, model_descriptor)
                mark_unit_complete(
                    bootstrapped,
                    participant_unit_key("jit", participant_id),
                    jit_fp,
                    [briefing_path],
                    metadata={
                        "participant_id": participant_id,
                        "bootstrapped": True,
                        "semantic_empty": semantic_empty,
                        "empty_reason": "no_meaningful_points" if semantic_empty else None,
                        "contract": jit_payload,
                    },
                )
            eim_path = paths["eim"]
            if eim_path.exists() and episode_path.exists():
                eim_payload, eim_fp = build_eim_contract(participant_id, fingerprint_file(episode_path), model_descriptor)
                mark_unit_complete(
                    bootstrapped,
                    participant_unit_key("eim", participant_id),
                    eim_fp,
                    [eim_path],
                    metadata={"participant_id": participant_id, "bootstrapped": True, "contract": eim_payload},
                )
        bootstrapped["state"] = summarize_manifest_state(bootstrapped)
        return bootstrapped

    def read_cache_manifest() -> Dict[str, Any]:
        manifest = safe_json_load(eim_cache_manifest_path, default=None)
        if isinstance(manifest, dict):
            manifest.setdefault("manifest_version", EIM_CACHE_MANIFEST_VERSION)
            manifest.setdefault("window_vectors", {})
            manifest.setdefault("history_checkpoints", {})
            return manifest
        return {
            "manifest_version": EIM_CACHE_MANIFEST_VERSION,
            "window_vectors": {},
            "history_checkpoints": {},
        }

    def write_cache_manifest(cache_manifest: Dict[str, Any]) -> None:
        atomic_write_json(eim_cache_manifest_path, cache_manifest)

    def checkpoint_contract_payload(model_descriptor: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "stage": "eim_history_checkpoint",
            "stage_version": CAPA_EIM_STAGE_VERSION,
            "snapshot_postprocess_version": EIM_SNAPSHOT_POSTPROCESS_VERSION,
            "meeting_id": meeting_id,
            "transcript_fingerprint": transcript_fingerprint,
            "model_descriptor": model_descriptor,
            "checkpoint_stride": EIM_CHECKPOINT_STRIDE,
        }

    def checkpoint_contract_fingerprint(model_descriptor: Dict[str, Any]) -> str:
        return fingerprint_contract(**checkpoint_contract_payload(model_descriptor))

    def checkpoint_path_for_idx(processed_idx: int) -> Path:
        return eim_checkpoint_dir / f"checkpoint_{processed_idx:06d}.json"

    def write_history_checkpoint(cache_manifest: Dict[str, Any], processed_idx: int, full_transcript: List[Dict[str, Any]], model_descriptor: Dict[str, Any], source: str) -> None:
        if processed_idx < 0:
            return
        checkpoint_path = checkpoint_path_for_idx(processed_idx)
        utt_id = full_transcript[processed_idx].get("utterance_id") if processed_idx < len(full_transcript) else None
        payload = {
            "contract_fingerprint": checkpoint_contract_fingerprint(model_descriptor),
            "contract": checkpoint_contract_payload(model_descriptor),
            "processed_through_idx": processed_idx,
            "processed_through_utterance_id": utt_id,
            "source": source,
            "master_state": pipeline.eim.export_master_state(),
        }
        atomic_write_json(checkpoint_path, payload)
        cache_manifest.setdefault("history_checkpoints", {})[str(checkpoint_path)] = {
            "contract_fingerprint": payload["contract_fingerprint"],
            "processed_through_idx": processed_idx,
            "source": source,
        }

    def episode_context_ids(ep: CAPAEpisode) -> List[str]:
        return [utt.get("utterance_id") for utt in (ep.context_last_N or []) if isinstance(utt, dict) and utt.get("utterance_id")]

    def build_window_cache_entry(ep: CAPAEpisode, model_descriptor: Dict[str, Any]) -> Dict[str, Any]:
        context_ids = episode_context_ids(ep)
        cue_idx = ep.cue_index if ep.cue_index is not None else full_utt_index.get(ep.cue_utt_id, 10**9)
        history_processed_through_idx = max(-1, max(0, int(cue_idx) - (ep.N - 1)) - 1)
        contract = {
            "stage": "eim_window_vector",
            "stage_version": CAPA_EIM_STAGE_VERSION,
            "snapshot_postprocess_version": EIM_SNAPSHOT_POSTPROCESS_VERSION,
            "meeting_id": meeting_id,
            "transcript_fingerprint": transcript_fingerprint,
            "model_descriptor": model_descriptor,
            "cue_utt_id": ep.cue_utt_id,
            "cue_idx": cue_idx,
            "context_ids": context_ids,
        }
        contract_fingerprint = fingerprint_contract(**contract)
        return {
            "window_key": contract_fingerprint,
            "contract": contract,
            "contract_fingerprint": contract_fingerprint,
            "cache_path": eim_window_dir / f"{contract_fingerprint}.json",
            "cue_idx": cue_idx,
            "history_processed_through_idx": history_processed_through_idx,
            "context": ep.context_last_N,
            "test_case_id": ep.test_case_id,
            "participant_id": ep.represented_user,
        }

    def load_cached_window_vector(window_entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        payload = safe_json_load(window_entry["cache_path"], default=None)
        if not isinstance(payload, dict):
            return None
        if payload.get("contract_fingerprint") != window_entry["contract_fingerprint"]:
            return None
        if not payload.get("eim_output"):
            return None
        return payload

    def write_cached_window_vector(cache_manifest: Dict[str, Any], window_entry: Dict[str, Any], eim_output: Dict[str, Any]) -> None:
        payload = {
            "contract_fingerprint": window_entry["contract_fingerprint"],
            "contract": window_entry["contract"],
            "cue_idx": window_entry["cue_idx"],
            "history_processed_through_idx": window_entry["history_processed_through_idx"],
            "processed_through_utterance_id": full_transcript[window_entry["history_processed_through_idx"]].get("utterance_id") if window_entry["history_processed_through_idx"] >= 0 else None,
            "test_case_id": window_entry["test_case_id"],
            "participant_id": window_entry["participant_id"],
            "eim_output": eim_output,
        }
        atomic_write_json(window_entry["cache_path"], payload)
        cache_manifest.setdefault("window_vectors", {})[window_entry["window_key"]] = {
            "path": str(window_entry["cache_path"]),
            "cue_idx": window_entry["cue_idx"],
            "history_processed_through_idx": window_entry["history_processed_through_idx"],
            "contract_fingerprint": window_entry["contract_fingerprint"],
        }

    def restore_best_eim_bootstrap(cache_manifest: Dict[str, Any], first_missing_cue_idx: int, model_descriptor: Dict[str, Any]) -> Tuple[int, str]:
        checkpoint_fp = checkpoint_contract_fingerprint(model_descriptor)
        best_idx = -1
        best_source = "fresh"
        best_state: Optional[Dict[str, Any]] = None

        for checkpoint_path in sorted(eim_checkpoint_dir.glob("checkpoint_*.json")):
            payload = safe_json_load(checkpoint_path, default=None)
            if not isinstance(payload, dict):
                continue
            if payload.get("contract_fingerprint") != checkpoint_fp:
                continue
            processed_idx = int(payload.get("processed_through_idx", -1))
            if processed_idx <= first_missing_cue_idx and processed_idx > best_idx:
                best_idx = processed_idx
                best_source = f"checkpoint:{checkpoint_path.name}"
                best_state = payload.get("master_state")

        for vector_path in sorted(eim_window_dir.glob("*.json")):
            payload = safe_json_load(vector_path, default=None)
            if not isinstance(payload, dict):
                continue
            contract = payload.get("contract") or {}
            if contract.get("transcript_fingerprint") != transcript_fingerprint:
                continue
            if contract.get("model_descriptor") != model_descriptor:
                continue
            if contract.get("stage_version") != CAPA_EIM_STAGE_VERSION:
                continue
            cue_idx = int(payload.get("cue_idx", -1))
            if cue_idx > first_missing_cue_idx:
                continue
            if cue_idx <= best_idx:
                continue
            eim_output = payload.get("eim_output") or {}
            best_idx = cue_idx
            best_source = f"window:{vector_path.name}"
            best_state = {
                "topic_summary_log": eim_output.get("topic_summary_log") or {},
                "participant_models": eim_output.get("participant_models") or {},
                "identity_aliases": eim_output.get("identity_aliases") or {},
            }

        pipeline.eim.reset_master_state()
        if best_state:
            pipeline.eim.load_master_state(best_state)
        return best_idx, best_source

    transcript_payload = read_json_required(Path(transcript_path))
    full_transcript = transcript_to_list(transcript_payload)
    if not full_transcript:
        results["success"] = False
        results["error"] = f"Unsupported or empty transcript payload: {transcript_path}"
        return results

    profile_segment, testing_segment = pipeline.transcript_processor.split_transcript(full_transcript, profile_split_ratio)
    all_participants, participants = requested_participants_from_transcript(full_transcript)
    results["participants"] = participants

    print("\n>> Loading and splitting transcript...", flush=True)
    print(f"   - Total utterances: {len(full_transcript)}")
    print(f"   - Profile segment: {len(profile_segment)} utterances")
    print(f"   - Testing segment: {len(testing_segment)} utterances")
    print(f"   - All participants: {', '.join(all_participants)}")
    if participant_subset or target_participant:
        print(f"   - Processing participant subset: {', '.join(participants)}")
    else:
        print("   - Processing all participants")

    transcript_fingerprint = fingerprint_transcript_file(Path(transcript_path))
    testing_segment_fingerprint = stable_segment_fingerprint(testing_segment)
    model_descriptor = build_model_descriptor()
    current_prep_base_contract = {
        "stage": "prep",
        "stage_version": PIPELINE_RUNTIME_VERSION,
        "meeting_id": meeting_id,
        "transcript_fingerprint": transcript_fingerprint,
        "profile_split_ratio": profile_split_ratio,
        "selector": selector,
        "capa_N": capa_N,
        "capa_k": capa_k,
        "freeze_anchor_inventory": bool(freeze_anchor_inventory),
        "anchor_seed_k": int(anchor_seed_k or capa_k),
        "cap_meeting": cap_meeting,
        "cap_per_participant": cap_per_participant,
        "min_gap": min_gap,
        "jit_enable_sliding_windows": jit_enable_sliding_windows,
        "jit_window_size": jit_window_size,
        "jit_window_stride": jit_window_stride,
        "jit_emit_evidence_sidecar": jit_emit_evidence_sidecar,
        "jit_enable_llm_cleanup": jit_enable_llm_cleanup,
        "model_descriptor": model_descriptor,
    }
    requested_paths = {participant_id: prep_paths_for_participant(output_root, meeting_id, participant_id, capa_k) for participant_id in participants}
    anchor_inventory_source_root = Path(anchor_inventory_dir).resolve() if anchor_inventory_dir else None
    static_source_paths = {
        participant_id: prep_paths_for_participant(static_prep_source_root, meeting_id, participant_id, capa_k)
        for participant_id in participants
    } if static_prep_source_root else {}
    static_source_manifest = (
        load_manifest(static_prep_source_root / "prep_manifest.json", default_version=PREP_MANIFEST_VERSION)
        if static_prep_source_root
        else None
    )
    full_utt_index = {utt["utterance_id"]: idx for idx, utt in enumerate(full_transcript) if isinstance(utt, dict) and utt.get("utterance_id")}

    profile_files: Dict[str, str] = {}
    normalized_profile_files: Dict[str, str] = {}
    briefing_files: Dict[str, str] = {}
    anchor_inventory_files: Dict[str, str] = {}
    anchor_inventory_counts: Dict[str, int] = {}
    capa_episode_files: Dict[str, str] = {}
    capa_episode_counts: Dict[str, int] = {}
    k_diagnostics_files: Dict[str, str] = {}
    capa_eim_files: Dict[str, str] = {}
    capa_eim_counts: Dict[str, int] = {}
    failed_units: List[str] = []

    def refresh_meeting_k_diagnostics_summary() -> None:
        """Rebuild meeting-level k diagnostics from all participant diagnostics.

        Resume can regenerate only a subset of participants. Reading the
        per-participant files avoids dropping summaries for reused participants.
        """
        participant_summaries: List[Dict[str, Any]] = []
        for participant_id in participants:
            paths = requested_paths[participant_id]
            diagnostics_payload = safe_json_load(paths["k_diagnostics"], default={})
            summary = (diagnostics_payload or {}).get("summary") or {}
            if summary:
                participant_summaries.append(summary)
        if not participant_summaries:
            return
        meeting_summary_path = k_diagnostics_output_dir / f"k_diagnostics_summary_for_{meeting_id}_k{capa_k}.json"
        atomic_write_json(
            meeting_summary_path,
            summarize_meeting_k_diagnostics(
                meeting_id=meeting_id,
                k=capa_k,
                inventory_reference_k=inventory_reference_k,
                participant_summaries=participant_summaries,
            ),
        )

    def ensure_jit_briefings() -> None:
        print("\n>> Generating JIT briefings...", flush=True)
        for participant_id in participants:
            paths = requested_paths[participant_id]
            jit_payload, jit_contract_fp = build_jit_contract(participant_id, testing_segment_fingerprint, model_descriptor)
            jit_unit = (prep_manifest.get("units") or {}).get(participant_unit_key("jit", participant_id))
            if resume and not force and unit_is_reusable(jit_unit, jit_contract_fp, [paths["briefing"]]):
                print(f"   - Reusing JIT briefing for {participant_id}")
                briefing_files[participant_id] = str(paths["briefing"])
                continue
            if static_prep_source_root and not force:
                source_briefing_path = static_source_paths[participant_id]["briefing"]
                if import_reusable_stage_output(
                    source_manifest=static_source_manifest,
                    unit_key=participant_unit_key("jit", participant_id),
                    contract_fingerprint=jit_contract_fp,
                    source_path=source_briefing_path,
                    destination_path=paths["briefing"],
                ):
                    briefing_files[participant_id] = str(paths["briefing"])
                    briefing_payload = safe_json_load(paths["briefing"], default={}) or {}
                    semantic_empty = not (briefing_payload.get("jit_briefing_points") or [])
                    mark_unit_complete(
                        prep_manifest,
                        participant_unit_key("jit", participant_id),
                        jit_contract_fp,
                        [paths["briefing"]],
                        metadata={
                            "participant_id": participant_id,
                            "contract": jit_payload,
                            "semantic_empty": semantic_empty,
                            "empty_reason": "no_meaningful_points" if semantic_empty else None,
                            "imported_from": str(source_briefing_path),
                        },
                    )
                    prep_manifest["state"] = summarize_manifest_state(prep_manifest)
                    save_manifest(manifest_path, prep_manifest)
                    print(f"   - Imported JIT briefing for {participant_id} from {source_briefing_path}")
                    continue
            mark_unit_in_progress(prep_manifest, participant_unit_key("jit", participant_id), jit_contract_fp, {"participant_id": participant_id, "contract": jit_payload})
            prep_manifest["state"] = summarize_manifest_state(prep_manifest)
            save_manifest(manifest_path, prep_manifest)
            generated = pipeline.briefing_generator.generate_briefings_for_meeting(
                transcript_path=transcript_path,
                output_directory=str(briefings_dir),
                split_ratio=profile_split_ratio,
                meeting_id=meeting_id,
                target_participant=participant_id,
                enable_sliding_windows=jit_enable_sliding_windows,
                window_size=jit_window_size,
                window_stride=jit_window_stride,
                emit_evidence_sidecar=jit_emit_evidence_sidecar,
                enable_llm_cleanup=jit_enable_llm_cleanup,
            )
            if paths["briefing"].exists() and participant_id in generated:
                briefing_files[participant_id] = str(paths["briefing"])
                briefing_payload = safe_json_load(paths["briefing"], default={}) or {}
                semantic_empty = not (briefing_payload.get("jit_briefing_points") or [])
                mark_unit_complete(
                    prep_manifest,
                    participant_unit_key("jit", participant_id),
                    jit_contract_fp,
                    [paths["briefing"]],
                    metadata={
                        "participant_id": participant_id,
                        "contract": jit_payload,
                        "semantic_empty": semantic_empty,
                        "empty_reason": "no_meaningful_points" if semantic_empty else None,
                    },
                )
                prep_manifest["state"] = summarize_manifest_state(prep_manifest)
                save_manifest(manifest_path, prep_manifest)
            else:
                error = "jit_generation_failed"
                print(f"   - [ERROR] Failed to generate JIT briefing for {participant_id}")
                mark_unit_failed(prep_manifest, participant_unit_key("jit", participant_id), jit_contract_fp, {"participant_id": participant_id, "error": error, "contract": jit_payload})
                prep_manifest["state"] = summarize_manifest_state(prep_manifest)
                save_manifest(manifest_path, prep_manifest)
                failed_units.append(participant_unit_key("jit", participant_id))

    def delegate_inventory_text_for(participant_id: str) -> str:
        paths = requested_paths[participant_id]
        profile_payload = safe_json_load(paths["profile"], default={}) or {}
        briefing_payload = safe_json_load(paths["briefing"], default={}) or {}
        return build_delegate_inventory_text(profile_payload, briefing_payload)

    def delegate_inventory_payloads_for(participant_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        paths = requested_paths[participant_id]
        profile_payload = safe_json_load(paths["profile"], default={}) or {}
        briefing_payload = safe_json_load(paths["briefing"], default={}) or {}
        return profile_payload, briefing_payload

    def run_semantic_target_availability_verifier(
        *,
        participant_id: str,
        selector_item: Dict[str, Any],
        episode: CAPAEpisode,
        stage1: Dict[str, Any],
        inventory_evidence: List[Dict[str, Any]],
    ) -> Optional[TargetAvailabilityVerifierReport]:
        target_ids = [str(rid) for rid in selector_item.get("response_utterance_ids") or []]
        target_rows = []
        for utt in episode.gt_next_window or []:
            if not isinstance(utt, dict):
                continue
            uid = str(utt.get("utterance_id") or "")
            if target_ids and uid not in target_ids:
                continue
            target_rows.append(
                {
                    "utterance_id": uid,
                    "speaker_id": utt.get("speaker_id"),
                    "utterance_content": utt.get("utterance_content") or utt.get("content") or "",
                }
            )
        evidence_ids = set(stage1.get("target_availability_evidence_ids") or [])
        top_evidence = []
        for item in inventory_evidence:
            if not isinstance(item, dict):
                continue
            if item.get("evidence_id") in evidence_ids or len(top_evidence) < 8:
                top_evidence.append(
                    {
                        "evidence_id": item.get("evidence_id"),
                        "source": item.get("source"),
                        "core_text": item.get("core_text") or item.get("text"),
                        "support_utt_ids": item.get("support_utt_ids") or [],
                        "anchor_phrases": item.get("anchor_phrases") or [],
                        "context_utt_ids": item.get("context_utt_ids") or [],
                    }
                )
            if len(top_evidence) >= 10:
                break
        system_prompt = (
            "You are a narrow CAPA prep verifier. Decide whether the represented user's "
            "official recall target is fairly available from that participant's JIT/profile inventory. "
            "Return only JSON matching the schema. Verdicts: supported means same actionable core "
            "proposition is available; testable_partial means a concrete directly usable facet is "
            "available and no essential constraint is missing or contradicted; unsupported_generic "
            "means only broad topical/generic support; unsupported_missing means required facts, "
            "constraints, design mechanisms, decisions, or rationale are missing."
        )
        user_prompt = json.dumps(
            {
                "participant_id": participant_id,
                "official_target": {
                    "topic_summary": selector_item.get("topic_summary"),
                    "official_target_text": selector_item.get("official_target_text"),
                    "response_utterances": target_rows,
                },
                "deterministic_stage1": stage1,
                "participant_inventory_evidence": top_evidence,
                "decision_rule": (
                    "Keep supported and concrete testable_partial only. Reject vague broad "
                    "preference/research/marketability overlap when the target is a specific "
                    "constraint, decision, mechanism, or rationale."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        try:
            response = pipeline.llm_client.call_with_retry(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1000,
                json_required=True,
                response_model=TargetAvailabilityVerifierReport,
                temperature=0.0,
                telemetry_label="prep.target_availability_verifier",
                telemetry_phase="prep",
            )
            if hasattr(response, "model_dump"):
                return response
            if isinstance(response, dict):
                return TargetAvailabilityVerifierReport(**response)
        except Exception as exc:
            if debug_episodes:
                print(f"       [WARN] Semantic target availability verifier failed for {participant_id}: {exc}")
        return None

    def apply_official_episode_gate(
        *,
        participant_id: str,
        selector_item: Dict[str, Any],
        episode: CAPAEpisode,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        gated_item = dict(selector_item or {})
        window_ok, window_reason, gated_item = enforce_official_response_window(
            selector_item=gated_item,
            episode=episode,
            represented_user=participant_id,
        )
        if not window_ok:
            return False, window_reason, gated_item
        quality = assess_episode_quality(
            selector_item=gated_item,
            episode=episode,
            represented_user=participant_id,
        )
        gated_item.update(quality)
        if quality.get("quality_status") != "kept":
            return False, str(quality.get("quality_drop_reason") or "low_substance"), gated_item

        profile_payload, briefing_payload = delegate_inventory_payloads_for(participant_id)
        inventory_evidence = build_delegate_inventory_evidence(profile_payload, briefing_payload)
        availability = assess_target_availability_stage1(
            selector_item=gated_item,
            episode=episode,
            inventory_text=delegate_inventory_text_for(participant_id),
            inventory_evidence=inventory_evidence,
        )
        if availability.get("target_availability_requires_semantic_review"):
            semantic = run_semantic_target_availability_verifier(
                participant_id=participant_id,
                selector_item=gated_item,
                episode=episode,
                stage1=availability,
                inventory_evidence=inventory_evidence,
            )
            if semantic is not None:
                sem = semantic.model_dump()
                verdict = sem.get("verdict")
                availability.update(
                    target_available_to_delegate=verdict in {"supported", "testable_partial"},
                    target_availability_verdict=verdict,
                    target_availability_method="semantic_stage2",
                    target_availability_reason=f"semantic_stage2_{verdict}",
                    target_availability_evidence_ids=sem.get("evidence_ids") or [],
                    target_availability_missing_facets=sem.get("missing_facets") or [],
                    target_availability_rationale=sem.get("rationale") or "",
                    target_availability_requires_semantic_review=False,
                )
            else:
                availability.update(
                    target_availability_method="deterministic_stage1_semantic_unavailable",
                    target_availability_rationale=(
                        str(availability.get("target_availability_rationale") or "")
                        + " Semantic verifier unavailable; deterministic verdict used."
                    ).strip(),
                    target_availability_requires_semantic_review=False,
                )
        gated_item.update(availability)
        if not bool(availability.get("target_available_to_delegate")):
            return False, "target_unavailable_to_delegate", gated_item
        gated_item = canonicalize_official_target(
            selector_item=gated_item,
            episode=episode,
            inventory_evidence=inventory_evidence,
        )
        try:
            repaired_units: List[Dict[str, Any]] = []
            for unit in episode.idea_units or []:
                if hasattr(unit, "model_dump"):
                    repaired_units.append(unit.model_dump(exclude_none=True))
                elif isinstance(unit, dict):
                    repaired_units.append(dict(unit))
            if repaired_units:
                repaired_units[0].update(gated_item)
            else:
                repaired_units = [dict(gated_item)]
            episode.idea_units = [IdeaUnit(**unit) for unit in repaired_units]
        except Exception:
            pass
        return True, "", gated_item

    manifest_path = output_root / "prep_manifest.json"
    lock_path = output_root / ".prep.lock"

    try:
        with file_lock(lock_path):
            prep_manifest = bootstrap_manifest_if_needed(load_manifest(manifest_path, default_version=PREP_MANIFEST_VERSION), model_descriptor, requested_paths)
            save_manifest(manifest_path, prep_manifest)

            print("\n>> Generating user profiles...", flush=True)
            profile_segment_string = pipeline.profile_generator.format_utterances_for_prompt(profile_segment)
            for participant_id in participants:
                paths = requested_paths[participant_id]
                current_profile_fp = fingerprint_file(paths["profile"])
                profile_payload, profile_contract_fp = build_profile_contract(participant_id, transcript_fingerprint, model_descriptor)
                normalized_payload, normalized_contract_fp = build_normalized_profile_contract(participant_id, current_profile_fp)
                profile_unit = (prep_manifest.get("units") or {}).get(participant_unit_key("profile", participant_id))
                normalized_unit = (prep_manifest.get("units") or {}).get(participant_unit_key("normalized_profile", participant_id))
                if resume and not force and unit_is_reusable(profile_unit, profile_contract_fp, [paths["profile"]]) and unit_is_reusable(normalized_unit, normalized_contract_fp, [paths["profile"]]):
                    print(f"   - Reusing profile for {participant_id}", flush=True)
                    profile_files[participant_id] = str(paths["profile"])
                    normalized_profile_files[participant_id] = str(paths["profile"])
                    continue
                if static_prep_source_root and not force:
                    source_profile_path = static_source_paths[participant_id]["profile"]
                    if import_reusable_stage_output(
                        source_manifest=static_source_manifest,
                        unit_key=participant_unit_key("profile", participant_id),
                        contract_fingerprint=profile_contract_fp,
                        source_path=source_profile_path,
                        destination_path=paths["profile"],
                    ):
                        current_profile_fp = fingerprint_file(paths["profile"])
                        normalized_payload, normalized_contract_fp = build_normalized_profile_contract(participant_id, current_profile_fp)
                        mark_unit_complete(
                            prep_manifest,
                            participant_unit_key("profile", participant_id),
                            profile_contract_fp,
                            [paths["profile"]],
                            metadata={
                                "participant_id": participant_id,
                                "contract": profile_payload,
                                "normalized_output_only": True,
                                "imported_from": str(source_profile_path),
                            },
                        )
                        mark_unit_complete(
                            prep_manifest,
                            participant_unit_key("normalized_profile", participant_id),
                            normalized_contract_fp,
                            [paths["profile"]],
                            metadata={
                                "participant_id": participant_id,
                                "contract": normalized_payload,
                                "imported_from": str(source_profile_path),
                            },
                        )
                        prep_manifest["state"] = summarize_manifest_state(prep_manifest)
                        save_manifest(manifest_path, prep_manifest)
                        profile_files[participant_id] = str(paths["profile"])
                        normalized_profile_files[participant_id] = str(paths["profile"])
                        print(f"   - Imported profile for {participant_id} from {source_profile_path}", flush=True)
                        continue

                print(f"   - Generating profile for {participant_id}...", flush=True)
                mark_unit_in_progress(prep_manifest, participant_unit_key("profile", participant_id), profile_contract_fp, {"participant_id": participant_id, "contract": profile_payload})
                mark_unit_in_progress(prep_manifest, participant_unit_key("normalized_profile", participant_id), normalized_contract_fp, {"participant_id": participant_id, "contract": normalized_payload})
                prep_manifest["state"] = summarize_manifest_state(prep_manifest)
                save_manifest(manifest_path, prep_manifest)
                try:
                    user_profile = pipeline.profile_generator.generate_rich_user_profile(
                        participant_speaker_id=participant_id,
                        meeting_transcript_segment_string=profile_segment_string,
                    )
                    if not user_profile:
                        raise RuntimeError("profile_generation_returned_none")
                    payload = user_profile.model_dump()
                    profile_payload_json = payload.get("UserProfile", payload)
                    atomic_write_json(paths["profile"], profile_payload_json)
                    current_profile_fp = fingerprint_file(paths["profile"])
                    normalized_payload, normalized_contract_fp = build_normalized_profile_contract(participant_id, current_profile_fp)
                    mark_unit_complete(
                        prep_manifest,
                        participant_unit_key("profile", participant_id),
                        profile_contract_fp,
                        [paths["profile"]],
                        metadata={"participant_id": participant_id, "contract": profile_payload, "normalized_output_only": True},
                    )
                    mark_unit_complete(
                        prep_manifest,
                        participant_unit_key("normalized_profile", participant_id),
                        normalized_contract_fp,
                        [paths["profile"]],
                        metadata={"participant_id": participant_id, "contract": normalized_payload},
                    )
                    prep_manifest["state"] = summarize_manifest_state(prep_manifest)
                    save_manifest(manifest_path, prep_manifest)
                    profile_files[participant_id] = str(paths["profile"])
                    normalized_profile_files[participant_id] = str(paths["profile"])
                    print(f"     [OK] Saved CAPA profile -> {paths['profile'].name}")
                except Exception as exc:
                    error = str(exc)
                    print(f"     [ERROR] Failed to generate profile for {participant_id}: {error}")
                    mark_unit_failed(prep_manifest, participant_unit_key("profile", participant_id), profile_contract_fp, {"participant_id": participant_id, "error": error, "contract": profile_payload})
                    mark_unit_failed(prep_manifest, participant_unit_key("normalized_profile", participant_id), normalized_contract_fp, {"participant_id": participant_id, "error": error, "contract": normalized_payload})
                    prep_manifest["state"] = summarize_manifest_state(prep_manifest)
                    save_manifest(manifest_path, prep_manifest)
                    failed_units.extend([
                        participant_unit_key("profile", participant_id),
                        participant_unit_key("normalized_profile", participant_id),
                    ])

            # JIT/profile inventory is now an input to official episode validity, so
            # briefings must exist before selector outputs are finalized.
            ensure_jit_briefings()

            anchor_inventory_enabled = selector == "llm-idea" and (freeze_anchor_inventory or anchor_inventory_dir)
            inventory_reference_k = int(anchor_seed_k or capa_k) if anchor_inventory_enabled else None
            anchor_records_by_participant: Dict[str, List[Dict[str, Any]]] = {}
            if (freeze_anchor_inventory or anchor_inventory_dir) and selector != "llm-idea":
                raise ValueError("Frozen anchor inventory is only supported with selector='llm-idea'")

            if anchor_inventory_enabled:
                print("\n>> Freezing anchor inventory (CAPA mode)...", flush=True)
                from capa.cue_selector_llm import build_transcript_table
                from capa.idea_unit_selector_llm import select_idea_units
                from capa.episode_utils import build_id_index

                seed_k = int(anchor_seed_k or capa_k)
                selector_max_items = selector_shortlist_size(cap_per_participant)
                start_uid = testing_segment[0]["utterance_id"]
                end_uid = testing_segment[-1]["utterance_id"]
                table = build_transcript_table(full_transcript, full_utt_index, start_uid, end_uid)
                id_to_idx = build_id_index(full_transcript)

                for participant_id in participants:
                    paths = requested_paths[participant_id]
                    normalized_profile_fp = fingerprint_file(paths["profile"])
                    source_inventory_path = None
                    source_fingerprint = None
                    if anchor_inventory_dir:
                        source_inventory_path = resolve_anchor_inventory_source_path(participant_id)
                        source_fingerprint = fingerprint_file(source_inventory_path)
                    anchor_payload, anchor_contract_fp = build_anchor_inventory_contract(
                        participant_id,
                        testing_segment_fingerprint,
                        normalized_profile_fp,
                        model_descriptor,
                        source_fingerprint,
                    )
                    anchor_unit = (prep_manifest.get("units") or {}).get(participant_unit_key("anchor_inventory", participant_id))
                    if resume and not force and unit_is_reusable(anchor_unit, anchor_contract_fp, [paths["anchor_inventory"]]):
                        print(f"   - Reusing anchor inventory for {participant_id}")
                        anchor_inventory_files[participant_id] = str(paths["anchor_inventory"])
                        anchor_records = load_anchor_inventory_records(paths["anchor_inventory"])
                        anchor_records_by_participant[participant_id] = anchor_records
                        anchor_inventory_counts[participant_id] = len(anchor_records)
                        continue

                    mark_unit_in_progress(
                        prep_manifest,
                        participant_unit_key("anchor_inventory", participant_id),
                        anchor_contract_fp,
                        {"participant_id": participant_id, "contract": anchor_payload},
                    )
                    prep_manifest["state"] = summarize_manifest_state(prep_manifest)
                    save_manifest(manifest_path, prep_manifest)

                    try:
                        if source_inventory_path:
                            if not source_inventory_path.exists():
                                raise FileNotFoundError(f"anchor inventory file not found: {source_inventory_path}")
                            anchor_records = load_anchor_inventory_records(source_inventory_path)
                        else:
                            prof_path = normalized_profile_files.get(participant_id) or str(paths["profile"])
                            role_summary = pipeline._role_summary_from_profile(prof_path) or participant_id
                            sel = select_idea_units(
                                client=pipeline.llm_client,
                                meeting_id=meeting_id,
                                participant_id=participant_id,
                                role_summary=role_summary,
                                transcript_table=table,
                                N=capa_N,
                                k=seed_k,
                                max_items=selector_max_items,
                            )
                            raw_units = sel.get("idea_units") or []
                            units = sorted(raw_units, key=lambda u: float(u.get("importance_score") or 0.0), reverse=True)
                            valid_units: List[Dict[str, Any]] = []
                            for unit in units:
                                ok, why, normalized_unit = normalize_selector_item(
                                    item=unit,
                                    id_to_idx=id_to_idx,
                                    transcript=full_transcript,
                                    represented_user=participant_id,
                                    k=seed_k,
                                )
                                if ok:
                                    valid_units.append(normalized_unit)
                                    if selector_max_items is not None and len(valid_units) >= selector_max_items:
                                        break
                                elif debug_episodes:
                                    print(f"     [DBG] DROP frozen anchor {participant_id}:{unit.get('unit_id')} -> {why}")
                            anchor_records = freeze_anchor_inventory_records(
                                meeting_id=meeting_id,
                                participant_id=participant_id,
                                units=valid_units,
                                id_to_idx=id_to_idx,
                                inventory_reference_k=seed_k,
                                selector=selector,
                            )

                        write_anchor_inventory_records(paths["anchor_inventory"], anchor_records)
                        anchor_inventory_files[participant_id] = str(paths["anchor_inventory"])
                        anchor_records_by_participant[participant_id] = anchor_records
                        anchor_inventory_counts[participant_id] = len(anchor_records)
                        mark_unit_complete(
                            prep_manifest,
                            participant_unit_key("anchor_inventory", participant_id),
                            anchor_contract_fp,
                            [paths["anchor_inventory"]],
                            metadata={"participant_id": participant_id, "contract": anchor_payload, "anchor_count": len(anchor_records)},
                        )
                        prep_manifest["state"] = summarize_manifest_state(prep_manifest)
                        save_manifest(manifest_path, prep_manifest)
                        print(f"     [OK] Saved {len(anchor_records)} frozen anchors for {participant_id} -> {paths['anchor_inventory']}")
                    except Exception as exc:
                        error = str(exc)
                        print(f"     [ERROR] Failed to freeze anchor inventory for {participant_id}: {error}")
                        mark_unit_failed(
                            prep_manifest,
                            participant_unit_key("anchor_inventory", participant_id),
                            anchor_contract_fp,
                            {"participant_id": participant_id, "error": error, "contract": anchor_payload},
                        )
                        prep_manifest["state"] = summarize_manifest_state(prep_manifest)
                        save_manifest(manifest_path, prep_manifest)
                        failed_units.append(participant_unit_key("anchor_inventory", participant_id))

            print("\n>> Generating CAPA episodes (CAPA mode)...", flush=True)
            episodes_to_generate: List[str] = []
            for participant_id in participants:
                paths = requested_paths[participant_id]
                normalized_profile_fp = fingerprint_file(paths["profile"])
                episode_payload, episode_contract_fp = build_episode_contract(
                    participant_id,
                    testing_segment_fingerprint,
                    normalized_profile_fp,
                    fingerprint_file(paths["briefing"]),
                    model_descriptor,
                    fingerprint_file(paths["anchor_inventory"]) if anchor_inventory_enabled else None,
                )
                episode_unit = (prep_manifest.get("units") or {}).get(participant_unit_key("episodes", participant_id))
                reusable_outputs = [paths["episodes"]]
                if anchor_inventory_enabled or selector in {"llm-idea", "llm-contrib"}:
                    reusable_outputs.append(paths["k_diagnostics"])
                if resume and not force and unit_is_reusable(episode_unit, episode_contract_fp, reusable_outputs):
                    print(f"   - Reusing CAPA episodes for {participant_id}")
                    capa_episode_files[participant_id] = str(paths["episodes"])
                    existing_episodes = safe_json_load(paths["episodes"], default=[])
                    capa_episode_counts[participant_id] = len(existing_episodes or [])
                    if paths["k_diagnostics"].exists():
                        k_diagnostics_files[participant_id] = str(paths["k_diagnostics"])
                    continue
                if not paths["profile"].exists():
                    error = "normalized_profile_missing"
                    print(f"   - [WARN] Skipping episode generation for {participant_id}: {error}")
                    mark_unit_failed(prep_manifest, participant_unit_key("episodes", participant_id), episode_contract_fp, {"participant_id": participant_id, "error": error, "contract": episode_payload})
                    prep_manifest["state"] = summarize_manifest_state(prep_manifest)
                    save_manifest(manifest_path, prep_manifest)
                    failed_units.append(participant_unit_key("episodes", participant_id))
                    continue
                if anchor_inventory_enabled and not requested_paths[participant_id]["anchor_inventory"].exists():
                    error = "anchor_inventory_missing"
                    print(f"   - [WARN] Skipping episode generation for {participant_id}: {error}")
                    mark_unit_failed(prep_manifest, participant_unit_key("episodes", participant_id), episode_contract_fp, {"participant_id": participant_id, "error": error, "contract": episode_payload})
                    prep_manifest["state"] = summarize_manifest_state(prep_manifest)
                    save_manifest(manifest_path, prep_manifest)
                    failed_units.append(participant_unit_key("episodes", participant_id))
                    continue
                if not paths["briefing"].exists():
                    error = "jit_briefing_missing"
                    print(f"   - [WARN] Skipping episode generation for {participant_id}: {error}")
                    mark_unit_failed(prep_manifest, participant_unit_key("episodes", participant_id), episode_contract_fp, {"participant_id": participant_id, "error": error, "contract": episode_payload})
                    prep_manifest["state"] = summarize_manifest_state(prep_manifest)
                    save_manifest(manifest_path, prep_manifest)
                    failed_units.append(participant_unit_key("episodes", participant_id))
                    continue
                episodes_to_generate.append(participant_id)
                mark_unit_in_progress(prep_manifest, participant_unit_key("episodes", participant_id), episode_contract_fp, {"participant_id": participant_id, "contract": episode_payload})
            prep_manifest["state"] = summarize_manifest_state(prep_manifest)
            save_manifest(manifest_path, prep_manifest)

            episodes_by_participant: Dict[str, List[CAPAEpisode]] = {}
            diagnostics_by_participant: Dict[str, Dict[str, Any]] = {}
            if episodes_to_generate:
                if anchor_inventory_enabled:
                    print(f"   - [frozen-anchor] Rebuilding CAPA episodes for {len(episodes_to_generate)} participant(s) (N={capa_N}, k={capa_k})...")
                    for participant_id in episodes_to_generate:
                        anchor_records = anchor_records_by_participant.get(participant_id)
                        if anchor_records is None:
                            anchor_records = load_anchor_inventory_records(requested_paths[participant_id]["anchor_inventory"])
                            anchor_records_by_participant[participant_id] = anchor_records
                        ranked_anchor_records = sorted(
                            anchor_records,
                            key=lambda record: float(record.get("importance_score") or 0.0),
                            reverse=True,
                        )
                        participant_episodes: List[CAPAEpisode] = []
                        raw_diagnostics: List[Dict[str, Any]] = []
                        kept_diagnostics: List[Dict[str, Any]] = []
                        for episode_rank, record in enumerate(ranked_anchor_records, start=1):
                            record = dict(record)
                            record["_episode_rank"] = episode_rank
                            rebuilt = rebuild_episode_from_anchor_record(
                                record=record,
                                transcript=full_transcript,
                                represented_user=participant_id,
                                N=capa_N,
                                k=capa_k,
                            )
                            diag_row = {
                                "participant_id": participant_id,
                                "meeting_id": meeting_id,
                                "anchor_utt_id": record.get("anchor_utt_id"),
                                "unit_id": record.get("unit_id"),
                                "status": rebuilt.get("status"),
                                "drop_reason": rebuilt.get("drop_reason"),
                                "cue_mode": rebuilt.get("cue_mode"),
                                "importance_score": record.get("importance_score"),
                                "preferred_cue_gap": rebuilt.get("preferred_cue_gap"),
                                "effective_cue_gap": rebuilt.get("effective_cue_gap"),
                                "cap_rank": None,
                            }
                            episode = rebuilt.get("episode")
                            if episode is not None:
                                if episode.cue_utt_id in full_utt_index:
                                    global_index = full_utt_index[episode.cue_utt_id]
                                    episode.cue_index = global_index
                                    refresh_episode_identity(episode)
                                gate_ok, gate_reason, gated_record = apply_official_episode_gate(
                                    participant_id=participant_id,
                                    selector_item=record,
                                    episode=episode,
                                )
                                diag_row.update({
                                    "quality_status": gated_record.get("quality_status"),
                                    "quality_drop_reason": gated_record.get("quality_drop_reason"),
                                    "target_available_to_delegate": gated_record.get("target_available_to_delegate"),
                                    "target_availability_reason": gated_record.get("target_availability_reason"),
                                    "target_availability_score": gated_record.get("target_availability_score"),
                                    "target_availability_verdict": gated_record.get("target_availability_verdict"),
                                    "target_availability_method": gated_record.get("target_availability_method"),
                                    "target_availability_evidence_ids": gated_record.get("target_availability_evidence_ids"),
                                    "target_availability_missing_facets": gated_record.get("target_availability_missing_facets"),
                                    "target_availability_rationale": gated_record.get("target_availability_rationale"),
                                    "context_already_covered": gated_record.get("context_already_covered"),
                                    "context_coverage_overlap": gated_record.get("context_coverage_overlap"),
                                    "response_substance_score": gated_record.get("response_substance_score"),
                                    "selector_response_utterance_ids": gated_record.get("selector_response_utterance_ids"),
                                    "selector_topic_summary": gated_record.get("selector_topic_summary"),
                                    "trimmed_response_utterance_ids": gated_record.get("trimmed_response_utterance_ids"),
                                    "expanded_response_utterance_ids": gated_record.get("expanded_response_utterance_ids"),
                                    "official_constituent_expansion_reason": gated_record.get("official_constituent_expansion_reason"),
                                    "official_response_id_count": gated_record.get("official_response_id_count"),
                                    "official_target_text": gated_record.get("official_target_text"),
                                    "canonical_target_proposition": gated_record.get("canonical_target_proposition"),
                                    "tested_facet_text": gated_record.get("tested_facet_text"),
                                    "summary_rewritten": gated_record.get("summary_rewritten"),
                                    "summary_rewrite_reason": gated_record.get("summary_rewrite_reason"),
                                    "response_window_trimmed": gated_record.get("response_window_trimmed"),
                                    "trimmed_response_outside_window": gated_record.get("trimmed_response_outside_window"),
                                    "dropped_response_outside_window": gated_record.get("dropped_response_outside_window"),
                                    "response_ids_outside_window": gated_record.get("response_ids_outside_window"),
                                })
                                diag_row["test_case_id"] = episode.test_case_id
                                diag_row["cue_test_case_id"] = episode.cue_test_case_id
                                diag_row["cue_anchor_id"] = episode.cue_test_case_id
                                diag_row["episode_k"] = int(episode.k)
                                if gate_ok:
                                    participant_episodes.append(episode)
                                    kept_diagnostics.append(diag_row)
                                else:
                                    diag_row["status"] = "dropped"
                                    diag_row["drop_reason"] = gate_reason
                            raw_diagnostics.append(diag_row)
                        participant_episodes = enforce_episode_cap(
                            participant_id=participant_id,
                            participant_episodes=participant_episodes,
                            kept_diagnostics=kept_diagnostics,
                            cap_per_participant=cap_per_participant,
                            debug_episodes=debug_episodes,
                        )
                        participant_episodes.sort(key=lambda e: (e.cue_index if e.cue_index is not None else 10**9))
                        episodes_by_participant[participant_id] = participant_episodes
                        summary = summarize_k_diagnostics(
                            participant_id=participant_id,
                            meeting_id=meeting_id,
                            k=capa_k,
                            frozen_anchor_count=len(ranked_anchor_records),
                            diagnostics=[
                                {
                                    **diag_row,
                                    "episode": next((ep for ep in participant_episodes if ep.test_case_id == diag_row.get("test_case_id")), None),
                                }
                                for diag_row in raw_diagnostics
                            ],
                        )
                        diagnostics_payload = {
                            "meeting_id": meeting_id,
                            "participant_id": participant_id,
                            "k": int(capa_k),
                            "inventory_reference_k": inventory_reference_k,
                            "summary": summary,
                            "diagnostics": raw_diagnostics,
                        }
                        diagnostics_by_participant[participant_id] = diagnostics_payload
                elif selector == "dense":
                    print(f"   - [dense] Generating CAPA episodes for {len(episodes_to_generate)} participant(s) (N={capa_N}, k={capa_k})...")
                    episodes_by_participant = generate_dense_episodes_multi_participant(
                        transcript=testing_segment,
                        participants=episodes_to_generate,
                        N=capa_N,
                        k=capa_k,
                        step=1,
                    )
                elif selector == "llm-idea":
                    print(f"   - [llm-idea] Selecting Idea Units per participant (N={capa_N}, k={capa_k})...")
                    from capa.cue_selector_llm import build_transcript_table
                    from capa.idea_unit_selector_llm import select_idea_units
                    from capa.episode_utils import build_episode_from_contrib, build_id_index, has_future_rep

                    start_uid = testing_segment[0]["utterance_id"]
                    end_uid = testing_segment[-1]["utterance_id"]
                    table = build_transcript_table(full_transcript, full_utt_index, start_uid, end_uid)
                    id_to_idx = build_id_index(full_transcript)
                    selector_max_items = selector_shortlist_size(cap_per_participant)

                    for participant_id in episodes_to_generate:
                        prof_path = normalized_profile_files.get(participant_id) or str(requested_paths[participant_id]["profile"])
                        role_summary = pipeline._role_summary_from_profile(prof_path) or participant_id
                        sel = select_idea_units(
                            client=pipeline.llm_client,
                            meeting_id=meeting_id,
                            participant_id=participant_id,
                            role_summary=role_summary,
                            transcript_table=table,
                            N=capa_N,
                            k=capa_k,
                            max_items=selector_max_items,
                        )
                        raw_units = sel.get("idea_units") or []
                        if debug_episodes:
                            print(f"     [DBG] {participant_id}: selector returned {len(raw_units)} idea_unit(s)")
                        units = sorted(raw_units, key=lambda u: float(u.get("importance_score") or 0.0), reverse=True)
                        participant_episodes: List[CAPAEpisode] = []
                        raw_diagnostics: List[Dict[str, Any]] = []
                        kept_diagnostics: List[Dict[str, Any]] = []
                        for episode_rank, unit in enumerate(units, start=1):
                            unit = dict(unit)
                            unit["_episode_rank"] = episode_rank
                            uid = unit.get("unit_id") or ""
                            if debug_episodes:
                                print(f"       [DBG] Normalize {participant_id}:{uid} cue={unit.get('cue_utterance_id')}@{unit.get('cue_idx')} -> first_resp_idx={unit.get('first_response_idx')}; imp={unit.get('importance_score')} type={unit.get('cue_type')}")
                            ok, why, norm_unit = normalize_selector_item(
                                item=unit,
                                id_to_idx=id_to_idx,
                                transcript=full_transcript,
                                represented_user=participant_id,
                                k=capa_k,
                            )
                            if not ok:
                                raw_diagnostics.append(
                                    build_selector_diag_row(
                                        participant_id=participant_id,
                                        meeting_id=meeting_id,
                                        selector_item=dict(unit),
                                        status="dropped",
                                        drop_reason=why,
                                        episode=None,
                                    )
                                )
                                if debug_episodes:
                                    print(f"       [DROP] {participant_id}:{uid} invalid idea_unit: {why}")
                                continue
                            ep = build_episode_from_contrib(
                                contrib=norm_unit,
                                transcript=full_transcript,
                                represented_user=participant_id,
                                N=capa_N,
                                k_used=capa_k,
                            )
                            try:
                                ep.idea_units = [IdeaUnit(**norm_unit)]
                            except Exception:
                                try:
                                    ep.idea_units = [IdeaUnit(**norm_unit)]
                                except Exception:
                                    ep.idea_units = [norm_unit]
                            gate_ok, gate_reason, gated_unit = apply_official_episode_gate(
                                participant_id=participant_id,
                                selector_item=norm_unit,
                                episode=ep,
                            )
                            norm_unit.update(gated_unit)
                            if not gate_ok:
                                raw_diagnostics.append(
                                    build_selector_diag_row(
                                        participant_id=participant_id,
                                        meeting_id=meeting_id,
                                        selector_item=gated_unit,
                                        status="dropped",
                                        drop_reason=gate_reason,
                                        episode=ep,
                                    )
                                )
                                if debug_episodes:
                                    print(f"       [DROP] {participant_id}:{uid} quality/availability gate: {gate_reason}")
                                continue

                            present, contentful, _ = has_future_rep(ep.gt_next_window or [], participant_id)
                            if present and contentful:
                                participant_episodes.append(ep)
                                diag_row = build_selector_diag_row(
                                    participant_id=participant_id,
                                    meeting_id=meeting_id,
                                    selector_item=gated_unit,
                                    status="kept",
                                    drop_reason=None,
                                    episode=ep,
                                )
                                raw_diagnostics.append(diag_row)
                                kept_diagnostics.append(diag_row)
                                if debug_episodes:
                                    print(f"       [KEEP] {participant_id}:{uid} present+contentful in k_window (len={len(ep.gt_next_window or [])})")
                            else:
                                raw_diagnostics.append(
                                    build_selector_diag_row(
                                        participant_id=participant_id,
                                        meeting_id=meeting_id,
                                        selector_item=gated_unit,
                                        status="dropped",
                                        drop_reason="not_present_or_not_contentful",
                                        episode=ep,
                                    )
                                )
                                if debug_episodes:
                                    print(f"       [DROP] {participant_id}:{uid} present={present} contentful={contentful}")
                        participant_episodes = enforce_episode_cap(
                            participant_id=participant_id,
                            participant_episodes=participant_episodes,
                            kept_diagnostics=kept_diagnostics,
                            cap_per_participant=cap_per_participant,
                            debug_episodes=debug_episodes,
                        )
                        episodes_by_participant[participant_id] = participant_episodes
                        summary = summarize_k_diagnostics(
                            participant_id=participant_id,
                            meeting_id=meeting_id,
                            k=capa_k,
                            frozen_anchor_count=len(units),
                            diagnostics=raw_diagnostics,
                        )
                        diagnostics_by_participant[participant_id] = {
                            "meeting_id": meeting_id,
                            "participant_id": participant_id,
                            "k": int(capa_k),
                            "inventory_reference_k": inventory_reference_k,
                            "summary": summary,
                            "diagnostics": raw_diagnostics,
                        }
                        if debug_episodes:
                            print(f"     [DBG] {participant_id}: kept {len(participant_episodes)}/{len(units)} after validation/contentfulness/cap")
                else:
                    print(f"   - [llm-contrib] Selecting contribution spans and cues per participant (N={capa_N}, k={capa_k})...")
                    from capa.contribution_selector_llm import select_contributions
                    from capa.cue_selector_llm import build_transcript_table
                    from capa.episode_utils import build_episode_from_contrib, build_id_index, has_future_rep

                    start_uid = testing_segment[0]["utterance_id"]
                    end_uid = testing_segment[-1]["utterance_id"]
                    table = build_transcript_table(full_transcript, full_utt_index, start_uid, end_uid)
                    id_to_idx = build_id_index(full_transcript)
                    selector_max_items = selector_shortlist_size(cap_per_participant)
                    for participant_id in episodes_to_generate:
                        prof_path = normalized_profile_files.get(participant_id) or str(requested_paths[participant_id]["profile"])
                        role_summary = pipeline._role_summary_from_profile(prof_path) or participant_id
                        sel = select_contributions(
                            client=pipeline.llm_client,
                            meeting_id=meeting_id,
                            participant_id=participant_id,
                            role_summary=role_summary,
                            transcript_table=table,
                            N=capa_N,
                            k=capa_k,
                            max_items=selector_max_items,
                        )
                        raw_contribs = sel.get("contributions") or []
                        contribs = sorted(raw_contribs, key=lambda c: float(c.get("importance_score") or 0.0), reverse=True)
                        participant_episodes: List[CAPAEpisode] = []
                        raw_diagnostics: List[Dict[str, Any]] = []
                        kept_diagnostics: List[Dict[str, Any]] = []
                        for episode_rank, contrib in enumerate(contribs, start=1):
                            contrib = dict(contrib)
                            contrib["_episode_rank"] = episode_rank
                            ok, why, norm_contrib = normalize_selector_item(
                                item=contrib,
                                id_to_idx=id_to_idx,
                                transcript=full_transcript,
                                represented_user=participant_id,
                                k=capa_k,
                            )
                            if not ok:
                                raw_diagnostics.append(
                                    build_selector_diag_row(
                                        participant_id=participant_id,
                                        meeting_id=meeting_id,
                                        selector_item=dict(contrib),
                                        status="dropped",
                                        drop_reason=why,
                                        episode=None,
                                    )
                                )
                                continue
                            ep = build_episode_from_contrib(
                                contrib=norm_contrib,
                                transcript=full_transcript,
                                represented_user=participant_id,
                                N=capa_N,
                                k_used=capa_k,
                            )
                            gate_ok, gate_reason, gated_contrib = apply_official_episode_gate(
                                participant_id=participant_id,
                                selector_item=norm_contrib,
                                episode=ep,
                            )
                            norm_contrib.update(gated_contrib)
                            if not gate_ok:
                                raw_diagnostics.append(
                                    build_selector_diag_row(
                                        participant_id=participant_id,
                                        meeting_id=meeting_id,
                                        selector_item=gated_contrib,
                                        status="dropped",
                                        drop_reason=gate_reason,
                                        episode=ep,
                                    )
                                )
                                continue

                            present, contentful, _ = has_future_rep(ep.gt_next_window or [], participant_id)
                            if present and contentful:
                                participant_episodes.append(ep)
                                diag_row = build_selector_diag_row(
                                    participant_id=participant_id,
                                    meeting_id=meeting_id,
                                    selector_item=gated_contrib,
                                    status="kept",
                                    drop_reason=None,
                                    episode=ep,
                                )
                                raw_diagnostics.append(diag_row)
                                kept_diagnostics.append(diag_row)
                            else:
                                raw_diagnostics.append(
                                    build_selector_diag_row(
                                        participant_id=participant_id,
                                        meeting_id=meeting_id,
                                        selector_item=gated_contrib,
                                        status="dropped",
                                        drop_reason="not_present_or_not_contentful",
                                        episode=ep,
                                    )
                                )
                        participant_episodes = enforce_episode_cap(
                            participant_id=participant_id,
                            participant_episodes=participant_episodes,
                            kept_diagnostics=kept_diagnostics,
                            cap_per_participant=cap_per_participant,
                            debug_episodes=debug_episodes,
                        )
                        episodes_by_participant[participant_id] = participant_episodes
                        summary = summarize_k_diagnostics(
                            participant_id=participant_id,
                            meeting_id=meeting_id,
                            k=capa_k,
                            frozen_anchor_count=len(contribs),
                            diagnostics=raw_diagnostics,
                        )
                        diagnostics_by_participant[participant_id] = {
                            "meeting_id": meeting_id,
                            "participant_id": participant_id,
                            "k": int(capa_k),
                            "inventory_reference_k": inventory_reference_k,
                            "summary": summary,
                            "diagnostics": raw_diagnostics,
                        }

                for participant_id, participant_episodes in episodes_by_participant.items():
                    for episode in participant_episodes:
                        refresh_episode_identity(episode)
                assert_unique_episode_ids(episodes_by_participant, meeting_id=meeting_id)

                for participant_id in episodes_to_generate:
                    episode_payload, episode_contract_fp = build_episode_contract(
                        participant_id,
                        testing_segment_fingerprint,
                        fingerprint_file(requested_paths[participant_id]["profile"]),
                        fingerprint_file(requested_paths[participant_id]["briefing"]),
                        model_descriptor,
                        fingerprint_file(requested_paths[participant_id]["anchor_inventory"]) if anchor_inventory_enabled else None,
                    )
                    try:
                        participant_episodes = episodes_by_participant.get(participant_id, [])
                        for ep in participant_episodes:
                            if ep.cue_utt_id in full_utt_index:
                                global_index = full_utt_index[ep.cue_utt_id]
                                ep.cue_index = global_index
                                refresh_episode_identity(ep)
                        participant_episodes.sort(key=lambda e: (e.cue_index if e.cue_index is not None else 10**9))
                        write_episodes_to_file(participant_episodes, str(requested_paths[participant_id]["episodes"]))
                        capa_episode_files[participant_id] = str(requested_paths[participant_id]["episodes"])
                        capa_episode_counts[participant_id] = len(participant_episodes)
                        output_paths = [requested_paths[participant_id]["episodes"]]
                        diagnostics_payload = diagnostics_by_participant.get(participant_id) or {}
                        if diagnostics_payload:
                            atomic_write_json(requested_paths[participant_id]["k_diagnostics"], diagnostics_payload)
                            k_diagnostics_files[participant_id] = str(requested_paths[participant_id]["k_diagnostics"])
                            output_paths.append(requested_paths[participant_id]["k_diagnostics"])
                        mark_unit_complete(
                            prep_manifest,
                            participant_unit_key("episodes", participant_id),
                            episode_contract_fp,
                            output_paths,
                            metadata={"participant_id": participant_id, "contract": episode_payload, "episode_count": len(participant_episodes)},
                        )
                        prep_manifest["state"] = summarize_manifest_state(prep_manifest)
                        save_manifest(manifest_path, prep_manifest)
                        print(f"     [OK] Saved {len(participant_episodes)} CAPA episodes for {participant_id} -> {requested_paths[participant_id]['episodes']}")
                    except Exception as exc:
                        error = str(exc)
                        print(f"     [ERROR] Failed to save CAPA episodes for {participant_id}: {error}")
                        mark_unit_failed(prep_manifest, participant_unit_key("episodes", participant_id), episode_contract_fp, {"participant_id": participant_id, "error": error, "contract": episode_payload})
                        prep_manifest["state"] = summarize_manifest_state(prep_manifest)
                        save_manifest(manifest_path, prep_manifest)
                        failed_units.append(participant_unit_key("episodes", participant_id))
            if anchor_inventory_enabled or selector in {"llm-idea", "llm-contrib"}:
                refresh_meeting_k_diagnostics_summary()

            def compute_incremental_eim(stale_participants: List[str]) -> None:
                if not stale_participants:
                    return
                episodes_by_participant_for_eim: Dict[str, List[CAPAEpisode]] = {}
                for participant_id in stale_participants:
                    payload = safe_json_load(requested_paths[participant_id]["episodes"], default=[])
                    episodes_by_participant_for_eim[participant_id] = [CAPAEpisode.model_validate(ep) for ep in (payload or [])]

                unique_windows: Dict[str, Dict[str, Any]] = {}
                consumers: Dict[str, List[Tuple[str, str]]] = {}
                for participant_id, episodes in episodes_by_participant_for_eim.items():
                    for ep in episodes:
                        window_entry = build_window_cache_entry(ep, model_descriptor)
                        window_key = window_entry["window_key"]
                        if window_key not in unique_windows:
                            unique_windows[window_key] = window_entry
                            consumers[window_key] = []
                        consumers[window_key].append((participant_id, ep.test_case_id))

                window_outputs: Dict[str, Dict[str, Any]] = {}
                with file_lock(eim_cache_lock_path):
                    cache_manifest = read_cache_manifest()
                    missing_windows: List[Dict[str, Any]] = []
                    for window_key, window_entry in unique_windows.items():
                        cached_payload = load_cached_window_vector(window_entry)
                        if cached_payload:
                            window_outputs[window_key] = cached_payload["eim_output"]
                        else:
                            missing_windows.append(window_entry)

                    if missing_windows:
                        missing_windows.sort(key=lambda entry: (entry["cue_idx"], entry["test_case_id"]))
                        first_missing_cue_idx = missing_windows[0]["cue_idx"]
                        bootstrap_idx, bootstrap_source = restore_best_eim_bootstrap(cache_manifest, first_missing_cue_idx, model_descriptor)
                        profile_processed_idx = len(profile_segment) - 1
                        current_processed_idx = bootstrap_idx
                        if current_processed_idx < profile_processed_idx:
                            pipeline.eim.reset_master_state()
                            pipeline.eim.process_history(profile_segment)
                            current_processed_idx = profile_processed_idx
                            write_history_checkpoint(cache_manifest, current_processed_idx, full_transcript, model_descriptor, "profile_segment")
                        last_checkpoint_bucket = current_processed_idx // EIM_CHECKPOINT_STRIDE if current_processed_idx >= 0 else -1
                        if bootstrap_source != "fresh":
                            print(f"   - Restored EIM bootstrap from {bootstrap_source}")
                        for window_entry in missing_windows:
                            history_target = window_entry["history_processed_through_idx"]
                            if history_target > current_processed_idx:
                                history_chunk = full_transcript[current_processed_idx + 1: history_target + 1]
                                if history_chunk:
                                    pipeline.eim.process_history(history_chunk)
                                current_processed_idx = history_target
                                current_bucket = current_processed_idx // EIM_CHECKPOINT_STRIDE if current_processed_idx >= 0 else -1
                                if current_bucket > last_checkpoint_bucket:
                                    write_history_checkpoint(cache_manifest, current_processed_idx, full_transcript, model_descriptor, f"history_stride_{current_processed_idx}")
                                    last_checkpoint_bucket = current_bucket
                            vec = pipeline.eim.analyze_snapshot(window_entry["context"])
                            if vec is None:
                                raise RuntimeError(f"eim_snapshot_failed:{window_entry['test_case_id']}")
                            eim_output = vec.model_dump()
                            write_cached_window_vector(cache_manifest, window_entry, eim_output)
                            window_outputs[window_entry["window_key"]] = eim_output
                        write_cache_manifest(cache_manifest)

                for participant_id in stale_participants:
                    entries = []
                    for ep in episodes_by_participant_for_eim.get(participant_id, []):
                        window_entry = build_window_cache_entry(ep, model_descriptor)
                        eim_output = window_outputs.get(window_entry["window_key"])
                        if eim_output is None:
                            continue
                        entries.append({"test_case_id": ep.test_case_id, "eim_output": eim_output})
                    atomic_write_json(requested_paths[participant_id]["eim"], entries)
                    capa_eim_files[participant_id] = str(requested_paths[participant_id]["eim"])
                    capa_eim_counts[participant_id] = len(entries)

            print("\n>> Precomputing CAPA EIM states...", flush=True)
            stale_eim_participants: List[str] = []
            for participant_id in participants:
                paths = requested_paths[participant_id]
                episode_file_fp = fingerprint_file(paths["episodes"])
                eim_payload, eim_contract_fp = build_eim_contract(participant_id, episode_file_fp, model_descriptor)
                eim_unit = (prep_manifest.get("units") or {}).get(participant_unit_key("eim", participant_id))
                if resume and not force and unit_is_reusable(eim_unit, eim_contract_fp, [paths["eim"]]):
                    print(f"   - Reusing CAPA EIM states for {participant_id}")
                    capa_eim_files[participant_id] = str(paths["eim"])
                    existing_states = safe_json_load(paths["eim"], default=[])
                    capa_eim_counts[participant_id] = len(existing_states or [])
                    continue
                if not paths["episodes"].exists():
                    error = "episodes_missing"
                    print(f"   - [WARN] Skipping EIM generation for {participant_id}: {error}")
                    mark_unit_failed(prep_manifest, participant_unit_key("eim", participant_id), eim_contract_fp, {"participant_id": participant_id, "error": error, "contract": eim_payload})
                    prep_manifest["state"] = summarize_manifest_state(prep_manifest)
                    save_manifest(manifest_path, prep_manifest)
                    failed_units.append(participant_unit_key("eim", participant_id))
                    continue
                stale_eim_participants.append(participant_id)
                mark_unit_in_progress(prep_manifest, participant_unit_key("eim", participant_id), eim_contract_fp, {"participant_id": participant_id, "contract": eim_payload})
            prep_manifest["state"] = summarize_manifest_state(prep_manifest)
            save_manifest(manifest_path, prep_manifest)
            try:
                compute_incremental_eim(stale_eim_participants)
                for participant_id in stale_eim_participants:
                    eim_payload, eim_contract_fp = build_eim_contract(participant_id, fingerprint_file(requested_paths[participant_id]["episodes"]), model_descriptor)
                    mark_unit_complete(
                        prep_manifest,
                        participant_unit_key("eim", participant_id),
                        eim_contract_fp,
                        [requested_paths[participant_id]["eim"]],
                        metadata={"participant_id": participant_id, "contract": eim_payload, "state_count": capa_eim_counts.get(participant_id, 0)},
                    )
                prep_manifest["state"] = summarize_manifest_state(prep_manifest)
                save_manifest(manifest_path, prep_manifest)
                for participant_id in stale_eim_participants:
                    print(f"     [OK] Saved {capa_eim_counts.get(participant_id, 0)} CAPA EIM states for {participant_id} -> {requested_paths[participant_id]['eim']}")
            except Exception as exc:
                error = str(exc)
                print(f"   - [ERROR] Incremental EIM generation failed: {error}")
                for participant_id in stale_eim_participants:
                    eim_payload, eim_contract_fp = build_eim_contract(participant_id, fingerprint_file(requested_paths[participant_id]["episodes"]), model_descriptor)
                    mark_unit_failed(prep_manifest, participant_unit_key("eim", participant_id), eim_contract_fp, {"participant_id": participant_id, "error": error, "contract": eim_payload})
                    failed_units.append(participant_unit_key("eim", participant_id))
                prep_manifest["state"] = summarize_manifest_state(prep_manifest)
                save_manifest(manifest_path, prep_manifest)

            results["generated_files"]["profiles"] = profile_files
            results["generated_files"]["capa_user_profiles"] = normalized_profile_files
            if anchor_inventory_enabled:
                results["generated_files"]["capa_anchor_inventory"] = anchor_inventory_files
                results["generated_files"]["capa_k_diagnostics"] = k_diagnostics_files
            results["generated_files"]["capa_episodes"] = capa_episode_files
            results["generated_files"]["capa_eim_states"] = capa_eim_files
            results["generated_files"]["briefings"] = briefing_files
            results["processing_stats"] = {
                "profiles_generated": len(profile_files),
                "capa_profiles_normalized": len(normalized_profile_files),
                "briefings_generated": len(briefing_files),
                "capa_anchor_inventory_generated": sum(anchor_inventory_counts.values()),
                "capa_episodes_generated": sum(capa_episode_counts.values()),
                "capa_eim_states_generated": sum(capa_eim_counts.values()),
            }

            prep_manifest["state"] = summarize_manifest_state(prep_manifest)
            save_manifest(manifest_path, prep_manifest)

        print(f"\n{'=' * 60}")
        print(f"PIPELINE COMPLETE FOR {meeting_id}")
        print(f"{'=' * 60}")
        print(f"[OK] User Profiles: {results['processing_stats']['profiles_generated']}/{len(participants)}")
        if anchor_inventory_enabled:
            print(f"[OK] Frozen Anchors: {results['processing_stats'].get('capa_anchor_inventory_generated', 0)} total")
        print(f"[OK] CAPA Episodes: {results['processing_stats'].get('capa_episodes_generated', 0)} total")
        print(f"[OK] JIT Briefings: {results['processing_stats']['briefings_generated']}/{len(participants)}")
        print(f"[OK] CAPA EIM States: {results['processing_stats'].get('capa_eim_states_generated', 0)} total")

        if failed_units:
            results["success"] = False
            results["error"] = f"Stage 1 units failed: {', '.join(sorted(set(failed_units)))}"
        else:
            results["success"] = True
        return results
    except Exception as exc:
        results["success"] = False
        results["error"] = str(exc)
        return results
