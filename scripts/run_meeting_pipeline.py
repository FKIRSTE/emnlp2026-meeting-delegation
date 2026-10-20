#!/usr/bin/env python
"""
run_meeting_pipeline.py — Single Meeting E2E Orchestrator

Runs the complete pipeline for one meeting:
1. Data generation (shared Stage 1 prep)
2. Simulation for each participant (variant-specific)
3. Evaluation (variant-specific)
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from raw_baseline_support import (
    build_delegate_capabilities,
    build_delegate_runtime_metadata,
    is_raw_baseline_variant,
    normalize_raw_context_n,
)
from pipeline_resume import (
    EVAL_MANIFEST_VERSION,
    EVAL_STAGE_VERSION,
    PIPELINE_RUNTIME_VERSION,
    PREP_MANIFEST_VERSION,
    SIMULATION_MANIFEST_VERSION,
    atomic_write_json,
    fingerprint_contract,
    fingerprint_file,
    load_manifest,
    summarize_manifest_state,
    unit_is_reusable,
)
from prep_resume import EPISODE_NORMALIZATION_VERSION


def setup_logging(output_dir: Path, meeting_id: str) -> logging.Logger:
    """Setup dual logging: console + file."""
    log_file = output_dir / "pipeline.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"pipeline.{meeting_id}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    return logger


def extract_meeting_id(input_path: Path) -> str:
    name = input_path.stem
    if name.endswith("_parsed"):
        name = name[:-7]
    return name


def extract_participants(input_path: Path) -> List[str]:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        participants = {
            utt.get("speaker") or utt.get("participant_id") or utt.get("speaker_id")
            for utt in data
            if isinstance(utt, dict)
        }
        return sorted(p for p in participants if p)

    if isinstance(data, dict) and "participants" in data:
        return list(data["participants"].keys())

    if isinstance(data, dict) and "utterances" in data:
        participants = {
            utt.get("speaker") or utt.get("participant_id") or utt.get("speaker_id")
            for utt in data["utterances"]
            if isinstance(utt, dict)
        }
        return sorted(p for p in participants if p)

    return []


def materialize_debug_truncated_input(input_path: Path, output_dir: Path, max_utterances: int, logger: logging.Logger) -> Path:
    if max_utterances <= 0:
        return input_path

    with open(input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    truncated_payload: Any = payload
    original_len = None

    if isinstance(payload, list):
        original_len = len(payload)
        if original_len > max_utterances:
            truncated_payload = payload[:max_utterances]
    elif isinstance(payload, dict):
        if isinstance(payload.get("utterances"), list):
            original_len = len(payload["utterances"])
            if original_len > max_utterances:
                truncated_payload = dict(payload)
                truncated_payload["utterances"] = list(payload["utterances"][:max_utterances])
        elif isinstance(payload.get("transcript"), list):
            original_len = len(payload["transcript"])
            if original_len > max_utterances:
                truncated_payload = dict(payload)
                truncated_payload["transcript"] = list(payload["transcript"][:max_utterances])

    if original_len is None or original_len <= max_utterances:
        return input_path

    debug_path = output_dir / f"{input_path.stem}__DEBUG_first_{max_utterances}.json"
    should_write = True
    if debug_path.exists():
        try:
            with open(debug_path, "r", encoding="utf-8") as f:
                existing_payload = json.load(f)
            should_write = existing_payload != truncated_payload
        except Exception:
            should_write = True

    if should_write:
        atomic_write_json(debug_path, truncated_payload)
    logger.warning(
        "[DEBUG ONLY] Using first %s utterances from %s via %s. This only happens because --debug-max-utterances was passed.",
        max_utterances,
        input_path,
        debug_path,
    )
    return debug_path


def make_stage_result(
    ok: bool,
    *,
    returncode: Optional[int] = None,
    timed_out: bool = False,
    failure_kind: Optional[str] = None,
    message: Optional[str] = None,
    command: Optional[List[str]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "ok": ok,
        "returncode": returncode,
        "timed_out": timed_out,
        "failure_kind": failure_kind,
        "message": message,
    }
    if command is not None:
        payload["command"] = list(command)
    return payload



def run_command(cmd: List[str], logger: logging.Logger, stage: str, timeout_minutes: int = 180) -> Dict[str, Any]:
    logger.info(f"[{stage}] Running: {' '.join(cmd)}")

    log_file_handle = None
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.flush()
            log_file_handle = open(handler.baseFilename, "a", encoding="utf-8")
            break

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    try:
        if log_file_handle:
            log_file_handle.write(f"\n{'=' * 60}\n[{stage}] SUBPROCESS OUTPUT START\n{'=' * 60}\n")
            log_file_handle.flush()
            result = subprocess.run(
                cmd,
                stdout=log_file_handle,
                stderr=subprocess.STDOUT,
                timeout=timeout_minutes * 60,
                env=env,
            )
            log_file_handle.write(f"\n{'=' * 60}\n[{stage}] SUBPROCESS OUTPUT END\n{'=' * 60}\n")
            log_file_handle.flush()
            log_file_handle.close()
        else:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_minutes * 60,
                env=env,
            )
            if result.stdout:
                logger.debug(f"[{stage}] stdout:\n{result.stdout}")
            if result.stderr:
                logger.debug(f"[{stage}] stderr:\n{result.stderr}")

        if result.returncode != 0:
            logger.error(f"[{stage}] Failed with return code {result.returncode}")
            return make_stage_result(
                False,
                returncode=result.returncode,
                failure_kind="subprocess_error",
                message=f"{stage} subprocess exited with return code {result.returncode}",
                command=cmd,
            )

        logger.info(f"[{stage}] Completed successfully")
        return make_stage_result(True, returncode=0, message=f"{stage} completed successfully", command=cmd)
    except subprocess.TimeoutExpired:
        logger.error(f"[{stage}] TIMEOUT after {timeout_minutes} minutes")
        if log_file_handle and not log_file_handle.closed:
            log_file_handle.write(f"\n[{stage}] TIMEOUT - subprocess killed\n")
            log_file_handle.close()
        return make_stage_result(
            False,
            timed_out=True,
            failure_kind="timeout",
            message=f"{stage} timed out after {timeout_minutes} minutes",
            command=cmd,
        )
    except Exception as exc:
        logger.error(f"[{stage}] Exception: {exc}")
        if log_file_handle and not log_file_handle.closed:
            log_file_handle.close()
        return make_stage_result(
            False,
            failure_kind="exception",
            message=str(exc),
            command=cmd,
        )


# =============================================================================
# Resume helpers
# =============================================================================

def get_file_mtime(path: Path) -> Optional[float]:
    return path.stat().st_mtime if path.exists() else None


def get_latest_mtime(files: List[Path]) -> Optional[float]:
    mtimes = [get_file_mtime(path) for path in files if path.exists()]
    return max(mtimes) if mtimes else None


def load_pipeline_status(output_dir: Path) -> Dict[str, Any]:
    status_file = output_dir / "pipeline_status.json"
    if not status_file.exists():
        return {}
    try:
        with open(status_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _unique_participants(participants: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for participant in participants:
        if not participant or participant in seen:
            continue
        seen.add(participant)
        ordered.append(participant)
    return ordered


def discover_complete_simulation_participants(output_dir: Path) -> List[str]:
    manifest = load_manifest(output_dir / "simulation_manifest.json", default_version=SIMULATION_MANIFEST_VERSION)
    participants: List[str] = []
    for unit_key, unit in (manifest.get("units") or {}).items():
        if not str(unit_key).startswith("participant::"):
            continue
        if (unit or {}).get("state") != "complete":
            continue
        output_fingerprints = (unit or {}).get("output_fingerprints") or {}
        if not output_fingerprints:
            continue
        outputs_ok = True
        for path_str, expected_fingerprint in output_fingerprints.items():
            current_fingerprint = fingerprint_file(Path(path_str))
            if current_fingerprint is None or (expected_fingerprint and current_fingerprint != expected_fingerprint):
                outputs_ok = False
                break
        if not outputs_ok:
            continue
        metadata = (unit or {}).get("metadata") or {}
        participant_id = metadata.get("participant_id") or str(unit_key).split("participant::", 1)[-1]
        participants.append(participant_id)
    return sorted(_unique_participants(participants))


def resolve_pipeline_participants(output_dir: Path, requested_participants: List[str], *, include_requested: bool) -> List[str]:
    effective = discover_complete_simulation_participants(output_dir)
    if include_requested:
        effective = _unique_participants([*effective, *requested_participants])
    return effective or list(requested_participants)


def discover_simulation_participant_states(output_dir: Path, requested_participants: List[str]) -> Dict[str, List[str]]:
    manifest = load_manifest(output_dir / "simulation_manifest.json", default_version=SIMULATION_MANIFEST_VERSION)
    completed = set(discover_complete_simulation_participants(output_dir))
    failed: List[str] = []
    in_progress: List[str] = []
    missing: List[str] = []

    for participant in requested_participants:
        if participant in completed:
            continue
        unit = (manifest.get("units") or {}).get(f"participant::{participant}") or {}
        state = unit.get("state")
        if state == "failed":
            failed.append(participant)
        elif state == "in_progress":
            in_progress.append(participant)
        else:
            missing.append(participant)

    return {
        "complete": sorted(p for p in requested_participants if p in completed),
        "failed": failed,
        "in_progress": in_progress,
        "missing": missing,
        "incomplete": [*failed, *in_progress, *missing],
    }


def write_pipeline_status(
    output_dir: Path,
    meeting_id: str,
    prep_dir: Path,
    participants: List[str],
    args: argparse.Namespace,
    status: str,
    *,
    effective_participants: Optional[List[str]] = None,
    stage_states: Optional[Dict[str, Any]] = None,
    current_unit: Optional[str] = None,
    elapsed_seconds: Optional[float] = None,
    retryable: Optional[bool] = None,
) -> None:
    capability_flags = build_delegate_capabilities(args.delegate_variant, args.no_recalibration)
    runtime_metadata = build_delegate_runtime_metadata(args.delegate_variant, args.raw_context_N, args.capa_k)
    requested_participants = list(participants)
    effective_participants = _unique_participants(effective_participants or requested_participants)
    payload = {
        "meeting_id": meeting_id,
        "status": status,
        "results_dir": str(output_dir),
        "shared_prep_dir": str(prep_dir),
        "participants": effective_participants,
        "requested_participants": requested_participants,
        "effective_participants": effective_participants,
        "model": args.model,
        "capa_N": args.capa_N,
        "capa_k": args.capa_k,
        "freeze_anchor_inventory": bool(getattr(args, "freeze_anchor_inventory", False)),
        "anchor_seed_k": getattr(args, "anchor_seed_k", None),
        "anchor_inventory_dir": str(getattr(args, "anchor_inventory_dir", "") or ""),
        "static_prep_source_dir": str(getattr(args, "static_prep_source_dir", "") or ""),
        "eim_cache_dir": str(getattr(args, "eim_cache_dir", "") or ""),
        "stage_states": stage_states or {},
        "current_unit": current_unit,
        "pipeline_runtime_version": PIPELINE_RUNTIME_VERSION,
        "episode_normalization_version": EPISODE_NORMALIZATION_VERSION,
        **capability_flags,
        **runtime_metadata,
    }
    if elapsed_seconds is not None:
        payload["elapsed_seconds"] = elapsed_seconds
    if retryable is not None:
        payload["retryable"] = retryable
    atomic_write_json(output_dir / "pipeline_status.json", payload)


def build_eval_contract(
    output_dir: Path,
    prep_dir: Path,
    meeting_id: str,
    participants: List[str],
    args: argparse.Namespace,
    *,
    requested_participants: Optional[List[str]] = None,
    evaluation_state: str = "complete",
    missing_participants: Optional[List[str]] = None,
) -> Dict[str, Any]:
    sim_files: Dict[str, Optional[str]] = {}
    artifact_files: Dict[str, Optional[str]] = {}
    episode_files: Dict[str, Optional[str]] = {}
    for participant in participants:
        slug = participant.replace(" ", "_")
        sim_path = output_dir / f"capa_sim_{slug}_{meeting_id}.json"
        artifact_path = output_dir / "artifacts" / f"capa_artifacts_{participant}_{meeting_id}.json"
        episode_path = prep_dir / "capa_episodes" / f"capa_episodes_for_{slug}_from_{meeting_id}.json"
        sim_files[str(sim_path)] = fingerprint_file(sim_path)
        artifact_files[str(artifact_path)] = fingerprint_file(artifact_path)
        episode_files[str(episode_path)] = fingerprint_file(episode_path)
    requested = list(requested_participants or participants)
    effective = list(participants)
    missing = list(missing_participants or [p for p in requested if p not in effective])
    payload = {
        "stage": "evaluation",
        "stage_version": EVAL_STAGE_VERSION,
        "meeting_id": meeting_id,
        "participants": effective,
        "requested_participants": requested,
        "effective_participants": effective,
        "missing_participants": missing,
        "evaluation_state": evaluation_state,
        "mode": "idea",
        "ctx_strict": args.ctx_strict,
        "rel_strict": args.rel_strict,
        "ctx_loose": args.ctx_loose,
        "rel_loose": args.rel_loose,
        "capa_k": args.capa_k,
        "sim_files": sim_files,
        "artifact_files": artifact_files,
        "episode_files": episode_files,
        "delegate_variant": args.delegate_variant,
        "model": args.model,
        "freeze_anchor_inventory": bool(getattr(args, "freeze_anchor_inventory", False)),
        "anchor_seed_k": getattr(args, "anchor_seed_k", None),
        "anchor_inventory_dir": str(getattr(args, "anchor_inventory_dir", "") or ""),
        "episode_normalization_version": EPISODE_NORMALIZATION_VERSION,
        **build_delegate_runtime_metadata(args.delegate_variant, args.raw_context_N, args.capa_k),
    }
    return {
        "payload": payload,
        "fingerprint": fingerprint_contract(**payload),
    }


def check_eval_manifest_complete(output_dir: Path, meeting_id: str, participants: List[str], prep_dir: Path, args: argparse.Namespace, logger: logging.Logger) -> bool:
    eval_file = output_dir / "evals" / meeting_id / "meeting_metrics.json"
    manifest_path = output_dir / "eval_manifest.json"
    if not eval_file.exists():
        return False
    manifest = load_manifest(manifest_path, default_version=EVAL_MANIFEST_VERSION)
    unit = (manifest.get("units") or {}).get("meeting") or {}
    if unit.get("state") != "complete":
        return False
    effective_participants = resolve_pipeline_participants(output_dir, participants, include_requested=False)
    contract = build_eval_contract(
        output_dir,
        prep_dir,
        meeting_id,
        effective_participants,
        args,
        requested_participants=participants,
        evaluation_state="complete",
        missing_participants=[p for p in participants if p not in effective_participants],
    )
    reusable = unit_is_reusable(unit, contract["fingerprint"], [eval_file])
    if reusable:
        logger.debug("[AutoResume] Stage 3 complete via eval_manifest.json")
    return reusable


def write_eval_manifest(
    output_dir: Path,
    meeting_id: str,
    participants: List[str],
    prep_dir: Path,
    args: argparse.Namespace,
    *,
    evaluation_state: str = "complete",
    effective_participants: Optional[List[str]] = None,
    missing_participants: Optional[List[str]] = None,
) -> None:
    manifest_path = output_dir / "eval_manifest.json"
    manifest = load_manifest(manifest_path, default_version=EVAL_MANIFEST_VERSION)
    effective = list(effective_participants or resolve_pipeline_participants(output_dir, participants, include_requested=False))
    missing = list(missing_participants or [p for p in participants if p not in effective])
    contract = build_eval_contract(
        output_dir,
        prep_dir,
        meeting_id,
        effective,
        args,
        requested_participants=participants,
        evaluation_state=evaluation_state,
        missing_participants=missing,
    )
    eval_file = output_dir / "evals" / meeting_id / "meeting_metrics.json"
    manifest.setdefault("units", {})["meeting"] = {
        "state": evaluation_state,
        "contract_fingerprint": contract["fingerprint"],
        "metadata": contract["payload"],
        "output_fingerprints": {str(eval_file): fingerprint_file(eval_file)},
    }
    manifest["state"] = summarize_manifest_state(manifest)
    atomic_write_json(manifest_path, manifest)


def pipeline_status_matches(
    output_dir: Path,
    meeting_id: str,
    prep_dir: Path,
    participants: List[str],
    delegate_variant: str,
    no_recalibration: bool,
    model: str,
    raw_context_N: int,
    capa_k: Optional[int] = None,
    freeze_anchor_inventory: bool = False,
    anchor_seed_k: Optional[int] = None,
    anchor_inventory_dir: Optional[str] = None,
    static_prep_source_dir: Optional[str] = None,
    eim_cache_dir: Optional[str] = None,
) -> bool:
    status = load_pipeline_status(output_dir)
    if not status:
        return False
    expected = build_delegate_capabilities(delegate_variant, no_recalibration)
    runtime_metadata = build_delegate_runtime_metadata(delegate_variant, raw_context_N, capa_k)
    if status.get("meeting_id") != meeting_id:
        return False
    if status.get("delegate_variant") != delegate_variant:
        return False
    if Path(status.get("shared_prep_dir", str(prep_dir))).resolve() != prep_dir.resolve():
        return False
    requested = status.get("requested_participants") or status.get("participants")
    if requested != participants:
        return False
    if status.get("model") != model:
        return False
    if status.get("capa_k") != capa_k:
        return False
    if bool(status.get("freeze_anchor_inventory")) != bool(freeze_anchor_inventory):
        return False
    if status.get("anchor_seed_k") != anchor_seed_k:
        return False
    if status.get("anchor_inventory_dir", "") != str(anchor_inventory_dir or ""):
        return False
    if status.get("static_prep_source_dir", "") != str(static_prep_source_dir or ""):
        return False
    if status.get("eim_cache_dir", "") != str(eim_cache_dir or ""):
        return False
    if status.get("episode_normalization_version") != EPISODE_NORMALIZATION_VERSION:
        return False
    for key, value in expected.items():
        if key == "delegate_variant":
            continue
        if status.get(key) != value:
            return False
    for key, value in runtime_metadata.items():
        if status.get(key) != value:
            return False
    return True


def check_stage1_complete(
    prep_dir: Path,
    participants: List[str],
    meeting_id: str,
    input_mtime: float,
    logger: logging.Logger,
    *,
    require_anchor_inventory: bool = False,
    capa_k: Optional[int] = None,
) -> bool:
    manifest_path = prep_dir / "prep_manifest.json"
    manifest = load_manifest(manifest_path, default_version=PREP_MANIFEST_VERSION)
    if manifest.get("manifest_version") == PREP_MANIFEST_VERSION and any(
        "::" in key for key in (manifest.get("units") or {})
    ):
        files: List[Path] = []
        for participant in participants:
            slug = participant.replace(" ", "_")
            participant_files = [
                prep_dir / "capa_episodes" / f"capa_episodes_for_{slug}_from_{meeting_id}.json",
                prep_dir / "jit_briefings" / f"{slug}_briefing_from_{meeting_id}_partB.json",
                prep_dir / "user_profiles" / f"capa_profile_for_{slug}_in_{meeting_id}.json",
                prep_dir / "capa_eim_states" / f"capa_eim_states_for_{slug}_in_{meeting_id}.json",
            ]
            anchor_inventory_path = prep_dir / "capa_anchor_inventory" / f"anchor_inventory_for_{slug}_from_{meeting_id}.json"
            k_diag_path = prep_dir / "capa_k_diagnostics" / f"k_diagnostics_for_{slug}_from_{meeting_id}_k{capa_k}.json" if require_anchor_inventory and capa_k is not None else None
            contracts = {
                f"profile::{participant}": [participant_files[2]],
                f"normalized_profile::{participant}": [participant_files[2]],
                f"episodes::{participant}": [path for path in [participant_files[0], k_diag_path] if path is not None],
                f"jit::{participant}": [participant_files[1]],
                f"eim::{participant}": [participant_files[3]],
            }
            if require_anchor_inventory:
                contracts[f"anchor_inventory::{participant}"] = [anchor_inventory_path]
            for unit_key, output_paths in contracts.items():
                unit = (manifest.get("units") or {}).get(unit_key)
                if not unit or unit.get("state") != "complete":
                    logger.debug(f"[AutoResume] Stage 1 incomplete: {unit_key} not complete")
                    return False
                output_fingerprints = unit.get("output_fingerprints") or {}
                for path in output_paths:
                    if not path.exists():
                        logger.debug(f"[AutoResume] Stage 1 incomplete: missing {path.name}")
                        return False
                    current_fp = fingerprint_file(path)
                    if current_fp is None or output_fingerprints.get(str(path)) != current_fp:
                        logger.debug(f"[AutoResume] Stage 1 incomplete: fingerprint mismatch for {path.name}")
                        return False
                files.extend(output_paths)
        oldest_output = min(get_file_mtime(path) or 0 for path in files) if files else 0
        if oldest_output < input_mtime:
            logger.debug("[AutoResume] Stage 1 outputs are stale")
            return False
        logger.debug(f"[AutoResume] Stage 1 complete via manifest v2: {len(files)} files verified")
        return True

    required_dirs = [
        prep_dir / "capa_episodes",
        prep_dir / "jit_briefings",
        prep_dir / "user_profiles",
        prep_dir / "capa_eim_states",
    ]
    if require_anchor_inventory:
        required_dirs.extend([prep_dir / "capa_anchor_inventory", prep_dir / "capa_k_diagnostics"])
    if not all(path.exists() for path in required_dirs):
        logger.debug("[AutoResume] Stage 1 incomplete: missing prep directories")
        return False

    files: List[Path] = []
    for participant in participants:
        slug = participant.replace(" ", "_")
        participant_files = [
            prep_dir / "capa_episodes" / f"capa_episodes_for_{slug}_from_{meeting_id}.json",
            prep_dir / "jit_briefings" / f"{slug}_briefing_from_{meeting_id}_partB.json",
            prep_dir / "user_profiles" / f"capa_profile_for_{slug}_in_{meeting_id}.json",
            prep_dir / "capa_eim_states" / f"capa_eim_states_for_{slug}_in_{meeting_id}.json",
        ]
        if require_anchor_inventory:
            participant_files.extend(
                [
                    prep_dir / "capa_anchor_inventory" / f"anchor_inventory_for_{slug}_from_{meeting_id}.json",
                    prep_dir / "capa_k_diagnostics" / f"k_diagnostics_for_{slug}_from_{meeting_id}_k{capa_k}.json",
                ]
            )
        missing = [path for path in participant_files if not path.exists()]
        if missing:
            logger.debug(f"[AutoResume] Stage 1 incomplete: missing {missing[0].name}")
            return False
        files.extend(participant_files)

    oldest_output = min(get_file_mtime(path) or 0 for path in files)
    if oldest_output < input_mtime:
        logger.debug("[AutoResume] Stage 1 outputs are stale")
        return False

    logger.debug(f"[AutoResume] Stage 1 complete: {len(files)} files verified")
    return True


def check_stage2_complete_for_participant(
    output_dir: Path,
    participant: str,
    meeting_id: str,
    stage1_mtime: float,
    logger: logging.Logger,
) -> bool:
    slug = participant.replace(" ", "_")
    sim_file = output_dir / f"capa_sim_{slug}_{meeting_id}.json"
    artifact_file = output_dir / "artifacts" / f"capa_artifacts_{participant}_{meeting_id}.json"

    if not sim_file.exists() or not artifact_file.exists():
        logger.debug(f"[AutoResume] Stage 2/{participant}: missing simulation or artifact file")
        return False

    oldest_output = min(get_file_mtime(sim_file) or 0, get_file_mtime(artifact_file) or 0)
    if oldest_output < stage1_mtime:
        logger.debug(f"[AutoResume] Stage 2/{participant}: outputs are stale")
        return False

    logger.debug(f"[AutoResume] Stage 2/{participant}: complete")
    return True


def check_stage3_complete(output_dir: Path, meeting_id: str, stage2_mtime: float, logger: logging.Logger) -> bool:
    eval_file = output_dir / "evals" / meeting_id / "meeting_metrics.json"
    if not eval_file.exists():
        logger.debug("[AutoResume] Stage 3 incomplete: no meeting_metrics.json")
        return False
    eval_mtime = get_file_mtime(eval_file)
    if eval_mtime and eval_mtime < stage2_mtime:
        logger.debug("[AutoResume] Stage 3 output is stale")
        return False
    logger.debug("[AutoResume] Stage 3 complete")
    return True


def get_stage1_mtime(prep_dir: Path, participants: List[str], meeting_id: str, *, include_anchor_inventory: bool = False, capa_k: Optional[int] = None) -> float:
    files: List[Path] = []
    for participant in participants:
        slug = participant.replace(" ", "_")
        files.extend(
            [
                prep_dir / "capa_episodes" / f"capa_episodes_for_{slug}_from_{meeting_id}.json",
                prep_dir / "jit_briefings" / f"{slug}_briefing_from_{meeting_id}_partB.json",
                prep_dir / "user_profiles" / f"capa_profile_for_{slug}_in_{meeting_id}.json",
                prep_dir / "capa_eim_states" / f"capa_eim_states_for_{slug}_in_{meeting_id}.json",
            ]
        )
        if include_anchor_inventory and capa_k is not None:
            files.extend(
                [
                    prep_dir / "capa_anchor_inventory" / f"anchor_inventory_for_{slug}_from_{meeting_id}.json",
                    prep_dir / "capa_k_diagnostics" / f"k_diagnostics_for_{slug}_from_{meeting_id}_k{capa_k}.json",
                ]
            )
    return get_latest_mtime(files) or 0


def get_stage2_mtime(output_dir: Path, participants: List[str], meeting_id: str) -> float:
    files: List[Path] = []
    for participant in participants:
        slug = participant.replace(" ", "_")
        files.extend(
            [
                output_dir / f"capa_sim_{slug}_{meeting_id}.json",
                output_dir / "artifacts" / f"capa_artifacts_{participant}_{meeting_id}.json",
            ]
        )
    return get_latest_mtime(files) or 0


# =============================================================================
# Stage runners
# =============================================================================

def run_data_generation(
    input_path: Path,
    prep_dir: Path,
    meeting_id: str,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Dict[str, Any]:
    cmd = [
        sys.executable, "src/data_pipeline.py",
        "--input", str(input_path),
        "--output", str(prep_dir),
        "--meeting-id", meeting_id,
        "--mode", "capa",
        "--capa-N", str(args.capa_N),
        "--capa-k", str(args.capa_k),
        "--selector", "llm-idea",
        "--cap-meeting", str(args.cap_meeting),
        "--cap-per-participant", str(args.cap_per_participant),
        "--min-gap", str(args.min_gap),
        "--model", args.model,
        "--debug-capa-episodes",
        "--jit-enable-sliding-windows",
        "--jit-enable-llm-cleanup",
    ]
    if getattr(args, "freeze_anchor_inventory", False):
        cmd.append("--freeze-anchor-inventory")
    if getattr(args, "anchor_inventory_dir", None):
        cmd.extend(["--anchor-inventory", str(args.anchor_inventory_dir)])
    if getattr(args, "anchor_seed_k", None) is not None:
        cmd.extend(["--anchor-seed-k", str(args.anchor_seed_k)])
    if getattr(args, "static_prep_source_dir", None):
        cmd.extend(["--static-prep-source-dir", str(args.static_prep_source_dir)])
    if getattr(args, "eim_cache_dir", None):
        cmd.extend(["--eim-cache-dir", str(args.eim_cache_dir)])
    if getattr(args, "participants", "all").lower() != "all":
        cmd.extend(["--participants", args.participants])
    if getattr(args, "force", False):
        cmd.append("--force")
    else:
        cmd.append("--resume")
    return run_command(cmd, logger, "DATA_GEN")


def run_simulation(
    input_path: Path,
    output_dir: Path,
    prep_dir: Path,
    meeting_id: str,
    participant: str,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Dict[str, Any]:
    cmd = [
        sys.executable, "src/participant_simulation.py",
        "--participant", participant,
        "--meeting", meeting_id,
        "--agent-type", "two",
        "--mode", "capa",
        "--delegate-variant", args.delegate_variant,
        "--capa-episodes-dir", str(prep_dir / "capa_episodes"),
        "--capa-eim-dir", str(prep_dir / "capa_eim_states"),
        "--briefings-dir", str(prep_dir / "jit_briefings"),
        "--user-profiles-dir", str(prep_dir / "user_profiles"),
        "--output-dir", str(output_dir),
        "--enable-artifact-tracking",
        "--model", args.model,
        "--late-window-k", str(args.capa_k),
    ]
    if is_raw_baseline_variant(args.delegate_variant):
        cmd.extend([
            "--raw-context-N", str(args.raw_context_N),
            "--transcript-path", str(input_path),
        ])
    if args.delegate_variant == "capa" and args.no_recalibration:
        cmd.append("--no-recalibration")
    if getattr(args, "force", False):
        cmd.append("--force")
    else:
        cmd.append("--resume")
    return run_command(cmd, logger, f"SIM:{participant}")


def run_evaluation(
    output_dir: Path,
    prep_dir: Path,
    meeting_id: str,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Dict[str, Any]:
    cmd = [
        sys.executable, "scripts/capa_eval.py",
        "--meeting", meeting_id,
        "--results-dir", str(output_dir),
        "--episodes-dir", str(prep_dir / "capa_episodes"),
        "--mode", "idea",
        "--strict-missing-artifacts",
        "--ctx-strict", str(args.ctx_strict),
        "--rel-strict", str(args.rel_strict),
        "--ctx-loose", str(args.ctx_loose),
        "--rel-loose", str(args.rel_loose),
    ]
    return run_command(cmd, logger, "EVAL")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete CAPA pipeline for a single meeting")
    parser.add_argument("--input", required=True, help="Path to meeting JSON file")
    parser.add_argument("--output-dir", required=True, help="Variant-specific output directory for this meeting")
    parser.add_argument("--shared-prep-dir", default=None, help="Optional shared Stage 1 prep directory for this meeting")

    parser.add_argument("--participants", default="all", help="Comma-separated list of participants, or 'all'")

    parser.add_argument("--capa-N", type=int, default=20, help="CAPA window size N")
    parser.add_argument("--capa-k", type=int, default=5, help="CAPA window size k")
    parser.add_argument("--freeze-anchor-inventory", action="store_true", help="Freeze anchor inventory before rebuilding k-specific CAPA episodes")
    parser.add_argument("--anchor-inventory", "--anchor-inventory-dir", dest="anchor_inventory_dir", default=None, help="Optional directory containing frozen anchor inventory JSON files")
    parser.add_argument("--anchor-seed-k", type=int, default=None, help="Seed k used when freezing anchor inventory with llm-idea")
    parser.add_argument("--static-prep-source-dir", default=None, help="Optional meeting-specific prep dir providing reusable user_profiles and jit_briefings")
    parser.add_argument("--eim-cache-dir", default=None, help="Optional EIM cache root; defaults to <shared-prep-dir>/_eim_cache")
    parser.add_argument("--cap-meeting", type=int, default=20, help="JIT cap per meeting")
    parser.add_argument("--cap-per-participant", type=int, default=5, help="JIT cap per participant")
    parser.add_argument("--min-gap", type=int, default=0, help="Min gap for JIT")

    parser.add_argument("--ctx-strict", type=float, default=0.75, help="Strict context threshold")
    parser.add_argument("--rel-strict", type=float, default=0.75, help="Strict relevance threshold")
    parser.add_argument("--ctx-loose", type=float, default=0.65, help="Loose context threshold")
    parser.add_argument("--rel-loose", type=float, default=0.65, help="Loose relevance threshold")

    parser.add_argument("--delegate-variant", choices=["capa", "raw-modular", "raw-modular-fair", "mono-raw", "mono-raw-reflection", "mono-raw-reflection-v2"], default="capa", help="CAPA delegate runtime variant")
    parser.add_argument("--no-recalibration", action="store_true", help="Disable recalibrator (CAPA only)")
    parser.add_argument("--raw-context-N", type=int, default=50, help="Raw-baseline transcript window size")
    parser.add_argument("--mono-raw-context-N", type=int, default=None, help="Deprecated alias for --raw-context-N")
    parser.add_argument("--model", default="gpt-4o", help="LLM model to use; must exist in config_extended.json")
    parser.add_argument("--debug-max-utterances", type=int, default=None, help="Optional debug-only transcript truncation cap. When set, use only the first N utterances from the input transcript.")

    parser.add_argument("--skip-generation", action="store_true", help="Skip data generation (use existing)")
    parser.add_argument("--skip-simulation", action="store_true", help="Skip simulation (use existing)")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip evaluation")
    parser.add_argument("--force", action="store_true", help="Force re-run all stages, ignore existing files")

    args = parser.parse_args()
    args.raw_context_N = normalize_raw_context_n(args.raw_context_N, args.mono_raw_context_N)

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    prep_dir = Path(args.shared_prep_dir).resolve() if args.shared_prep_dir else output_dir
    meeting_id = extract_meeting_id(input_path)
    os.environ["MD_ACTIVE_MODEL"] = args.model

    logger = setup_logging(output_dir, meeting_id)
    logger.info("=" * 60)
    logger.info(f"PIPELINE START: {meeting_id}")

    if args.debug_max_utterances is not None:
        input_path = materialize_debug_truncated_input(input_path, output_dir, args.debug_max_utterances, logger)

    logger.info(f"Input: {input_path}")
    logger.info(f"Results: {output_dir}")
    logger.info(f"Shared Prep: {prep_dir}")
    logger.info(f"Delegate Variant: {args.delegate_variant}")
    if is_raw_baseline_variant(args.delegate_variant):
        logger.info(f"Raw Context N: {args.raw_context_N}")
    logger.info(f"Model: {args.model}")
    logger.info("=" * 60)

    if args.delegate_variant != "capa" and args.no_recalibration:
        logger.warning("--no-recalibration is ignored for raw delegate variants")

    start_time = datetime.now()

    participants = extract_participants(input_path) if args.participants.lower() == "all" else [p.strip() for p in args.participants.split(",") if p.strip()]
    logger.info(f"Participants: {participants}")

    stage_states: Dict[str, Any] = {
        "prep": {"state": "pending", "result": None},
        "simulation": {
            "state": "pending",
            "requested_participants": list(participants),
            "completed_participants": [],
            "failed_participants": [],
            "in_progress_participants": [],
            "missing_participants": list(participants),
            "participant_results": {},
            "result": None,
        },
        "evaluation": {
            "state": "pending",
            "requested_participants": list(participants),
            "effective_participants": [],
            "missing_participants": list(participants),
            "full": False,
            "reused": False,
            "result": None,
        },
    }
    write_pipeline_status(
        output_dir,
        meeting_id,
        prep_dir,
        participants,
        args,
        "running",
        effective_participants=resolve_pipeline_participants(output_dir, participants, include_requested=True),
        stage_states=stage_states,
        retryable=True,
    )

    if not args.skip_generation:
        stage_states["prep"]["state"] = "running"
        write_pipeline_status(
            output_dir,
            meeting_id,
            prep_dir,
            participants,
            args,
            "running",
            effective_participants=resolve_pipeline_participants(output_dir, participants, include_requested=True),
            stage_states=stage_states,
            current_unit="prep",
            retryable=True,
        )
        logger.info("[STAGE 1/3] Data Generation")
        prep_result = run_data_generation(input_path, prep_dir, meeting_id, args, logger)
        stage_states["prep"]["result"] = prep_result
        stage_states["prep"]["state"] = "complete" if prep_result["ok"] else "failed"
        if not prep_result["ok"]:
            logger.error("Data generation failed. Aborting downstream stages.")
    else:
        logger.info("[STAGE 1/3] Skipping data generation (--skip-generation)")
        stage_states["prep"] = {
            "state": "skipped",
            "result": make_stage_result(True, message="Stage 1 skipped by operator"),
        }

    if stage_states["prep"]["state"] != "failed":
        if not args.skip_simulation:
            logger.info("[STAGE 2/3] Simulation")
            stage_states["simulation"]["state"] = "running"
            for participant in participants:
                write_pipeline_status(
                    output_dir,
                    meeting_id,
                    prep_dir,
                    participants,
                    args,
                    "running",
                    effective_participants=resolve_pipeline_participants(output_dir, participants, include_requested=True),
                    stage_states=stage_states,
                    current_unit=f"simulation:{participant}",
                    retryable=True,
                )
                participant_result = run_simulation(input_path, output_dir, prep_dir, meeting_id, participant, args, logger)
                stage_states["simulation"]["participant_results"][participant] = participant_result
                if not participant_result["ok"]:
                    logger.error(f"Simulation failed for {participant}. Continuing with others.")
        else:
            logger.info("[STAGE 2/3] Skipping simulation (--skip-simulation)")

        simulation_summary = discover_simulation_participant_states(output_dir, participants)
        stage_states["simulation"]["completed_participants"] = simulation_summary["complete"]
        stage_states["simulation"]["failed_participants"] = simulation_summary["failed"]
        stage_states["simulation"]["in_progress_participants"] = simulation_summary["in_progress"]
        stage_states["simulation"]["missing_participants"] = simulation_summary["missing"]

        if not simulation_summary["incomplete"]:
            stage_states["simulation"]["state"] = "complete"
            stage_states["simulation"]["result"] = make_stage_result(True, returncode=0, message="All requested participant simulations are complete")
        elif simulation_summary["complete"]:
            stage_states["simulation"]["state"] = "partial"
            stage_states["simulation"]["result"] = make_stage_result(False, failure_kind="partial_simulation", message="Some requested participant simulations failed or are incomplete")
        else:
            stage_states["simulation"]["state"] = "failed"
            stage_states["simulation"]["result"] = make_stage_result(False, failure_kind="simulation_failed", message="No requested participant simulations completed successfully")

        if args.skip_simulation:
            stage_states["simulation"]["result"]["message"] = (
                "Stage 2 skipped by operator; existing participant outputs are complete"
                if stage_states["simulation"]["state"] == "complete"
                else "Stage 2 skipped by operator; required participant outputs remain incomplete"
            )
    else:
        logger.info("[STAGE 2/3] Skipping simulation (prep failed)")
        stage_states["simulation"]["state"] = "blocked"
        stage_states["simulation"]["result"] = make_stage_result(False, failure_kind="prep_failed", message="Stage 2 blocked because Stage 1 failed")

    completed_participants = list(stage_states["simulation"].get("completed_participants", []))
    missing_participants = [participant for participant in participants if participant not in completed_participants]

    if stage_states["prep"]["state"] == "failed":
        stage_states["evaluation"] = {
            "state": "blocked",
            "requested_participants": list(participants),
            "effective_participants": completed_participants,
            "missing_participants": missing_participants,
            "full": False,
            "reused": False,
            "result": make_stage_result(False, failure_kind="prep_failed", message="Stage 3 blocked because Stage 1 failed"),
        }
    elif args.skip_evaluation:
        logger.info("[STAGE 3/3] Skipping evaluation (--skip-evaluation)")
        stage_states["evaluation"] = {
            "state": "skipped",
            "requested_participants": list(participants),
            "effective_participants": completed_participants,
            "missing_participants": missing_participants,
            "full": False,
            "reused": False,
            "result": make_stage_result(False, failure_kind="evaluation_skipped", message="Stage 3 skipped by operator"),
        }
    elif not completed_participants:
        logger.info("[STAGE 3/3] Skipping evaluation (no complete simulation outputs)")
        stage_states["evaluation"] = {
            "state": "blocked",
            "requested_participants": list(participants),
            "effective_participants": [],
            "missing_participants": list(participants),
            "full": False,
            "reused": False,
            "result": make_stage_result(False, failure_kind="missing_simulation_outputs", message="Stage 3 blocked because no requested participant simulations completed successfully"),
        }
    elif not missing_participants and not args.force and check_eval_manifest_complete(output_dir, meeting_id, participants, prep_dir, args, logger):
        logger.info("[STAGE 3/3] Skipping (eval_manifest.json is current)")
        stage_states["evaluation"] = {
            "state": "complete",
            "requested_participants": list(participants),
            "effective_participants": completed_participants,
            "missing_participants": [],
            "full": True,
            "reused": True,
            "result": make_stage_result(True, returncode=0, message="Evaluation manifest is current; reusing existing full evaluation"),
        }
    else:
        logger.info("[STAGE 3/3] Evaluation")
        stage_states["evaluation"] = {
            "state": "running",
            "requested_participants": list(participants),
            "effective_participants": completed_participants,
            "missing_participants": missing_participants,
            "full": False,
            "reused": False,
            "result": None,
        }
        write_pipeline_status(
            output_dir,
            meeting_id,
            prep_dir,
            participants,
            args,
            "running",
            effective_participants=resolve_pipeline_participants(output_dir, participants, include_requested=True),
            stage_states=stage_states,
            current_unit="evaluation",
            retryable=True,
        )
        evaluation_result = run_evaluation(output_dir, prep_dir, meeting_id, args, logger)
        if not evaluation_result["ok"]:
            logger.error("Evaluation failed.")
            stage_states["evaluation"] = {
                "state": "failed",
                "requested_participants": list(participants),
                "effective_participants": completed_participants,
                "missing_participants": missing_participants,
                "full": False,
                "reused": False,
                "result": evaluation_result,
            }
        else:
            evaluation_state = "complete" if not missing_participants else "partial"
            write_eval_manifest(
                output_dir,
                meeting_id,
                participants,
                prep_dir,
                args,
                evaluation_state=evaluation_state,
                effective_participants=completed_participants,
                missing_participants=missing_participants,
            )
            if evaluation_state == "partial":
                evaluation_stage_result = dict(evaluation_result)
                evaluation_stage_result.update({
                    "ok": False,
                    "failure_kind": "partial_evaluation",
                    "message": "Evaluation completed on a partial participant set; diagnostic only",
                })
            else:
                evaluation_stage_result = evaluation_result
            stage_states["evaluation"] = {
                "state": evaluation_state,
                "requested_participants": list(participants),
                "effective_participants": completed_participants,
                "missing_participants": missing_participants,
                "full": evaluation_state == "complete",
                "reused": False,
                "result": evaluation_stage_result,
            }

    if stage_states["prep"]["state"] == "failed" or stage_states["evaluation"]["state"] == "failed":
        meeting_status = "failed"
    elif not completed_participants:
        meeting_status = "failed"
    elif missing_participants:
        meeting_status = "partial"
    elif stage_states["evaluation"]["state"] != "complete":
        meeting_status = "partial"
    else:
        meeting_status = "success"

    retryable = meeting_status != "success"
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("=" * 60)
    logger.info(f"PIPELINE {meeting_status.upper()}: {meeting_id} in {elapsed:.1f}s")
    logger.info("=" * 60)

    write_pipeline_status(
        output_dir,
        meeting_id,
        prep_dir,
        participants,
        args,
        meeting_status,
        effective_participants=completed_participants,
        stage_states=stage_states,
        elapsed_seconds=elapsed,
        retryable=retryable,
    )

    sys.exit(0 if meeting_status == "success" else 1)


if __name__ == "__main__":
    main()
