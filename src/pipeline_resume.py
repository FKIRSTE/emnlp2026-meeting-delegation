from __future__ import annotations

import hashlib
import json
import os
import socket
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

PREP_MANIFEST_VERSION = "prep_manifest_v2"
SIMULATION_MANIFEST_VERSION = "simulation_manifest_v1"
EVAL_MANIFEST_VERSION = "eval_manifest_v1"
PIPELINE_RUNTIME_VERSION = "pipeline_runtime_resume_v1"

PROFILE_STAGE_VERSION = "profile_stage_v1"
NORMALIZED_PROFILE_STAGE_VERSION = "normalized_profile_stage_v1"
ANCHOR_INVENTORY_STAGE_VERSION = "anchor_inventory_stage_v1"
EPISODE_STAGE_VERSION = "episode_stage_v1"
CAPA_EIM_STAGE_VERSION = "capa_eim_stage_v1"
JIT_STAGE_VERSION = "jit_stage_v1"
SIMULATION_STAGE_VERSION = "simulation_stage_v2_delegate_yield"
EVAL_STAGE_VERSION = "eval_stage_v1"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def fingerprint_payload(payload: Any) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def fingerprint_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_files(paths: Iterable[Path]) -> Dict[str, Optional[str]]:
    return {str(path): fingerprint_file(path) for path in paths}


def fingerprint_transcript_file(path: Path) -> str:
    digest = fingerprint_file(path)
    if digest is None:
        raise FileNotFoundError(f"Transcript file not found: {path}")
    return digest


def fingerprint_contract(**kwargs: Any) -> str:
    return fingerprint_payload(kwargs)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False, default=str))


@contextmanager
def file_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "created_at": time.time(),
    }
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    fd = None
    try:
        fd = os.open(str(lock_path), flags)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass


def load_manifest(path: Path, *, default_version: str) -> Dict[str, Any]:
    if not path.exists():
        return {
            "manifest_version": default_version,
            "state": "not_started",
            "units": {},
            "updated_at": None,
        }
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("manifest must be a JSON object")
        payload.setdefault("manifest_version", default_version)
        payload.setdefault("state", "not_started")
        payload.setdefault("units", {})
        payload.setdefault("updated_at", None)
        return payload
    except Exception:
        return {
            "manifest_version": default_version,
            "state": "corrupt",
            "units": {},
            "updated_at": None,
        }


def save_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    atomic_write_json(path, manifest)


def unit_is_reusable(unit: Optional[Dict[str, Any]], contract_fingerprint: str, output_paths: Iterable[Path]) -> bool:
    if not unit or unit.get("state") != "complete":
        return False
    if unit.get("contract_fingerprint") != contract_fingerprint:
        return False
    for path in output_paths:
        if not path.exists():
            return False
    output_fingerprints = unit.get("output_fingerprints") or {}
    for path in output_paths:
        current = fingerprint_file(path)
        if current is None or fingerprint_for_equivalent_path(output_fingerprints, path) != current:
            return False
    return True


def fingerprint_for_equivalent_path(output_fingerprints: Dict[str, Optional[str]], path: Path) -> Optional[str]:
    candidates = {str(path), os.path.normpath(str(path))}
    try:
        resolved = path.resolve()
        candidates.add(str(resolved))
        candidates.add(os.path.normpath(str(resolved)))
        try:
            candidates.add(os.path.normpath(os.path.relpath(resolved, Path.cwd())))
        except ValueError:
            pass
    except Exception:
        resolved = None

    for candidate in candidates:
        if candidate in output_fingerprints:
            return output_fingerprints.get(candidate)

    if resolved is None:
        return None
    for raw_key, fingerprint in output_fingerprints.items():
        try:
            key_path = Path(raw_key)
            key_resolved = key_path.resolve() if key_path.is_absolute() else (Path.cwd() / key_path).resolve()
        except Exception:
            continue
        if key_resolved == resolved:
            return fingerprint
    return None


def mark_unit_in_progress(manifest: Dict[str, Any], unit_key: str, contract_fingerprint: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    manifest.setdefault("units", {})[unit_key] = {
        "state": "in_progress",
        "contract_fingerprint": contract_fingerprint,
        "metadata": metadata or {},
        "output_fingerprints": {},
    }
    manifest["state"] = "in_progress"


def mark_unit_complete(manifest: Dict[str, Any], unit_key: str, contract_fingerprint: str, output_paths: Iterable[Path], metadata: Optional[Dict[str, Any]] = None) -> None:
    manifest.setdefault("units", {})[unit_key] = {
        "state": "complete",
        "contract_fingerprint": contract_fingerprint,
        "metadata": metadata or {},
        "output_fingerprints": fingerprint_files(output_paths),
    }


def mark_unit_failed(manifest: Dict[str, Any], unit_key: str, contract_fingerprint: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    manifest.setdefault("units", {})[unit_key] = {
        "state": "failed",
        "contract_fingerprint": contract_fingerprint,
        "metadata": metadata or {},
        "output_fingerprints": {},
    }


def summarize_manifest_state(manifest: Dict[str, Any]) -> str:
    units = manifest.get("units") or {}
    if not units:
        return manifest.get("state", "not_started")
    if all((unit or {}).get("state") == "complete" for unit in units.values()):
        return "complete"
    if any((unit or {}).get("state") == "partial" for unit in units.values()):
        return "partial"
    if any((unit or {}).get("state") == "in_progress" for unit in units.values()):
        return "in_progress"
    if any((unit or {}).get("state") == "failed" for unit in units.values()):
        if any((unit or {}).get("state") == "complete" for unit in units.values()):
            return "partial"
        return "failed"
    if any((unit or {}).get("state") == "complete" for unit in units.values()):
        return "partial"
    return manifest.get("state", "not_started")
