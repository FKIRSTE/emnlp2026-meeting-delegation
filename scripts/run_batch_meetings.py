#!/usr/bin/env python
"""
run_batch_meetings.py — Batch Runner for Multiple Meetings

Runs run_meeting_pipeline.py for all meetings in one or more directories with
variant-aware results roots and optional shared Stage 1 prep reuse.
"""

import argparse
import fnmatch
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from raw_baseline_support import (
    build_delegate_capabilities,
    build_delegate_runtime_metadata,
    normalize_raw_context_n,
)
from pipeline_resume import EVAL_MANIFEST_VERSION, SIMULATION_MANIFEST_VERSION, fingerprint_file, load_manifest

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("batch_runner")

MEETING_NUMERIC_ID_RE = re.compile(r"^(?P<prefix>.*?)(?P<number>\d+)(?P<suffix>[^\d]*)$")


def extract_meeting_id(input_path: Path) -> str:
    name = input_path.stem
    if name.endswith("_parsed"):
        name = name[:-7]
    return name


def parse_input_dirs_csv(raw_value: Optional[str]) -> List[str]:
    if not raw_value:
        return []
    values: List[str] = []
    for item in str(raw_value).split(","):
        value = item.strip()
        if value:
            values.append(value)
    return values


def normalize_input_dirs(input_dirs_csv: Optional[str], input_dir_args: Optional[List[str]]) -> List[Path]:
    ordered_dirs: List[Path] = []
    seen: Set[Path] = set()
    for raw_value in parse_input_dirs_csv(input_dirs_csv) + list(input_dir_args or []):
        resolved = Path(raw_value).resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered_dirs.append(resolved)
    return ordered_dirs


def parse_meeting_numeric_id(meeting_id: str) -> Optional[Tuple[str, int, str]]:
    match = MEETING_NUMERIC_ID_RE.fullmatch(str(meeting_id).strip())
    if not match:
        return None
    return (
        match.group("prefix"),
        int(match.group("number")),
        match.group("suffix"),
    )


def parse_skip_rules(raw_skip: Optional[str]) -> List[Dict[str, Any]]:
    if not raw_skip:
        return []

    rules: List[Dict[str, Any]] = []
    for raw_token in str(raw_skip).split(","):
        token = raw_token.strip()
        if not token:
            continue
        if ".." in token:
            parts = [part.strip() for part in token.split("..")]
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(
                    f"Malformed skip range '{token}'. Expected syntax '<start_meeting_id>..<end_meeting_id>'."
                )
            start = parse_meeting_numeric_id(parts[0])
            end = parse_meeting_numeric_id(parts[1])
            if start is None or end is None:
                raise ValueError(
                    f"Malformed skip range '{token}'. Range endpoints must look like concrete meeting IDs."
                )
            start_prefix, start_number, start_suffix = start
            end_prefix, end_number, end_suffix = end
            if start_prefix != end_prefix or start_suffix != end_suffix:
                raise ValueError(
                    f"Invalid skip range '{token}'. Range endpoints must share the same non-numeric prefix and suffix."
                )
            if start_number > end_number:
                raise ValueError(
                    f"Invalid skip range '{token}'. Range start must be <= range end."
                )
            rules.append({
                "kind": "range",
                "raw": token,
                "prefix": start_prefix,
                "suffix": start_suffix,
                "start": start_number,
                "end": end_number,
            })
            continue
        if "*" in token or "?" in token:
            rules.append({"kind": "glob", "raw": token, "pattern": token})
            continue
        rules.append({"kind": "exact", "raw": token, "meeting_id": token})
    return rules


def should_skip_meeting(meeting_id: str, skip_rules: List[Dict[str, Any]]) -> bool:
    parsed_meeting = None
    for rule in skip_rules:
        kind = rule.get("kind")
        if kind == "exact" and meeting_id == rule["meeting_id"]:
            return True
        if kind == "glob" and fnmatch.fnmatchcase(meeting_id, rule["pattern"]):
            return True
        if kind == "range":
            if parsed_meeting is None:
                parsed_meeting = parse_meeting_numeric_id(meeting_id)
            if parsed_meeting is None:
                continue
            prefix, number, suffix = parsed_meeting
            if prefix == rule["prefix"] and suffix == rule["suffix"] and rule["start"] <= number <= rule["end"]:
                return True
    return False


def discover_meetings(
    input_dirs: List[Path],
    meetings_filter: Optional[List[str]] = None,
    skip_rules: Optional[List[Dict[str, Any]]] = None,
) -> List[Path]:
    filter_set = set(meetings_filter) if meetings_filter else None
    active_skip_rules = list(skip_rules or [])
    files: List[Path] = []
    selected_meetings: Dict[str, Path] = {}

    for input_dir in input_dirs:
        for path in sorted(input_dir.glob("*_parsed.json")):
            meeting_id = extract_meeting_id(path)
            if filter_set and meeting_id not in filter_set:
                continue
            if should_skip_meeting(meeting_id, active_skip_rules):
                continue
            previous_path = selected_meetings.get(meeting_id)
            if previous_path is not None:
                raise ValueError(
                    f"Duplicate meeting_id '{meeting_id}' selected from multiple input dirs: "
                    f"{previous_path} and {path}"
                )
            selected_meetings[meeting_id] = path
            files.append(path)

    return files


def resolve_meeting_anchor_inventory_dir(anchor_inventory_root: Optional[str], meeting_id: str) -> str:
    if not anchor_inventory_root:
        return ""
    root = Path(anchor_inventory_root).resolve()
    if root.is_file():
        return str(root)
    if any(root.glob("anchor_inventory_for_*_from_*.json")):
        return str(root)
    if (root / "capa_anchor_inventory").exists():
        return str(root / "capa_anchor_inventory")
    meeting_dir = root / meeting_id
    if (meeting_dir / "capa_anchor_inventory").exists():
        return str(meeting_dir / "capa_anchor_inventory")
    if any(meeting_dir.glob("anchor_inventory_for_*_from_*.json")):
        return str(meeting_dir)
    if meeting_dir.exists():
        return str(meeting_dir)
    return str(root)


def resolve_meeting_static_prep_dir(static_prep_root: Optional[str], meeting_id: str) -> str:
    if not static_prep_root:
        return ""
    root = Path(static_prep_root).resolve()
    if (root / "user_profiles").exists() or (root / "jit_briefings").exists() or root.name == meeting_id:
        return str(root)
    return str(root / meeting_id)


def resolve_meeting_eim_cache_dir(eim_cache_root: Optional[str], meeting_id: str) -> str:
    if not eim_cache_root:
        return ""
    root = Path(eim_cache_root).resolve()
    if (root / "window_vectors").exists() or (root / "history_checkpoints").exists() or root.name == meeting_id:
        return str(root)
    return str(root / meeting_id)


def load_batch_log(output_dir: Path) -> Dict[str, Any]:
    log_file = output_dir / "batch_log.json"
    if log_file.exists():
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as exc:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = output_dir / f"batch_log.corrupt_{stamp}.json"
            try:
                backup.write_text(log_file.read_text(encoding="utf-8"), encoding="utf-8")
                logger.warning("Ignoring corrupt batch_log.json (%s); backed up to %s", exc, backup)
            except Exception:
                logger.warning("Ignoring corrupt batch_log.json (%s); backup failed", exc)
    return {"started_at": datetime.now().isoformat(), "meetings": {}}


def save_batch_log(output_dir: Path, log_data: Dict[str, Any]) -> None:
    log_file = output_dir / "batch_log.json"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = log_file.with_suffix(".json.tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_file, log_file)


def load_pipeline_status(meeting_output_dir: Path) -> Dict[str, Any]:
    status_file = meeting_output_dir / "pipeline_status.json"
    if not status_file.exists():
        return {}
    try:
        with open(status_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def discover_complete_simulation_participants(meeting_output_dir: Path) -> List[str]:
    manifest = load_manifest(meeting_output_dir / "simulation_manifest.json", default_version=SIMULATION_MANIFEST_VERSION)
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
    return sorted(set(participants))


def requested_participants_for_resume(meeting_output_dir: Path, args: argparse.Namespace) -> List[str]:
    if args.participants.lower() != "all":
        return [p.strip() for p in args.participants.split(",") if p.strip()]
    status = load_pipeline_status(meeting_output_dir)
    requested = status.get("requested_participants") or status.get("participants") or []
    return [p for p in requested if p]


def simulation_outputs_match_requested(meeting_output_dir: Path, args: argparse.Namespace) -> bool:
    requested = requested_participants_for_resume(meeting_output_dir, args)
    if not requested:
        return False
    completed = set(discover_complete_simulation_participants(meeting_output_dir))
    return all(participant in completed for participant in requested)


def pipeline_status_matches(
    meeting_output_dir: Path,
    meeting_id: str,
    args: argparse.Namespace,
    meeting_prep_dir: Path,
) -> bool:
    status = load_pipeline_status(meeting_output_dir)
    if not status:
        return False
    if status.get("meeting_id") != meeting_id:
        return False
    if status.get("status") != "success":
        return False
    if bool(status.get("retryable", False)):
        return False
    stage_states = status.get("stage_states") or {}
    if (stage_states.get("simulation") or {}).get("state") not in {None, "complete"}:
        return False
    if (stage_states.get("evaluation") or {}).get("state") not in {None, "complete"}:
        return False
    if status.get("delegate_variant") != args.delegate_variant:
        return False
    if status.get("model") != args.model:
        return False
    if status.get("capa_k") != args.capa_k:
        return False
    if bool(status.get("freeze_anchor_inventory")) != bool(getattr(args, "freeze_anchor_inventory", False)):
        return False
    if status.get("anchor_seed_k") != getattr(args, "anchor_seed_k", None):
        return False
    expected_anchor_inventory_dir = resolve_meeting_anchor_inventory_dir(
        getattr(args, "anchor_inventory_root", None),
        meeting_id,
    )
    if status.get("anchor_inventory_dir", "") != expected_anchor_inventory_dir:
        return False
    expected_static_prep_dir = resolve_meeting_static_prep_dir(
        getattr(args, "static_prep_root", None),
        meeting_id,
    )
    if status.get("static_prep_source_dir", "") != expected_static_prep_dir:
        return False
    expected_eim_cache_dir = resolve_meeting_eim_cache_dir(
        getattr(args, "eim_cache_root", None),
        meeting_id,
    )
    if status.get("eim_cache_dir", "") != expected_eim_cache_dir:
        return False
    if Path(status.get("shared_prep_dir", str(meeting_prep_dir))).resolve() != meeting_prep_dir.resolve():
        return False
    if args.participants.lower() != "all":
        requested = status.get("requested_participants") or status.get("participants")
        if requested != [p.strip() for p in args.participants.split(",") if p.strip()]:
            return False
    expected = build_delegate_capabilities(args.delegate_variant, args.no_recalibration)
    runtime_metadata = build_delegate_runtime_metadata(args.delegate_variant, args.raw_context_N, args.capa_k)
    for key, value in expected.items():
        if status.get(key) != value:
            return False
    for key, value in runtime_metadata.items():
        if status.get(key) != value:
            return False
    return True


def eval_manifest_matches(meeting_output_dir: Path, meeting_id: str, args: argparse.Namespace) -> bool:
    metrics_file = meeting_output_dir / "evals" / meeting_id / "meeting_metrics.json"
    if not metrics_file.exists():
        return False
    manifest = load_manifest(meeting_output_dir / "eval_manifest.json", default_version=EVAL_MANIFEST_VERSION)
    unit = (manifest.get("units") or {}).get("meeting") or {}
    if unit.get("state") != "complete":
        return False
    metadata = unit.get("metadata") or {}
    if metadata.get("evaluation_state") not in {None, "complete"}:
        return False
    if metadata.get("missing_participants") not in (None, []):
        return False
    if metadata.get("ctx_strict") != getattr(args, "ctx_strict", 0.75):
        return False
    if metadata.get("rel_strict") != getattr(args, "rel_strict", 0.75):
        return False
    if metadata.get("ctx_loose") != getattr(args, "ctx_loose", 0.60):
        return False
    if metadata.get("rel_loose") != getattr(args, "rel_loose", 0.60):
        return False
    if metadata.get("capa_k") != getattr(args, "capa_k", None):
        return False
    if bool(metadata.get("freeze_anchor_inventory")) != bool(getattr(args, "freeze_anchor_inventory", False)):
        return False
    if metadata.get("anchor_seed_k") != getattr(args, "anchor_seed_k", None):
        return False
    expected_anchor_inventory_dir = resolve_meeting_anchor_inventory_dir(
        getattr(args, "anchor_inventory_root", None),
        meeting_id,
    )
    if metadata.get("anchor_inventory_dir", "") != expected_anchor_inventory_dir:
        return False
    if metadata.get("mode") != "idea":
        return False
    return True


def is_meeting_complete(
    meeting_output_dir: Path,
    meeting_id: str,
    args: argparse.Namespace,
    meeting_prep_dir: Path,
) -> bool:
    metrics_file = meeting_output_dir / "evals" / meeting_id / "meeting_metrics.json"
    if not metrics_file.exists():
        return False
    return (
        pipeline_status_matches(meeting_output_dir, meeting_id, args, meeting_prep_dir)
        and simulation_outputs_match_requested(meeting_output_dir, args)
        and eval_manifest_matches(meeting_output_dir, meeting_id, args)
    )


def run_meeting_pipeline(
    input_path: Path,
    meeting_output_dir: Path,
    meeting_prep_dir: Path,
    args: argparse.Namespace,
    dry_run: bool = False,
) -> Dict[str, Any]:
    anchor_inventory_dir = resolve_meeting_anchor_inventory_dir(
        getattr(args, "anchor_inventory_root", None),
        extract_meeting_id(input_path),
    )
    static_prep_source_dir = resolve_meeting_static_prep_dir(
        getattr(args, "static_prep_root", None),
        extract_meeting_id(input_path),
    )
    eim_cache_dir = resolve_meeting_eim_cache_dir(
        getattr(args, "eim_cache_root", None),
        extract_meeting_id(input_path),
    )
    cmd = [
        sys.executable, "scripts/run_meeting_pipeline.py",
        "--input", str(input_path),
        "--output-dir", str(meeting_output_dir),
        "--shared-prep-dir", str(meeting_prep_dir),
        "--participants", args.participants,
        "--delegate-variant", args.delegate_variant,
        "--raw-context-N", str(args.raw_context_N),
        "--model", args.model,
        "--capa-N", str(args.capa_N),
        "--capa-k", str(args.capa_k),
        "--freeze-anchor-inventory" if getattr(args, "freeze_anchor_inventory", False) else "",
        "--cap-meeting", str(args.cap_meeting),
        "--cap-per-participant", str(args.cap_per_participant),
        "--min-gap", str(args.min_gap),
        "--ctx-strict", str(args.ctx_strict),
        "--rel-strict", str(args.rel_strict),
        "--ctx-loose", str(args.ctx_loose),
        "--rel-loose", str(args.rel_loose),
    ]
    cmd = [token for token in cmd if token != ""]
    if anchor_inventory_dir:
        cmd.extend(["--anchor-inventory", anchor_inventory_dir])
    if static_prep_source_dir:
        cmd.extend(["--static-prep-source-dir", static_prep_source_dir])
    if eim_cache_dir:
        cmd.extend(["--eim-cache-dir", eim_cache_dir])
    if getattr(args, "anchor_seed_k", None) is not None:
        cmd.extend(["--anchor-seed-k", str(args.anchor_seed_k)])
    if args.delegate_variant == "capa" and args.no_recalibration:
        cmd.append("--no-recalibration")
    if getattr(args, "force", False):
        cmd.append("--force")
    if getattr(args, "skip_generation", False):
        cmd.append("--skip-generation")
    if getattr(args, "skip_simulation", False):
        cmd.append("--skip-simulation")
    if getattr(args, "skip_evaluation", False):
        cmd.append("--skip-evaluation")

    if dry_run:
        logger.info(f"[DRY-RUN] Would run: {' '.join(cmd)}")
        return {"ok": True, "status": "dry_run", "returncode": 0, "pipeline_status": {}}

    logger.info(f"Running: {' '.join(cmd[:8])} ...")
    try:
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "MD_ACTIVE_MODEL": getattr(args, "model", "gpt-4o")}
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        pipeline_status = load_pipeline_status(meeting_output_dir)
        meeting_status = pipeline_status.get("status") or ("success" if result.returncode == 0 else "failed")
        if result.returncode != 0:
            logger.error(f"Pipeline failed with return code {result.returncode}")
            if result.stdout:
                logger.error(f"=== STDOUT ===\n{result.stdout}")
            if result.stderr:
                logger.error(f"=== STDERR ===\n{result.stderr}")
            return {
                "ok": False,
                "status": meeting_status,
                "returncode": result.returncode,
                "pipeline_status": pipeline_status,
            }
        return {
            "ok": meeting_status == "success",
            "status": meeting_status,
            "returncode": result.returncode,
            "pipeline_status": pipeline_status,
        }
    except Exception as exc:
        logger.error(f"Exception: {exc}")
        return {"ok": False, "status": "failed", "returncode": None, "pipeline_status": {}, "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CAPA pipeline for multiple meetings in batch")
    parser.add_argument("--input-dir", action="append", default=None, help="Directory containing *_parsed.json files; can be repeated")
    parser.add_argument("--input-dirs", default=None, help="Comma-separated list of directories containing *_parsed.json files")
    parser.add_argument("--output-dir", required=True, help="Root output directory for all variants and meetings")
    parser.add_argument("--shared-prep-root", default=None, help="Optional shared Stage 1 prep root across variants")
    parser.add_argument("--static-prep-root", default=None, help="Optional root containing meeting-specific reusable user_profiles and jit_briefings")
    parser.add_argument("--eim-cache-root", default=None, help="Optional root for meeting-scoped shared EIM caches")

    parser.add_argument("--meetings", default=None, help="Comma-separated list of meeting IDs to process")
    parser.add_argument(
        "--skip",
        default=None,
        help="Comma-separated exact meeting IDs, inclusive ranges (<start>..<end>), or glob patterns to exclude",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N meetings")
    parser.add_argument("--participants", default="all", help="Comma-separated participants or 'all'")

    parser.add_argument("--capa-N", type=int, default=20, help="CAPA window size N")
    parser.add_argument("--capa-k", type=int, default=5, help="CAPA window size k")
    parser.add_argument("--freeze-anchor-inventory", action="store_true", help="Freeze anchor inventory before rebuilding k-specific CAPA episodes")
    parser.add_argument("--anchor-inventory-root", "--anchor-inventory-dir", dest="anchor_inventory_root", default=None, help="Optional root directory containing frozen anchor inventory JSON files")
    parser.add_argument("--anchor-seed-k", type=int, default=None, help="Seed k used when freezing anchor inventory with llm-idea")
    parser.add_argument("--cap-meeting", type=int, default=20, help="JIT cap per meeting")
    parser.add_argument("--cap-per-participant", type=int, default=5, help="JIT cap per participant")
    parser.add_argument("--min-gap", type=int, default=0, help="Min gap for JIT")

    parser.add_argument("--ctx-strict", type=float, default=0.75, help="Strict context threshold")
    parser.add_argument("--rel-strict", type=float, default=0.75, help="Strict relevance threshold")
    parser.add_argument("--ctx-loose", type=float, default=0.60, help="Loose context threshold")
    parser.add_argument("--rel-loose", type=float, default=0.60, help="Loose relevance threshold")

    parser.add_argument("--delegate-variant", choices=["capa", "raw-modular", "raw-modular-fair", "mono-raw", "mono-raw-reflection", "mono-raw-reflection-v2"], default="capa", help="CAPA delegate runtime variant")
    parser.add_argument("--no-recalibration", action="store_true", help="Disable recalibrator (CAPA only)")
    parser.add_argument("--raw-context-N", type=int, default=50, help="Raw-baseline transcript window size")
    parser.add_argument("--mono-raw-context-N", type=int, default=None, help="Deprecated alias for --raw-context-N")

    parser.add_argument("--model", default="gpt-4o", help="LLM model to use; must exist in config_extended.json")
    parser.add_argument("--resume", action="store_true", help="Skip already-completed meetings")
    parser.add_argument("--force", action="store_true", help="Force re-run each selected meeting")
    parser.add_argument("--skip-generation", action="store_true", help="Skip Stage 1 in each selected meeting")
    parser.add_argument("--skip-simulation", action="store_true", help="Skip Stage 2 in each selected meeting")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip Stage 3 in each selected meeting")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run without executing")

    args = parser.parse_args()

    args.raw_context_N = normalize_raw_context_n(args.raw_context_N, args.mono_raw_context_N)

    input_dirs = normalize_input_dirs(args.input_dirs, args.input_dir)
    if not input_dirs:
        logger.error("At least one input directory must be provided via --input-dirs and/or --input-dir")
        sys.exit(1)
    missing_input_dirs = [path for path in input_dirs if not path.is_dir()]
    if missing_input_dirs:
        for path in missing_input_dirs:
            logger.error(f"Input directory not found: {path}")
        sys.exit(1)

    base_output_root = Path(args.output_dir).resolve()
    variant_output_root = base_output_root / args.delegate_variant
    shared_prep_root = Path(args.shared_prep_root).resolve() if args.shared_prep_root else (base_output_root / "_prep")

    if args.delegate_variant != "capa" and args.no_recalibration:
        logger.warning("--no-recalibration is ignored for raw delegate variants")

    meetings_filter = [m.strip() for m in args.meetings.split(",") if m.strip()] if args.meetings else None
    try:
        skip_rules = parse_skip_rules(args.skip)
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)
    try:
        meeting_files = discover_meetings(input_dirs, meetings_filter, skip_rules)
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)

    if args.limit:
        meeting_files = meeting_files[:args.limit]

    if not meeting_files:
        joined_inputs = ", ".join(str(path) for path in input_dirs)
        logger.error(f"No meeting files found in selected input dirs: {joined_inputs}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info(f"BATCH RUN: {len(meeting_files)} meetings")
    logger.info(f"Input dirs: {[str(path) for path in input_dirs]}")
    logger.info(f"Variant Output Root: {variant_output_root}")
    logger.info(f"Shared Prep Root: {shared_prep_root}")
    if args.static_prep_root:
        logger.info(f"Static Prep Root: {Path(args.static_prep_root).resolve()}")
    if args.eim_cache_root:
        logger.info(f"Shared EIM Cache Root: {Path(args.eim_cache_root).resolve()}")
    logger.info(f"Delegate Variant: {args.delegate_variant}")
    logger.info("=" * 60)

    batch_log = load_batch_log(variant_output_root)
    batch_log["config"] = {
        "input_dirs": [str(path) for path in input_dirs],
        "output_root": str(base_output_root),
        "variant_output_root": str(variant_output_root),
        "shared_prep_root": str(shared_prep_root),
        "participants": args.participants,
        "capa_N": args.capa_N,
        "capa_k": args.capa_k,
        "cap_meeting": args.cap_meeting,
        "cap_per_participant": args.cap_per_participant,
        "min_gap": args.min_gap,
        "freeze_anchor_inventory": args.freeze_anchor_inventory,
        "anchor_inventory_root": args.anchor_inventory_root,
        "anchor_seed_k": args.anchor_seed_k,
        "static_prep_root": args.static_prep_root,
        "eim_cache_root": args.eim_cache_root,
        "ctx_strict": args.ctx_strict,
        "rel_strict": args.rel_strict,
        "ctx_loose": args.ctx_loose,
        "rel_loose": args.rel_loose,
        "no_recalibration": args.no_recalibration,
        "resume": args.resume,
        "limit": args.limit,
        "meetings_filter": meetings_filter,
        "skip": args.skip,
        "model": args.model,
        **build_delegate_capabilities(args.delegate_variant, args.no_recalibration),
        **build_delegate_runtime_metadata(args.delegate_variant, args.raw_context_N, args.capa_k),
    }
    if len(input_dirs) == 1:
        batch_log["config"]["input_dir"] = str(input_dirs[0])
    save_batch_log(variant_output_root, batch_log)

    completed = 0
    partial = 0
    failed = 0
    skipped = 0

    for index, meeting_file in enumerate(meeting_files, start=1):
        meeting_id = extract_meeting_id(meeting_file)
        meeting_output_dir = variant_output_root / meeting_id
        meeting_prep_dir = shared_prep_root / meeting_id

        logger.info(f"\n[{index}/{len(meeting_files)}] Processing: {meeting_id}")

        if args.resume and not args.force and is_meeting_complete(meeting_output_dir, meeting_id, args, meeting_prep_dir):
            logger.info("  → Skipping (already complete for this variant/config)")
            skipped += 1
            batch_log["meetings"][meeting_id] = {
                "status": "skipped",
                "input_path": str(meeting_file),
                "output_dir": str(meeting_output_dir),
                "shared_prep_dir": str(meeting_prep_dir),
                "capa_k": args.capa_k,
                "anchor_inventory_dir": resolve_meeting_anchor_inventory_dir(args.anchor_inventory_root, meeting_id),
                "static_prep_source_dir": resolve_meeting_static_prep_dir(args.static_prep_root, meeting_id),
                "eim_cache_dir": resolve_meeting_eim_cache_dir(args.eim_cache_root, meeting_id),
                **build_delegate_runtime_metadata(args.delegate_variant, args.raw_context_N, args.capa_k),
            }
            save_batch_log(variant_output_root, batch_log)
            continue

        batch_log["meetings"][meeting_id] = {
            "status": "running",
            "started": datetime.now().isoformat(),
            "input_path": str(meeting_file),
            "output_dir": str(meeting_output_dir),
            "shared_prep_dir": str(meeting_prep_dir),
            "capa_k": args.capa_k,
            "anchor_inventory_dir": resolve_meeting_anchor_inventory_dir(args.anchor_inventory_root, meeting_id),
            "static_prep_source_dir": resolve_meeting_static_prep_dir(args.static_prep_root, meeting_id),
            "eim_cache_dir": resolve_meeting_eim_cache_dir(args.eim_cache_root, meeting_id),
            **build_delegate_capabilities(args.delegate_variant, args.no_recalibration),
            **build_delegate_runtime_metadata(args.delegate_variant, args.raw_context_N, args.capa_k),
        }
        save_batch_log(variant_output_root, batch_log)

        start = datetime.now()
        run_result = run_meeting_pipeline(meeting_file, meeting_output_dir, meeting_prep_dir, args, dry_run=args.dry_run)
        elapsed = (datetime.now() - start).total_seconds()

        meeting_status = run_result.get("status") or ("dry_run" if args.dry_run else "failed")
        log_status = "complete" if meeting_status == "success" else meeting_status
        log_payload = {
            "status": log_status,
            "elapsed": elapsed,
            "input_path": str(meeting_file),
            "output_dir": str(meeting_output_dir),
            "shared_prep_dir": str(meeting_prep_dir),
            "capa_k": args.capa_k,
            "anchor_inventory_dir": resolve_meeting_anchor_inventory_dir(args.anchor_inventory_root, meeting_id),
            "static_prep_source_dir": resolve_meeting_static_prep_dir(args.static_prep_root, meeting_id),
            "eim_cache_dir": resolve_meeting_eim_cache_dir(args.eim_cache_root, meeting_id),
            "returncode": run_result.get("returncode"),
            "pipeline_status": run_result.get("pipeline_status") or {},
            **build_delegate_capabilities(args.delegate_variant, args.no_recalibration),
            **build_delegate_runtime_metadata(args.delegate_variant, args.raw_context_N, args.capa_k),
        }

        if args.dry_run:
            skipped += 1
            batch_log["meetings"][meeting_id] = log_payload
            logger.info(f"  → Dry run prepared in {elapsed:.1f}s")
        elif meeting_status == "success":
            completed += 1
            batch_log["meetings"][meeting_id] = log_payload
            logger.info(f"  → Completed in {elapsed:.1f}s")
        elif meeting_status == "partial":
            partial += 1
            batch_log["meetings"][meeting_id] = log_payload
            logger.error(f"  → Partial after {elapsed:.1f}s")
        else:
            failed += 1
            batch_log["meetings"][meeting_id] = log_payload
            logger.error(f"  → Failed after {elapsed:.1f}s")

        save_batch_log(variant_output_root, batch_log)

    batch_log["completed_at"] = datetime.now().isoformat()
    batch_log["summary"] = {
        "total": len(meeting_files),
        "completed": completed,
        "partial": partial,
        "failed": failed,
        "skipped": skipped,
    }
    save_batch_log(variant_output_root, batch_log)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"BATCH COMPLETE: {completed} done, {partial} partial, {failed} failed, {skipped} skipped")
    logger.info(f"Variant results in: {variant_output_root}")
    logger.info(f"Shared prep in: {shared_prep_root}")
    logger.info(f"{'=' * 60}")

    sys.exit(1 if (partial > 0 or failed > 0) else 0)


if __name__ == "__main__":
    main()
