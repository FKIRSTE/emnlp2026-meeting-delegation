from __future__ import annotations

"""
Artifact loader for CAPA evals.

Responsibilities:
- Load per-participant artifact JSON emitted by CapaArtifactTracker.save_artifacts.
- Normalize into a stream of step records with safe accessors.
- Provide helpers to group by episode and participant for aggregation.

This module is I/O light and pure-Python to keep tests simple.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
import json


@dataclass
class StepRecord:
    meeting_id: str
    participant_id: str
    test_case_id: Optional[str]
    delta: Optional[int]
    anchor_utt_id: Optional[str]
    # Captured components
    controller_output: Optional[dict]
    env_pred: Optional[dict]
    env_judge: Optional[dict]
    delegate_judge: Optional[dict]
    delegate_judge_input: Optional[dict]
    recalibrator_output: Optional[dict]
    # Instrumentation
    delegate_utterance: Optional[str]
    gt_window_speakers: Optional[List[str]]
    gt_window_ids: Optional[List[str]]
    represented_in_window: Optional[bool]
    gt_anchor_text: Optional[str]
    # Idea-unit / banality extensions
    gt_is_banal: Optional[bool]
    matched_unit_id: Optional[str]
    unit_timing_delta: Optional[int]
    delegate_judge_failure: Optional[dict]
    # Meta
    step_key: Optional[str]


def _safe_get(d: Optional[dict], path: List[str], default=None):
    cur = d or {}
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


def load_artifact_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_steps_from_artifacts(payload: Dict[str, Any]) -> Iterator[StepRecord]:
    meeting_id = payload.get("meeting_id") or "unknown"
    participant_id = payload.get("participant_id") or "unknown"
    steps = payload.get("step_artifacts") or []
    for s in steps:
        yield StepRecord(
            meeting_id=meeting_id,
            participant_id=participant_id,
            test_case_id=s.get("test_case_id"),
            delta=s.get("delta"),
            anchor_utt_id=s.get("anchor_utt_id"),
            controller_output=s.get("controller_output"),
            env_pred=s.get("environment_predictor_output"),
            env_judge=s.get("environment_judge_output"),
            delegate_judge=s.get("delegate_judge_output"),
            delegate_judge_input=s.get("delegate_judge_input"),
            recalibrator_output=s.get("recalibrator_output"),
            delegate_utterance=s.get("delegate_utterance"),
            gt_window_speakers=s.get("gt_window_speakers"),
            gt_window_ids=s.get("gt_window_ids"),
            represented_in_window=s.get("represented_in_window"),
            gt_anchor_text=s.get("gt_anchor_text"),
            gt_is_banal=_safe_get(s.get("delegate_judge_output"), ["gt_is_banal"], None) or _safe_get(s.get("environment_judge_output"), ["gt_is_banal"], None),
            matched_unit_id=_safe_get(s.get("delegate_judge_output"), ["matched_unit_id"], None),
            unit_timing_delta=_safe_get(s.get("delegate_judge_output"), ["unit_timing_delta"], None),
            delegate_judge_failure=s.get("delegate_judge_failure"),
            step_key=s.get("step_key"),
        )


def discover_artifact_files(results_dir: Path, meeting_id: str) -> List[Path]:
    """Find artifact files for a meeting under results_dir/artifacts.

    Supports both filename patterns noted in eval_plan.md (with and without _meeting suffix).
    """
    artifacts_dir = results_dir / "artifacts"
    if not artifacts_dir.exists():
        return []
    out: List[Path] = []
    for p in artifacts_dir.glob("capa_artifacts_*_*.json"):
        out.append(p)
    out += list(artifacts_dir.glob("capa_artifacts_*_*_meeting.json"))
    # De-duplicate
    uniq: Dict[str, Path] = {str(p.resolve()): p for p in out}
    # Filter by meeting_id by opening files (files are small)
    filtered: List[Path] = []
    for p in uniq.values():
        try:
            data = load_artifact_file(p)
            if (data.get("meeting_id") or "").lower() == (meeting_id or "").lower():
                filtered.append(p)
        except Exception:
            continue
    return filtered


def group_steps_by_episode(steps: Iterable[StepRecord]) -> Dict[str, List[StepRecord]]:
    by_ep: Dict[str, List[StepRecord]] = {}
    for s in steps:
        if s.test_case_id:
            key = f"{s.participant_id}\t{s.test_case_id}"
        else:
            key = s.step_key or f"{s.meeting_id}_{s.participant_id}_unknown"
        by_ep.setdefault(key, []).append(s)
    # ensure ordering by delta when available
    for k in list(by_ep.keys()):
        by_ep[k] = sorted(by_ep[k], key=lambda x: (x.delta is None, x.delta))
    return by_ep
