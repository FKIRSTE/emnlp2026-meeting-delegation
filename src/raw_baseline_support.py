from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from raw_context import slice_transcript_window


RAW_BASELINE_VARIANTS = frozenset({"raw-modular", "raw-modular-fair", "mono-raw", "mono-raw-reflection", "mono-raw-reflection-v2"})
DEFAULT_RAW_CONTEXT_N = 50
RAW_AGENT_REGIMES = {
    "mono-raw": "mono_raw_v3",
    "mono-raw-reflection": "mono_raw_reflection_v1",
    "mono-raw-reflection-v2": "mono_raw_reflection_v2",
    "raw-modular": "raw_modular_v2",
    "raw-modular-fair": "raw_modular_fair_v1",
}
DELEGATE_JUDGE_REGIME = "delegate_judge_v3_calibrated"


def is_raw_baseline_variant(delegate_variant: Optional[str]) -> bool:
    return delegate_variant in RAW_BASELINE_VARIANTS


def normalize_raw_context_n(
    raw_context_n: Optional[int],
    mono_raw_context_n: Optional[int] = None,
    *,
    default: int = DEFAULT_RAW_CONTEXT_N,
) -> int:
    resolved = int(raw_context_n or default)
    if mono_raw_context_n is not None:
        alias_value = int(mono_raw_context_n)
        if raw_context_n not in (None, default) and resolved != alias_value:
            raise ValueError("--raw-context-N and --mono-raw-context-N disagree; use only one or keep them equal")
        resolved = alias_value
    return max(1, resolved)


def build_raw_runtime_metadata(delegate_variant: str) -> Dict[str, Any]:
    if not is_raw_baseline_variant(delegate_variant):
        return {}
    return {
        "raw_agent_regime": RAW_AGENT_REGIMES[delegate_variant],
    }


def build_delegate_runtime_metadata(
    delegate_variant: str,
    raw_context_n: int,
    capa_k: Optional[int] = None,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "delegate_judge_regime": DELEGATE_JUDGE_REGIME,
    }
    if capa_k is not None:
        metadata["late_window_k"] = int(capa_k)
    if is_raw_baseline_variant(delegate_variant):
        metadata["raw_context_N"] = raw_context_n
    metadata.update(build_raw_runtime_metadata(delegate_variant))
    return metadata


def build_delegate_capabilities(
    delegate_variant: str,
    no_recalibration: bool,
    *,
    use_two_tier: bool = True,
) -> Dict[str, Any]:
    if delegate_variant in {"raw-modular", "raw-modular-fair"}:
        uses_helpers = True
    elif delegate_variant in {"mono-raw", "mono-raw-reflection", "mono-raw-reflection-v2"}:
        uses_helpers = False
    else:
        uses_helpers = bool(use_two_tier)

    uses_explicit_state = delegate_variant == "capa"
    uses_feedback = delegate_variant == "capa" and bool(use_two_tier) and not no_recalibration
    uses_predictor = delegate_variant == "capa" and bool(use_two_tier)
    uses_env_judge = delegate_variant == "capa" and bool(use_two_tier)
    uses_delegate_judge = is_raw_baseline_variant(delegate_variant) or (delegate_variant == "capa" and bool(use_two_tier))

    capabilities = {
        "delegate_variant": delegate_variant,
        "uses_explicit_state": uses_explicit_state,
        "uses_helpers": uses_helpers,
        "uses_feedback": uses_feedback,
        "uses_predictor": uses_predictor,
        "uses_env_judge": uses_env_judge,
        "uses_delegate_judge": uses_delegate_judge,
    }
    if delegate_variant in {"mono-raw-reflection", "mono-raw-reflection-v2"}:
        capabilities["uses_runtime_reflection_critic"] = True
        capabilities["uses_reflection_memory"] = True
    return capabilities


def prepare_raw_delegate_judge_context(
    *,
    delegate_variant: str,
    evaluation_context: Dict[str, Any],
    helper_context: Optional[Dict[str, Any]] = None,
    uses_helper_priors_in_judge: bool = False,
) -> Dict[str, Any]:
    prepared = dict(helper_context or {})
    prepared.setdefault("evaluation_context", evaluation_context)
    prepared["uses_helper_priors_in_judge"] = uses_helper_priors_in_judge
    prepared.setdefault("delegate_variant", delegate_variant)
    return prepared


class RawTranscriptWindowResolver:
    """Loads and reconstructs expanded raw transcript windows for raw baselines."""

    def __init__(self, raw_context_n: int, transcript_path: Optional[str] = None):
        self.raw_context_n = max(1, int(raw_context_n or DEFAULT_RAW_CONTEXT_N))
        self.transcript_path = str(Path(transcript_path).resolve()) if transcript_path else None
        self._transcript_utterances: Optional[List[Dict[str, Any]]] = None
        self._transcript_index_by_id: Dict[str, int] = {}

    def ensure_episode_compatibility(self, episodes: Sequence[Any]) -> None:
        max_episode_n = max((int(getattr(ep, "N", 0)) for ep in episodes), default=0)
        if self.raw_context_n > max_episode_n and not self.transcript_path:
            raise ValueError(
                f"raw baseline context N={self.raw_context_n} exceeds stored episode N={max_episode_n}; "
                "pass --transcript-path to reconstruct longer context"
            )

    def resolve_window(
        self,
        episode: Any,
        delta: int,
        snapshot_window: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        desired_window = self.raw_context_n
        if desired_window <= len(snapshot_window):
            return list(snapshot_window[-desired_window:])

        transcript_utterances = self._load_transcript_utterances()
        cue_index = getattr(episode, "cue_index", None)
        if cue_index is None:
            cue_index = self._transcript_index_by_id.get(getattr(episode, "cue_utt_id", None))
        if cue_index is None and snapshot_window:
            cue_index = self._transcript_index_by_id.get(snapshot_window[-1].get("utterance_id"))
        if cue_index is None:
            raise ValueError(
                f"Could not determine cue index for raw-baseline episode {getattr(episode, 'test_case_id', 'unknown')}"
            )

        end_index = int(cue_index) + int(delta)
        return slice_transcript_window(transcript_utterances, end_index=end_index, window_size=desired_window)

    def _extract_transcript_utterances(self, payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, list):
            return [utt for utt in payload if isinstance(utt, dict)]
        if isinstance(payload, dict):
            for key in ("utterances", "transcript", "meeting_transcript"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [utt for utt in value if isinstance(utt, dict)]
                if isinstance(value, dict):
                    nested = value.get("utterances")
                    if isinstance(nested, list):
                        return [utt for utt in nested if isinstance(utt, dict)]
        return []

    def _load_transcript_utterances(self) -> List[Dict[str, Any]]:
        if self._transcript_utterances is not None:
            return self._transcript_utterances
        if not self.transcript_path:
            raise ValueError("raw-baseline longer-context reconstruction requires --transcript-path")
        transcript_file = Path(self.transcript_path)
        if not transcript_file.exists():
            raise FileNotFoundError(f"Transcript file not found: {transcript_file}")
        with open(transcript_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        utterances = self._extract_transcript_utterances(payload)
        if not utterances:
            raise ValueError(f"Could not extract transcript utterances from {transcript_file}")
        self._transcript_utterances = utterances
        self._transcript_index_by_id = {
            utt.get("utterance_id"): idx
            for idx, utt in enumerate(utterances)
            if isinstance(utt, dict) and utt.get("utterance_id")
        }
        return utterances
