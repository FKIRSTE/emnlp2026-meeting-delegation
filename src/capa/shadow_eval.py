from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from ..environment_interpreter import EnvironmentInterpretationModule
    from ..llm_client import LLMClient
    from ..schemas import EIMMasterState, EIMMeetingStateVector
    from .delegate_judge import CapaDelegateJudge
    from .episode_utils import build_official_idea_targets
    from .types import DelegateErrorReport
except ImportError:
    from environment_interpreter import EnvironmentInterpretationModule
    from llm_client import LLMClient
    from schemas import EIMMasterState, EIMMeetingStateVector
    from capa.delegate_judge import CapaDelegateJudge
    from capa.episode_utils import build_official_idea_targets
    from capa.types import DelegateErrorReport


@dataclass
class ShadowEvalResult:
    prediction: Optional[dict]
    environment_report: Optional[dict]
    delegate_report: Optional[DelegateErrorReport]
    next_state: Optional[EIMMeetingStateVector]
    capa_step_successful: bool


class ShadowCapaEvaluator:
    """Hidden delegate-judge-only evaluation for transcript-only baselines."""

    def __init__(self, client: LLMClient, late_window_k: int = 5):
        self.client = client
        self.delegate_judge = CapaDelegateJudge(client, late_window_k=late_window_k)

    def seed_state(
        self,
        precomputed_state: Optional[EIMMeetingStateVector],
        snapshot_utterances: List[Dict[str, Any]],
    ) -> Optional[EIMMeetingStateVector]:
        if precomputed_state is not None:
            return self._refresh_snapshot_analysis(precomputed_state, snapshot_utterances)
        if not snapshot_utterances:
            return None
        return self._analyze_snapshot_with_master_state(None, snapshot_utterances)

    def step(
        self,
        *,
        eim_state: Optional[EIMMeetingStateVector],
        last_utterances: List[Dict[str, Any]],
        gt_window: List[Dict[str, Any]],
        controller_output: Optional[Any],
        generator_output: Optional[Any],
        delegate_payload: Optional[Dict[str, Any]] = None,
        updated_snapshot_window: Optional[List[Dict[str, Any]]] = None,
        episode: Optional[object] = None,
        artifact_tracker: Optional[object] = None,
    ) -> ShadowEvalResult:
        state = eim_state or self.seed_state(None, last_utterances)

        prediction = None
        environment_report = None
        delegate_report = None

        controller_payload = delegate_payload or self._serialize_controller_payload(controller_output)
        if artifact_tracker:
            artifact_tracker.capture_delegate_judge_input(controller_payload)

        if state is not None and controller_payload:
            delegate_report = self.delegate_judge.judge_delegate(
                controller_output=controller_payload,
                generator_output=generator_output,
                eim_state=state,
                gt_window=gt_window,
                idea_targets=self._build_idea_targets(episode, last_utterances, gt_window),
            )
        if artifact_tracker:
            artifact_tracker.capture_delegate_judge_output(delegate_report)
            capture_delegate_trace = getattr(artifact_tracker, "capture_delegate_judge_trace", None)
            if callable(capture_delegate_trace):
                capture_delegate_trace(getattr(self.delegate_judge, "last_reconciliation_trace", None))

        next_state = self.advance_state(state, updated_snapshot_window)
        capa_step_successful = bool(delegate_report is not None)
        return ShadowEvalResult(
            prediction=prediction,
            environment_report=environment_report,
            delegate_report=delegate_report,
            next_state=next_state,
            capa_step_successful=capa_step_successful,
        )

    def advance_state(
        self,
        current_state: Optional[EIMMeetingStateVector],
        updated_snapshot_window: Optional[List[Dict[str, Any]]],
    ) -> Optional[EIMMeetingStateVector]:
        if not updated_snapshot_window:
            return current_state
        next_state = self._analyze_snapshot_with_master_state(current_state, updated_snapshot_window)
        return next_state or current_state

    def _clone_state(self, state: EIMMeetingStateVector) -> EIMMeetingStateVector:
        if hasattr(state, "model_dump"):
            return EIMMeetingStateVector(**state.model_dump())
        return EIMMeetingStateVector(**dict(state))

    def _refresh_snapshot_analysis(
        self,
        state: EIMMeetingStateVector,
        snapshot_utterances: List[Dict[str, Any]],
    ) -> EIMMeetingStateVector:
        cloned_state = self._clone_state(state)
        if not snapshot_utterances:
            return cloned_state
        refreshed_state = self._analyze_snapshot_with_master_state(cloned_state, snapshot_utterances)
        return refreshed_state or cloned_state

    def _analyze_snapshot_with_master_state(
        self,
        state: Optional[EIMMeetingStateVector],
        snapshot_utterances: List[Dict[str, Any]],
    ) -> Optional[EIMMeetingStateVector]:
        eim = EnvironmentInterpretationModule(self.client)
        if state is not None:
            eim.master_state = EIMMasterState(
                topic_summary_log=dict(state.topic_summary_log or {}),
                participant_models=dict(state.participant_models or {}),
                identity_aliases=dict(getattr(state, "identity_aliases", {}) or {}),
            )
        return eim.analyze_snapshot(snapshot_utterances)

    def _serialize_controller_payload(self, controller_output: Optional[Any]) -> Dict[str, Any]:
        if controller_output is None:
            return {}
        if hasattr(controller_output, "model_dump"):
            return controller_output.model_dump()
        if isinstance(controller_output, dict):
            return controller_output
        return {}

    def _build_idea_targets(
        self,
        episode: Optional[object],
        last_utterances: List[Dict[str, Any]],
        gt_window: List[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        try:
            return build_official_idea_targets(
                episode,
                context_utterances=last_utterances,
                gt_window=gt_window,
            )
        except Exception:
            return None
