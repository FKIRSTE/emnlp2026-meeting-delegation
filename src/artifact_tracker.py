"""
CAPA Artifact Tracker - Lightweight tracking for CAPA system intermediate outputs

This module provides centralized artifact collection and saving for the CAPA system,
capturing Controller outputs, Judge assessments, and Recalibrator patches for analysis.
"""

import json
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime

from schemas import (
    CapaStepArtifacts, 
    CapaSimulationArtifacts,
    ControllerOutput,
    EIMMasterState
)
from pipeline_resume import atomic_write_json
from raw_baseline_support import DELEGATE_JUDGE_REGIME

try:
    from capa.types import PredictionResponse, ErrorReport, DelegateErrorReport
except ImportError:
    # Handle case where CAPA is not available
    PredictionResponse = None
    ErrorReport = None
    DelegateErrorReport = None


class CapaArtifactTracker:
    """Lightweight artifact tracker for CAPA intermediate outputs"""
    
    def __init__(self, 
                 participant_id: Optional[str] = None, 
                 meeting_id: Optional[str] = None, 
                 output_dir: Optional[Path] = None):
        self.participant_id = participant_id or "unknown"
        self.meeting_id = meeting_id or "unknown" 
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        
        # Dynamic context attributes that can be updated during simulation
        self.test_case_id: Optional[str] = None
        self.delta: Optional[int] = None
        self.anchor_utt_id: Optional[str] = None
        
        # Step-level artifacts storage
        self.step_artifacts: List[Dict] = []
    
    def get_simulation_artifacts(self) -> Dict:
        """Get collected simulation artifacts in a simple format"""
        return {
            "participant_id": self.participant_id,
            "meeting_id": self.meeting_id,
            "delegate_judge_regime": DELEGATE_JUDGE_REGIME,
            "timestamp": datetime.now().isoformat(),
            "step_artifacts": self.step_artifacts
        }
        
        print(f"[ArtifactTracker] Initialized for {participant_id} - {meeting_id}")
    
    def start_episode(self, episode_id: str):
        """Initialize artifact collection for episode"""
        self.current_episode_id = episode_id
        if episode_id not in self.simulation_artifacts.artifacts_by_episode:
            self.simulation_artifacts.artifacts_by_episode[episode_id] = []
        print(f"[ArtifactTracker] Started episode: {episode_id}")
    
    def start_step(self, delta: int, anchor_utt_id: str):
        """Initialize artifact collection for step"""
        if not self.current_episode_id:
            print("[ArtifactTracker] Warning: start_step called without active episode.")
            return
            
        self.current_step_artifacts = CapaStepArtifacts(
            test_case_id=self.current_episode_id,
            delta=delta,
            anchor_utt_id=anchor_utt_id,
            participant_id=self.participant_id,
            meeting_id=self.meeting_id
        )
        print(f"[ArtifactTracker] Started step: {self.current_episode_id} delta={delta} anchor={anchor_utt_id}")

    def _ensure_current_step(self):
        """Ensure we have a current step artifact dict with proper identification"""
        if not hasattr(self, '_current_step') or self._current_step is None:
            self._current_step = {
                # Essential identification - needed to link back to CAPA simulation
                "test_case_id": self.test_case_id,
                "delta": self.delta,
                "anchor_utt_id": self.anchor_utt_id,
                # Context for analysis
                "participant_id": self.participant_id,
                "meeting_id": self.meeting_id,
                "delegate_judge_regime": DELEGATE_JUDGE_REGIME,
                "timestamp": datetime.now().isoformat(),
                # Step-specific artifact container
                "helper_io": []
            }

    def _step_context_label(self) -> str:
        test_case = None
        delta = None
        if hasattr(self, '_current_step') and self._current_step:
            test_case = self._current_step.get('test_case_id')
            delta = self._current_step.get('delta')
        if test_case is None:
            test_case = self.test_case_id
        if delta is None:
            delta = self.delta
        delta_display = '?' if delta is None else delta
        return f"test_case={test_case or 'unknown'}, delta={delta_display}"

    def capture_delegate_utterance(self, utterance: Optional[str]):
        """Capture the final generated utterance text for this step (may be empty string).

        Stored under key 'delegate_utterance' in the current step artifact to enable
        delegate recall (lexical) and downstream analyses without revisiting simulation logs.
        """
        if utterance is None:
            utterance = ""
        self._ensure_current_step()
        self._current_step["delegate_utterance"] = utterance
        print(f"[ArtifactTracker] Captured delegate utterance ({self._step_context_label()}).")

    def capture_step_context(self, gt_window: Optional[List[Dict]], represented_user: Optional[str]):
        """Capture compact GT-step context and presence of represented user in the window.

        - gt_window_speakers: list of speaker_ids in the provided window order
        - gt_anchor_text: utterance_content for the anchor (gt_window[0]) if available
        - gt_window_ids: utterance_ids for the window (t+1..t+k) to support idea-anchor tagging in evals
        - represented_in_window: whether represented_user appears in gt_window speakers
        """
        if not gt_window:
            return
        self._ensure_current_step()
        try:
            speakers = [u.get("speaker_id") for u in gt_window if isinstance(u, dict)]
            self._current_step["gt_window_speakers"] = speakers
            ids = [u.get("utterance_id") for u in gt_window if isinstance(u, dict)]
            self._current_step["gt_window_ids"] = ids
            anchor_text = None
            if isinstance(gt_window[0], dict):
                anchor_text = gt_window[0].get("utterance_content") or gt_window[0].get("content")
            if anchor_text is not None:
                self._current_step["gt_anchor_text"] = anchor_text
            if represented_user is not None:
                self._current_step["represented_in_window"] = bool(represented_user in speakers)
        except Exception:
            # Best-effort capture; ignore failures
            pass

    def capture_delegate_judge_input(self, delegate_payload: Optional[Dict[str, Any]]):
        """Capture the sanitized controller payload provided to the delegate judge."""
        if not delegate_payload:
            return
        self._ensure_current_step()
        try:
            sanitized_payload = json.loads(json.dumps(delegate_payload, ensure_ascii=False))
        except (TypeError, ValueError):
            sanitized_payload = delegate_payload
        self._current_step["delegate_judge_input"] = sanitized_payload
        print(f"[ArtifactTracker] Recorded delegate judge input ({self._step_context_label()}).")

    def capture_helper_io(self, helper_name: str, payload: Optional[Dict], response: Optional[Dict]):
        """Record helper tool inputs and outputs for later analysis.

        Payload/response are deep-copied to preserve their call-time state even
        if the underlying objects are mutated later in the pipeline.
        """
        self._ensure_current_step()
        # Defensive deep copy via JSON round-trip where possible; fall back to
        # original objects if they are not JSON-serializable.
        try:
            payload_copy = json.loads(json.dumps(payload, ensure_ascii=False)) if payload is not None else None
        except (TypeError, ValueError):
            payload_copy = payload
        try:
            response_copy = json.loads(json.dumps(response, ensure_ascii=False)) if response is not None else None
        except (TypeError, ValueError):
            response_copy = response
        entry = {
            "name": helper_name,
            "payload": payload_copy,
            "response": response_copy,
        }
        self._current_step.setdefault("helper_io", []).append(entry)

    def capture_controller_output(self, controller_output: ControllerOutput):
        """Capture Controller output - key for debugging elimination logic"""
        self._ensure_current_step()
        self._current_step["controller_output"] = (
            controller_output.model_dump(exclude_none=True)
            if hasattr(controller_output, 'model_dump')
            else controller_output
        )
        print(f"[ArtifactTracker] Captured controller output ({self._step_context_label()}).")
    
    def capture_predictor_output(self, predictor_output):
        """Capture CAPA Environment Predictor output."""
        if predictor_output is None:
            return
        self._ensure_current_step()
        if hasattr(predictor_output, 'model_dump'):
            self._current_step["environment_predictor_output"] = predictor_output.model_dump()
        else:
            self._current_step["environment_predictor_output"] = predictor_output
        print(f"[ArtifactTracker] Captured environment predictor output ({self._step_context_label()}).")
    
    def capture_judge_output(self, judge_output):
        """Capture Environment Judge output."""
        if judge_output is None:
            return
        self._ensure_current_step() 
        if hasattr(judge_output, 'model_dump'):
            self._current_step["environment_judge_output"] = judge_output.model_dump()
        else:
            self._current_step["environment_judge_output"] = judge_output
        print(f"[ArtifactTracker] Captured environment judge output ({self._step_context_label()}).")
    
    def capture_environment_judge_output(self, judge_output):
        """Capture Environment Judge output (alias for compatibility)"""
        self.capture_judge_output(judge_output)
    
    def capture_delegate_judge_output(self, delegate_judge_output):
        """Capture Delegate Judge output"""
        self._ensure_current_step()
        if delegate_judge_output and hasattr(delegate_judge_output, 'model_dump'):
            self._current_step["delegate_judge_output"] = delegate_judge_output.model_dump()
        else:
            self._current_step["delegate_judge_output"] = delegate_judge_output
        print(f"[ArtifactTracker] Captured delegate judge output ({self._step_context_label()}).")

    def capture_delegate_judge_trace(self, delegate_trace: Optional[Dict[str, Any]]):
        """Capture verifier and reconciliation traces emitted by the delegate judge."""
        if not delegate_trace:
            return
        self._ensure_current_step()
        try:
            sanitized_trace = json.loads(json.dumps(delegate_trace, ensure_ascii=False))
        except (TypeError, ValueError):
            sanitized_trace = delegate_trace
        for key, value in (sanitized_trace or {}).items():
            self._current_step[key] = value
        print(f"[ArtifactTracker] Captured delegate judge trace ({self._step_context_label()}).")
    
    def capture_recalibrator_output(self, recalibrator_output, patch_applied: bool = False):
        """Capture Recalibrator output and patch application status."""
        if recalibrator_output is None and not patch_applied:
            return
        self._ensure_current_step()
        if recalibrator_output is not None and hasattr(recalibrator_output, 'model_dump'):
            self._current_step["recalibrator_output"] = recalibrator_output.model_dump()
        else:
            self._current_step["recalibrator_output"] = recalibrator_output
        self._current_step["recalibrator_patch_applied"] = patch_applied
        print(f"[ArtifactTracker] Captured recalibrator output ({self._step_context_label()}), patch_applied={patch_applied}.")

    def capture_recalibrator_snapshot_analysis(self, snapshot_analysis: Optional[Dict]):
        """Capture recalibrator-provided snapshot_analysis for the step (if any)."""
        if snapshot_analysis is None:
            return
        self._ensure_current_step()
        try:
            sanitized = json.loads(json.dumps(snapshot_analysis, ensure_ascii=False))
        except (TypeError, ValueError):
            sanitized = snapshot_analysis
        self._current_step["recalibrator_snapshot_analysis"] = sanitized
        print(f"[ArtifactTracker] Captured recalibrator snapshot_analysis ({self._step_context_label()}).")

    def capture_recalibrator_guidance(self, guidance: Optional[Dict]):
        """Capture GuidanceBundle (if any)."""
        if not guidance:
            return
        self._ensure_current_step()
        try:
            sanitized = json.loads(json.dumps(guidance, ensure_ascii=False))
        except (TypeError, ValueError):
            sanitized = guidance
        self._current_step["recalibrator_guidance"] = sanitized
        print(f"[ArtifactTracker] Captured recalibrator guidance ({self._step_context_label()}).")
    
    def finalize_step(self, capa_step_successful: bool = None):
        """Finalize current step and add to artifacts list with full identification"""
        if hasattr(self, '_current_step') and self._current_step is not None:
            if capa_step_successful is not None:
                self._current_step["capa_step_successful"] = capa_step_successful
            
            # Add a comprehensive step identifier for easy lookup
            step_key = (
                f"{self._current_step.get('participant_id', 'unknown')}::"
                f"{self._current_step.get('test_case_id', 'unknown')}::"
                f"delta_{self._current_step.get('delta', 'unknown')}"
            )
            self._current_step["step_key"] = step_key
            
            self.step_artifacts.append(self._current_step)
            self._current_step = None
            print(f"[ArtifactTracker] Finalized step artifact: {step_key}.")
    
    def start_episode(self, episode_id: str):
        """Start tracking a new episode (optional for compatibility)"""
        pass
        
    def start_step(self, delta: int, anchor_utt_id: str):
        """Start tracking a new step (optional for compatibility)"""
        self.delta = delta
        self.anchor_utt_id = anchor_utt_id
    
    def save_artifacts(self, output_dir: Path) -> Path:
        """Save all collected artifacts to a separate file in artifacts subdirectory"""
        # Create artifacts subdirectory
        artifacts_dir = output_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate artifact filename matching the main simulation file pattern
        artifact_file = artifacts_dir / f"capa_artifacts_{self.participant_id}_{self.meeting_id}.json"
        
        # Get artifacts in simple format
        artifacts_data = self.get_simulation_artifacts()
        
        atomic_write_json(artifact_file, artifacts_data)
        
        print(f"[ArtifactTracker] Saved {len(self.step_artifacts)} artifacts to {artifact_file}.")
        
        return artifact_file
    
    def get_artifact_file_reference(self, artifact_file_path: Path) -> Dict:
        """Get a compact reference to the saved artifacts file for inclusion in main simulation output"""
        return {
            "artifacts_file": str(artifact_file_path.name),
            "artifacts_directory": str(artifact_file_path.parent.relative_to(artifact_file_path.parent.parent)),
            "total_step_artifacts": len(self.step_artifacts),
            "participant_id": self.participant_id,
            "meeting_id": self.meeting_id,
            "summary": self.get_artifact_summary()
        }
    
    def get_artifact_summary(self) -> Dict:
        """Get summary statistics focusing on intermediate CAPA components with full identification"""
        return {
            'participant_id': self.participant_id,
            'meeting_id': self.meeting_id,
            'total_capa_steps': len(self.step_artifacts),
            'step_keys': [a.get('step_key') for a in self.step_artifacts],
            # Intermediate component tracking (not in standard CAPA sim)
            'controller_decisions_captured': sum(1 for a in self.step_artifacts if a.get('controller_output')),
            'environment_predictions_captured': sum(1 for a in self.step_artifacts if a.get('environment_predictor_output')), 
            'environment_judge_assessments_captured': sum(1 for a in self.step_artifacts if a.get('environment_judge_output')),
            'delegate_judge_assessments_captured': sum(1 for a in self.step_artifacts if a.get('delegate_judge_output')),
            'recalibrator_patches_captured': sum(1 for a in self.step_artifacts if a.get('recalibrator_output')),
            'successful_eim_patches': sum(1 for a in self.step_artifacts if a.get('recalibrator_patch_applied')),
            'successful_capa_steps': sum(1 for a in self.step_artifacts if a.get('capa_step_successful')),
            # New: coverage for added instrumentation
            'delegate_utterances_captured': sum(1 for a in self.step_artifacts if 'delegate_utterance' in (a or {})),
            'gt_context_captured': sum(1 for a in self.step_artifacts if 'gt_window_speakers' in (a or {})),
        }


class CapaArtifactAnalyzer:
    """Analysis utilities for CAPA artifacts"""
    
    def __init__(self, participant_id: str, meeting_id: str, results_dir: Path):
        self.participant_id = participant_id
        self.meeting_id = meeting_id
        self.results_dir = Path(results_dir)
        
        # Load artifacts
        self.artifact_data = self._load_artifact_data()
    
    def _load_artifact_data(self) -> Optional[CapaSimulationArtifacts]:
        """Load artifact data from file"""
        artifact_file = self.results_dir / f"capa_artifacts_{self.participant_id}_{self.meeting_id}.json"
        
        if not artifact_file.exists():
            print(f"[ArtifactAnalyzer] Artifact file not found: {artifact_file}.")
            return None
        
        try:
            with open(artifact_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return CapaSimulationArtifacts.model_validate(data)
        except Exception as e:
            print(f"[ArtifactAnalyzer] Failed to load artifacts: {e}.")
            return None
    
    def get_elimination_contradictions(self) -> List[Dict]:
        """Find cases where elimination mentioned but point still used"""
        if not self.artifact_data:
            return []
            
        contradictions = []
        
        for artifact in self.artifact_data.artifacts:
            controller_output = artifact.controller_output
            if not controller_output or not controller_output.helper_candidates_digest:
                continue

            digest_map = {item.candidate_id: item for item in controller_output.helper_candidates_digest if getattr(item, 'candidate_id', None)}
            chosen_id = controller_output.active_candidate_id
            selected_point = controller_output.selected_point

            chosen_item = digest_map.get(chosen_id) if chosen_id else None
            if not chosen_item and selected_point:
                for item in digest_map.values():
                    if self._text_similarity(item.text or '', selected_point) > 0.8:
                        chosen_item = item
                        break

            if not chosen_item:
                continue

            treatment = (chosen_item.recommended_treatment or '').lower()
            if treatment in {'skip'}:
                contradictions.append({
                    'test_case_id': artifact.test_case_id,
                    'delta': artifact.delta,
                    'anchor_utt_id': artifact.anchor_utt_id,
                    'candidate_id': chosen_item.candidate_id,
                    'candidate_text': chosen_item.text,
                    'recommended_treatment': chosen_item.recommended_treatment,
                    'selected_point': selected_point,
                    'selection_reason': controller_output.selection_reason
                })

        return contradictions
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity measure"""
        if not text1 or not text2:
            return 0.0
        
        # Simple word-based similarity for now
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def get_judge_assessment_summary(self) -> Dict:
        """Summarize judge assessments across all steps"""
        if not self.artifact_data:
            return {}
            
        env_judge_stats = []
        delegate_judge_stats = []
        
        for artifact in self.artifact_data.artifacts:
            if artifact.environment_judge_output:
                env_judge_stats.append({
                    'intent_score': artifact.environment_judge_output.intent_score,
                    'topic_score': artifact.environment_judge_output.topic_score,
                    'floor_score': artifact.environment_judge_output.floor_score,
                    'category': artifact.environment_judge_output.category
                })
            
            if artifact.delegate_judge_output:
                delegate_judge_stats.append({
                    'decision_appropriateness': artifact.delegate_judge_output.decision_appropriateness,
                    'context_appropriateness': artifact.delegate_judge_output.context_appropriateness,
                    'relevance': artifact.delegate_judge_output.relevance,
                    'grounding': artifact.delegate_judge_output.grounding,
                    'category': artifact.delegate_judge_output.category,
                    'redundancy': artifact.delegate_judge_output.redundancy
                })
        
        return {
            'environment_judge_assessments': len(env_judge_stats),
            'delegate_judge_assessments': len(delegate_judge_stats),
            'env_avg_intent_score': sum(s['intent_score'] for s in env_judge_stats) / len(env_judge_stats) if env_judge_stats else 0,
            'env_avg_topic_score': sum(s['topic_score'] for s in env_judge_stats) / len(env_judge_stats) if env_judge_stats else 0,
            'delegate_avg_decision_appropriateness': sum(s['decision_appropriateness'] for s in delegate_judge_stats) / len(delegate_judge_stats) if delegate_judge_stats else 0,
            'delegate_redundancy_flags': sum(1 for s in delegate_judge_stats if s['redundancy']) if delegate_judge_stats else 0
        }
