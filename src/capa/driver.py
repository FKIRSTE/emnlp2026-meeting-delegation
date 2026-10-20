"""
CAPA Driver: orchestrates predict -> judge -> (delegate judge) -> recalibrate -> patch.
"""


from typing import List, Dict, Optional

from .types import PredictionResponse, ErrorReport, EpisodicEvent, DelegateErrorReport
from .predictor import CapaPredictor
from .judge import CapaJudge
from .recalibrator import CapaRecalibrator
from .delegate_judge import CapaDelegateJudge
from .episode_utils import build_official_idea_targets
try:
    from ..schemas import EIMMasterState, EIMMeetingStateVector
except ImportError:
    from schemas import EIMMasterState, EIMMeetingStateVector


class CapaDriver:
    def __init__(self, predictor: CapaPredictor, judge: CapaJudge, recalibrator: CapaRecalibrator, delegate_judge: CapaDelegateJudge = None,
                 recalibrator_drives_state: bool = False, guidance_enabled: bool = False, disable_recalibration: bool = False):
        self.predictor = predictor
        self.judge = judge
        self.recalibrator = recalibrator
        self.delegate_judge = delegate_judge
        self.recalibrator_drives_state = recalibrator_drives_state
        self.guidance_enabled = guidance_enabled
        self.disable_recalibration = disable_recalibration  # Ablation flag

    def step(
        self,
        eim_state: EIMMeetingStateVector,
        last_utterances: List[Dict],
        current_gt_turn: Optional[Dict],
        apply_patch_fn,
        episode: Optional[object] = None,
        controller_output: Optional[object] = None,
        generator_output: Optional[object] = None,
        gt_window: Optional[List[Dict]] = None,
        artifact_tracker: Optional[object] = None,
        apply_guidance_fn=None,
        # Tracking info for guidance computation
        matched_unit_count: int = 0,
        total_unit_count: int = 0,
        is_in_cooldown: bool = False,
        episode_contribution_count: int = 0,
        record_match_fn=None,  # Optional callback: record_match_fn(unit_id) for match tracking
        participant_id: Optional[str] = None,  # Who the delegate represents
        matched_idea_unit_ids: Optional[set] = None,  # Set of already-matched unit IDs for coverage check
    ) -> Optional[EIMMasterState]:
        """
        Run a single CAPA step in shadow mode. Returns the diff applied or None.
        `apply_patch_fn` should be a callable like EIM.apply_patch.
        """
        pred = self.predictor.predict(eim_state, last_utterances)
        
        # CAPTURE PREDICTOR OUTPUT
        if artifact_tracker:
            artifact_tracker.capture_predictor_output(pred)
        
        if not pred:
            return None

        gtw = gt_window if gt_window is not None else ([current_gt_turn] if current_gt_turn else [])

        env_report = self.judge.judge_prediction(pred, eim_state, gtw)
        
        # CAPTURE ENVIRONMENT JUDGE OUTPUT
        if artifact_tracker:
            artifact_tracker.capture_environment_judge_output(env_report)
        
        if not env_report:
            return None

        # Optional delegate judge
        delegate_report = None
        if self.delegate_judge and controller_output is not None:
            if artifact_tracker:
                capture_delegate_input = getattr(artifact_tracker, "capture_delegate_judge_input", None)
                if callable(capture_delegate_input):
                    capture_delegate_input(controller_output)
            try:
                idea_targets = build_official_idea_targets(
                    episode,
                    context_utterances=last_utterances,
                    gt_window=gtw,
                )
            except Exception:
                idea_targets = None

            delegate_report = self.delegate_judge.judge_delegate(
                controller_output=controller_output,
                generator_output=generator_output,
                eim_state=eim_state,
                gt_window=gtw,
                idea_targets=idea_targets,
            )
            # CAPTURE DELEGATE JUDGE OUTPUT
            if artifact_tracker:
                artifact_tracker.capture_delegate_judge_output(delegate_report)
                capture_delegate_trace = getattr(artifact_tracker, "capture_delegate_judge_trace", None)
                if callable(capture_delegate_trace):
                    capture_delegate_trace(getattr(self.delegate_judge, "last_reconciliation_trace", None))
            
            # Record matched unit if callback provided
            if delegate_report and callable(record_match_fn):
                if getattr(delegate_report, 'matched_unit_id', None):
                    try:
                        record_match_fn(delegate_report.matched_unit_id)
                    except Exception:
                        pass

        patch_applied = False
        diff = None
        if self.recalibrator_drives_state and not self.disable_recalibration:
            # Extract delegate utterance for recalibrator context
            delegate_utt = None
            if generator_output is not None:
                if hasattr(generator_output, 'utterance'):
                    delegate_utt = getattr(generator_output, 'utterance', None)
                elif hasattr(generator_output, 'response'):
                    delegate_utt = getattr(generator_output, 'response', None)
                elif isinstance(generator_output, dict):
                    delegate_utt = generator_output.get('utterance') or generator_output.get('response')
                if delegate_utt:
                    delegate_utt = str(delegate_utt).strip() or None
            # New path: request both diff and snapshot_analysis
            
            # --- LATENCY FIX: Update tracking info locally for this step ---
            # The Recalibrator needs the state AS IF the current action has been processed.
            # We must reconcile the "pre-action" stats passed in arguments with the "post-action" findings in delegate_report.
            # Ensure integer types (guard against string values being passed)
            local_matched = int(matched_unit_count) if matched_unit_count else 0
            local_total = int(total_unit_count) if total_unit_count else 0
            local_cooldown = bool(is_in_cooldown)
            local_contribs = int(episode_contribution_count) if episode_contribution_count else 0

            if delegate_report:
                # Check if this step resulted in a match or contribution
                # We consider it a match if matched_unit_id is set (confirmed match)
                # OR if matched_utt_id is set (candidate match)
                has_match = bool(getattr(delegate_report, "matched_unit_id", None)) or bool(getattr(delegate_report, "matched_utt_id", None))
                
                if has_match:
                    local_matched += 1
                    local_contribs += 1
                    local_cooldown = True # Enter cooldown immediately after contribution

            # Check if delegate already covered the anchor's idea unit in an earlier delta
            # This enables nuanced silence reasoning (silence justified if already covered vs missed opportunity)
            delegate_covered_anchor_earlier = False
            anchor_turn = gtw[0] if gtw else (current_gt_turn or {})
            anchor_utt_id_for_check = anchor_turn.get("utterance_id")
            if episode and hasattr(episode, 'idea_units') and episode.idea_units and matched_idea_unit_ids:
                for iu in episode.idea_units:
                    # IdeaUnit can be dict or Pydantic object, so handle both
                    iu_anchor = iu.get("anchor_utt_id") if isinstance(iu, dict) else getattr(iu, 'anchor_utt_id', None)
                    if iu_anchor == anchor_utt_id_for_check:
                        iu_unit_id = iu.get("unit_id") if isinstance(iu, dict) else getattr(iu, 'unit_id', None)
                        if iu_unit_id and iu_unit_id in matched_idea_unit_ids:
                            delegate_covered_anchor_earlier = True
                        break

            fusion = self.recalibrator.build_fusion(
                eim_state=eim_state,
                last_utterances=last_utterances,
                current_gt_turn=(gtw[0] if gtw else (current_gt_turn or {})),
                env_error=env_report,
                delegate_error=delegate_report,
                delegate_utterance=delegate_utt,
                matched_unit_count=local_matched,
                total_unit_count=local_total,
                is_in_cooldown=local_cooldown,
                episode_contribution_count=local_contribs,
                participant_id=participant_id,
                delegate_covered_anchor_earlier=delegate_covered_anchor_earlier,
            )
            anchor_utt_id = None
            if gtw and len(gtw) > 0 and isinstance(gtw[0], dict):
                anchor_utt_id = gtw[0].get("utterance_id")
            if anchor_utt_id is None and current_gt_turn:
                anchor_utt_id = current_gt_turn.get("utterance_id")
            if fusion:
                diff = fusion.diff
                # Prefer explicit anchor from fusion if provided
                anchor_utt_id = fusion.anchor_utterance_id or anchor_utt_id
                # Apply patch and overwrite snapshot_analysis if provided
                patch_applied = apply_patch_fn(
                    diff if diff else EIMMasterState(topic_summary_log={}, participant_models={}),
                    anchor_utt_id,
                    new_snapshot_analysis=fusion.snapshot_analysis,
                    recompute_snapshot=(fusion.snapshot_analysis is None),
                )
                # Apply guidance (gated by guidance_enabled for ablation studies)
                if self.guidance_enabled and fusion.guidance and callable(apply_guidance_fn):
                    try:
                        apply_guidance_fn(fusion.guidance)
                    except Exception:
                        pass
                # Capture recalibrator extras
                if artifact_tracker:
                    try:
                        sa_payload = fusion.snapshot_analysis.model_dump() if hasattr(fusion.snapshot_analysis, 'model_dump') else fusion.snapshot_analysis
                        artifact_tracker.capture_recalibrator_snapshot_analysis(sa_payload)
                    except Exception:
                        pass
                    try:
                        gd_payload = fusion.guidance.model_dump() if (fusion.guidance is not None and hasattr(fusion.guidance, 'model_dump')) else fusion.guidance
                        artifact_tracker.capture_recalibrator_guidance(gd_payload)
                    except Exception:
                        pass
        else:
            # Ablation path: No recalibrator LLM call - use EIM's native snapshot analysis
            # This still updates topic_summary_log, participant_models, and snapshot_analysis
            # via EIM.analyze_snapshot (triggered by recompute_snapshot=True)
            anchor_utt_id = None
            if gtw and len(gtw) > 0 and isinstance(gtw[0], dict):
                anchor_utt_id = gtw[0].get("utterance_id")
            if anchor_utt_id is None and current_gt_turn:
                anchor_utt_id = current_gt_turn.get("utterance_id")
            
            # Apply empty diff but trigger snapshot recompute via EIM.
            # Use the module-level compatibility import above; importing via
            # "..schemas" here fails when the simulator is launched as a script.
            empty_diff = EIMMasterState(topic_summary_log={}, participant_models={})
            patch_applied = apply_patch_fn(
                empty_diff,
                anchor_utt_id,
                recompute_snapshot=True  # EIM.analyze_snapshot will handle all updates
            )
            diff = empty_diff  # Native EIM refresh succeeded; no guidance/fusion was produced.

        # CAPTURE RECALIBRATOR OUTPUT
        if artifact_tracker:
            artifact_tracker.capture_recalibrator_output(diff, patch_applied)
            artifact_tracker.finalize_step(capa_step_successful=bool(diff and patch_applied))
        
        return diff if patch_applied else None
