"""
Participant Simulation Module

This module provides real participant simulation with test cases, running the
complete meeting delegate system with actual meeting data and comparing outputs
against ground truth responses.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

from pydantic import ValidationError

# Add current directory to path for robust importing
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from config import Config
from llm_client import LLMClient
from llm_telemetry import make_telemetry_path
from transcript_processor import TranscriptProcessor
from user_profile_generator import UserProfileGenerator
from test_case_generator import TestCaseGenerator
from jit_briefing_generator import JITBriefingGenerator
from environment_interpreter import EnvironmentInterpretationModule
from shared_memory import SharedMemoryModule
from single_tier_agent import SingleTierAgent
from two_tier_agent import TwoTierUserModule
from raw_agents import MonoRawAgent, MonoRawReflectionAgent, MonoRawReflectionV2Agent, RawModularAgent, RawModularFairAgent
from raw_baseline_support import (
    RawTranscriptWindowResolver,
    build_delegate_capabilities,
    build_delegate_runtime_metadata,
    is_raw_baseline_variant,
    normalize_raw_context_n,
)
from pipeline_resume import (
    SIMULATION_MANIFEST_VERSION,
    SIMULATION_STAGE_VERSION,
    atomic_write_json,
    file_lock,
    fingerprint_contract,
    fingerprint_file,
    load_manifest,
    mark_unit_complete,
    mark_unit_failed,
    mark_unit_in_progress,
    save_manifest,
    summarize_manifest_state,
    unit_is_reusable,
)
from schemas import DelegateResponse, EIMMeetingStateVector, UserProfile, CAPAEpisode
from capa.episode_runner import run_capa_episode
from artifact_tracker import CapaArtifactTracker
from capa.shadow_eval import ShadowCapaEvaluator


def _episode_test_case_id(ep: CAPAEpisode) -> str:
    return ep.test_case_id or f"{ep.meeting_id}_{ep.represented_user}_t{ep.cue_index}"


def _prune_episode_progress(
    progress_metadata: Dict[str, Any],
    episodes: List[CAPAEpisode],
) -> Dict[str, Any]:
    current_ids = {_episode_test_case_id(ep) for ep in episodes}
    return {
        test_case_id: progress
        for test_case_id, progress in (progress_metadata or {}).items()
        if test_case_id in current_ids
    }


class ParticipantSimulation:
    """
    Real participant simulation with test cases and ground truth comparison.
    
    This class orchestrates the complete meeting delegate pipeline for a specific
    participant, running through their test cases and generating delegate responses
    that can be compared against ground truth.
    """
    
    def __init__(self, 
                 config_path: str = None,
                 use_two_tier: bool = False,
                 delegate_variant: str = "capa",
                 raw_context_N: int = 50,
                 mono_raw_context_N: Optional[int] = None,
                 late_window_k_override: Optional[int] = None,
                 transcript_path: Optional[str] = None):
        """
        Initialize the participant simulation system.
        
        Args:
            config_path: Path to the configuration file. If None, auto-detects location.
            use_two_tier: Whether to use two-tier agent (True) or single-tier (False)
            delegate_variant: CAPA runtime variant: capa | raw-modular | raw-modular-fair | mono-raw | mono-raw-reflection | mono-raw-reflection-v2
        """
        self.config = Config(config_path)  # Auto-detects if config_path is None
        self.llm_client = LLMClient(self.config)
        self.use_two_tier = use_two_tier
        self.delegate_variant = delegate_variant
        self.is_raw_variant = is_raw_baseline_variant(delegate_variant)
        self.raw_context_N = normalize_raw_context_n(raw_context_N, mono_raw_context_N)
        self.raw_window_resolver: Optional[RawTranscriptWindowResolver] = (
            RawTranscriptWindowResolver(self.raw_context_N, transcript_path) if self.is_raw_variant else None
        )
        self.transcript_path = self.raw_window_resolver.transcript_path if self.raw_window_resolver else None

        capa_cfg = getattr(self.config, 'capa', {}) or {}
        late_window_k = int(late_window_k_override if late_window_k_override is not None else capa_cfg.get('late_window_k', capa_cfg.get('k', 5)))
        self.late_window_k = late_window_k
        self.shadow_capa_evaluator: Optional[ShadowCapaEvaluator] = None

        if delegate_variant == "raw-modular-fair":
            self.agent = RawModularFairAgent(self.llm_client)
            self.agent_type = "Raw-Modular-Fair"
        elif delegate_variant == "raw-modular":
            self.agent = RawModularAgent(self.llm_client)
            self.agent_type = "Raw-Modular"
        elif delegate_variant == "mono-raw":
            self.agent = MonoRawAgent(self.llm_client)
            self.agent_type = "Mono-Raw"
        elif delegate_variant == "mono-raw-reflection":
            self.agent = MonoRawReflectionAgent(self.llm_client)
            self.agent_type = "Mono-Raw-Reflection"
        elif delegate_variant == "mono-raw-reflection-v2":
            self.agent = MonoRawReflectionV2Agent(self.llm_client)
            self.agent_type = "Mono-Raw-Reflection-V2"
        elif use_two_tier:
            self.agent = TwoTierUserModule(self.llm_client, late_window_k_override=late_window_k)
            self.agent_type = "Two-Tier"
        else:
            self.agent = SingleTierAgent(self.llm_client)
            self.agent_type = "Single-Tier"

        if self.is_raw_variant:
            self.shadow_capa_evaluator = ShadowCapaEvaluator(self.llm_client, late_window_k=late_window_k)

        print(f"Initialized ParticipantSimulation with {self.agent_type} Agent (variant={self.delegate_variant})")

    def get_delegate_capabilities(self) -> Dict[str, Any]:
        return build_delegate_capabilities(
            self.delegate_variant,
            no_recalibration=bool(getattr(self.agent, 'disable_recalibration', False)),
            use_two_tier=self.use_two_tier,
        )

    def get_delegate_runtime_metadata(self) -> Dict[str, Any]:
        return build_delegate_runtime_metadata(
            self.delegate_variant,
            self.raw_context_N,
            self.late_window_k,
        )

    def _get_raw_window_resolver(self) -> Optional[RawTranscriptWindowResolver]:
        if not is_raw_baseline_variant(getattr(self, "delegate_variant", None)):
            return None
        resolver = getattr(self, "raw_window_resolver", None)
        if resolver is None:
            resolver = RawTranscriptWindowResolver(getattr(self, "raw_context_N", 50), getattr(self, "transcript_path", None))
            self.raw_window_resolver = resolver
        return resolver

    @staticmethod
    def _extract_action_name(controller_output: Optional[Any]) -> Optional[str]:
        if controller_output is None:
            return None
        if hasattr(controller_output, 'action_details'):
            action_details = getattr(controller_output, 'action_details', None)
            if action_details is not None:
                return getattr(action_details, 'action_name', None)
        if isinstance(controller_output, dict):
            return ((controller_output.get('action_details') or {}).get('action_name'))
        return None

    @staticmethod
    def _extract_delegate_spoke(delegate_response: Optional[Any], action_name: Optional[str] = None) -> Optional[bool]:
        """Use the final utterance as the source of truth, with action fallback only if unavailable."""
        delegate_text: Optional[Any] = None
        if delegate_response is not None:
            if hasattr(delegate_response, 'response'):
                delegate_text = getattr(delegate_response, 'response', None)
            elif isinstance(delegate_response, dict):
                delegate_text = delegate_response.get('response')
                if delegate_text is None:
                    delegate_text = delegate_response.get('utterance')
        if delegate_text is not None:
            return bool(str(delegate_text).strip())
        if action_name:
            return action_name != "Remain_Silent"
        return None

    def _resolve_raw_prompt_window(
        self,
        episode: CAPAEpisode,
        delta: int,
        snapshot_window: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        resolver = self._get_raw_window_resolver()
        return resolver.resolve_window(episode, delta, snapshot_window) if resolver else list(snapshot_window)

    def _ensure_raw_context_compatible(self, episodes: List[CAPAEpisode]) -> None:
        resolver = self._get_raw_window_resolver()
        if resolver:
            resolver.ensure_episode_compatibility(episodes)

    def _resolve_prompt_snapshot_window(
        self,
        episode: CAPAEpisode,
        delta: int,
        snapshot_window: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return self._resolve_raw_prompt_window(episode, delta, snapshot_window) if self.is_raw_variant else list(snapshot_window)

    def _seed_shadow_state(
        self,
        episode: CAPAEpisode,
        eim_vec: Optional[EIMMeetingStateVector],
    ) -> Optional[EIMMeetingStateVector]:
        if not self.shadow_capa_evaluator:
            return None
        initial_raw_window = self._resolve_prompt_snapshot_window(episode, 0, list(episode.context_last_N))
        return self.shadow_capa_evaluator.seed_state(eim_vec, initial_raw_window)

    def _run_delegate_turn(
        self,
        *,
        smm: SharedMemoryModule,
        snapshot_window: List[Dict[str, Any]],
        prompt_snapshot_window: List[Dict[str, Any]],
        participant_id: str,
        gt_window: List[Dict[str, Any]],
        episode: CAPAEpisode,
        delta: int,
    ) -> Optional[DelegateResponse]:
        if self.is_raw_variant:
            return self.agent.run_turn(
                smm=smm,
                snapshot_utterances=snapshot_window,
                participant_id=participant_id,
                gt_next_window=gt_window,
                episode=episode,
                delta_index=delta,
                prompt_snapshot_utterances=prompt_snapshot_window,
            )
        return self.agent.run_turn(
            smm=smm,
            snapshot_utterances=snapshot_window,
            participant_id=participant_id,
            gt_next_window=gt_window,
            episode=episode,
            delta_index=delta,
        )

    def _artifact_tracking_supported(self) -> bool:
        return hasattr(self.agent, 'artifact_tracker')

    def _record_shadow_delegate_outcome(
        self,
        smm: SharedMemoryModule,
        delegate_report: Optional[Any],
        snapshot_window: List[Dict[str, Any]],
        delta_index: int,
    ) -> None:
        if delegate_report is None:
            return
        matched_unit_id = getattr(delegate_report, 'matched_unit_id', None)
        if not matched_unit_id or matched_unit_id in smm.matched_idea_units:
            return
        smm.record_idea_unit_match(matched_unit_id)
        current_step = len(snapshot_window) if snapshot_window else 0
        smm.record_delegate_contribution(current_step, delta_index=delta_index)
    
    def load_test_cases(self, test_cases_dir: str, participant_id: str, meeting_id: str) -> List[Dict[str, Any]]:
        """
        Load test cases for a specific participant and meeting.
        
        Args:
            test_cases_dir: Directory containing test case files
            participant_id: ID of the participant (e.g., "Project Manager")
            meeting_id: ID of the meeting (e.g., "Meeting31")
            
        Returns:
            List of test case dictionaries
        """
        test_case_filename = f"test_cases_for_{participant_id.replace(' ', '_')}_from_{meeting_id}.json"
        test_case_path = os.path.join(test_cases_dir, test_case_filename)
        
        if not os.path.exists(test_case_path):
            raise FileNotFoundError(f"Test case file not found at '{test_case_path}'")
        
        with open(test_case_path, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
        
        print(f"Loaded {len(test_cases)} test cases for {participant_id} from {meeting_id}")
        return test_cases
    
    def load_jit_briefing(self, briefings_dir: str, participant_id: str, meeting_id: str) -> List[str]:
        """
        Load JIT briefing for a specific participant and meeting.
        
        Args:
            briefings_dir: Directory containing JIT briefing files
            participant_id: ID of the participant
            meeting_id: ID of the meeting
            
        Returns:
            List of briefing points
        """
        briefing_filename = f"{participant_id.replace(' ', '_')}_briefing_from_{meeting_id}_partB.json"
        briefing_path = os.path.join(briefings_dir, briefing_filename)
        
        if not os.path.exists(briefing_path):
            print(f"Warning: JIT briefing file not found at '{briefing_path}'. Using empty briefing.")
            return []
        
        with open(briefing_path, 'r', encoding='utf-8') as f:
            briefing_data = json.load(f)
            jit_briefing_points = briefing_data.get("jit_briefing_points", [])
            evidence_rows = briefing_data.get("points_evidence") or []
            if isinstance(jit_briefing_points, list) and isinstance(evidence_rows, list) and evidence_rows:
                evidence_by_text = {
                    str(row.get("text") or "").strip(): row
                    for row in evidence_rows
                    if isinstance(row, dict) and str(row.get("text") or "").strip()
                }
                enriched_points = []
                for point in jit_briefing_points:
                    point_text = point if isinstance(point, str) else str(point or "")
                    evidence = evidence_by_text.get(point_text.strip()) or {}
                    if evidence:
                        enriched_points.append(
                            {
                                "text": point_text,
                                "anchor_phrases": list(evidence.get("anchor_phrases") or []),
                                "framing": evidence.get("framing"),
                            }
                        )
                    else:
                        enriched_points.append(point_text)
                jit_briefing_points = enriched_points
        
        print(f"Loaded {len(jit_briefing_points)} JIT briefing points for {participant_id}")
        return jit_briefing_points
    
    def load_eim_states(self, eim_outputs_dir: str, participant_id: str, meeting_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Load pre-generated EIM states for the participant and meeting.

        Args:
            eim_outputs_dir: Directory containing EIM output files
            participant_id: ID of the participant
            meeting_id: ID of the meeting
            
        Returns:
            List of EIM state dictionaries, or None if not found
        """
        eim_filename = f"eim_states_for_{participant_id.replace(' ', '_')}_in_{meeting_id}.json"
        eim_path = os.path.join(eim_outputs_dir, eim_filename)
        
        if not os.path.exists(eim_path):
            print(f"Warning: EIM states file not found at '{eim_path}'. Will generate states on-demand.")
            return None
        
        with open(eim_path, 'r', encoding='utf-8') as f:
            eim_states = json.load(f)

        print(f"Loaded {len(eim_states)} pre-generated EIM states for {participant_id}")
        return eim_states

    def _parse_user_profile_file(self, profile_path: Path) -> UserProfile:
        """Parse and validate a stored user profile JSON file."""
        with open(profile_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        profile_blob = payload.get('UserProfile', payload)
        try:
            return UserProfile(**profile_blob)
        except ValidationError as exc:
            raise ValueError(f"User profile at {profile_path} is not a valid UserProfile") from exc

    def load_capa_user_profile(self,
                               user_profiles_dir: str,
                               participant_id: str,
                               meeting_id: str) -> Tuple[UserProfile, Path]:
        """Locate and load a user profile for CAPA-mode simulations."""
        base_dir = Path(user_profiles_dir)
        if not base_dir.exists():
            raise FileNotFoundError(f"User profile directory not found: {base_dir}")

        participant_slug = participant_id.replace(' ', '_')
        candidates = [
            base_dir / f"capa_profile_for_{participant_slug}_in_{meeting_id}.json",
            base_dir / f"{participant_slug}_profile_from_{meeting_id}_partA.json",
        ]

        for candidate in candidates:
            if candidate.exists():
                profile = self._parse_user_profile_file(candidate)
                return profile, candidate

        # Fallback: glob for any matching profile artefact
        for candidate in base_dir.glob("*.json"):
            stem = candidate.stem
            if participant_slug in stem and meeting_id in stem and 'profile' in stem:
                profile = self._parse_user_profile_file(candidate)
                return profile, candidate

        raise FileNotFoundError(
            f"No user profile found for participant '{participant_id}' in meeting '{meeting_id}'"
        )

    def run_single_tier_simulation(self, 
                                 test_cases: List[Dict[str, Any]], 
                                 jit_briefing_points: List[str],
                                 meeting_id: str) -> List[Dict[str, Any]]:
        """
        Run simulation using the single-tier agent.
        
        Args:
            test_cases: List of test case dictionaries
            jit_briefing_points: JIT briefing points
            meeting_id: The meeting ID (should use the CLI parameter, not extract from test_case_id)
            
        Returns:
            List of simulation results
        """        
        return self.agent.run_simulation_for_participant(
            test_cases=test_cases,
            jit_briefing_points=jit_briefing_points,
            participant_id=test_cases[0]['principal_participant_id'],
            meeting_id=meeting_id
        )
    
    def run_two_tier_simulation(self, 
                              test_cases: List[Dict[str, Any]], 
                              jit_briefing_points: List[str],
                              eim_states: Optional[List[Dict[str, Any]]] = None,
                              artifact_tracker: Optional[CapaArtifactTracker] = None) -> List[Dict[str, Any]]:
        """
        Run simulation using the two-tier agent with SMM integration.
        
        Args:
            test_cases: List of test case dictionaries
            jit_briefing_points: JIT briefing points
            eim_states: Pre-generated EIM states (optional)
            
        Returns:
            List of simulation results
        """
        results = []
        
        # Initialize SMM with user profile
        smm = SharedMemoryModule()
        if test_cases:
            user_profile_dict = test_cases[0]['ground_truth_user_profile']
            # Convert dictionary to UserProfile object
            user_profile = UserProfile(**user_profile_dict)
            smm.load_user_profile(user_profile)
            smm.load_jit_briefing(jit_briefing_points)
        
        print(f"Starting two-tier simulation for {len(test_cases)} test cases...")
        
        # Create a mapping from test_case_id to EIM state for efficient lookup
        eim_state_lookup = {}
        if eim_states:
            for eim_entry in eim_states:
                test_case_id = eim_entry.get('test_case_id')
                if test_case_id:
                    eim_state_lookup[test_case_id] = eim_entry.get('eim_output')
        
        for i, test_case in enumerate(test_cases):
            test_case_id = test_case.get('test_case_id', 'N/A')
            print(f"\n--- Processing Test Case {i+1}/{len(test_cases)} ({test_case_id}) ---")
            
            try:
                # Get EIM state for this specific test case using test_case_id
                eim_output = eim_state_lookup.get(test_case_id)
                if eim_output:
                    # Convert dictionary to EIMMeetingStateVector object
                    eim_state_obj = EIMMeetingStateVector(**eim_output)
                    # Update SMM with EIM state
                    smm.update_working_memory(eim_state_obj)
                    print(f"  > Used pre-generated EIM state for {test_case_id}")
                else:
                    print(f"  > Warning: No EIM state available for test case {test_case_id}")
                    # For now, we'll skip cases without EIM states
                    results.append({
                        "test_case": test_case,
                        "delegate_output": {"error": f"No EIM state available for {test_case_id}"}
                    })
                    continue
                
                # Run two-tier agent
                snapshot_utterances = test_case['input_snapshot_utterances']
                participant_id = test_case['principal_participant_id']
                
                # Set up artifact tracker for this test case if provided
                if artifact_tracker:
                    # Initialize artifact tracker with test case context
                    artifact_tracker.test_case_id = test_case_id
                    artifact_tracker.participant_id = participant_id
                    artifact_tracker.meeting_id = test_case.get('meeting_id', 'unknown')
                    artifact_tracker.anchor_utt_id = None  # No anchor for legacy simulation
                    artifact_tracker.delta = None  # No delta for legacy simulation
                    
                    # Set agent's artifact tracker
                    self.agent.artifact_tracker = artifact_tracker
                
                delegate_response = self.agent.run_turn(smm, snapshot_utterances, participant_id)
                
                # Clear artifact tracker after use
                if artifact_tracker:
                    self.agent.artifact_tracker = None
                
                delegate_output_dict = delegate_response.model_dump() if delegate_response else {"error": "Failed to generate valid response."}
                
                result_entry = {
                    "test_case": test_case,
                    "delegate_output": delegate_output_dict,
                    "smm_state_summary": smm.get_memory_summary()
                }
                results.append(result_entry)
                
                if delegate_response:
                    print(f"  > Delegate Thoughts: {delegate_response.thoughts[:200]}...")
                    print(f"  > Delegate Speak:   '{delegate_response.response}'")
                    print(f"  > Ground Truth:     '{test_case['ground_truth_response_text']}'")
                else:
                    print("  > Delegate failed to produce a valid output.")
                
            except Exception as e:
                print(f"  > Error processing test case: {e}")
                results.append({
                    "test_case": test_case,
                    "delegate_output": {"error": str(e)}
                })
        
        return results
    
    def run_participant_simulation(self,
                                 participant_id: str,
                                 meeting_id: str,
                                 test_cases_dir: str = "benchmark_test_cases",
                                 briefings_dir: str = "jit_briefings",
                                 eim_outputs_dir: str = "eim_outputs",
                                 output_dir: str = "simulation_outputs",
                                 enable_artifact_tracking: bool = False) -> Dict[str, Any]:
        """
        Run complete simulation for a specific participant.
        
        Args:
            participant_id: ID of the participant (e.g., "Project Manager")
            meeting_id: ID of the meeting (e.g., "Meeting31")
            test_cases_dir: Directory containing test cases
            briefings_dir: Directory containing JIT briefings
            eim_outputs_dir: Directory containing EIM states
            output_dir: Directory to save simulation results
            
        Returns:
            Dictionary with simulation results and metadata
        """
        print(f"=" * 60)
        print(f"RUNNING PARTICIPANT SIMULATION")
        print(f"Participant: {participant_id}")
        print(f"Meeting: {meeting_id}")
        print(f"Agent Type: {self.agent_type}")
        print(f"=" * 60)
        
        try:
            # Initialize artifact tracker if requested
            artifact_tracker = None
            if enable_artifact_tracking and self._artifact_tracking_supported():
                artifact_tracker = CapaArtifactTracker(participant_id=participant_id, meeting_id=meeting_id)
                print("[OK] Artifact tracking enabled")
            
            # Load all required data
            test_cases = self.load_test_cases(test_cases_dir, participant_id, meeting_id)
            jit_briefing_points = self.load_jit_briefing(briefings_dir, participant_id, meeting_id)
            eim_states = self.load_eim_states(eim_outputs_dir, participant_id, meeting_id) if self.use_two_tier else None
            
            # Run appropriate simulation
            if self.use_two_tier:
                results = self.run_two_tier_simulation(test_cases, jit_briefing_points, eim_states, artifact_tracker)
            else:
                results = self.run_single_tier_simulation(test_cases, jit_briefing_points, meeting_id)
            
            # Prepare output
            simulation_output = {
                "metadata": {
                    "participant_id": participant_id,
                    "meeting_id": meeting_id,
                    "agent_type": self.agent_type,
                    "total_test_cases": len(test_cases),
                    "successful_cases": len([r for r in results if "error" not in r["delegate_output"]]),
                    "jit_briefing_points_count": len(jit_briefing_points),
                    "artifact_tracking_enabled": artifact_tracker is not None,
                    **self.get_delegate_capabilities()
                },
                "results": results
            }
            
            # Add artifact tracking data if available
            if artifact_tracker:
                # Save artifacts to separate file in artifacts subdirectory
                artifact_file_path = artifact_tracker.save_artifacts(Path(output_dir))
                # Add reference to artifacts file in main simulation output
                simulation_output["artifacts_reference"] = artifact_tracker.get_artifact_file_reference(artifact_file_path)
                print(f"[ARTIFACTS] Captured {len(artifact_tracker.step_artifacts)} step artifacts")
            
            # Save results
            os.makedirs(output_dir, exist_ok=True)
            agent_suffix = "two_tier" if self.use_two_tier else "single_tier"
            output_filename = f"simulation_{participant_id.replace(' ', '_')}_{meeting_id}_{agent_suffix}.json"
            output_path = os.path.join(output_dir, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(simulation_output, f, indent=2, ensure_ascii=False)
            
            print(f"\n{'='*60}")
            print(f"SIMULATION COMPLETE")
            print(f"Results saved to: {output_path}")
            print(f"Success rate: {simulation_output['metadata']['successful_cases']}/{simulation_output['metadata']['total_test_cases']}")
            print(f"{'='*60}")
            
            return simulation_output
            
        except Exception as e:
            print(f"[ERROR] Simulation failed: {e}")
            raise


    # ===== CAPA (Unified Episodes) =====
    def load_capa_episodes(self, episodes_dir: str, participant_id: str, meeting_id: str) -> List[CAPAEpisode]:
        """Load unified CAPA episodes for a participant/meeting."""
        path = os.path.join(episodes_dir, f"capa_episodes_for_{participant_id.replace(' ', '_')}_from_{meeting_id}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"CAPA episodes file not found at '{path}'")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        episodes: List[CAPAEpisode] = []
        for item in data:
            try:
                episodes.append(CAPAEpisode(**item))
            except Exception:
                continue
        print(f"Loaded {len(episodes)} CAPA episodes for {participant_id} from {meeting_id}")
        return episodes

    def load_capa_eim_states(self, eim_dir: str, participant_id: str, meeting_id: str) -> Dict[str, Dict[str, Any]]:
        """Load precomputed CAPA EIM states (mapping test_case_id -> eim_output dict)."""
        path = os.path.join(eim_dir, f"capa_eim_states_for_{participant_id.replace(' ', '_')}_in_{meeting_id}.json")
        if not os.path.exists(path):
            print(f"Warning: CAPA EIM states not found at '{path}'. Episodes will compute EIM_t on the fly.")
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            entries = json.load(f)
        mapping: Dict[str, Dict[str, Any]] = {}
        for entry in entries:
            tcid = entry.get('test_case_id')
            eim_output = entry.get('eim_output')
            if tcid and eim_output:
                mapping[tcid] = eim_output
        print(f"Loaded {len(mapping)} CAPA EIM states for {participant_id}")
        return mapping

    def _simulation_manifest_path(self, output_dir: str) -> Path:
        return Path(output_dir) / "simulation_manifest.json"

    def _simulation_lock_path(self, output_dir: str) -> Path:
        return Path(output_dir) / ".simulation.lock"

    def _participant_unit_key(self, participant_id: str) -> str:
        return f"participant::{participant_id}"

    def _participant_simulation_paths(self, output_dir: str, participant_id: str, meeting_id: str) -> Dict[str, Path]:
        slug = participant_id.replace(' ', '_')
        base = Path(output_dir)
        return {
            "simulation": base / f"capa_sim_{slug}_{meeting_id}.json",
            "artifacts": base / "artifacts" / f"capa_artifacts_{participant_id}_{meeting_id}.json",
        }

    def _participant_dependency_paths(
        self,
        participant_id: str,
        meeting_id: str,
        episodes_dir: str,
        eim_states_dir: str,
        jit_briefings_dir: str,
        user_profiles_dir: str,
    ) -> Dict[str, Path]:
        slug = participant_id.replace(' ', '_')
        return {
            "episodes": Path(episodes_dir) / f"capa_episodes_for_{slug}_from_{meeting_id}.json",
            "eim_states": Path(eim_states_dir) / f"capa_eim_states_for_{slug}_in_{meeting_id}.json",
            "briefing": Path(jit_briefings_dir) / f"{slug}_briefing_from_{meeting_id}_partB.json",
            "profile": Path(user_profiles_dir) / f"capa_profile_for_{slug}_in_{meeting_id}.json",
        }

    def _participant_checkpoint_paths(self, output_dir: str, participant_id: str, test_case_id: str) -> Dict[str, Path]:
        slug = participant_id.replace(' ', '_')
        safe_case = test_case_id.replace('/', '_')
        base = Path(output_dir) / "_checkpoints" / slug
        return {
            "steps": base / "episodes" / f"{safe_case}.json",
            "artifacts": base / "artifacts" / f"{safe_case}.json",
        }

    def _build_simulation_contract(
        self,
        participant_id: str,
        meeting_id: str,
        dependency_paths: Dict[str, Path],
        enable_artifact_tracking: bool,
    ) -> Dict[str, Any]:
        dependency_fingerprints = {name: fingerprint_file(path) for name, path in dependency_paths.items()}
        if self.is_raw_variant and self.transcript_path:
            dependency_fingerprints["transcript"] = fingerprint_file(Path(self.transcript_path))
        contract_payload = {
            "stage": "simulation",
            "stage_version": SIMULATION_STAGE_VERSION,
            "participant_id": participant_id,
            "meeting_id": meeting_id,
            "delegate_variant": self.delegate_variant,
            "use_two_tier": self.use_two_tier,
            "enable_artifact_tracking": enable_artifact_tracking,
            "model_name": getattr(self.config, 'active_model_name', None),
            "provider": getattr(self.config, 'provider', None),
            "model": getattr(self.config, 'model', None),
            "late_window_k": self.late_window_k,
            "raw_context_N": self.raw_context_N if self.is_raw_variant else None,
            **self.get_delegate_capabilities(),
            **self.get_delegate_runtime_metadata(),
            "dependency_fingerprints": dependency_fingerprints,
        }
        return {
            "payload": contract_payload,
            "fingerprint": fingerprint_contract(**contract_payload),
            "dependency_fingerprints": dependency_fingerprints,
        }

    def _build_episode_contract(self, participant_contract_fingerprint: str, episode: CAPAEpisode) -> str:
        return fingerprint_contract(
            stage="simulation_episode",
            stage_version=SIMULATION_STAGE_VERSION,
            participant_contract_fingerprint=participant_contract_fingerprint,
            episode=episode.model_dump(),
        )

    @staticmethod
    def _load_json_file(path: Path) -> Dict[str, Any]:
        with open(path, 'r', encoding='utf-8') as handle:
            return json.load(handle)

    def _load_existing_simulation_output(self, path: Path) -> Dict[str, Any]:
        return self._load_json_file(path)

    def _validate_episode_k_consistency(self, episodes: List[CAPAEpisode]) -> None:
        expected_k = int(self.late_window_k)
        mismatched = sorted({
            int(ep.k)
            for ep in episodes
            if getattr(ep, "k", None) is not None and int(ep.k) != expected_k
        })
        if mismatched:
            raise ValueError(
                f"Episode/runtime k mismatch: runtime late_window_k={expected_k}, "
                f"episode k values={mismatched}. Regenerate prep or pass matching --capa-k/--late-window-k."
            )

    def run_capa_simulation(self,
                             participant_id: str,
                             meeting_id: str,
                             episodes_dir: str = "capa_episodes",
                             eim_states_dir: str = "capa_eim_states",
                             jit_briefings_dir: str = "jit_briefings",
                             output_dir: str = "capa_sim_outputs",
                             user_profiles_dir: str = "user_profiles",
                             enable_artifact_tracking: bool = False,
                             resume: bool = False,
                             force: bool = False) -> Dict[str, Any]:
        """Run CAPA evaluation simulation over unified episodes for a participant.
        Each episode is processed with a delta-loop over k steps with strict anchoring."""
        os.makedirs(output_dir, exist_ok=True)
        output_paths = self._participant_simulation_paths(output_dir, participant_id, meeting_id)
        manifest_path = self._simulation_manifest_path(output_dir)
        dependency_paths = self._participant_dependency_paths(
            participant_id=participant_id,
            meeting_id=meeting_id,
            episodes_dir=episodes_dir,
            eim_states_dir=eim_states_dir,
            jit_briefings_dir=jit_briefings_dir,
            user_profiles_dir=user_profiles_dir,
        )
        contract = self._build_simulation_contract(
            participant_id=participant_id,
            meeting_id=meeting_id,
            dependency_paths=dependency_paths,
            enable_artifact_tracking=enable_artifact_tracking,
        )
        participant_unit_key = self._participant_unit_key(participant_id)

        with file_lock(self._simulation_lock_path(output_dir)):
            manifest = load_manifest(manifest_path, default_version=SIMULATION_MANIFEST_VERSION)
            existing_unit = (manifest.get('units') or {}).get(participant_unit_key)
            reusable_outputs = [output_paths['simulation']]
            if enable_artifact_tracking:
                reusable_outputs.append(output_paths['artifacts'])
            if resume and not force and unit_is_reusable(existing_unit, contract['fingerprint'], reusable_outputs):
                return self._load_existing_simulation_output(output_paths['simulation'])

            existing_progress = (existing_unit or {}).get('metadata', {}).get('episode_progress', {}) or {}
            if (existing_unit or {}).get('contract_fingerprint') == contract['fingerprint']:
                progress_metadata = dict(existing_progress)
            else:
                progress_metadata = {}
            unit_metadata = {
                "participant_id": participant_id,
                "meeting_id": meeting_id,
                "dependency_fingerprints": contract['dependency_fingerprints'],
                "episode_progress": progress_metadata,
            }
            mark_unit_in_progress(manifest, participant_unit_key, contract['fingerprint'], unit_metadata)
            save_manifest(manifest_path, manifest)

            current_test_case_id: Optional[str] = None
            current_episode_contract: Optional[str] = None
            per_episode_results: List[Dict[str, Any]] = []
            aggregated_step_artifacts: List[Dict[str, Any]] = []

            try:
                episodes = self.load_capa_episodes(episodes_dir, participant_id, meeting_id)
                self._validate_episode_k_consistency(episodes)
                pruned_progress = _prune_episode_progress(progress_metadata, episodes)
                if len(pruned_progress) != len(progress_metadata):
                    progress_metadata = pruned_progress
                    manifest['units'][participant_unit_key]['metadata']['episode_progress'] = progress_metadata
                    save_manifest(manifest_path, manifest)
                eim_map = self.load_capa_eim_states(eim_states_dir, participant_id, meeting_id)

                if self.is_raw_variant:
                    self._ensure_raw_context_compatible(episodes)

                user_profile, profile_path = self.load_capa_user_profile(
                    user_profiles_dir=user_profiles_dir,
                    participant_id=participant_id,
                    meeting_id=meeting_id,
                )
                print(f"Loaded user profile for {participant_id} from {profile_path}")

                jit_briefing_points = self.load_jit_briefing(
                    briefings_dir=jit_briefings_dir,
                    participant_id=participant_id,
                    meeting_id=meeting_id
                )

                if self.is_raw_variant:
                    print("[RAW] Shadow eval mode: delegate judge only; predictor/env judge/recalibrator disabled.")

                smm = SharedMemoryModule()
                smm.load_user_profile(user_profile)
                smm.load_jit_briefing(jit_briefing_points)
                if self.delegate_variant == "capa" and not self.use_two_tier:
                    print("Warning: CAPA runner expects Two-Tier agent; continuing anyway.")

                artifact_tracking_enabled = enable_artifact_tracking and self._artifact_tracking_supported()

                for ep in episodes:
                    tcid = _episode_test_case_id(ep)
                    current_test_case_id = tcid
                    checkpoint_paths = self._participant_checkpoint_paths(output_dir, participant_id, tcid)
                    episode_contract = self._build_episode_contract(contract['fingerprint'], ep)
                    current_episode_contract = episode_contract
                    progress_entry = progress_metadata.get(tcid)
                    episode_output_paths = [checkpoint_paths['steps'], checkpoint_paths['artifacts']]

                    if resume and not force and unit_is_reusable(progress_entry, episode_contract, episode_output_paths):
                        step_payload = self._load_json_file(checkpoint_paths['steps'])
                        artifact_payload = self._load_json_file(checkpoint_paths['artifacts'])
                        per_episode_results.append({
                            "test_case_id": tcid,
                            "steps": step_payload.get('steps', []),
                        })
                        aggregated_step_artifacts.extend(artifact_payload.get('step_artifacts', []))
                        continue

                    progress_metadata[tcid] = {
                        "state": "in_progress",
                        "contract_fingerprint": episode_contract,
                        "output_fingerprints": {},
                    }
                    manifest['units'][participant_unit_key]['metadata']['episode_progress'] = progress_metadata
                    save_manifest(manifest_path, manifest)

                    try:
                        eim_output = eim_map.get(tcid)
                        eim_vec = EIMMeetingStateVector(**eim_output) if eim_output else None
                        smm.clear_working_memory()
                        episode_tracker = CapaArtifactTracker(participant_id=participant_id, meeting_id=meeting_id) if artifact_tracking_enabled else None
                        steps = self._run_capa_episode_with_tracking(
                            episode=ep,
                            smm=smm,
                            participant_id=participant_id,
                            meeting_id=meeting_id,
                            eim_vec=eim_vec,
                            artifact_tracker=episode_tracker,
                        )
                        step_artifacts = list(episode_tracker.step_artifacts) if episode_tracker else []
                        atomic_write_json(checkpoint_paths['steps'], {
                            "participant_id": participant_id,
                            "meeting_id": meeting_id,
                            "test_case_id": tcid,
                            "steps": steps,
                        })
                        atomic_write_json(checkpoint_paths['artifacts'], {
                            "participant_id": participant_id,
                            "meeting_id": meeting_id,
                            "test_case_id": tcid,
                            "step_artifacts": step_artifacts,
                        })
                        progress_metadata[tcid] = {
                            "state": "complete",
                            "contract_fingerprint": episode_contract,
                            "output_fingerprints": {
                                str(checkpoint_paths['steps']): fingerprint_file(checkpoint_paths['steps']),
                                str(checkpoint_paths['artifacts']): fingerprint_file(checkpoint_paths['artifacts']),
                            },
                        }
                        manifest['units'][participant_unit_key]['metadata']['episode_progress'] = progress_metadata
                        save_manifest(manifest_path, manifest)
                        per_episode_results.append({
                            "test_case_id": tcid,
                            "steps": steps,
                        })
                        aggregated_step_artifacts.extend(step_artifacts)
                    except Exception as exc:
                        progress_metadata[tcid] = {
                            "state": "failed",
                            "contract_fingerprint": episode_contract,
                            "output_fingerprints": {},
                            "error_type": exc.__class__.__name__,
                            "error_message": str(exc),
                        }
                        manifest['units'][participant_unit_key]['metadata']['episode_progress'] = progress_metadata
                        save_manifest(manifest_path, manifest)
                        raise

                result = {
                    "participant_id": participant_id,
                    "meeting_id": meeting_id,
                    "delegate_variant": self.delegate_variant,
                    "capabilities": self.get_delegate_capabilities(),
                    **self.get_delegate_runtime_metadata(),
                    "total_episodes": len(episodes),
                    "results": per_episode_results,
                }

                if artifact_tracking_enabled:
                    aggregate_tracker = CapaArtifactTracker(participant_id=participant_id, meeting_id=meeting_id)
                    aggregate_tracker.step_artifacts = aggregated_step_artifacts
                    artifact_file_path = aggregate_tracker.save_artifacts(Path(output_dir))
                    result["artifacts_reference"] = aggregate_tracker.get_artifact_file_reference(artifact_file_path)
                    print(f"[ARTIFACTS] Captured {len(aggregated_step_artifacts)} CAPA step artifacts")
                elif output_paths['artifacts'].exists():
                    output_paths['artifacts'].unlink(missing_ok=True)

                atomic_write_json(output_paths['simulation'], result)
                complete_paths = [output_paths['simulation']]
                if artifact_tracking_enabled:
                    complete_paths.append(output_paths['artifacts'])
                mark_unit_complete(
                    manifest,
                    participant_unit_key,
                    contract['fingerprint'],
                    complete_paths,
                    metadata={
                        "participant_id": participant_id,
                        "meeting_id": meeting_id,
                        "dependency_fingerprints": contract['dependency_fingerprints'],
                        "episode_progress": progress_metadata,
                        "total_episodes": len(episodes),
                    },
                )
                manifest['state'] = summarize_manifest_state(manifest)
                save_manifest(manifest_path, manifest)
                print(f"CAPA simulation complete: {output_paths['simulation']}")
                return result
            except Exception as exc:
                mark_unit_failed(
                    manifest,
                    participant_unit_key,
                    contract['fingerprint'],
                    metadata={
                        "participant_id": participant_id,
                        "meeting_id": meeting_id,
                        "dependency_fingerprints": contract['dependency_fingerprints'],
                        "episode_progress": progress_metadata,
                        "failed_test_case_id": current_test_case_id,
                        "failed_episode_contract": current_episode_contract,
                        "error_type": exc.__class__.__name__,
                        "error_message": str(exc),
                    },
                )
                manifest['state'] = summarize_manifest_state(manifest)
                save_manifest(manifest_path, manifest)
                raise

    def _run_capa_episode_with_tracking(self,
                                      episode: CAPAEpisode,
                                      smm: SharedMemoryModule,
                                      participant_id: str,
                                      meeting_id: str,
                                      eim_vec: Optional[EIMMeetingStateVector] = None,
                                      artifact_tracker: Optional[CapaArtifactTracker] = None) -> List[Dict[str, Any]]:
        """
        Run a CAPA episode with artifact tracking support.
        This is a modified version of run_capa_episode that integrates artifact tracking.
        """
        records: List[Dict[str, Any]] = []

        if artifact_tracker:
            artifact_tracker.participant_id = participant_id
            artifact_tracker.meeting_id = meeting_id
            self.agent.artifact_tracker = artifact_tracker

        telemetry_test_case_id = _episode_test_case_id(episode)
        self.llm_client.set_telemetry_context(
            meeting_id=meeting_id,
            participant_id=participant_id,
            test_case_id=telemetry_test_case_id,
            episode_k=episode.k,
            episode_N=episode.N,
        )

        shadow_state: Optional[EIMMeetingStateVector] = None
        if self.delegate_variant == "capa":
            if eim_vec is not None:
                smm.update_working_memory(eim_vec)
                try:
                    smm.mark_wm_seeded_at_cue()
                except Exception:
                    pass
            else:
                _eim = EnvironmentInterpretationModule(self.agent.client)
                if smm.get_working_memory() is None:
                    vec = _eim.analyze_snapshot(episode.context_last_N)
                    if vec is not None:
                        smm.update_working_memory(vec)
                try:
                    smm.mark_wm_seeded_at_cue()
                except Exception:
                    pass
        else:
            smm.clear_working_memory()
            shadow_state = self._seed_shadow_state(episode, eim_vec)

        snapshot_window: List[Dict[str, Any]] = list(episode.context_last_N)

        for delta in range(episode.k):
            gt_window = episode.gt_next_window[delta:]
            if not gt_window:
                break

            decision_started = time.perf_counter()
            self.llm_client.set_telemetry_context(
                delta=delta,
                anchor_utt_id=gt_window[0].get("utterance_id"),
                decision_id=f"{participant_id}::{telemetry_test_case_id}::delta_{delta}",
            )

            if artifact_tracker:
                tcid = episode.test_case_id or f"{episode.meeting_id}_{episode.represented_user}_t{episode.cue_index}"
                artifact_tracker.test_case_id = tcid
                artifact_tracker.delta = delta
                artifact_tracker.anchor_utt_id = gt_window[0].get("utterance_id")

            if self.is_raw_variant:
                for attr_name in (
                    'last_controller_output',
                    'last_generator_output',
                    'last_delegate_judge_payload',
                    'last_helper_context',
                    'last_helper_digest_data',
                    'last_raw_context',
                    'last_mono_raw_prompt_context',
                    'last_reflection_payload',
                    'last_reflection_output',
                    'last_reflection_memory_before',
                    'last_reflection_memory_after',
                ):
                    if hasattr(self.agent, attr_name):
                        setattr(self.agent, attr_name, None)

            prompt_snapshot_window = self._resolve_prompt_snapshot_window(episode, delta, snapshot_window)
            delegate_response = self._run_delegate_turn(
                smm=smm,
                snapshot_window=snapshot_window,
                prompt_snapshot_window=prompt_snapshot_window,
                participant_id=participant_id,
                gt_window=gt_window,
                episode=episode,
                delta=delta,
            )

            observed_turn = gt_window[0]
            if hasattr(self.agent, "update_reflection_after_observed_turn"):
                self.agent.update_reflection_after_observed_turn(
                    smm=smm,
                    participant_id=participant_id,
                    episode=episode,
                    delta_index=delta,
                    prompt_snapshot_utterances=prompt_snapshot_window,
                    observed_turn=observed_turn,
                    delegate_response=delegate_response,
                )

            updated_snapshot_window = list(snapshot_window) + [observed_turn]
            if len(updated_snapshot_window) > episode.N:
                updated_snapshot_window = updated_snapshot_window[-episode.N:]
            prompt_updated_snapshot_window = (
                self._resolve_raw_prompt_window(episode, delta + 1, updated_snapshot_window)
                if self.is_raw_variant
                else list(updated_snapshot_window)
            )

            capa_step_successful = None
            if self.is_raw_variant and self.shadow_capa_evaluator:
                shadow_result = self.shadow_capa_evaluator.step(
                    eim_state=shadow_state,
                    last_utterances=prompt_snapshot_window,
                    gt_window=gt_window,
                    controller_output=getattr(self.agent, 'last_controller_output', None),
                    generator_output=getattr(self.agent, 'last_generator_output', None),
                    delegate_payload=getattr(self.agent, 'last_delegate_judge_payload', None),
                    updated_snapshot_window=prompt_updated_snapshot_window,
                    episode=episode,
                    artifact_tracker=artifact_tracker,
                )
                shadow_state = shadow_result.next_state
                capa_step_successful = shadow_result.capa_step_successful
                self._record_shadow_delegate_outcome(
                    smm=smm,
                    delegate_report=shadow_result.delegate_report,
                    snapshot_window=snapshot_window,
                    delta_index=delta,
                )

            if artifact_tracker:
                artifact_tracker.finalize_step(capa_step_successful=capa_step_successful)

            self.llm_client.record_telemetry_span(
                "decision_end_to_end",
                time.perf_counter() - decision_started,
                phase="decision",
                extra={
                    "capa_step_successful": capa_step_successful,
                    "delegate_variant": self.delegate_variant,
                },
            )

            snapshot_window = updated_snapshot_window
            action_name = self._extract_action_name(getattr(self.agent, 'last_controller_output', None))
            spoke = self._extract_delegate_spoke(delegate_response, action_name)
            records.append({
                "delta": delta,
                "anchor_utt_id": observed_turn.get("utterance_id"),
                "action_name": action_name,
                "spoke": spoke,
                "delegate_response": delegate_response.model_dump() if delegate_response else None,
            })

        if artifact_tracker:
            self.agent.artifact_tracker = None

        self.llm_client.clear_telemetry_context(
            "test_case_id",
            "episode_k",
            "episode_N",
            "delta",
            "anchor_utt_id",
            "decision_id",
        )

        return records

def run_participant_simulation_cli():
    """
    Command-line interface for running participant simulations.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Run participant simulation for Meeting Delegate system")
    parser.add_argument("--participant", required=True, help="Participant ID (e.g., 'Project Manager')")
    parser.add_argument("--meeting", required=True, help="Meeting ID (e.g., 'Meeting31')")
    parser.add_argument("--agent-type", choices=["single", "two"], default="single", help="Agent type to use")
    parser.add_argument("--mode", choices=["legacy", "capa"], default="capa", help="Simulation mode")
    parser.add_argument("--test-cases-dir", default="benchmark_test_cases", help="Legacy test cases directory")
    parser.add_argument("--briefings-dir", default="jit_briefings", help="JIT briefings directory")
    parser.add_argument("--eim-dir", default="eim_outputs", help="Legacy EIM outputs directory")
    parser.add_argument("--capa-episodes-dir", default="capa_episodes", help="CAPA episodes directory")
    parser.add_argument("--capa-eim-dir", default="capa_eim_states", help="CAPA EIM states directory")
    parser.add_argument("--output-dir", default="simulation_outputs", help="Output directory")
    parser.add_argument("--user-profiles-dir", default="user_profiles", help="User profiles directory")
    parser.add_argument("--enable-artifact-tracking", action="store_true", help="Enable CAPA artifact tracking")
    parser.add_argument("--delegate-variant", choices=["capa", "raw-modular", "raw-modular-fair", "mono-raw", "mono-raw-reflection", "mono-raw-reflection-v2"], default="capa", help="CAPA delegate runtime variant")
    parser.add_argument("--no-recalibration", action="store_true", help="Disable recalibrator (ablation)")
    parser.add_argument("--raw-context-N", type=int, default=50, help="Raw-baseline transcript window size")
    parser.add_argument("--mono-raw-context-N", type=int, default=None, help="Deprecated alias for --raw-context-N")
    parser.add_argument("--transcript-path", default=None, help="Original meeting transcript JSON for raw-baseline longer-context reconstruction")
    parser.add_argument("--capa-k", "--late-window-k", dest="late_window_k", type=int, default=None, help="Override runtime late window k for CAPA judges/recalibrator and raw shadow eval")
    parser.add_argument("--model", default=None, help="Optional active model override; must exist in config_extended.json")
    parser.add_argument("--resume", action="store_true", help="Resume from simulation_manifest/checkpoints when compatible")
    parser.add_argument("--force", action="store_true", help="Ignore simulation checkpoints and recompute participant outputs")
    parser.add_argument("--disable-llm-telemetry", action="store_true", help="Do not write per-call token/latency JSONL logs")
    
    args = parser.parse_args()

    if args.model:
        os.environ["MD_ACTIVE_MODEL"] = args.model

    args.raw_context_N = normalize_raw_context_n(args.raw_context_N, args.mono_raw_context_N)
    
    # Create simulation instance
    simulation = ParticipantSimulation(
        use_two_tier=(args.agent_type == "two"),
        delegate_variant=args.delegate_variant,
        raw_context_N=args.raw_context_N,
        late_window_k_override=args.late_window_k,
        transcript_path=args.transcript_path,
    )
    if not args.disable_llm_telemetry:
        telemetry_path = make_telemetry_path(
            Path(args.output_dir),
            stage="simulation",
            participant_id=args.participant,
        )
        simulation.llm_client.configure_telemetry(
            telemetry_path,
            base_context={
                "stage": "simulation",
                "meeting_id": args.meeting,
                "participant_id": args.participant,
                "delegate_variant": args.delegate_variant,
            },
        )
        print(f"[LLM TELEMETRY] Writing {telemetry_path}")
    
    if args.mode == "legacy":
        results = simulation.run_participant_simulation(
            participant_id=args.participant,
            meeting_id=args.meeting,
            test_cases_dir=args.test_cases_dir,
            briefings_dir=args.briefings_dir,
            eim_outputs_dir=args.eim_dir,
            output_dir=args.output_dir,
            enable_artifact_tracking=args.enable_artifact_tracking
        )
    else:
        # CAPA δ-loop simulation over unified episodes
        # Apply ablation flags if requested
        if args.no_recalibration:
            if args.delegate_variant == "capa" and hasattr(simulation.agent, 'disable_recalibration'):
                simulation.agent.disable_recalibration = True
                if hasattr(simulation.agent, '_capa_driver') and simulation.agent._capa_driver:
                    simulation.agent._capa_driver.disable_recalibration = True
                print("[WARN] Recalibration DISABLED (ablation mode)")
            else:
                print("[WARN] --no-recalibration is ignored for raw delegate variants")
        
        results = simulation.run_capa_simulation(
            participant_id=args.participant,
            meeting_id=args.meeting,
            episodes_dir=args.capa_episodes_dir,
            eim_states_dir=args.capa_eim_dir,
            jit_briefings_dir=args.briefings_dir,
            output_dir=args.output_dir if args.output_dir else "capa_sim_outputs",
            user_profiles_dir=args.user_profiles_dir,
            enable_artifact_tracking=args.enable_artifact_tracking,
            resume=args.resume,
            force=args.force,
        )
    
    print("[DONE] Simulation completed successfully!")


if __name__ == "__main__":
    run_participant_simulation_cli()
