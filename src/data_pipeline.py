"""
Data Preparation Pipeline

This module orchestrates the complete data preparation workflow from raw
transcripts to simulation-ready data. It coordinates transcript processing,
profile generation, test case creation, and JIT briefing generation.
"""

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

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
from capa.episode_generator import generate_dense_episodes, generate_dense_episodes_multi_participant, write_episodes_to_file
from capa.episode_precompute import precompute_eim_for_episodes, write_eim_states_to_file
from capa.episode_utils import is_banal
from schemas import CAPAEpisode
from prep_resume import run_capa_prep
from pipeline_resume import (
    PREP_MANIFEST_VERSION,
    PIPELINE_RUNTIME_VERSION,
    PROFILE_STAGE_VERSION,
    NORMALIZED_PROFILE_STAGE_VERSION,
    EPISODE_STAGE_VERSION,
    CAPA_EIM_STAGE_VERSION,
    JIT_STAGE_VERSION,
    atomic_write_json,
    atomic_write_text,
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


class DataPreparationPipeline:
    """
    Orchestrates the complete data preparation pipeline for the Meeting Delegate system.
    
    This pipeline takes raw meeting transcripts and generates all the necessary data
    structures for simulation: user profiles, test cases, JIT briefings, and EIM states.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the data preparation pipeline.
        
        Args:
            config_path: Path to the configuration file (None for auto-detection)
        """
        self.config = Config(config_path)
        self.llm_client = LLMClient(self.config)
        
        # Initialize all processors
        self.transcript_processor = TranscriptProcessor()
        self.profile_generator = UserProfileGenerator(self.llm_client)
        self.test_case_generator = TestCaseGenerator(self.llm_client)
        self.briefing_generator = JITBriefingGenerator(self.llm_client)
        self.eim = EnvironmentInterpretationModule(self.llm_client)  # Fixed: use self.llm_client as 'client' parameter
        
        print("Initialized Data Preparation Pipeline with all processors")

    def _role_summary_from_profile(self, profile_path: Optional[str]) -> Optional[str]:
        """Load a concise 1�2 line role summary for a participant from a normalized profile JSON.

        Tries fields commonly present in our normalized profiles. Falls back to None if missing.
        """
        if not profile_path or not os.path.exists(profile_path):
            return None
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                prof = json.load(f)
        except Exception:
            return None
        # Try inferred_role_and_goals first
        try:
            irg = prof.get('inferred_role_and_goals') or {}
            primary_role = irg.get('primary_role')
            goals = irg.get('high_level_goals') or []
            if primary_role:
                summary = str(primary_role)
                if goals:
                    g = ", ".join([str(x) for x in goals[:2]])
                    summary = f"{summary}. Focus: {g}."
                # Append aliases if present
                aliases = prof.get('identity_aliases') or []
                if isinstance(aliases, list) and aliases:
                    # keep up to 2 concise aliases
                    ali = ", ".join([str(a) for a in aliases[:2]])
                    summary = f"{summary} Aliases: {ali}."
                return summary
        except Exception:
            pass
        # Fallback: core_info knowledge areas
        try:
            core = prof.get('core_info') or {}
            kas = core.get('knowledge_areas') or []
            if kas:
                g = ", ".join([str(x) for x in kas[:2]])
                summary = f"Focus: {g}."
                aliases = prof.get('identity_aliases') or []
                if isinstance(aliases, list) and aliases:
                    ali = ", ".join([str(a) for a in aliases[:2]])
                    summary = f"{summary} Aliases: {ali}."
                return summary
        except Exception:
            pass
        return None


    @staticmethod
    def _sanitize_contrib_responses(
        contrib: Dict[str, Any],
        id_to_idx: Dict[str, int],
        transcript: List[Dict[str, Any]],
        represented_user: str,
    ) -> Tuple[bool, str, int]:
        """Ensure response_utterance_ids reference the represented user's substantive turns.

        Attempts small corrections by re-pointing adjacent participant utterances when possible.
        Returns (ok_after_filter, reason_if_not, adjustments_made).
        """
        resp_ids = list(contrib.get("response_utterance_ids") or [])
        if not resp_ids:
            return False, "no response ids provided", 0
        filtered: List[str] = []
        adjustments = 0
        for rid in resp_ids:
            idx = id_to_idx.get(rid)
            if idx is None:
                adjustments += 1
                continue
            if transcript[idx].get("speaker_id") == represented_user:
                filtered.append(rid)
                continue
            replacement = None
            for delta in (1, -1):
                neighbor_idx = idx + delta
                if neighbor_idx < 0 or neighbor_idx >= len(transcript):
                    continue
                if transcript[neighbor_idx].get("speaker_id") != represented_user:
                    continue
                text = transcript[neighbor_idx].get("utterance_content") or transcript[neighbor_idx].get("content") or ""
                if not is_banal(text):
                    replacement = transcript[neighbor_idx].get("utterance_id")
                    break
            if replacement:
                filtered.append(replacement)
                adjustments += 1
            else:
                adjustments += 1
        if not filtered:
            return False, "no participant responses after filtering", adjustments
        seen: set[str] = set()
        ordered: List[str] = []
        for rid in filtered:
            if rid and rid not in seen:
                seen.add(rid)
                ordered.append(rid)
        ordered.sort(key=lambda rid: id_to_idx.get(rid, 10**9))
        contrib["response_utterance_ids"] = ordered
        contrib["first_response_idx"] = id_to_idx.get(ordered[0])
        return True, "", adjustments

    def process_single_transcript(self,
                                 transcript_path: str,
                                 output_dir: str,
                                 meeting_id: str,
                                 profile_split_ratio: float = 0.6,
                                  target_participant: str = None,
                                 participant_subset: Optional[List[str]] = None,
                                 generate_test_cases: bool = True,
                                 generate_eim_states: bool = True,
                                 capa_N: int = 20,
                                 capa_k: int = 5,
                                 freeze_anchor_inventory: bool = False,
                                 anchor_inventory_dir: Optional[str] = None,
                                 anchor_seed_k: Optional[int] = None,
                                 static_prep_source_dir: Optional[str] = None,
                                 eim_cache_dir: Optional[str] = None,
                                 selector: str = "dense",
                                 cap_meeting: int = 20,
                                 cap_per_participant: int = 5,
                                 min_gap: int = 5,
                                 debug_episodes: bool = False,
                                 # JIT improvement flags (downstream-safe; defaults preserve legacy behavior)
                                 jit_enable_sliding_windows: bool = False,
                                 jit_window_size: int = 60,
                                 jit_window_stride: int = 20,
                                 jit_emit_evidence_sidecar: bool = True,
                                 jit_enable_llm_cleanup: bool = False,
                                 resume: bool = False,
                                 force: bool = False,
                                 ) -> Dict[str, Any]:
        """
        Process a single transcript through the complete pipeline.
        
        Args:
            transcript_path: Path to the parsed transcript JSON file
            output_dir: Base output directory for all generated data
            meeting_id: Identifier for the meeting
            profile_split_ratio: Ratio for splitting transcript (profile vs test data)
            target_participant: If specified, only process this participant (e.g., 'Marketing')
            
        Returns:
            Dictionary with processing results and file paths
        """
        print(f"=" * 60, flush=True)
        print(f"PROCESSING TRANSCRIPT: {meeting_id}", flush=True)
        print(f"Source: {transcript_path}")
        print(f"Output: {output_dir}")
        print(f"=" * 60)

        if not generate_test_cases and not generate_eim_states:
            return run_capa_prep(
                self,
                transcript_path=transcript_path,
                output_dir=output_dir,
                meeting_id=meeting_id,
                profile_split_ratio=profile_split_ratio,
                target_participant=target_participant,
                participant_subset=participant_subset,
                capa_N=capa_N,
                capa_k=capa_k,
                freeze_anchor_inventory=freeze_anchor_inventory,
                anchor_inventory_dir=anchor_inventory_dir,
                anchor_seed_k=anchor_seed_k,
                static_prep_source_dir=static_prep_source_dir,
                eim_cache_dir=eim_cache_dir,
                selector=selector,
                cap_meeting=cap_meeting,
                cap_per_participant=cap_per_participant,
                min_gap=min_gap,
                debug_episodes=debug_episodes,
                jit_enable_sliding_windows=jit_enable_sliding_windows,
                jit_window_size=jit_window_size,
                jit_window_stride=jit_window_stride,
                jit_emit_evidence_sidecar=jit_emit_evidence_sidecar,
                jit_enable_llm_cleanup=jit_enable_llm_cleanup,
                resume=resume,
                force=force,
            )
        
        # Create output directories
        profiles_dir = os.path.join(output_dir, "user_profiles")
        test_cases_dir = os.path.join(output_dir, "test_cases")
        briefings_dir = os.path.join(output_dir, "jit_briefings")
        eim_outputs_dir = os.path.join(output_dir, "eim_outputs")
        capa_episodes_dir = os.path.join(output_dir, "capa_episodes")
        capa_eim_dir = os.path.join(output_dir, "capa_eim_states")
        
        for dir_path in [profiles_dir, test_cases_dir, briefings_dir, eim_outputs_dir, capa_episodes_dir, capa_eim_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        results = {
            "meeting_id": meeting_id,
            "transcript_path": transcript_path,
            "output_directories": {
                "profiles": profiles_dir,
                "test_cases": test_cases_dir,
                "briefings": briefings_dir,
                "eim_outputs": eim_outputs_dir
            },
            "generated_files": {},
            "participants": [],
            "processing_stats": {}
        }
        
        try:
            # Step 1: Load and split transcript
            print("\n>> Loading and splitting transcript...", flush=True)
            with open(transcript_path, 'r', encoding='utf-8') as f:
                full_transcript = json.load(f)
            
            profile_segment, testing_segment = self.transcript_processor.split_transcript(
                full_transcript, profile_split_ratio
            )
            
            # Extract unique participants
            all_participants = list(set(utt['speaker_id'] for utt in full_transcript))
            
            # Filter participants if target specified
            if target_participant:
                if target_participant in all_participants:
                    participants = [target_participant]
                    print(f"   - Filtering to target participant: {target_participant}")
                else:
                    print(f"   - [ERROR] Target participant '{target_participant}' not found in transcript")
                    print(f"   - Available participants: {', '.join(all_participants)}")
                    participants = []
            else:
                participants = all_participants
            
            results["participants"] = participants
            
            print(f"   - Total utterances: {len(full_transcript)}")
            print(f"   - Profile segment: {len(profile_segment)} utterances")
            print(f"   - Testing segment: {len(testing_segment)} utterances")
            print(f"   - All participants: {', '.join(all_participants)}")
            if target_participant:
                print(f"   - Processing participant: {target_participant}")
            else:
                print(f"   - Processing all participants")

            manifest_path = Path(output_dir) / "prep_manifest.json"
            model_descriptor = {
                "active_model_name": getattr(self.config, "active_model_name", None),
                "provider": getattr(self.config, "provider", None),
                "model": getattr(self.config, "model", None),
            }
            prep_contract_payload = {
                "stage": "prep",
                "stage_version": PIPELINE_RUNTIME_VERSION,
                "meeting_id": meeting_id,
                "transcript_fingerprint": fingerprint_transcript_file(Path(transcript_path)),
                "profile_split_ratio": profile_split_ratio,
                "participants": sorted(participants),
                "target_participant": target_participant,
                "generate_test_cases": generate_test_cases,
                "generate_eim_states": generate_eim_states,
                "selector": selector,
                "capa_N": capa_N,
                "capa_k": capa_k,
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
            prep_contract_fingerprint = fingerprint_contract(**prep_contract_payload)
            prep_output_paths = []
            for participant_id in participants:
                slug = participant_id.replace(' ', '_')
                prep_output_paths.append(Path(profiles_dir) / f"capa_profile_for_{slug}_in_{meeting_id}.json")
                prep_output_paths.append(Path(briefings_dir) / f"{slug}_briefing_from_{meeting_id}_partB.json")
                prep_output_paths.append(Path(capa_episodes_dir) / f"capa_episodes_for_{slug}_from_{meeting_id}.json")
                prep_output_paths.append(Path(capa_eim_dir) / f"capa_eim_states_for_{slug}_in_{meeting_id}.json")

            with file_lock(Path(output_dir) / ".prep.lock"):
                prep_manifest = load_manifest(manifest_path, default_version=PREP_MANIFEST_VERSION)
                meeting_unit = (prep_manifest.get("units") or {}).get("meeting")
                if resume and not force and unit_is_reusable(meeting_unit, prep_contract_fingerprint, prep_output_paths):
                    profile_files = {
                        participant_id: str(Path(profiles_dir) / f"capa_profile_for_{participant_id.replace(' ', '_')}_in_{meeting_id}.json")
                        for participant_id in participants
                    }
                    briefing_files = {
                        participant_id: str(Path(briefings_dir) / f"{participant_id.replace(' ', '_')}_briefing_from_{meeting_id}_partB.json")
                        for participant_id in participants
                    }
                    capa_episode_files = {
                        participant_id: str(Path(capa_episodes_dir) / f"capa_episodes_for_{participant_id.replace(' ', '_')}_from_{meeting_id}.json")
                        for participant_id in participants
                    }
                    capa_eim_files = {
                        participant_id: str(Path(capa_eim_dir) / f"capa_eim_states_for_{participant_id.replace(' ', '_')}_in_{meeting_id}.json")
                        for participant_id in participants
                    }
                    results["generated_files"]["profiles"] = profile_files
                    results["generated_files"]["capa_user_profiles"] = profile_files
                    results["generated_files"]["briefings"] = briefing_files
                    results["generated_files"]["capa_episodes"] = capa_episode_files
                    results["generated_files"]["capa_eim_states"] = capa_eim_files
                    results["processing_stats"] = {
                        "profiles_generated": len(profile_files),
                        "capa_profiles_normalized": len(profile_files),
                        "briefings_generated": len(briefing_files),
                        "capa_episodes_generated": len(capa_episode_files),
                        "capa_eim_states_generated": len(capa_eim_files),
                    }
                    results["success"] = True
                    return results
                mark_unit_in_progress(prep_manifest, "meeting", prep_contract_fingerprint, {
                    "meeting_id": meeting_id,
                    "participants": participants,
                    "contract": prep_contract_payload,
                })
                save_manifest(manifest_path, prep_manifest)
            
            # Step 2: Generate user profiles
            print("\n>> Generating user profiles...", flush=True)
            profile_files = {}
            
            # In CAPA mode, avoid writing duplicate raw+normalized profiles
            normalized_profile_files: Dict[str, str] = {}
            
            for participant_id in participants:
                print(f"   - Generating profile for {participant_id}...", flush=True)
                
                try:
                    # Format profile segment for LLM prompt
                    profile_segment_string = self.profile_generator.format_utterances_for_prompt(profile_segment)
                    
                    user_profile = self.profile_generator.generate_rich_user_profile(
                        participant_speaker_id=participant_id,
                        meeting_transcript_segment_string=profile_segment_string
                    )
                    
                    if user_profile:
                        if generate_test_cases:
                            profile_filename = f"{participant_id.replace(' ', '_')}_profile_from_{meeting_id}_partA.json"
                            profile_path = os.path.join(profiles_dir, profile_filename)
                            with open(profile_path, 'w', encoding='utf-8') as f:
                                f.write(user_profile.model_dump_json(indent=2))
                            profile_files[participant_id] = profile_path
                            print(f"     [OK] Saved to {profile_filename}")
                        else:
                            normalized_filename = (f"capa_profile_for_{participant_id.replace(' ', '_')}_in_{meeting_id}.json")
                            normalized_path = os.path.join(profiles_dir, normalized_filename)
                            payload = user_profile.model_dump()
                            profile_payload = payload.get('UserProfile', payload)
                            with open(normalized_path, 'w', encoding='utf-8') as nf:
                                json.dump(profile_payload, nf, indent=2, ensure_ascii=False)
                            profile_files[participant_id] = normalized_path
                            normalized_profile_files[participant_id] = normalized_path
                            print(f"     [OK] Saved CAPA profile -> {normalized_filename}")
                    else:
                        print(f"     ? Failed to generate profile for {participant_id}")
                        
                except Exception as e:
                    print(f"     ? Error generating profile for {participant_id}: {e}")
                    print(f"     Error type: {type(e).__name__}")
                    import traceback
                    print(f"     Full traceback: {traceback.format_exc()}")
            
            results["generated_files"]["profiles"] = profile_files
            results["processing_stats"]["profiles_generated"] = len(profile_files)

            if generate_test_cases:
                # Normalize profiles for CAPA consumption (strip LLM thoughts, keep schema only)
                normalized_profile_files: Dict[str, str] = {}
                for participant_id, profile_path in profile_files.items():
                    if not profile_path or not os.path.exists(profile_path):
                        continue
                    try:
                        with open(profile_path, 'r', encoding='utf-8') as pf:
                            raw_profile = json.load(pf)
                        profile_payload = raw_profile.get('UserProfile', raw_profile)
                        normalized_filename = (
                            f"capa_profile_for_{participant_id.replace(' ' , '_')}_in_{meeting_id}.json"
                        )
                        normalized_path = os.path.join(profiles_dir, normalized_filename)
                        with open(normalized_path, 'w', encoding='utf-8') as nf:
                            json.dump(profile_payload, nf, indent=2, ensure_ascii=False)
                        normalized_profile_files[participant_id] = normalized_path
                        print(f"     [OK] Normalized CAPA profile saved -> {normalized_filename}")
                    except Exception as e:
                        print(f"     [WARN] Failed to normalize profile for {participant_id}: {e}")

            if normalized_profile_files:
                results["generated_files"]["capa_user_profiles"] = normalized_profile_files
                results["processing_stats"]["capa_profiles_normalized"] = len(normalized_profile_files)

            # Step 3: Generate test cases (legacy) or CAPA episodes
            test_case_files = {}
            total_test_cases = 0
            if generate_test_cases:
                print("\n>> Generating test cases...")
                # Delegate the entire test case generation process to the TestCaseGenerator
                test_case_files = self.test_case_generator.generate_test_cases_for_meeting(
                    transcript_path=transcript_path,
                    user_profiles_directory=profiles_dir,
                    output_directory=test_cases_dir,
                    split_ratio=profile_split_ratio,
                    meeting_id=meeting_id,
                    target_participant=target_participant
                )
                # Calculate stats based on the returned file paths
                if test_case_files:
                    for path in test_case_files.values():
                        try:
                            with open(path, 'r', encoding='utf-8') as f:
                                total_test_cases += len(json.load(f))
                        except (FileNotFoundError, json.JSONDecodeError):
                            continue
                print(f"   [OK] Test case generation complete. Found {total_test_cases} total instances.")
            else:
                print("\n>> Generating CAPA episodes (CAPA mode)...", flush=True)
                # Build CAPA episodes from the testing segment (Part B) only, per participant
                # Prepare utterance index over full transcript for cue indexing
                full_utt_index = {utt['utterance_id']: i for i, utt in enumerate(full_transcript)}

                capa_episode_files: Dict[str, str] = {}
                capa_episode_counts: Dict[str, int] = {}

                # Generate episodes once for all participants
                if selector == "dense":
                    print(f"   - [dense] Generating CAPA episodes for {len(participants)} participant(s) (N={capa_N}, k={capa_k})...")
                    episodes_by_participant = generate_dense_episodes_multi_participant(
                        transcript=testing_segment,
                        participants=participants,
                        N=capa_N,
                        k=capa_k,
                        step=1,
                    )
                elif selector == "llm-idea":
                    print(f"   - [llm-idea] Selecting Idea Units per participant (N={capa_N}, k={capa_k})...")
                    from capa.cue_selector_llm import build_transcript_table
                    from capa.idea_unit_selector_llm import select_idea_units
                    from capa.episode_utils import (
                        build_id_index,
                        validate_idea_unit,
                        build_episode_from_contrib,
                        has_future_rep,
                    )
                    from schemas import IdeaUnit

                    start_uid = testing_segment[0]['utterance_id']
                    end_uid = testing_segment[-1]['utterance_id']
                    table = build_transcript_table(full_transcript, full_utt_index, start_uid, end_uid)

                    episodes_by_participant: Dict[str, List[CAPAEpisode]] = {}

                    normalized_map: Dict[str, str] = {}
                    try:
                        normalized_map = results.get("generated_files", {}).get("capa_user_profiles", {})
                    except Exception:
                        normalized_map = {}

                    id_to_idx = build_id_index(full_transcript)

                    for p in participants:
                        prof_path = normalized_map.get(p) or os.path.join(profiles_dir, f"capa_profile_for_{p.replace(' ', '_')}_in_{meeting_id}.json")
                        role_summary = self._role_summary_from_profile(prof_path) or p

                        sel = select_idea_units(
                            client=self.llm_client,
                            meeting_id=meeting_id,
                            participant_id=p,
                            role_summary=role_summary,
                            transcript_table=table,
                            N=capa_N,
                            k=capa_k,
                        )
                        raw_units = sel.get("idea_units") or []
                        if debug_episodes:
                            print(f"     [DBG] {p}: selector returned {len(raw_units)} idea_unit(s)")
                        units = sorted(raw_units, key=lambda u: float(u.get("importance_score") or 0.0), reverse=True)

                        eps: List[CAPAEpisode] = []
                        for u in units:
                            uid = u.get("unit_id") or ""
                            if debug_episodes:
                                print(f"       [DBG] Validate {p}:{uid} cue={u.get('cue_utterance_id')}@{u.get('cue_idx')} -> first_resp_idx={u.get('first_response_idx')}; imp={u.get('importance_score')} type={u.get('cue_type')}")
                            ok, why, k_used = validate_idea_unit(
                                unit=u,
                                id_to_idx=id_to_idx,
                                transcript=full_transcript,
                                represented_user=p,
                                N=capa_N,
                                k=capa_k,
                                k_max=capa_k + 3,  # Allow slight flexibility (k+3) but not excessive
                            )
                            if not ok:
                                if debug_episodes:
                                    print(f"       [DROP] {p}:{uid} invalid idea_unit: {why}")
                                continue
                            ep = build_episode_from_contrib(
                                contrib=u,
                                transcript=full_transcript,
                                represented_user=p,
                                N=capa_N,
                                k_used=k_used,
                            )
                            # Attach episode-local idea_units (only this unit; anchor convenience)
                            try:
                                anchor_utt_id = (u.get("response_utterance_ids") or [None])[0]
                                norm_unit = dict(u)
                                norm_unit["anchor_utt_id"] = anchor_utt_id
                                # Coerce to IdeaUnit model for clean serialization
                                ep.idea_units = [IdeaUnit(**norm_unit)]
                            except Exception:
                                try:
                                    ep.idea_units = [IdeaUnit(**u)]
                                except Exception:
                                    ep.idea_units = [u]
                            present, contentful, _ = has_future_rep(ep.gt_next_window or [], p)
                            if present and contentful:
                                eps.append(ep)
                                if debug_episodes:
                                    print(f"       [KEEP] {p}:{uid} present+contentful in k_window (len={len(ep.gt_next_window or [])})")
                            else:
                                if debug_episodes:
                                    toks: List[str] = []
                                    for utt in ep.gt_next_window or []:
                                        if utt.get('speaker_id') == p:
                                            txt = utt.get('utterance_content') or utt.get('content') or ''
                                            ban = is_banal(txt)
                                            toks.append(f"{utt.get('utterance_id')} tok={len(str(txt).split())} ban={ban}")
                                    detail = ", ".join(toks) if toks else "no_user_turns"
                                    print(f"       [DROP] {p}:{uid} present={present} contentful={contentful}; user_turns: {detail}")
                        episodes_by_participant[p] = eps
                        if debug_episodes:
                            print(f"     [DBG] {p}: kept {len(eps)}/{len(units)} after validation/contentfulness")
                else:
                    print(f"   - [llm-contrib] Selecting contribution spans and cues per participant (N={capa_N}, k={capa_k})...")
                    # Build global index map and testing bounds
                    from capa.cue_selector_llm import build_transcript_table
                    from capa.contribution_selector_llm import select_contributions
                    from capa.episode_utils import (
                        build_id_index,
                        validate_contribution,
                        build_episode_from_contrib,
                        has_future_rep,
                    )

                    start_uid = testing_segment[0]['utterance_id']
                    end_uid = testing_segment[-1]['utterance_id']
                    table = build_transcript_table(full_transcript, full_utt_index, start_uid, end_uid)

                    episodes_by_participant: Dict[str, List[CAPAEpisode]] = {}

                    # Try to discover normalized profile files captured earlier in this run
                    normalized_map: Dict[str, str] = {}
                    try:
                        normalized_map = results.get("generated_files", {}).get("capa_user_profiles", {})
                    except Exception:
                        normalized_map = {}

                    id_to_idx = build_id_index(full_transcript)

                    for p in participants:
                        # Prefer role summary from normalized profile if available
                        prof_path = normalized_map.get(p) or os.path.join(profiles_dir, f"capa_profile_for_{p.replace(' ', '_')}_in_{meeting_id}.json")
                        role_summary = self._role_summary_from_profile(prof_path) or p

                        sel = select_contributions(
                            client=self.llm_client,
                            meeting_id=meeting_id,
                            participant_id=p,
                            role_summary=role_summary,
                            transcript_table=table,
                            N=capa_N,
                            k=capa_k,
                        )
                        raw_contribs = sel.get("contributions") or []
                        if debug_episodes:
                            print(f"     [DBG] {p}: selector returned {len(raw_contribs)} contribution(s)")
                        # Sort by importance_score desc (fallback 0.0)
                        contribs = sorted(raw_contribs, key=lambda c: float(c.get("importance_score") or 0.0), reverse=True)


                        eps: List[CAPAEpisode] = []
                        for c in contribs:
                            c_id = c.get("contribution_id") or ""
                            cue_uid = c.get("cue_utterance_id")
                            cue_idx = c.get("cue_idx")
                            first_idx = c.get("first_response_idx")
                            if debug_episodes:
                                print(f"       [DBG] Validate {p}:{c_id} cue={cue_uid}@{cue_idx} -> first_resp_idx={first_idx}; imp={c.get('importance_score')} type={c.get('cue_type')}")
                            ok, why, k_used = validate_contribution(
                                contrib=c,
                                id_to_idx=id_to_idx,
                                transcript=full_transcript,
                                represented_user=p,
                                N=capa_N,
                                k=capa_k,
                                k_max=capa_k + 3,  # Allow slight flexibility (k+3) but not excessive
                            )
                            if not ok:
                                lower = (why or "").lower()
                                if "not by represented user" in lower or "unknown response id" in lower:
                                    san_ok, san_reason, san_adj = self._sanitize_contrib_responses(
                                        contrib=c,
                                        id_to_idx=id_to_idx,
                                        transcript=full_transcript,
                                        represented_user=p,
                                    )
                                    if san_ok and san_adj:
                                        if debug_episodes:
                                            print(f"       [DBG] {p}:{c_id} sanitize adjusted {san_adj} response id(s)")
                                        cue_idx = c.get("cue_idx")
                                        first_idx = c.get("first_response_idx")
                                        ok, why, k_used = validate_contribution(
                                            contrib=c,
                                            id_to_idx=id_to_idx,
                                            transcript=full_transcript,
                                            represented_user=p,
                                            N=capa_N,
                                            k=capa_k,
                                            k_max=capa_k + 3,  # Allow slight flexibility (k+3)
                                        )
                                    else:
                                        if debug_episodes:
                                            reason = san_reason or "sanitize produced no participant responses"
                                            print(f"       [DROP] {p}:{c_id} sanitize failed: {reason}")
                                        continue
                            if not ok:
                                if debug_episodes:
                                    print(f"       [DROP] {p}:{c_id} invalid contribution: {why}")
                                continue
                            if debug_episodes:
                                delta = None
                                try:
                                    if c.get("first_response_idx") is not None and c.get("cue_idx") is not None:
                                        delta = int(c.get("first_response_idx")) - int(c.get("cue_idx"))
                                except (TypeError, ValueError):
                                    delta = None
                                print(f"       [OK ] {p}:{c_id} validated; k_used={k_used}; delta(first-cue)={delta}")
                            ep = build_episode_from_contrib(
                                contrib=c,
                                transcript=full_transcript,
                                represented_user=p,
                                N=capa_N,
                                k_used=k_used,
                            )
                            present, contentful, _ = has_future_rep(ep.gt_next_window or [], p)
                            if present and contentful:
                                eps.append(ep)
                                if debug_episodes:
                                    print(f"       [KEEP] {p}:{c_id} present+contentful in k_window (len={len(ep.gt_next_window or [])})")
                            else:
                                if debug_episodes:
                                    # Provide per-utterance token counts and banality flags for the represented user
                                    toks: List[str] = []
                                    for utt in ep.gt_next_window or []:
                                        if utt.get('speaker_id') == p:
                                            txt = utt.get('utterance_content') or utt.get('content') or ''
                                            ban = is_banal(txt)
                                            toks.append(f"{utt.get('utterance_id')} tok={len(str(txt).split())} ban={ban}")
                                    detail = ", ".join(toks) if toks else "no_user_turns"
                                    print(f"       [DROP] {p}:{c_id} present={present} contentful={contentful}; user_turns: {detail}")
                        episodes_by_participant[p] = eps
                        if debug_episodes:
                            print(f"     [DBG] {p}: kept {len(eps)}/{len(contribs)} after validation/contentfulness")

                # Reindex and save per participant
                for participant_id, episodes in episodes_by_participant.items():
                    # Reindex cue_index and test_case_id to full transcript indices using cue_utt_id
                    for ep in episodes:
                        if ep.cue_utt_id in full_utt_index:
                            global_index = full_utt_index[ep.cue_utt_id]
                            ep.cue_index = global_index
                            # Fix test_case_id to use global transcript index for consistency
                            ep.test_case_id = f"{ep.meeting_id}_{ep.represented_user}_t{global_index}"
                    # Sort episodes chronologically by cue_index
                    try:
                        episodes.sort(key=lambda e: (e.cue_index if e.cue_index is not None else 10**9))
                    except Exception:
                        pass
                    
                    ep_path = os.path.join(capa_episodes_dir, f"capa_episodes_for_{participant_id.replace(' ', '_')}_from_{meeting_id}.json")
                    write_episodes_to_file(episodes, ep_path)
                    capa_episode_files[participant_id] = ep_path
                    capa_episode_counts[participant_id] = len(episodes)
                    print(f"     [OK] Saved {len(episodes)} CAPA episodes for {participant_id} -> {ep_path}")

                results["generated_files"]["capa_episodes"] = capa_episode_files
                results["processing_stats"]["capa_episodes_generated"] = sum(capa_episode_counts.values())

                # 4. Deduplicated EIM precompute per unique cue+context (single pass)
                print("\n>> Precomputing CAPA EIM states (deduplicated per unique window)...", flush=True)

                def _ctx_ids(ep: CAPAEpisode) -> List[str]:
                    return [u.get('utterance_id') for u in (ep.context_last_N or [])]

                def _ctx_hash(ids: List[str]) -> str:
                    joined = "|".join(ids)
                    return hashlib.sha256(joined.encode('utf-8')).hexdigest()[:16]

                unique_eps: Dict[str, CAPAEpisode] = {}
                consumers: Dict[str, List[str]] = {}
                total_eps = 0

                for p, eps in episodes_by_participant.items():
                    for ep in eps:
                        total_eps += 1
                        ids = _ctx_ids(ep)
                        ukey = f"{ep.cue_utt_id}__{_ctx_hash(ids)}"
                        if ukey not in unique_eps:
                            rep = CAPAEpisode(
                                meeting_id=ep.meeting_id,
                                represented_user=ep.represented_user,
                                cue_utt_id=ep.cue_utt_id,
                                cue_index=ep.cue_index,
                                N=ep.N,
                                k=ep.k,
                                context_last_N=ep.context_last_N,
                                gt_next_window=ep.gt_next_window,
                                test_case_id=f"UKEY:{ukey}",
                            )
                            unique_eps[ukey] = rep
                            consumers[ukey] = []
                        else:
                            # Safety: ensure exact same window
                            prior_ids = _ctx_ids(unique_eps[ukey])
                            if prior_ids != ids:
                                # Force a distinct key using full JSON hash when contexts differ
                                full_hash = hashlib.sha256(json.dumps(ep.context_last_N, ensure_ascii=False).encode('utf-8')).hexdigest()[:16]
                                ukey = f"{ep.cue_utt_id}__{full_hash}"
                                rep = CAPAEpisode(
                                    meeting_id=ep.meeting_id,
                                    represented_user=ep.represented_user,
                                    cue_utt_id=ep.cue_utt_id,
                                    cue_index=ep.cue_index,
                                    N=ep.N,
                                    k=ep.k,
                                    context_last_N=ep.context_last_N,
                                    gt_next_window=ep.gt_next_window,
                                    test_case_id=f"UKEY:{ukey}",
                                )
                                unique_eps[ukey] = rep
                                consumers[ukey] = []
                        consumers[ukey].append(ep.test_case_id)

                unique_list = list(unique_eps.values())
                # Reuse the pipeline's EIM instance for a single chronological pass
                precomp_unique = precompute_eim_for_episodes(
                    transcript=full_transcript,
                    episodes=unique_list,
                    eim=self.eim,
                )
                # Map back UKEY -> EIM vector
                ukey_to_vec: Dict[str, Dict[str, Any]] = {}
                for entry in precomp_unique:
                    tcid = entry.get('test_case_id', '')
                    if isinstance(tcid, str) and tcid.startswith('UKEY:'):
                        ukey = tcid.split('UKEY:', 1)[1]
                        ukey_to_vec[ukey] = entry.get('eim_output')

                # Fan-out per participant
                capa_eim_files: Dict[str, str] = {}
                total_states = 0
                tcid_to_part: Dict[str, str] = {}
                for p, eps in episodes_by_participant.items():
                    for ep in eps:
                        tcid_to_part[ep.test_case_id] = p

                per_part_entries: Dict[str, List[Dict[str, Any]]] = {p: [] for p in participants}
                for ukey, tcids in consumers.items():
                    vec = ukey_to_vec.get(ukey)
                    if vec is None:
                        continue
                    for tcid in tcids:
                        p = tcid_to_part.get(tcid)
                        if not p:
                            continue
                        per_part_entries[p].append({
                            'test_case_id': tcid,
                            'eim_output': vec,
                        })
                        total_states += 1

                for p, entries in per_part_entries.items():
                    out_path = os.path.join(capa_eim_dir, f"capa_eim_states_for_{p.replace(' ', '_')}_in_{meeting_id}.json")
                    with open(out_path, 'w', encoding='utf-8') as f:
                        json.dump(entries, f, indent=2, ensure_ascii=False)
                    capa_eim_files[p] = out_path
                    print(f"     [OK] Saved {len(entries)} CAPA EIM states for {p} -> {out_path}")

                results["generated_files"]["capa_eim_states"] = capa_eim_files
                results["processing_stats"]["capa_eim_states_generated"] = total_states
                print(f"   => Unique windows: {len(unique_list)} | Episodes total: {total_eps} | Fan-out states: {total_states}")
            results["generated_files"]["test_cases"] = test_case_files
            results["processing_stats"]["test_cases_generated"] = total_test_cases
            
            # Step 4: Generate JIT briefings
            print("\n>> Generating JIT briefings...")
            
            briefing_files = self.briefing_generator.generate_briefings_for_meeting(
                transcript_path=transcript_path,
                output_directory=briefings_dir,
                split_ratio=profile_split_ratio,
                meeting_id=meeting_id,
                target_participant=target_participant,
                enable_sliding_windows=jit_enable_sliding_windows,
                window_size=jit_window_size,
                window_stride=jit_window_stride,
                emit_evidence_sidecar=jit_emit_evidence_sidecar,
                enable_llm_cleanup=jit_enable_llm_cleanup,
            )
            
            results["generated_files"]["briefings"] = briefing_files
            results["processing_stats"]["briefings_generated"] = len(briefing_files)
            print(f"   [OK] JIT briefing generation complete. Found {len(briefing_files)} total briefings.")
            
            # Step 5: Generate EIM states (legacy) or CAPA EIM (single pass)
            if generate_eim_states:
                print("\n>> Generating EIM states...")

                # Efficiently process the entire meeting chronologically while preserving per-participant output.
                # First, build the EIM's master state using the initial part of the transcript.
                print("   - Building initial EIM master state from profile segment...")
                # Direct chunk processing without seeding
                print("   - Using direct chunk processing of profile segment...")
                # Then run chunked history to refine and extend with diffs
                self.eim.process_history(profile_segment)
                # Trace summary of master state after profile segment
                try:
                    topics_total = len(getattr(self.eim.master_state, 'topic_summary_log', {}) or {})
                    participants_total = len(getattr(self.eim.master_state, 'participant_models', {}) or {})
                    aliases_total = len(getattr(self.eim.master_state, 'identity_aliases', {}) or {})
                    print(f"[EIM-TRACE] PROFILE_DONE | topics={topics_total}, participants={participants_total}, aliases={aliases_total}")
                except Exception:
                    print("[EIM-TRACE] PROFILE_DONE | <unable to summarize>")

                # Collect all test cases from all participants into a single list.
                all_test_cases = []
                for participant_id in participants:
                    if participant_id in test_case_files:
                        try:
                            with open(test_case_files[participant_id], 'r', encoding='utf-8') as f:
                                all_test_cases.extend(json.load(f))
                        except (FileNotFoundError, json.JSONDecodeError):
                            continue

                # Sort all test cases chronologically based on their appearance in the full transcript.
                utterance_index_map = {utt['utterance_id']: i for i, utt in enumerate(full_transcript)}
                all_test_cases.sort(key=lambda tc: utterance_index_map.get(tc.get('input_snapshot_utterances', [{}])[0].get('utterance_id'), float('inf')))

                print(f"   - Processing {len(all_test_cases)} total test cases chronologically...")

                # Process the rest of the transcript chronologically, generating a state vector at each test case.
                eim_results_in_memory = []
                last_processed_index = len(profile_segment) - 1

                for test_case in all_test_cases:
                    try:
                        # Process "gap" utterances between the last processed point and this snapshot.
                        snapshot_start_id = test_case['input_snapshot_utterances'][0]['utterance_id']
                        snapshot_start_index = utterance_index_map[snapshot_start_id]

                        gap_utterances = full_transcript[last_processed_index + 1:snapshot_start_index]
                        if gap_utterances:
                            try:
                                gid_start = gap_utterances[0].get('utterance_id')
                                gid_end = gap_utterances[-1].get('utterance_id')
                                print(f"[EIM-TRACE] GAP_BEFORE_SNAPSHOT ids={gid_start}..{gid_end} len={len(gap_utterances)}")
                            except Exception:
                                print(f"[EIM-TRACE] GAP_BEFORE_SNAPSHOT len={len(gap_utterances)}")
                            self.eim.process_history(gap_utterances)

                        # Analyze the current snapshot to get the unique meeting state vector for this test case.
                        try:
                            sid_start = test_case['input_snapshot_utterances'][0]['utterance_id']
                            sid_end = test_case['input_snapshot_utterances'][-1]['utterance_id']
                            print(f"[EIM-TRACE] SNAPSHOT_CALL ids={sid_start}..{sid_end} len={len(test_case['input_snapshot_utterances'])}")
                        except Exception:
                            print(f"[EIM-TRACE] SNAPSHOT_CALL len={len(test_case['input_snapshot_utterances'])}")
                        meeting_state_vector = self.eim.analyze_snapshot(test_case['input_snapshot_utterances'])

                        # Save the cue-aligned EIM state (state at decision time)
                        if meeting_state_vector:
                            try:
                                sa = getattr(meeting_state_vector, 'snapshot_analysis', None)
                                micro = getattr(sa, 'effective_micro_topic', None) if sa else None
                                floor = getattr(sa, 'conversational_floor_status', None) if sa else None
                                topics_total = len(getattr(self.eim.master_state, 'topic_summary_log', {}) or {})
                                participants_total = len(getattr(self.eim.master_state, 'participant_models', {}) or {})
                                print(f"[EIM-TRACE] SNAPSHOT_RESULT id={meeting_state_vector.last_processed_utterance_id} micro={micro} floor={floor} topics={topics_total} participants={participants_total}")
                            except Exception:
                                print(f"[EIM-TRACE] SNAPSHOT_RESULT id={getattr(meeting_state_vector, 'last_processed_utterance_id', '?')}")
                            eim_payload = meeting_state_vector.model_dump()
                            # Include identity aliases from master state as of this moment (pre response processing)
                            try:
                                if getattr(self.eim.master_state, 'identity_aliases', None):
                                    eim_payload['identity_aliases'] = self.eim.master_state.identity_aliases
                            except Exception:
                                pass

                            eim_results_in_memory.append({
                                "test_case_id": test_case['test_case_id'],
                                "participant_id": test_case.get('participant_id', test_case.get('principal_participant_id')),
                                "eim_output": eim_payload
                            })

                        # Process the post-snapshot "gap" up to the end of the response (if any)
                        resp_ids = test_case.get('scenario_info_from_llm', {}).get('Response_IDs', [])
                        last_snapshot_id = test_case['input_snapshot_utterances'][-1]['utterance_id']
                        last_snapshot_index = utterance_index_map[last_snapshot_id]
                        if resp_ids:
                            last_response_id = resp_ids[-1]
                            last_response_index = utterance_index_map[last_response_id]
                            if last_response_index > last_snapshot_index:
                                gap_after_snapshot = full_transcript[last_snapshot_index + 1:last_response_index + 1]
                                if gap_after_snapshot:
                                    try:
                                        gid2_start = gap_after_snapshot[0].get('utterance_id')
                                        gid2_end = gap_after_snapshot[-1].get('utterance_id')
                                        print(f"[EIM-TRACE] GAP_AFTER_SNAPSHOT ids={gid2_start}..{gid2_end} len={len(gap_after_snapshot)}")
                                    except Exception:
                                        print(f"[EIM-TRACE] GAP_AFTER_SNAPSHOT len={len(gap_after_snapshot)}")
                                    self.eim.process_history(gap_after_snapshot)
                            last_processed_index = last_response_index
                        else:
                            # No explicit response range; set pointer to the end of snapshot
                            last_processed_index = last_snapshot_index

                        # Note: We intentionally do NOT overwrite the saved EIM state with post-response state.

                    except Exception as e:
                        print(f"     [ERROR] Error generating EIM state for test case {test_case.get('test_case_id', 'N/A')}: {e}")

                # Group the in-memory results by participant and save them to separate files.
                print("   - Saving EIM states to per-participant files...")
                eim_files = {}
                participant_eim_data = {p_id: [] for p_id in participants}
                for result in eim_results_in_memory:
                    participant_eim_data[result['participant_id']].append(result)

                for participant_id, data in participant_eim_data.items():
                    if not data:
                        continue
                    try:
                        eim_filename = f"eim_states_for_{participant_id.replace(' ', '_')}_in_{meeting_id}.json"
                        eim_path = os.path.join(eim_outputs_dir, eim_filename)
                        with open(eim_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                        eim_files[participant_id] = eim_path
                        print(f"     ? Saved {len(data)} EIM states for {participant_id} to {eim_filename}")
                    except Exception as e:
                        print(f"     [ERROR] Error saving EIM states for {participant_id}: {e}")
                results["generated_files"]["eim_states"] = eim_files
                results["processing_stats"]["eim_states_generated"] = len(eim_files)
            
            # Final summary
            print(f"\n{'='*60}")
            print(f"PIPELINE COMPLETE FOR {meeting_id}")
            print(f"{'='*60}")
            print(f"[OK] User Profiles: {results['processing_stats']['profiles_generated']}/{len(participants)}")
            if generate_test_cases:
                print(f"[OK] Test Cases: {results['processing_stats']['test_cases_generated']} total")
            else:
                print(f"[OK] CAPA Episodes: {results['processing_stats'].get('capa_episodes_generated', 0)} total")
            print(f"[OK] JIT Briefings: {results['processing_stats']['briefings_generated']}/{len(participants)}")
            # Print EIM summary depending on mode
            if generate_eim_states:
                print(f"[OK] EIM States: {results['processing_stats']['eim_states_generated']}/{len(participants)}")
            else:
                print(f"[OK] CAPA EIM States: {results['processing_stats'].get('capa_eim_states_generated', 0)} total")
            
            with file_lock(Path(output_dir) / ".prep.lock"):
                prep_manifest = load_manifest(manifest_path, default_version=PREP_MANIFEST_VERSION)
                mark_unit_complete(
                    prep_manifest,
                    "meeting",
                    prep_contract_fingerprint,
                    prep_output_paths,
                    metadata={
                        "meeting_id": meeting_id,
                        "participants": participants,
                        "contract": prep_contract_payload,
                        "generated_files": results.get("generated_files", {}),
                    },
                )
                prep_manifest["state"] = summarize_manifest_state(prep_manifest)
                save_manifest(manifest_path, prep_manifest)
            results["success"] = True
            return results
            
        except Exception as e:
            print(f"[ERROR] Pipeline failed for {meeting_id}: {e}")
            results["success"] = False
            results["error"] = str(e)
            with file_lock(Path(output_dir) / ".prep.lock"):
                prep_manifest = load_manifest(manifest_path, default_version=PREP_MANIFEST_VERSION)
                prep_manifest["state"] = "failed"
                save_manifest(manifest_path, prep_manifest)
            return results
    
    def process_transcript_directory(self,
                                   input_dir: str,
                                   output_dir: str,
                                   profile_split_ratio: float = 0.6) -> Dict[str, Any]:
        """
        Process all transcripts in a directory through the pipeline.
        
        Args:
            input_dir: Directory containing parsed transcript JSON files
            output_dir: Base output directory for all generated data
            profile_split_ratio: Ratio for splitting transcripts
            
        Returns:
            Dictionary with processing results for all transcripts
        """
        print(f"=" * 80)
        print(f"BATCH PROCESSING TRANSCRIPT DIRECTORY")
        print(f"Input: {input_dir}")
        print(f"Output: {output_dir}")
        print(f"=" * 80)
        
        # Find all transcript files
        transcript_files = []
        for file_path in Path(input_dir).glob("*_parsed.json"):
            meeting_id = file_path.stem.replace("_parsed", "")
            transcript_files.append((str(file_path), meeting_id))
        
        print(f"Found {len(transcript_files)} transcript files to process")
        
        batch_results = {
            "input_directory": input_dir,
            "output_directory": output_dir,
            "total_transcripts": len(transcript_files),
            "processed_transcripts": 0,
            "successful_transcripts": 0,
            "failed_transcripts": 0,
            "results": {}
        }
        
        # Process each transcript
        for transcript_path, meeting_id in transcript_files:
            print(f"\n{'='*20} Processing {meeting_id} {'='*20}")
            
            try:
                result = self.process_single_transcript(
                    transcript_path=transcript_path,
                    output_dir=output_dir,
                    meeting_id=meeting_id,
                    profile_split_ratio=profile_split_ratio
                )
                
                batch_results["results"][meeting_id] = result
                batch_results["processed_transcripts"] += 1
                
                if result["success"]:
                    batch_results["successful_transcripts"] += 1
                else:
                    batch_results["failed_transcripts"] += 1
                    
            except Exception as e:
                print(f"[ERROR] Failed to process {meeting_id}: {e}")
                batch_results["results"][meeting_id] = {
                    "success": False,
                    "error": str(e)
                }
                batch_results["processed_transcripts"] += 1
                batch_results["failed_transcripts"] += 1
        
        # Final batch summary
        print(f"\n{'='*80}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"Total transcripts: {batch_results['total_transcripts']}")
        print(f"Processed: {batch_results['processed_transcripts']}")
        print(f"Successful: {batch_results['successful_transcripts']}")
        print(f"Failed: {batch_results['failed_transcripts']}")
        print(f"Success rate: {batch_results['successful_transcripts']}/{batch_results['total_transcripts']}")
        
        return batch_results

    # NOT USED rn, since episodes are identical across participants
    # ===== CAPA (Unified Episodes + EIM states) =====
    def generate_capa_data(
        self,
        transcript_path: str,
        output_dir: str,
        meeting_id: str,
        N: int = 12,
        k: int = 5,
        target_participant: Optional[str] = None,
        step: int = 1,
    ) -> Dict[str, Any]:
        """
        Generate unified CAPA episodes and precomputed EIM states.

        - Builds dense episodes per participant with history N and future k.
        - Precomputes EIMMeetingStateVector at each episode cue in a single pass.

        Returns a dict with episode/eim file paths per participant.
        """
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = json.load(f)

        if not transcript:
            raise ValueError("Transcript is empty; cannot generate CAPA data")

        # Determine participants
        participants = sorted({utt.get('speaker_id') for utt in transcript if 'speaker_id' in utt})
        if target_participant:
            if target_participant in participants:
                participants = [target_participant]
            else:
                print(f"Warning: target participant '{target_participant}' not found; generating for none.")
                participants = []

        # Output dirs
        episodes_dir = os.path.join(output_dir, "benchmark_test_cases", "capa_episodes")
        eim_dir = os.path.join(output_dir, "benchmark_test_cases", "capa_eim_states")
        os.makedirs(episodes_dir, exist_ok=True)
        os.makedirs(eim_dir, exist_ok=True)

        episode_files: Dict[str, str] = {}
        eim_files: Dict[str, str] = {}

        for p in participants:
            print(f"[CAPA] Generating episodes for participant: {p}")
            episodes = generate_dense_episodes(transcript=transcript, represented_user=p, N=N, k=k, step=step)
            ep_path = os.path.join(episodes_dir, f"capa_episodes_for_{p.replace(' ', '_')}_from_{meeting_id}.json")
            write_episodes_to_file(episodes, ep_path)
            episode_files[p] = ep_path
            print(f"[CAPA] Saved {len(episodes)} episodes to {ep_path}")

            # Precompute EIM at cue t for these episodes using a fresh EIM
            eim = EnvironmentInterpretationModule(self.llm_client)
            precomp = precompute_eim_for_episodes(transcript=transcript, episodes=episodes, eim=eim)
            eim_path = os.path.join(eim_dir, f"capa_eim_states_for_{p.replace(' ', '_')}_in_{meeting_id}.json")
            write_eim_states_to_file(precomp, eim_path)
            eim_files[p] = eim_path
            print(f"[CAPA] Saved {len(precomp)} EIM states to {eim_path}")

        return {
            "meeting_id": meeting_id,
            "participants": participants,
            "episode_files": episode_files,
            "eim_state_files": eim_files,
            "success": True,
        }


def run_data_pipeline_cli():
    """
    Command-line interface for running the data preparation pipeline.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Run data preparation pipeline for Meeting Delegate system")
    parser.add_argument("--input", required=True, help="Input transcript file or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--meeting-id", help="Meeting ID (required for single file processing)")
    parser.add_argument("--split-ratio", type=float, default=0.6, help="Profile/testing split ratio")
    parser.add_argument("--participant", help="Deprecated alias for processing only a specific participant (e.g., 'Marketing', 'Project Manager')")
    parser.add_argument("--participants", help="Optional comma-separated participant subset for CAPA-mode prep")
    parser.add_argument("--mode", choices=["legacy", "capa"], default="legacy", help="Data generation mode")
    parser.add_argument("--capa-N", type=int, default=20, help="History length N for CAPA episodes")
    parser.add_argument("--capa-k", type=int, default=5, help="Future window k for CAPA episodes")
    parser.add_argument("--freeze-anchor-inventory", action="store_true", help="Freeze per-participant anchor inventory before rebuilding k-specific episodes")
    parser.add_argument("--anchor-inventory", "--anchor-inventory-dir", dest="anchor_inventory_dir", default=None, help="Directory containing frozen per-participant anchor inventory JSON files")
    parser.add_argument("--anchor-seed-k", type=int, default=None, help="Seed k used when freezing anchor inventory with llm-idea")
    parser.add_argument("--static-prep-source-dir", default=None, help="Optional meeting-specific prep dir providing reusable user_profiles and jit_briefings")
    parser.add_argument("--eim-cache-dir", default=None, help="Optional EIM cache root; defaults to <output>/_eim_cache")
    parser.add_argument("--selector", choices=["dense", "llm", "llm-idea"], default="dense", help="Episode selection strategy in CAPA mode")
    parser.add_argument("--cap-meeting", type=int, default=20, help="Max total episodes per meeting in LLM selector mode")
    parser.add_argument("--cap-per-participant", type=int, default=5, help="Max episodes per participant in LLM selector mode")
    parser.add_argument("--min-gap", type=int, default=5, help="Min utterance gap between selected cues per participant in LLM selector mode")
    parser.add_argument("--debug-capa-episodes", action="store_true", help="Verbose debug prints for CAPA episode validation/contentfulness filters")
    # JIT improvements (downstream-safe; optional)
    parser.add_argument("--jit-enable-sliding-windows", action="store_true", help="Enable sliding-window JIT extraction (testing-segment only)")
    parser.add_argument("--jit-window-size", type=int, default=50, help="JIT sliding window size (utterances)")
    parser.add_argument("--jit-window-stride", type=int, default=30, help="JIT sliding window stride (utterances)")
    parser.add_argument("--jit-emit-evidence-sidecar", action="store_true", help="Emit evidence sidecar JSON if evidence is returned")
    parser.add_argument("--jit-enable-llm-cleanup", action="store_true", help="Use an LLM pass to dedupe/merge final bullets before saving")
    parser.add_argument("--model", default=None, help="Optional active model override; must exist in config_extended.json")
    parser.add_argument("--resume", action="store_true", help="Reuse prep outputs when prep_manifest.json matches the current contract")
    parser.add_argument("--force", action="store_true", help="Ignore prep_manifest.json and recompute prep outputs")
    parser.add_argument("--disable-llm-telemetry", action="store_true", help="Do not write per-call token/latency JSONL logs")
    
    args = parser.parse_args()

    if args.model:
        os.environ["MD_ACTIVE_MODEL"] = args.model
    
    participant_subset = None
    if args.participants:
        participant_subset = [p.strip() for p in args.participants.split(",") if p.strip()]
    if args.participant:
        participant_subset = participant_subset or []
        if args.participant not in participant_subset:
            participant_subset.append(args.participant)

    # Create pipeline instance
    pipeline = DataPreparationPipeline()
    if not args.disable_llm_telemetry:
        telemetry_path = make_telemetry_path(Path(args.output), stage="prep")
        pipeline.llm_client.configure_telemetry(
            telemetry_path,
            base_context={"stage": "prep", "meeting_id": args.meeting_id},
        )
        print(f"[LLM TELEMETRY] Writing {telemetry_path}")
    
    # Determine if processing single file or directory
    if os.path.isfile(args.input):
        if not args.meeting_id:
            # Try to extract meeting ID from filename
            filename = os.path.basename(args.input)
            if filename.endswith("_parsed.json"):
                args.meeting_id = filename.replace("_parsed.json", "")
            else:
                raise ValueError("Meeting ID required for single file processing")
        
        if args.mode == "legacy":
            result = pipeline.process_single_transcript(
                transcript_path=args.input,
                output_dir=args.output,
                meeting_id=args.meeting_id,
                profile_split_ratio=args.split_ratio,
                target_participant=args.participant,
                participant_subset=participant_subset,
                generate_test_cases=True,
                generate_eim_states=True,
                jit_enable_sliding_windows=args.jit_enable_sliding_windows,
                jit_window_size=args.jit_window_size,
                jit_window_stride=args.jit_window_stride,
                jit_emit_evidence_sidecar=args.jit_emit_evidence_sidecar,
                jit_enable_llm_cleanup=args.jit_enable_llm_cleanup,
                resume=args.resume,
                force=args.force,
            )
            if result["success"]:
                print("[DONE] Single transcript processing completed successfully!")
            else:
                print(f"[ERROR] Processing failed: {result.get('error', 'Unknown error')}")
                exit(1)
        else:
            # CAPA mode: generate profiles + JIT + CAPA episodes + CAPA EIM states
            result = pipeline.process_single_transcript(
                transcript_path=args.input,
                output_dir=args.output,
                meeting_id=args.meeting_id,
                profile_split_ratio=args.split_ratio,
                target_participant=args.participant,
                participant_subset=participant_subset,
                generate_test_cases=False,
                generate_eim_states=False,
                capa_N=args.capa_N,
                capa_k=args.capa_k,
                freeze_anchor_inventory=args.freeze_anchor_inventory,
                anchor_inventory_dir=args.anchor_inventory_dir,
                anchor_seed_k=args.anchor_seed_k,
                static_prep_source_dir=args.static_prep_source_dir,
                eim_cache_dir=args.eim_cache_dir,
                selector=args.selector,
                cap_meeting=args.cap_meeting,
                cap_per_participant=args.cap_per_participant,
                min_gap=args.min_gap,
                debug_episodes=args.debug_capa_episodes,
                jit_enable_sliding_windows=args.jit_enable_sliding_windows,
                jit_window_size=args.jit_window_size,
                jit_window_stride=args.jit_window_stride,
                jit_emit_evidence_sidecar=args.jit_emit_evidence_sidecar,
                jit_enable_llm_cleanup=args.jit_enable_llm_cleanup,
                resume=args.resume,
                force=args.force,
            )
            if not result["success"]:
                print(f"[ERROR] Processing failed: {result.get('error', 'Unknown error')}")
                exit(1)
            print("Generated CAPA data (episodes + EIM states)")
    
    elif os.path.isdir(args.input):
        # Process directory
        if args.mode == "legacy":
            results = pipeline.process_transcript_directory(
                input_dir=args.input,
                output_dir=args.output,
                profile_split_ratio=args.split_ratio
            )
            if results["failed_transcripts"] == 0:
                print("[DONE] Batch processing completed successfully!")
            else:
                print(f"[WARN] Batch processing completed with {results['failed_transcripts']} failures")
                exit(1)
        else:
            # CAPA mode: iterate over files and run profiles+JIT + CAPA episodes + CAPA EIM
            processed = 0
            for file_path in Path(args.input).glob("*_parsed.json"):
                meeting_id = file_path.stem.replace("_parsed", "")
                res = pipeline.process_single_transcript(
                    transcript_path=str(file_path),
                    output_dir=args.output,
                    meeting_id=meeting_id,
                    profile_split_ratio=args.split_ratio,
                    target_participant=args.participant,
                participant_subset=participant_subset,
                    generate_test_cases=False,
                    generate_eim_states=False,
                    capa_N=args.capa_N,
                    capa_k=args.capa_k,
                    freeze_anchor_inventory=args.freeze_anchor_inventory,
                    anchor_inventory_dir=args.anchor_inventory_dir,
                    anchor_seed_k=args.anchor_seed_k,
                    static_prep_source_dir=args.static_prep_source_dir,
                    eim_cache_dir=args.eim_cache_dir,
                    selector=args.selector,
                    cap_meeting=args.cap_meeting,
                    cap_per_participant=args.cap_per_participant,
                    min_gap=args.min_gap,
                    jit_enable_sliding_windows=args.jit_enable_sliding_windows,
                    jit_window_size=args.jit_window_size,
                    jit_window_stride=args.jit_window_stride,
                    jit_emit_evidence_sidecar=args.jit_emit_evidence_sidecar,
                    jit_enable_llm_cleanup=args.jit_enable_llm_cleanup,
                    resume=args.resume,
                    force=args.force,
                )
                if not res.get("success"):
                    print(f"[ERROR] Processing failed for {meeting_id}: {res.get('error','unknown')}")
                    continue
                print(f"Generated CAPA data for {meeting_id}")
                processed += 1
            print(f"[DONE] CAPA batch processed {processed} transcripts")
    
    else:
        print(f"[ERROR] Input path not found: {args.input}")
        exit(1)


if __name__ == "__main__":
    run_data_pipeline_cli()
