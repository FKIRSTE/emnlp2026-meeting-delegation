from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Set

from pydantic import ValidationError

from llm_client import LLMClient
from raw_context import (
    MonoRawPromptContext,
    RawModularPromptContext,
    build_eval_only_context,
    build_mono_raw_prompt_context,
    build_raw_modular_prompt_context,
    build_transcript_participant_roster,
    mono_raw_prompt_context_to_payload,
    raw_briefing_point_text,
)
from raw_baseline_support import prepare_raw_delegate_judge_context
from raw_prompt_bundle import (
    build_mono_raw_system_prompt,
    build_mono_raw_user_prompt,
    build_raw_controller_system_prompt,
    build_raw_controller_user_prompt,
    build_raw_coverage_instructions,
    build_raw_curator_instructions,
    build_raw_speaker_instructions,
)
from schemas import (
    ActionDetails,
    ControllerLLMOutput,
    ControllerOutput,
    DelegateActionLogEntry,
    DelegateResponse,
    GeneratorOutput,
    MonoRawLLMOutput,
    MonoRawReflectionCriticOutput,
)
from shared_memory import SharedMemoryModule
from two_tier_agent import TwoTierUserModule


class TranscriptOnlyAgentBase(TwoTierUserModule):
    """Shared runtime for transcript-only CAPA baselines."""

    raw_delegate_variant: str = ""
    fair_contract: bool = False

    def __init__(self, client: LLMClient):
        super().__init__(client)
        self.capa_enabled = False
        self._capa_driver = None
        self._capa_predictor = None
        self._capa_env_judge = None
        self._capa_delegate_judge = None
        self._capa_recalibrator = None
        self.last_controller_output: Optional[ControllerOutput] = None
        self.last_generator_output: Optional[GeneratorOutput] = None
        self.last_delegate_judge_payload: Optional[Dict[str, Any]] = None
        self.last_helper_context: Optional[Dict[str, Any]] = None
        self.last_helper_digest_data: Optional[List[Dict[str, Any]]] = None
        self.last_raw_context: Optional[RawModularPromptContext] = None
        self.last_mono_raw_prompt_context: Optional[MonoRawPromptContext] = None
        self.last_reflection_payload: Optional[Dict[str, Any]] = None
        self.last_reflection_output: Optional[Dict[str, Any]] = None
        self.last_reflection_memory_before: Optional[List[str]] = None
        self.last_reflection_memory_after: Optional[List[str]] = None
        self.raw_controller_system_prompt = build_raw_controller_system_prompt(fair_contract=self.fair_contract)
        self.mono_raw_system_prompt = build_mono_raw_system_prompt()

    def _build_profile_payload(self, smm: SharedMemoryModule) -> Dict[str, Any]:
        profile = smm.get_user_profile()
        if not profile or not hasattr(profile, "model_dump"):
            return {}
        payload = profile.model_dump()
        payload.pop("identity_aliases", None)
        return payload

    def _safe_json(self, payload: Any) -> str:
        if payload is None:
            return "null"
        try:
            return json.dumps(payload, indent=2, ensure_ascii=False)
        except Exception:
            return "null"

    def _format_jit_points(self, points: Sequence[str]) -> str:
        rows = [
            f"- {raw_briefing_point_text(point).strip()}"
            for point in points or []
            if raw_briefing_point_text(point).strip()
        ]
        return "\n".join(rows) if rows else "- None"

    def _format_action_log(self, entries: Sequence[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for entry in entries or []:
            cue = entry.get("cue_utterance_id") or "unknown"
            action = entry.get("action") or "Unknown"
            spoken = (entry.get("spoken_content") or "").strip()
            line = f"- cue={cue}; action={action}"
            if spoken:
                line += f"; spoken={spoken}"
            lines.append(line)
        return "\n".join(lines) if lines else "- None"

    def _call_structured_output(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: Any,
        max_tokens: int = 4000,
        temperature: float = 0.1,
    ) -> Optional[Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            response = self.client.call_with_retry(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                json_required=True,
                response_model=response_model,
                telemetry_label="raw_agent.structured_call",
                telemetry_phase="floor",
            )
            if not response:
                return None
            if hasattr(response, "model_dump"):
                return response
            return self.client.validate_json_response(response, response_model)
        except (ValidationError, Exception) as exc:
            print(f"  [RAW] Structured call failed: {exc}")
            return None

    def _build_digest_models(self, raw_list: Optional[Sequence[Any]]) -> List[ControllerOutput.CandidateDigestItem]:
        models: List[ControllerOutput.CandidateDigestItem] = []
        for item in raw_list or []:
            if isinstance(item, ControllerOutput.CandidateDigestItem):
                models.append(item)
                continue
            if not isinstance(item, dict):
                continue
            filtered = {
                "candidate_id": item.get("candidate_id"),
                "text": item.get("text"),
                "source": item.get("source"),
                "suitability": item.get("suitability"),
                "stance_function": item.get("stance_function"),
                "recommended_treatment": item.get("recommended_treatment"),
                "suggested_usage": item.get("suggested_usage"),
                "relevance_score": item.get("relevance_score"),
                "novelty_score": item.get("novelty_score"),
                "helper_notes": item.get("helper_notes"),
                "residual_contribution_value": item.get("residual_contribution_value"),
                "viability_status": item.get("viability_status"),
                "viability_reason": item.get("viability_reason"),
                "blocking_reason": item.get("blocking_reason"),
            }
            if filtered["candidate_id"] and filtered["text"]:
                try:
                    models.append(ControllerOutput.CandidateDigestItem(**filtered))
                except Exception:
                    continue
        return models

    def _inventory_digest(self, raw_context: RawModularPromptContext) -> List[Dict[str, Any]]:
        digest: List[Dict[str, Any]] = []
        for entry in raw_context.available_candidates or raw_context.candidate_inventory:
            coverage_status = dict(entry.get("coverage_status") or {})
            if coverage_status.get("covered_in_current_episode"):
                treatment = "Skip"
            elif coverage_status.get("covered_in_previous_episodes"):
                treatment = "Reinforce"
            else:
                treatment = "Fresh"
            digest_entry = {
                    "candidate_id": entry.get("candidate_id"),
                    "text": entry.get("text"),
                    "core_text": entry.get("core_text"),
                    "source": entry.get("source"),
                    "inventory_tier": entry.get("inventory_tier"),
                    "stance_function": entry.get("stance_function"),
                    "suitability": "Medium",
                    "recommended_treatment": treatment,
                    "suggested_usage": "Skip" if treatment == "Skip" else ("Reinforce" if treatment == "Reinforce" else "CorePoint"),
                    "relevance_score": None,
                    "novelty_score": None,
                    "helper_notes": f"Transcript-only inventory candidate ({entry.get('inventory_tier', 'primary')}).",
                    "coverage_status": coverage_status,
                    "composite_score": None,
            }
            if self.fair_contract:
                digest_entry.update(self._candidate_viability_details(digest_entry))
            digest.append(digest_entry)
        return digest

    def _find_candidate(self, candidate_id: Optional[str], helper_digest_data: Optional[Sequence[Any]]) -> Optional[Dict[str, Any]]:
        if not candidate_id:
            return None
        for item in helper_digest_data or []:
            if isinstance(item, dict) and item.get("candidate_id") == candidate_id:
                return item
        return None

    def _hydrate_controller_output(
        self,
        llm_output: Any,
        helper_digest_data: Optional[List[Dict[str, Any]]],
        *,
        speaker_output: Optional[Dict[str, Any]] = None,
        banned_ids: Optional[Set[str]] = None,
    ) -> ControllerOutput:
        banned_ids = banned_ids or set()
        if isinstance(llm_output, ControllerOutput):
            controller_output = llm_output
        else:
            controller_output = ControllerOutput(
                action_details=llm_output.action_details,
                content_plan=llm_output.content_plan,
                active_candidate_id=llm_output.active_candidate_id,
                candidate_decision_trace=getattr(llm_output, "candidate_decision_trace", None),
            )

        digest_models = self._build_digest_models(helper_digest_data)
        if digest_models:
            controller_output.helper_candidates_digest = digest_models

        if controller_output.active_candidate_id and not self._find_candidate(controller_output.active_candidate_id, helper_digest_data):
            controller_output.active_candidate_id = None

        if controller_output.active_candidate_id and controller_output.active_candidate_id in banned_ids:
            controller_output.action_details.action_name = "Remain_Silent"
            controller_output.content_plan = None
            controller_output.active_candidate_id = None
            note = "Hard override: candidate was already covered in the current transcript/episode."
            controller_output.candidate_decision_trace = (
                f"{controller_output.candidate_decision_trace} {note}".strip()
                if controller_output.candidate_decision_trace
                else note
            )

        candidate = self._find_candidate(controller_output.active_candidate_id, helper_digest_data)
        if candidate:
            controller_output.selected_point = candidate.get("text")
            controller_output.selection_reason = controller_output.candidate_decision_trace or "Selected the active raw candidate."
        elif controller_output.content_plan and controller_output.content_plan.knowledge_to_convey and not controller_output.selected_point:
            controller_output.selected_point = controller_output.content_plan.knowledge_to_convey
            controller_output.selection_reason = controller_output.candidate_decision_trace or "Used content_plan.knowledge_to_convey as the selected point."

        if speaker_output:
            ranking = speaker_output.get("speaker_ranking") or []
            if ranking:
                controller_output.speaker_recommendation = ranking[0].get("speaker")
            elif speaker_output.get("recommended_speaker"):
                controller_output.speaker_recommendation = speaker_output.get("recommended_speaker")
            confidence = speaker_output.get("confidence")
            if confidence is not None:
                try:
                    controller_output.speaker_recommendation_confidence = float(confidence)
                except (TypeError, ValueError):
                    pass

        action_name = (controller_output.action_details.action_name or "").strip()
        if action_name == "Remain_Silent":
            controller_output.helper_global_suggestion = "Remain_Silent"
        elif action_name == "Acknowledge":
            controller_output.helper_global_suggestion = "Acknowledge"
        else:
            controller_output.helper_global_suggestion = "Speak"

        return controller_output

    def _build_silent_controller_output(
        self,
        reason: str,
        helper_digest_data: Optional[List[Dict[str, Any]]],
    ) -> ControllerOutput:
        controller_output = ControllerOutput(
            content_plan=None,
            action_details=ActionDetails(
                action_name="Remain_Silent",
                scope="Concise",
                justification=reason,
            ),
            candidate_decision_trace=reason,
        )
        digest_models = self._build_digest_models(helper_digest_data)
        if digest_models:
            controller_output.helper_candidates_digest = digest_models
        controller_output.helper_global_suggestion = "Remain_Silent"
        return controller_output

    def _prepare_delegate_judge_context(
        self,
        *,
        smm: SharedMemoryModule,
        participant_id: Optional[str],
        snapshot_utterances: List[Dict[str, Any]],
        helper_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        evaluation_context = build_eval_only_context(
            smm=smm,
            snapshot_utterances=snapshot_utterances,
            participant_id=participant_id or "Delegate",
        )
        return prepare_raw_delegate_judge_context(
            delegate_variant=self.raw_delegate_variant,
            evaluation_context=evaluation_context,
            helper_context=helper_context,
        )

    def _handle_episode_boundary(self, smm: SharedMemoryModule, episode: Optional[object]) -> None:
        previous_episode = self._current_episode_id
        self._update_episode_boundary(episode)
        if episode and previous_episode is not None and self._current_episode_id != previous_episode:
            smm.reset_episode_tracking()
            smm.recalibration_guidance = None

    def _finalize_turn_state(
        self,
        *,
        smm: SharedMemoryModule,
        participant_id: Optional[str],
        snapshot_utterances: List[Dict[str, Any]],
        gt_next_window: Optional[List[Dict[str, Any]]],
        controller_output: ControllerOutput,
        generator_output: GeneratorOutput,
        helper_context: Dict[str, Any],
        helper_digest_data: Optional[List[Dict[str, Any]]],
        raw_context: Optional[RawModularPromptContext],
    ) -> DelegateResponse:
        helper_context = self._prepare_delegate_judge_context(
            smm=smm,
            participant_id=participant_id,
            snapshot_utterances=snapshot_utterances,
            helper_context=helper_context,
        )

        if self.artifact_tracker:
            self.artifact_tracker.capture_controller_output(controller_output)
            try:
                if gt_next_window:
                    self.artifact_tracker.capture_step_context(gt_next_window, participant_id)
                self.artifact_tracker.capture_delegate_utterance(generator_output.utterance if generator_output else "")
            except Exception:
                pass

        delegate_judge_payload = self._build_delegate_judge_payload(
            controller_output,
            helper_digest_data,
            participant_id=participant_id,
            snapshot_utterances=snapshot_utterances,
            helper_context=helper_context,
        )

        self.last_controller_output = controller_output
        self.last_generator_output = generator_output
        self.last_delegate_judge_payload = delegate_judge_payload
        self.last_helper_context = helper_context
        self.last_helper_digest_data = helper_digest_data
        self.last_raw_context = raw_context

        final_response = generator_output.utterance if generator_output else None
        delegate_response = DelegateResponse(
            thoughts=controller_output.action_details.justification,
            response=final_response,
        )

        if controller_output.action_details.action_name != "Remain_Silent":
            anchor_utt_id = None
            if gt_next_window and gt_next_window[0]:
                anchor_utt_id = gt_next_window[0].get("utterance_id")
            if not anchor_utt_id and snapshot_utterances:
                anchor_utt_id = snapshot_utterances[-1].get("utterance_id")

            candidate = self._find_candidate(controller_output.active_candidate_id, helper_digest_data)
            if candidate and anchor_utt_id:
                self._update_candidate_coverage(
                    candidate_id=controller_output.active_candidate_id,
                    candidate_text=candidate.get("text") or "",
                    utterance_id=anchor_utt_id,
                    source=candidate.get("source"),
                )

            smm.log_delegate_action(
                DelegateActionLogEntry(
                    utterance_id_of_action=anchor_utt_id or "unknown_cue",
                    action_taken=controller_output.action_details.action_name,
                    point_made_summary=final_response,
                )
            )

        return delegate_response


class RawModularAgent(TranscriptOnlyAgentBase):
    """Transcript-only modular delegate with helper stages and controller/generator split."""

    raw_delegate_variant = "raw-modular"
    fair_contract = False

    FORBIDDEN_FAIR_PAYLOAD_KEYS = {
        "eim_state",
        "snapshot_analysis",
        "micro_topic_analysis",
        "topic_summary_log",
        "participant_models",
        "identity_aliases",
        "recalibration_guidance",
        "gt_next_window",
        "idea_targets",
        "official_idea_targets",
        "target_response_ids",
        "current_episode_target_response_ids",
        "support_utt_ids",
        "anchor_phrases",
        "context_utt_ids",
        "framing",
    }

    def _raw_curator_instructions(self) -> str:
        return build_raw_curator_instructions(fair_contract=self.fair_contract)

    def _raw_coverage_instructions(self) -> str:
        return build_raw_coverage_instructions(fair_contract=self.fair_contract)

    def _raw_speaker_instructions(self) -> str:
        return build_raw_speaker_instructions(fair_contract=self.fair_contract)

    def _find_forbidden_fair_payload_keys(self, payload: Any, path: str = "$") -> List[str]:
        if not self.fair_contract:
            return []
        found: List[str] = []
        if isinstance(payload, dict):
            for key, value in payload.items():
                key_text = str(key)
                next_path = f"{path}.{key_text}"
                if key_text in self.FORBIDDEN_FAIR_PAYLOAD_KEYS:
                    found.append(next_path)
                found.extend(self._find_forbidden_fair_payload_keys(value, next_path))
        elif isinstance(payload, list):
            for idx, value in enumerate(payload):
                found.extend(self._find_forbidden_fair_payload_keys(value, f"{path}[{idx}]"))
        return found

    def _assert_no_fair_payload_leakage(self, payload: Any, *, label: str) -> None:
        leaks = self._find_forbidden_fair_payload_keys(payload)
        if leaks:
            raise ValueError(f"raw-modular-fair payload leakage in {label}: {', '.join(leaks[:8])}")

    def _sanitize_raw_fair_payload(self, payload: Any) -> Any:
        if not self.fair_contract:
            return payload
        if isinstance(payload, dict):
            return {
                key: self._sanitize_raw_fair_payload(value)
                for key, value in payload.items()
                if str(key) not in self.FORBIDDEN_FAIR_PAYLOAD_KEYS
            }
        if isinstance(payload, list):
            return [self._sanitize_raw_fair_payload(value) for value in payload]
        return payload

    def _build_raw_curator_payload(self, smm: SharedMemoryModule, raw_context: RawModularPromptContext) -> Dict[str, Any]:
        payload = {
            "helper_name": "candidate_curator",
            "instructions": self._raw_curator_instructions(),
            "output_schema": {
                "ranked_candidates": [
                    {
                        "candidate_id": "string",
                        "text": "string",
                        "source": "JIT|Profile",
                        "suitability": "High|Medium|Low",
                        "relevance_score": "1-10 integer",
                        "novelty_score": "1-10 integer",
                        "intent_alignment_score": "1-10 integer (how directly the candidate answers the current ask or meeting need)",
                        "intent_alignment_factor": "0.5|0.8|1.2|1.5 (multiplier based on how directly the candidate addresses the current ask or need)",
                        "composite_score": "number (ranking score computed from relevance, novelty, and intent alignment)",
                        "suggested_usage": "CorePoint|Reinforce|Acknowledge|Skip (how central this candidate should be if used this turn)",
                        "helper_notes": "short string",
                        "recommended_treatment": (
                            "Fresh|ExpandOnOthers|Partial|Reinforce|Acknowledge|Skip (best treatment for this candidate in the current moment)"
                            if self.fair_contract
                            else "Fresh|Reinforce|Acknowledge|Skip (best treatment for this candidate in the current moment)"
                        ),
                        "residual_contribution_value": "High|Medium|Low|None (required for Partial/ExpandOnOthers in fair mode)",
                        "stance_function": "support|proposal|constraint|caution|question|facilitation|neutral",
                    }
                ],
                "commentary": "short string",
            },
            "context": {
                "represented_participant_id": raw_context.participant_id,
                "profile": self._build_profile_payload(smm),
                "delegate_action_log": raw_context.delegate_action_log,
                "conversation_window": raw_context.conversation_window,
                "candidate_inventory": raw_context.available_candidates,
                "covered_candidates_readonly": raw_context.covered_candidates_readonly,
            },
        }
        self._assert_no_fair_payload_leakage(payload, label="candidate_curator")
        return payload

    def _build_raw_coverage_payload(
        self,
        raw_context: RawModularPromptContext,
        curator_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = {
            "helper_name": "coverage_evaluator",
            "instructions": self._raw_coverage_instructions(),
            "output_schema": {
                "annotated_candidates": [
                    {
                        "candidate_id": "string (candidate being evaluated)",
                        "coverage_by_delegate": {
                            "in_transcript_window": "bool (true if the delegate already made the same point in the current transcript window)",
                            "in_delegate_history": "bool (true if the delegate already made the same point in recent delegate history)",
                            "in_current_delegate_episode": "bool (true if candidate_coverage_flags show the delegate already used this point earlier in the current delegate episode)",
                            "in_previous_delegate_episodes": "bool (true if candidate_coverage_flags show the delegate used this point in previous delegate episodes)",
                        },
                        "coverage_by_others": {
                            "in_transcript_window": "bool (true if another participant already made the same point in the current transcript window)",
                        },
                        "coverage_completeness": "none|partial|full (how fully the point is already covered)",
                        "covered_aspects": ["string (aspect already covered)"],
                        "uncovered_aspects": ["string (aspect still available to add)"],
                        "residual_contribution_value": "High|Medium|Low|None (remaining value if the delegate still speaks)",
                        "contribution_angle": "short string (best angle if the delegate still contributes)",
                        "framing_suggestion": "short string (how to frame the contribution)",
                        "expand_on_reference": {
                            "speaker": "string (speaker being expanded on)",
                            "utterance_id": "string (utterance in the current transcript window being expanded on)",
                        },
                        "recommended_treatment": (
                            "Fresh|ExpandOnOthers|Partial|Reinforce|Acknowledge|Skip (best treatment for this candidate now)"
                            if self.fair_contract
                            else "Fresh|Reinforce|Acknowledge|Skip (best treatment for this candidate now)"
                        ),
                        "coverage_explanation": "short string (why this treatment is appropriate)",
                        "requires_historical_framing": "bool (true if a reinforcement should acknowledge prior delegate coverage)",
                        "suggested_framing_approaches": ["string (brief framing suggestion)"],
                    }
                ],
                "confidence": "0..1 float (overall confidence in the coverage analysis)",
                "global_notes": "short string (optional overall note)",
            },
            "context": {
                "represented_participant_id": raw_context.participant_id,
                "conversation_window": raw_context.conversation_window,
                "ranked_candidates": curator_output.get("ranked_candidates") or [],
                "delegate_action_log": raw_context.delegate_action_log,
                "candidate_coverage_flags": raw_context.candidate_coverage_flags,
            },
        }
        self._assert_no_fair_payload_leakage(payload, label="coverage_evaluator")
        return payload

    def _build_raw_speaker_payload(
        self,
        raw_context: RawModularPromptContext,
        delegate_content_summary: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        payload = {
            "helper_name": "speaker_scorer",
            "instructions": self._raw_speaker_instructions(),
            "output_schema": {
                "speaker_ranking": [
                    {
                        "speaker": "participant_id",
                        "score": "0..1 float",
                        "match_level": "MICRO|MESO|MACRO|NONE (how closely the speaker or candidate fits the current discussion level)",
                        "topic_engagement_role": "current_owner|active|adjacent|idle (how centrally this speaker is engaged in the current topic flow)",
                        "is_delegate": "bool",
                    }
                ],
                "delegate_rank_position": "integer",
                "delegate_viability": {
                    "has_viable_candidate": "bool",
                    "match_level": "MICRO|MESO|MACRO|NONE",
                    "best_viable_fit": "HIGH|MEDIUM|LOW|NONE (overall quality of the delegate's best viable candidate given content fit and timing)",
                    "viable_candidate_count": "integer",
                },
                "overall_assessment": "Proceed|Deferral|Acknowledge (overall speaker-side assessment of whether the delegate is timely to speak)",
                "confidence": "0..1 float",
                "rationale": "short string",
                "speak_or_silent_hint": "Proceed|Deferral|Acknowledge (practical recommendation on whether the delegate should speak now)",
                "notes": "short string",
            },
            "context": {
                "represented_participant_id": raw_context.participant_id,
                "participant_roster": build_transcript_participant_roster(raw_context.conversation_window),
                "delegate_content_summary": delegate_content_summary,
                "delegate_action_log": raw_context.delegate_action_log,
                "conversation_window": raw_context.conversation_window,
            },
        }
        self._assert_no_fair_payload_leakage(payload, label="speaker_scorer")
        return payload

    @staticmethod
    def _coerce_helper_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "1"}:
                return True
            if lowered in {"false", "no", "0", "", "none", "null"}:
                return False
        return bool(value)

    def _extract_raw_coverage_flag(self, payload: Any, *keys: str) -> bool:
        if not isinstance(payload, dict):
            return False
        for key in keys:
            if key in payload:
                return self._coerce_helper_bool(payload.get(key))
        return False

    def _normalize_raw_coverage_output(self, coverage_output: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not isinstance(coverage_output, dict):
            return coverage_output

        normalized_candidates: List[Dict[str, Any]] = []
        for item in coverage_output.get("annotated_candidates") or []:
            if not isinstance(item, dict):
                continue

            coverage_by_delegate = item.get("coverage_by_delegate") or {}
            coverage_by_others = item.get("coverage_by_others") or {}

            normalized_item = dict(item)
            normalized_item["coverage_by_delegate"] = {
                "in_snapshot": self._extract_raw_coverage_flag(coverage_by_delegate, "in_transcript_window", "in_snapshot"),
                "in_topic_log": self._extract_raw_coverage_flag(coverage_by_delegate, "in_delegate_history", "in_topic_log"),
                "in_current_episode": self._extract_raw_coverage_flag(coverage_by_delegate, "in_current_delegate_episode", "in_current_episode"),
                "in_previous_episodes": self._extract_raw_coverage_flag(coverage_by_delegate, "in_previous_delegate_episodes", "in_previous_episodes"),
            }
            normalized_item["coverage_by_others"] = {
                "in_snapshot": self._extract_raw_coverage_flag(coverage_by_others, "in_transcript_window", "in_snapshot"),
                "in_topic_log": self._extract_raw_coverage_flag(coverage_by_others, "in_topic_log"),
            }
            normalized_candidates.append(normalized_item)

        normalized_output = dict(coverage_output)
        normalized_output["annotated_candidates"] = normalized_candidates
        return normalized_output

    def _build_raw_helper_digest(
        self,
        raw_context: RawModularPromptContext,
        curator_output: Optional[Dict[str, Any]],
        coverage_output: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not curator_output or not curator_output.get("ranked_candidates"):
            return self._inventory_digest(raw_context)

        coverage_by_id = {
            item.get("candidate_id"): item
            for item in (coverage_output or {}).get("annotated_candidates", []) or []
            if isinstance(item, dict) and item.get("candidate_id")
        }
        inventory_by_id = {entry.get("candidate_id"): entry for entry in raw_context.candidate_inventory}
        digest: List[Dict[str, Any]] = []
        for item in curator_output.get("ranked_candidates") or []:
            if not isinstance(item, dict) or not item.get("candidate_id"):
                continue
            candidate_id = item.get("candidate_id")
            inventory_entry = inventory_by_id.get(candidate_id, {})
            coverage_entry = coverage_by_id.get(candidate_id, {})
            coverage_status = dict(inventory_entry.get("coverage_status") or {})
            recommended_treatment = (
                coverage_entry.get("recommended_treatment")
                or item.get("recommended_treatment")
                or ("Skip" if coverage_status.get("covered_in_current_episode") else ("Reinforce" if coverage_status.get("covered_in_previous_episodes") else "Fresh"))
            )
            digest_entry = {
                    "candidate_id": candidate_id,
                    "text": item.get("text") or inventory_entry.get("text"),
                    "core_text": inventory_entry.get("core_text") or item.get("text"),
                    "source": item.get("source") or inventory_entry.get("source"),
                    "inventory_tier": inventory_entry.get("inventory_tier"),
                    "stance_function": item.get("stance_function") or inventory_entry.get("stance_function"),
                    "suitability": item.get("suitability") or "Medium",
                    "recommended_treatment": recommended_treatment,
                    "suggested_usage": item.get("suggested_usage") or ("Skip" if recommended_treatment == "Skip" else ("Reinforce" if recommended_treatment == "Reinforce" else "CorePoint")),
                    "relevance_score": item.get("relevance_score"),
                    "novelty_score": item.get("novelty_score"),
                    "helper_notes": coverage_entry.get("coverage_explanation") or item.get("helper_notes"),
                    "coverage_status": coverage_status,
                    "composite_score": item.get("composite_score"),
                    "residual_contribution_value": coverage_entry.get("residual_contribution_value") or item.get("residual_contribution_value"),
                    "coverage": coverage_entry if coverage_entry else None,
            }
            if self.fair_contract:
                digest_entry.update(self._candidate_viability_details(digest_entry))
            digest.append(digest_entry)
        return digest

    def _run_raw_helper_sequence(
        self,
        smm: SharedMemoryModule,
        raw_context: RawModularPromptContext,
    ) -> Dict[str, Any]:
        curator_payload = self._build_raw_curator_payload(smm, raw_context)
        curator_output = self._invoke_helper_tool("candidate_curator", curator_payload)

        coverage_payload = None
        coverage_output = None
        if curator_output and curator_output.get("ranked_candidates"):
            coverage_payload = self._build_raw_coverage_payload(raw_context, curator_output)
            coverage_output = self._invoke_helper_tool("coverage_evaluator", coverage_payload)
            coverage_output = self._normalize_raw_coverage_output(coverage_output)

        helper_digest = self._build_raw_helper_digest(raw_context, curator_output, coverage_output)
        delegate_content_summary = self._build_delegate_content_summary(helper_digest)

        speaker_payload = self._build_raw_speaker_payload(raw_context, delegate_content_summary)
        speaker_output = self._invoke_helper_tool("speaker_scorer", speaker_payload)

        banned_ids: Set[str]
        ban_reasons: Dict[str, str]
        if coverage_output:
            banned_ids, ban_reasons = self._compute_banned_candidates(coverage_output)
        else:
            banned_ids, ban_reasons = set(), {}

        return {
            "_raw_context": raw_context,
            "curator_payload": curator_payload,
            "curator_output": curator_output,
            "coverage_payload": coverage_payload,
            "coverage_output": coverage_output,
            "speaker_payload": speaker_payload,
            "speaker_output": speaker_output,
            "helper_candidates_digest": helper_digest,
            "banned_candidate_ids": list(banned_ids),
            "ban_reasons": ban_reasons,
        }

    def controller_phase(
        self,
        smm: SharedMemoryModule,
        snapshot_utterances: List[Dict],
        participant_id: Optional[str] = None,
        helper_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[ControllerOutput]:
        profile = smm.get_user_profile()
        raw_context = (helper_context or {}).get("_raw_context") if isinstance(helper_context, dict) else None
        if not profile or raw_context is None:
            print("  [RAW] Controller failed: missing profile or raw context.")
            return None

        helper_context = helper_context or {}
        helper_digest_data = helper_context.get("helper_candidates_digest") or []
        helper_summary = self._build_helper_summary(
            curator_output=helper_context.get("curator_output"),
            coverage_output=helper_context.get("coverage_output"),
            speaker_output=helper_context.get("speaker_output"),
            helper_digest_data=helper_digest_data,
            banned_candidate_ids=set(helper_context.get("banned_candidate_ids") or []),
        )
        helper_summary = self._sanitize_raw_fair_payload(helper_summary)

        candidate_inventory_payload = {
            "available_candidates": raw_context.available_candidates,
            "covered_candidates_readonly": raw_context.covered_candidates_readonly,
            "candidate_coverage_flags": raw_context.candidate_coverage_flags,
        }
        candidate_inventory_payload = self._sanitize_raw_fair_payload(candidate_inventory_payload)

        user_prompt = build_raw_controller_user_prompt(
            participant_id=participant_id or "Delegate",
            profile_json=self._safe_json(self._build_profile_payload(smm)),
            action_log_text=self._format_action_log(raw_context.delegate_action_log),
            candidate_inventory_json=self._safe_json(candidate_inventory_payload),
            helper_summary_json=self._safe_json(helper_summary),
            conversation_window_text=self._format_utterances_for_prompt(snapshot_utterances),
            fair_contract=self.fair_contract,
        )
        self._assert_no_fair_payload_leakage(
            {
                "candidate_inventory": candidate_inventory_payload,
                "helper_summary": helper_summary,
                "raw_context": raw_context.__dict__,
            },
            label="controller_user_prompt",
        )

        llm_output = self._call_structured_output(
            system_prompt=self.raw_controller_system_prompt,
            user_prompt=user_prompt,
            response_model=ControllerLLMOutput,
            max_tokens=4000,
            temperature=0.1,
        )
        if not llm_output:
            return None
        controller_output = self._hydrate_controller_output(
            llm_output,
            helper_digest_data,
            speaker_output=helper_context.get("speaker_output"),
            banned_ids=set(helper_context.get("banned_candidate_ids") or []),
        )
        if self.fair_contract:
            self._apply_raw_fair_controller_guards(
                controller_output,
                helper_summary=helper_summary,
                helper_digest_data=helper_digest_data,
                helper_context=helper_context,
                snapshot_utterances=snapshot_utterances,
                participant_id=participant_id,
                base_messages=[
                    {"role": "system", "content": self.raw_controller_system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        return controller_output

    def _apply_raw_fair_controller_guards(
        self,
        controller_output: ControllerOutput,
        *,
        helper_summary: Dict[str, Any],
        helper_digest_data: Optional[List[Dict[str, Any]]],
        helper_context: Dict[str, Any],
        snapshot_utterances: List[Dict],
        participant_id: Optional[str],
        base_messages: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        inconsistent, best_candidate, veto_reason = self._controller_silence_inconsistent(
            controller_output,
            helper_summary=helper_summary,
            snapshot_utterances=snapshot_utterances,
            participant_id=participant_id,
        )
        if inconsistent and best_candidate and base_messages:
            try:
                repair_prompt = (
                    "Raw-modular-fair controller consistency check: you chose Remain_Silent, "
                    "but the raw-transcript helper outputs contain a viable candidate and no concrete "
                    "raw-transcript silence veto was found. Re-evaluate once. If you remain silent, cite "
                    "a concrete veto from the visible transcript such as represented participant just spoke, "
                    "floor directed elsewhere, all candidates low fit, unsupported/repetitive content, or strong "
                    "speaker-scorer deferral grounded in raw transcript evidence.\n\n"
                    f"Best viable candidate:\n{json.dumps(best_candidate, ensure_ascii=False, indent=2)}"
                )
                repair_response = self.client.call_with_retry(
                    model=self.model_name,
                    messages=base_messages + [{"role": "user", "content": repair_prompt}],
                    max_tokens=2200,
                    temperature=0.1,
                    json_required=True,
                    response_model=ControllerLLMOutput,
                    telemetry_label="raw_agent.controller_consistency_retry",
                    telemetry_phase="floor",
                )
                if repair_response:
                    repaired = self._hydrate_controller_output(
                        repair_response,
                        helper_digest_data,
                        speaker_output=helper_context.get("speaker_output"),
                        banned_ids=set(helper_context.get("banned_candidate_ids") or []),
                    )
                    repaired_inconsistent, _, repaired_veto = self._controller_silence_inconsistent(
                        repaired,
                        helper_summary=helper_summary,
                        snapshot_utterances=snapshot_utterances,
                        participant_id=participant_id,
                    )
                    if not repaired_inconsistent:
                        controller_output.action_details = repaired.action_details
                        controller_output.content_plan = repaired.content_plan
                        controller_output.active_candidate_id = repaired.active_candidate_id
                        controller_output.candidate_decision_trace = repaired.candidate_decision_trace
                        controller_output.selected_point = repaired.selected_point
                        controller_output.selection_reason = repaired.selection_reason
                        if repaired_veto:
                            note = f"Raw fair retry retained silence with veto: {repaired_veto}."
                            controller_output.candidate_decision_trace = (
                                f"{controller_output.candidate_decision_trace} {note}".strip()
                                if controller_output.candidate_decision_trace
                                else note
                            )
            except Exception as err:
                print(f"  [RAW] Raw fair controller consistency retry failed: {err}")

        inconsistent, best_candidate, _ = self._controller_silence_inconsistent(
            controller_output,
            helper_summary=helper_summary,
            snapshot_utterances=snapshot_utterances,
            participant_id=participant_id,
        )
        if inconsistent and best_candidate:
            self._apply_safe_controller_fallback(
                controller_output,
                best_candidate=best_candidate,
                snapshot_utterances=snapshot_utterances,
                participant_id=participant_id,
            )
        try:
            self._apply_participant_stance_consistency_guard(
                controller_output,
                helper_summary=helper_summary,
            )
        except Exception:
            pass
        action_name = (controller_output.action_details.action_name or "").strip()
        if action_name == "Remain_Silent":
            controller_output.helper_global_suggestion = "Remain_Silent"
        elif action_name == "Acknowledge":
            controller_output.helper_global_suggestion = "Acknowledge"
        else:
            controller_output.helper_global_suggestion = "Speak"

    def run_turn(
        self,
        smm: SharedMemoryModule,
        snapshot_utterances: List[Dict],
        participant_id: Optional[str] = None,
        gt_next_window: Optional[List[Dict]] = None,
        episode: Optional[object] = None,
        delta_index: int = 0,
        prompt_snapshot_utterances: Optional[List[Dict]] = None,
    ) -> Optional[DelegateResponse]:
        print("  [RAW] Starting raw-modular turn execution...")
        participant_id = participant_id or "Delegate"
        self._handle_episode_boundary(smm, episode)
        prompt_snapshot = list(prompt_snapshot_utterances or snapshot_utterances)

        raw_context = build_raw_modular_prompt_context(
            smm=smm,
            snapshot_utterances=prompt_snapshot,
            participant_id=participant_id,
            coverage_flags=self.candidate_coverage_flags,
            fair_projection=self.fair_contract,
        )
        helper_context: Dict[str, Any] = {
            "_raw_context": raw_context,
            "helper_candidates_digest": self._inventory_digest(raw_context),
        }
        helper_digest_data = helper_context["helper_candidates_digest"]

        helper_context = self._run_raw_helper_sequence(smm, raw_context)
        helper_digest_data = helper_context.get("helper_candidates_digest") or helper_digest_data
        controller_output = self.controller_phase(smm, prompt_snapshot, participant_id, helper_context)
        if not controller_output:
            controller_output = self._build_silent_controller_output(
                "Controller phase failed; remaining silent.",
                helper_digest_data,
            )

        if controller_output.action_details.action_name != "Remain_Silent":
            profile = smm.get_user_profile()
            generator_output = self.generator_phase(controller_output, profile, prompt_snapshot) if profile else None
            if not generator_output:
                generator_output = GeneratorOutput(utterance="")
        else:
            generator_output = GeneratorOutput(utterance="")

        return self._finalize_turn_state(
            smm=smm,
            participant_id=participant_id,
            snapshot_utterances=prompt_snapshot,
            gt_next_window=gt_next_window,
            controller_output=controller_output,
            generator_output=generator_output,
            helper_context=helper_context,
            helper_digest_data=helper_digest_data,
            raw_context=raw_context,
        )


class RawModularFairAgent(RawModularAgent):
    """Fair raw-transcript modular ablation with CAPA-like contracts and no EIM runtime state."""

    raw_delegate_variant = "raw-modular-fair"
    fair_contract = True


class MonoRawAgent(TranscriptOnlyAgentBase):
    """Single-call transcript-only delegate for the system-claim baseline."""

    raw_delegate_variant = "mono-raw"

    def _get_reflection_memory_mode(self) -> str:
        return "v1"

    def _get_reflection_memory_text(self, episode: Optional[object], participant_id: str) -> Optional[str]:
        return None

    def run_turn(
        self,
        smm: SharedMemoryModule,
        snapshot_utterances: List[Dict],
        participant_id: Optional[str] = None,
        gt_next_window: Optional[List[Dict]] = None,
        episode: Optional[object] = None,
        delta_index: int = 0,
        prompt_snapshot_utterances: Optional[List[Dict]] = None,
    ) -> Optional[DelegateResponse]:
        print("  [RAW] Starting mono-raw turn execution...")
        participant_id = participant_id or "Delegate"
        self._handle_episode_boundary(smm, episode)

        prompt_snapshot = list(prompt_snapshot_utterances or snapshot_utterances)
        prompt_context = build_mono_raw_prompt_context(
            smm=smm,
            snapshot_utterances=prompt_snapshot,
            participant_id=participant_id,
        )
        prompt_payload = mono_raw_prompt_context_to_payload(prompt_context)
        helper_context: Dict[str, Any] = {}
        helper_digest_data: List[Dict[str, Any]] = []
        self.last_mono_raw_prompt_context = prompt_context

        if self.artifact_tracker:
            self.artifact_tracker.capture_helper_io("mono_raw_prompt_context", prompt_payload, None)

        user_prompt = build_mono_raw_user_prompt(
            participant_id=participant_id,
            profile_text=prompt_context.profile_text,
            jit_briefing_text=self._format_jit_points(prompt_context.jit_briefing_points),
            action_log_text=self._format_action_log(prompt_context.delegate_action_log),
            conversation_window_text=self._format_utterances_for_prompt(prompt_snapshot),
            reflection_memory_text=self._get_reflection_memory_text(episode, participant_id),
            reflection_memory_mode=self._get_reflection_memory_mode(),
        )

        llm_output = self._call_structured_output(
            system_prompt=self.mono_raw_system_prompt,
            user_prompt=user_prompt,
            response_model=MonoRawLLMOutput,
            max_tokens=5000,
            temperature=0.15,
        )
        if not llm_output:
            controller_output = self._build_silent_controller_output(
                "Mono-raw call failed; remaining silent.",
                helper_digest_data,
            )
            generator_output = GeneratorOutput(utterance="")
            return self._finalize_turn_state(
                smm=smm,
                participant_id=participant_id,
                snapshot_utterances=prompt_snapshot,
                gt_next_window=gt_next_window,
                controller_output=controller_output,
                generator_output=generator_output,
                helper_context=helper_context,
                helper_digest_data=helper_digest_data,
                raw_context=None,
            )

        controller_output = self._hydrate_controller_output(
            llm_output,
            helper_digest_data,
            banned_ids=set(),
        )
        response_text = (llm_output.response or "").strip() if getattr(llm_output, "response", None) else ""
        if not controller_output.selected_point:
            if controller_output.content_plan and controller_output.content_plan.knowledge_to_convey:
                controller_output.selected_point = controller_output.content_plan.knowledge_to_convey
                controller_output.selection_reason = controller_output.selection_reason or controller_output.candidate_decision_trace
            elif response_text:
                controller_output.selected_point = response_text
                controller_output.selection_reason = controller_output.selection_reason or controller_output.candidate_decision_trace or "Used generated response as the selected point."
        if controller_output.action_details.action_name == "Remain_Silent":
            generator_output = GeneratorOutput(utterance="")
        else:
            generator_output = GeneratorOutput(utterance=response_text)

        return self._finalize_turn_state(
            smm=smm,
            participant_id=participant_id,
            snapshot_utterances=prompt_snapshot,
            gt_next_window=gt_next_window,
            controller_output=controller_output,
            generator_output=generator_output,
            helper_context=helper_context,
            helper_digest_data=helper_digest_data,
            raw_context=None,
        )


class MonoRawReflectionAgent(MonoRawAgent):
    """Transcript-only delegate with scoped post-action procedural reflection memory."""

    raw_delegate_variant = "mono-raw-reflection"
    REFLECTION_MEMORY_LESSON_COUNT_FOR_CURATION = 10
    REFLECTION_MEMORY_TOKEN_CHARS_FOR_CURATION = 4000

    STRUCTURAL_LEAKAGE_PATTERNS = (
        re.compile(r"\b(?:utterance_id|response_utterance_ids|gt_next_window|official_idea|official target|target_response_ids)\b", re.I),
        re.compile(r"\b(?:strict recall|loose recall|recall credit|ground truth|delegate_judge|matched_unit_id|capa_step_successful)\b", re.I),
        re.compile(r"\b(?:test_case_id|episode_id|anchor_utt_id|current_episode_target_response_ids)\b", re.I),
        re.compile(r"\bMeeting\d+(?:test|train|val)?_\d+\b", re.I),
        re.compile(r"\b[A-Za-z]+(?:test|train|val)?_\d{2,}\b"),
    )

    def __init__(self, client: LLMClient):
        super().__init__(client)
        self._reflection_memory_by_scope: Dict[str, List[str]] = {}

    def _reflection_scope_key(self, episode: Optional[object], participant_id: str) -> str:
        meeting_id = getattr(episode, "meeting_id", None) or "unknown_meeting"
        return f"{meeting_id}::{participant_id or 'Delegate'}"

    def _get_reflection_memory(self, episode: Optional[object], participant_id: str) -> List[str]:
        scope = self._reflection_scope_key(episode, participant_id)
        return list(self._reflection_memory_by_scope.get(scope, []))

    def _set_reflection_memory(self, episode: Optional[object], participant_id: str, memory: Sequence[str]) -> None:
        scope = self._reflection_scope_key(episode, participant_id)
        self._reflection_memory_by_scope[scope] = [str(item).strip() for item in memory or [] if str(item).strip()]

    def _format_reflection_memory(self, memory: Sequence[str]) -> str:
        rows = [f"- {str(item).strip()}" for item in memory or [] if str(item).strip()]
        return "\n".join(rows) if rows else ""

    def _get_reflection_memory_text(self, episode: Optional[object], participant_id: str) -> Optional[str]:
        return self._format_reflection_memory(self._get_reflection_memory(episode, participant_id))

    def _reflection_memory_needs_curation(self, memory: Sequence[str]) -> bool:
        serialized = "\n".join(str(item) for item in memory or [])
        return (
            len([item for item in memory or [] if str(item).strip()]) > self.REFLECTION_MEMORY_LESSON_COUNT_FOR_CURATION
            or len(serialized) > self.REFLECTION_MEMORY_TOKEN_CHARS_FOR_CURATION
        )

    def _sanitize_reflection_memory(self, raw_memory: Sequence[Any]) -> tuple[List[str], List[str]]:
        sanitized: List[str] = []
        drop_reasons: List[str] = []
        for index, item in enumerate(raw_memory or []):
            text = re.sub(r"\s+", " ", str(item or "")).strip()
            if not text:
                continue
            if "\n" in str(item):
                drop_reasons.append(f"item_{index}: multiline_span")
                continue
            if "{" in text or "}" in text or "[" in text or "]" in text:
                drop_reasons.append(f"item_{index}: json_like_span")
                continue
            if len(text) > 700:
                drop_reasons.append(f"item_{index}: copied_or_overlong_span")
                continue
            if any(pattern.search(text) for pattern in self.STRUCTURAL_LEAKAGE_PATTERNS):
                drop_reasons.append(f"item_{index}: structural_or_eval_leakage")
                continue
            if text not in sanitized:
                sanitized.append(text)
        return sanitized, drop_reasons

    def _reflection_critic_system_prompt(self) -> str:
        return """
You are a transcript-only reflection critic for a meeting delegate baseline.
Your job is to maintain compact procedural guidance for the same represented participant within this meeting.

Use only the provided raw transcript context, participant profile/briefing text, prior reflection memory, the delegate's latest action, and the single next observed transcript turn.
Do not infer from hidden ground truth, official targets, future windows, CAPA state, helper outputs, judge labels, or recall outcomes.

Return no_update when the next observed turn is phatic, a backchannel, fragmentary, disfluent, or too ambiguous to support a useful procedural lesson.
Do not punish silence merely because the next turn contains substantive content; identify a concrete available cue or profile/briefing-backed reason before adding a lesson.
Prefer reusable procedural lessons about opportunity recognition, unsupported agreement, missed participant-local constraints, redundancy, timing, or grounding.
Do not include utterance IDs, episode IDs, target IDs, judge/eval labels, JSON field names, or copied transcript spans in memory.
Return the full updated memory list, preserving useful prior lessons and merging overlapping ones when update_mode=curate_memory.
""".strip()

    def _reflection_critic_user_prompt(self, payload: Dict[str, Any]) -> str:
        return (
            "Update the reflection memory if a reliable procedural lesson is warranted.\n\n"
            f"Reflection critic payload:\n{self._safe_json(payload)}\n\n"
            "Return one JSON object matching the schema exactly."
        )

    def _postprocess_reflection_output(
        self,
        raw_output: Dict[str, Any],
        payload: Dict[str, Any],
        prior_memory: Sequence[str],
    ) -> Dict[str, Any]:
        return raw_output

    def _strip_turn_for_reflection(self, turn: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(turn, dict):
            return {"speaker_id": None, "content": ""}
        return {
            "speaker_id": turn.get("speaker_id"),
            "content": turn.get("utterance_content") or turn.get("content") or "",
        }

    def _strip_window_for_reflection(self, turns: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self._strip_turn_for_reflection(turn) for turn in turns or [] if isinstance(turn, dict)]

    def _strip_action_log_for_reflection(self, entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            rows.append(
                {
                    "action": entry.get("action"),
                    "spoken_content": entry.get("spoken_content"),
                }
            )
        return rows

    def update_reflection_after_observed_turn(
        self,
        *,
        smm: SharedMemoryModule,
        participant_id: str,
        episode: Optional[object],
        delta_index: int,
        prompt_snapshot_utterances: Sequence[Dict[str, Any]],
        observed_turn: Dict[str, Any],
        delegate_response: Optional[DelegateResponse],
    ) -> None:
        participant_id = participant_id or "Delegate"
        prior_memory = self._get_reflection_memory(episode, participant_id)
        prompt_context = self.last_mono_raw_prompt_context
        profile_text = prompt_context.profile_text if prompt_context else ""
        jit_points = list(prompt_context.jit_briefing_points or []) if prompt_context else []
        action_log = list(prompt_context.delegate_action_log or []) if prompt_context else []
        controller_output = self.last_controller_output
        generator_output = self.last_generator_output
        action_name = None
        action_justification = None
        if controller_output and getattr(controller_output, "action_details", None):
            action_name = controller_output.action_details.action_name
            action_justification = controller_output.action_details.justification
        update_mode = "curate_memory" if self._reflection_memory_needs_curation(prior_memory) else "step_update"
        payload = {
            "update_mode": update_mode,
            "participant_id": participant_id,
            "profile_text": profile_text,
            "jit_briefing_points": jit_points,
            "prior_reflection_memory": prior_memory,
            "recent_delegate_action_log": self._strip_action_log_for_reflection(action_log),
            "visible_conversation_window": self._strip_window_for_reflection(prompt_snapshot_utterances),
            "delegate_action": {
                "action_name": action_name,
                "justification": action_justification,
                "utterance": (generator_output.utterance if generator_output else None)
                or (delegate_response.response if delegate_response else None)
                or "",
            },
            "next_observed_turn": self._strip_turn_for_reflection(observed_turn),
        }

        self.last_reflection_payload = payload
        self.last_reflection_memory_before = list(prior_memory)

        critic_output = self._call_structured_output(
            system_prompt=self._reflection_critic_system_prompt(),
            user_prompt=self._reflection_critic_user_prompt(payload),
            response_model=MonoRawReflectionCriticOutput,
            max_tokens=2500,
            temperature=0.1,
        )

        raw_output: Dict[str, Any]
        if critic_output and hasattr(critic_output, "model_dump"):
            raw_output = critic_output.model_dump()
        else:
            raw_output = {
                "verdict": "no_update",
                "rationale": "Reflection critic failed or returned invalid output; preserving prior memory.",
                "updated_memory": prior_memory,
                "drop_reason": "critic_failed",
            }
        raw_output = self._postprocess_reflection_output(raw_output, payload, prior_memory)

        verdict = str(raw_output.get("verdict") or "no_update").strip().lower()
        candidate_memory = raw_output.get("updated_memory") or []
        if verdict == "update" and not candidate_memory and raw_output.get("lesson"):
            candidate_memory = list(prior_memory) + [raw_output.get("lesson")]
        if verdict == "no_update" and not candidate_memory:
            candidate_memory = list(prior_memory)

        sanitized_memory, drop_reasons = self._sanitize_reflection_memory(candidate_memory)
        if drop_reasons and not sanitized_memory:
            sanitized_memory = list(prior_memory)
        self._set_reflection_memory(episode, participant_id, sanitized_memory)

        artifact_output = {
            "critic_output": raw_output,
            "memory_before": prior_memory,
            "memory_after": sanitized_memory,
            "sanitization_drop_reasons": drop_reasons,
            "scope": self._reflection_scope_key(episode, participant_id),
            "delta_index": delta_index,
        }
        self.last_reflection_output = artifact_output
        self.last_reflection_memory_after = list(sanitized_memory)

        if self.artifact_tracker:
            self.artifact_tracker.capture_helper_io(
                "mono_raw_reflection_critic",
                payload,
                artifact_output,
            )


class MonoRawReflectionV2Agent(MonoRawReflectionAgent):
    """Conservative transcript-only reflection baseline with conditional memory."""

    raw_delegate_variant = "mono-raw-reflection-v2"

    ALLOWED_POST_SPEECH_FEEDBACK = {
        "correction",
        "contradiction",
        "over_bundling",
        "profile_contradiction",
        "redundancy",
        "unsupported_agreement",
        "unsupported_claim",
    }
    GENERIC_LESSON_PATTERNS = (
        re.compile(r"\b(be|speak|contribute|share)\s+proactive(?:ly)?\b", re.I),
        re.compile(r"\b(?:include|share|explore|cover)\s+all\s+relevant\b", re.I),
        re.compile(r"\bensure\s+(?:your|their)\s+(?:perspective|insights?|input)\s+(?:is|are)\s+considered\b", re.I),
        re.compile(r"\bwhen\s+the\s+discussion\s+aligns\s+with\s+your\s+expertise\b", re.I),
    )
    CONDITIONAL_LESSON_PATTERN = re.compile(
        r"Trigger:\s*(?P<trigger>.+?)\s*\|\s*Do:\s*(?P<do>.+?)\s*\|\s*Avoid:\s*(?P<avoid>.+?)\s*\|\s*Confidence:\s*(?P<confidence>high|medium)\s*$",
        re.I,
    )

    def _get_reflection_memory_mode(self) -> str:
        return "v2"

    def _format_reflection_memory(self, memory: Sequence[str]) -> str:
        rows = [f"- {str(item).strip()}" for item in memory or [] if str(item).strip()]
        return "\n".join(rows) if rows else ""

    def _reflection_critic_system_prompt(self) -> str:
        return """
You are a conservative transcript-only reflection critic for a meeting delegate baseline.
Your job is to maintain compact conditional procedural guidance for the same represented participant within this meeting.

Use only the provided raw transcript context, participant profile/briefing text, prior reflection memory, the delegate's latest action, and the single next observed transcript turn.
Do not infer from hidden ground truth, official targets, future windows, CAPA state, helper outputs, judge labels, or recall outcomes.

Memory format:
- Every retained lesson must be one single line:
  Trigger: <specific raw-context situation> | Do: <one concrete adjustment> | Avoid: <one concrete failure mode> | Confidence: high|medium
- Do not write generic lessons such as "be proactive", "share all relevant ideas", or "ensure your perspective is considered".
- Lessons should improve discrimination: which specific point to make, when to stay silent, and what same-topic substitution or over-bundling to avoid.

Update policy:
- If the delegate spoke, default to no_update. Add a lesson only when the next turn directly reveals correction, contradiction, redundancy, unsupported agreement, unsupported claim, profile contradiction, or clear over-bundling.
- If the delegate remained silent, add missed_opportunity only when the next turn contains a concrete profile/JIT-supported point that was plausibly available from the visible transcript context.
- Do not add a lesson merely because the next turn contains additional related substantive content.
- Return no_update when the next observed turn is phatic, a backchannel, fragmentary, disfluent, or too ambiguous to support a useful lesson.
- Return the full updated memory list, preserving useful prior conditional lessons and merging overlapping ones when update_mode=curate_memory.
""".strip()

    def _reflection_critic_user_prompt(self, payload: Dict[str, Any]) -> str:
        return (
            "Update the conditional reflection memory only if the strict v2 update policy is satisfied.\n"
            "All updated_memory entries must use exactly: "
            "Trigger: ... | Do: ... | Avoid: ... | Confidence: high|medium\n\n"
            f"Reflection critic payload:\n{self._safe_json(payload)}\n\n"
            "Return one JSON object matching the schema exactly."
        )

    def _lesson_is_generic(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in self.GENERIC_LESSON_PATTERNS)

    def _sanitize_reflection_memory(self, raw_memory: Sequence[Any]) -> tuple[List[str], List[str]]:
        sanitized: List[str] = []
        drop_reasons: List[str] = []
        for index, item in enumerate(raw_memory or []):
            text = re.sub(r"\s+", " ", str(item or "")).strip()
            if not text:
                continue
            if "\n" in str(item):
                drop_reasons.append(f"item_{index}: multiline_span")
                continue
            if "{" in text or "}" in text or "[" in text or "]" in text:
                drop_reasons.append(f"item_{index}: json_like_span")
                continue
            if len(text) > 700:
                drop_reasons.append(f"item_{index}: copied_or_overlong_span")
                continue
            if any(pattern.search(text) for pattern in self.STRUCTURAL_LEAKAGE_PATTERNS):
                drop_reasons.append(f"item_{index}: structural_or_eval_leakage")
                continue
            if not self.CONDITIONAL_LESSON_PATTERN.search(text):
                drop_reasons.append(f"item_{index}: missing_conditional_fields")
                continue
            if self._lesson_is_generic(text):
                drop_reasons.append(f"item_{index}: generic_reflection_lesson")
                continue
            if text not in sanitized:
                sanitized.append(text)
        return sanitized, drop_reasons

    def _no_update_output(self, raw_output: Dict[str, Any], prior_memory: Sequence[str], reason: str) -> Dict[str, Any]:
        output = dict(raw_output or {})
        output["verdict"] = "no_update"
        output["updated_memory"] = list(prior_memory)
        output["drop_reason"] = reason
        output["rationale"] = f"{output.get('rationale') or ''} [v2 gate: {reason}]".strip()
        return output

    def _postprocess_reflection_output(
        self,
        raw_output: Dict[str, Any],
        payload: Dict[str, Any],
        prior_memory: Sequence[str],
    ) -> Dict[str, Any]:
        verdict = str((raw_output or {}).get("verdict") or "no_update").strip().lower()
        if verdict not in {"update", "curate_memory"}:
            return raw_output

        delegate_action = payload.get("delegate_action") or {}
        action_name = str(delegate_action.get("action_name") or "").strip()
        utterance = str(delegate_action.get("utterance") or "").strip()
        delegate_spoke = bool(utterance) or action_name not in {"", "Remain_Silent"}
        feedback_type = str((raw_output or {}).get("feedback_type") or "").strip().lower()
        evidence_quality = str((raw_output or {}).get("evidence_quality") or "").strip().lower()
        next_turn = payload.get("next_observed_turn") or {}
        next_speaker = str(next_turn.get("speaker_id") or "").strip()
        participant_id = str(payload.get("participant_id") or "").strip()

        if delegate_spoke and feedback_type not in self.ALLOWED_POST_SPEECH_FEEDBACK:
            return self._no_update_output(
                raw_output,
                prior_memory,
                "post_speech_update_requires_direct_problem",
            )

        if not delegate_spoke and feedback_type in {"missed_opportunity", "opportunity_recognition"}:
            if evidence_quality not in {"high", "medium"}:
                return self._no_update_output(
                    raw_output,
                    prior_memory,
                    "missed_opportunity_requires_medium_or_high_evidence",
                )
            if next_speaker != participant_id and evidence_quality != "high":
                return self._no_update_output(
                    raw_output,
                    prior_memory,
                    "non_participant_next_turn_requires_high_evidence",
                )

        return raw_output
