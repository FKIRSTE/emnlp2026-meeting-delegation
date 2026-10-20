"""
Environment Interpretation Module (EIM)

This module implements the Environment Interpretation Module that provides stateful
meeting context analysis. The EIM maintains a master state of the meeting and can
perform deep analysis on specific snapshots while incorporating historical context.

Extracted from cells 21-26 of the original notebook.
"""

import json
import sys
import time
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import ValidationError

# Add current directory to path for robust importing
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from llm_client import LLMClient
from schemas import EIMMeetingStateVector, EIMMasterState, LastUtteranceContext
from capa.episode_utils import is_banal, is_fragmentary, response_substance_score


class EnvironmentInterpretationModule:
    """
    The Environment Interpretation Module (EIM) provides stateful meeting context analysis.
    
    Key Features:
    - Maintains a master state of meeting topics and participant models
    - Processes historical utterances in chunks with overlap
    - Performs deep analysis on snapshots using dual-focus approach
    - Provides rich MeetingStateVector outputs with validation
    """
    
    def __init__(self, client: LLMClient, history_chunk_size: int = 40, history_overlap_size: int = 10):
        """
        Initialize the EIM with LLM client and model configuration.
        
        Args:
            client: LLMClient instance for LLM interactions
            history_chunk_size: The number of utterances to include in each history processing chunk.
            history_overlap_size: The number of utterances to overlap between consecutive chunks.
        """
        self.client = client
        self.llm_client = client  # For compatibility with tests
        self.model_name = client.model  # Extract model name from the LLM client
        self.HISTORY_CHUNK_SIZE = history_chunk_size
        self.HISTORY_OVERLAP_SIZE = history_overlap_size
        # For backwards compatibility with tests
        self.history_chunk_size = history_chunk_size
        self.history_overlap_size = history_overlap_size
        
        # Initialize empty master state
        self.master_state = EIMMasterState(
            topic_summary_log={},
            participant_models={}
        )

        # Debug trace toggle (set EIM_TRACE=0 to disable)
        self._trace_enabled = os.getenv("EIM_TRACE", "1") != "0"

    def reset_master_state(self) -> None:
        self.master_state = EIMMasterState(
            topic_summary_log={},
            participant_models={},
            identity_aliases={},
        )

    def export_master_state(self) -> Dict[str, Any]:
        return self.master_state.model_dump()

    def load_master_state(self, payload: Dict[str, Any]) -> None:
        state_payload = payload or {}
        self.master_state = EIMMasterState.model_validate(state_payload)
        if getattr(self.master_state, "identity_aliases", None) is None:
            self.master_state.identity_aliases = {}

    # --- Debug helpers ----------------------------------------------------
    def _trace(self, msg: str):
        if self._trace_enabled:
            print(f"[EIM-TRACE] {msg}")

    def _summarize_master_state(self) -> str:
        try:
            topics_total = len(getattr(self.master_state, 'topic_summary_log', {}) or {})
            participants_total = len(getattr(self.master_state, 'participant_models', {}) or {})
            aliases_total = len(getattr(self.master_state, 'identity_aliases', {}) or {})
            # Find most recent up to 3 topics by last_updated_at_utterance_id
            def _utt_num(u):
                try:
                    return int(str(u).split('_')[-1]) if u else -1
                except Exception:
                    return -1
            recent = []
            for tlabel, tentry in (self.master_state.topic_summary_log or {}).items():
                recent.append((tlabel, getattr(tentry, 'status', None), getattr(tentry, 'last_updated_at_utterance_id', None)))
            recent.sort(key=lambda x: _utt_num(x[2]), reverse=True)
            recent_str = ", ".join(
                f"{lbl}({status or '?'}, {last or '-'})" for lbl, status, last in recent[:3]
            )
            return f"topics={topics_total}, participants={participants_total}, aliases={aliases_total}, recent=[{recent_str}]"
        except Exception:
            return "<summary failed>"

    def _merge_state_updates(self, diff: EIMMasterState):
        """
        Merges a 'diff' or 'patch' object into the main master_state.

        This method updates the existing state with new information
        from the diff object, handling nested structures like topics and participant models.

        Args:
            diff: An EIMMasterState object containing only the updates.
        """
        # Merge topic_summary_log
        self._trace("MERGE_START: applying diff to master state")
        def _utt_num(utt_id: str) -> int:
            try:
                if not utt_id:
                    return -1
                # Expect pattern like MeetingX_123
                return int(str(utt_id).split('_')[-1])
            except Exception:
                return -1

        # Collect newly ongoing topics' latest utterance numbers for heuristic
        new_ongoing_latest = -1
        for tkey, tval in diff.topic_summary_log.items():
            if getattr(tval, 'status', None) == 'Ongoing' and getattr(tval, 'last_updated_at_utterance_id', None):
                new_ongoing_latest = max(new_ongoing_latest, _utt_num(tval.last_updated_at_utterance_id))

            ## for testing:
            else:
                self._trace(
                    f"merge getattr check: status={getattr(tval, 'status', None)} last={getattr(tval, 'last_updated_at_utterance_id', None)}"
                )

        for topic_key, topic_value in diff.topic_summary_log.items():
            if topic_key not in self.master_state.topic_summary_log:
                # If the topic is entirely new, add it.
                self.master_state.topic_summary_log[topic_key] = topic_value
            else:
                # If the topic exists, update its fields.
                existing_topic = self.master_state.topic_summary_log[topic_key]
                if topic_value.key_points:
                    existing_topic.key_points.extend(topic_value.key_points)
                    # Dedupe key_points by utterance_id, preserve order
                    seen = set()
                    deduped = []
                    for kp in existing_topic.key_points:
                        uid = getattr(kp, 'utterance_id', None)
                        if uid in seen:
                            continue
                        seen.add(uid)
                        deduped.append(kp)
                    existing_topic.key_points = deduped
                # Merge open_questions if present
                if hasattr(topic_value, 'open_questions') and topic_value.open_questions:
                    if not hasattr(existing_topic, 'open_questions') or existing_topic.open_questions is None:
                        existing_topic.open_questions = []
                    # Append new questions/updates
                    existing_topic.open_questions.extend(topic_value.open_questions)
                    # Dedupe by source_utterance_id with LATEST-WINS semantics so updates (e.g., resolved flags) override earlier entries
                    by_sid = {}
                    order = []
                    for oq in existing_topic.open_questions:
                        sid = getattr(oq, 'source_utterance_id', None)
                        if sid is None:
                            # Keep non-standard entries as-is (should be rare)
                            sid = f"__no_sid__:{id(oq)}"
                        # Track order (first occurrence) but overwrite value (last wins)
                        if sid not in by_sid:
                            order.append(sid)
                        by_sid[sid] = oq
                    # Rebuild list preserving first-seen order but with latest values
                    existing_topic.open_questions = [by_sid[sid] for sid in order]
                # Merge decisions if present
                if hasattr(topic_value, 'decisions') and topic_value.decisions:
                    if not hasattr(existing_topic, 'decisions') or existing_topic.decisions is None:
                        existing_topic.decisions = []
                    existing_topic.decisions.extend(topic_value.decisions)
                    # Dedupe decisions by source_utterance_id, preserve order
                    seen_d = set()
                    deduped_d = []
                    for de in existing_topic.decisions:
                        sid = getattr(de, 'source_utterance_id', None)
                        if sid in seen_d:
                            continue
                        seen_d.add(sid)
                        deduped_d.append(de)
                    existing_topic.decisions = deduped_d
                if topic_value.status:
                    existing_topic.status = topic_value.status
                if topic_value.last_updated_at_utterance_id:
                    existing_topic.last_updated_at_utterance_id = topic_value.last_updated_at_utterance_id

        # Merge participant_models
        for participant_key, participant_value in diff.participant_models.items():
            if participant_key not in self.master_state.participant_models:
                # If the participant is new, add them.
                self.master_state.participant_models[participant_key] = participant_value
            else:
                # If the participant exists, update their model.
                existing_model = self.master_state.participant_models[participant_key]
                if participant_value.inferred_expertise_keywords:
                    for keyword in participant_value.inferred_expertise_keywords:
                        if keyword not in existing_model.inferred_expertise_keywords:
                            existing_model.inferred_expertise_keywords.append(keyword)
                
                if participant_value.recent_dialogue_act:
                    existing_model.recent_dialogue_act = participant_value.recent_dialogue_act

                if participant_value.expressed_stance_on_topics:
                    for topic_key, stance_list in participant_value.expressed_stance_on_topics.items():
                        if topic_key not in existing_model.expressed_stance_on_topics:
                            existing_model.expressed_stance_on_topics[topic_key] = []
                        existing_model.expressed_stance_on_topics[topic_key].extend(stance_list)
                        # Dedupe by source_utterance_id, preserve order
                        seen = set()
                        deduped = []
                        for st in existing_model.expressed_stance_on_topics[topic_key]:
                            sid = getattr(st, 'source_utterance_id', None)
                            if sid in seen:
                                continue
                            seen.add(sid)
                            deduped.append(st)
                        existing_model.expressed_stance_on_topics[topic_key] = deduped

        # Merge identity_aliases if present in diff
        try:
            if hasattr(diff, 'identity_aliases') and diff.identity_aliases:
                if not getattr(self.master_state, 'identity_aliases', None):
                    self.master_state.identity_aliases = {}
                added_aliases = 0
                for name, mapped in diff.identity_aliases.items():
                    if name not in self.master_state.identity_aliases:
                        self.master_state.identity_aliases[name] = mapped
                        added_aliases += 1
                if added_aliases:
                    self._trace(f"MERGE_ALIASES: +{added_aliases} new identity aliases")
        except Exception:
            pass

        # Topic status heuristic: if a different topic is now ongoing far later, conservatively conclude stale ongoing topics
        try:
            if new_ongoing_latest >= 0:
                for tkey, tval in self.master_state.topic_summary_log.items():
                    if getattr(tval, 'status', None) == 'Ongoing':
                        last_num = _utt_num(getattr(tval, 'last_updated_at_utterance_id', None))
                        # If this topic wasn't updated in a long time relative to new ongoing topic, conclude it.
                        # Threshold of 15 utterances; conservative.
                        if last_num >= 0 and new_ongoing_latest - last_num >= 15 and tkey not in diff.topic_summary_log:
                            tval.status = 'Concluded'
        except Exception:
            pass

        self._trace(f"MERGE_DONE: {self._summarize_master_state()}")

    def apply_patch(self, diff: EIMMasterState, current_turn_utterance_id: Optional[str] = None) -> bool:
        """
        Public API to apply an EIMMasterState diff to the master state with
        optional no-future-anchor enforcement.

        Args:
            diff: EIMMasterState containing only updates (diff/patch semantics).
            current_turn_utterance_id: If provided, all new anchors in the diff
                must be <= this utterance id (by numeric suffix) to prevent
                leakage from future content.

        Returns:
            True if merged; False if a violation was detected and nothing merged.
        """
        def _utt_num(u: Optional[str]) -> int:
            try:
                if not u:
                    return -1
                return int(str(u).split('_')[-1])
            except Exception:
                return -1

        if current_turn_utterance_id:
            ceiling = _utt_num(current_turn_utterance_id)
            # Validate topic anchors (key_points, open_questions, decisions)
            for tkey, tval in (diff.topic_summary_log or {}).items():
                try:
                    for kp in (getattr(tval, 'key_points', None) or []):
                        if _utt_num(getattr(kp, 'utterance_id', None)) > ceiling:
                            print(f"  [EIM Patch] ✖ Future anchor in key_point for topic '{tkey}'.")
                            return False
                    for oq in (getattr(tval, 'open_questions', None) or []):
                        if _utt_num(getattr(oq, 'source_utterance_id', None)) > ceiling:
                            print(f"  [EIM Patch] ✖ Future anchor in open_question for topic '{tkey}'.")
                            return False
                    for de in (getattr(tval, 'decisions', None) or []):
                        if _utt_num(getattr(de, 'source_utterance_id', None)) > ceiling:
                            print(f"  [EIM Patch] ✖ Future anchor in decision for topic '{tkey}'.")
                            return False
                except Exception:
                    # Be conservative if inspection fails
                    print(f"  [EIM Patch] ✖ Unable to validate anchors for topic '{tkey}'.")
                    return False

            # Validate participant stance anchors
            for pkey, pval in (diff.participant_models or {}).items():
                try:
                    st_map = getattr(pval, 'expressed_stance_on_topics', {}) or {}
                    for topic_label, st_list in st_map.items():
                        for st in st_list:
                            if _utt_num(getattr(st, 'source_utterance_id', None)) > ceiling:
                                print(f"  [EIM Patch] ✖ Future anchor in stance for participant '{pkey}' topic '{topic_label}'.")
                                return False
                except Exception:
                    print(f"  [EIM Patch] ✖ Unable to validate stances for participant '{pkey}'.")
                    return False

        self._merge_state_updates(diff)
        return True

    def _format_utterances_for_prompt(self, utterances: List[Dict]) -> str:
        """
        Format a list of utterances for inclusion in LLM prompts.
        
        Args:
            utterances: List of utterance dictionaries with speaker_id, utterance_content, utterance_id
            
        Returns:
            Formatted string for prompt inclusion
        """
        formatted_lines = []
        for utt in utterances:
            speaker = utt.get('speaker_id', 'Unknown')
            content = utt.get('utterance_content', '')
            utt_id = utt.get('utterance_id', 'Unknown')
            formatted_lines.append(f"({speaker}): {content} ({utt_id})")
        return "\n".join(formatted_lines)

    def _process_single_history_chunk(self, utterances_chunk: List[Dict]) -> bool:
        """
        Process a single chunk of historical utterances to generate a "diff" and
        merge it into the master state.
        
        Args:
            utterances_chunk: List of utterances to process
            
        Returns:
            bool: True if processing succeeded, False if validation failed
        """
        if not utterances_chunk:
            return True

        # Trace before processing
        try:
            first_id = utterances_chunk[0].get('utterance_id')
            last_id = utterances_chunk[-1].get('utterance_id')
            self._trace(f"HISTORY_BEFORE_CHUNK ids={first_id}..{last_id} | {self._summarize_master_state()}")
        except Exception:
            self._trace(f"HISTORY_BEFORE_CHUNK | {self._summarize_master_state()}")

        chunk_str = self._format_utterances_for_prompt(utterances_chunk)
        current_state_str = self.master_state.model_dump_json(indent=2)

        system_prompt = """You are an expert, meticulous meeting analyst. Your task is to analyze meeting utterances and update the meeting state accordingly.

        ## CRITICAL: Cold Start vs. Update Mode
        **COLD START MODE:** If the `Current_State` has empty topic_summary_log={{}} and/or participant_models={{}}, you are in COLD START MODE.
        - You MUST extract substantial content from this chunk and create initial topics and participant models.
        - Emit 1-3 macro topics with clear labels (e.g., "Project Overview: <name>", "Meeting Agenda", "Initial Requirements").
        - Extract 3-6 key points from this chunk, focusing on project goals, constraints, responsibilities, and foundational information.
        - Do NOT return empty topic_summary_log or participant_models in cold start mode.
        - When uncertain, prefer creating a broad topic with key points rather than returning nothing.
        
        **UPDATE MODE:** If the `Current_State` already contains topics and participants, you are in UPDATE MODE.
        - Output a "diff" containing ONLY new or updated information.
        - Do NOT replicate existing data that hasn't changed.

        ## Core Task: State Analysis & Update
        1. **Check Current_State**: Determine if you're in COLD START MODE (empty state) or UPDATE MODE (existing state).
        2. **Process New_Utterances_Chunk**: Extract substantive content including project overview, goals, constraints, agenda items, responsibilities, and stakeholder priorities.
        3. **Generate appropriate output**: Full initial state for cold start, or diff for updates.

        Utterance Format
        - Each line uses the form: "(SpeakerID): content (UtteranceID)". Use these to anchor speaker and utterance_id.

        ## Key Guidelines:
        1. **Topic Management**:
            - COLD START: Create initial macro topics covering project overview, meeting objectives, and key areas discussed.
            - UPDATE: Add new topics or update existing ones based on topic shifts.
            - When a topic shift is detected in UPDATE mode, mark previous "Ongoing" topics as "Concluded".
            - Do not append vaguely related points to broad, existing topics - Prefer granularity over broad, and overly generic topics.
        2. **Key Point Extraction**: Extract substantive points including project overview, goals, constraints, agenda items, responsibilities, and stakeholder priorities. Do not add simple acknowledgments.
        3. **Participant Modeling (MUST NOT BE SPARSE)**: 
            - COLD START: Create initial participant models for ALL speakers in the chunk.
            - UPDATE: Update existing participants or add new ones with each chunk.
            
            **Enrichment Requirements (for each substantive utterance):**
            - **Expertise Keywords**: Add when participant demonstrates deep knowledge (explains mechanisms, cites data, provides technical details)—NOT just mentions a term.
            - **Stances**: Add when participant takes a position (agrees, disagrees, expresses concern, proposes, questions). Each stance needs topic, summary, and source_utterance_id.
            - **Dialogue Act**: Update recent_dialogue_act to reflect their most recent contribution type (Questioning, Proposing, Explaining, Agreeing, Challenging, etc.)
            
            **ANTI-SPARSITY CHECK**: If a participant made substantive utterances in this chunk but their model update is empty, you likely missed something. Review their utterances for stances, expertise, or dialogue acts.
            
            - **Stances:** Append new stance entries, don't overwrite existing ones.
        4. **Identity Aliases**: Map personal names to participant IDs/roles when confident. Create mappings in these cases:
            - **Explicit Introduction**: Direct statements like "Ron as HR Specialist" or "Let's hear from Sarah in Engineering"
            - **Response Pattern Recognition**: When a personal name is mentioned and consistently the same participant responds immediately after (≥2 instances in current or historical context)
            - **Role-Based Context**: When a personal name is used in role-specific contexts that clearly match a participant's expertise/function
            - **Direct Address Response**: When someone uses a personal name to address someone and a specific participant consistently responds as if being addressed
            Only create mappings when the evidence is clear and unambiguous. When uncertain, omit the mapping.
        5. **Non-Empty Diff Rule (UPDATE)**: If this chunk contains ≥2 substantive utterances that introduce new points, do not return an empty diff. (e.g., provide data, explain a complex mechanism, offer a detailed pro/con analysis, ...). Simply mentioning a term is **NOT** expertise.
        6. **Label Specificity**: Avoid generic labels like "Discussion/Topic"; prefer object/project-oriented labels (e.g., "Remote Control Inputs", "Cost Ceiling & BOM").
        
        ---
        
        ## Output Schema
        - Your output MUST be a JSON object representing the "diff" of changes. It must conform to the `EIMMasterState` schema, but only include fields and info that are new or have been updated.
        - The JSON object must strictly adhere to the following `EIMMasterState` schema:
        ```json
        {{
            "topic_summary_log": {{
                "<Topic Label as String Key>": {{
                "label": "(string) The human-readable name of the topic. MUST match the dictionary key.",
                "last_updated_at_utterance_id": "(string or null) The utterance_id of the last key point added to this topic.",
                "key_points": [
                    {{
                    "point": "(string) A one-sentence summary of a substantive contribution.",
                    "speaker": "(string) The speaker_id of the person who made the point.",
                    "utterance_id": "(string) The utterance_id where the point was made."
                    }}
                ],
                "status": "(string) The current status of the topic. MUST be either 'Ongoing' or 'Concluded'."
                }}
            }},
            "participant_models": {{
                "<Speaker ID as String Key>": {{
                    "inferred_expertise_keywords": [
                        "(string) A keyword representing a domain of expertise demonstrated by this participant."
                    ],
                    "expressed_stance_on_topics": {{
                        "<Topic Label as String Key>": [
                        {{
                            "stance_summary": "(string) A concise summary of the participant's expressed opinion or position on this topic.",
                            "source_utterance_id": "(string) The utterance_id where this stance was expressed."
                        }}
                        ]
                    }},
                "recent_dialogue_act": "(string or null) The dialogue act of the participant's most recent utterance in the chunk (e.g., 'Questioning', 'Proposing', ...)."
                }}
            }},
            "identity_aliases": {{
                "<Personal Name>": "(string) Participant ID or role this name refers to"
            }}
        }}
        ```
        """

        user_prompt_template = """Please analyze the new utterances chunk and provide a "diff" of the changes to the meeting state.

        ---
        ## EXAMPLES (for reference)

        ### Example 1: Cold Start (Processing the first chunk of a meeting)

        **Current State:**
        ```json
        {{
            "topic_summary_log": {{}},
            "participant_models": {{}}
        }}
        ```
        **New Utterances Chunk:**
        (Manager): Alright everyone, let's kick off the Q3 planning meeting. The main goal today is to finalize the roadmap for the new 'Orion' feature. (MeetingX_001)
        (Lead Dev): Sounds good. I've reviewed the initial spec, and my main concern is the database choice. (MeetingX_002)
        (Manager): Good point. Let's start there. (MeetingX_003)

        **Desired Output (The "diff" object containing only new information):**
        ```json
        {{
            "topic_summary_log": {{
                "Q3 Orion Feature Roadmap": {{
                    "label": "Q3 Orion Feature Roadmap",
                    "last_updated_at_utterance_id": "MeetingX_001",
                    "key_points": [
                        {{"point": "Manager defined the meeting's main goal: to finalize the roadmap for the 'Orion' feature.", "speaker": "Manager", "utterance_id": "MeetingX_001"}}
                    ],
                    "status": "Ongoing"
                }},
                "Database Choice for Orion": {{
                    "label": "Database Choice for Orion",
                    "last_updated_at_utterance_id": "MeetingX_002",
                    "key_points": [
                        {{"point": "Lead Dev raised a concern about the database choice for the feature.", "speaker": "Lead Dev", "utterance_id": "MeetingX_002"}}
                    ],
                    "status": "Ongoing"
                }}
            }},
            "participant_models": {{
                "Manager": {{
                    "inferred_expertise_keywords": ["Project Planning", "Roadmapping"],
                    "expressed_stance_on_topics": {{}},
                    "recent_dialogue_act": "Acknowledgement"
                }},
                "Lead Dev": {{
                    "inferred_expertise_keywords": ["Technical Specification", "Database Architecture"],
                    "expressed_stance_on_topics": {{
                        "Database Choice for Orion": [
                            {{
                                "stance_summary": "Expressed concern about the database choice.",
                                "source_utterance_id": "MeetingX_002"
                            }}
                        ]
                    }},
                    "recent_dialogue_act": "Stating a concern"
                }}
            }}
        }}
        ```

        ### Example 2a: Kickoff + Overview (Cold Start)

        **Current State:**
        ```json
        {{
            "topic_summary_log": {{}},
            "participant_models": {{}}
        }}
        ```
        **New Utterances Chunk:**
        (Manager): Thanks all. Today we're starting the “Project Aurora” device discussion. We need alignment on scope and stakeholders. (MeetingA_001)
        (Marketing): Our target customer is home users; price ceiling $99 to stay competitive. (MeetingA_002)
        (Industrial Designer): Priorities should be ergonomics and durability; we can't compromise accessibility. (MeetingA_003)

        **Desired Output (Diff with macro topic and high-level points):**
        ```json
        {{
          "topic_summary_log": {{
            "Project Overview: Aurora Device": {{
              "label": "Project Overview: Aurora Device",
              "last_updated_at_utterance_id": "MeetingA_003",
              "key_points": [
                {{"point": "Manager opened Project Aurora and asked for scope and stakeholder alignment.", "speaker": "Manager", "utterance_id": "MeetingA_001"}},
                {{"point": "Marketing stated target customers are home users with a $99 price ceiling.", "speaker": "Marketing", "utterance_id": "MeetingA_002"}},
                {{"point": "Industrial Designer emphasized ergonomics, durability, and accessibility priorities.", "speaker": "Industrial Designer", "utterance_id": "MeetingA_003"}}
              ],
              "status": "Ongoing"
            }}
          }},
          "participant_models": {{
            "Manager": {{"inferred_expertise_keywords": [], "expressed_stance_on_topics": {{}}, "recent_dialogue_act": "Meeting kickoff"}},
            "Marketing": {{"inferred_expertise_keywords": ["Market Strategy"], "expressed_stance_on_topics": {{}}, "recent_dialogue_act": "Information Sharing"}},
            "Industrial Designer": {{"inferred_expertise_keywords": [], "expressed_stance_on_topics": {{}}, "recent_dialogue_act": "Prioritization"}}
          }}
        }}
        ```

        ### Example 2b: Logistics + Constraints (Cold Start)

        **Current State:**
        ```json
        {{
            "topic_summary_log": {{}},
            "participant_models": {{}}
        }}
        ```
        **New Utterances Chunk:**
        (Manager): Let's confirm agenda: goals, constraints, then timeline. (MeetingB_001)
        (Engineering): Constraint: Bluetooth coexistence issues in crowded homes; we must validate robustness. (MeetingB_002)
        (UX): We need quick onboarding; first-use success must be trivial. (MeetingB_003)

        **Desired Output (Diff with macro topics and high-level points):**
        ```json
        {{
          "topic_summary_log": {{
            "Agenda & Objectives": {{
              "label": "Agenda & Objectives",
              "last_updated_at_utterance_id": "MeetingB_001",
              "key_points": [
                {{"point": "Manager set agenda: goals, constraints, then timeline.", "speaker": "Manager", "utterance_id": "MeetingB_001"}}
              ],
              "status": "Ongoing"
            }},
            "Initial Constraints & Requirements": {{
              "label": "Initial Constraints & Requirements",
              "last_updated_at_utterance_id": "MeetingB_003",
              "key_points": [
                {{"point": "Engineering raised Bluetooth coexistence concerns in home environments.", "speaker": "Engineering", "utterance_id": "MeetingB_002"}},
                {{"point": "UX emphasized trivial onboarding and first-use success.", "speaker": "UX", "utterance_id": "MeetingB_003"}}
              ],
              "status": "Ongoing"
            }}
          }},
          "participant_models": {{
            "Engineering": {{"inferred_expertise_keywords": ["Wireless Reliability"], "expressed_stance_on_topics": {{}}, "recent_dialogue_act": "Raising a constraint"}},
            "UX": {{"inferred_expertise_keywords": ["Onboarding"], "expressed_stance_on_topics": {{}}, "recent_dialogue_act": "Setting a requirement"}}
          }}
        }}
        ```

        ### Example 3: State Update (Processing a subsequent chunk)

        **Current State:**
        ```json
        {{
            "topic_summary_log": {{
                "Q3 Orion Feature Roadmap": {{ "label": "Q3 Orion Feature Roadmap", "last_updated_at_utterance_id": "MeetingX_001", "key_points": [ {{"point": "...", "speaker": "...", "utterance_id": "..."}} ], "status": "Ongoing" }},
                "Database Choice for Orion": {{ "label": "Database Choice for Orion", "last_updated_at_utterance_id": "MeetingX_041", "key_points": [ {{"point": "...", "speaker": "...", "utterance_id": "..."}} ], "status": "Ongoing" }}
            }},
            "participant_models": {{
                "Manager": {{ "inferred_expertise_keywords": ["Project Management"], ... }},
                "Lead Dev": {{
                    "inferred_expertise_keywords": ["Database Architecture"],
                    "expressed_stance_on_topics": {{
                        "Database Choice for Orion": [
                            {{"stance_summary": "Expressed concern about the database choice.", "source_utterance_id": "MeetingX_002"}}
                        ]
                    }}, ...
                }}
            }}
        }}
        ```
        **New Utterance Chunk:**
        (Manager): Great, so we're aligned on the database. Let's move on. (MeetingX_044)
        (Lead Dev): To be specific, the current proposal uses a relational database, but our user data is unstructured. A NoSQL solution like MongoDB would be a much better fit for scalability. (MeetingX_045)
        (UX Designer): From a user perspective, does the database choice affect the UI response time? That's my main concern. (MeetingX_046)

        **Desired Output (The "diff" object containing only new information from the current chunk):**
        ```json
        {{
            "topic_summary_log": {{
                "Q3 Orion Feature Roadmap": {{
                    "label": "Q3 Orion Feature Roadmap",
                    "status": "Concluded"
                }},
                "Database Choice for Orion": {{
                    "label": "Database Choice for Orion",
                    "last_updated_at_utterance_id": "MeetingX_046",
                    "key_points": [
                        {{ "point": "Lead Dev advocated for a NoSQL solution like MongoDB, citing better scalability for unstructured user data.", "speaker": "Lead Dev", "utterance_id": "MeetingX_045" }},
                        {{ "point": "UX Designer raised a question about the impact of database choice on UI response time.", "speaker": "UX Designer", "utterance_id": "MeetingX_046" }}
                    ],
                    "status": "Ongoing"
                }}
            }},
            "participant_models": {{
                "Lead Dev": {{
                    "inferred_expertise_keywords": ["NoSQL", "Scalability"],
                    "expressed_stance_on_topics": {{
                        "Database Choice for Orion": [
                            {{ "stance_summary": "Advocates for a NoSQL solution due to scalability.", "source_utterance_id": "MeetingX_045" }}
                        ]
                    }},
                    "recent_dialogue_act": "Proposing a solution"
                }},
                "UX Designer": {{
                    "inferred_expertise_keywords": ["User Experience", "UI Performance"],
                    "expressed_stance_on_topics": {{
                        "Database Choice for Orion": [
                            {{ "stance_summary": "Concerned about the impact on UI response time.", "source_utterance_id": "MeetingX_046" }}
                        ]
                    }},
                    "recent_dialogue_act": "Questioning"
                }}
            }},
            "identity_aliases": {{}}
        }}
        ```

        ### Example 4: Alias Mapping (Personal Name → Role/Participant)

        **Current State:**
        ```json
        {{
            "topic_summary_log": {{}},
            "participant_models": {{
                "HR": {{ "inferred_expertise_keywords": [], "expressed_stance_on_topics": {{}}, "recent_dialogue_act": null }},
                "Engineering": {{ "inferred_expertise_keywords": [], "expressed_stance_on_topics": {{}}, "recent_dialogue_act": null }}
            }},
            "identity_aliases": {{}}
        }}
        ```
        **New Utterances Chunk:**
        (Manager): We'll hear from Ron as HR Specialist next about team dynamics. (MeetingY_010)

        **Desired Output (The "diff" object containing only new information):**
        ```json
        {{
            "topic_summary_log": {{}},
            "participant_models": {{}},
            "identity_aliases": {{
                "Ron": "HR"
            }}
        }}
        ```

        ### Example 5: Pattern-Based Identity Inference

        **Current State:**
        ```json
        {{
            "topic_summary_log": {{ "Security Review": {{ "label": "Security Review", "status": "Ongoing", ... }} }},
            "participant_models": {{
                "Security Lead": {{ "inferred_expertise_keywords": ["Security"], "recent_dialogue_act": "Information sharing" }},
                "Product Manager": {{ "inferred_expertise_keywords": ["Project Planning"], "recent_dialogue_act": "Questioning" }}
            }},
            "identity_aliases": {{}}
        }}
        ```
        **New Utterances Chunk:**
        (Product Manager): What does Alex think about the authentication approach? (MeetingZ_015)
        (Security Lead): I believe we should implement OAuth 2.0 with PKCE for better security. (MeetingZ_016)
        (Product Manager): Thanks Alex, that's exactly what I needed to know. (MeetingZ_017)

        **Desired Output (Pattern recognition: "Alex" addressed → Security Lead responds → Manager thanks "Alex"):**
        ```json
        {{
            "topic_summary_log": {{}},
            "participant_models": {{
                "Security Lead": {{
                    "recent_dialogue_act": "Providing technical recommendation"
                }},
                "Product Manager": {{
                    "recent_dialogue_act": "Acknowledgment"
                }}
            }},
            "identity_aliases": {{
                "Alex": "Security Lead"
            }}
        }}
        ```
        ---
        
        ## YOUR TASK:
        
        **Current State:**
        {current_state_str}

        **New Utterances Chunk:**
        {chunk_str}

        Now, provide the "diff" of changes as a valid JSON object that follows the EIMMasterState schema.
        """

        # Escape braces in JSON content to prevent format conflicts with .format()
        escaped_current_state = current_state_str.replace('{', '{{').replace('}', '}}')
        escaped_chunk_str = chunk_str.replace('{', '{{').replace('}', '}}')
        
        filled_user_prompt = user_prompt_template.format(
            current_state_str=escaped_current_state,
            chunk_str=escaped_chunk_str
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": filled_user_prompt}
        ]

        try:
            # DEBUG: Log the prompt being sent to LLM for first chunk
            is_empty_state = (
                len(self.master_state.topic_summary_log) == 0 and 
                len(self.master_state.participant_models) == 0
            )
            if is_empty_state:
                print(f"  [EIM Debug] COLD START detected - sending to LLM:")
                print(f"    Current state: {current_state_str[:200]}...")
                print(f"    Chunk utterances: {len(utterances_chunk)}")
                if utterances_chunk:
                    print(f"    First utterance: {utterances_chunk[0].get('utterance_id')} - {utterances_chunk[0].get('utterance_content', '')[:100]}...")
                    print(f"    Last utterance: {utterances_chunk[-1].get('utterance_id')} - {utterances_chunk[-1].get('utterance_content', '')[:100]}...")

            # Get JSON response using json_required=True (no instructor)
            json_response = self.client.call_with_retry(
                model=self.model_name,
                messages=messages,
                max_tokens=4096,
                temperature=0.1,
                json_required=True,  # Force JSON output without instructor
                telemetry_label="perceiver.history",
                telemetry_phase="pre_turn",
            )
            
            # Manually validate the JSON against the EIM schema
            diff_response = None
            if json_response:
                diff_response = self.client.validate_json_response(json_response, EIMMasterState)
            
            # DEBUG: Log what LLM actually returned for first chunk
            if is_empty_state and diff_response:
                print(f"  [EIM Debug] LLM response for COLD START:")
                print(f"    Topics returned: {len(diff_response.topic_summary_log or {})}")
                print(f"    Participants returned: {len(diff_response.participant_models or {})}")
                if diff_response.topic_summary_log:
                    for topic_name, topic_data in diff_response.topic_summary_log.items():
                        print(f"      Topic: {topic_name} ({len(topic_data.key_points or [])} key points)")
                if diff_response.participant_models:
                    for participant_id in diff_response.participant_models.keys():
                        print(f"      Participant: {participant_id}")
            
            if diff_response:
                # Trace diff summary
                try:
                    t_keys = list((diff_response.topic_summary_log or {}).keys())
                    p_keys = list((diff_response.participant_models or {}).keys())
                    concluded = [k for k, v in (diff_response.topic_summary_log or {}).items() if getattr(v, 'status', '').lower().startswith('conclud')]
                    aliases = list((getattr(diff_response, 'identity_aliases', {}) or {}).keys())
                    self._trace(
                        f"HISTORY_DIFF topics+~={len(t_keys)} concluded={len(concluded)} participants~={len(p_keys)} aliases+={len(aliases)}"
                    )
                except Exception:
                    self._trace("HISTORY_DIFF <unable to summarize>")
                self._merge_state_updates(diff_response)
                # Trace after merge
                try:
                    self._trace(f"HISTORY_AFTER_CHUNK | {self._summarize_master_state()}")
                except Exception:
                    pass
                return True
            return False
        except (ValidationError, json.JSONDecodeError) as e:
            print(f"  [EIM History] Validation/JSON Error processing chunk: {e}")
            return False
        except Exception as e:
            print(f"  [EIM History] Unexpected error processing chunk: {e}")
            return False

    def _process_history_chunk_with_fallback(self, utterances_chunk: List[Dict], depth: int = 0) -> bool:
        """
        Process a history chunk with graceful fallback: if validation fails,
        recursively split the chunk into smaller subchunks (with light overlap)
        until processing succeeds or chunks are too small.

        Args:
            utterances_chunk: The chunk to process
            depth: Current recursion depth

        Returns:
            True if any processing succeeded (and merged), else False
        """
        MIN_CHUNK = 6
        MAX_DEPTH = 3

        if not utterances_chunk:
            return True

        ok = self._process_single_history_chunk(utterances_chunk)
        if ok:
            return True

        # Fallback: split if possible
        if depth >= MAX_DEPTH or len(utterances_chunk) <= MIN_CHUNK:
            print("  [EIM History] Fallback exhausted; skipping this chunk.")
            return False

        mid = max(1, len(utterances_chunk) // 2)
        overlap = max(1, self.HISTORY_OVERLAP_SIZE // 2)
        left = utterances_chunk[:mid]
        right_start = max(0, mid - overlap)
        right = utterances_chunk[right_start:]

        print(f"  [EIM History] Splitting chunk (len={len(utterances_chunk)}) into left(len={len(left)}) and right(len={len(right)}) for retry...")
        left_ok = self._process_history_chunk_with_fallback(left, depth + 1)
        right_ok = self._process_history_chunk_with_fallback(right, depth + 1)
        return left_ok or right_ok

    def process_history(self, utterances: List[Dict]):
        """
        Process a long list of historical utterances chunk by chunk with overlap,
        building the internal master_state. Uses adaptive chunking with larger 
        chunks for early meeting content.
        
        Args:
            utterances: List of utterance dictionaries to process
        """
        if not utterances:
            return

        print(f"  [EIM History] Processing {len(utterances)} historical utterances in chunks...")
        self._trace(
            f"HISTORY_START total={len(utterances)} base_chunk_size={self.HISTORY_CHUNK_SIZE} base_overlap={self.HISTORY_OVERLAP_SIZE}"
        )
        
        # Calculate adaptive chunking thresholds
        early_threshold = int(len(utterances) * 0.4)  # First 40% gets larger chunks
        mid_threshold = int(len(utterances) * 0.7)    # Next 30% gets medium chunks
        
        start = 0
        total_utterances = len(utterances)
        
        while start < len(utterances):
            # Determine chunk size based on position in meeting
            if start < early_threshold:
                # Early meeting: use larger chunks for better context
                chunk_size = min(50, self.HISTORY_CHUNK_SIZE * 2)
                overlap = min(20, self.HISTORY_OVERLAP_SIZE * 2)
                context_type = "EARLY"
            elif start < mid_threshold:
                # Mid meeting: use slightly larger chunks
                chunk_size = min(45, int(self.HISTORY_CHUNK_SIZE * 1.5))
                overlap = min(15, int(self.HISTORY_OVERLAP_SIZE * 1.5))
                context_type = "MID"
            else:
                # Late meeting: use normal chunks
                chunk_size = self.HISTORY_CHUNK_SIZE
                overlap = self.HISTORY_OVERLAP_SIZE
                context_type = "LATE"
            
            end = start + chunk_size
            if end > total_utterances: 
                end = total_utterances
                
            chunk = utterances[start:end]
            if not chunk: 
                break

            print(f"    - Processing {context_type} chunk: utterances index {start} to {end-1} (size={len(chunk)})")
            self._trace(f"ADAPTIVE_CHUNK type={context_type} start={start} end={end-1} size={len(chunk)} overlap={overlap}")
            
            success = self._process_history_chunk_with_fallback(chunk, depth=0)
            if not success:
                print("    - Skipping this chunk due to repeated validation errors; continuing with next chunk.")

            # Create overlap for next chunk using adaptive overlap
            start += chunk_size - overlap

        print("  [EIM History] Master state build complete.")
        self._trace(f"HISTORY_DONE | {self._summarize_master_state()}")

    def _derive_last_utterance_intent(self, text: str) -> str:
        stripped = str(text or "").strip()
        if not stripped:
            return "Fragment"
        if is_banal(stripped) or is_fragmentary(stripped):
            return "Acknowledgment"
        if "?" in stripped:
            return "Direct Question"
        return "Statement"

    def _focus_too_close_to_noise(self, focus: str, final_content: str) -> bool:
        focus_tokens = {tok for tok in re.findall(r"[a-z0-9]+", str(focus or "").lower()) if len(tok) > 2}
        final_tokens = {tok for tok in re.findall(r"[a-z0-9]+", str(final_content or "").lower()) if len(tok) > 2}
        if not focus_tokens or not final_tokens:
            return False
        return len(focus_tokens & final_tokens) / max(1, min(len(focus_tokens), len(final_tokens))) >= 0.75

    def _derive_active_focus_from_substantive_tail(self, snapshot_utterances: List[Dict]) -> Optional[str]:
        substantive: List[Dict[str, str]] = []
        for utt in reversed(snapshot_utterances or []):
            if not isinstance(utt, dict):
                continue
            content = str(utt.get("utterance_content") or utt.get("content") or "").strip()
            if not content or is_banal(content) or is_fragmentary(content) or response_substance_score(content) < 0.35:
                continue
            substantive.append(
                {
                    "speaker": str(utt.get("speaker_id") or "Unknown"),
                    "content": re.sub(r"\s+", " ", content),
                }
            )
            if len(substantive) >= 2:
                break
        if not substantive:
            return None
        substantive.reverse()
        if len(substantive) == 1:
            row = substantive[0]
            return f"{row['speaker']}'s active substantive point: {row['content'][:180]}"
        return "Active exchange: " + " / ".join(f"{row['speaker']}: {row['content'][:120]}" for row in substantive)

    def _infer_direct_target(self, text: str) -> Optional[str]:
        raw = str(text or "").strip()
        lowered = raw.lower()
        if not lowered:
            return None
        candidate_names = set((self.master_state.participant_models or {}).keys())
        for alias, mapped in (getattr(self.master_state, "identity_aliases", {}) or {}).items():
            if mapped:
                candidate_names.add(str(alias))
                candidate_names.add(str(mapped))
        for candidate in sorted(candidate_names, key=len, reverse=True):
            token = str(candidate or "").strip()
            if not token:
                continue
            escaped = re.escape(token.lower())
            direct_patterns = (
                rf"^\s*{escaped}\s*[,:\?]",
                rf"\b{escaped}\s*,\s*(can|could|would|will|do|does|did|should|what|how|why|when)\b",
                rf"\b(can|could|would|will|do|does|did|should)\s+{escaped}\b",
                rf"\b(to|from|for)\s+{escaped}\s*[:\?]?$",
            )
            if any(re.search(pattern, lowered) for pattern in direct_patterns):
                return token
        return None

    def _postprocess_snapshot_response(
        self,
        response: EIMMeetingStateVector,
        snapshot_utterances: List[Dict],
    ) -> EIMMeetingStateVector:
        """Enforce non-negotiable EIM freshness invariants after LLM validation."""
        if not response or not snapshot_utterances:
            return response
        final_utt = snapshot_utterances[-1] or {}
        final_id = str(final_utt.get("utterance_id") or "")
        final_speaker = str(final_utt.get("speaker_id") or "")
        final_content = str(final_utt.get("utterance_content") or final_utt.get("content") or "")

        response.last_processed_utterance_id = final_id or response.last_processed_utterance_id
        snapshot_analysis = getattr(response, "snapshot_analysis", None)
        micro = getattr(snapshot_analysis, "micro_topic_analysis", None) if snapshot_analysis else None
        if micro is not None:
            previous_luv = getattr(micro, "last_utterance_verbatim", None)
            previous_id = getattr(previous_luv, "utterance_id", None) if previous_luv else None
            previous_target = getattr(previous_luv, "target_addressed", None) if previous_luv else None
            target_addressed = previous_target if previous_id == final_id else self._infer_direct_target(final_content)
            micro.last_utterance_verbatim = LastUtteranceContext(
                speaker=final_speaker,
                content=final_content,
                utterance_id=final_id,
                intent_type=(
                    getattr(previous_luv, "intent_type", None)
                    if previous_id == final_id and getattr(previous_luv, "intent_type", None)
                    else self._derive_last_utterance_intent(final_content)
                ),
                target_addressed=target_addressed,
            )
            if is_banal(final_content) or is_fragmentary(final_content):
                current_focus = str(getattr(micro, "immediate_focus", "") or "").strip()
                if not current_focus or self._focus_too_close_to_noise(current_focus, final_content):
                    repaired_focus = self._derive_active_focus_from_substantive_tail(snapshot_utterances[:-1])
                    if repaired_focus:
                        micro.immediate_focus = repaired_focus
                        try:
                            micro.immediate_focus_sanity_flag = "repaired_from_noisy_final_utterance"
                        except Exception:
                            pass
                elif not getattr(micro, "immediate_focus_sanity_flag", None):
                    try:
                        micro.immediate_focus_sanity_flag = "checked_noisy_final_utterance"
                    except Exception:
                        pass

        if is_banal(final_content) or is_fragmentary(final_content):
            try:
                snapshot_analysis.key_points_made_in_snapshot = [
                    kp for kp in (snapshot_analysis.key_points_made_in_snapshot or [])
                    if getattr(kp, "utterance_id", None) != final_id
                ]
            except Exception:
                pass
            try:
                for model in (response.participant_models or {}).values():
                    stance_map = getattr(model, "expressed_stance_on_topics", None) or {}
                    for topic, stances in list(stance_map.items()):
                        stance_map[topic] = [
                            stance for stance in (stances or [])
                            if getattr(stance, "source_utterance_id", None) != final_id
                        ]
            except Exception:
                pass
        return response

    def analyze_snapshot(self, snapshot_utterances: List[Dict]) -> Optional[EIMMeetingStateVector]:
        """
        Perform deep, dual-focus analysis on a snapshot, generating a diff to update
        the master_state and returning a comprehensive state vector.
        
        Args:
            snapshot_utterances: List of utterances to analyze as the current snapshot
            
        Returns:
            EIMMeetingStateVector or None if analysis failed
        """
        if not snapshot_utterances:
            return None

        # Trace before snapshot analysis
        try:
            first_id = snapshot_utterances[0].get('utterance_id')
            last_id = snapshot_utterances[-1].get('utterance_id')
            self._trace(f"SNAPSHOT_START ids={first_id}..{last_id} | {self._summarize_master_state()}")
        except Exception:
            self._trace(f"SNAPSHOT_START | {self._summarize_master_state()}")

        historical_context_str = self.master_state.model_dump_json(indent=2)
        snapshot_str = self._format_utterances_for_prompt(snapshot_utterances)

        system_prompt = """You are an expert meeting analyst. Your task is to synthesize a meeting's full history with a specific, immediate snapshot to produce a comprehensive analysis of the current meeting state. This involves two key outputs: a "diff" of changes to the long-term state and a detailed analysis of the snapshot itself.

        ## Core Task: Dual-Focus Analysis & Diff Generation
        1.  **Analyze the `Immediate_Snapshot`** to understand the "present moment." This involves identifying the overall (macro) topic of the snapshot, the specific micro-topic of the last few utterances, the conversational state, the floor status, and the key points made *within the snapshot*. This analysis should be self-contained in the `snapshot_analysis` part of your output.
        2.  **Compare the `Immediate_Snapshot` against the `Historical_Context`** to identify what is new.
        3.  **Generate a "diff" or "patch" object** containing ONLY the new or updated information for the `topic_summary_log` and `participant_models`. Do NOT replicate any data from the `Historical_Context` that has not changed.

        ## Instructions for `snapshot_analysis` fields (Analysis of the present moment):
        -   `micro_topic_analysis`: Hierarchical topic analysis with MESO and MICRO granularity levels:
            - `broader_context` (MESO-level): SPECIFIC thematic focus of the last 4-5 SUBSTANTIVE utterances. Must be MORE SPECIFIC than snapshot_major_topic.
            - `immediate_focus` (MICRO-level): VERY SPECIFIC active focus of the last 1-2 RELEVANT SUBSTANTIVE utterances. Filter phatic, banal, backchannel, fragmentary, and disfluent noise before deciding this field. Must be MORE SPECIFIC than broader_context.
            - `topic_continuity`: "CONTINUATION" if immediate_focus continues the broader_context, "SHIFT" if the latest substantive utterance introduces new focus
            - `last_utterance_verbatim`: Include speaker, exact content, utterance_id, intent_type, and target_addressed for the very last utterance

            **CRITICAL: Topic Granularity Gradient**
            The three topic levels MUST differ in specificity (zoom in progressively):
            
            - `snapshot_major_topic` (MACRO): Broad theme of the ENTIRE snapshot window
              → E.g., "Q4 API development planning meeting"
              
            - `broader_context` (MESO): SPECIFIC sub-topic of last 4-5 SUBSTANTIVE utterances
              → E.g., "Comparing OAuth 2.0 vs API key authentication approaches"
              → Must be MORE SPECIFIC than snapshot_major_topic
              
            - `immediate_focus` (MICRO): VERY SPECIFIC focus of last 1-2 SUBSTANTIVE utterances
              → E.g., "Security lead's concerns about token expiry edge cases in OAuth flow"
              → Must be MORE SPECIFIC than broader_context
            
            **VALIDATION (perform before finalizing):**
            1. Is broader_context more specific than snapshot_major_topic? (Not same/similar wording!)
            2. Is immediate_focus more specific than broader_context?
            3. If they're nearly identical → REVISE to add granularity
            
            **ANTI-PATTERN (WRONG):**
            - snapshot_major_topic: "API security discussion"
            - broader_context: "API security discussion"
            - immediate_focus: "Security measures"
            → All nearly identical, no gradient!
            
            **CORRECT PATTERN:**
            - snapshot_major_topic: "Q4 API development planning meeting"
            - broader_context: "Evaluating OAuth 2.0 implementation trade-offs"
            - immediate_focus: "Security lead asking about token refresh failure edge cases"
            → Each level zooms in more specifically

            **Micro-Topic Freshness Verification**
            Before finalizing `micro_topic_analysis`, perform this verification:
            1. Examine the exact final turn in `last_utterance_verbatim.content`, but first decide whether it is substantive or phatic/noisy.
            2. If the final turn is substantive, ask: "Does my proposed `immediate_focus` describe what the final substantive turn and its immediate setup are DOING or TALKING ABOUT?"
            3. If the final turn is phatic/noisy, ignore it for `immediate_focus` and summarize the most recent active substantive focus instead.
            4. If the immediate_focus describes a PRIOR discussion focus but the latest substantive utterance introduces NEW content/question/topic, this is a SHIFT

            **Staleness Detection Heuristics:**
            - If `immediate_focus` references concepts NOT present or implied in the last 1-2 relevant substantive utterances → it's likely stale
            - If the latest substantive utterance introduces a question, enumeration, new proposal, or explicit topic change but `immediate_focus` describes prior ongoing discussion → update it

            **Robustness for Banal/Phatic Utterances:**
            - If the last 1-2 utterances are banal/phatic (e.g., 'okay', 'mm-hmm', disfluencies, backchannels), do NOT let them dominate `immediate_focus`
            - In such cases, look further back to the last SUBSTANTIVE utterance to determine the true `immediate_focus`
            - `broader_context` should still reflect the ongoing topical arc, not be swayed by phatic acknowledgments

        -   `snapshot_major_topic` (MACRO-level): The broad theme/agenda of the ENTIRE snapshot window. Must be LESS SPECIFIC than broader_context and immediate_focus.
        -   `conversational_state`: Must be one of: 'Problem Diagnosis', 'Brainstorming Solutions', 'Decision Making', 'Social Banter', 'Information Sharing', 'Planning', or any other relevant state, if applicable.
        -   `conversational_floor_status`: Determined by the **last few utterances**. Must be one of:
            -   `'Open_Floor'`: The default state after a simple statement or the conclusion of a topic. The barrier to entry for a Chime_In is low.
            -   `'Open_Question_To_Group'`: A speaker has asked a question to the whole group. The barrier to entry for a relevant Implicit Cue is very low.
            -   `'Directed_Turn (to: SpeakerX)'`: The highest priority signal. The floor belongs exclusively to SpeakerX. All others must be silent.
            -   `'Dyadic_Exchange (between: SpeakerX, SpeakerY)'`: A rapid back-and-forth between two participants over the last few utterances with minimal interjections. If a third party interjects or a facilitator speaks broadly, prefer `Open_Floor` or `Open_Question_To_Group`.
            -   `'Turn_Yielded_Back (to: SpeakerX)'`: A participant has just finished answering a question asked by SpeakerX. The floor is now implicitly yielded back to SpeakerX to acknowledge the answer or ask a follow-up. Others should generally wait.
        -   `floor_rationale`: A one-line rationale explaining why this floor status was selected (e.g., "last utterance is a direct question to <alias>").
        -   `key_points_made_in_snapshot`: Extract a summary of every **substantive point** made *within the snapshot*.
        
            **KEY POINT SPECIFICITY REQUIREMENT:**
            Each key_point must be SELF-CONTAINED and UNAMBIGUOUS:
            
            1. **RESOLVE ANAPHORA**: If utterance contains vague references ("it", "that approach", "this feature"), consult preceding 2-3 utterances to determine the specific referent and include it.
            
            2. **SELF-CONTAINMENT TEST**: Would someone reading ONLY this key_point understand the specific topic without seeing prior context? If not, add context.
            
            3. **LENGTH GUIDANCE**: 15-30 words with essential context; avoid verbatim quotes.
            
            4. **NO HALLUCINATION**: Only include context explicitly present in preceding utterances.
            
            **BEFORE/AFTER EXAMPLE:**
            - Utterance: "I think that's a good approach, but we need to consider the edge cases"
            - Prior context: Discussion about implementing OAuth 2.0 refresh token rotation
            
            ❌ BAD: "Engineer agreed the approach is good but raised concerns about edge cases"
              → "the approach" and "edge cases" are vague
            
            ✅ GOOD: "Engineer endorsed OAuth 2.0 refresh token rotation but raised concerns about token expiry edge cases during network failures"
              → Resolves vague references with specific context


        ## Instructions for `topic_summary_log` and `participant_models` (The "diff" object):
        -   **Topic Management**:
            - Analyze the topic of each new utterance in the snapshot. If it starts a new, more specific topic/sub-topic, create a new entry in the `topic_summary_log`.
            - If it continues or revisits an existing topic, find the existing entry and append any new key points to its `key_points` list.
            - When a topic shift is detected, you **MUST** find the previous "Ongoing" topic(s) in the `topic_summary_log` and change its/their `status` to `"Concluded"`.
            - Note: **Do not append vaguely related points to broad, existing topics. Prefer granularity.**
        -   **Participant Modeling**:
            - **CRITICAL RULE FOR EXPERTISE:** Only add a keyword to a participant's `inferred_expertise_keywords` if they demonstrate **deep knowledge** about it. Simply mentioning a term is **NOT** expertise.
            - **Stances:** When a participant expresses a clear stance on a topic, **APPEND** a new `StanceEntry` object to the list for that topic in their `expressed_stance_on_topics`. **DO NOT OVERWRITE** previous stances.
        -   **Identity Aliases**: Create mappings between personal names and participant IDs/roles when confident:
            - **Explicit Introduction**: Direct statements like "Sarah from Engineering" or "Let's hear from Mike in Marketing"
            - **Response Pattern Recognition**: When a personal name is mentioned and the same participant consistently responds immediately after
            - **Direct Address Response**: When someone uses a personal name to address someone and a specific participant responds as if being addressed
            - **Role Context Clues**: When a personal name is used in contexts that clearly match a participant's expertise/function
            Only create mappings when evidence is clear and unambiguous. Cross-reference with historical context when available.

        ## Output Schema
        - Your output MUST be a complete JSON object that conforms to the `EIMMeetingStateVector` schema.
        - The `topic_summary_log` and `participant_models` fields should ONLY contain the "diff" of changes.
        - The `snapshot_analysis` should be a complete analysis of the `Immediate_Snapshot`.
        - The `last_processed_utterance_id` should be the final utterance in the snapshot.
        ```json
        {{
            "last_processed_utterance_id": "(string) The utterance_id of the last utterance in the snapshot.",
            "snapshot_analysis": {{
                "micro_topic_analysis": {{
                    "broader_context": "(string) Topic label for the last 2-3 utterances as a conversational unit",
                    "immediate_focus": "(string) Active focus of the last relevant substantive utterances after filtering phatic/noisy final turns",
                    "topic_continuity": "(string) 'CONTINUATION' or 'SHIFT'",
                    "last_utterance_verbatim": {{
                        "speaker": "(string) Speaker of the last utterance",
                        "content": "(string) Exact text content",
                        "utterance_id": "(string) Unique identifier",
                        "intent_type": "(string) e.g., 'Direct Question', 'Statement', 'Request'",
                        "target_addressed": "(string, optional) Who is being addressed, if specific"
                    }}
                }},
                "snapshot_major_topic": "(string) The macro topic of the snapshot window.",
                "conversational_state": "(string) The current state of the conversation (e.g., 'Brainstorming').",
                "conversational_floor_status": "(string) The status of who has the right to speak.",
                "floor_rationale": "(string, optional) One-line rationale for the floor decision.",
                "key_points_made_in_snapshot": [
                    {{
                        "point": "(string) A one-sentence summary of a substantive contribution from the snapshot.",
                        "speaker": "(string) The speaker_id of the person who made the point.",
                        "utterance_id": "(string) The utterance_id where the point was made."
                    }}
                ]
            }},
            "topic_summary_log": {{ 
                "...": {{
                    "label": "(string) The human-readable name of the topic. MUST match the dictionary key.",
                    "last_updated_at_utterance_id": "(string or null) The utterance_id of the last key point added to this topic.",
                    "key_points": [ {{...}} ],
                    "status": "(string) The current status of the topic. MUST be either 'Ongoing' or 'Concluded'."
                }}
            }}, // DIFF ONLY
            "participant_models": {{ 
                "...": {{
                    "inferred_expertise_keywords": [ "(string) A keyword representing a domain of expertise demonstrated by this participant." ],
                    "expressed_stance_on_topics": {{ "...": [ {{...}} ] }},
                    "recent_dialogue_act": "(string or null) The dialogue act of the participant's most recent utterance in the chunk (e.g., 'Questioning', 'Proposing', ...)."
                }}
            }}, // DIFF ONLY
            "identity_aliases": {{
                "<Personal Name>": "(string) Participant ID or role this name refers to"
            }}
        }}
        ```
        """

        user_prompt_template = """Please synthesize the historical context with the immediate snapshot to produce the complete meeting state vector, including a "diff" of the changes, but detailed analysis of the current snapshot in the `snapshot_analysis`.

        ---
        ## EXAMPLE

        **Historical Context:**
        ```json
        {{
            "topic_summary_log": {{
                "Q3 Orion Feature Roadmap": {{ "label": "Q3 Orion Feature Roadmap", "status": "Ongoing", "last_updated_at_utterance_id": "MeetingX_001", "key_points": [ ... ] }},
                "Database Choice for Orion": {{ "label": "Database Choice for Orion", "status": "Ongoing", "last_updated_at_utterance_id": "MeetingX_041", "key_points": [ ... ] }}
            }},
            "participant_models": {{ ... }}
        }}
        ```
        **Immediate Snapshot:**
        (Manager): Great, so we're aligned on the database. Let's move on. (MeetingX_044)
        (Lead Dev): To be specific, the current proposal uses a relational database, but our user data is unstructured. A NoSQL solution like MongoDB would be a much better fit for scalability. (MeetingX_045)
        (UX Designer): From a user perspective, does the database choice affect the UI response time? That's my main concern. (MeetingX_046)

        **Desired Output (A complete `snapshot_analysis` and a `diff` for other fields):**
        ```json
        {{
            "last_processed_utterance_id": "MeetingX_046",
            "snapshot_analysis": {{
                "micro_topic_analysis": {{
                    "broader_context": "Database Choice Technical Discussion",
                    "immediate_focus": "UI Response Time Impact Assessment",
                    "topic_continuity": "SHIFT",
                    "last_utterance_verbatim": {{
                        "speaker": "UX Designer",
                        "content": "From a user perspective, does the database choice affect the UI response time? That's my main concern.",
                        "utterance_id": "MeetingX_046",
                        "intent_type": "Direct Question",
                        "target_addressed": "Group"
                    }}
                }},
                "snapshot_major_topic": "Database Choice for Orion",
                "conversational_state": "Problem Diagnosis",
                "conversational_floor_status": "Open_Question_To_Group",
                "floor_rationale": "UX Designer asked a question to the whole group about UI response time",
                "key_points_made_in_snapshot": [
                    {{ "point": "Manager signals conclusion of the 'Database Choice for Orion' topic.", "speaker": "Manager", "utterance_id": "MeetingX_044" }},
                    {{ "point": "Lead Dev advocated for a NoSQL solution like MongoDB, citing better scalability for unstructured user data.", "speaker": "Lead Dev", "utterance_id": "MeetingX_045" }},
                    {{ "point": "UX Designer raised a question about the impact of database choice on UI response time.", "speaker": "UX Designer", "utterance_id": "MeetingX_046" }}
                ]
            }},
            "topic_summary_log": {{
                "Q3 Orion Feature Roadmap": {{
                    "label": "Q3 Orion Feature Roadmap",
                    "status": "Concluded"
                }},
                "Database Choice for Orion": {{
                    "label": "Database Choice for Orion",
                    "last_updated_at_utterance_id": "MeetingX_046",
                    "key_points": [
                        {{ "point": "Lead Dev advocated for a NoSQL solution like MongoDB, citing better scalability for unstructured user data.", "speaker": "Lead Dev", "utterance_id": "MeetingX_045" }},
                        {{ "point": "UX Designer raised a question about the impact of database choice on UI response time.", "speaker": "UX Designer", "utterance_id": "MeetingX_046" }}
                    ],
                    "status": "Ongoing"
                }}
            }},
            "participant_models": {{
                "Lead Dev": {{
                    "inferred_expertise_keywords": ["NoSQL", "Scalability"],
                    "expressed_stance_on_topics": {{
                        "Database Choice for Orion": [
                            {{ "stance_summary": "Advocates for a NoSQL solution due to scalability.", "source_utterance_id": "MeetingX_045" }}
                        ]
                    }},
                    "recent_dialogue_act": "Proposing a solution"
                }},
                "UX Designer": {{
                    "inferred_expertise_keywords": ["User Experience", "UI Performance"],
                    "expressed_stance_on_topics": {{
                        "Database Choice for Orion": [
                            {{ "stance_summary": "Concerned about the impact on UI response time.", "source_utterance_id": "MeetingX_046" }}
                        ]
                    }},
                    "recent_dialogue_act": "Questioning"
                }}
            }}
        }}
        ```
        ---
        
        ## YOUR TASK:
        
        **Historical Context:**
        {historical_context_str}

        **Immediate Snapshot:**
        {snapshot_str}

        Now, provide the complete, synthesized `EIMMeetingStateVector` as a valid JSON object, with `topic_summary_log` and `participant_models` containing only the diff.
        """

        # Escape braces in JSON content to prevent format conflicts with .format()
        escaped_historical_context = historical_context_str.replace('{', '{{').replace('}', '}}')
        escaped_snapshot_str = snapshot_str.replace('{', '{{').replace('}', '}}')
        
        filled_user_prompt = user_prompt_template.format(
            historical_context_str=escaped_historical_context,
            snapshot_str=escaped_snapshot_str
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": filled_user_prompt}
        ]

        try:
            # Get JSON response using json_required=True (no instructor)
            json_response = self.client.call_with_retry(
                model=self.model_name,
                messages=messages,
                max_tokens=4096,
                temperature=0.1,
                json_required=True,  # Force JSON output without instructor
                telemetry_label="perceiver.snapshot",
                telemetry_phase="pre_turn",
            )
            
            # Manually validate the JSON against the EIM schema
            response_with_diff = None
            if json_response:
                response_with_diff = self.client.validate_json_response(json_response, EIMMeetingStateVector)
                if response_with_diff:
                    response_with_diff = self._postprocess_snapshot_response(response_with_diff, snapshot_utterances)
            
            if response_with_diff:
                # Convert Pydantic objects to dictionaries before creating EIMMasterState
                topic_log_dict = {}
                if hasattr(response_with_diff, 'topic_summary_log') and response_with_diff.topic_summary_log:
                    topic_log_dict = {k: v.model_dump() if hasattr(v, 'model_dump') else v 
                                    for k, v in response_with_diff.topic_summary_log.items()}
                
                participant_models_dict = {}
                if hasattr(response_with_diff, 'participant_models') and response_with_diff.participant_models:
                    participant_models_dict = {k: v.model_dump() if hasattr(v, 'model_dump') else v 
                                             for k, v in response_with_diff.participant_models.items()}
                
                # Create the diff object to merge
                diff_to_merge = EIMMasterState(
                    topic_summary_log=topic_log_dict,
                    participant_models=participant_models_dict
                )
                
                # Trace snapshot diff summary (note: identity_aliases from snapshot are not merged here)
                try:
                    t_keys = list((response_with_diff.topic_summary_log or {}).keys())
                    p_keys = list((response_with_diff.participant_models or {}).keys())
                    concluded = [k for k, v in (response_with_diff.topic_summary_log or {}).items() if getattr(v, 'status', '').lower().startswith('conclud')]
                    aliases = list((getattr(response_with_diff, 'identity_aliases', {}) or {}).keys())
                    sa = getattr(response_with_diff, 'snapshot_analysis', None)
                    sa_bits = []
                    if sa:
                        # Handle both new enhanced structure and legacy structure
                        micro_topic = getattr(sa, 'effective_micro_topic', None)
                        if micro_topic:
                            sa_bits.append(f"micro={micro_topic}")
                        if hasattr(sa, 'micro_topic_analysis') and sa.micro_topic_analysis:
                            sa_bits.append(f"continuity={sa.micro_topic_analysis.topic_continuity}")
                        if getattr(sa, 'snapshot_major_topic', None):
                            sa_bits.append(f"major={sa.snapshot_major_topic}")
                        if getattr(sa, 'conversational_floor_status', None):
                            sa_bits.append(f"floor={sa.conversational_floor_status}")
                    sa_str = ", ".join(sa_bits)
                    self._trace(
                        f"SNAPSHOT_DIFF topics+~={len(t_keys)} concluded={len(concluded)} participants~={len(p_keys)} aliases_in_resp={len(aliases)} | {sa_str}"
                    )
                except Exception:
                    self._trace("SNAPSHOT_DIFF <unable to summarize>")

                # Merge the diff into the master state
                self._merge_state_updates(diff_to_merge)
                
                # Construct the final, complete state vector to return
                # Convert snapshot_analysis to dictionary if it's a Pydantic object
                snapshot_analysis_dict = response_with_diff.snapshot_analysis
                if hasattr(response_with_diff.snapshot_analysis, 'model_dump'):
                    snapshot_analysis_dict = response_with_diff.snapshot_analysis.model_dump()
                
                final_state_vector = EIMMeetingStateVector(
                    last_processed_utterance_id=response_with_diff.last_processed_utterance_id,
                    snapshot_analysis=snapshot_analysis_dict,
                    topic_summary_log=self.master_state.topic_summary_log,
                    participant_models=self.master_state.participant_models,
                    identity_aliases=getattr(self.master_state, 'identity_aliases', {}) or {}
                )
                
                print(f"  [EIM Snapshot] Successfully analyzed snapshot ending with {final_state_vector.last_processed_utterance_id}.")
                self._trace(
                    f"SNAPSHOT_DONE id={final_state_vector.last_processed_utterance_id} | {self._summarize_master_state()}"
                )
                return final_state_vector
            
            return None

        except (ValidationError, json.JSONDecodeError) as e:
            print(f"  [EIM Snapshot] ❌ Validation failed for snapshot. Error: {e}")
            return None
        except Exception as e:
            print(f"  [EIM Snapshot] ❌ API call failed: {e}")
            return None

    def format_meeting_history_for_prompt(self, utterances: List[Dict]) -> str:
        """
        Format meeting history for prompt inclusion in LLM calls.
        Helper method for testing and external use.
        
        Args:
            utterances: List of utterance dictionaries
            
        Returns:
            Formatted string representation of the meeting history
        """
        return self._format_utterances_for_prompt(utterances)

    def format_snapshot_for_prompt(self, utterances: List[Dict]) -> str:
        """
        Format a snapshot of utterances for prompt inclusion in LLM calls.
        Helper method for testing and external use.
        
        Args:
            utterances: List of utterance dictionaries representing a snapshot
            
        Returns:
            Formatted string representation of the snapshot
        """
        return self._format_utterances_for_prompt(utterances)
