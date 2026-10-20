"""
CAPA Recalibrator (LLM-driven fusion): merges environment and delegate error
reports with the current ground-truth turn to produce an EIMMasterState diff.
All updates are append-only and anchored to the current utterance.
"""


from typing import Dict, Optional, List, Any
import json
try:
    from ..llm_client import LLMClient
    from ..schemas import (
        EIMMasterState,
        EIMMeetingStateVector,
        SnapshotAnalysis,
    )
except ImportError:
    from llm_client import LLMClient
    from schemas import (
        EIMMasterState,
        EIMMeetingStateVector,
        SnapshotAnalysis,
    )
from .types import ErrorReport, DelegateErrorReport, RecalibrationOutput, ControllerGuidance, GuidanceBundle


class CapaRecalibrator:
    def __init__(self, client: LLMClient, late_window_k: int = 5, guidance_enabled: bool = False):
        self.client = client
        self.model_name = client.model
        self.k = late_window_k
        self.guidance_enabled = guidance_enabled

    def _format_utterances(self, utterances: List[Dict]) -> str:
        lines = []
        for u in utterances or []:
            sp = u.get("speaker_id", "Unknown")
            tx = u.get("utterance_content", "")
            uid = u.get("utterance_id", "Unknown")
            lines.append(f"({sp}): {tx} ({uid})")
        return "\n".join(lines)

    def _compact_snapshot(self, eim_state: EIMMeetingStateVector) -> Dict:
        try:
            sa = eim_state.snapshot_analysis.model_dump()
        except Exception:
            sa = {}
        topics = {}
        try:
            for lbl, entry in (eim_state.topic_summary_log or {}).items():
                # Tail of key points / open questions / decisions keeps prompt small
                kp_tail = []
                for kp in (entry.key_points or [])[-3:]:
                    kp_tail.append({
                        "point": kp.point,
                        "speaker": kp.speaker,
                        "utterance_id": kp.utterance_id,
                        "kind": getattr(kp, 'kind', None),
                    })
                topics[lbl] = {
                    "label": entry.label,
                    "last_updated_at_utterance_id": entry.last_updated_at_utterance_id,
                    "status": entry.status,
                    "key_points_tail": kp_tail,
                    "open_questions_tail": [
                        {
                            "question_summary": oq.question_summary,
                            "raised_by": oq.raised_by,
                            "source_utterance_id": oq.source_utterance_id,
                        }
                        for oq in (getattr(entry, 'open_questions', []) or [])[-3:]
                    ],
                    "decisions_tail": [
                        {
                            "decision_summary": de.decision_summary,
                            "decided_by": de.decided_by,
                            "source_utterance_id": de.source_utterance_id,
                            "status": de.status,
                        }
                        for de in (getattr(entry, 'decisions', []) or [])[-3:]
                    ],
                }
        except Exception:
            topics = {}

        participants = {}
        try:
            for pid, pm in (eim_state.participant_models or {}).items():
                # Only minimal ToM fields to keep compact
                stances_min = {}
                try:
                    for tlabel, st_list in (pm.expressed_stance_on_topics or {}).items():
                        if st_list:
                            st = st_list[-1]
                            stances_min[tlabel] = {
                                "stance_summary": getattr(st, 'stance_summary', None),
                                "source_utterance_id": getattr(st, 'source_utterance_id', None),
                            }
                except Exception:
                    stances_min = {}
                participants[pid] = {
                    "recent_dialogue_act": getattr(pm, 'recent_dialogue_act', None),
                    "engagement_score": getattr(pm, 'engagement_score', None),
                    "inferred_expertise_keywords": list(getattr(pm, 'inferred_expertise_keywords', []) or []),
                    "last_stances": stances_min,
                }
        except Exception:
            participants = {}

        try:
            aliases = dict(getattr(eim_state, 'identity_aliases', {}) or {})
        except Exception:
            aliases = {}

        return {
            "snapshot_analysis": sa,
            "topic_summary_log": topics,
            "participant_models": participants,
            "identity_aliases": aliases,
        }

    def build_fusion(
        self,
        eim_state: EIMMeetingStateVector,
        last_utterances: List[Dict],
        current_gt_turn: Dict,
        env_error: ErrorReport,
        delegate_error: Optional[DelegateErrorReport] = None,
        delegate_utterance: Optional[str] = None,
        # Tracking info for guidance computation
        matched_unit_count: int = 0,
        total_unit_count: int = 0,
        is_in_cooldown: bool = False,
        episode_contribution_count: int = 0,
        participant_id: Optional[str] = None,  # Who the delegate represents
        delegate_covered_anchor_earlier: bool = False,  # If delegate already matched anchor's idea unit
    ) -> Optional[RecalibrationOutput]:
        """
        Produce EIMMasterState diff, SnapshotAnalysis, and optionally ControllerGuidance.
        Guidance is only produced when self.guidance_enabled=True.
        """
        compact = self._compact_snapshot(eim_state)
        # Render current context lines (t-N..t)
        ctx_lines = self._format_utterances(last_utterances)
        # JSON blocks
        env_json = json.dumps(env_error.model_dump(), ensure_ascii=False, indent=2) if env_error else "null"
        compact_json = json.dumps(compact, ensure_ascii=False, indent=2)
        gt_json = json.dumps({
            "speaker_id": current_gt_turn.get("speaker_id"),
            "utterance_id": current_gt_turn.get("utterance_id"),
            "utterance_content": current_gt_turn.get("utterance_content"),
        }, ensure_ascii=False, indent=2)
        # Note: delegate_error and delegate_utterance are passed directly to _generate_guidance
        # (guidance is produced by Feedback Coach, not included in main state-update prompt)


        system_prompt = (
            "You are an expert meeting state updater. After observing what actually happened in a meeting turn (the 'anchor'), "
            "you update the structured meeting state to reflect reality.\n\n"
            "---\n\n"
            "## Schema Reference\n\n"
            "**KeyPoint:** point (str), speaker (str), utterance_id (str), kind ('fact'|'proposal'|'decision'|'question'|'novel_insight'|'opposing_view'|'confirmation')\n\n"
            "**OpenQuestion:** question_summary (str), raised_by (str), source_utterance_id (str), resolved (bool), resolved_at_utterance_id (str|null)\n\n"
            "**DecisionEntry:** decision_summary (str), decided_by (str), source_utterance_id (str), status ('Tentative'|'Final'), confidence (0-1)\n\n"
            "---\n\n"
            "## Inputs\n\n"
            "1. **Context Window** (t-N+1..t): Prior utterances BEFORE anchor\n"
            "2. **Anchor GT Turn** (t+1): The turn that just occurred — your PRIMARY extraction source\n"
            "3. **Environment ErrorReport**: Predictor judgment (category: Aligned-Immediate/Late, Missed-Cue, Novel, Contradiction, Hallucination, Banal; flags: gt_is_banal, novelty)\n"
            "4. **Compact EIM Snapshot**: Current state BEFORE this turn (topic_summary_log, participant_models)\n\n"
            "---\n\n"
            "## Category Semantics (from Environment ErrorReport)\n\n"
            "| Category | Diff Action |\n"
            "|----------|-------------|\n"
            "| Aligned-Immediate/Late | Append key_point from anchor (if substantive) |\n"
            "| Topic-Only | Append key_point, note partial alignment |\n"
            "| Missed-Cue | Append key_point (valid content predictor missed) |\n"
            "| Novel | Create/extend topic with kind='novel_insight' |\n"
            "| Contradiction | Append key_point with kind='opposing_view' |\n"
            "| Hallucination | Ignore prediction; extract anchor if substantive |\n"
            "| Banal | Usually no key_point (see Banal Anchor Handling) |\n\n"
            "---\n\n"
            "## CORE EXTRACTION RULES\n\n"
            "### Rule 1: Anchor-Only for EIM Updates\n"
            "ALL diff content (key_points, decisions, stances) MUST come from the ANCHOR turn.\n"
            "- Delegate behavior → guidance signals ONLY, never EIM updates\n\n"
            "### Rule 2: Anti-Sparsity (CRITICAL)\n"
            "Before returning, verify diff is NOT sparse:\n"
            "1. If anchor is substantive (gt_is_banal=false) → diff.key_points MUST have ≥1 entry\n"
            "2. If snapshot has 3+ key_points NOT in topic_log → diff.key_points MUST have ≥2 entries\n"
            "3. Empty diff when snapshot has content = EXTRACTION FAILURE\n\n"
            "### Rule 3: Key Point Extraction\n"
            "For substantive anchors, extract:\n"
            "- KeyPoint: Concise summary with RESOLVED anaphora (self-contained)\n"
            "  - BAD: 'PM agreed the approach is good' (what approach?)\n"
            "  - GOOD: 'PM endorsed OAuth token rotation but raised cost concerns'\n"
            "- OpenQuestion: If anchor asks a question → add both KeyPoint(kind='question') AND OpenQuestion\n"
            "- DecisionEntry: If anchor concludes → add both KeyPoint(kind='decision') AND DecisionEntry(status='Tentative')\n"
            "- Participant update: Update recent_dialogue_act, stances, expertise_keywords for anchor speaker\n\n"
            "### Rule 4: Topic Updates\n"
            "- New topic: CREATE entry with status='Ongoing'\n"
            "- Existing topic: APPEND key_point\n"
            "- Concluded topic revived: Set status='Ongoing', append key_point\n"
            "- Mark concluded: ONLY when anchor explicitly signals end\n\n"
            "### Rule 5: Questions & Decisions\n"
            "**OpenQuestion:**\n"
            "- CREATE: Anchor asks question → add OpenQuestion + KeyPoint(kind='question')\n"
            "- RESOLVE: Anchor answers visible prior question → re-emit OpenQuestion with resolved=true, resolved_at_utterance_id=anchor\n\n"
            "**DecisionEntry (Immutable):**\n"
            "- CREATE: Anchor reaches conclusion → add DecisionEntry(status='Tentative') + KeyPoint(kind='decision')\n"
            "- FINALIZE: Anchor reaffirms prior decision → emit NEW DecisionEntry(status='Final') anchored to THIS turn (never update old entry)\n\n"
            "---\n\n"
            "## Banal Anchor Handling (gt_is_banal=true)\n\n"
            "When anchor is phatic ('Yeah', 'Okay', etc.):\n"
            "- **Diff**: Still check anti-sparsity. If snapshot has 3+ substantive points NOT in topic_log → migrate top 2-3 to diff\n"
            "- **Snapshot**: Complete all fields; immediate_focus reflects last SUBSTANTIVE utterance (not the banal one)\n\n"
            "---\n\n"
            "## Snapshot Analysis\n\n"
            "The snapshot_analysis represents window (t-N+1..t+1). MUST be fresh, not copied from prior.\n\n"
            "**Required Fields:**\n"
            "- snapshot_major_topic (MACRO): Broad theme of full window\n"
            "- micro_topic_analysis:\n"
            "  - broader_context (MESO): Sub-topic of last 4-5 substantive utterances (MORE specific than MACRO)\n"
            "  - immediate_focus (MICRO): Focus of last 1-2 substantive utterances (MORE specific than MESO)\n"
            "  - topic_continuity: 'CONTINUATION' or 'SHIFT'\n"
            "  - last_utterance_verbatim: Speaker, content, utterance_id, intent_type, target_addressed for ANCHOR\n"
            "- conversational_state: Problem Diagnosis | Brainstorming | Decision Making | Information Sharing\n"
            "- conversational_floor_status: Open_Floor | Open_Question_To_Group | Directed_Turn (to: X) | Dyadic_Exchange (between: X, Y) | Turn_Yielded_Back (to: X)\n"
            "- floor_rationale: One-line explanation\n"
            "- key_points_made_in_snapshot: Summary of substantive points in window (INCLUDE anchor if substantive)\n\n"
            "**Topic Granularity Examples (MUST differ in specificity):**\n"
            "- MACRO: 'Q4 API development planning'\n"
            "- MESO: 'Comparing OAuth 2.0 vs API key authentication'\n"
            "- MICRO: 'Security lead asking about token refresh edge cases'\n"
            "→ If MACRO ≈ MESO ≈ MICRO → REVISE to add granularity!\n\n"
            "**Freshness Check:**\n"
            "- last_utterance_verbatim MUST match anchor\n"
            "- immediate_focus MUST reflect anchor content (not stale prior focus)\n"
            "- Example: prior='budget discussion', anchor='What's the API latency?' → immediate_focus='API latency inquiry'\n\n"
            "---\n\n"
            +
            "## Output Contract (STRICT)\n\n"
            "Return JSON with these keys:\n\n"
            "1. **diff**: `EIMMasterState` patch\n"
            "   - Must match schema: `topic_summary_log` (dict of topics) and `participant_models` (dict of participants).\n"
            "   - **CRITICAL**: `open_questions` and `decisions` lists must be nested INSIDE the relevant topic object in `topic_summary_log`.\n"
            "   - Every entry references anchor utterance_id\n\n"
            "2. **snapshot_analysis**: Fresh SnapshotAnalysis object\n"
            "   - Complete for window (t-N+1..t+1)\n"
            "   - Always populated, even for banal anchor\n\n"
            "3. **anchor_utterance_id**: String matching anchor's utterance_id\n\n"
            +
            "---\n\n"
            "## PRE-OUTPUT VERIFICATION CHECKLIST (MANDATORY)\n\n"
            "Before returning, verify:\n"
            "1. **Diff Completeness**: If anchor is substantive (gt_is_banal=false) → diff.key_points MUST have ≥1 entry\n"
            "2. **Snapshot Migration**: If snapshot_key_points has 3+ entries NOT in topic_log → diff.key_points MUST have ≥2 entries\n"
            "3. **Participant Update**: If anchor speaker said something substantive → participant_models should update stance or dialogue_act\n"
            "4. **Anchor Freshness**: snapshot_analysis.micro_topic_analysis.last_utterance_verbatim MUST match anchor\n"
            "5. **EMPTY DIFF = FAILURE**: Empty key_points for substantive anchor indicates extraction failure. Re-read anchor.\n\n"
            "---\n\n"
            "## GOLD EXAMPLES (Reference these for output format)\n\n"
            "### EXAMPLE 1: Substantive Anchor (Decision)\n\n"
            "**Input Scenario:**\n"
            "- Context: Discussion about API latency optimization\n"
            "- Anchor: (PM): \"Let's go with Option B - it reduces latency by 40%.\" (Meeting_205)\n"
            "- gt_is_banal: false\n"
            "- Prior snapshot has 3 key_points about latency\n\n"
            "**Expected Output:**\n"
            "```json\n"
            "{\n"
            '  "diff": {\n'
            '    "topic_summary_log": {\n'
            '      "API Latency Optimization": {\n'
            '        "label": "API Latency Optimization",\n'
            '        "last_updated_at_utterance_id": "Meeting_205",\n'
            '        "status": "Ongoing",\n'
            '        "key_points": [\n'
            '          {\n'
            '            "point": "PM decided to proceed with Option B due to 40% latency reduction",\n'
            '            "speaker": "Project Manager",\n'
            '            "utterance_id": "Meeting_205",\n'
            '            "kind": "decision"\n'
            '          }\n'
            '        ],\n'
            '        "open_questions": [],\n'
            '        "decisions": [\n'
            '          {\n'
            '            "decision_summary": "Team will implement Option B for latency optimization",\n'
            '            "decided_by": "Project Manager",\n'
            '            "source_utterance_id": "Meeting_205",\n'
            '            "status": "Tentative",\n'
            '            "confidence": 0.8\n'
            '          }\n'
            '        ]\n'
            '      }\n'
            '    },\n'
            '    "participant_models": {\n'
            '      "Project Manager": {\n'
            '        "recent_dialogue_act": "Deciding",\n'
            '        "inferred_expertise_keywords": ["Project Coordination"],\n'
            '        "expressed_stance_on_topics": {\n'
            '          "API Latency Optimization": [{\n'
            '            "stance_summary": "Favors Option B for latency benefits",\n'
            '            "source_utterance_id": "Meeting_205"\n'
            '          }]\n'
            '        }\n'
            '      }\n'
            '    },\n'
            '    "identity_aliases": {}\n'
            '  },\n'
            '  "snapshot_analysis": {\n'
            '    "snapshot_major_topic": "Q4 API Development",\n'
            '    "micro_topic_analysis": {\n'
            '      "broader_context": "Comparing latency optimization approaches",\n'
            '      "immediate_focus": "PM decision on Option B for 40% latency reduction",\n'
            '      "topic_continuity": "CONTINUATION",\n'
            '      "last_utterance_verbatim": {\n'
            '        "speaker": "Project Manager",\n'
            '        "content": "Let\'s go with Option B - it reduces latency by 40%.",\n'
            '        "utterance_id": "Meeting_205",\n'
            '        "intent_type": "Decision",\n'
            '        "target_addressed": "Group"\n'
            '      }\n'
            '    },\n'
            '    "conversational_state": "Decision Making",\n'
            '    "conversational_floor_status": "Open_Floor",\n'
            '    "floor_rationale": "PM made a decision, floor open for reactions",\n'
            '    "key_points_made_in_snapshot": [\n'
            '      {"point": "Engineer proposed Option A with 25% improvement", "speaker": "Engineer", "utterance_id": "Meeting_201", "kind": "proposal"},\n'
            '      {"point": "Designer raised concerns about Option A complexity", "speaker": "Designer", "utterance_id": "Meeting_203", "kind": "opposing_view"},\n'
            '      {"point": "PM decided on Option B for 40% latency reduction", "speaker": "Project Manager", "utterance_id": "Meeting_205", "kind": "decision"}\n'
            '    ]\n'
            '  },\n'
            '  "anchor_utterance_id": "Meeting_205"\n'
            "}\n"
            "```\n\n"
            "**Why this is correct:**\n"
            "- diff.key_points: 1 entry (anchor content extracted)\n"
            "- diff.decisions: Captures the decision formally\n"
            "- participant_models: PM's stance and dialogue_act updated\n"
            "- snapshot_analysis: last_utterance_verbatim matches anchor exactly\n"
            "- key_points_made_in_snapshot: Includes anchor as the last entry\n"
            +
            "\n"
            "---\n\n"
            "### EXAMPLE 2: Banal Anchor (Acknowledgment with Snapshot Migration)\n\n"
            "**Input Scenario:**\n"
            "- Context: Prior discussion had 4 substantive points\n"
            "- Anchor: (Designer): \"Okay.\" (Meeting_210)\n"
            "- gt_is_banal: true\n"
            "- Prior snapshot has 4 key_points NOT in topic_log\n\n"
            "**Expected Output:**\n"
            "```json\n"
            "{\n"
            '  "diff": {\n'
            '    "topic_summary_log": {\n'
            '      "API Latency Optimization": {\n'
            '        "label": "API Latency Optimization",\n'
            '        "last_updated_at_utterance_id": "Meeting_210",\n'
            '        "status": "Ongoing",\n'
            '        "key_points": [\n'
            '          {\n'
            '            "point": "Engineer proposed caching strategy for API responses",\n'
            '            "speaker": "Engineer",\n'
            '            "utterance_id": "Meeting_208",\n'
            '            "kind": "proposal"\n'
            '          },\n'
            '          {\n'
            '            "point": "PM decided on Option B for latency optimization",\n'
            '            "speaker": "Project Manager",\n'
            '            "utterance_id": "Meeting_205",\n'
            '            "kind": "decision"\n'
            '          }\n'
            '        ],\n'
            '        "open_questions": [],\n'
            '        "decisions": []\n'
            '      }\n'
            '    },\n'
            '    "participant_models": {\n'
            '      "Designer": {\n'
            '        "recent_dialogue_act": "Acknowledging",\n'
            '        "inferred_expertise_keywords": [],\n'
            '        "expressed_stance_on_topics": {}\n'
            '      }\n'
            '    },\n'
            '    "identity_aliases": {}\n'
            '  },\n'
            '  "snapshot_analysis": {\n'
            '    "snapshot_major_topic": "Q4 API Development",\n'
            '    "micro_topic_analysis": {\n'
            '      "broader_context": "Discussing API caching and latency improvements",\n'
            '      "immediate_focus": "Designer acknowledgment of caching proposal",\n'
            '      "topic_continuity": "CONTINUATION",\n'
            '      "last_utterance_verbatim": {\n'
            '        "speaker": "Designer",\n'
            '        "content": "Okay.",\n'
            '        "utterance_id": "Meeting_210",\n'
            '        "intent_type": "Acknowledgment",\n'
            '        "target_addressed": "Group"\n'
            '      }\n'
            '    },\n'
            '    "conversational_state": "Information Sharing",\n'
            '    "conversational_floor_status": "Open_Floor",\n'
            '    "floor_rationale": "Designer acknowledged; floor open",\n'
            '    "key_points_made_in_snapshot": [\n'
            '      {"point": "PM decided on Option B for latency optimization", "speaker": "Project Manager", "utterance_id": "Meeting_205", "kind": "decision"},\n'
            '      {"point": "Engineer proposed caching strategy for API responses", "speaker": "Engineer", "utterance_id": "Meeting_208", "kind": "proposal"}\n'
            '    ]\n'
            '  },\n'
            '  "anchor_utterance_id": "Meeting_210"\n'
            "}\n"
            "```\n\n"
            "**Why this is correct:**\n"
            "- Anchor is banal, but diff STILL has key_points (migrated from snapshot)\n"
            "- The 2 most significant prior points are added to diff.key_points\n"
            "- participant_models: Only dialogue_act updated (banal anchor has no stance)\n"
            "- snapshot_analysis: last_utterance_verbatim matches anchor (even though banal)\n"
            "- This prevents sparsity by migrating transient snapshot points to persistent log\n\n"
            "---\n\n"
            "### EXAMPLE 3: Question-Asking Anchor (OpenQuestion + KeyPoint Dual Creation)\n\n"
            "**Input Scenario:**\n"
            "- Context: Prior discussion about budget allocation for Q4\n"
            "- Anchor: (Engineer): \"What's the expected latency requirement for the new API endpoint?\" (Meeting_215)\n"
            "- gt_is_banal: false\n"
            "- This is a TOPIC SHIFT from budget to API latency\n\n"
            "**Expected Output:**\n"
            "```json\n"
            "{\n"
            '  "diff": {\n'
            '    "topic_summary_log": {\n'
            '      "API Latency Requirements": {\n'
            '        "label": "API Latency Requirements",\n'
            '        "last_updated_at_utterance_id": "Meeting_215",\n'
            '        "status": "Ongoing",\n'
            '        "key_points": [\n'
            '          {\n'
            '            "point": "Engineer asked about the expected latency requirement for the new API endpoint",\n'
            '            "speaker": "Engineer",\n'
            '            "utterance_id": "Meeting_215",\n'
            '            "kind": "question"\n'
            '          }\n'
            '        ],\n'
            '        "open_questions": [\n'
            '          {\n'
            '            "question_summary": "What is the expected latency requirement for the new API endpoint?",\n'
            '            "raised_by": "Engineer",\n'
            '            "source_utterance_id": "Meeting_215",\n'
            '            "resolved": false,\n'
            '            "resolved_at_utterance_id": null\n'
            '          }\n'
            '        ],\n'
            '        "decisions": []\n'
            '      }\n'
            '    },\n'
            '    "participant_models": {\n'
            '      "Engineer": {\n'
            '        "recent_dialogue_act": "Questioning",\n'
            '        "inferred_expertise_keywords": ["API Development"],\n'
            '        "expressed_stance_on_topics": {}\n'
            '      }\n'
            '    },\n'
            '    "identity_aliases": {}\n'
            '  },\n'
            '  "snapshot_analysis": {\n'
            '    "snapshot_major_topic": "Q4 Planning and Technical Requirements",\n'
            '    "micro_topic_analysis": {\n'
            '      "broader_context": "Transitioning from budget discussion to technical requirements",\n'
            '      "immediate_focus": "Engineer inquiry about API latency requirements",\n'
            '      "topic_continuity": "SHIFT",\n'
            '      "last_utterance_verbatim": {\n'
            '        "speaker": "Engineer",\n'
            '        "content": "What\'s the expected latency requirement for the new API endpoint?",\n'
            '        "utterance_id": "Meeting_215",\n'
            '        "intent_type": "Question",\n'
            '        "target_addressed": "Group"\n'
            '      }\n'
            '    },\n'
            '    "conversational_state": "Questioning",\n'
            '    "conversational_floor_status": "Open_Question_To_Group",\n'
            '    "floor_rationale": "Engineer posed open question to the group; awaiting response",\n'
            '    "key_points_made_in_snapshot": [\n'
            '      {"point": "Engineer asked about API latency requirements", "speaker": "Engineer", "utterance_id": "Meeting_215", "kind": "question"}\n'
            '    ]\n'
            '  },\n'
            '  "anchor_utterance_id": "Meeting_215"\n'
            "}\n"
            "```\n\n"
            "**Why this is correct:**\n"
            "- BOTH KeyPoint(kind='question') AND OpenQuestion created (Rule 5 compliance)\n"
            "- New topic 'API Latency Requirements' created (anchor introduced new subject)\n"
            "- topic_continuity='SHIFT' because anchor changed focus from budget to API\n"
            "- conversational_floor_status='Open_Question_To_Group' (not Open_Floor)\n"
            "- immediate_focus reflects anchor's actual question, not prior budget discussion\n\n"
            "---\n\n"
            "No prose outside JSON."
        )



        user_prompt = (
            f"Context Window (t-{self.k}..t):\n{ctx_lines}\n\n"
            f"Anchor GT Turn (t+1):\n{gt_json}\n\n"
            f"Environment ErrorReport:\n{env_json}\n\n"
            f"Current Compact EIM Snapshot:\n{compact_json}\n\n"
            "Produce the JSON update:"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        json_response = self.client.call_with_retry(
            model=self.model_name,
            messages=messages,
            max_tokens=4000,
            temperature=0.1,
            json_required=True,
            telemetry_label="recalibrator.fusion",
            telemetry_phase="post_turn",
        )
        if not json_response:
            return None

        # Expect a dict with 'diff', 'snapshot_analysis', 'anchor_utterance_id', and optional 'guidance'
        try:
            # Parse the JSON string to get the object structure
            obj: Dict[str, Any] = json.loads(json_response) if isinstance(json_response, str) else json_response
            
            # Validate diff and snapshot separately against schemas, if present
            diff_obj = None
            if obj.get("diff") is not None:
                try:
                    diff_payload = obj.get("diff")
                    # Guard against malformed list payloads
                    if isinstance(diff_payload, list):
                        diff_obj = None
                    else:
                        diff_obj = self.client.validate_json_response(
                            json.dumps(diff_payload),
                            EIMMasterState
                        )
                except Exception:
                    diff_obj = None
                
            sa_obj = None
            if obj.get("snapshot_analysis") is not None:
                try:
                    # Convert snapshot_analysis dict to JSON string for validate_json_response
                    sa_obj = self.client.validate_json_response(
                        json.dumps(obj.get("snapshot_analysis")), 
                        SnapshotAnalysis
                    )
                except Exception:
                    sa_obj = None
            
            # === STEP 2: Feedback Coach (if guidance enabled) ===
            guidance_obj = None
            if self.guidance_enabled and sa_obj is not None:
                guidance_obj = self._generate_guidance(
                    delegate_error=delegate_error,
                    delegate_utterance=delegate_utterance,
                    snapshot_analysis=sa_obj,
                    anchor_turn=current_gt_turn,
                    context_window=last_utterances,
                    matched_unit_count=matched_unit_count,
                    total_unit_count=total_unit_count,
                    is_in_cooldown=is_in_cooldown,
                    episode_contribution_count=episode_contribution_count,
                    participant_id=participant_id,
                    delegate_covered_anchor_earlier=delegate_covered_anchor_earlier,
                )
            
            anchor_id = obj.get("anchor_utterance_id") or current_gt_turn.get("utterance_id")
            return RecalibrationOutput(
                diff=diff_obj,
                snapshot_analysis=sa_obj,
                guidance=guidance_obj,
                anchor_utterance_id=anchor_id,
            )
        except Exception as e:
            print(f"[Recalibrator] ERROR: Unexpected exception in build_fusion: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _feedback_coach_prompt(self) -> str:
        """System prompt for the Feedback Coach helper."""
        return '''You are an expert evaluator providing performance feedback for an AI meeting delegate.

## Your Role

You assess how well the delegate's contribution aligned with what actually happened in the meeting (the "ground truth anchor"). Your feedback helps the delegate calibrate future decisions about when to speak, what to say, and how to stay on-topic.

## Context You Receive

1. **Recent conversation context** — The last few utterances before the anchor, showing conversation flow
2. **Anchor turn** — What actually happened in the meeting (speaker, content) — this is ground truth
3. **Delegate's action** — What the delegate said (or "Silent" if they didn't speak)
4. **Delegate judge report** — Evaluation of the delegate's action with quality flags and scores:
   - `matched_utt_id`: If the delegate matched ground truth, this is the matched ID (null = no match)
   - `timing_delta`: Timing relative to anchor:
     - 0 = exact match (delegate spoke at anchor turn)
     - N > 0 = early (delegate spoke N turns BEFORE anchor — anticipatory)
     - null = no match or silent
   - `floor_violation`: true if the delegate mishandled the conversational floor
   - `floor_violation_kind`: "intrusion" (spoke when shouldn't) or "omission" (silent when should speak)
   - `off_topic`: true if the delegate diverged from the current discussion focus
   - `redundancy`: true if the delegate repeated already-covered content
   - `category`: Overall assessment ("Good", "FloorViolation", "OffTopic", "Redundant", etc.)
5. **Updated meeting state** — The floor status and immediate focus AFTER the anchor occurred
6. **Tracking signals** — Qualitative indicators (not exact counts to avoid target chasing):
   - `coverage_status`: "none" | "low" | "moderate" | "high" | "no_targets"
   - `contribution_level`: "none" | "light" | "moderate" | "active"
   - `is_in_cooldown`: true if delegate just made a matched contribution (expires after ~2 turns)
7. **Prior coverage** — Whether delegate already covered the anchor's content:
   - `delegate_covered_anchor_earlier`: true if delegate matched this anchor's idea unit in an earlier turn

## How to Analyze

**Compare the delegate's action against the anchor:**
- Did the delegate speak when the anchor shows it was their turn? (match vs omission)
- Did the delegate stay silent when someone else took the floor? (appropriate_silence vs intrusion)
- Did the delegate's content align with the anchor's topic? (relevant vs off_topic)
- Did the delegate add value or repeat what was said? (fresh vs redundant)

**Use all available signals holistically:**
- The `matched_utt_id` and `timing_delta` tell you if the delegate's contribution matched ground truth
- The `floor_violation` flags tell you about turn-taking behavior
- The `off_topic` and `redundancy` flags tell you about content quality
- The `anchor_content` and `immediate_focus` tell you what the conversation is actually about

## Output: Guidance Object

Produce a JSON with these fields:

### last_action_outcome (REQUIRED — derive from delegate action + match status)

```
IF delegate_spoke = true:
  IF matched_utt_id is non-null:
    - timing_delta = 0 → "matched (exact timing)"
    - timing_delta > 0 → "matched (N turns early)" where N = timing_delta
  ELSE → "no_match"

IF delegate_spoke = false:
  **Nuanced silence reasoning:**
  - IF anchor_is_for_delegate = true (anchor shows delegate's turn in ground truth):
    - IF delegate_covered_anchor_earlier = true → "silence_justified (already covered)"
      (Delegate was silent but this is fine because they covered this content earlier)
    - ELIF floor_violation = true AND floor_violation_kind = "omission" → "missed_opportunity"
      (Delegate should have spoken but didn't, and never covered this content)
    - ELSE → "appropriate_silence"
  - ELSE (anchor is for someone else):
    - IF floor_violation = false → "appropriate_silence"
    - IF floor_violation = true → check floor_violation_kind
```

### contribution_pressure (based on coverage status)

- coverage_status = "no_targets" → "neutral"
- coverage_status = "low" or "none" AND is_in_cooldown = false → "dial-up" (opportunities remain)
- coverage_status = "high" OR is_in_cooldown = true → "dial-down" (contributed enough)
- else → "neutral"

### post_contribution_cooldown

- Pass-through of is_in_cooldown value (true/false)

### contribution_balance

- contribution_level = "none" or "light" → "under"
- contribution_level = "active" → "over"
- else → "balanced"

### error_correction (null if no issues detected)

Synthesize specific feedback when errors occurred. Reference the actual context:
- If off_topic: "Prior contribution about [what delegate said] diverged from [immediate_focus]. Re-align with current topic."
- If redundancy: "Prior contribution repeated [covered content]. Add novel perspective or wait for new opportunity."
- If floor_violation (intrusion): "Spoke while floor was directed to [other participant]. Respect turn-taking signals."
- If floor_violation (omission): "Floor was open with opportunity, but delegate remained silent. Consider engaging when relevant."

### guidance_summary (ALWAYS populate — one sentence synthesis)

Synthesize the key takeaway. Reference the anchor and delegate action. Examples:
- "Delegate matched the anchor on API design discussion; good timing and relevance."
- "Silence was appropriate as Product Owner and Tech Lead were discussing sprint priorities."
- "Missed opportunity: Scrum Master asked for blockers but delegate stayed silent."
- "Off-topic: Anchor was about budget review but delegate discussed hiring plans."

## Worked Examples

### Example 1: Delegate Spoke and Matched

**Situation:**
- Anchor: Tech Lead said "We should prioritize the authentication module for the Q1 release."
- Delegate utterance: "Agreed, let's focus on authentication first since it blocks other features."
- matched_utt_id: "sprint_47", timing_delta: 0
- floor_violation: false, off_topic: false, category: "Good"

**Analysis:** Delegate spoke, matched the anchor (timing=0), no violations. This is a successful contribution.

**Output:**
```json
{
  "last_action_outcome": "matched (timing=0)",
  "contribution_pressure": "dial-down",
  "post_contribution_cooldown": true,
  "contribution_balance": "balanced",
  "error_correction": null,
  "guidance_summary": "Successful contribution — matched anchor on authentication prioritization."
}
```

### Example 2: Delegate Silent Appropriately

**Situation:**
- Anchor: Marketing Lead said "I think the blue color scheme resonates better with our target demographic."
- Delegate utterance: Silent
- matched_utt_id: null, floor_violation: false
- floor_status: "Dyadic_Exchange (between: Marketing Lead, Creative Director)"

**Analysis:** Delegate was silent because Marketing Lead and Creative Director were in a dyadic exchange. No floor violation. Silence was appropriate.

**Output:**
```json
{
  "last_action_outcome": "appropriate_silence",
  "contribution_pressure": "neutral",
  "post_contribution_cooldown": false,
  "contribution_balance": "under",
  "error_correction": null,
  "guidance_summary": "Appropriate silence — Marketing Lead and Creative Director were discussing color scheme."
}
```

### Example 3: Delegate Spoke but Off-Topic

**Situation:**
- Anchor: Finance Director asked "What's the projected ROI for this campaign?"
- Delegate utterance: "I think we should explore TikTok as an additional channel."
- matched_utt_id: null, off_topic: true, redundancy: false
- immediate_focus: "Campaign ROI projections"

**Analysis:** Delegate spoke about channel expansion when the anchor was clearly about ROI. This is off-topic.

**Output:**
```json
{
  "last_action_outcome": "no_match",
  "contribution_pressure": "dial-up",
  "post_contribution_cooldown": false,
  "contribution_balance": "under",
  "error_correction": "Prior contribution about TikTok channel diverged from ROI projection discussion. Re-align with current topic.",
  "guidance_summary": "Off-topic — anchor was about ROI projections but delegate discussed channel expansion."
}
```

### Example 4: Missed Opportunity

**Situation:**
- Anchor: Scrum Master asked "Does anyone have blockers for this sprint?"
- Delegate utterance: Silent
- floor_violation: true, floor_violation_kind: "omission"
- floor_status: "Open_Question_To_Group"

**Analysis:** Scrum Master explicitly asked for group input, floor was open, but delegate stayed silent. This is a missed opportunity.

**Output:**
```json
{
  "last_action_outcome": "missed_opportunity",
  "contribution_pressure": "dial-up",
  "post_contribution_cooldown": false,
  "contribution_balance": "under",
  "error_correction": "Floor was open with explicit request for blockers. Consider engaging when opportunities arise.",
  "guidance_summary": "Missed opportunity — Scrum Master invited input but delegate remained silent."
}
```

Return ONLY the JSON object.'''

    def _generate_guidance(
        self,
        delegate_error: Optional[DelegateErrorReport],
        delegate_utterance: Optional[str],
        snapshot_analysis: SnapshotAnalysis,
        anchor_turn: Dict,
        context_window: List[Dict],
        matched_unit_count: int,
        total_unit_count: int,
        is_in_cooldown: bool,
        episode_contribution_count: int,
        participant_id: Optional[str] = None,
        delegate_covered_anchor_earlier: bool = False,  # If delegate already covered anchor's content
    ) -> Optional[ControllerGuidance]:
        """
        Dedicated LLM call to produce guidance signals.
        Called AFTER main recalibration completes.
        Receives the finalized snapshot_analysis (updated post-anchor).
        """
        delegate_spoke = bool(delegate_utterance)
        
        # Extract fields from delegate_error, with safe defaults
        if delegate_error:
            matched_utt_id = delegate_error.matched_utt_id
            timing_delta = delegate_error.timing_delta
            floor_violation = delegate_error.floor_violation
            floor_violation_kind = delegate_error.floor_violation_kind
            off_topic = delegate_error.off_topic
            redundancy = delegate_error.redundancy
            category = delegate_error.category
            gt_is_banal = delegate_error.gt_is_banal
        else:
            matched_utt_id = None
            timing_delta = None
            floor_violation = False
            floor_violation_kind = None
            off_topic = False
            redundancy = False
            category = "Good"
            gt_is_banal = False
        
        # Extract from snapshot_analysis
        floor_status = getattr(snapshot_analysis, 'conversational_floor_status', 'Open_Floor')
        micro_topic = getattr(snapshot_analysis, 'micro_topic_analysis', None)
        immediate_focus = getattr(micro_topic, 'immediate_focus', None) if micro_topic else None
        conversational_state = getattr(snapshot_analysis, 'conversational_state', None)
        
        # Build flat context for coach
        flat_context = {
            "delegate_spoke": delegate_spoke,
            "delegate_utterance": delegate_utterance or "Silent",
            "matched_utt_id": matched_utt_id,
            "timing_delta": timing_delta,
            "floor_violation": floor_violation,
            "floor_violation_kind": floor_violation_kind,
            "off_topic": off_topic,
            "redundancy": redundancy,
            "category": category,
            "gt_is_banal": gt_is_banal,
            "floor_status": floor_status,
            "immediate_focus": immediate_focus,
            "conversational_state": conversational_state,
            "anchor_speaker": anchor_turn.get("speaker_id"),
            "anchor_content": anchor_turn.get("utterance_content"),
            # Identity context
            "represented_participant": participant_id,
            "anchor_is_for_delegate": (anchor_turn.get("speaker_id") == participant_id) if participant_id else None,
            # Qualitative coverage signal (avoid exact numbers to prevent target chasing)
            "coverage_status": self._coverage_status(matched_unit_count, total_unit_count),
            "is_in_cooldown": is_in_cooldown,
            "contribution_level": self._contribution_level(episode_contribution_count),
            # Prior coverage check for nuanced silence reasoning
            "delegate_covered_anchor_earlier": delegate_covered_anchor_earlier,
        }
        
        # Format recent context (last 3 utterances for brevity)
        ctx_summary = self._format_utterances(context_window[-3:]) if context_window else "No prior context."
        
        system_prompt = self._feedback_coach_prompt()
        user_prompt = (
            f"Recent conversation context:\n{ctx_summary}\n\n"
            f"Current situation:\n{json.dumps(flat_context, indent=2)}\n\n"
            "Produce the guidance JSON:"
        )
        
        response = self.client.call_with_retry(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=2000,  # Increased for GPT-5-mini reasoning overhead (was 500)
            temperature=0.1,
            json_required=True,
            telemetry_label="recalibrator.feedback_coach",
            telemetry_phase="post_turn",
        )
        
        if not response:
            return None
        
        try:
            obj = json.loads(response) if isinstance(response, str) else response
            return ControllerGuidance.model_validate(obj)
        except Exception:
            return None

    def _coverage_status(self, matched: int, total: int) -> str:
        """Convert numeric coverage to qualitative signal."""
        if total == 0:
            return "no_targets"  # No idea units to match
        ratio = matched / total
        if ratio >= 0.8:
            return "high"  # Covered most targets
        elif ratio >= 0.5:
            return "moderate"  # Covered about half
        elif ratio > 0:
            return "low"  # Some coverage but gaps remain
        else:
            return "none"  # No coverage yet

    def _contribution_level(self, count: int) -> str:
        """Convert numeric contribution count to qualitative signal."""
        if count == 0:
            return "none"
        elif count == 1:
            return "light"
        elif count == 2:
            return "moderate"
        else:
            return "active"  # 3+ contributions
