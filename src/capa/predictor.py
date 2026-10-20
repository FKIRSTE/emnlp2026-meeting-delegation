"""
CAPA Predictor: produces a one-turn PredictionResponse from compact context.
"""

from typing import List, Dict, Optional
from pydantic import ValidationError
import json

from .types import PredictionResponse
try:
    # When imported from tests or other contexts (relative import)
    from ..llm_client import LLMClient
    from ..schemas import EIMMeetingStateVector
except ImportError:
    # When imported from two_tier_agent (direct src context)
    from llm_client import LLMClient
    from schemas import EIMMeetingStateVector


class CapaPredictor:
    def __init__(self, client: LLMClient, last_n: int = 12):
        self.client = client
        self.model_name = client.model
        self.last_n = last_n

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
        # Topics with tails for signal
        topics = {}
        try:
            for lbl, entry in (eim_state.topic_summary_log or {}).items():
                if getattr(entry, 'status', '').lower().startswith('ongoing'):
                    kp_tail = []
                    for kp in (entry.key_points or [])[-6:]:
                        kp_tail.append({
                            "point": kp.point,
                            "speaker": kp.speaker,
                            "utterance_id": kp.utterance_id,
                            "kind": getattr(kp, 'kind', None)
                        })
                    topics[lbl] = {
                        "label": entry.label,
                        "last_updated_at_utterance_id": entry.last_updated_at_utterance_id,
                        "status": entry.status,
                        "key_points_tail": kp_tail,
                        "open_questions_tail": [
                            {"question_summary": oq.question_summary, "raised_by": oq.raised_by, "source_utterance_id": oq.source_utterance_id}
                            for oq in (getattr(entry, 'open_questions', []) or [])[-2:]
                        ],
                        "decisions_tail": [
                            {"decision_summary": de.decision_summary, "decided_by": de.decided_by, "source_utterance_id": de.source_utterance_id, "status": de.status}
                            for de in (getattr(entry, 'decisions', []) or [])[-2:]
                        ]
                    }
        except Exception:
            topics = {}
        # Participants (richer ToM for speaker prediction)
        participants = {}
        try:
            for pid, pm in (eim_state.participant_models or {}).items():
                # Extract stances (most recent per topic)
                stances_summary = {}
                try:
                    for tlabel, st_list in (pm.expressed_stance_on_topics or {}).items():
                        if st_list:
                            st = st_list[-1]  # Most recent stance
                            stances_summary[tlabel] = {
                                "stance_summary": getattr(st, 'stance_summary', None),
                                "source_utterance_id": getattr(st, 'source_utterance_id', None),
                            }
                except Exception:
                    stances_summary = {}
                
                participants[pid] = {
                    "recent_dialogue_act": getattr(pm, 'recent_dialogue_act', None),
                    "engagement_score": getattr(pm, 'engagement_score', None),
                    "expertise_keywords": getattr(pm, 'inferred_expertise_keywords', []) or [],
                    "stances": stances_summary,
                }
        except Exception:
            participants = {}
        try:
            aliases = dict(getattr(eim_state, 'identity_aliases', {}) or {})
        except Exception:
            aliases = {}
        return {
            "snapshot_analysis": sa,
            "ongoing_topics": topics,
            "participants": participants,
            "identity_aliases": aliases,
        }

    def predict(
        self,
        eim_state: EIMMeetingStateVector,
        last_utterances: List[Dict],
    ) -> Optional[PredictionResponse]:
        """
        Produce a one-turn, intent-level forecast. Intent is coarse (Agree/Disagree/Question/Propose/Inform/Meta/Acknowledge),
        with a short gist and micro-topic. The output is not verbatim text.
        """
        try:
            compact_state = self._compact_snapshot(eim_state)
            last_lines = self._format_utterances(last_utterances[-self.last_n:])

            system_prompt = (
                "You are a meeting dynamics forecaster. Predict the next conversational turn at an intent level.\n\n"
                "## Input Context (use for forecasting):\n"
                "You receive:\n"
                "1. **Compact EIM Snapshot** (JSON):\n"
                "   - `snapshot_analysis`: Current meeting state (major_topic, micro_topic_analysis, conversational_state)\n"
                "   - `ongoing_topics`: Active discussion threads with key_points, open_questions, decisions\n"
                "   - `participants`: Per-participant data:\n"
                "     - `recent_dialogue_act`: Their last action (Questioning, Proposing, Agreeing, etc.)\n"
                "     - `engagement_score`: How active they are (0-1)\n"
                "     - `expertise_keywords`: Topics they're expert in \n"
                "     - `stances`: Their positions on topics \n"
                "   - `identity_aliases`: Name mappings for speaker resolution\n"
                "2. **Last N Utterances**: Recent conversation flow (who spoke, what was said)\n\n"
                "**Note**: Some fields may be empty or sparse occasionally (e.g., `decisions_tail`, `open_questions_tail`, `expertise_keywords`, `stances`). Use what is available so that you can still make a rough prediction.\n\n"
                "## Forecasting Strategy:\n"
                "- **Speaker prediction**: Check who was just asked (DIRECTED_TO), who has expertise on active topic, who has been quiet but engaged\n"
                "- **Intent prediction**: Infer from prior stances (will they agree/disagree?), open questions (will they answer?), their dialogue pattern, ...\n"
                "- **Topic prediction**: Use hierarchical topic levels from snapshot_analysis:\n"
                "  * Prefer `immediate_focus` (MICRO level) for next-turn topic prediction - it captures the most recent conversational direction\n"
                "  * Use `broader_context` (MESO level) as fallback if immediate_focus appears banal\n"
                "  * `snapshot_major_topic` (MACRO) is too broad for precise topic prediction\n"
                "- **MICRO-TOPIC VERIFICATION NOTE**: The EIM's `snapshot_analysis.micro_topic_analysis.immediate_focus` may occasionally be stale (not reflecting the most recent utterance's focus). Before relying on it:\n"
                "  - Cross-check against BOTH: (a) the `snapshot_analysis.micro_topic_analysis` object AND (b) the raw Last 1-2 Utterances' content (if these are banal/phatic, look further back to the last substantive utterance)\n"
                "  - If the last utterance introduces NEW content not captured in `immediate_focus`, derive your topic prediction from the raw utterance content instead\n"
                "  - ROBUSTNESS: If the last 1-2 utterances are banal/phatic (e.g., 'okay', 'mm-hmm', disfluencies), look further back to the last substantive utterance for topic context\n\n"
                "- **Floor prediction**: Check if last speaker yielded, if someone was asked directly, if floor is open\n\n"
                "## Output Schema (JSON):\n"
                "{\n"
                "  \"speaker_id\": string (MUST be from EIM participant list),\n"
                "  \"intent_summary\": string (concise 5-10 word gist of intended content),\n"
                "  \"micro_topic\": string (short topic phrase based on EIM snapshot or subtopics—maybe snapshot_analysis and the last few utterances could be helpful here),\n"
                "  \"floor_effect\": string (TAKES_FLOOR | KEEPS_FLOOR | YIELDS_FLOOR | DIRECTED_TO:speaker_id),\n"
                "  \"confidence\": float (0.0-1.0)\n"
                "}\n\n"
                "## Intent Types (implicit in intent_summary):\n"
                "- Question: Seeking information (\"Ask about timeline\", \"Question migration approach\")\n"
                "- Propose: Suggesting action (\"Propose caching solution\", \"Suggest microservices\")\n"
                "- Agree: Expressing support (\"Support budget proposal\", \"Agree with approach\")\n"
                "- Disagree: Expressing concern (\"Challenge timeline\", \"Raise concerns about cost\")\n"
                "- Inform: Sharing facts (\"Report Q1 results\", \"Share performance data\")\n"
                "- Meta: Meeting management (\"Suggest table discussion\", \"Propose move to next topic\")\n"
                "- Acknowledge: Simple response (\"Acknowledge point\", \"Thank presenter\")\n\n"
                "## Granularity Guidelines:\n"
                "- intent_summary: Brief gist (5-10 words), NOT implementation details or specific numbers\n"
                "- micro_topic: Infer where conversation is headed based on snapshot_analysis + recent utterances (may refine/shift from major_topic)\n"
                "- speaker_id: ONLY from EIM participants/identity_aliases - NEVER invent new speakers\n"
                "- Output is intent-level forecast, NOT verbatim text\n\n"
                "## Floor Effect Guidelines (objective meeting dynamics):\n"
                "- TAKES_FLOOR: Participant initiates new speaking turn in open discussion\n"
                "- KEEPS_FLOOR: Participant continues from their previous utterance without yielding\n"
                "- DIRECTED_TO:X: Response explicitly expected from participant X (question/request)\n"
                "- YIELDS_FLOOR: Participant wraps up and opens floor to others\n\n"
                "## Avoid Hallucination (critical):\n"
                "- Use ONLY speakers from EIM participants list\n"
                "- Use ONLY topics/subtopics from EIM snapshot or ongoing_topics\n"
                "- DO NOT invent specific numbers, dates, or names not in EIM context\n"
                "- If context is unclear/insufficient, use confidence < 0.5\n\n"
                "## Examples:\n\n"
                "Ex1 (Directed Response): {\"speaker_id\": \"QA\", \"intent_summary\": \"Answer diagnostics question\", \"micro_topic\": \"API Latency\", \"floor_effect\": \"DIRECTED_TO:QA\", \"confidence\": 0.85}\n\n"
                "Ex2 (Topic Evolution): major_topic='Performance', recent talk about Redis → {\"speaker_id\": \"Backend\", \"intent_summary\": \"Propose Redis caching\", \"micro_topic\": \"Caching Strategy\", \"floor_effect\": \"TAKES_FLOOR\", \"confidence\": 0.75}\n\n"
                "Ex3 (Speaker Continues): LeadDev just proposed microservices, still has floor → {\"speaker_id\": \"LeadDev\", \"intent_summary\": \"Explain scalability benefits\", \"micro_topic\": \"Microservices\", \"floor_effect\": \"KEEPS_FLOOR\", \"confidence\": 0.80}\n\n"
                "Ex4 (Low Confidence): Sparse EIM, unclear next speaker → {\"speaker_id\": \"PM\", \"intent_summary\": \"Continue discussion\", \"micro_topic\": \"Project Status\", \"floor_effect\": \"TAKES_FLOOR\", \"confidence\": 0.35}\n"
            )

            compact_state_json = json.dumps(compact_state, ensure_ascii=False, indent=2)
            user_prompt = (
                f"Compact EIM Snapshot (JSON):\n{compact_state_json}\n\n"
                f"Last {min(self.last_n, len(last_utterances))} Utterances (lines):\n{last_lines}\n\n"
                "Output JSON schema: {\n"
                "  \"speaker_id\": string|null,\n"
                "  \"intent_summary\": string,\n"
                "  \"micro_topic\": string|null,\n"
                "  \"floor_effect\": string|null,\n"
                "  \"confidence\": float (0..1)\n"
                "}"
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            json_response = self.client.call_with_retry(
                model=self.model_name,
                messages=messages,
                max_tokens=2000,  # Increased for GPT-5-mini reasoning overhead (was 500)
                temperature=0.2,
                json_required=True,
                response_model=PredictionResponse,  # Structured output for GPT-5-mini/Gemini
                telemetry_label="predictor",
                telemetry_phase="post_turn",
            )
            if not json_response:
                return None
            # Handle both Pydantic model (structured output) and raw JSON (fallback)
            if hasattr(json_response, 'model_dump'):
                return json_response  # Already a PredictionResponse
            pred = self.client.validate_json_response(json_response, PredictionResponse)
            return pred
        except (ValidationError, Exception):
            return None
