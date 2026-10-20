"""
CAPA Env Judge implementation
"""
from typing import List, Dict, Optional, Tuple
from pydantic import ValidationError
import json
import re
try:
    from ..schemas import EIMMeetingStateVector
    from ..llm_client import LLMClient
except ImportError:
    from schemas import EIMMeetingStateVector
    from llm_client import LLMClient
from .types import PredictionResponse, ErrorReport


class CapaJudge:
    def __init__(self, client: LLMClient, late_window_k: int = 5):
        self.client = client
        self.model_name = client.model
        self.k = late_window_k
        # Thresholds (aligned with rubric)
        self.tau_intent = 0.65
        self.tau_topic = 0.65
        self.tau_floor = 0.60
        self.tau_align = 0.62

    def _format_utterances(self, utterances: List[Dict]) -> str:
        lines = []
        for u in utterances or []:
            sp = u.get("speaker_id", "Unknown")
            # Fallback: some parts of codebase use 'content' instead of 'utterance_content'
            tx = u.get("utterance_content") or u.get("content", "")
            uid = u.get("utterance_id", "Unknown")
            if not tx.strip():
                print(f"[WARN] GT utterance {uid} has empty content - judge may produce unreliable scores")
            lines.append(f"({sp}): {tx} ({uid})")
        return "\n".join(lines)

    def _compact_snapshot(self, eim_state: EIMMeetingStateVector) -> Dict:
        """Keep snapshot_analysis, topic summary with key_points tail, participants, and aliases."""
        # Build trimmed topic_summary_log with key_points tail for topic grounding
        topics = {}
        try:
            for lbl, entry in (eim_state.topic_summary_log or {}).items():
                kp_tail = []
                for kp in (entry.key_points or [])[-5:]:  # Last 5 key points
                    kp_tail.append({
                        "point": kp.point,
                        "speaker": kp.speaker,
                    })
                topics[lbl] = {
                    "status": getattr(entry, 'status', None),
                    "key_points_tail": kp_tail,
                }
        except Exception:
            topics = {}
        try:
            sa = eim_state.snapshot_analysis.model_dump()
        except Exception:
            sa = {}
        # Add participant IDs for hallucination detection (STEP 2 in prompt)
        try:
            participant_ids = list((eim_state.participant_models or {}).keys())
        except Exception:
            participant_ids = []
        # Add identity aliases for speaker resolution
        try:
            aliases = dict(getattr(eim_state, 'identity_aliases', {}) or {})
        except Exception:
            aliases = {}
        return {
            "snapshot_analysis": sa,
            "topic_summary": topics,  # Richer than just labels
            "participant_ids": participant_ids,
            "identity_aliases": aliases,
        }

    def _enforce_k_window(self, gt_window: List[Dict]) -> List[Dict]:
        """
        CRITICAL: Enforce k-window limit to prevent future context leakage.
        Only return the first k utterances (t+1 through t+k).
        """
        if not gt_window:
            return []
        return gt_window[:self.k]

    def _enforce_mapping(self, er: ErrorReport, prediction: PredictionResponse = None, gt_window: List[Dict] = None) -> ErrorReport:
        """
        Deterministic mapping safety net; 
        Priority order:
        1. Keep LLM-assigned special categories (Contradiction, Banal, Hallucination)
        2. Check speaker matching for Aligned categories
        3. Apply threshold-based rules for alignment categories
        4. Fallback to Missed-Cue if no clear match
        """
        # Keep LLM-only categories when assigned
        keep = {"Contradiction", "Banal", "Hallucination"}
        if er.category in keep:
            return er

        # Banal enforcement: if anchor is banal AND no substantive match found, force Banal
        gt_is_banal = getattr(er, 'gt_is_banal', None)
        if gt_is_banal is True and er.matched_utt_id is None:
            er.category = "Banal"
            return er

        intent = er.intent_score or 0.0
        topic = er.topic_score or 0.0
        floor = er.floor_score or 0.0
        # CRITICAL: null timing_delta means no match; don't default to 0 (immediate)
        td = er.timing_delta  # Keep as None if null
        novelty = getattr(er, 'novelty', None)

        # Check speaker matching for Aligned-* categories
        # CRITICAL: If matched_utt_id is null, no match exists -> can't be Aligned-*
        speaker_matches = False  # Default: False (require explicit match)
        if prediction and gt_window and er.matched_utt_id:
            predicted_speaker = prediction.speaker_id
            # Find the matched GT utterance
            matched_gt = None
            for utt in gt_window:
                if utt.get("utterance_id") == er.matched_utt_id:
                    matched_gt = utt
                    break
            if matched_gt:
                gt_speaker = matched_gt.get("speaker_id", "")
                speaker_matches = (predicted_speaker == gt_speaker)

        # Aligned-Immediate: all thresholds met at t+1 AND speaker matches AND td is valid
        if intent >= self.tau_intent and topic >= self.tau_topic and floor >= self.tau_floor:
            # CRITICAL: td must not be None (requires matched_utt_id to exist)
            if td is not None and td == 0:
                if speaker_matches:
                    er.category = "Aligned-Immediate"
                else:
                    er.category = "Topic-Only"
                return er
            # Aligned-Late: all thresholds met at 1 <= td <= k-1 AND speaker matches
            if td is not None and 1 <= td < self.k:
                if speaker_matches:
                    er.category = "Aligned-Late"
                else:
                    er.category = "Topic-Only"
                return er

        # Topic-Only: topic aligns but intent is moderate (0.30-0.64) or floor fails
        if topic >= self.tau_topic:
            if 0.30 <= intent < self.tau_intent:
                er.category = "Topic-Only"
                return er
            # Also Topic-Only if intent/topic pass but floor fails
            if intent >= self.tau_intent and floor < self.tau_floor:
                er.category = "Topic-Only"
                return er

        # Missed-Cue: poor alignment or beyond window
        if intent < 0.30 and topic < 0.50:
            if novelty is True:
                er.category = "Novel"
            else:
                er.category = "Missed-Cue"
            return er

        # Simple fallback: if no clear match, use Missed-Cue or Novel
        # (Removed composite alignment fallback which violated Topic-Only semantics)
        if novelty is True:
            er.category = "Novel"
        else:
            er.category = "Missed-Cue"

        return er

    def _build_system_prompt(self) -> str:

        return f"""You are an impartial evaluator assessing the quality of meeting turn predictions.

**Task**: Compare a forecasted next turn with observed ground-truth (GT) turns in a {self.k}-utterance window (t+1 through t+{self.k}), then return a structured JSON assessment.

---

## Evaluation Process:

1. **Find Best Content Match**: Identify the GT utterance with highest semantic overlap (intent + topic)
   - Prioritize substantive contributions over acknowledgments
   - Match to the utterance that best captures the predicted idea, even if speaker differs
   - Set matched_utt_id = that utterance, or null if no viable match exists

2. **Score Three Dimensions** (SCORING REFERENCE matters):
   
   **SCORING REFERENCE**:
   - If matched_utt_id EXISTS: score intent_score and topic_score against THE MATCHED UTTERANCE
   - If matched_utt_id is NULL (no viable match): score against THE ANCHOR (t+1)
   - floor_score ALWAYS reflects anchor (t+1) floor state
   
   **Dimensions**:
   - **intent_score**: How well does predicted intent match REFERENCE GT content?
   - **topic_score**: How closely does predicted micro_topic align with REFERENCE GT micro_topic?
   - **floor_score**: How well does predicted floor behavior match anchor (t+1) turn-taking?
   - NOTE: Speaker identity affects category assignment, NOT individual dimension scores

3. **Assign Category**: Use the decision flow (below) considering scores AND speaker match

---

## Scoring Dimensions (each scored 0.0-1.0):

1. **intent_score** - Semantic alignment of predicted intent with GT content
   
   **CRITICAL**: Consider both CONTENT similarity AND SPEECH ACT type (PROPOSE, QUESTION, INFORM, CONCERN, etc.).
   
   **Scoring Guide**:
   - **0.85-1.0**: Same speech act + same topic (e.g., both propose Redis caching)
   - **0.65-0.84**: Same speech act + topic, minor content differences
   - **0.35-0.64**: Same topic, DIFFERENT speech act (proposal vs question) or moderate alignment
   - **0.10-0.34**: Opposing stance (support vs concern) or weakly related
   - **0.0-0.09**: Completely unrelated
   
   Examples:
   - "Propose Redis caching" vs "Ask about current caching" -> 0.4-0.5 (same topic, different act)
   - "Concern about API security" vs "Concern about API security" -> 0.9-1.0 (perfect match)
   - "Support budget increase" vs "Concern about budget increase" -> 0.1-0.2 (opposing stance)
   
   **Note**: Predictor outputs coarse summaries (5-10 words); don't penalize for lack of implementation details.

2. **topic_score** - Alignment of predicted micro_topic with REFERENCE GT micro_topic
   
   Reference = matched_utt_id if exists, else anchor (t+1).
   This measures MICRO-TOPIC proximity (specific focus), not macro-topic overlap.
   
   **MICRO-TOPIC VERIFICATION NOTE**: The EIM snapshot's `snapshot_analysis.micro_topic_analysis.immediate_focus` may occasionally be stale.
   When scoring topic alignment, cross-check `immediate_focus` against BOTH: (a) the EIM snapshot AND (b) the raw Ground Truth Window utterances.
   If `immediate_focus` describes a prior discussion focus while the last utterance introduces something new,
   weight the raw utterance content more heavily in your assessment.
   ROBUSTNESS: If the last 1-2 utterances are banal/phatic (disfluencies, backchannels like 'mm-hmm'), look further back to the last substantive utterance.
   
   **Scoring Guide**:
   - **0.85-1.0**: Exact micro_topic match (same specific focus)
   - **0.65-0.84**: Same micro_topic area, minor phrasing differences
   - **0.40-0.64**: Related domain, DIFFERENT micro_topic focus
   - **0.20-0.39**: Weakly related (same macro topic, different sub-area)
   - **0.0-0.19**: Unrelated micro_topics

3. **floor_score** - Alignment of predicted floor behavior with GT turn-taking dynamics
   
   **Floor Types** (from predictor):
   - **TAKES_FLOOR**: Initiates new speaking turn (new contribution, topic shift, or interruption)
   - **KEEPS_FLOOR**: Continues from previous utterance without yielding (multi-turn elaboration)
   - **YIELDS_FLOOR**: Wraps up and opens floor to others (explicit handoff or natural conclusion)
   - **DIRECTED_TO:X**: Response explicitly expected from participant X (direct question/request)
   
   **Scoring Guide** (focus on turn-taking dynamics, not speaker identity):
   - **1.0**: Exact floor type match (e.g., both TAKES_FLOOR)
   - **0.85-0.95**: Same floor behavior category
     * TAKES_FLOOR ↔ KEEPS_FLOOR (both active contribution)
     * DIRECTED_TO:X ↔ DIRECTED_TO:Y (both directed, different target)
   - **0.6-0.8**: Related floor dynamics
     * TAKES_FLOOR vs YIELDS_FLOOR (active initiation vs handoff)
     * DIRECTED_TO:X vs actual X responds (expected vs realized)
   - **0.3-0.5**: Opposite floor dynamics
     * TAKES_FLOOR predicted but speaker was responding to direct question
     * DIRECTED_TO:X predicted but floor was open (no explicit cue)
   - **0.0-0.2**: Complete mismatch
     * Active contribution predicted but utterance was passive acknowledgment
     * Multi-turn continuation predicted but speaker took floor for first time
   
   **Note**: Evaluate the TURN-TAKING PATTERN, not who took the turn. Speaker identity affects category assignment separately.

### **Additional Fields**:
- **timing_delta**: Position of matched utterance (0 for t+1, 1 for t+2, ..., k-1 for t+k; NULL if no match)
- **matched_utt_id**: ID of the best-matching GT utterance (or NULL if no match)

---

## **Category Assignment Decision Flow**:

**STEP 1 (gt_is_banal)**: Is the **ANCHOR GT turn (t+1)** phatic/banal/backchannel (ok/okay/yeah/yup/mm-hmm/uh-huh/thanks/right/{{vocalsound}}/{{disfmarker}}, or <=5-token acknowledgment) with no substantive content?
- ALWAYS set `gt_is_banal` to true/false based on t+1 content ONLY.
- This does NOT automatically mean category=Banal (see STEP 1b and category definition below).
- Purpose: Downstream recalibrator uses this to gate factual updates.

**STEP 1b (matching eligibility)**: When finding best content match in the k-window:
- If prediction is **substantive** (not just acknowledgment/thanks), SKIP banal GT utterances as match candidates.
- Match to the first substantive GT that aligns with predicted content.
- If no substantive GT matches, set `matched_utt_id=null` and `timing_delta=null`.

**STEP 2**: Check for structural hallucination (speaker existence)
- Does predicted speaker exist in EIM participant_ids OR identity_aliases?
- (If predicted speaker is an alias key in identity_aliases, treat as valid - resolve to canonical participant)
- If speaker is NOT in participant_ids AND NOT an alias key -> **Hallucination** (structural)

**STEP 3**: Check speaker match **(MANDATORY FOR ALIGNED CATEGORIES)**
- Compare predicted speaker_id with matched GT utterance's speaker_id
- **If speakers MATCH** + all scores meet thresholds -> Eligible for **Aligned-Immediate** / **Aligned-Late**
- **If speakers DON'T MATCH** -> Maximum category is **Topic-Only** (even with perfect scores)

**STEP 4**: Apply scoring thresholds
- **High alignment** (intent>=0.65, topic>=0.65, floor>=0.60, speaker matches):
  * timing_delta = 0 -> **Aligned-Immediate**
  * 1 <= timing_delta <= k-1 -> **Aligned-Late**
- **Topic match only** (topic>=0.60, but intent<0.65 OR floor<0.60 OR speaker mismatch) -> **Topic-Only**
- **Same topic, opposing stance** (topic>=0.60, intent<0.30) -> **Contradiction**
- **No match / Complete miss** (matched_utt_id=null OR intent<0.30, topic<0.30) -> **Missed-Cue** or **Novel**
  * Apply the "Was it already in play?" inferability test (see definitions below)

**SCORING vs CATEGORY DISTINCTION** (CRITICAL):
- **topic_score** = semantic proximity (can be HIGH even for Novel if topic is related)
- **category** = predictability (Novel = unpredictable; Missed-Cue = predictable but missed)

---

**Category Definitions**:

1. **Aligned-Immediate**
   - Perfect match at (t+1): intent>=0.65, topic>=0.65, floor>=0.60, timing_delta=0, speaker matches
   - All dimensions meet thresholds, immediate timing

2. **Aligned-Late**
   - Strong match at (t+2) or later: intent>=0.65, topic>=0.65, floor>=0.60, 1<=timing_delta<=k-1, speaker matches
   - All dimensions meet thresholds, delayed timing

3. **Topic-Only**
   - Topic matches but other criteria fail: topic>=0.60, AND (intent<0.65 OR floor<0.60 OR speaker mismatch)
   - Common causes: Different speech act, wrong speaker, low floor alignment

4. **Contradiction**
   - Same topic, opposing stance: topic>=0.60, intent<0.30
   - Example: Predict "support budget increase" -> GT "concerned about budget increase"

5. **Hallucination**
   - **Structural hallucination ONLY**: Predicted speaker does NOT exist in EIM participant_ids AND is NOT an alias
   - This is a fabrication of a non-existent meeting participant
   - Note: Topic-related issues (ungrounded topics) should be scored low on topic_score, not labeled Hallucination

**ANCHOR-BASED CATEGORIES (Missed-Cue, Novel, Banal)**:
These categories are determined by the **ANCHOR (t+1)** characteristics, NOT by whether prediction matched somewhere in the window.
- Scores may be computed against matched_utt_id (if exists), but CATEGORY assignment reflects anchor behavior.

6. **Missed-Cue** (novelty=false)
   - Prediction missed GT that was **inferable** from available context
   - **"Was it already in play?" test**: Answer YES -> Missed-Cue
   - Inferable when anchor:
     * Answers an open question from last turns or topic_summary's open_questions
     * Continues/elaborates on ongoing topic or key_point thread
     * Responds to direct floor handoff or explicit request
   - Example: open_question="What's timeline?", GT="We can ship in 2 weeks" -> Missed-Cue (answers in-play question)

7. **Novel** (novelty=true)
   - GT introduces **exogenous** information NOT inferable from predictor's context
   - Can be ON-TOPIC (high topic_score) but still Novel if content wasn't predictable
   - **"Was it already in play?" test**: Answer NO -> Novel
   - Novel when anchor introduces:
     * External decision: "Leadership decided...", "Legal just told us..."
     * New incident: "We got an alert...", "Production is down"
     * New data from outside: "Just got user testing results", "Latest metrics show..."
     * Surprise constraint: "Budget was cut by 20%"
   - Do NOT penalize as missed predictable cue; treat as unpredictable arrival

8. **Banal**
   - Assign when ALL of these are true:
     1. `gt_is_banal=true` (anchor t+1 is phatic/acknowledgment)
     2. Prediction was **substantive** (expected real contribution, not just ack/thanks)
     3. No substantive GT in window matched the prediction (`matched_utt_id=null` or matched GT is also banal)
   - This means: predictor expected substance, but anchor was phatic and nothing in window delivered the predicted content.
   - Note: If t+1 is banal but t+2..t+k contains the predicted substantive idea, match to that and assign Aligned-Late/Topic-Only (not Banal).

---

**Examples** (curated for critical edge cases):

**Example 1 - Topic-Only (Speech Act Mismatch)**: Same topic, different speech act type
```
Prediction: {{ "speaker_id": "Backend", "intent_summary": "Propose Redis caching implementation", "micro_topic": "Caching Strategy", "floor_effect": "TAKES_FLOOR" }}
GT t+1: "(Backend): Has anyone looked into what caching strategies we're currently using?"
Scores: intent=0.45 (PROPOSE ≠ QUESTION, same topic but different speech act), topic=0.85, floor=0.85, timing_delta=0
Category: Topic-Only
Reason: High topic/floor, same speaker, BUT intent_score is moderate because speech acts differ (proposal vs question). This prevents Aligned-Immediate.
```

**Example 2 - Topic-Only (CRITICAL: Speaker Mismatch Rule)**: Perfect scores but wrong speaker
```
Prediction: {{ "speaker_id": "Security", "intent_summary": "Raise API authentication concerns", "micro_topic": "API Security", "floor_effect": "TAKES_FLOOR" }}
GT t+1: "(Backend): I'm concerned about authentication issues on our API endpoints."
Scores: intent=0.90, topic=0.88, floor=0.85, timing_delta=0
Category: Topic-Only
Reason: High scores on ALL dimensions, BUT speaker mismatch (Security ≠ Backend) prevents Aligned-Immediate. Speaker match is MANDATORY for Aligned-* categories.
```

**Example 3 - Contradiction**: Same topic, opposing stance
```
Prediction: {{ "speaker_id": "Eve", "intent_summary": "Support marketing budget increase", "micro_topic": "Budget", "floor_effect": "TAKES_FLOOR" }}
GT t+1: "(Eve): I'm concerned about the marketing budget increase. We should focus on ROI first."
Scores: intent=0.10 (opposing), topic=0.85, floor=0.90, timing_delta=0
Category: Contradiction
Reason: Topic matches but intent expresses opposite stance (support vs concern).
```

**Example 4 - Hallucination**: Non-existent speaker
```
Prediction: {{ "speaker_id": "Marketing_Director", "intent_summary": "Report Q1 campaign results", "micro_topic": "Q1 Marketing", "floor_effect": "TAKES_FLOOR" }}
EIM Participants: [CEO, Finance, Data_Analyst]  // NO Marketing_Director!
GT t+1: "(Data_Analyst): Product analytics show good user engagement growth."
Category: Hallucination
Reason: Predicted speaker doesn't exist in meeting. Structural hallucination regardless of topic overlap.
```

**Example 5a - Novel (exogenous event, off-topic)**: GT introduces genuinely new event not in EIM
```
Prediction: {{ "speaker_id": "Product", "intent_summary": "Propose email campaign features", "micro_topic": "Q2 Features", "floor_effect": "TAKES_FLOOR" }}
EIM snapshot_analysis: "Q2 product roadmap features discussion", topics include "marketing automation", "email campaigns"
GT t+1: "(Engineering): Hold on - I just got an alert that production is down. API gateway returning 503 errors."
Scores: intent=0.0, topic=0.0, floor=0.1, timing_delta=null, novelty=true
Category: Novel
"Was it in play?" test: NO - production outage is exogenous event, not inferable from roadmap discussion.
```

**Example 5b - Novel (exogenous decision, ON-TOPIC)**: High topic_score but still Novel
```
Prediction: {{ "speaker_id": "PM", "intent_summary": "Discuss Q2 roadmap priorities", "micro_topic": "Roadmap", "floor_effect": "TAKES_FLOOR" }}
EIM snapshot_analysis: "Q2 roadmap prioritization", no budget constraints mentioned
GT t+1: "(PM): CFO just informed us - budget is cut by 20% effective immediately. We need to reprioritize."
Scores: intent=0.30, topic=0.65 (still roadmap-related!), floor=0.80, timing_delta=0, novelty=true
Category: Novel (NOT Missed-Cue despite high topic_score)
"Was it in play?" test: NO - budget cut is exogenous decision from leadership, not inferable from EIM context.
```

**Example 5c - Missed-Cue (inferable from open question)**: GT was predictable but predictor missed
```
Prediction: {{ "speaker_id": "Backend", "intent_summary": "Propose microservices architecture", "micro_topic": "Architecture", "floor_effect": "TAKES_FLOOR" }}
EIM topic_summary: open_questions include "What's the expected timeline for phase 1?"
GT t+1: "(PM): We can ship phase 1 in 2 weeks if we stick to the current scope."
matched_utt_id: null (prediction about Architecture doesn't match Timeline answer)
Scores (vs anchor): intent=0.15, topic=0.35 (Architecture vs Timeline = different micro_topics), floor=0.60, timing_delta=null, novelty=false
Category: Missed-Cue (NOT Novel - answer was in play)
"Was it in play?" test: YES - GT answers an explicit open_question in topic_summary. Predictor should have anticipated a timeline answer.
```

**Example 6 - Banal**: Substantive prediction, banal anchor, no substantive match in window
```
Prediction: {{ "speaker_id": "UX", "intent_summary": "Present user testing findings", "micro_topic": "Testing Results", "floor_effect": "TAKES_FLOOR" }}
GT t+1: "(PM): Yeah, sounds good." (banal ack - ANCHOR)
GT t+2: "(PM): Let's move on to the next topic." (meta, not testing - NOT a match)
gt_is_banal: true (anchor is phatic)
matched_utt_id: null (no UX, no testing content in window)
Scores: intent=0.10, topic=0.15, floor=0.60
Category: Banal
Reason: Prediction expected UX to present testing findings (substantive), but anchor was a banal ack and no utterance in window delivered the predicted content. Per STEP 1b, we don't force a match to banal/unrelated content.
```

**Example 7 - Aligned-Late despite banal anchor**: Banal t+1, but substantive match at t+2
```
Prediction: {{ "speaker_id": "UX", "intent_summary": "Present user testing findings", "micro_topic": "Testing Results", "floor_effect": "TAKES_FLOOR" }}
GT t+1: "(PM): Yeah, sounds good." (banal ack - ANCHOR)
GT t+2: "(UX): So in our testing, we saw a 20% improvement in task completion."
gt_is_banal: true (anchor is phatic - used by recalibrator)
matched_utt_id: GT t+2 (substantive match found!)
Scores: intent=0.85, topic=0.90, floor=0.85, timing_delta=1
Category: Aligned-Late (NOT Banal - substantive match exists)
Reason: Although anchor (t+1) is banal, t+2 contains the predicted substantive content from UX about testing. Per STEP 1b, we match to substantive GT. Speaker matches, scores pass thresholds, timing_delta=1 -> Aligned-Late.
```

---

## Output Format
Return ONLY the JSON object below (no additional text):

```json
{{
  "intent_score": <float 0.0-1.0>,
  "topic_score": <float 0.0-1.0>,
  "floor_score": <float 0.0-1.0>,
  "matched_utt_id": <string|null>,
  "timing_delta": <int 0 to k-1 | null>,
  "category": <string>,
  "novelty": <boolean>,
  "gt_is_banal": <boolean>,
  "details": <string>
}}
```

## Field Specifications:
- **intent_score**, **topic_score**, **floor_score**: Float 0.0-1.0 (score dimension independently of speaker)
- **matched_utt_id**: Utterance ID of best content match, or null
- **timing_delta**: 0 for t+1, 1 for t+2, ..., k-1 for t+k; null if no match
- **category**: One of: Aligned-Immediate, Aligned-Late, Topic-Only, Contradiction, Hallucination, Missed-Cue, Novel, Banal
- **novelty**: Boolean (true only for Missed-Cue when GT introduces genuinely new information per STEP 4)
- **details**: 1-2 sentence explanation

**No Forced Match**:
- If no GT turn in the current k-window reaches minimal content alignment, set matched_utt_id=null and timing_delta=null. Do not pick a placeholder turn.

**Pre-Output Verification Checklist**:
1. ✓ Scores are dimension-only (floor_score ignores speaker identity)
2. ✓ Aligned-* categories require speaker match (enforced via STEP 3)
3. ✓ Category assignment matches reasoning in details field

Return only the JSON object."""

    def judge_prediction(
        self,
        prediction: PredictionResponse,
        eim_state: EIMMeetingStateVector,
        gt_window: List[Dict],
    ) -> Optional[ErrorReport]:
        """
        Primary entrypoint for single-mode evaluation.
        
        CRITICAL: gt_window will be trimmed to enforce k-window limit (t+1..t+k).
        This method is read-only with respect to working memory and never leaks
        future context into prompts.
        """
        try:
            # ENFORCE K-WINDOW LIMIT
            gt_window = self._enforce_k_window(gt_window)

            # Prepare compact state and inputs
            compact_state = self._compact_snapshot(eim_state)
            prediction_json = json.dumps(prediction.model_dump(), ensure_ascii=False, indent=2)
            compact_state_json = json.dumps(compact_state, ensure_ascii=False, indent=2)
            gt_formatted = self._format_utterances(gt_window)

            system_prompt = self._build_system_prompt()

            user_prompt = (
                f"**K-Window Limit**: {self.k} (evaluate only t+1 through t+{self.k})\n\n"
                f"**Prediction (JSON)**:\n{prediction_json}\n\n"
                f"**Compact EIM Snapshot (JSON)**:\n{compact_state_json}\n\n"
                f"**Ground Truth Window** (t+1 to t+{self.k}):\n{gt_formatted}\n\n"
                "Return only the JSON ErrorReport as specified above."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            json_response = self.client.call_with_retry(
                model=self.model_name,
                messages=messages,
                max_tokens=3000,  # Increased for GPT-5-mini reasoning overhead (was 1000)
                temperature=0.0,
                json_required=True,
                response_model=ErrorReport,  # Structured output for GPT-5-mini/Gemini
                telemetry_label="environment_judge",
                telemetry_phase="post_turn",
            )

            if not json_response:
                return None

            # Handle both Pydantic model (structured output) and raw JSON (fallback)
            if hasattr(json_response, 'model_dump'):
                report = json_response  # Already an ErrorReport
            else:
                report = self.client.validate_json_response(json_response, ErrorReport)
            if not report:
                return None

            # Enforce deterministic mapping to stabilize categories (with speaker matching)
            report = self._enforce_mapping(report, prediction, gt_window)

            return report

        except (ValidationError, Exception) as e:
            print(f"JUDGE ERROR: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
