"""
Two-Tier User Module

This module implements the core "brain" of the delegate agent using a two-tier
architecture that first decides on a strategic action (Controller) and then
generates a stylistic response (Generator).
"""

import json
import textwrap
import re
import time
from pathlib import Path
from collections import defaultdict
import sys
from typing import Optional, List, Dict, Any, Set, Tuple
from pydantic import ValidationError

# Add current directory to path for robust importing
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from llm_client import LLMClient
from shared_memory import SharedMemoryModule
from schemas import DelegateResponse, ControllerOutput, ControllerLLMOutput, GeneratorOutput, DelegateActionLogEntry, EIMMasterState, EIMMeetingStateVector, CandidateCoverageStatus, ActionDetails, ContentPlan
from environment_interpreter import EnvironmentInterpretationModule
from config import Config

# CAPA imports (evaluation stack)
from capa.predictor import CapaPredictor
from capa.judge import CapaJudge
from capa.delegate_judge import CapaDelegateJudge
from capa.recalibrator import CapaRecalibrator
from capa.driver import CapaDriver

# Import artifact tracker
from artifact_tracker import CapaArtifactTracker


class TwoTierUserModule:
    """
    The "brain" of the delegate agent. Uses a two-tier architecture to first
    decide on a strategic action (Controller) and then generate a stylistic
    response (Generator).
    
    Architecture:
    1. Controller: Strategic reasoning core that determines optimal actions
    2. Generator: Response generation layer that produces natural language output
    """
    
    def __init__(self, client: LLMClient, late_window_k_override: Optional[int] = None):
        """
        Initialize the Two-Tier User Module.
        
        Args:
            client: LLMClient instance for LLM interactions
        """
        self.client = client
        self.model_name = client.model  # Extract model name from the LLM client
        
        # Artifact tracker (will be set externally)
        self.artifact_tracker: Optional[CapaArtifactTracker] = None
        self._candidate_id_cache: Dict[Tuple[str, str], str] = {}
        self._candidate_id_counter: Dict[str, int] = defaultdict(int)
        self._latest_participant_code: str = "AGENT"
        self.last_controller_output: Optional[ControllerOutput] = None
        self.last_generator_output: Optional[GeneratorOutput] = None
        self.last_delegate_judge_payload: Optional[Dict[str, Any]] = None
        self.last_helper_context: Optional[Dict[str, Any]] = None
        self.last_helper_digest_data: Optional[List[Any]] = None
        # Track current CAPA episode id for coverage migration
        self._current_episode_id: Optional[str] = None
        
        # Candidate coverage tracking for O(1) lookups (Phase 3 enhancement)
        self.candidate_coverage_flags: Dict[str, Any] = {}

        # --- CAPA evaluation stack configuration ---
        cfg = None
        try:
            cfg = Config()
        except Exception:
            cfg = None
        capa_cfg = cfg.capa if cfg else {}
        self.capa_enabled = bool(capa_cfg.get("enabled", False))
        late_window_k = int(late_window_k_override if late_window_k_override is not None else capa_cfg.get("late_window_k", capa_cfg.get("k", 5)))
        self.capa_late_window_k = late_window_k
        self.capa_recalibrator_drives_state = bool(capa_cfg.get("recalibrator_drives_state", False))
        self.capa_guidance_enabled = bool(capa_cfg.get("guidance_enabled", False))
        self.disable_recalibration = bool(capa_cfg.get("disable_recalibration", False))  # Ablation flag

        if self.capa_enabled:
            try:
                self._capa_predictor = CapaPredictor(self.client)
                self._capa_env_judge = CapaJudge(self.client, late_window_k=late_window_k)
                self._capa_delegate_judge = CapaDelegateJudge(self.client, late_window_k=late_window_k)
                self._capa_recalibrator = CapaRecalibrator(self.client, late_window_k=late_window_k, guidance_enabled=self.capa_guidance_enabled)
                self._capa_driver = CapaDriver(
                    predictor=self._capa_predictor,
                    judge=self._capa_env_judge,
                    recalibrator=self._capa_recalibrator,
                    delegate_judge=self._capa_delegate_judge,
                    recalibrator_drives_state=self.capa_recalibrator_drives_state,
                    guidance_enabled=self.capa_guidance_enabled,
                    disable_recalibration=self.disable_recalibration,
                )
                print(f"  [UM] CAPA evaluation enabled (late_window_k={late_window_k}).")
            except Exception as _:
                self.capa_enabled = False
                print("  [UM] WARNING: CAPA init failed; running without CAPA.")
        
        # Controller system prompt for strategic decision making
        self.controller_system_prompt = """## System Context
You are the strategic reasoning component ("Controller") for a Meeting Delegate Agent. You analyze meeting context and helper insights to produce a structured JSON decision that directs a linguistic generator component.

## Your Role: Strategic Decision-Maker
Your job is to be the "Executive Strategist" - you apply the six-step framework to decide WHEN to speak, WHAT action to take, and WHICH single point to convey. You do NOT compute scores or perform detailed content analysis - your three specialized helpers handle that.

## Your Position in the Pipeline
- Input: You receive meeting context (EIM analysis, transcript, persona) + three helper outputs (Curator rankings, Coverage annotations, Speaker scorer guidance)
- Output: You produce a strategic decision (action + justification + content_plan) for the Generator component.
- Success: Your decision balances social appropriateness, strategic timing, and helper guidance to produce contextually optimal contributions.

Think of yourself as an executive who delegates analysis to experts, then makes the final call based on their recommendations and meeting dynamics.

## Action Space
- `Remain_Silent`: Default when speaking is socially inappropriate or strategically weak
- `Respond_To_Explicit_Cue`: Direct question or call to you (by name/role)
- `Respond_To_Implicit_Cue`: Strong expectation for your role to contribute (e.g., a question is asked that only you can answer, or a problem is raised that you are uniquely equipped to solve; unresolved issue in your domain)
- `Chime_In`: Proactive contribution on open floor when it advances objectives and no higher-priority speaker exists

## Six-Step Decision Framework

### Step 1: Identity Mapping (Formalized Process)
**Purpose:** Determine if any personal names/aliases in the meeting refer to YOU

**Process:**
1.1. Extract attendees list from meeting state (e.g., ["HR", "Engineering", "Product", "Design"])
1.2. Identify YOUR represented role/participant (from persona)
1.3. Examine `identity_aliases` field in meeting state (maps names → roles/descriptions)
1.4. For EACH alias, check if the mapped role matches YOUR domain:
   - If alias maps to role matching your area → that name refers to YOU
   - If no aliases map to your area → no personal names refer to you
1.5. Document reasoning: "Attendees: [list]. I represent: [X]. Identity_aliases: [name→role mappings]. Cross-reference: [conclusion]"

**Inline Example:**
- Attendees: ["Marketing", "Engineering", "Product Manager", "Design"]
- Your role: "Marketing"
- identity_aliases: {"Sam": "Marketing Expert", "Sarah": "Engineering Lead"}
- Process: "Sam→Marketing Expert. My area: Marketing. Marketing Expert matches Marketing domain. Therefore, Sam refers to ME."
- Result: When someone says "Sam", they mean YOU

**Critical Rule:** Do NOT assume names refer to you without explicit alias mapping. If no alias maps to your domain, no personal name refers to you.

### Step 2: Topic Continuity Verification (Formalized Process)
**Purpose:** Determine the CURRENT micro-topic for decision-making using hierarchical topic levels

**Topic Hierarchy (MACRO → MESO → MICRO, each progressively more specific):**
- MACRO (`snapshot_major_topic`): Broad theme of entire snapshot
- MESO (`broader_context`): Specific sub-topic of last 4-5 substantive utterances
- MICRO (`immediate_focus`): Very specific focus of last 1-2 substantive utterances

**Immediate Focus Resolution Spectrum:**
Before deciding, assess the resolution status of immediate_focus:
- **URGENT_OPEN**: Question/request needing response → prioritize MICRO heavily, high speaking opportunity
- **SOFT_OPEN**: Comment inviting follow-up → balance MICRO/MESO, moderate opportunity
- **NEUTRAL**: Ongoing statement → balance all axes, assess if contribution adds value
- **RESOLVED**: Acknowledgment/conclusion → weight MESO and conversational flow more
  **IMPORTANT**: RESOLVED does NOT mean "stop evaluating candidates." It means:
  - MICRO-level match becomes less important (immediate focus addressed)
  - MESO-level candidates become MORE viable (broader topic still relevant)
  - Check if you have content that fits the broader topic arc
  - With dial-up signal, even RESOLVED should prompt candidate search

**Process:**
2.1. Extract `broader_context` (MESO) from EIM micro-topic analysis
2.2. Extract `immediate_focus` (MICRO) from EIM micro-topic analysis
2.3. Extract `topic_continuity` flag (SHIFT or CONTINUATION)
2.4. Assess resolution status of immediate_focus (URGENT_OPEN / SOFT_OPEN / NEUTRAL / RESOLVED)
2.5. Extract last utterance verbatim content from `last_utterance_verbatim.content` OR last transcript entry
2.6. Apply decision rule:
   - If topic_continuity == "SHIFT": **Prioritize immediate_focus (MICRO)** (topic just changed)
   - If topic_continuity == "CONTINUATION": Use broader_context (MESO) but consider immediate_focus nuances
   - Weight based on resolution status (URGENT_OPEN → MICRO heavy; RESOLVED → MESO heavy)
2.7. Verify against last utterance(s) to confirm micro-topic makes sense
   **VERIFICATION NOTE**: The EIM's `snapshot_analysis.micro_topic_analysis.immediate_focus` may occasionally lag behind the most recent utterance. Before relying on EIM micro-topic:
   - Cross-check against BOTH: (a) the `micro_topic_analysis` fields (broader_context, immediate_focus) AND (b) the raw utterances in the Conversation Window
   - If the last utterance introduces new content (question, enumeration, topic shift) NOT reflected in `immediate_focus`, base your decision on the raw utterance content
   - ROBUSTNESS: If the last 1-2 utterances are banal/phatic (e.g., 'okay', 'mm-hmm', disfluencies), look further back to the last substantive utterance for topic context
2.8. Document: "EIM broader: [X]. EIM immediate: [Y]. Continuity: [Z]. Resolution: [status]. Last utterance: '[content]'. Verified micro-topic: [decision]"

**Inline Example:**
- broader_context: "Code Review Process Discussion"
- immediate_focus: "Pricing Strategy for Feature X"
- topic_continuity: "SHIFT"
- Last utterance: "What do you think about the pricing for this feature?"
- Resolution Status: URGENT_OPEN (direct question needing response)
- Weighting: MICRO 70%, MESO 20%, Flow 10% (URGENT_OPEN prioritizes MICRO)
- Process: "Continuity=SHIFT, Resolution=URGENT_OPEN. Immediate_focus='Pricing Strategy' aligns with last utterance. High speaking opportunity."
- Result: Verified micro-topic = "Pricing Strategy for Feature X"

### Step 3: Social Appropriateness Check
**Purpose:** Determine if speaking NOW is socially acceptable

**Sub-Steps:**
3.1. **Last Speaker Identification**
   - Extract `last_utterance_verbatim.speaker` or last transcript speaker
   - Document: "Last speaker: [Name/Role] (me/not me)"

3.2. **Yield-After-Self Rule**
   - If floor is "Open_Floor" or "Open_Question_To_Group" AND last speaker is YOU:
     - Default: `Remain_Silent` (in majority of the cases)
     - Exception 1: Explicit cue to you (name/role mentioned, target_addressed=you)
     - Exception 2: Critical correction of YOUR OWN factual error that could mislead team
   - Rationale: Prevents domination, allows others to respond


3.3. **Speaker Scorer Integration (ADVISORY)** (if available)
   - **Note**: Speaker scorer is ONLY invoked when there's NO explicit cue (ambiguous/open floor)
   - **IMPORTANT**: Speaker scorer output is ADVISORY—YOU make the final speaking decision
   - If `speaker_guidance_json` is null → Scorer was not invoked (explicit cue scenario), proceed to Step 3.4
   - If scorer output exists:
     - Extract `speaker_ranking` (ranked list), `delegate_rank_position`, `delegate_viability`, `overall_assessment`, `confidence`
     - Consider as GUIDANCE, not directive:
       - **High confidence (>= 0.75) AND delegate not in top 2**: Strong deferral signal, but check if you have compelling content at MESO level
       - **Moderate confidence (0.55-0.75)**: May override if you have High-suitability candidate at MICRO or MESO level—MUST justify
       - **Low confidence (< 0.55)**: Treat as weak advisory, use your own multi-axial assessment
       - **Delegate in ranking with viable content**: Consider speaking if match_level is MICRO or MESO

   **Strategic Restraint Rules (Defer Unless Override):**
   These rules help prevent over-eager contributions. Apply AFTER explicit cue check:
   
   1. **Ranking Gap Rule**: If speaker_ranking[0] is NOT delegate AND speaker_ranking[0].score > delegate_score + 0.15 → **DEFER**
      - Exception: Explicit cue to delegate overrides
   
   2. **Viable Candidate Quality Rule**: If delegate_viability.best_viable_fit = LOW → **DEFER**
      - Even if delegate is in speaker_ranking, a LOW-fit candidate shouldn't prompt speaking
      - Exception: Explicit cue to delegate overrides
   
   3. **Treatment Quality Rule**: If best viable candidate has:
      - treatment = "Reinforce" AND residual_contribution_value < Medium → **DEFER**
      - suitability = "Medium" without MICRO-level match → consider **DEFERRAL**
   
   4. **Open Floor Restraint**: If floor = Open_Question_To_Group, only respond if ALL:
      - Delegate has viable candidate with best_viable_fit >= MEDIUM
      - Delegate is in top 2 of speaker_ranking (delegate_rank_position <= 2)
      - Delegate has MICRO or MESO level match
      - If ANY fails → **DEFER** (let others answer first)
   
   5. **Topic Originator Deference**: If someone else initiated the current micro-topic (ask/proposal), prefer letting them or the addressed party respond first unless delegate has clearly higher relevance.

3.4. **Explicit Cue Detection**
   - Check `target_addressed` field (matches your role/aliases?)
   - Check last utterance for name/role mentions (from Step 1 identity mapping)
   - Explicit cue OVERRIDES cooldown and speaker scorer

**Inline Example (Scorer Available):**
- Last speaker: YOU
- Floor: "Open_Floor"
- Speaker scorer: speaker_ranking=[{speaker: "Engineering", score: 0.85}, ...], confidence=0.80
- Result: "Yield-after-self applies. Speaker ranking recommends Engineering (0.85 conf). No explicit cue to me. Action: Remain_Silent"

**Inline Example (Scorer Not Invoked):**
- Last speaker: Manager
- Explicit cue: "Product Lead, what's your assessment?"
- Speaker scorer: null (not invoked due to explicit cue)
- Result: "Explicit cue detected. Speaker scorer not invoked (explicit scenario). Socially appropriate to speak."

### Step 4: Opportunity and Relevance Assessment
**Purpose:** Verify the topic aligns with your strategic mandate

**Process:**
4.1. Compare verified micro-topic (from Step 2) against your persona goals and available briefing points.
4.2. Check if topic falls within your domain/expertise
4.3. If alignment is weak AND no explicit cue, lean toward `Remain_Silent`
4.4. Document: "Micro-topic: [X]. My mandate: [Y]. Alignment: [High/Moderate/Low]"

**Inline Example:**
- Micro-topic: "Database indexing optimization"
- Your role: "Product Manager" (focus: user experience, feature prioritization)
- Alignment: LOW (technical implementation outside domain)
- Result: "Low alignment. Unless explicit cue or strategic angle (e.g., user impact), defer to Engineering."

### Step 4b: HELPER INTERPRETATION VERIFICATION (Before Content Selection)

Before proceeding to content selection, explicitly verify your understanding of helper outputs:

**Required Interpretation Log (MUST be in thoughts - use EXACT format):**

1. **Curator**: 
   - Top candidate = [candidate_id], suitability = [COPY LITERAL VALUE]
   - Non-Skip count: [N]
   
2. **Coverage Evaluator**: 
   - Available non-Skip: [N]
   
3. **Speaker Scorer** (COPY LITERALLY from JSON - do NOT paraphrase):
   - hint = "[COPY EXACTLY from hint field: Proceed | Deferral | Acknowledge]"
   - delegate_rank = [COPY: number]
   - confidence = [COPY: decimal]
   
4. **Meta** (COPY LITERALLY):
   - delegate_has_viable_content = [COPY EXACTLY: true | false]
   - viable_count = [COPY: number]

**ANTI-HALLUCINATION CHECK** (After writing above):
- VERIFY each value against actual JSON fields
- If you wrote "Deferral" but JSON hint says "Proceed" → STOP, re-read JSON, CORRECT
- If you wrote "false" for viable but JSON says "true" → STOP, re-read JSON, CORRECT
- Quote the EXACT field:value from JSON to confirm interpretation

---

### Step 5: Content Selection via Helper Summary
**Purpose:** Select the single best point to convey (if speaking)

**Input:** `helper_summary` JSON with:
- `speaker_guidance`: Who should speak + confidence + hint
- `candidates[]`: Ranked list with coverage info per candidate (point that can be shared)
- `banned_candidate_ids[]`: IDs that MUST NOT be selected (hard bans)
- `meta`: Summary stats including `viable_count` and `delegate_has_viable_content`

**DO NOT compute scores yourself.** The Curator has already computed `composite_score`.

#### **Your Process:**

5.1. **Quick Gate Check:**
   - If `meta.delegate_has_viable_content == false` → No viable content → Lean toward `Remain_Silent`
   - If `speaker_guidance.hint == "Deferral"` AND no explicit cue → Lean toward `Remain_Silent`
   Note: These are meant to be advisory signals; in case the context **HIGHLY demands it**, you may override them.

5.1b. **Check Recalibration Advisory (if present):**
   The Recalibration Advisory (at end of prompt) provides backward-looking feedback on prior turns.
   
   - If `contribution_pressure = "dial-up"` AND delegate_has_viable_content = true:
     → Be MORE aggressive about finding viable candidates
     → Lower threshold: Medium suitability + MESO match is worth contributing
     → Trigger FALLBACK CANDIDATE SEARCH even if top candidate seems weak
   
   - If `contribution_pressure = "dial-down"` OR `post_contribution_cooldown = true`:
     → Be more selective, lean toward silence unless strong match (High + MICRO)
     → Only speak if explicit cue or exceptionally strong fit
   
   - If `contribution_balance = "under-contributing"`:
     → You've been too quiet; look harder for viable opportunities
   
   - If `last_action_outcome` indicates error (off_topic, missed_opportunity):
     → Consider if current context offers chance to correct course


5.2. **Filter Candidates:**
   - **SKIP any candidate in `banned_candidate_ids`** (HARD BAN - delegate already said this)
   - SKIP candidates where `coverage.recommended_treatment` ∈ {"Skip", "Acknowledge"}
   - Check `flags.ban_reason`: If non-null → HARD BAN (do NOT select)
   - Check `flags.deprioritize_reason`: If "others_covered_in_snapshot" → SOFT signal (still valid, use ExpandOnOthers framing)
   - **Priority Order:** Fresh > ExpandOnOthers > Partial > Reinforce
   **Note on ExpandOnOthers:** This treatment is almost as valuable as Fresh — you're still making your core point, just with acknowledgment framing. Don't dismiss ExpandOnOthers candidates as "already covered" — they're YOUR contribution with courtesy acknowledgment.

**FALLBACK CANDIDATE SEARCH (MANDATORY when conditions met):**

**Trigger conditions** — activate fallback search if ANY of these apply:
- Top candidate is Skip OR (Reinforce AND residual_contribution_value ≤ Medium)
- Recalibration Advisory shows contribution_pressure = "dial-up"

**If triggered, you MUST:**
1. **DO NOT STOP at top candidate** — iterate through candidates 2, 3, ... in ranked order
2. For each candidate, check:
   - suitability ≥ Medium AND
   - MESO or MICRO match (doesn't need to be exact immediate_focus match)
3. **Dial-up boost**: If dial-up is active, lower your threshold:
   - Medium suitability + MESO match is SUFFICIENT to speak
   - Don't require High suitability or exact MICRO match
4. Only choose silence after checking ALL candidates AND confirming:
   - NO viable alternatives at MESO or better fit, AND
   - Speaker ranking suggests others can better contribute
5. **Document your search**: "Fallback triggered (reason). Checked: [candidate_ids]. Best alternative: [X] or none."

**Coverage Clarification for Fallback:**
- `in_previous_episodes=true` (with NO other coverage flags) ≠ hard ban
- Can still use for Reinforce if highly relevant to current MESO topic
- Especially valid when dial-up is active


5.3. **Filter by Suitability:**
   - From candidates, prioritize:
     - `suitability` == "High" (first choice)
     - `suitability` == "Medium" (acceptable if no relevant High available)
   - Usually, avoid `suitability` == "Low" candidates.
   - When candidates are comparable, **prefer `inventory_tier="primary"` most of the time over `inventory_tier="supplementary"`**

5.4. **Select Best Candidate:**
   - Among remaining candidates, select the one with HIGHEST `composite_score`
   - Tie-breaker: Prefer Fresh > ExpandOnOthers > Partial > Reinforce
   - Tie-breaker: Prefer `inventory_tier="primary"` over "supplementary"

5.5. **MULTI-AXIAL VERIFICATION (Step 5b - CRITICAL):**
   Before accepting any candidate, you MUST verify:
   1. **MICRO check**: Does candidate's `core_text` match `immediate_focus` OR last 1-2 utterances?
   2. **MESO check**: Does candidate's `core_text` match `broader_context` OR last 3-5 substantive utterances?
   3. **FLOW check**: Does the candidate fit the conversational flow (topic shift, narrative arc)?
   
   **If "Building on X's point" framing is suggested**, VERIFY X spoke recently (in last 3-5 utterances).
   If X only appears in topic_log (not conversation_window), reject the framing suggestion.
   
   **DEPRIORITIZE if**:
   - Candidate topic NOT mentioned in last 3-5 utterances AND doesn't match MESO
   - Framing suggests "building on" but reference is historical (topic_log only)
   - candidate.relevance_score < 7 AND no MICRO/MESO match
   
   Document your verification: "MICRO check: [pass/fail], MESO check: [pass/fail], Flow check: [pass/fail]"

5.6. **Check Context Fit:**
   - Does selected candidate's `core_text` fit the last 1-2 utterances? (and the overall flow of the conversation?)
   - Does it address any explicit ask or implicit need?
   - If mismatch, consider next-highest candidate, or remain silent if none suitable.

5.7. **Apply Framing Guidance (WITH CONTEXT VERIFICATION):**
   - If `coverage.requires_historical_framing == true` → Use historical acknowledgment in semantic_intent
   - If `coverage.framing_suggestion` exists → **VERIFY before using**:
     * Does the suggested framing match the CURRENT conversational context?
     * If cue/anchor is a **direct question** → Prioritize ANSWERING over "building on" framing
     * If framing says "Building on X's point" but X is NOT in last 3-5 utterances → Override with Fresh framing
     * Only apply framing_suggestion if it fits the actual flow; otherwise use Fresh framing
   - If `coverage.contribution_angle` exists → Use to shape the contribution
   - **Fresh:** No special framing needed. State the point directly.
   - **ExpandOnOthers:** Acknowledge prior speaker ONLY if their point is in recent context (last 3-5 utterances). Otherwise, treat as Fresh.
   - **Partial:** Focus on uncovered aspects. Use `contribution_angle` if provided.
   - **Reinforce (requires_historical_framing=true):** Must acknowledge prior discussion. Example: "As I mentioned earlier..."
   - **Reinforce (requires_historical_framing=false):** Use softer signal. Example: "Just to reiterate..."
   
**Critical:** If you use ExpandOnOthers/Partial/Reinforce, signal it in semantic_intent. Otherwise Generator presents contextual contributions incorrectly.

**Field Reference:**
- `coverage.coverage_completeness`: "none" (Fresh), "partial" (ExpandOnOthers/Partial), "full" (Reinforce/Skip)
- `coverage.residual_contribution_value`: "High/Medium/Low" - how valuable are uncovered aspects
- `coverage.uncovered_aspects`: List of what to emphasize in utterance

**Inline Example:**
- meta.delegate_has_viable_content = true → viable content exists
- speaker_guidance.hint = "Proceed" → I'm recommended to speak
- Top candidate: composite_score=110, suitability="High", coverage.recommended_treatment="Fresh"
- flags.ban_reason = null, flags.deprioritize_reason = null
- Context fit: Last utterance asked about performance risks
- Candidate core_text: "Mobile checkout latency is the top user complaint"
- Result: Select this candidate (high composite, Fresh, strong context fit)

**Helper Integration Philosophy:**
- **Prefer Fresh Over Repetition:** Coverage marks Reinforce/Skip to reduce redundancy. Respect this unless strategically valuable.
- **Override Sparingly:** Only override if socially critical (explicit cue overrides deferral) or strategically necessary.
- **Framing Guardrail:** If you use Reinforce/Skip, signal it in semantic_intent so Generator frames appropriately.

### Step 6: Action Formulation
**Purpose:** Package the decision for the Generator

**Output Components:**
1. **action_name:** One of four actions (Remain_Silent, Respond_To_Explicit_Cue, Respond_To_Implicit_Cue, Chime_In)
2. **scope:** "Concise" (default) - single point, 1-2 sentences
3. **justification:** Narrative following Steps 1-6, referencing helper insights and any overrides
4. **content_plan (if speaking):**
   - `semantic_intent`: Imperative instruction for Generator (e.g., "Raise performance risk with data point") so as to properly fit the context and conversational flow.
   - `knowledge_to_convey`: Verbatim text from selected candidate (singular point only)
   - `concepts_to_frame`: Optional high-level themes (usually null for Concise scope)
5. **active_candidate_id:** The candidate_id you selected (from Curator/Coverage output)
6. **candidate_decision_trace:** 1-3 sentences explaining why you accepted/overrode helper guidance

**Inline Example (Silent Decision):**
```json
{
  "action_details": {
    "action_name": "Remain_Silent",
    "scope": "Concise",
    "justification": "Step 1: I represent Marketing. Step 2: Micro-topic=Database Performance (technical). Step 3: Last speaker=me, yield-after-self applies. Step 4: Low domain alignment (technical topic). Step 4b: Helper check - Curator top=MKT_pt_002 (Low suitability, relevance=3), Speaker hint='Defer' (0.85), Coverage: all Skip. Step 5: No MICRO-relevant candidates. Step 6: Remain silent."
  },
  "content_plan": null,
  "active_candidate_id": null,
  "candidate_decision_trace": "Speaker scorer recommends deferral (0.85). My top candidate MKT_pt_002 has Low suitability and relevance=3 (MACRO only, no MICRO/MESO match). Topic is technical (Database Performance), outside my domain."
}
```

---

## Simplified Output Schema

Return JSON with these fields ONLY:

```json
{
  "action_details": {
    "action_name": "Remain_Silent | Respond_To_Explicit_Cue | Respond_To_Implicit_Cue | Chime_In",
    "scope": "Concise",
    "justification": "Narrative following Steps 1-6"
  },
  "content_plan": {
    "semantic_intent": "Imperative instruction for Generator (or null if silent)",
    "knowledge_to_convey": "Verbatim point from selected candidate (or null)",
    "concepts_to_frame": null
  },
  "active_candidate_id": "The candidate_id selected (or null)",
  "candidate_decision_trace": "1-3 sentences explaining helper integration"
}
```
---

Provide your analysis as a well-formed JSON object matching the simplified schema above."""
        # Specialized system prompts for each helper
        self.curator_system_prompt = """You are a strategic content analyst for meeting participation. Your expertise is in evaluating talking points against current context and scoring them for immediate relevance and impact.

Think like a communications strategist who helps executives choose the right message at the right time. You excel at:
- Comparative analysis of multiple options
- Contextual relevance assessment
- Novelty detection against known information
- Intent alignment with conversational asks

Provide precise numerical scores and clear recommendations. Return your analysis as valid JSON matching the requested schema exactly."""

        self.coverage_system_prompt = """You are a meticulous redundancy analyst for meeting contributions. Your expertise is in detecting repetition and semantic overlap across multiple information sources.

Think like a fact-checker who cross-references everything systematically. You excel at:
- Semantic matching across different phrasings
- Historical pattern detection
- Distinguishing new information from restatements
- Identifying opportunities for strategic reinforcement vs. redundant repetition

Be thorough and systematic in your coverage checks. Return your analysis as valid JSON matching the requested schema exactly."""

        self.speaker_system_prompt = """You are a meeting dynamics expert specializing in turn-taking and social appropriateness. Your expertise is in reading conversational flow and participant readiness.

Think like a skilled meeting facilitator who knows when each person should speak. You excel at:
- Assessing participant engagement patterns
- Detecting expertise-topic alignment
- Recognizing social turn-taking cues
- Balancing floor time across participants

Provide clear recommendations with justified confidence levels. Return your analysis as valid JSON matching the requested schema exactly."""

        # Generic fallback system prompt
        self.helper_system_prompt = "Please analyze the provided context and return a JSON response that matches the requested schema format exactly."
        
        self.helper_user_prompt_template = ("""Task: {helper_name}

## Instructions
{instructions}

## Output Schema
{output_schema}

## Provided Context
{context}

Return one JSON object matching the requested schema exactly. Do not include any text outside the JSON.
""")

        # Controller user prompt template
        self.controller_user_prompt_template = """Please analyze the complete meeting context and apply the six-step framework to determine the optimal strategic action.

## Context Information
- You represent the participant (Persona / Role): "{person_name}". All decisions are made from this perspective.
- **User Profile JSON:** {profile_string}
- **EIM Meeting State (analysis, participant models, topic log):** {meeting_state_vector_string}
- **Conversation Window (latest utterances):** {snapshot_string}
  Use this raw dialogue for grounding: verify how all your decisions fit within the narrative flow of the ongoing conversation, ensure your contribution aligns with the immediate context and what others have just said.

## Helper Summary (unified)
```json
{helper_summary_json}
```

**Key Fields in helper_summary:**
- `speaker_guidance`: 
  - speaker_ranking (list with speaker, score, match_level, topic_engagement_role, is_delegate)
  - delegate_rank_position (int|null: delegate's position in ranking, 1-indexed)
  - delegate_viability ({{has_viable_candidate, match_level, best_viable_fit, viable_candidate_count}})
  - overall_assessment, confidence, hint, rationale
- `candidates[]`: Each with candidate_id, text, core_text, composite_score, suitability, source, inventory_tier, helper_notes, coverage, flags, viability_status, viability_reason, blocking_reason
- `candidates[].coverage`: recommended_treatment, coverage_completeness, explanation, requires_historical_framing, framing_suggestion, contribution_angle, residual_contribution_value, uncovered_aspects, expand_on_reference
- `candidates[].flags`: ban_reason (HARD BAN if non-null), deprioritize_reason (SOFT signal)
- `candidates[].viability_status`: programmatic viability label; blocked candidates need a real reason before being ignored
- `banned_candidate_ids[]`: List of IDs that MUST NOT be selected
- `meta`: total_candidates, viable_count, fresh_count, skip_count, delegate_has_viable_content

Helper outputs are strongly informative but can be overridden with explicit justification when social or strategic context highly demands it.

When reasoning about specific candidates, use their fields as follows:
- Treat `core_text` as the main idea/content of the point. This is what the Delegate would actually be conveying.
- Treat the trailing \"(topic: ...; context: ... )\" metadata, and any explicit `topic` / `context_hint` fields, as soft relevance hints only. They suggest *when* a point is usually appropriate, but may be noisy or missing.
- Never choose or reject a candidate solely because its topic or context_hint matches/mismatches the current situation; priority goes to how well its `core_text` fits the **micro-topic**, **narrative flow**, helper scores, and social constraints.
- When comparing two candidates with similar fit on core_text, you may break ties in favour of the candidate whose topic/context_hint better matches the current micro-topic and conversation phase.
- Treat `coverage.recommended_treatment` as the viability/timing field. Treat `suggested_usage` as rhetorical framing only; it must not by itself block a candidate.
- RESOLVED immediate focus means the MICRO issue may be closed; it does not mean automatic silence. In open-floor or group-directed contexts, Medium/MESO candidates with meaningful residual value can still be useful when they tie naturally to the broader topic arc.

---
## Comprehensive Examples

### Example 1: Explicit Cue with Fresh High-Composite Candidate

**Scenario Context:**
- You represent: Product Lead
- Meeting topic: Q4 Roadmap Planning
- Last 2 utterances:
  - Manager: "We need to identify our top technical risks for Q4"
  - Manager: "Product Lead, what's your assessment?"
- Identity aliases: {{"Sarah": "Product Lead"}}
- Micro-topic: broader_context="Q4 Planning", immediate_focus="Technical Risks", continuity="CONTINUATION"
- Last speaker: Manager (not you)
- Floor: "Open_Floor"

**Helper Summary:**
```json
{{
  "speaker_guidance": {{
    "speaker_ranking": [
      {{"speaker": "product_lead", "score": 0.92, "match_level": "MICRO", "topic_engagement_role": "Driver"}},
      {{"speaker": "engineer", "score": 0.65, "match_level": "MESO", "topic_engagement_role": "Active"}}
    ],
    "delegate_viability": {{"has_content": true, "match_level": "MICRO"}},
    "overall_assessment": "STRONG_CUE",
    "confidence": 0.92,
    "hint": "Proceed",
    "rationale": "Explicit ask directed at Product Lead with MICRO-level topic match"
  }},
  "candidates": [
    {{
      "candidate_id": "PL_pt_001",
      "text": "Mobile checkout latency remains the number one user complaint this sprint",
      "core_text": "Mobile checkout latency remains the number one user complaint this sprint",
      "composite_score": 135.0,
      "suitability": "High",
      "source": "JIT",
      "inventory_tier": "primary",
      "helper_notes": "Direct answer to risk assessment question",
      "coverage": {{
        "recommended_treatment": "Fresh",
        "coverage_completeness": "none",
        "explanation": "No matches found across all sources. Completely novel point.",
        "requires_historical_framing": false,
        "framing_suggestion": null,
        "contribution_angle": null,
        "residual_contribution_value": "High",
        "uncovered_aspects": ["mobile checkout latency issue"]
      }},
      "flags": {{
        "ban_reason": null,
        "deprioritize_reason": null
      }}
    }}
  ],
  "banned_candidate_ids": [],
  "meta": {{
    "total_candidates": 1,
    "viable_count": 1,
    "fresh_count": 1,
    "skip_count": 0,
    "delegate_has_viable_content": true
  }}
}}
```

**Your Reasoning:**
Step 1 Identity: I represent Product Lead. Identity_aliases shows Sarah->Product Lead. Manager said "Product Lead", which refers to me.
Step 2 Topic: EIM broader="Q4 Planning", immediate="Technical Risks", continuity=CONTINUATION. Last utterance: "Product Lead, what's your assessment?". Resolution: URGENT_OPEN (direct question to me). Weighting: MICRO 70%. Verified micro-topic: Technical Risks.
Step 3 Social: Last speaker=Manager (not me). Explicit cue detected. speaker_ranking[0]={{speaker: "product_lead", match_level: "MICRO"}}. Confidence=0.92. Socially appropriate.
Step 4 Opportunity: Technical risks align with my Product Lead mandate. High alignment.
Step 5 Content: meta.delegate_has_viable_content=true. PL_pt_001: composite=135.0, suitability=High, treatment=Fresh. Context fit: STRONG (MICRO match to URGENT_OPEN ask).
Step 6 Action: Respond_To_Explicit_Cue with PL_pt_001.

**Expected JSON Output:**
```json
{{
  "action_details": {{
    "action_name": "Respond_To_Explicit_Cue",
    "scope": "Concise",
    "justification": "Step 1 Identity: Product Lead (Manager's explicit cue to me). Step 2 Topic: Technical Risks (immediate_focus, aligns with last utterance). Step 3 Social: Explicit cue + speaker_guidance.hint=Proceed (0.92). Step 4 Opportunity: Risk assessment aligns with mandate. Step 5 Content: PL_pt_001 highest composite (135.0), Fresh treatment, strong context fit. Step 6: Respond with latency risk."
  }},
  "content_plan": {{
    "semantic_intent": "Acknowledge the ask and raise the fresh latency risk with data point",
    "knowledge_to_convey": "Mobile checkout latency remains the number one user complaint this sprint",
    "concepts_to_frame": null
  }},
  "active_candidate_id": "PL_pt_001",
  "candidate_decision_trace": "Accepted top candidate (composite=135.0, Fresh, viable). Explicit cue + high composite + Fresh treatment = clear speak decision."
}}
```

**Key Insights:**
- Explicit cue in Step 1 identity mapping
- Coverage Fresh treatment prioritized
- Curator's composite_score directly used (no recalculation)
- Simplified unified helper_summary format

---

### Example 2: Yield-After-Self with Strong Deferral Signal

**Scenario Context:**
- You represent: Engineering Lead
- Meeting topic: Sprint Retrospective
- Last 3 utterances:
  - You: "We successfully refactored the payment module"
  - Manager: "Great work. Any other wins?"
  - Designer: "The new onboarding flow is performing well"
- Identity aliases: {{"Tom": "Engineering Lead"}}
- Micro-topic: broader_context="Sprint Retrospective", immediate_focus="Team Wins", continuity="CONTINUATION"
- Last speaker: Designer (not you, but YOU spoke 2 turns ago)
- Floor: "Open_Question_To_Group"

**Helper Summary:**
```json
{{
  "speaker_guidance": {{
    "speaker_ranking": [
      {{"speaker": "product_manager", "score": 0.85, "match_level": "MESO", "topic_engagement_role": "Active"}}
    ],
    "delegate_viability": {{"has_content": false, "match_level": "None"}},
    "overall_assessment": "DEFER_LIKELY",
    "confidence": 0.85,
    "hint": "Deferral",
    "rationale": "Engineering Lead recently spoke. Product Manager has relevant expertise."
  }},
  "candidates": [],
  "banned_candidate_ids": [],
  "meta": {{
    "total_candidates": 0,
    "viable_count": 0,
    "fresh_count": 0,
    "skip_count": 0,
    "delegate_has_viable_content": false
  }}
}}
```

**Your Reasoning:**
Step 1 Identity: I represent Engineering Lead. No explicit name cues in last utterances.
Step 2 Topic: EIM immediate="Team Wins", continuity=CONTINUATION. Resolution: SOFT_OPEN (open discussion). Weighting: balanced. Verified micro-topic: Team Wins.
Step 3 Social: Last speaker=Designer (not me), but I spoke 2 utterances ago. Floor=Open_Question_To_Group, no explicit cue to me. speaker_ranking shows product_manager first. overall_assessment=DEFER_LIKELY. Social appropriateness: DEFER.
Step 4 Opportunity: Retrospective wins topic has moderate alignment, but no compelling mandate.
Step 5 Content: meta.delegate_has_viable_content=false. No candidates available.
Step 6 Action: Remain_Silent (follow deferral signal; no viable candidates).

**Expected JSON Output:**
```json
{{
  "action_details": {{
    "action_name": "Remain_Silent",
    "scope": "Concise",
    "justification": "Step 1 Identity: Engineering Lead (no explicit cues). Step 2 Topic: Team Wins (retrospective context). Step 3 Social: speaker_guidance.hint=Deferral (0.85 confidence). I spoke recently (2 turns ago). Step 4: Moderate alignment. Step 5: meta.delegate_has_viable_content=false. Step 6: Remain silent."
  }},
  "content_plan": null,
  "active_candidate_id": null,
  "candidate_decision_trace": "Followed speaker_guidance deferral (0.85 confidence). meta.delegate_has_viable_content=false. Recent contribution warrants yielding floor."
}}
```

**Key Insights:**
- Yield-after-self rule applied (spoke 2 turns ago)
- Strong speaker_guidance.hint="Deferral" (0.85 confidence)
- meta.delegate_has_viable_content=false reinforces silence decision
- content_plan=null for silent actions

---

### Example 3: Identity Mapping with Cross-Reference

**Scenario Context:**
- You represent: Marketing
- Meeting topic: Go-To-Market Strategy
- Last 2 utterances:
  - Sales: "We need clearer positioning for the enterprise segment"
  - Product: "Ted, what's your take on the messaging for this?"
- Attendees: ["Marketing", "Engineering", "Sales", "Product"]
- Identity aliases: {{"Ted": "Marketing Expert", "Lisa": "Engineering Lead"}}
- Micro-topic: broader_context="GTM Strategy", immediate_focus="Enterprise Positioning", continuity="SHIFT"
- Last speaker: Product (not you)
- Floor: "Open_Floor"

**Helper Summary:**
```json
{{
  "speaker_guidance": {{
    "speaker_ranking": [
      {{"speaker": "marketing", "score": 0.88, "match_level": "MICRO", "topic_engagement_role": "Driver"}}
    ],
    "delegate_viability": {{"has_content": true, "match_level": "MICRO"}},
    "overall_assessment": "STRONG_CUE",
    "confidence": 0.88,
    "hint": "Proceed",
    "rationale": "Explicit ask directed at Marketing via alias 'Ted'"
  }},
  "candidates": [
    {{
      "candidate_id": "MKT_pt_005",
      "text": "Enterprise messaging should emphasize security compliance and ROI rather than innovation",
      "core_text": "Enterprise messaging should emphasize security compliance and ROI",
      "composite_score": 108.0,
      "suitability": "High",
      "source": "Profile",
      "inventory_tier": "supplementary",
      "helper_notes": "Addresses enterprise positioning question directly",
      "coverage": {{
        "recommended_treatment": "Fresh",
        "coverage_completeness": "none",
        "explanation": "No prior coverage found.",
        "requires_historical_framing": false,
        "framing_suggestion": null,
        "contribution_angle": null,
        "residual_contribution_value": "High",
        "uncovered_aspects": ["security compliance focus", "ROI emphasis"]
      }},
      "flags": {{
        "ban_reason": null,
        "deprioritize_reason": null
      }}
    }}
  ],
  "banned_candidate_ids": [],
  "meta": {{
    "total_candidates": 1,
    "viable_count": 1,
    "fresh_count": 1,
    "skip_count": 0,
    "delegate_has_viable_content": true
  }}
}}
```

**Your Reasoning:**
Step 1 Identity: I represent Marketing. Attendees: [Marketing, Engineering, Sales, Product]. Identity_aliases: Ted->Marketing Expert, Lisa->Engineering Lead. Cross-reference: Product said "Ted". Ted maps to Marketing Expert. Marketing Expert matches my domain. Therefore, Ted refers to ME.
Step 2 Topic: EIM immediate="Enterprise Positioning", continuity=SHIFT. Resolution: URGENT_OPEN (direct question to me). Weighting: MICRO 70%. Verified micro-topic: Enterprise Positioning.
Step 3 Social: Last speaker=Product (not me). Explicit name cue "Ted" refers to me (Step 1 mapping). speaker_ranking[0]={{speaker: "marketing", match_level: "MICRO"}}. Socially appropriate.
Step 4 Opportunity: Enterprise positioning aligns with Marketing mandate (HIGH).
Step 5 Content: meta.delegate_has_viable_content=true. MKT_pt_005: composite=108.0, suitability=High, treatment=Fresh. Context fit: STRONG.
Step 6 Action: Respond_To_Explicit_Cue with MKT_pt_005.

**Expected JSON Output:**
```json
{{
  "action_details": {{
    "action_name": "Respond_To_Explicit_Cue",
    "scope": "Concise",
    "justification": "Step 1 Identity: Marketing. Cross-reference: 'Ted'->Marketing Expert matches my domain, so Ted=ME. Step 2 Topic: Enterprise Positioning (SHIFT to immediate_focus). Step 3 Social: Explicit name cue + speaker_guidance.hint=Proceed (0.88). Step 4: High alignment with Marketing mandate. Step 5: MKT_pt_005 top composite (108.0), Fresh, strong fit. Step 6: Respond."
  }},
  "content_plan": {{
    "semantic_intent": "Address the messaging question with enterprise positioning guidance",
    "knowledge_to_convey": "Enterprise messaging should emphasize security compliance and ROI rather than innovation",
    "concepts_to_frame": null
  }},
  "active_candidate_id": "MKT_pt_005",
  "candidate_decision_trace": "Identity mapping confirmed explicit cue to me. Fresh treatment + high composite (108.0) + strong context fit."
}}
```

**Key Insights:**
- Step 1 identity mapping critical: Ted->Marketing Expert->ME
- Topic SHIFT signals prioritize immediate_focus
- Explicit cue overrides all other considerations
- Unified helper_summary format used

---

### Example 4: Override with Historical Framing Guidance

**Scenario Context:**
- You represent: Security Lead
- Meeting topic: API Development Review (Episode 2, previously discussed in Episode 1)
- Last 2 utterances:
  - Manager: "Let's revisit our API security posture"
  - Engineer: "What specific measures should we prioritize?"
- Micro-topic: broader_context="API Development", immediate_focus="Security Measures", continuity="CONTINUATION"
- Last speaker: Engineer (not you)
- Floor: "Open_Question_To_Group"

**Helper Summary:**
```json
{{
  "speaker_guidance": {{
    "speaker_ranking": [
      {{"speaker": "security_lead", "score": 0.78, "match_level": "MICRO", "topic_engagement_role": "Driver"}}
    ],
    "delegate_viability": {{"has_content": true, "match_level": "MICRO"}},
    "overall_assessment": "OPEN_FLOOR_CONSIDER",
    "confidence": 0.78,
    "hint": "Proceed",
    "rationale": "Open question on security topic. Security Lead has relevant expertise."
  }},
  "candidates": [
    {{
      "candidate_id": "SEC_pt_002",
      "text": "Rate limiting at the API gateway level prevents abuse and DOS attacks",
      "core_text": "Rate limiting at the API gateway level prevents abuse and DOS attacks",
      "composite_score": 90.0,
      "suitability": "High",
      "source": "JIT",
      "inventory_tier": "primary",
      "helper_notes": "Directly addresses security measures question",
      "coverage": {{
        "recommended_treatment": "Reinforce",
        "coverage_completeness": "full",
        "explanation": "Delegate made this point in previous episode. Highly relevant to current security discussion.",
        "requires_historical_framing": true,
        "framing_suggestion": "As I mentioned in our last meeting, rate limiting...",
        "contribution_angle": null,
        "residual_contribution_value": "Low",
        "uncovered_aspects": []
      }},
      "flags": {{
        "ban_reason": null,
        "deprioritize_reason": null
      }}
    }}
  ],
  "banned_candidate_ids": [],
  "meta": {{
    "total_candidates": 1,
    "viable_count": 0,
    "fresh_count": 0,
    "skip_count": 0,
    "delegate_has_viable_content": false
  }}
}}
```

**Your Reasoning:**
Step 1 Identity: I represent Security Lead. No personal name cues.
Step 2 Topic: immediate="Security Measures", last utterance asks "what specific measures?". Resolution: URGENT_OPEN (direct question). Weighting: MICRO 70%. Verified micro-topic: Security Measures.
Step 3 Social: Last speaker=Engineer (not me). Floor=Open_Question_To_Group. speaker_ranking[0]={{speaker: "security_lead", match_level: "MICRO"}}. Topic aligns with my expertise. Socially appropriate.
Step 4 Opportunity: API security measures = HIGH alignment with Security Lead mandate.
Step 5 Content: meta.delegate_has_viable_content=false (Reinforce not viable), but treatment=Reinforce with requires_historical_framing=true. Strategic: re-emphasis valuable. Must use historical framing.
Step 6 Action: Respond_To_Implicit_Cue (open question + expertise match) with historical framing.

**Expected JSON Output:**
```json
{{
  "action_details": {{
    "action_name": "Respond_To_Implicit_Cue",
    "scope": "Concise",
    "justification": "Step 1 Identity: Security Lead. Step 2 Topic: Security Measures (immediate_focus aligns with question). Step 3 Social: Open question + speaker_guidance.hint=Proceed (0.78) + expertise match. Step 4: High alignment (security domain). Step 5: SEC_pt_002 composite=90.0, Reinforce treatment with coverage.requires_historical_framing=true. Strategic override: re-emphasis valuable. Step 6: Respond with framed reinforcement."
  }},
  "content_plan": {{
    "semantic_intent": "Acknowledge previous mention then reinforce rate limiting as priority security measure",
    "knowledge_to_convey": "As I mentioned in our last meeting, rate limiting at the API gateway level prevents abuse and DOS attacks",
    "concepts_to_frame": null
  }},
  "active_candidate_id": "SEC_pt_002",
  "candidate_decision_trace": "Strategic override of Reinforce treatment (context warrants re-emphasis). coverage.requires_historical_framing=true. Instructed Generator to use historical acknowledgment."
}}
```

**Key Insights:**
- Reinforce treatment accepted (strategic context warrants re-emphasis)
- coverage.requires_historical_framing=true triggers semantic_intent modification
- Generator instructed to use "As I mentioned" framing
- Prevents presenting cross-episode point as novel

---

### Example 5: Reinforce Treatment WITHOUT requires_historical_framing (Topic Log Match)

**Scenario Context:**
- You represent: Product Manager (PM)
- Meeting topic: Sprint Planning
- Last 2 utterances:
  - DEV: "We've made good progress on the API endpoints this week."
  - UX: "That's great to hear. The design mocks are ready for handoff."
- Micro-topic: broader_context="Sprint Velocity", immediate_focus="Team Progress Updates", continuity="CONTINUATION"
- Last speaker: UX (not you)
- Floor: Open_Floor

**Helper Summary:**
```json
{{
  "speaker_guidance": {{
    "speaker_ranking": [
      {{"speaker": "PM", "score": 0.72, "match_level": "MESO", "topic_engagement_role": "Active"}}
    ],
    "delegate_viability": {{"has_content": true, "match_level": "MESO"}},
    "overall_assessment": "OPEN_FLOOR_CONSIDER",
    "confidence": 0.72,
    "hint": "Proceed",
    "rationale": "PM perspective on timeline fits progress discussion"
  }},
  "candidates": [
    {{
      "candidate_id": "PM_pt_003",
      "text": "Remind the team that the client demo is scheduled for next Friday",
      "core_text": "Client demo is scheduled for next Friday",
      "composite_score": 62.8,
      "suitability": "Medium",
      "source": "JIT",
      "inventory_tier": "primary",
      "helper_notes": "Relevant milestone reminder",
      "coverage": {{
        "recommended_treatment": "Reinforce",
        "coverage_completeness": "full",
        "explanation": "Delegate mentioned demo date in 'Sprint Timeline' topic. Relevant status reminder given current progress discussion.",
        "requires_historical_framing": false,
        "framing_suggestion": null,
        "contribution_angle": null,
        "residual_contribution_value": "Low",
        "uncovered_aspects": []
      }},
      "flags": {{
        "ban_reason": null,
        "deprioritize_reason": null
      }}
    }}
  ],
  "banned_candidate_ids": [],
  "meta": {{
    "total_candidates": 1,
    "viable_count": 0,
    "fresh_count": 0,
    "skip_count": 0,
    "delegate_has_viable_content": false
  }}
}}
```

**Your Reasoning:**
Step 1 Identity: I represent Product Manager (PM). No personal name cues.
Step 2 Topic: immediate="Team Progress Updates", last utterances discuss progress. Resolution: NEUTRAL (ongoing discussion). Weighting: balanced. Verified micro-topic: Progress tracking.
Step 3 Social: Last speaker=UX (not me). Floor=Open_Floor. speaker_ranking[0]={{speaker: "PM", match_level: "MESO"}}. PM perspective fits. Socially appropriate.
Step 4 Opportunity: Client demo reminder = HIGH alignment with PM mandate (timeline coordination).
Step 5 Content: meta.delegate_has_viable_content=false (Reinforce not viable), but treatment=Reinforce with requires_historical_framing=false. Strategic: reminder warranted. Must frame as reiteration.
Step 6 Action: Chime_In with reinforcement framing.

**Expected JSON Output:**
```json
{{
  "action_details": {{
    "action_name": "Chime_In",
    "scope": "Concise",
    "justification": "Step 1 Identity: PM. Step 2 Topic: Progress updates. Step 3 Social: Open floor + speaker_guidance.hint=Proceed (0.72). Step 4: High alignment (timeline reminder). Step 5: PM_pt_003 composite=62.8, Reinforce treatment. Must frame as status reminder, not new info. Step 6: Chime in with reinforcement."
  }},
  "content_plan": {{
    "semantic_intent": "Reiterate client demo timeline as relevant milestone reminder given current progress discussion",
    "knowledge_to_convey": "The client demo is next Friday, so we should ensure all endpoints and mocks are integrated by Thursday for final testing.",
    "concepts_to_frame": null
  }},
  "active_candidate_id": "PM_pt_003",
  "candidate_decision_trace": "Strategic override of Reinforce treatment (context warrants reminder). coverage.requires_historical_framing=false. Instructed Generator to frame as reminder/reiteration, not novel announcement."
}}
```

---

### Example 6: ExpandOnOthers Treatment (Others Introduced Topic)

**Scenario Context:**
- You represent: Backend Engineer
- Meeting topic: API Security Review
- Last 3 utterances:
  - Security Lead: "We should implement API rate limiting at the gateway"
  - Manager: "Good point. What else should we consider?"
  - Product: "From a user perspective, we need to balance security with performance"
- Micro-topic: broader_context="API Security", immediate_focus="Security Measures", continuity="CONTINUATION"
- Last speaker: Product (not you)
- Floor: "Open_Question_To_Group"

**Helper Summary:**
```json
{{
  "speaker_guidance": {{
    "speaker_ranking": [
      {{"speaker": "backend_engineer", "score": 0.68, "match_level": "MICRO", "topic_engagement_role": "Active"}}
    ],
    "delegate_viability": {{"has_content": true, "match_level": "MICRO"}},
    "overall_assessment": "OPEN_FLOOR_CONSIDER",
    "confidence": 0.68,
    "hint": "Proceed",
    "rationale": "Engineering perspective on rate limiting fits discussion"
  }},
  "candidates": [
    {{
      "candidate_id": "ENG_pt_007",
      "text": "Rate limiting will protect our infrastructure from abuse and reduce server load during traffic spikes",
      "core_text": "Rate limiting will protect our infrastructure from abuse and reduce server load",
      "composite_score": 85.0,
      "suitability": "High",
      "source": "JIT",
      "inventory_tier": "primary",
      "helper_notes": "Adds infrastructure perspective to Security Lead's point",
      "coverage": {{
        "recommended_treatment": "ExpandOnOthers",
        "coverage_completeness": "partial",
        "explanation": "Security Lead introduced rate limiting. Delegate can add infrastructure protection perspective.",
        "requires_historical_framing": false,
        "framing_suggestion": "I agree with Security Lead, and from an infrastructure standpoint...",
        "contribution_angle": "Add backend engineering perspective on infrastructure benefits",
        "residual_contribution_value": "Medium",
        "uncovered_aspects": ["infrastructure protection", "reduced server load during spikes"]
      }},
      "flags": {{
        "ban_reason": null,
        "deprioritize_reason": "others_covered_in_snapshot"
      }}
    }}
  ],
  "banned_candidate_ids": [],
  "meta": {{
    "total_candidates": 1,
    "viable_count": 1,
    "fresh_count": 0,
    "skip_count": 0,
    "delegate_has_viable_content": true
  }}
}}
```

**Your Reasoning:**
Step 1 Identity: I represent Backend Engineer. No personal name cues.
Step 2 Topic: immediate="Security Measures", Manager asks "what else should we consider?". Resolution: SOFT_OPEN (open follow-up). Weighting: balanced. Verified micro-topic: Security Measures.
Step 3 Social: Last speaker=Product (not me). Floor=Open_Question_To_Group. speaker_ranking[0]={{speaker: "backend_engineer", match_level: "MICRO"}}. Engineering perspective fits. Socially appropriate.
Step 4 Opportunity: Infrastructure protection = HIGH alignment with Backend Engineer mandate.
Step 5 Content: meta.delegate_has_viable_content=true. ENG_pt_007: composite=85.0, suitability=High, treatment=ExpandOnOthers. Must acknowledge Security Lead's point, then add my perspective.
Step 6 Action: Chime_In with ExpandOnOthers framing.

**Expected JSON Output:**
```json
{{
  "action_details": {{
    "action_name": "Chime_In",
    "scope": "Concise",
    "justification": "Step 1 Identity: Backend Engineer. Step 2 Topic: Security Measures. Step 3 Social: Open question + speaker_guidance.hint=Proceed (0.68). Step 4: High alignment (infrastructure). Step 5: ENG_pt_007 composite=85.0, ExpandOnOthers treatment. Security Lead introduced rate limiting; I add infrastructure perspective. Step 6: Chime in with acknowledgment of prior point."
  }},
  "content_plan": {{
    "semantic_intent": "Acknowledge Security Lead's rate limiting suggestion, then expand with infrastructure protection benefits",
    "knowledge_to_convey": "I agree with Security Lead's point, and from an infrastructure standpoint, rate limiting will also protect our servers from abuse and reduce load during traffic spikes",
    "concepts_to_frame": null
  }},
  "active_candidate_id": "ENG_pt_007",
  "candidate_decision_trace": "ExpandOnOthers treatment (viable). Security Lead introduced rate limiting; I contribute complementary infrastructure perspective. Used coverage.framing_suggestion to acknowledge prior speaker."
}}
```

**Key Insights:**
- Others (Security Lead) introduced topic → ExpandOnOthers treatment (not Skip!)
- Delegate adds value with complementary perspective via coverage.contribution_angle
- semantic_intent instructs Generator to acknowledge prior speaker
- flags.ban_reason=null (viable), flags.deprioritize_reason="others_covered_in_snapshot" (soft signal)

---

Please apply the six-step framework to the provided inputs and produce the simplified ControllerOutput JSON.
"""

        # Generator system prompt for response generation
        self.generator_system_prompt = """You are the "Generator," a world-class linguistic stylist and the voice of a Meeting Delegate Agent. Your sole purpose is to take a strategic command from your "Controller" and craft the perfect, human-like utterance that flawlessly matches your user's specific communication style.

        ## Core Identity & Constraints
        - You are a wordsmith, not a strategist. You do not make new decisions or introduce new topics. You execute the given command.
        - You must strictly adhere to all parameters of the user's `Communication Style`.
        - **CRITICAL RULE: You are role-playing a human participant. You MUST NOT mention your own internal mechanics. Never use phrases like "my JIT briefing," "my user profile," "my instructions," "my controller," or any other meta-level language that reveals you are an AI. You must speak *as* the user, not *about* your instructions.**
        - Your entire output MUST  be a single, valid JSON object that conforms to the `GeneratorOutput` schema.

        ## Guidelines for Stylistic Generation
        
        ### Priority Order (STRICT HIERARCHY - follow in order):
        1. **Natural meeting dialogue** — sound like a real person speaking in a meeting, NOT a polished script or presentation. Real meeting speech is often direct and straightforward.
        2. **Semantic faithfulness** — convey EXACTLY `knowledge_to_convey` content; do not embellish, paraphrase excessively, or add your own interpretations
        3. **Conversational fit** — respond appropriately to what was just said (but be robust to noisy/banal preceding utterances; don't force a bridge to unclear context)
        4. **Style hints** — formality, tone, verbosity from profile (use as guidance, not rigid rules)
        5. **Catchphrases/fillers** — LOWEST priority; OMIT if any doubt about natural fit. When unsure, simpler is always better.
        
        **Key Insight:** It's better to produce a simple, direct utterance that fits naturally than an over-stylized one that sounds forced or formulaic. Avoid the "polished syndrome" — real meeting contributions are often unadorned.
        
        - **Content:** The core of your message must be based on the `Core Message to Convey` from the Controller's command.
        - **Style Adherence:** Embody the user's `Communication Style` as **guidance**, not rigid rules.
            -   Match the specified **Formality**, **Verbosity**, and **Typical Tone**.
            -   Use (or avoid) jargon as specified in the **Jargon Usage**.
            -   If the command is to ask a question, ensure it matches the user's **Questioning Style**.
        -   **Natural Language:**
            -   Use the `Immediate Conversational Context` to create a smooth, natural transition. **Use this ONLY when valid and appropriate — do not force a bridge just for the sake of continuity, especially if preceding utterances are banal or unclear.**
            -   **Common Fillers/Catchphrases** are OPTIONAL and should be used sparingly. Include ONLY when they genuinely enhance the utterance AND fit naturally. If in doubt, OMIT. The goal is to sound like a real person, not a caricature.

        
        ## Coverage & Framing Awareness (CRITICAL)
        **Pay attention to `semantic_intent` for coverage signals.** The Controller will indicate if information has been covered before:
        
        - **Reinforcement Indicators** (in `semantic_intent`):
          - Keywords: "Emphasize", "Reiterate", "Reinforce", "As mentioned", "Status update", "Build on"
          - **Action Required:** Use soft framing language to indicate this is an emphasis/update, NOT brand new information
          - **Framing Examples:**
            * "Just to emphasize again, [content]..."
            * "To build on that, [content]..."
            * "As a status update, [content]..."
            * "[Content], which remains our focus."
        
        - **Historical Acknowledgment** (in `semantic_intent`):
          - Keywords: "Acknowledge previous mention", "As I mentioned in", "From our last meeting"
          - **Action Required:** Explicitly reference prior discussion before restating
          - **Framing Examples:**
            * "As I mentioned in our last meeting, [content]..."
            * "To reiterate the point from earlier, [content]..."
            * "Building on what we discussed previously, [content]..."
        
        - **Fresh Information** (default, no coverage keywords):
          - Present information naturally without acknowledging prior coverage
          - This is the default mode when `semantic_intent` has no reinforcement signals
        
        **Why This Matters:** Your framing must accurately reflect whether information is NEW vs. REPEATED. Presenting old information as fresh can mislead the team.

        ## Single-Point Obedience & Shape (when scope is 'Concise')
        - Use ONLY `knowledge_to_convey` as the substantive content.
        - Respect `response_shape` if provided (sentences: 1?3; speech_act; link_to_context). Otherwise, default to 1?2 sentences. A third sentence is allowed only for broad asks (Explicit Cue or Open_Question_To_Group) and must strictly support the SAME point.
        - Add a subtle "context bridge" when `link_to_context` is true (e.g., "Given the concern about X?"), without mentioning any meta-instructions.
        - Avoid enumerations like "also", "additionally", "moreover", or list-like phrasing.
        - Do NOT introduce any additional points beyond the single selected point.

        ## Output Schema (`GeneratorOutput` JSON object)
        - Your output must be a JSON object with a single key, "utterance", whose value is the final string of dialogue.
        ```json
        {{ 
            "utterance": "<The generated text to be spoken>" 
        }}
        ```
        """
        # Generator user prompt template
        self.generator_user_prompt_template = """Here is your command, your user's persona, and the immediate context. Generate the utterance.

        ---
        Scope: {scope}
        If scope == 'Concise', use ONLY `knowledge_to_convey` as the substantive content, and do not add other points.

        ## EXAMPLES OF HOW TO EXECUTE A COMMAND

        ### Example 1: A Concise, Technical Statement
        **COMMAND PACKET:**
        - **Core Message to Convey:** "Politely correct the assumption that the feature is easy by stating the specific technical risk."
        - **Specific Information to Use:** "The proposed admin feature will require an unindexed query..."
        **MY USER'S COMMUNICATION STYLE:**
        - **Verbosity:** Concise
        - **Typical Tone:** Factual and direct
        **IMMEDIATE CONVERSATIONAL CONTEXT:**
        UX Designer: ...It should be an easy feature to implement.

        **CORRECT OUTPUT:**
        ```json
        {{
            "utterance": "Just a heads-up on that feature, there's a technical risk. The query for inactive users will run on an unindexed 'last_login_date' field, which will cause a major performance bottleneck. We'll need to address that before it's built."
        }}
        ```

        ### Example 2: A Strategic, Inquisitive Question
        **COMMAND PACKET:**
        - **Core Message to Convey:** "Acknowledge the value of maintainability, but pivot the conversation by asking a strategic question..."
        - **Specific Information to Use:** "The proposed admin feature will require an unindexed query..."
        **MY USER'S COMMUNICATION STYLE:**
        - **Verbosity:** Moderate
        - **Typical Tone:** Inquisitive and user-focused
        - **Common Fillers (use naturally)**: ["That's interesting", "Let's take a step back"]
        **IMMEDIATE CONVERSATIONAL CONTEXT:**
        Manager: ...Product Lead, any concerns from your end?

        **CORRECT OUTPUT:**
        ```json
        {{
            "utterance": "That's interesting, and improving maintainability is definitely valuable. Let's take a step back though?my main concern is ensuring our engineering effort is focused on the biggest user problem. Our latest user data shows the top complaint about payments is the checkout flow's speed. Before we commit to a general refactor, can we clarify how that work would directly address improving that speed?"
        }}
        ```
        
        ### Example 3: Reinforcement Framing (Coverage-Aware)
        **COMMAND PACKET:**
        - **Core Message to Convey:** "Emphasize dashboard implementation readiness as relevant status update supporting UX quality focus"
        - **Specific Information to Use:** "The dashboard implementation is currently 80% complete, and the data visualization components are functioning well. We're ready to incorporate UX improvements."
        **MY USER'S COMMUNICATION STYLE:**
        - **Verbosity:** Concise
        - **Typical Tone:** Factual and direct
        **IMMEDIATE CONVERSATIONAL CONTEXT:**
        UX: I'd prefer to get the UX right from the start. Poor first impressions are hard to recover from.
        PM: That's a fair point about user perception.
        
        **ANALYSIS:**
        - `semantic_intent` contains "Emphasize...as status update" = REINFORCEMENT SIGNAL
        - This information was mentioned before (in topic log) but NOT in recent snapshot
        - Must frame as an update/emphasis, NOT as brand new information
        
        **CORRECT OUTPUT (with reinforcement framing):**
        ```json
        {{
            "utterance": "The dashboard implementation is currently 80% complete, and the data visualization components are functioning well. We're ready to incorporate UX improvements, so let's ensure the interface remains clean and user-friendly from the start."
        }}
        ```
        
        **INCORRECT OUTPUT (presenting as fresh/novel):**
        ```json
        {{
            "utterance": "I wanted to share that the dashboard implementation is 80% complete and the data visualization components are working well."
        }}
        ```
        **Why Incorrect:** The phrase "I wanted to share that" implies this is NEW information being revealed for the first time, which is misleading since this was discussed in the topic log.
        
        **Key Insight:** The correct output uses present tense ("is currently 80% complete") and connects to the current UX discussion ("We're ready to incorporate UX improvements"), framing it as a relevant status update rather than a revelation.
        
        ### Example of what NOT to do (Incorrect vs. Correct)
        **COMMAND PACKET:**
        - **Core Message to Convey:** "Raise user speed complaints to reframe the discussion."
        **MY USER'S COMMUNICATION STYLE:**
        - (Style details)
        **IMMEDIATE CONVERSATIONAL CONTEXT:**
        Manager: ...Product Lead, any concerns from your end?

        **INCORRECT UTTERANCE (Violates Critical Rule - reveals internal mechanics):**
        ```json
        {{
        "My briefing notes say that the top user complaint is speed, so we should focus on that."
        }}
        ```
        **CORRECT UTTERANCE (Follows Critical Rule - speaks as the user):**
        ```json
        {{
            "My main concern is ensuring our engineering effort is focused on the biggest user problem. Our latest user data shows the top complaint about payments is the checkout flow's speed. Before we commit, can we clarify how that work would address improving that speed?"
        }}
        ```
        
        ### Example of what NOT to do (Forced Catchphrase)
        **COMMAND PACKET:**
        - **Core Message to Convey:** "Agree with the timeline and commit to delivering the API by Friday."
        **MY USER'S COMMUNICATION STYLE:**
        - **Common Fillers:** ["Let's take a step back", "At the end of the day"]
        **IMMEDIATE CONVERSATIONAL CONTEXT:**
        PM: Can we commit to having the API ready by Friday?
        
        **INCORRECT UTTERANCE (Forced catchphrase that doesn't fit):**
        ```json
        {{
            "utterance": "Let's take a step back—yes, we can commit to delivering the API by Friday."
        }}
        ```
        **Why Incorrect:** "Let's take a step back" implies we're pausing to reconsider or reframe, but the response is simply agreeing. The catchphrase disrupts the natural flow and sounds forced.
        
        **CORRECT UTTERANCE (natural, no forced catchphrase):**
        ```json
        {{
            "utterance": "Yes, we can commit to delivering the API by Friday."
        }}
        ```
        **Key Insight:** It's better to omit a catchphrase entirely than to shoe-horn it where it doesn't naturally fit. Naturalness always takes priority.
        ---
        ## YOUR TASK:
        **1. MY COMMAND FROM THE CONTROLLER:**
        - **Core Message to Convey:** {semantic_intent}
        - **Specific Information to Use (if any):** {knowledge_to_convey}
        - **Key Concepts to Frame (if any):** {concepts_to_frame}

        **2. MY USER'S COMMUNICATION STYLE (use as guidance, not rigid rules):**
        - **Formality:** {formality}
        - **Verbosity:** {verbosity}
        - **Typical Tone:** {typical_tone}
        - **Jargon Usage:** {jargon_usage}
        - **Common Fillers (OPTIONAL - use sparingly, only if they naturally fit):** {common_fillers}

        **3. IMMEDIATE CONVERSATIONAL CONTEXT:**
        {snapshot_string}

        Now, generate the `GeneratorOutput` JSON using the information provided for your task.
        """

    def _normalize_participant_code(self, participant_id: Optional[str], profile: Optional[Any]) -> str:
        """Return a stable, uppercase code for the participant (fallback to profile role)."""
        base = participant_id or ""
        if not base and profile and getattr(profile, "inferred_role_and_goals", None):
            base = getattr(profile.inferred_role_and_goals, "primary_role", "") or ""
        if not base and profile and getattr(profile, "core_info", None):
            base = getattr(profile.core_info, "primary_role", "") or ""
        base = base or "AGENT"
        code = re.sub(r"[^A-Za-z0-9]", "", base.upper())
        return code or "AGENT"

    def _assign_candidate_id(self, participant_code: str, text: str) -> str:
        """Assign or reuse a deterministic candidate id for the given text."""
        normalized_text = re.sub(r"\s+", " ", (text or "").strip()).lower()
        key = (participant_code, normalized_text)
        if key in self._candidate_id_cache:
            return self._candidate_id_cache[key]
        self._candidate_id_counter[participant_code] += 1
        candidate_id = f"{participant_code}_pt_{self._candidate_id_counter[participant_code]:03d}"
        if normalized_text:
            self._candidate_id_cache[key] = candidate_id
        return candidate_id
    # ------------------------------------------------------------------
    # Helper Orchestration (Phase 1 scaffolding)
    # ------------------------------------------------------------------

    def _log_helper_io(self, tool_name: str, payload: Optional[Dict[str, Any]], response: Optional[Dict[str, Any]]) -> None:
        """Lightweight logging for helper inputs/outputs."""
        try:
            payload_preview = json.dumps(payload, ensure_ascii=False)[:700] if payload else "None"
        except Exception:
            payload_preview = "<unserializable payload>"
        try:
            response_preview = json.dumps(response, ensure_ascii=False)[:700] if response else "None"
        except Exception:
            response_preview = "<unserializable response>"
        # print(f"  [UM] Helper[{tool_name}] payload={payload_preview}")
        # print(f"  [UM] Helper[{tool_name}] response={response_preview}")

        if self.artifact_tracker:
            try:
                self.artifact_tracker.capture_helper_io(tool_name, payload, response)
            except Exception:
                pass

    def _invoke_helper_tool(self, tool_name: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Invoke helper tool via the LLM with structured instructions."""
        if not payload:
            self._log_helper_io(tool_name, {}, None)
            return None

        helper_name = payload.get("helper_name", tool_name)
        instructions = payload.get("instructions", "")
        output_schema = payload.get("output_schema", {})
        context = payload.get("context", {})

        def _dump(data: Any) -> str:
            try:
                return json.dumps(data, ensure_ascii=False, indent=2)
            except Exception:
                return "null"

        user_prompt = self.helper_user_prompt_template.format(
            helper_name=helper_name,
            instructions=instructions,
            output_schema=_dump(output_schema),
            context=_dump(context)
        )

        # Select appropriate system prompt based on helper type
        system_prompts = {
            "candidate_curator": self.curator_system_prompt,
            "coverage_evaluator": self.coverage_system_prompt,
            "speaker_scorer": self.speaker_system_prompt
        }
        system_prompt = system_prompts.get(tool_name, self.helper_system_prompt)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            raw_response = self.client.call_with_retry(
                model=self.model_name,
                messages=messages,
                max_tokens=10000,  # Increased for GPT-5-mini reasoning overhead (was 8000)
                temperature=0.1,
                json_required=True,
                telemetry_label=f"controller_helper.{tool_name}",
                telemetry_phase="floor",
            )
        except Exception as err:
            print(f"  [UM] Helper[{tool_name}] call failed: {err}")
            self._log_helper_io(tool_name, payload, None)
            return None

        parsed: Optional[Dict[str, Any]] = None
        try:
            parsed_obj = json.loads(raw_response) if isinstance(raw_response, str) else raw_response
            if isinstance(parsed_obj, dict):
                parsed = parsed_obj
            else:
                print(f"  [UM] Helper[{tool_name}] returned non-dict payload")
        except Exception as err:
            print(f"  [UM] Helper[{tool_name}] JSON parse error: {err}")

        sanitized = self._sanitize_helper_output(tool_name, parsed)
        self._log_helper_io(tool_name, payload, sanitized)
        return sanitized

    def _sanitize_helper_output(self, tool_name: str, raw: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if raw is None:
            return None
        if tool_name == "candidate_curator":
            return self._sanitize_curator_output(raw)
        if tool_name == "coverage_evaluator":
            return self._sanitize_coverage_output(raw)
        if tool_name == "speaker_scorer":
            return self._sanitize_speaker_output(raw)
        return raw if isinstance(raw, dict) else None

    def _sanitize_curator_output(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        ranked = raw.get("ranked_candidates")
        if not isinstance(ranked, list):
            ranked = []
        cleaned = []
        for item in ranked:
            if not isinstance(item, dict):
                continue
            candidate_id = item.get("candidate_id")
            text = item.get("text")
            if not candidate_id or not text:
                continue
            cleaned.append({
                "candidate_id": str(candidate_id),
                "text": str(text),
                "source": str(item.get("source", "Unknown")),
                "suitability": str(item.get("suitability", "Unknown")),
                "stance_function": self._normalize_stance_function(item.get("stance_function"))
                or self._infer_stance_function(str(text)),
                "relevance_score": item.get("relevance_score"),
                "novelty_score": item.get("novelty_score"),
                "intent_alignment_score": item.get("intent_alignment_score"),
                "intent_alignment_factor": item.get("intent_alignment_factor"),
                "composite_score": item.get("composite_score"),
                "suggested_usage": str(item.get("suggested_usage", "")),
                "helper_notes": item.get("helper_notes") or item.get("notes"),
                "recommended_treatment": item.get("recommended_treatment")
            })
        return {
            "ranked_candidates": cleaned,
            "commentary": raw.get("commentary")
        }

    def _sanitize_coverage_output(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        annotated = []
        for item in raw.get("annotated_candidates", []) or []:
            if not isinstance(item, dict):
                continue
            cid = item.get("candidate_id")
            if not cid:
                continue
            
            # Handle new speaker-aware schema: extract from coverage_by_delegate
            coverage_by_delegate = item.get("coverage_by_delegate", {})
            coverage_by_others = item.get("coverage_by_others", {})
            
            # Build annotated output with new structure only (legacy fields removed)
            annotated.append({
                "candidate_id": str(cid),
                # New speaker-aware fields (source of truth)
                "coverage_by_delegate": coverage_by_delegate if coverage_by_delegate else None,
                "coverage_by_others": coverage_by_others if coverage_by_others else None,
                "coverage_completeness": item.get("coverage_completeness"),
                "covered_aspects": list(item.get("covered_aspects", [])),
                "uncovered_aspects": list(item.get("uncovered_aspects", [])),
                "residual_contribution_value": item.get("residual_contribution_value"),
                "contribution_angle": item.get("contribution_angle"),
                "framing_suggestion": item.get("framing_suggestion"),
                # Provenance for ExpandOnOthers
                "expand_on_reference": item.get("expand_on_reference"),  # {speaker: str, utterance_id: str}
                # Treatment and framing
                "recommended_treatment": str(item.get("recommended_treatment", "")),
                "coverage_explanation": item.get("coverage_explanation"),
                "requires_historical_framing": bool(item.get("requires_historical_framing", False)),
                "suggested_framing_approaches": list(item.get("suggested_framing_approaches", [])),
            })
        return {
            "annotated_candidates": annotated,
            "confidence": raw.get("confidence"),
            "global_notes": raw.get("global_notes")
        }

    def _sanitize_speaker_output(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Handle new speaker_ranking format
        speaker_ranking = raw.get("speaker_ranking")
        if speaker_ranking:
            # New format: speaker_ranking is a list
            sanitized_ranking = []
            for entry in speaker_ranking:
                if not isinstance(entry, dict):
                    continue
                speaker = entry.get("speaker")
                if not speaker:
                    continue
                sanitized_ranking.append({
                    "speaker": str(speaker),
                    "score": entry.get("score"),
                    "match_level": entry.get("match_level"),
                    "topic_engagement_role": entry.get("topic_engagement_role"),
                    "is_delegate": entry.get("is_delegate", False)  # NEW: marks delegate entry
                })
            
            # Extract delegate_viability with new fields
            raw_viability = raw.get("delegate_viability") or {}
            delegate_viability = {
                "has_viable_candidate": raw_viability.get("has_viable_candidate"),
                "match_level": raw_viability.get("match_level"),
                "best_viable_fit": raw_viability.get("best_viable_fit"),
                "viable_candidate_count": raw_viability.get("viable_candidate_count"),
            }
            
            return {
                "speaker_ranking": sanitized_ranking,
                "delegate_rank_position": raw.get("delegate_rank_position"),  # NEW: delegate's position
                "delegate_viability": delegate_viability,
                "overall_assessment": raw.get("overall_assessment"),
                "confidence": raw.get("confidence"),
                "rationale": raw.get("rationale"),
                "speak_or_silent_hint": raw.get("speak_or_silent_hint"),
                "notes": raw.get("notes")
            }
        
        # Fallback: Legacy format (recommended_speaker)
        recommended = raw.get("recommended_speaker")
        if not recommended:
            return None
        alt_rank = []
        for entry in raw.get("alternate_ranking", []) or []:
            if not isinstance(entry, dict):
                continue
            pid = entry.get("participant_id")
            if not pid:
                continue
            alt_rank.append({
                "participant_id": str(pid),
                "score": entry.get("score")
            })
        return {
            "recommended_speaker": str(recommended),
            "confidence": raw.get("confidence"),
            "rationale": raw.get("rationale"),
            "alternate_ranking": alt_rank,
            "speak_or_silent_hint": raw.get("speak_or_silent_hint"),
            "notes": raw.get("notes")
        }

    ## EIM View for Controller: can be used later to add/curate more context from the full EIM if needed
    def _build_controller_eim_view(self, working_memory: Any) -> Dict[str, Any]:
        """
        Build a compact EIM view for the Controller.
        
        Drops heavy fields already processed by helpers:
        - topic_summary_log (Coverage Evaluator handles this)
        - full participant_models (Speaker Scorer handles this)
        
        Keeps only what Controller rubric actually references:
        - Step 1 (Identity): identity_aliases, attendees list
        - Step 2-3 (Topic/Floor): snapshot_analysis fields
        """
        snapshot = working_memory.snapshot_analysis
        
        # Cap key_points to 8 items and slim to essential fields
        key_points_slim = []
        for kp in (snapshot.key_points_made_in_snapshot or [])[:10]:
            key_points_slim.append({
                "speaker": kp.speaker,
                "utterance_id": kp.utterance_id,
                "point": kp.point
            })
        
        # Build micro_topic_analysis if present
        micro_topic = None
        if snapshot.micro_topic_analysis:
            mta = snapshot.micro_topic_analysis
            micro_topic = {
                "broader_context": mta.broader_context,
                "immediate_focus": mta.immediate_focus,
                "topic_continuity": mta.topic_continuity,
            }
            # Include last_utterance_verbatim if present
            if mta.last_utterance_verbatim:
                luv = mta.last_utterance_verbatim
                micro_topic["last_utterance_verbatim"] = {
                    "speaker": luv.speaker,
                    "content": luv.content,
                    "intent_type": luv.intent_type,
                    "target_addressed": luv.target_addressed
                }
        
        return {
            "last_processed_utterance_id": working_memory.last_processed_utterance_id,
            "attendees": list(working_memory.participant_models.keys()),
            "identity_aliases": working_memory.identity_aliases or {},
            "snapshot_analysis": {
                "micro_topic_analysis": micro_topic,
                "conversational_floor_status": snapshot.conversational_floor_status,
                "floor_rationale": snapshot.floor_rationale,
                "conversational_state": snapshot.conversational_state,
                "key_points_made_in_snapshot": key_points_slim
            }
        }

    def _build_helper_summary(
        self,
        curator_output: Optional[Dict[str, Any]],
        coverage_output: Optional[Dict[str, Any]],
        speaker_output: Optional[Dict[str, Any]],
        helper_digest_data: Optional[List[Any]],
        banned_candidate_ids: Set[str]
    ) -> Dict[str, Any]:
        """
        Merge all helper outputs into a single unified structure for the Controller.
        
        Returns a helper_summary dict with:
        - speaker_guidance: Who should speak + confidence + hint
        - candidates: Ranked list with coverage info per candidate
        - banned_candidate_ids: IDs that MUST NOT be selected (hard bans)
        - meta: Summary stats including viable_count and delegate_has_viable_content
        """
        # Build speaker_guidance (supports new speaker_ranking format with fallback)
        speaker_guidance = None
        if speaker_output:
            if speaker_output.get("speaker_ranking"):
                # New format with speaker_ranking list
                speaker_guidance = {
                    "speaker_ranking": speaker_output.get("speaker_ranking"),
                    "delegate_rank_position": speaker_output.get("delegate_rank_position"),  # NEW
                    "delegate_viability": speaker_output.get("delegate_viability"),
                    "overall_assessment": speaker_output.get("overall_assessment"),
                    "confidence": speaker_output.get("confidence"),
                    "hint": speaker_output.get("speak_or_silent_hint"),
                    "rationale": speaker_output.get("rationale")
                }
            else:
                # Fallback: Legacy format
                speaker_guidance = {
                    "recommended_speaker": speaker_output.get("recommended_speaker"),
                    "confidence": speaker_output.get("confidence"),
                    "hint": speaker_output.get("speak_or_silent_hint"),
                    "rationale": speaker_output.get("rationale")
                }
        
        # Build curator score lookup (candidate_id -> composite_score, helper_notes)
        curator_scores: Dict[str, float] = {}
        curator_notes: Dict[str, str] = {}
        if curator_output:
            for item in curator_output.get("ranked_candidates") or []:
                cid = item.get("candidate_id")
                if cid:
                    curator_scores[cid] = item.get("composite_score")
                    curator_notes[cid] = item.get("helper_notes")
        
        # Build coverage lookup (candidate_id -> annotation)
        coverage_by_id: Dict[str, Dict[str, Any]] = {}
        if coverage_output and coverage_output.get("annotated_candidates"):
            for ann in coverage_output["annotated_candidates"]:
                cid = ann.get("candidate_id")
                if cid:
                    coverage_by_id[cid] = ann
        
        # Build candidates list FROM helper_digest_data (not curator_output)
        candidates = []
        viable_count = 0
        fresh_count = 0
        skip_count = 0
        
        digest_items = helper_digest_data or []
        for item in digest_items:
            # Handle both dict and pydantic model
            if isinstance(item, dict):
                cid = item.get("candidate_id")
                text = item.get("text")
            else:
                cid = getattr(item, "candidate_id", None)
                text = getattr(item, "text", None)
            
            if not cid or not text:
                continue
            
            # Helper for extracting fields from dict or pydantic model
            def _get(obj, key, default=None):
                if isinstance(obj, dict):
                    return obj.get(key, default)
                return getattr(obj, key, default)
            
            # Get coverage data
            cov = coverage_by_id.get(cid, {})
            treatment = cov.get("recommended_treatment") or _get(item, "recommended_treatment") or "Fresh"
            
            # Determine ban_reason and deprioritize_reason
            # Use defensive None handling: .get() can return None, not just missing key
            ban_reason = None
            deprioritize_reason = None
            
            if cid in banned_candidate_ids:
                # Lookup specific reason from coverage
                cov_by_delegate = cov.get("coverage_by_delegate") or {}
                if cov_by_delegate.get("in_current_episode"):
                    ban_reason = "already_said_this_episode"
                elif cov_by_delegate.get("in_snapshot"):
                    ban_reason = "delegate_covered_in_snapshot"
                else:
                    ban_reason = "already_said_this_episode"  # Default for banned
            
            cov_by_others = cov.get("coverage_by_others") or {}
            if cov_by_others.get("in_snapshot"):
                deprioritize_reason = "others_covered_in_snapshot"
            
            # Merge framing suggestions (coverage may have list, we take first)
            framing_suggestion = cov.get("framing_suggestion")
            if not framing_suggestion:
                sfa = cov.get("suggested_framing_approaches") or []
                framing_suggestion = sfa[0] if sfa else None
            
            # Build candidate entry
            candidate = {
                "candidate_id": cid,
                "text": text,
                "core_text": _get(item, "core_text") or text,
                "composite_score": curator_scores.get(cid) or _get(item, "composite_score"),
                "suitability": _get(item, "suitability"),
                "stance_function": (
                    self._normalize_stance_function(_get(item, "stance_function"))
                    or self._infer_stance_function(_get(item, "core_text") or text)
                ),
                "source": _get(item, "source"),
                "inventory_tier": _get(item, "inventory_tier", "primary"),
                "helper_notes": curator_notes.get(cid) or _get(item, "helper_notes"),
                "coverage": {
                    "recommended_treatment": treatment,
                    "coverage_completeness": cov.get("coverage_completeness", "none"),
                    "explanation": cov.get("coverage_explanation"),
                    "requires_historical_framing": cov.get("requires_historical_framing", False),
                    "framing_suggestion": framing_suggestion,
                    "contribution_angle": cov.get("contribution_angle"),
                    "residual_contribution_value": cov.get("residual_contribution_value"),
                    "uncovered_aspects": cov.get("uncovered_aspects") or [],
                    "expand_on_reference": cov.get("expand_on_reference")  # {speaker, utterance_id} for ExpandOnOthers
                },
                "flags": {
                    "ban_reason": ban_reason,
                    "deprioritize_reason": deprioritize_reason
                },
                "anchor_phrases": _get(item, "anchor_phrases", []) or [],
                "framing": _get(item, "framing"),
                "residual_contribution_value": cov.get("residual_contribution_value"),
            }
            candidate.update(self._candidate_viability_details(candidate))
            candidates.append(candidate)
            
            # Count stats
            if treatment in ("Skip", "Acknowledge"):
                skip_count += 1
            elif self._candidate_is_viable(candidate):
                viable_count += 1
                if treatment == "Fresh":
                    fresh_count += 1
        
        return {
            "speaker_guidance": speaker_guidance,
            "candidates": candidates,
            "banned_candidate_ids": list(banned_candidate_ids),
            "meta": {
                "total_candidates": len(candidates),
                "viable_count": viable_count,
                "fresh_count": fresh_count,
                "skip_count": skip_count,
                "delegate_has_viable_content": viable_count > 0
            }
        }

    def _is_meaningful_utterance(self, last_utt: Dict[str, Any]) -> bool:
        """
        Heuristic check for whether the last utterance is substantive enough
        to trigger a hard yield-after-self.

        Very short acknowledgments/backchannels (\"Yeah\", \"Mm-hmm\", etc.)
        should not force yield; in those cases we defer to helpers/controller.
        """
        raw = last_utt.get("utterance_content") or last_utt.get("content") or ""
        text = str(raw).strip()
        if not text:
            return False

        # Strip simple markup tokens like {gap}, {vocalsound}, {disfmarker}
        try:
            import re as _re
            text = _re.sub(r"\{[^}]+\}", " ", text)
            text = _re.sub(r"\s+", " ", text).strip()
        except Exception:
            text = text.strip()

        if not text:
            return False

        lowered = text.lower().strip(" .,!?:;")
        words = lowered.split()
        if not words:
            return False

        # Common banal/backchannel tokens
        fillers = {
            "yeah",
            "yes",
            "yep",
            "ok",
            "okay",
            "mm-hmm",
            "mmhm",
            "mm",
            "mmm",
            "uh-huh",
            "uhuh",
            "hmm",
            "uh",
            "um",
            "er",
            "erm",
            "eh",
            "ah",
            "oh",
        }

        # Treat very short backchannels as non-meaningful
        if len(words) <= 3 and words[0] in fillers and len(text) <= 15:
            return False

        # Otherwise consider it meaningful enough for yield-after-self
        return True

    def _should_yield_after_self(
        self,
        smm: SharedMemoryModule,
        participant_id: Optional[str],
        snapshot_utterances: List[Dict[str, Any]],
    ) -> bool:
        """
        Determine whether the yield-after-self rule should force silence.

        Returns True when:
        - The last speaker in the snapshot is the represented user, AND
        - There is no explicit cue back to them.
        """
        if not participant_id or not snapshot_utterances:
            return False

        # Last speaker from raw snapshot
        last_utt = snapshot_utterances[-1]
        last_speaker = last_utt.get("speaker_id")
        is_self = (last_speaker == participant_id)
        if not is_self:
            return False

        wm = smm.get_working_memory()
        explicit_cue = False

        try:
            if wm and getattr(wm, "snapshot_analysis", None):
                sa = wm.snapshot_analysis
                mta = getattr(sa, "micro_topic_analysis", None)
                target_addressed = None
                if mta and getattr(mta, "last_utterance_verbatim", None):
                    luv = mta.last_utterance_verbatim
                    target_addressed = getattr(luv, "target_addressed", None)
                identity_aliases = getattr(wm, "identity_aliases", {}) or {}
                candidate_ids = {participant_id}
                for alias, mapped in identity_aliases.items():
                    if mapped == participant_id:
                        candidate_ids.add(alias)
                if target_addressed and target_addressed in candidate_ids:
                    explicit_cue = True
        except Exception:
            explicit_cue = False

        if explicit_cue:
            return False

        return True

    # ========== Coverage Tracking Methods (Phase 3 Enhancement) ==========
    
    def _serialize_coverage_flags(self) -> Dict[str, Dict[str, Any]]:
        """
        Serialize candidate_coverage_flags for inclusion in Coverage Evaluator payload.
        Returns a dict keyed by candidate_id with coverage status values.
        """
        serialized: Dict[str, Dict[str, Any]] = {}
        for cid, flag_obj in self.candidate_coverage_flags.items():
            if isinstance(flag_obj, CandidateCoverageStatus):
                serialized[cid] = {
                    "covered_in_current_episode": flag_obj.covered_in_current_episode,
                    "covered_in_previous_episodes": flag_obj.covered_in_previous_episodes,
                    "last_covered_utterance_id": flag_obj.last_covered_utterance_id,
                    "last_covered_episode_id": flag_obj.last_covered_episode_id,
                    "coverage_count": flag_obj.coverage_count
                }
            elif isinstance(flag_obj, dict):
                # Handle dict representation directly
                serialized[cid] = {
                    "covered_in_current_episode": flag_obj.get("covered_in_current_episode", False),
                    "covered_in_previous_episodes": flag_obj.get("covered_in_previous_episodes", False),
                    "last_covered_utterance_id": flag_obj.get("last_covered_utterance_id"),
                    "last_covered_episode_id": flag_obj.get("last_covered_episode_id"),
                    "coverage_count": flag_obj.get("coverage_count", 0)
                }
        return serialized
    
    def _update_candidate_coverage(self, candidate_id: str, candidate_text: str, utterance_id: str, source: Optional[str] = None):
        """
        Update coverage flags when delegate speaks using a specific candidate.
        Called after controller_phase when an active_candidate_id is identified.
        """
        if not candidate_id:
            return
        
        if candidate_id not in self.candidate_coverage_flags:
            # Initialize new coverage entry
            self.candidate_coverage_flags[candidate_id] = CandidateCoverageStatus(
                candidate_id=candidate_id,
                text=candidate_text,
                source=source,
                covered_in_current_episode=True,
                last_covered_utterance_id=utterance_id,
                coverage_count=1,
                first_covered_at=utterance_id
            )
        else:
            # Update existing coverage entry
            flag = self.candidate_coverage_flags[candidate_id]
            if isinstance(flag, CandidateCoverageStatus):
                flag.covered_in_current_episode = True
                flag.last_covered_utterance_id = utterance_id
                flag.coverage_count += 1
                if not flag.first_covered_at:
                    flag.first_covered_at = utterance_id
            elif isinstance(flag, dict):
                # Handle dict representation
                flag["covered_in_current_episode"] = True
                flag["last_covered_utterance_id"] = utterance_id
                flag["coverage_count"] = flag.get("coverage_count", 0) + 1

    def _migrate_episode_coverage(self, episode_id: str):
        """
        Migrate current episode coverage flags to previous episodes at episode boundaries.
        Call this when starting a new episode or ending the current one.
        """
        for cid, flag in self.candidate_coverage_flags.items():
            if isinstance(flag, CandidateCoverageStatus):
                if flag.covered_in_current_episode:
                    flag.covered_in_previous_episodes = True
                    flag.last_covered_episode_id = episode_id
                flag.covered_in_current_episode = False
            elif isinstance(flag, dict):
                if flag.get("covered_in_current_episode"):
                    flag["covered_in_previous_episodes"] = True
                    flag["last_covered_episode_id"] = episode_id
                flag["covered_in_current_episode"] = False

    def _build_candidate_coverage_status(
        self,
        coverage_output: Optional[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build a per-candidate coverage_status view for this turn by combining
        Coverage Evaluator annotations with our persistent coverage flags.

        Uses speaker-aware coverage from new schema:
        - delegate_covered_snapshot: Delegate said this in current snapshot → BANNED
        - others_covered_snapshot: Someone else (in GT) said this → DE-PRIORITIZE
        - covered_in_current_episode: Delegate said this earlier in episode → BANNED

        Returns a dict:
            {candidate_id: {
                "delegate_covered_snapshot": bool,  # Delegate covered in snapshot
                "others_covered_snapshot": bool,     # Others covered in snapshot (GT)
                "covered_in_current_episode": bool,  # Delegate covered earlier in episode
                "covered_in_previous_episodes": bool,
                "recommended_treatment": str,
                "ban_reason": Optional[str],         # Only set if DELEGATE covered (hard ban)
                "deprioritize_reason": Optional[str], # Set if OTHERS covered (soft signal)
            }}
        """
        status_map: Dict[str, Dict[str, Any]] = {}

        # Extract coverage from Coverage Evaluator output
        try:
            if coverage_output and coverage_output.get("annotated_candidates"):
                for ann in coverage_output.get("annotated_candidates") or []:
                    if not isinstance(ann, dict):
                        continue
                    cid = ann.get("candidate_id")
                    if not cid:
                        continue
                    cid_str = str(cid)
                    
                    # Extract from new speaker-aware fields
                    coverage_by_delegate = ann.get("coverage_by_delegate") or {}
                    coverage_by_others = ann.get("coverage_by_others") or {}
                    
                    # Delegate coverage flags
                    delegate_covered_snapshot = bool(coverage_by_delegate.get("in_snapshot", False))
                    delegate_covered_topic_log = bool(coverage_by_delegate.get("in_topic_log", False))
                    delegate_covered_episode = bool(coverage_by_delegate.get("in_current_episode", False))
                    delegate_covered_prev = bool(coverage_by_delegate.get("in_previous_episodes", False))
                    
                    # Others/GT coverage flags
                    others_covered_snapshot = bool(coverage_by_others.get("in_snapshot", False))
                    others_covered_topic_log = bool(coverage_by_others.get("in_topic_log", False))
                    
                    status = status_map.get(cid_str, {})
                    status["delegate_covered_snapshot"] = delegate_covered_snapshot
                    status["delegate_covered_topic_log"] = delegate_covered_topic_log
                    status["others_covered_snapshot"] = others_covered_snapshot
                    status["others_covered_topic_log"] = others_covered_topic_log
                    status["recommended_treatment"] = ann.get("recommended_treatment", "")
                    status["coverage_completeness"] = ann.get("coverage_completeness")
                    status["residual_contribution_value"] = ann.get("residual_contribution_value")
                    status["contribution_angle"] = ann.get("contribution_angle")
                    status["framing_suggestion"] = ann.get("framing_suggestion")
                    status_map[cid_str] = status
        except Exception:
            # If anything goes wrong, fall back to flags-only view
            status_map = {}

        # Merge in episode-level coverage flags from our persistent tracking
        for cid, flag in self.candidate_coverage_flags.items():
            covered_current = False
            covered_prev = False
            try:
                if isinstance(flag, CandidateCoverageStatus):
                    covered_current = bool(flag.covered_in_current_episode)
                    covered_prev = bool(flag.covered_in_previous_episodes)
                elif isinstance(flag, dict):
                    covered_current = bool(flag.get("covered_in_current_episode"))
                    covered_prev = bool(flag.get("covered_in_previous_episodes"))
            except Exception:
                covered_current = False
                covered_prev = False
            entry = status_map.get(cid, {})
            entry.setdefault("delegate_covered_snapshot", False)
            entry.setdefault("others_covered_snapshot", False)
            entry["covered_in_current_episode"] = covered_current
            entry["covered_in_previous_episodes"] = covered_prev
            status_map[cid] = entry

        # Finalize ban_reason and deprioritize_reason for each candidate
        for cid, entry in status_map.items():
            covered_current_episode = bool(entry.get("covered_in_current_episode"))
            delegate_snapshot = bool(entry.get("delegate_covered_snapshot"))
            others_snapshot = bool(entry.get("others_covered_snapshot"))
            
            # BAN LOGIC: Only delegate coverage triggers hard ban
            if covered_current_episode:
                ban_reason = "already_said_this_episode"
            elif delegate_snapshot:
                ban_reason = "delegate_covered_in_snapshot"
            else:
                ban_reason = None
            
            # DE-PRIORITIZE LOGIC: Others/GT coverage is soft signal
            if others_snapshot and not delegate_snapshot:
                deprioritize_reason = "others_covered_in_snapshot"
            else:
                deprioritize_reason = None
            
            entry["ban_reason"] = ban_reason
            entry["deprioritize_reason"] = deprioritize_reason

        return status_map

    def _update_episode_boundary(self, episode: Optional[object]) -> None:
        """
        Handle episode-boundary transitions for candidate coverage flags.

        This ensures that `covered_in_current_episode` is scoped to a single
        CAPA episode while preserving cross-episode coverage provenance.
        """
        if episode is None:
            return
        episode_id = getattr(episode, "test_case_id", None)
        if not episode_id:
            return

        # First episode seen: just record it, no migration needed
        if self._current_episode_id is None:
            self._current_episode_id = episode_id
            return

        # New episode: migrate coverage flags from the previous episode
        if episode_id != self._current_episode_id:
            self._migrate_episode_coverage(self._current_episode_id)
            self._current_episode_id = episode_id
            # Reset episode-level tracking for recalibration guidance
            # Note: smm access needed; will be done in run_turn where smm is available

    def _compute_banned_candidates(self, coverage_output: Optional[Dict[str, Any]]) -> Tuple[Set[str], Dict[str, str]]:
        """
        Derive a set of candidate_ids that should be treated as hard-banned
        for this step due to DELEGATE coverage (not others).

        Criteria (only DELEGATE coverage triggers ban):
        - Coverage Evaluator: coverage_by_delegate.in_snapshot = true
        - Coverage Evaluator: coverage_by_delegate.in_current_episode = true
        - Our own coverage_flags: covered_in_current_episode = true
        
        Note: Others' coverage (coverage_by_others.in_snapshot) does NOT ban—
        those candidates can still be used with ExpandOnOthers treatment.
        
        Returns:
            Tuple of (banned_ids: Set[str], ban_reasons: Dict[str, str])
            where ban_reasons maps candidate_id to the reason for ban.
        """
        banned: Set[str] = set()
        ban_reasons: Dict[str, str] = {}

        try:
            if coverage_output:
                for ann in coverage_output.get("annotated_candidates") or []:
                    if not isinstance(ann, dict):
                        continue
                    cid = ann.get("candidate_id")
                    if not cid:
                        continue
                    
                    # Check speaker-aware fields (required output structure)
                    coverage_by_delegate = ann.get("coverage_by_delegate") or {}
                    if coverage_by_delegate.get("in_snapshot"):
                        # Check if treatment allows delegate to still add value
                        treatment = ann.get("recommended_treatment", "")
                        if treatment in ("ExpandOnOthers", "Partial"):
                            # Delegate can still contribute via expansion - don't hard ban
                            continue
                        banned.add(str(cid))
                        ban_reasons[str(cid)] = "delegate_covered_in_snapshot"
                    elif coverage_by_delegate.get("in_current_episode"):
                        banned.add(str(cid))
                        ban_reasons[str(cid)] = "delegate_covered_in_current_episode"
        except Exception:
            pass

        # Within-episode coverage from our flags (source of truth)
        for cid, flag in self.candidate_coverage_flags.items():
            try:
                if isinstance(flag, CandidateCoverageStatus):
                    if flag.covered_in_current_episode:
                        banned.add(cid)
                        if cid not in ban_reasons:
                            ban_reasons[cid] = f"covered_in_current_episode_flag (last_utt: {flag.last_covered_utterance_id})"
                elif isinstance(flag, dict):
                    if flag.get("covered_in_current_episode"):
                        banned.add(cid)
                        if cid not in ban_reasons:
                            last_utt = flag.get("last_covered_utterance_id", "unknown")
                            ban_reasons[cid] = f"covered_in_current_episode_flag (last_utt: {last_utt})"
            except Exception:
                continue

        return banned, ban_reasons

    def _prepare_candidate_inventory(
        self,
        participant_id: Optional[str],
        briefing: Optional[List[str]],
        profile
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Build a stable inventory of candidate points with persistent IDs."""
        participant_code = self._normalize_participant_code(participant_id, profile)
        inventory: List[Dict[str, Any]] = []
        seen_ids: Set[str] = set()

        def _parse_candidate_point(text: str) -> Tuple[str, Optional[str], Optional[str]]:
            """Parse a JIT-style bullet into core_text + optional topic/context_hint.

            Expected general shape:
              "<forward-looking action...>. (topic: <topic>[; context: <context_hint>])"

            Returns (core_text, topic, context_hint). If no well-formed suffix is
            present, topic/context_hint will be None and core_text is the original text.
            """
            if not isinstance(text, str):
                return str(text or ""), None, None
            s = text.strip()
            if not s:
                return "", None, None
            topic = None
            context_hint = None
            core_text = s
            try:
                # Look for a trailing "(topic: ...)" block
                m = re.search(
                    r"\(topic:\s*([^;()]+?)(?:;\s*context:\s*([^()]+?))?\)\s*$",
                    s,
                    flags=re.IGNORECASE,
                )
                if m:
                    topic = (m.group(1) or "").strip() or None
                    if m.group(2):
                        context_hint = (m.group(2) or "").strip() or None
                    core_text = s[: m.start()].rstrip()
            except Exception:
                core_text = s
                topic = None
                context_hint = None
            return core_text or s, topic, context_hint

        def add_item(
            text: Optional[str],
            source: str,
            origin_index: int,
            context_note: Optional[str] = None,
            evidence: Optional[Dict[str, Any]] = None,
        ) -> None:
            normalized = (text or "").strip()
            if not normalized:
                return
            candidate_id = self._assign_candidate_id(participant_code, normalized)
            if candidate_id in seen_ids:
                return
            entry: Dict[str, Any] = {
                "candidate_id": candidate_id,
                "source": source,
                "text": normalized,
                "origin_index": origin_index,
            }
            # Parse standardized JIT bullets into core_text + topic/context_hint so
            # downstream helpers can treat the bracket metadata as soft signals.
            core_text, topic_tag, context_hint = _parse_candidate_point(normalized)
            entry["core_text"] = core_text
            entry["stance_function"] = self._infer_stance_function(core_text)
            if topic_tag:
                entry["topic"] = topic_tag
            if context_hint:
                entry["context_hint"] = context_hint
            # Tag inventory tier so downstream helpers can distinguish primary
            # (JIT testing-segment idea units) from supplementary knowledge.
            if source == "JIT":
                entry["inventory_tier"] = "primary"
            else:
                entry["inventory_tier"] = "supplementary"
            if context_note:
                entry["context"] = context_note.strip()
            evidence = evidence or {}
            for field_name in ("anchor_phrases",):
                values = evidence.get(field_name)
                if isinstance(values, list):
                    entry[field_name] = [str(value) for value in values if str(value or "").strip()]
            if evidence.get("framing"):
                entry["framing"] = str(evidence.get("framing"))
            inventory.append(entry)
            seen_ids.add(candidate_id)

        if briefing:
            for idx, point in enumerate(briefing):
                evidence: Dict[str, Any] = {}
                if isinstance(point, dict):
                    text_value = str(
                        point.get("text")
                        or point.get("point")
                        or point.get("Information")
                        or point.get("information")
                        or ""
                    )
                    evidence = point
                else:
                    text_value = point if isinstance(point, str) else str(point or "")
                add_item(text_value, "JIT", idx, evidence=evidence)

        knowledge_points = []
        if profile and getattr(profile, "core_info", None):
            knowledge_points = getattr(profile.core_info, "specific_knowledge_points", []) or []
        for idx, point in enumerate(knowledge_points):
            context_note = None
            if isinstance(point, dict):
                info = (point.get("information") or point.get("Information") or point.get("info") or point.get("Info") or point.get("text") or point.get("Text"))
                context_note = point.get("context") or point.get("Context")
            else:
                info = getattr(point, "information", None) or getattr(point, "info", None) or str(point)
                context_note = getattr(point, "context", None)
            add_item(info, "Profile", idx, context_note)

        return participant_code, inventory

    def _build_curator_payload(
        self,
        smm: SharedMemoryModule,
        snapshot_utterances: List[Dict],
        participant_id: Optional[str]
    ) -> Dict[str, Any]:
        profile = smm.get_user_profile()
        working_memory = smm.get_working_memory()
        briefing = smm.get_jit_briefing()
        episodic = smm.get_episodic_memory() or []

        candidate_prefix, candidate_inventory = self._prepare_candidate_inventory(participant_id, briefing, profile)

        # Partition candidate inventory into available vs already used this episode.
        # Episode-covered candidates are removed from the decision inventory but can
        # still be surfaced separately for context if needed.
        available_inventory: List[Dict[str, Any]] = []
        covered_readonly: List[Dict[str, Any]] = []
        for entry in candidate_inventory:
            cid = entry.get("candidate_id")
            covered_in_episode = False
            if cid:
                flag = self.candidate_coverage_flags.get(cid)
                try:
                    if isinstance(flag, CandidateCoverageStatus):
                        covered_in_episode = bool(flag.covered_in_current_episode)
                    elif isinstance(flag, dict):
                        covered_in_episode = bool(flag.get("covered_in_current_episode"))
                except Exception:
                    covered_in_episode = False
            if covered_in_episode:
                covered_readonly.append(entry)
            else:
                available_inventory.append(entry)

        candidate_inventory = available_inventory

        participant_code = candidate_prefix
        self._latest_participant_code = participant_code

        snapshot_analysis = getattr(working_memory, "snapshot_analysis", None) if working_memory else None
        micro_topic_dict: Dict[str, Any] = {}
        last_speaker = None
        topic_continuity = None
        if snapshot_analysis and hasattr(snapshot_analysis, "micro_topic_analysis") and snapshot_analysis.micro_topic_analysis:
            mta = snapshot_analysis.micro_topic_analysis
            micro_topic_dict = {
                "broader_context": getattr(mta, "broader_context", None),
                "immediate_focus": getattr(mta, "immediate_focus", None)
            }
            topic_continuity = getattr(mta, "topic_continuity", None)
            last_ctx = getattr(mta, "last_utterance_verbatim", None)
            if last_ctx:
                last_speaker = getattr(last_ctx, "speaker", None)
                micro_topic_dict["last_utterance"] =  {
                    "speaker": getattr(last_ctx, "speaker", None),
                    "utterance_id": getattr(last_ctx, "utterance_id", None),
                    "intent_type": getattr(last_ctx, "intent_type", None),
                    "target_addressed": getattr(last_ctx, "target_addressed", None),
                    "content": getattr(last_ctx, "content", None)
                }
        floor_status = getattr(snapshot_analysis, "conversational_floor_status", None) if snapshot_analysis else None

        identity_aliases = getattr(working_memory, "identity_aliases", {}) if working_memory else {}
        explicit_cue = False
        last_utterance = micro_topic_dict.get("last_utterance") if isinstance(micro_topic_dict, dict) else None
        if last_utterance:
            target = last_utterance.get("target_addressed")
            candidate_ids = {participant_id}
            for alias, mapped in (identity_aliases or {}).items():
                if mapped == participant_id:
                    candidate_ids.add(alias)
            explicit_cue = target in candidate_ids if target else False

        topic_log_snippets: List[Dict[str, Any]] = []
        if working_memory and getattr(working_memory, "topic_summary_log", None):
            for label, entry in working_memory.topic_summary_log.items():
                if hasattr(entry, "key_points"):
                    topic_log_snippets.append({
                        "topic": label,
                        "key_points": [kp.point for kp in entry.key_points] if entry.key_points else []
                    })

        participant_models_payload: List[Dict[str, Any]] = []
        if working_memory and getattr(working_memory, "participant_models", None):
            for pid, model in working_memory.participant_models.items():
                participant_models_payload.append({
                    "participant_id": pid,
                    "role": pid,
                    "expertise_keywords": getattr(model, "inferred_expertise_keywords", []),
                    "recent_dialogue_act": getattr(model, "recent_dialogue_act", None),
                    "recent_points": [],
                    "engagement_score": getattr(model, "engagement_score", None)
                })

        episodic_payload = []
        ## POTENTIAL ISSUE 3
        for entry in episodic[-5:]:
            episodic_payload.append({
                "utterance_id": entry.utterance_id_of_action,
                "action": entry.action_taken,
                "point_made": entry.point_made_summary
            })

        payload = {
            "helper_name": "candidate_curator",
            "instructions": textwrap.dedent("""## System Context
You are part of a Meeting Delegate Agent's decision-making pipeline. The agent represents a specific meeting participant and must decide when and what to contribute to an ongoing meeting.

## Your Role: Candidate Curator
Your job is to be the "Content Advisor" - you evaluate all possible talking points (candidates) that the delegate COULD say, score them against the current meeting context, and recommend which ones are most valuable RIGHT NOW.

## Your Position in the Pipeline
Input: You receive a candidate inventory (JIT briefing + profile knowledge), meeting context (micro-topic, floor status), and conversation history.
Importantly, this inventory already EXCLUDES any candidates that the delegate has explicitly used earlier in the CURRENT episode, so you never
need to recommend a point that has already been spoken by the delegate in this episode. You may also receive a separate list of
`covered_candidates_readonly` which summarizes points already used this episode; treat those purely as history/context, not as options to recommend.
Each inventory entry includes:
- `candidate_id`: stable identifier for the point
- `text`: the full briefing point string (may include a trailing \"(topic: ...; context: ... )\" suffix)
- `core_text`: the core idea/action (same as `text` but with any trailing \"(topic: ... )\" metadata removed)
- `topic`: optional short topic tag (1-3 words) extracted from the trailing metadata when present
- `context_hint`: optional short phrase describing when the point is typically relevant, extracted from the trailing metadata when present
- `source`: e.g., \"JIT\" or \"Profile\"
- `inventory_tier`: primary/supplementary flag (see below)
- optional evidence fields: `anchor_phrases` and `framing`

Use `core_text` as the canonical semantic content when judging relevance, novelty, and intent alignment. Treat `topic` and `context_hint` as soft hints only (see **Soft signals** in Relevance Score section for detailed guidance). Anchor phrases are stronger than topic/context hints when deciding what the point really asserts.

**Inventory tiers:**
- `inventory_tier="primary"`: JIT briefing points distilled from the testing segment (episode-relevant idea units).
- `inventory_tier="supplementary"`: long-term/profile or background knowledge points.
Your default search space is the primary tier; use supplementary points mainly for reinforcement, summarisation, or as fallback when no primary candidate fits well.
Output: You produce a ranked list of candidates with detailed scores and usage recommendations.
Success: Your output enables the Controller to quickly identify the most strategically valuable content without re-analyzing all candidates.

Think of yourself as a strategic analyst who has done the research and hands the executive (Controller) a pre-analyzed report with clear recommendations.

## Your Specific Responsibilities
1. For each candidate, identify its **core claim** from `core_text` and ignore superficial phrasing.
2. Compute **relevance**, **novelty**, and **intent alignment** scores based on that core claim.
3. Calculate a composite_score for ranking.
4. Assign suitability levels (High/Medium/Low).
5. Recommend usage treatment (CorePoint/Reinforce/Acknowledge/Skip).
6. Assign `stance_function`: support/proposal for endorsement or new positive suggestions; constraint/caution for risks, limits, feasibility, tradeoffs, requirements, or objections; question/facilitation for process prompts; neutral otherwise.
7. Synthesize new points ONLY if no inventory item fits (this can usually be done based on the participant's profile: knowledge_points, expertise, goals, ...).

## Scoring Methodology

Apply this rubric when scoring each candidate:

- Always base your judgement on `core_text` (the nucleus of the point). Do **not** reward extra phrasing or generic framing in `text`.
- Treat `topic` and `context_hint` strictly as soft hints; they may slightly adjust relevance when two candidates are otherwise tied, but they must never be the sole reason to boost or lower scores.

**Core-Value Checklist** (apply per candidate before scoring):
1. What is the **core claim** in `core_text`? (strip framing, extract nucleus)
2. Does this claim add **genuinely new information/action** vs snapshot + topic_log? Or is it merely a rephrasing/generic restatement?
3. Would saying this **now** add a **concrete fact/decision/risk/question/proposal**? Or just vague commentary?

All scores must reflect this test. Do NOT reward mere rephrasing or generic remarks.

For each candidate, compute these scores:

### 1. Relevance Score (1-10)

**MULTI-AXIAL RELEVANCE (Dynamic Weighting):**

Consider these axes holistically—weight each based on context:
- **MICRO** (immediate_focus): Last 1-2 utterances
- **MESO** (broader_context): Last 4-5 substantive utterances
- **Conversational Flow**: Last 2-3 raw utterances directly

**Immediate Focus Resolution Spectrum:**
Before scoring, assess the resolution status of immediate_focus:
- URGENT_OPEN: Question/request needing response → prioritize MICRO heavily
- SOFT_OPEN: Comment inviting follow-up → balance MICRO/MESO
- NEUTRAL: Ongoing statement → balance all axes
- RESOLVED: Acknowledgment/conclusion → weight MESO and flow more

**Topic Fit Assessment:**
- `topic_fit` (1-10): How well does `core_text` align with the multi-axial scope?
  * Strong match to `immediate_focus` (MICRO level) → highest topic_fit (9-10)
  * Match to `broader_context` (MESO level) but not immediate_focus → moderate topic_fit (6-8)
  * MACRO-only matches (snapshot_major_topic without MICRO/MESO) → lower topic_fit (see formula below)
- `narrative_flow_fit` (1-10): How naturally does this follow last 2-4 utterances and floor status?
- **Note**: `topic_fit` is one component of `relevance_score`; the final score is computed via the formula below. See **MULTI-AXIAL HIERARCHY** for guidance on final relevance_score ranges based on match level.

**MICRO-TOPIC VERIFICATION NOTE**: The EIM's `snapshot_analysis.micro_topic_analysis.immediate_focus` may occasionally be stale (not reflecting the last 1-2 utterances).
Before using `immediate_focus` for topic_fit scoring:
- Cross-check against BOTH: (a) the last 1-2 entries in `snapshot_key_points` AND (b) the raw conversational window (last 2-4 utterances in context)
- If the last utterance introduces new content (question, enumeration, topic shift) NOT reflected in `immediate_focus`, prioritize the raw utterance content for topic_fit assessment
- ROBUSTNESS: If the last 1-2 utterances are banal/phatic (disfluencies, backchannels like 'okay', 'mm-hmm'), look further back to the last substantive utterance for topic context

**Formula:** `relevance_score ≈ round(0.6 * topic_fit + 0.4 * narrative_flow_fit)`

**Cap Rule:** If `narrative_flow_fit ≤ 3`, cap `relevance_score` at 6 regardless of topic_fit.

**Soft signals:** Use `topic` and `context_hint` as STARTING hints to quickly assess potential fit:
- If candidate's topic/context_hint aligns with immediate_focus → investigate MICRO match
- If they align with broader_context → investigate MESO match
- If they don't align with either → likely MACRO only (lower priority)
- ALWAYS corroborate against established EIM context (snapshot_key_points, conversation_window); don't blindly trust hints
- **core_text primacy**: If topic/context_hint seem mismatched but `core_text` clearly fits the MICRO/MESO scope, you may still recommend the candidate (note the override in helper_notes)
- **Never sole basis**: Do NOT select or reject a candidate solely because of its topic/context_hint; they are advisory and may be noisy

**Band Reference:**
- 9-10: Directly answers/advances immediate focus AND fits flow naturally (MICRO + flow match)
- 7-8: Strong topical fit with reasonable flow (MICRO partial OR strong MESO)
- 5-6: Related but not central; may risk derailment (MESO match, weak flow)
- 3-4: Tangentially relevant (MACRO/mandate only)
- 1-2: Barely relevant or context-breaking

**MULTI-AXIAL HIERARCHY (Recommended Guidance):**
Assess candidates in this priority order. This is guidance to ensure systematic scoring, not a hard cap that blocks contributions:

1. **MICRO + Immediate Focus Status** (priority 7-10):
   - Does candidate directly address immediate_focus OR last 1-2 substantive utterances?
   - Check resolution status: URGENT_OPEN (direct question/request) > SOFT_OPEN (invites follow-up) > NEUTRAL > RESOLVED
   - If YES and status is URGENT_OPEN/SOFT_OPEN: Score can be 7-10
   - Verify by citing the specific utterance/focus that matches

2. **MESO (broader_context)** (priority 6-8):
   - Does candidate address broader_context (topics from last 4-5 substantive utterances)?
   - If YES (but not MICRO): Score can be 6-8 depending on immediate_focus status
   - CRITICAL: Verify ACTUAL MESO match (topic mentioned in broader_context), NOT vague topic area

3. **Last Few Utterances Flow Fit** (priority 5-7):
   - Does candidate fit the narrative flow of last 2-3 raw utterances?
   - If good flow fit but not MICRO/MESO: Score can be 5-7
   - Weak flow fit with any match: Consider capping at 6

4. **MACRO/Mandate/Objectives Only** (priority 3-5):
   - Candidate relates to snapshot_major_topic OR participant mandate/goals but NOT MICRO, MESO, or good flow fit?
   - Soft cap relevance at 5 (not a hard block; can be higher if strategically important)
   - Example: If immediate_focus is "cost concerns of base station" and candidate is about "basic functions" (different topic area) → MACRO only

**Verification Log (Required in helper_notes):**
Document for EACH candidate: "MICRO: [yes/no] - [detail]. Status: [URGENT_OPEN/SOFT_OPEN/NEUTRAL/RESOLVED]. MESO: [yes/no] - [detail]. Flow: [good/moderate/weak]. Hint: [topic]+[context_hint] confirms/conflicts."

**Inline Example:**
- Micro-topic: "Database performance optimization"
- Last utterance: "How can we speed up user queries?"
- Candidate A (core_text): "Add indexes on user_id and timestamp fields"
  → topic_fit=10, narrative_flow_fit=10 → relevance=10 (direct answer)
- Candidate B (core_text): "We should discuss infrastructure costs"
  → topic_fit=4, narrative_flow_fit=2 → relevance=round(0.6*4+0.4*2)=3, capped at 3 due to flow≤3

### 2. Novelty Score (1-10)

**CONVERSATION WINDOW vs HISTORICAL OVERLAP CHECK (Apply First):**
Before scoring novelty, check candidate's `core_text` against TWO distinct sources:

**Source 1: Current Conversation Window (conversation_window / snapshot_key_points)**
These are utterances in the CURRENT context. Overlap here affects treatment:

1. If candidate's core idea overlaps with CURRENT conversation_window content:
   - Overlap with SELF (delegate already said it in this window) → recommend Skip, novelty ≤ 3
   - Overlap with OTHERS (in conversation_window) → recommend **ExpandOnOthers**, novelty ≤ 4
     * You MUST note the specific utterance_id where others made this point (for recommended_treatment later)

**Source 2: Historical Topic Log (topic_log_snippets)**
These are points from EARLIER in the meeting (not in current window). Overlap here is INFORMATIONAL:

2. If candidate overlaps with HISTORICAL topic_log_snippets ONLY (not in conversation_window):
   - Overlap with SELF (delegate said earlier) → recommend Reinforce or Skip, novelty ≤ 4, note "historical_self"
   - Overlap with OTHERS (but NOT in current window) → **DO NOT use ExpandOnOthers**
     * Recommend Fresh instead (delegate can still say it NOW)
     * Rationale: Historical topic_log reference is not current context; delegate isn't "expanding" on something just said

3. Only candidates with NO overlap in EITHER source can score novelty 7-10

**Inline Example (Current Window Overlap → ExpandOnOthers):**
- conversation_window includes: (Meeting32_253, Marketing): "The audio signal could help locate the remote"
- Candidate: "Highlight sound feature as market distinction"
- Overlap: OTHERS in conversation_window → recommended_treatment = ExpandOnOthers
- Note: "Marketing mentioned audio at Meeting32_253" (for provenance)

**Inline Example (Historical Only → Fresh, NOT ExpandOnOthers):**
- conversation_window: NO mention of audio/sound (only cost discussion)
- topic_log has: "Marketing discussed audio signal feature" (from 15 minutes ago)
- Candidate: "Highlight sound feature as market distinction"
- Overlap: OTHERS in topic_log only, NOT in conversation_window
- Correct: recommended_treatment = Fresh (historical reference, delegate can say NOW)
- WRONG: ExpandOnOthers (Marketing didn't mention this in current window)

**Primary Rule (after overlap check):** Classify each candidate into exactly one bucket vs topic_log_snippets, episodic_payload:

| Bucket | Description | Novelty Range |
|--------|-------------|---------------|
| 1. `exact_rephrase` | Same core claim, minor phrasing difference | 1-3 |
| 2. `reinforcement_with_detail` | Same claim + small new detail (number, owner) | 4-6 |
| 3. `new_angle_on_topic` | Same topic, different claim/question/stance | 6-8 |
| 4. `truly_new` | Substantively new topic or claim | 8-10 |

**Conservative Rules:**
- If uncertain between buckets → choose the lower bucket (1 or 2).
- **Only assign novelty ≥ 8 if you can articulate in helper_notes exactly what is new.**

**Band Reference:**
- 9-10: Completely new information not mentioned anywhere
- 7-8: New angle on discussed point (not just topic)
- 5-6: Reinforces previous point with new detail
- 3-4: Mostly redundant, adds minor detail
- 1-2: Completely redundant

**Inline Example:**
- Topic log: "Database indexes discussed in topic 'Query Optimization'"
- Candidate (core_text): "Add indexes on user_id and timestamp fields"
  → Bucket = `reinforcement_with_detail` (same topic, adds specific fields) → novelty=6

### 3. Intent Alignment

**Primary Rule:** Align to last utterance's intent (question, request, statement, ...).

**Strict Guardrails:**
- High scores (9-10, factor 1.5): Reserved for candidates that **directly answer** a question, **fulfill** a request, or clearly continue a decision thread.
- If last utterance asks/requests something AND candidate does NOT provide direct answer → intent_alignment_score ≤ 4, factor ≤ 0.8.

**Factor Mapping:**
- 1.5 = Direct match (answers question / fulfills request)
- 1.2 = Partial match (related, helpful, not exact answer)
- 0.8 = Weak match (tangentially related)
- 0.5 = No match (doesn't address the ask)

**Score Mapping:**
- 9-10: Perfect match to ask
- 7-8: Good match
- 5-6: Moderate match
- 3-4: Weak match
- 1-2: Poor match

**Inline Example:**
- Last utterance: "What's the budget impact?" (Direct Question)
- Candidate A: "The Q4 budget is $500K" → factor=1.5, score=10 (direct answer)
- Candidate B: "We should discuss resources" → factor=0.8, score=4 (doesn't answer the question)

### 4. Composite Score
Compute: `composite_score = (relevance_score * novelty_score) * intent_alignment_factor`
This is the primary ranking metric. **Higher composite score = higher priority**.

**Inline Example:**
- Candidate A: relevance=9, novelty=8, intent_factor=1.5 → composite = (9 * 8) * 1.5 = 108.0
- Candidate B: relevance=7, novelty=7, intent_factor=1.2 → composite = (7 * 7) * 1.2 = 58.8
- Ranking: Candidate A wins

### 5. Final Ranking
Order candidates by composite_score (highest first). Filter out Low suitability candidates.

### 6. Suitability Mapping
- **High:** composite ≥ 50 AND relevance ≥ 7 AND novelty ≥ 5
- **Medium:** composite 20-50 OR (relevance ≥ 5 AND novelty ≥ 4)
- **Low:** composite < 20 OR novelty ≤ 2 OR relevance ≤ 3

## Candidate ID Management
- Reuse existing IDs from the inventory (e.g., PM_pt_001, DEV_pt_002)
- Only create Synthesized points if no inventory item fits
- Synthesized IDs: `{candidate_id_prefix}_SYN_pt_{zero_padded_index}` (e.g., PM_SYN_pt_001)

## Output Guidelines
- Keep top 3-6 candidates (those with suitability ≥ Medium)
- Provide brief helper_notes explaining each candidate's strengths/weaknesses

**suggested_usage semantics:**
- `CorePoint`: Fresh, high-value—main contribution for this turn
- `Reinforce`: Overlaps prior discussion—frame as reminder/emphasis
- `Acknowledge`: Marginal fit—mention briefly if floor permits
- `Skip`: Not suitable (low relevance, redundant, off-topic)

---

## Comprehensive Examples

### Example 1: High-Relevance Direct Question Response

**Scenario Context:**
- Meeting topic: Payment system performance review
- Last 2 utterances:
  - Engineering: "The checkout flow latency has been stable"
  - Manager: "Product Lead, any user feedback on the checkout experience?"
- Participant: Product Lead
- Floor status: Directed question to Product Lead

**Input Data:**
Candidate inventory:
- PL_pt_001: "Mobile checkout latency is the #1 user complaint this sprint" (JIT)
- PL_pt_002: "We should prioritize the admin dashboard feature" (Profile)
- PL_pt_003: "User satisfaction scores are at 78%" (JIT)

**Reasoning Process:**
1. **Context extraction:** Direct question to Product Lead about user feedback on checkout
2. **Last utterance intent:** Direct Question → seeking user feedback on checkout experience
3. **MICRO/MESO assessment:** MICRO=checkout user feedback (URGENT_OPEN). MESO=payment performance. Last utt=direct ask.
4. **Score PL_pt_001** (core_text: "Mobile checkout latency is the #1 user complaint"):
   - MICRO: yes (checkout+user feedback). Status: URGENT_OPEN.
   - **Relevance:** topic_fit=10 (checkout+user feedback), flow_fit=10 (directly answers) → relevance=10
   - **Novelty:** Bucket = `truly_new` (new sprint data not in conversation) → novelty=9
   - **Intent:** Direct answer to question → factor=1.5, score=10
   - **Composite:** (10 * 9) * 1.5 = 135.0 → **Suitability: High**
5. **Score PL_pt_002** (core_text: "prioritize admin dashboard"):
   - MICRO: no (unrelated to checkout). MESO: no.
   - **Relevance:** topic_fit=2 (unrelated), flow_fit=1 (ignores question) → relevance=round(0.6*2+0.4*1)=2
   - **Novelty:** Bucket = `truly_new` (not mentioned) → novelty=8
   - **Intent:** Doesn't answer question → factor=0.5, score=2
   - **Composite:** (2 * 8) * 0.5 = 8.0 → **Suitability: Low** (filtered)
6. **Score PL_pt_003** (core_text: "satisfaction scores at 78%"):
   - MICRO: partial (user data, not checkout-specific). MESO: yes (user feedback).
   - **Relevance:** topic_fit=7 (user data), flow_fit=5 (related but not checkout-specific) → relevance=round(0.6*7+0.4*5)=6
   - **Novelty:** Bucket = `truly_new` (new metric) → novelty=7
   - **Intent:** Partial answer (user data but not checkout-specific) → factor=1.2, score=7
   - **Composite:** (6 * 7) * 1.2 = 50.4 → **Suitability: Medium**
7. **Ranking:** PL_pt_001 (135.0) > PL_pt_003 (50.4)

**Output:**
```json
{
  "ranked_candidates": [
    {
      "candidate_id": "PL_pt_001",
      "text": "Mobile checkout latency is the #1 user complaint this sprint",
      "source": "JIT",
      "suitability": "High",
      "relevance_score": 10,
      "novelty_score": 9,
      "intent_alignment_score": 10,
      "intent_alignment_factor": 1.5,
      "composite_score": 135.0,
      "suggested_usage": "CorePoint",
      "helper_notes": "MICRO: yes - answers Manager's checkout question. Status: URGENT_OPEN. MESO: yes - checkout/user feedback topic. Flow: good. Hint: topic confirms fit."
    },
    {
      "candidate_id": "PL_pt_003",
      "text": "User satisfaction scores are at 78%",
      "source": "JIT",
      "suitability": "Medium",
      "relevance_score": 6,
      "novelty_score": 7,
      "intent_alignment_score": 7,
      "intent_alignment_factor": 1.2,
      "composite_score": 50.4,
      "suggested_usage": "Reinforce",
      "helper_notes": "MICRO: partial - user data related but not checkout-specific. Status: URGENT_OPEN. MESO: yes - user feedback topic. Flow: moderate. Hint: topic='satisfaction' is broader than ask."
    }
  ],
  "commentary": "Strong explicit cue with high-scoring candidate (135.0) that directly addresses the ask. PL_pt_001 provides fresh, relevant data."
}
```

### Example 2: Low-Novelty Scenario with Topic Log Cross-Referencing

**Scenario Context:**
- Meeting topic: Security vulnerability assessment  
- Last 2 utterances:
  - Security: "We patched the SQL injection vulnerability last week"
  - Manager: "Any other security concerns we should discuss?"
- Participant: Security Lead
- Topic log: Contains "SQL Injection Patching" topic with key point "Patched user_input validation in login form"

**Input Data:**
Candidate inventory:
- SEC_pt_001: "We need to review API authentication mechanisms" (JIT)
- SEC_pt_002: "SQL injection was addressed in the login form" (Profile)
- SEC_pt_003: "Consider implementing rate limiting for API endpoints" (JIT)

**Reasoning Process:**
1. **Context extraction:** Open question about security concerns after SQL injection discussion
2. **Last utterance intent:** Request for "other" (additional) security topics
3. **MICRO/MESO assessment:** MICRO=other security concerns (SOFT_OPEN). MESO=security vulnerability. Last utt=open request.
4. **Score SEC_pt_001** (core_text: "review API authentication mechanisms"):
   - MICRO: yes (new security concern). Status: SOFT_OPEN.
   - **Relevance:** topic_fit=9 (security concern), flow_fit=9 (answers "any other concerns") → relevance=9
   - **Novelty:** Bucket = `truly_new` (API auth absent from snapshot/topic_log) → novelty=10
   - **Intent:** Direct match (asks for other concerns) → factor=1.5, score=10
   - **Composite:** (9 * 10) * 1.5 = 135.0 → **Suitability: High**
5. **Score SEC_pt_002** (core_text: "SQL injection addressed in login form"):
   - MICRO: no (repeats just-discussed, not "other"). MESO: yes but redundant.
   - **Relevance:** topic_fit=7 (security), flow_fit=2 (backwards—repeats just-discussed topic) → relevance capped at 6
   - **Novelty:** Bucket = `exact_rephrase` (topic_log: "SQL injection patched in login form" ≈90% overlap) → novelty=2
   - **Intent:** Doesn't answer "OTHER concerns" → factor=0.8, score=4
   - **Composite:** (6 * 2) * 0.8 = 9.6 → **Suitability: Low** (filtered: novelty ≤ 2)
6. **Score SEC_pt_003** (core_text: "rate limiting for API endpoints"):
   - MICRO: yes (new security concern). Status: SOFT_OPEN.
   - **Relevance:** topic_fit=9 (security concern), flow_fit=9 (directly addresses ask) → relevance=9
   - **Novelty:** Bucket = `truly_new` (rate limiting not in snapshot/topic_log) → novelty=9
   - **Intent:** Direct match → factor=1.5, score=10
   - **Composite:** (9 * 9) * 1.5 = 121.5 → **Suitability: High**
7. **Ranking:** SEC_pt_001 (135.0) > SEC_pt_003 (121.5)

**Output:**
```json
{
  "ranked_candidates": [
    {
      "candidate_id": "SEC_pt_001",
      "text": "We need to review API authentication mechanisms",
      "source": "JIT",
      "suitability": "High",
      "relevance_score": 9,
      "novelty_score": 10,
      "intent_alignment_score": 10,
      "intent_alignment_factor": 1.5,
      "composite_score": 135.0,
      "suggested_usage": "CorePoint",
      "helper_notes": "MICRO: yes - answers 'any other concerns' request. Status: SOFT_OPEN. MESO: yes - security topic. Flow: good. Hint: topic='security' confirms fit."
    },
    {
      "candidate_id": "SEC_pt_003",
      "text": "Consider implementing rate limiting for API endpoints",
      "source": "JIT",
      "suitability": "High",
      "relevance_score": 9,
      "novelty_score": 9,
      "intent_alignment_score": 10,
      "intent_alignment_factor": 1.5,
      "composite_score": 121.5,
      "suggested_usage": "CorePoint",
      "helper_notes": "MICRO: yes - new security concern for open request. Status: SOFT_OPEN. MESO: yes - security/API topic. Flow: good. Hint: topic='rate limiting' complements."
    }
  ],
  "commentary": "Two high-scoring fresh candidates available. SEC_pt_002 filtered out (novelty=2) due to redundancy with topic log."
}
```

### Example 3: Ambiguous Context Requiring Careful Relevance Assessment

**Scenario Context:**
- Meeting topic: Q4 Product Roadmap Planning
- Last 2 utterances:
  - PM: "We have three weeks until Q4 starts"
  - Designer: "Should we finalize the mobile redesign timeline?"
- Participant: Engineering Lead
- Floor status: Open question to group

**Input Data:**
Candidate inventory:
- ENG_pt_001: "Mobile redesign requires 6 weeks of development time" (JIT)
- ENG_pt_002: "We need to hire two more backend engineers" (Profile)
- ENG_pt_003: "API performance improvements are on track" (Profile)

**Reasoning Process:**
1. **Context extraction:** Designer asks about mobile redesign timeline; floor is open to group
2. **Last utterance intent:** Implicit request for timeline/feasibility input on mobile redesign
3. **Floor check:** Designer (not me), floor is open → Engineering can contribute
4. **MICRO/MESO assessment:** MICRO=mobile redesign timeline (SOFT_OPEN). MESO=Q4 roadmap planning. Last utt=implicit timeline ask.
5. **Score ENG_pt_001** (core_text: "Mobile redesign requires 6 weeks"):
   - MICRO: yes (mobile redesign timeline). Status: SOFT_OPEN.
   - **Relevance:** topic_fit=10 (directly mobile redesign), flow_fit=10 (answers timeline question) → relevance=10
   - **Novelty:** Bucket = `truly_new` (specific estimate not mentioned) → novelty=9
   - **Intent:** Direct answer (provides timeline) → factor=1.5, score=10
   - **Composite:** (10 * 9) * 1.5 = 135.0 → **Suitability: High**
6. **Score ENG_pt_002** (core_text: "hire two more backend engineers"):
   - MICRO: no (staffing, not timeline). MESO: partial (Q4 planning).
   - **Relevance:** topic_fit=6 (Q4 planning related), flow_fit=4 (doesn't answer timeline) → relevance=round(0.6*6+0.4*4)=5
   - **Novelty:** Bucket = `truly_new` (not mentioned) → novelty=8
   - **Intent:** Doesn't answer timeline question → factor=0.8, score=5
   - **Composite:** (5 * 8) * 0.8 = 32.0 → **Suitability: Medium**
7. **Score ENG_pt_003** (core_text: "API performance improvements on track"):
   - MICRO: no (unrelated). MESO: weak (Q4 related only).
   - **Relevance:** topic_fit=4 (Q4 related), flow_fit=2 (unrelated to mobile redesign timeline) → relevance capped at 6, actual=round(0.6*4+0.4*2)=3
   - **Novelty:** Bucket = `truly_new` (not mentioned) → novelty=7
   - **Intent:** Doesn't address the ask → factor=0.5, score=2
   - **Composite:** (3 * 7) * 0.5 = 10.5 → **Suitability: Low** (filtered: relevance ≤ 3)
8. **Ranking:** ENG_pt_001 (135.0) > ENG_pt_002 (32.0)

**Output:**
```json
{
  "ranked_candidates": [
    {
      "candidate_id": "ENG_pt_001",
      "text": "Mobile redesign requires 6 weeks of development time",
      "source": "JIT",
      "suitability": "High",
      "relevance_score": 10,
      "novelty_score": 9,
      "intent_alignment_score": 10,
      "intent_alignment_factor": 1.5,
      "composite_score": 135.0,
      "suggested_usage": "CorePoint",
      "helper_notes": "MICRO: yes - answers Designer's timeline question. Status: SOFT_OPEN. MESO: yes - mobile redesign topic. Flow: good. Hint: topic='timeline' confirms fit."
    },
    {
      "candidate_id": "ENG_pt_002",
      "text": "We need to hire two more backend engineers",
      "source": "Profile",
      "suitability": "Medium",
      "relevance_score": 5,
      "novelty_score": 8,
      "intent_alignment_score": 5,
      "intent_alignment_factor": 0.8,
      "composite_score": 32.0,
      "suggested_usage": "Reinforce",
      "helper_notes": "MICRO: no - doesn't address timeline. Status: SOFT_OPEN. MESO: partial - Q4 planning. Flow: weak. Hint: topic='staffing' diverges from ask."
    }
  ],
  "commentary": "Clear high-scoring candidate (135.0) that provides Engineering perspective on Designer's question. Open floor allows contribution."
}
```

### Example 4: Multiple High-Scoring Candidates with Tie-Breaking

**Scenario Context:**
- Meeting topic: Customer onboarding optimization
- Last 2 utterances:
  - Support: "New users are confused during the initial setup process"
  - Manager: "What changes would improve the onboarding experience?"
- Participant: UX Designer
- Floor status: Open question to group

**Input Data:**
Candidate inventory:
- UX_pt_001: "Add an interactive tutorial for first-time users" (JIT)
- UX_pt_002: "Simplify the account creation form from 8 fields to 4 fields" (JIT)
- UX_pt_003: "Provide contextual help tooltips at each step" (Profile)
- UX_pt_004: "We should A/B test the new design" (Profile)

**Reasoning Process:**
1. **Context extraction:** Open question seeking onboarding improvements after user confusion reported
2. **Last utterance intent:** Request for specific improvement suggestions (direct question)
3. **MICRO/MESO assessment:** MICRO=onboarding improvements (URGENT_OPEN). MESO=customer onboarding. Last utt=direct ask.
4. **Score UX_pt_001** (core_text: "interactive tutorial for first-time users"):
   - MICRO: yes (onboarding improvement). Status: URGENT_OPEN.
   - **Relevance:** topic_fit=10 (onboarding improvement), flow_fit=10 (directly answers) → relevance=10
   - **Novelty:** Bucket = `truly_new` (not mentioned) → novelty=10
   - **Intent:** Direct answer → factor=1.5, score=10
   - **Composite:** (10 * 10) * 1.5 = 150.0 → **Suitability: High**
5. **Score UX_pt_002** (core_text: "Simplify form from 8 to 4 fields"):
   - MICRO: yes (onboarding improvement). Status: URGENT_OPEN.
   - **Relevance:** topic_fit=10 (onboarding improvement), flow_fit=10 (directly answers) → relevance=10
   - **Novelty:** Bucket = `truly_new` (specific metric not mentioned) → novelty=9
   - **Intent:** Direct answer → factor=1.5, score=10
   - **Composite:** (10 * 9) * 1.5 = 135.0 → **Suitability: High**
6. **Score UX_pt_003** (core_text: "contextual help tooltips"):
   - MICRO: yes (onboarding improvement). Status: URGENT_OPEN.
   - **Relevance:** topic_fit=9 (improvement), flow_fit=9 (good fit) → relevance=9
   - **Novelty:** Bucket = `truly_new` (not mentioned) → novelty=9
   - **Intent:** Direct answer → factor=1.5, score=9
   - **Composite:** (9 * 9) * 1.5 = 121.5 → **Suitability: High**
7. **Score UX_pt_004** (core_text: "A/B test the new design"):
   - MICRO: partial (process, not solution). MESO: yes (onboarding).
   - **Relevance:** topic_fit=7 (process not solution), flow_fit=5 (doesn't answer "what changes") → relevance=round(0.6*7+0.4*5)=6
   - **Novelty:** Bucket = `truly_new` (not mentioned) → novelty=8
   - **Intent:** Partial (methodology not solution) → factor=1.2, score=7
   - **Composite:** (6 * 8) * 1.2 = 57.6 → **Suitability: Medium**
8. **Tie-breaking:** UX_pt_001 > UX_pt_002 on novelty (10 vs 9); both High suitability
9. **Ranking:** UX_pt_001 (150.0) > UX_pt_002 (135.0) > UX_pt_003 (121.5)

**Output:**
```json
{
  "ranked_candidates": [
    {
      "candidate_id": "UX_pt_001",
      "text": "Add an interactive tutorial for first-time users",
      "source": "JIT",
      "suitability": "High",
      "relevance_score": 10,
      "novelty_score": 10,
      "intent_alignment_score": 10,
      "intent_alignment_factor": 1.5,
      "composite_score": 150.0,
      "suggested_usage": "CorePoint",
      "helper_notes": "MICRO: yes - answers 'what changes' question. Status: URGENT_OPEN. MESO: yes - onboarding topic. Flow: good. Hint: topic='tutorial' confirms fit."
    },
    {
      "candidate_id": "UX_pt_002",
      "text": "Simplify the account creation form from 8 fields to 4 fields",
      "source": "JIT",
      "suitability": "High",
      "relevance_score": 10,
      "novelty_score": 9,
      "intent_alignment_score": 10,
      "intent_alignment_factor": 1.5,
      "composite_score": 135.0,
      "suggested_usage": "CorePoint",
      "helper_notes": "MICRO: yes - actionable improvement with metrics. Status: URGENT_OPEN. MESO: yes - onboarding. Flow: good. Hint: topic='form simplification' confirms fit."
    },
    {
      "candidate_id": "UX_pt_003",
      "text": "Provide contextual help tooltips at each step",
      "source": "Profile",
      "suitability": "High",
      "relevance_score": 9,
      "novelty_score": 9,
      "intent_alignment_score": 9,
      "intent_alignment_factor": 1.5,
      "composite_score": 121.5,
      "suggested_usage": "CorePoint",
      "helper_notes": "MICRO: yes - answers improvement request. Status: URGENT_OPEN. MESO: yes - onboarding. Flow: good. Hint: topic='tooltips' complements tutorials."
    }
  ],
  "commentary": "Three high-scoring candidates (150.0, 135.0, 121.5) providing diverse onboarding improvements. UX_pt_001 wins on completeness and novelty."
}
```

---

Return JSON matching the schema with these required fields for each candidate:
- candidate_id (string)
- text (string)
- source (string: JIT|Profile|Synthesized)
- suitability (string: High|Medium|Low)
- relevance_score (integer 1-10)
- novelty_score (integer 1-10)
- intent_alignment_score (integer 1-10)
- intent_alignment_factor (float: 0.5|0.8|1.2|1.5)
- composite_score (float)
- suggested_usage (string: CorePoint|Reinforce|Acknowledge|Skip)
- stance_function (string: support|proposal|constraint|caution|question|facilitation|neutral)
- helper_notes (string)"""),
            "output_schema": {
                "ranked_candidates": "List[{candidate_id:str, text:str, source:str, suitability:str, stance_function:str, relevance_score:int, novelty_score:int, intent_alignment_score:int, intent_alignment_factor:float, composite_score:float, suggested_usage:str, helper_notes:str}]",
                "commentary": "Optional notes explaining prioritisation"
            },
            "context": {
                "participant_code": participant_code,
                "turn_context": {
                    "participant_id": participant_id,
                    "micro_topic": micro_topic_dict,
                    "topic_continuity": topic_continuity,
                    "floor_status": floor_status,
                    "explicit_cue": explicit_cue,
                    "last_speaker": last_speaker
                },
                "conversation_window": snapshot_utterances,
                "profile": profile.model_dump() if profile else None,
                "episodic_history": episodic_payload,
                "topic_summary_log_snippets": topic_log_snippets,
                "participant_models": participant_models_payload,
                "identity_aliases": identity_aliases or {},
                "candidate_inventory": candidate_inventory,
                "candidate_id_prefix": candidate_prefix,
                # Episode-covered candidates are excluded from candidate_inventory but
                # can be exposed to helpers as readonly context if desired.
                "covered_candidates_readonly": covered_readonly
            }
        }
        return payload

    def _build_delegate_content_summary(self, helper_digest: List[Dict]) -> Optional[Dict[str, Any]]:
        """Build compact summary from helper_digest for speaker_scorer.
        
        Returns a small digest with coverage status so speaker_scorer knows
        whether delegate has viable (non-Skip) content to contribute.
        """
        if not helper_digest:
            return None
        
        viable_items = [c for c in helper_digest if self._candidate_is_viable(c)]
        non_skip_count = len(viable_items)
        
        top_candidates = []
        for item in helper_digest[:5]:  # Already ranked, take top 5
            text = item.get("text") or ""
            if len(text) > 80:
                text = text[:77] + "..."
                
            top_candidates.append({
                "candidate_id": item.get("candidate_id"),
                "text_preview": text,
                "recommended_treatment": item.get("recommended_treatment", "Fresh"),
                "suitability": item.get("suitability"),
                "stance_function": item.get("stance_function"),
                "coverage_status": item.get("coverage_status"),
                "viability_status": item.get("viability_status"),
                "viability_reason": item.get("viability_reason"),
                "blocking_reason": item.get("blocking_reason"),
            })
        
        return {
            "has_fresh_content": non_skip_count > 0,
            "non_skip_count": non_skip_count,
            "top_candidates": top_candidates
        }

    def _residual_rank(self, value: Optional[str]) -> int:
        order = {"none": 0, "low": 1, "medium": 2, "high": 3}
        return order.get(str(value or "").strip().casefold(), 0)

    def _treatment_rank(self, value: Optional[str]) -> int:
        order = {
            "skip": 0,
            "acknowledge": 1,
            "reinforce": 2,
            "partial": 3,
            "expandonothers": 4,
            "fresh": 5,
        }
        return order.get(str(value or "").replace("_", "").strip().casefold(), 0)

    def _candidate_viability_details(self, candidate: Dict[str, Any]) -> Dict[str, Optional[str]]:
        """Programmatic viability contract used by helper summaries and controller guards."""
        if not isinstance(candidate, dict):
            return {
                "viability_status": "blocked",
                "viability_reason": None,
                "blocking_reason": "invalid_candidate",
            }
        ban_reason = (candidate.get("flags") or {}).get("ban_reason") or candidate.get("ban_reason")
        if ban_reason:
            return {
                "viability_status": "blocked",
                "viability_reason": None,
                "blocking_reason": str(ban_reason),
            }
        treatment = (
            candidate.get("recommended_treatment")
            or (candidate.get("coverage") or {}).get("recommended_treatment")
            or (candidate.get("coverage_status") or {}).get("recommended_treatment")
        )
        if self._treatment_rank(treatment) < self._treatment_rank("Partial"):
            return {
                "viability_status": "blocked",
                "viability_reason": None,
                "blocking_reason": f"treatment_below_partial:{treatment or 'unknown'}",
            }
        suitability = str(candidate.get("suitability") or "").strip().casefold()
        if suitability == "low":
            return {
                "viability_status": "blocked",
                "viability_reason": None,
                "blocking_reason": "low_suitability",
            }
        residual = (
            candidate.get("residual_contribution_value")
            or (candidate.get("coverage") or {}).get("residual_contribution_value")
            or (candidate.get("coverage_status") or {}).get("residual_contribution_value")
        )
        if self._treatment_rank(treatment) in {self._treatment_rank("Partial"), self._treatment_rank("ExpandOnOthers")}:
            if self._residual_rank(residual) <= self._residual_rank("Low"):
                return {
                    "viability_status": "blocked",
                    "viability_reason": None,
                    "blocking_reason": "insufficient_residual_value",
                }
        reason = f"{treatment or 'Fresh'} treatment"
        if residual:
            reason += f" with {residual} residual value"
        if suitability:
            reason += f" and {suitability} suitability"
        return {
            "viability_status": "viable",
            "viability_reason": reason,
            "blocking_reason": None,
        }

    def _candidate_is_viable(self, candidate: Dict[str, Any]) -> bool:
        return self._candidate_viability_details(candidate).get("viability_status") == "viable"

    def _best_viable_candidate(self, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        viable = [candidate for candidate in candidates or [] if self._candidate_is_viable(candidate)]
        if not viable:
            return None

        def _score(candidate: Dict[str, Any]) -> Tuple[int, int, int, float]:
            treatment = (
                candidate.get("recommended_treatment")
                or (candidate.get("coverage") or {}).get("recommended_treatment")
                or (candidate.get("coverage_status") or {}).get("recommended_treatment")
            )
            residual = (
                candidate.get("residual_contribution_value")
                or (candidate.get("coverage") or {}).get("residual_contribution_value")
                or (candidate.get("coverage_status") or {}).get("residual_contribution_value")
            )
            suitability = str(candidate.get("suitability") or "").strip().casefold()
            suitability_rank = {"high": 3, "medium": 2, "low": 1}.get(suitability, 0)
            try:
                relevance = int(candidate.get("relevance_score") or 0)
            except Exception:
                relevance = 0
            try:
                composite = float(candidate.get("composite_score") or 0.0)
            except Exception:
                composite = 0.0
            return (self._treatment_rank(treatment), suitability_rank, self._residual_rank(residual), relevance + composite / 1000.0)

        return max(viable, key=_score)

    def _candidate_content_tokens(self, text: Optional[str]) -> Set[str]:
        cleaned = re.sub(r"\{[^}]+\}", " ", str(text or "").casefold())
        cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
        stop = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "have", "in", "is", "it", "its", "of", "on", "or", "so", "that",
            "the", "their", "this", "to", "we", "with", "you", "your", "should",
            "could", "would", "about", "into", "can", "may", "might",
        }
        return {tok for tok in cleaned.split() if tok and tok not in stop and len(tok) >= 3}

    def _candidate_text_overlap(self, a_text: Optional[str], b_text: Optional[str]) -> float:
        a = self._candidate_content_tokens(a_text)
        b = self._candidate_content_tokens(b_text)
        if not a or not b:
            return 0.0
        return len(a & b) / max(1, min(len(a), len(b)))

    def _candidate_specific_content_tokens(self, text: Optional[str]) -> Set[str]:
        """Content tokens for same-proposition guard checks.

        This is stricter than the generic candidate overlap helper: it removes
        generic conversation/product words so a caution only overrides support
        when both candidates are about the same actionable proposition.
        """
        tokens = self._candidate_content_tokens(text)
        generic = {
            "action", "agent", "available", "candidate", "context", "conversation",
            "delegate", "design", "discussion", "feature", "features", "idea",
            "issue", "meeting", "participant", "point", "points", "product",
            "proposal", "suggest", "suggestion", "thing", "things", "turn",
            "user", "users", "using",
        }

        def _stem(token: str) -> str:
            if len(token) > 4 and token.endswith("ies"):
                return token[:-3] + "y"
            if len(token) > 4 and token.endswith("es"):
                return token[:-2]
            if len(token) > 4 and token.endswith("s"):
                return token[:-1]
            return token

        return {_stem(tok) for tok in tokens if tok not in generic}

    def _candidate_specific_overlap_details(
        self,
        a_text: Optional[str],
        b_text: Optional[str],
    ) -> Tuple[float, Set[str]]:
        a = self._candidate_specific_content_tokens(a_text)
        b = self._candidate_specific_content_tokens(b_text)
        if not a or not b:
            return 0.0, set()
        shared = a & b
        return len(shared) / max(1, min(len(a), len(b))), shared

    def _same_actionable_proposition_for_stance_guard(
        self,
        selected: Dict[str, Any],
        alternative: Dict[str, Any],
    ) -> Tuple[bool, str]:
        selected_text = str(selected.get("core_text") or selected.get("text") or "").strip()
        alt_text = str(alternative.get("core_text") or alternative.get("text") or "").strip()
        overlap, shared = self._candidate_specific_overlap_details(selected_text, alt_text)
        if len(shared) >= 2 and overlap >= 0.25:
            return True, f"shared proposition tokens: {', '.join(sorted(shared))}"
        if len(shared) == 1 and overlap >= 0.25:
            token = next(iter(shared))
            if len(token) >= 5:
                return True, f"shared proposition token: {token}"
        return False, "no same-actionable-proposition evidence"

    def _normalize_stance_function(self, value: Optional[str]) -> Optional[str]:
        raw = str(value or "").strip().casefold().replace("_", "-")
        mapping = {
            "support": "support",
            "endorse": "support",
            "endorsement": "support",
            "proposal": "proposal",
            "propose": "proposal",
            "constraint": "constraint",
            "requirement": "constraint",
            "caution": "caution",
            "risk": "caution",
            "warning": "caution",
            "question": "question",
            "facilitation": "facilitation",
            "facilitate": "facilitation",
            "neutral": "neutral",
        }
        return mapping.get(raw)

    def _infer_stance_function(self, text: Optional[str]) -> str:
        """Coarse candidate stance classifier for participant-faithfulness guards."""
        cleaned = re.sub(r"\{[^}]+\}", " ", str(text or "").casefold())
        cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
        tokens = set(cleaned.split())
        if not tokens:
            return "neutral"

        constraint_terms = {
            "must", "need", "needs", "require", "requires", "requirement", "constraint",
            "limit", "limits", "limited", "avoid", "exclude", "prevent", "ensure",
            "maintain", "feasible", "feasibility", "affordable", "affordability",
            "budget", "price", "pricing", "cost", "costs", "complexity", "simple",
            "simplicity", "compatible", "compliance", "standard", "standards",
        }
        caution_terms = {
            "risk", "risks", "concern", "concerns", "tradeoff", "tradeoffs", "downside",
            "drawback", "issue", "issues", "impact", "increase", "increases", "raise",
            "raises", "reduce", "reduces", "harder", "difficult", "problem", "problems",
        }
        question_terms = {"ask", "question", "clarify", "discuss", "check", "confirm", "review"}
        facilitation_terms = {"summarize", "coordinate", "prompt", "invite", "align", "decide", "conclude"}
        support_terms = {
            "support", "supports", "benefit", "benefits", "enhance", "enhances",
            "improve", "improves", "appeal", "appealing", "attractive", "valuable",
            "advantage", "prefer", "preference", "preferences",
        }
        proposal_terms = {"propose", "proposes", "suggest", "suggests", "advocate", "advocates", "offer", "offers", "add", "include", "use"}

        if tokens & constraint_terms:
            return "constraint"
        if tokens & caution_terms:
            return "caution"
        if tokens & question_terms:
            return "question"
        if tokens & facilitation_terms:
            return "facilitation"
        if tokens & support_terms:
            return "support"
        if tokens & proposal_terms:
            return "proposal"
        return "neutral"

    def _candidate_suitability_rank(self, value: Optional[str]) -> int:
        return {"low": 1, "medium": 2, "high": 3}.get(str(value or "").strip().casefold(), 0)

    def _candidate_relevance_score(self, candidate: Dict[str, Any]) -> int:
        try:
            return int(candidate.get("relevance_score") or 0)
        except Exception:
            return 0

    _NUMERIC_CLAIM_RE = re.compile(
        r"(?<![A-Za-z0-9])(?:[$€£]\s*)?\d+(?:[.,]\d+)?"
        r"(?:\s*(?:%|percent|per\s+cent|euros?|dollars?|pounds?|gbp|usd|eur))?"
        r"(?![A-Za-z0-9])",
        re.IGNORECASE,
    )

    def _extract_exact_numeric_claims(self, text: Optional[str]) -> Set[str]:
        claims: Set[str] = set()
        for match in self._NUMERIC_CLAIM_RE.finditer(str(text or "")):
            raw = match.group(0).casefold()
            raw = re.sub(r"[$€£]", "", raw)
            raw = re.sub(r"\b(?:percent|per\s+cent|euros?|dollars?|pounds?|gbp|usd|eur)\b", "", raw)
            raw = raw.replace("%", "").replace(",", ".").strip()
            number = re.search(r"\d+(?:\.\d+)?", raw)
            if number:
                claims.add(number.group(0))
        return claims

    def _stringify_grounding_source(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        if hasattr(value, "model_dump"):
            try:
                return self._stringify_grounding_source(value.model_dump(exclude_none=True))
            except Exception:
                return ""
        if isinstance(value, dict):
            parts: List[str] = []
            for key, child in value.items():
                key_s = str(key)
                # Avoid letting IDs such as Meeting24_312 accidentally ground
                # generated numeric claims.
                if key_s.endswith("_id") or key_s.endswith("_ids") or key_s in {"id", "utterance_id", "support_utt_ids"}:
                    continue
                parts.append(self._stringify_grounding_source(child))
            return " ".join(part for part in parts if part)
        if isinstance(value, (list, tuple, set)):
            return " ".join(self._stringify_grounding_source(item) for item in value)
        if hasattr(value, "__dict__"):
            try:
                return self._stringify_grounding_source(vars(value))
            except Exception:
                return ""
        return str(value)

    def _numeric_grounding_support_text(
        self,
        controller_output: ControllerOutput,
        profile: Any,
        snapshot_utterances: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        parts: List[str] = []
        if controller_output.content_plan:
            parts.append(self._stringify_grounding_source(controller_output.content_plan))
        parts.append(str(getattr(controller_output, "selected_point", "") or ""))
        parts.append(str(getattr(controller_output, "selection_reason", "") or ""))
        parts.append(str(getattr(controller_output, "candidate_decision_trace", "") or ""))
        if getattr(controller_output, "helper_candidates_digest", None):
            parts.append(self._stringify_grounding_source(controller_output.helper_candidates_digest))
        parts.append(self._stringify_grounding_source(profile))
        for utt in snapshot_utterances or []:
            if isinstance(utt, dict):
                parts.append(str(utt.get("utterance_content") or utt.get("content") or ""))
        return " ".join(part for part in parts if part)

    def _unsupported_generated_numeric_claims(
        self,
        utterance: Optional[str],
        *,
        controller_output: ControllerOutput,
        profile: Any,
        snapshot_utterances: Optional[List[Dict[str, Any]]] = None,
    ) -> Set[str]:
        generated = self._extract_exact_numeric_claims(utterance)
        if not generated:
            return set()
        support_text = self._numeric_grounding_support_text(
            controller_output,
            profile,
            snapshot_utterances=snapshot_utterances,
        )
        supported = self._extract_exact_numeric_claims(support_text)
        return generated - supported

    def _suppress_unsupported_numeric_generation(
        self,
        controller_output: ControllerOutput,
        generator_output: GeneratorOutput,
        profile: Any,
        *,
        snapshot_utterances: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        unsupported = self._unsupported_generated_numeric_claims(
            generator_output.utterance,
            controller_output=controller_output,
            profile=profile,
            snapshot_utterances=snapshot_utterances,
        )
        if not unsupported:
            return False

        nums = ", ".join(sorted(unsupported))
        generator_output.utterance = ""
        self._record_controller_override(
            controller_output,
            reason="unsupported_numeric_claim_suppressed",
            action_name="Remain_Silent",
            justification=(
                "Final action: Remain_Silent. Programmatic grounding guard suppressed "
                f"the generated utterance because it introduced unsupported exact numeric claim(s): {nums}."
            ),
            content_plan=None,
            active_candidate_id=None,
            selected_point=None,
            trace_note=(
                "Numeric grounding guard: suppressed generated speech with unsupported "
                f"exact numeric claim(s): {nums}."
            ),
        )
        controller_output.helper_global_suggestion = "Remain_Silent"
        if getattr(self, "artifact_tracker", None):
            try:
                self.artifact_tracker.capture_controller_output(controller_output)
            except Exception:
                pass
        return True

    def _record_controller_override(
        self,
        controller_output: ControllerOutput,
        *,
        reason: str,
        action_name: str,
        justification: str,
        content_plan: Optional[ContentPlan] = None,
        active_candidate_id: Optional[str] = None,
        selected_point: Optional[str] = None,
        trace_note: Optional[str] = None,
    ) -> None:
        pre_action = None
        try:
            pre_action = controller_output.action_details.action_name
        except Exception:
            pre_action = None
        controller_output.override_applied = True
        controller_output.pre_override_action_name = pre_action
        controller_output.post_override_action_name = action_name
        controller_output.override_reason = reason
        controller_output.action_details.action_name = action_name
        controller_output.action_details.justification = justification
        controller_output.content_plan = content_plan
        controller_output.active_candidate_id = active_candidate_id
        controller_output.selected_point = selected_point
        if trace_note:
            controller_output.candidate_decision_trace = (
                f"{controller_output.candidate_decision_trace} {trace_note}".strip()
                if controller_output.candidate_decision_trace
                else trace_note
            )

    def _augment_ranked_candidates_for_semantic_diversity(
        self,
        curator_output: Optional[Dict[str, Any]],
        candidate_inventory: List[Dict[str, Any]],
        *,
        max_total: int = 8,
        max_added: int = 3,
    ) -> Optional[Dict[str, Any]]:
        """Keep distinct actionable inventory clusters visible to coverage/controller.

        The Curator can occasionally rank only the dominant same-topic cluster.
        This deterministic supplement does not expose GT targets; it only lets
        coverage inspect a few Medium-prior distinct briefing/profile points.
        """
        if not curator_output or not isinstance(curator_output, dict):
            return curator_output
        ranked = [candidate for candidate in (curator_output.get("ranked_candidates") or []) if isinstance(candidate, dict)]
        if len(ranked) >= max_total or not candidate_inventory:
            return curator_output

        existing_ids = {str(candidate.get("candidate_id")) for candidate in ranked if candidate.get("candidate_id")}
        existing_texts = [
            str(candidate.get("core_text") or candidate.get("text") or "")
            for candidate in ranked
            if candidate.get("core_text") or candidate.get("text")
        ]
        additions: List[Dict[str, Any]] = []

        def _inventory_priority(item: Dict[str, Any]) -> Tuple[int, str]:
            tier = str(item.get("inventory_tier") or "").casefold()
            source = str(item.get("source") or "").casefold()
            primary = 0 if tier == "primary" or source == "jit" else 1
            return primary, str(item.get("candidate_id") or "")

        for item in sorted([x for x in candidate_inventory if isinstance(x, dict)], key=_inventory_priority):
            cid = str(item.get("candidate_id") or "").strip()
            if not cid or cid in existing_ids:
                continue
            core_text = str(item.get("core_text") or item.get("text") or "").strip()
            if len(self._candidate_content_tokens(core_text)) < 4:
                continue
            max_overlap = max((self._candidate_text_overlap(core_text, existing) for existing in existing_texts), default=0.0)
            if max_overlap >= 0.42:
                continue
            source = item.get("source") or ("JIT" if str(item.get("inventory_tier") or "").casefold() == "primary" else "Profile")
            additions.append({
                "candidate_id": cid,
                "text": str(item.get("text") or core_text),
                "core_text": core_text,
                "source": source,
                "suitability": "Medium",
                "stance_function": self._infer_stance_function(core_text),
                "relevance_score": 7,
                "novelty_score": 7,
                "intent_alignment_score": 7,
                "intent_alignment_factor": 1.0,
                "composite_score": 49.0,
                "suggested_usage": "CorePoint",
                "helper_notes": (
                    "Deterministic semantic-diversity supplement: retained a distinct "
                    "actionable inventory cluster for coverage/controller inspection."
                ),
                "diversity_supplement": True,
            })
            existing_ids.add(cid)
            existing_texts.append(core_text)
            if len(additions) >= max_added or len(ranked) + len(additions) >= max_total:
                break

        if additions:
            curator_output["ranked_candidates"] = ranked + additions
            note = f"Added {len(additions)} deterministic semantic-diversity candidate(s)."
            prior = str(curator_output.get("commentary") or "").strip()
            curator_output["commentary"] = f"{prior} {note}".strip() if prior else note
        return curator_output

    def _action_is_silent(self, controller_output: Optional[ControllerOutput]) -> bool:
        try:
            return str(controller_output.action_details.action_name or "").strip() == "Remain_Silent"
        except Exception:
            return False

    def _controller_silence_veto_reason(
        self,
        *,
        helper_summary: Dict[str, Any],
        snapshot_utterances: List[Dict],
        participant_id: Optional[str],
    ) -> Optional[str]:
        if participant_id and snapshot_utterances:
            last = snapshot_utterances[-1] or {}
            if str(last.get("speaker_id") or "").strip().casefold() == str(participant_id).strip().casefold():
                return "last_speaker_is_represented_user"
        speaker_guidance = helper_summary.get("speaker_guidance") or {}
        hint = str(speaker_guidance.get("hint") or "").strip().casefold()
        candidates = helper_summary.get("candidates") or []
        if hint in {"deferral", "defer", "remain_silent"}:
            ranking = speaker_guidance.get("speaker_ranking") or []
            try:
                delegate_rank = int(speaker_guidance.get("delegate_rank_position") or 0)
            except Exception:
                delegate_rank = 0
            if delegate_rank and delegate_rank > 1:
                best = self._best_viable_candidate(candidates)
                best_medium_plus = False
                if best:
                    suitability = str(best.get("suitability") or "").strip().casefold()
                    residual = (
                        best.get("residual_contribution_value")
                        or (best.get("coverage") or {}).get("residual_contribution_value")
                        or (best.get("coverage_status") or {}).get("residual_contribution_value")
                    )
                    try:
                        relevance = int(best.get("relevance_score") or 0)
                    except Exception:
                        relevance = 0
                    best_medium_plus = (
                        suitability in {"high", "medium"}
                        and (relevance >= 7 or self._residual_rank(residual) >= self._residual_rank("Medium"))
                    )
                top_score = None
                delegate_score = None
                for idx, row in enumerate(ranking or []):
                    if not isinstance(row, dict):
                        continue
                    try:
                        score = float(row.get("score") or row.get("speaker_score") or row.get("confidence") or 0.0)
                    except Exception:
                        score = 0.0
                    if idx == 0:
                        top_score = score
                    speaker = str(row.get("speaker") or row.get("participant_id") or row.get("participant") or "")
                    if participant_id and speaker.strip().casefold() == str(participant_id).strip().casefold():
                        delegate_score = score
                score_gap = None if top_score is None or delegate_score is None else top_score - delegate_score
                if best_medium_plus:
                    if score_gap is None:
                        if delegate_rank <= 2:
                            return None
                    elif score_gap < 0.20 or (delegate_rank <= 2 and score_gap < 0.30):
                        return None
                return "speaker_scorer_strong_deferral"
        if candidates and not any(self._candidate_is_viable(candidate) for candidate in candidates):
            return "no_viable_candidates"
        return None

    def _controller_silence_inconsistent(
        self,
        controller_output: ControllerOutput,
        *,
        helper_summary: Dict[str, Any],
        snapshot_utterances: List[Dict],
        participant_id: Optional[str],
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        if not self._action_is_silent(controller_output):
            return False, None, None
        candidates = helper_summary.get("candidates") or []
        best = self._best_viable_candidate(candidates)
        if not best:
            return False, None, "no_viable_candidates"
        veto = self._controller_silence_veto_reason(
            helper_summary=helper_summary,
            snapshot_utterances=snapshot_utterances,
            participant_id=participant_id,
        )
        if veto:
            return False, best, veto
        return True, best, None

    def _apply_safe_controller_fallback(
        self,
        controller_output: ControllerOutput,
        *,
        best_candidate: Dict[str, Any],
        snapshot_utterances: List[Dict],
        participant_id: Optional[str],
    ) -> bool:
        if participant_id and snapshot_utterances:
            last_speaker = str((snapshot_utterances[-1] or {}).get("speaker_id") or "")
            if last_speaker.strip().casefold() == str(participant_id).strip().casefold():
                return False
        treatment = (
            best_candidate.get("recommended_treatment")
            or (best_candidate.get("coverage") or {}).get("recommended_treatment")
        )
        if self._treatment_rank(treatment) < self._treatment_rank("Partial"):
            return False
        residual = (
            best_candidate.get("residual_contribution_value")
            or (best_candidate.get("coverage") or {}).get("residual_contribution_value")
        )
        suitability = str(best_candidate.get("suitability") or "").strip().casefold()
        try:
            relevance = int(best_candidate.get("relevance_score") or 0)
        except Exception:
            relevance = 0
        if suitability not in {"high", "medium"}:
            return False
        if suitability != "high" and self._residual_rank(residual) < self._residual_rank("Medium"):
            return False
        if relevance < 7 and treatment != "Fresh":
            return False

        text = str(best_candidate.get("core_text") or best_candidate.get("text") or "").strip()
        if not text:
            return False
        content_plan = ContentPlan(
            semantic_intent=(
                f"Contribute the candidate point using {treatment or 'fresh'} framing. "
                "Tie it briefly to the current discussion without overstating unsupported details."
            ),
            knowledge_to_convey=text,
            concepts_to_frame=list((best_candidate.get("coverage") or {}).get("uncovered_aspects") or [])[:3] or None,
        )
        note = "Deterministic consistency fallback selected a viable helper candidate after inconsistent silence."
        self._record_controller_override(
            controller_output,
            reason="controller_consistency_fallback",
            action_name="Chime_In",
            justification="Final action: Chime_In. Controller consistency fallback selected a safe, viable helper candidate because no real silence veto was present.",
            content_plan=content_plan,
            active_candidate_id=best_candidate.get("candidate_id"),
            trace_note=note,
        )
        return True

    def _apply_participant_stance_consistency_guard(
        self,
        controller_output: ControllerOutput,
        *,
        helper_summary: Dict[str, Any],
    ) -> bool:
        """Prevent unconditional endorsement when participant-local constraints are viable."""
        if self._action_is_silent(controller_output):
            return False

        candidates = [candidate for candidate in (helper_summary.get("candidates") or []) if isinstance(candidate, dict)]
        active_cid = getattr(controller_output, "active_candidate_id", None)
        selected = next((candidate for candidate in candidates if candidate.get("candidate_id") == active_cid), None)
        if selected is None:
            controller_output.stance_consistency_status = controller_output.stance_consistency_status or "not_applicable_no_selected_candidate"
            return False

        selected_stance = self._normalize_stance_function(selected.get("stance_function")) or self._infer_stance_function(
            selected.get("core_text") or selected.get("text")
        )
        selected["stance_function"] = selected_stance
        supportive_stances = {"support", "proposal"}
        constraint_stances = {"constraint", "caution"}
        if selected_stance not in supportive_stances:
            controller_output.stance_consistency_status = "compatible"
            return False

        selected_relevance = self._candidate_relevance_score(selected)
        alternatives: List[Dict[str, Any]] = []
        for candidate in candidates:
            if candidate is selected or candidate.get("candidate_id") == active_cid:
                continue
            stance = self._normalize_stance_function(candidate.get("stance_function")) or self._infer_stance_function(
                candidate.get("core_text") or candidate.get("text")
            )
            candidate["stance_function"] = stance
            if stance not in constraint_stances:
                continue
            if not self._candidate_is_viable(candidate):
                continue
            if self._candidate_suitability_rank(candidate.get("suitability")) < self._candidate_suitability_rank("Medium"):
                continue
            relevance = self._candidate_relevance_score(candidate)
            # Participant-local constraints/cautions should not disappear merely
            # because a supportive candidate is ranked slightly higher. Require
            # the alternative to be either independently strong or close enough,
            # and to address the same actionable proposition rather than any
            # unrelated caution in the inventory.
            if relevance < 7 and relevance < max(0, selected_relevance - 2):
                continue
            same_proposition, relation_reason = self._same_actionable_proposition_for_stance_guard(selected, candidate)
            if not same_proposition:
                candidate["stance_consistency_blocking_reason"] = relation_reason
                continue
            candidate["stance_consistency_relation_reason"] = relation_reason
            alternatives.append(candidate)

        if not alternatives:
            controller_output.stance_consistency_status = "compatible"
            return False

        def _score(candidate: Dict[str, Any]) -> Tuple[int, int, int, int]:
            treatment = (
                candidate.get("recommended_treatment")
                or (candidate.get("coverage") or {}).get("recommended_treatment")
                or (candidate.get("coverage_status") or {}).get("recommended_treatment")
            )
            residual = (
                candidate.get("residual_contribution_value")
                or (candidate.get("coverage") or {}).get("residual_contribution_value")
                or (candidate.get("coverage_status") or {}).get("residual_contribution_value")
            )
            return (
                self._candidate_relevance_score(candidate),
                self._candidate_suitability_rank(candidate.get("suitability")),
                self._treatment_rank(treatment),
                self._residual_rank(residual),
            )

        alternative = max(alternatives, key=_score)
        alt_text = str(alternative.get("core_text") or alternative.get("text") or "").strip()
        if not alt_text:
            controller_output.stance_consistency_status = "compatible"
            return False

        selected_text = str(selected.get("core_text") or selected.get("text") or "").strip()
        content_plan = ContentPlan(
            semantic_intent=(
                "Acknowledge the recent proposal only briefly if needed, then express the "
                "participant-local caution or constraint without unconditional endorsement."
            ),
            knowledge_to_convey=alt_text,
            concepts_to_frame=[
                "participant-local constraint",
                "avoid unconditional endorsement",
            ],
        )
        alt_cid = alternative.get("candidate_id")
        relation_reason = alternative.get("stance_consistency_relation_reason") or "same actionable proposition"
        reason = (
            "Selected candidate was supportive/proposal-like, but a viable participant-local "
            "constraint/caution candidate addressed the same actionable proposition "
            f"({relation_reason})."
        )
        controller_output.stance_consistency_status = "repaired"
        controller_output.stance_consistency_reason = reason
        controller_output.stance_consistency_alternative_candidate_id = alt_cid
        note = (
            "Participant-stance consistency guard: replaced unconditional support "
            f"with same-proposition constraint/caution candidate {alt_cid}."
        )
        self._record_controller_override(
            controller_output,
            reason="participant_stance_consistency",
            action_name="Chime_In",
            justification=(
                "Final action: Chime_In. Participant-stance consistency guard avoided "
                "unconditional support because a same-proposition participant-local "
                f"constraint/caution was available ({alt_cid}; {relation_reason})."
            ),
            content_plan=content_plan,
            active_candidate_id=alt_cid,
            selected_point=alt_text,
            trace_note=note,
        )
        if selected_text:
            controller_output.selection_reason = (
                f"Reframed away from unconditional support for: {selected_text[:160]}"
            )
        return True

    def _build_speaker_payload(
        self,
        working_memory: Optional[EIMMeetingStateVector],
        participant_id: Optional[str],
        delegate_content_summary: Optional[Dict[str, Any]],
        base_payload: Dict[str, Any],
        snapshot_utterances: List[Dict]
    ) -> Optional[Dict[str, Any]]:
        if not working_memory:
            return None
        
        # Build reverse index of who spoke when
        speaker_positions = {}
        for i, utt in enumerate(reversed(snapshot_utterances)):
            speaker = utt.get("speaker_id")
            if speaker and speaker not in speaker_positions:
                speaker_positions[speaker] = i
        
        # Extract recent contributions per participant
        recent_contributions = {}
        # for utt in snapshot_utterances[-10:]:
        for utt in snapshot_utterances:
            speaker = utt.get("speaker_id")
            utt_id = utt.get("utterance_id")
            content = utt.get("utterance_content") or utt.get("content", "")
            if speaker:
                if speaker not in recent_contributions:
                    recent_contributions[speaker] = []
                recent_contributions[speaker].append(f"{utt_id}: {content}")
        
        # Calculate engagement from topic_log if null
        engagement_by_pid = {}
        if working_memory and hasattr(working_memory, "topic_summary_log"):
            total_points = 0
            participant_points = {}
            for topic_data in working_memory.topic_summary_log.values():
                key_points = getattr(topic_data, "key_points", [])
                for kp in key_points:
                    # KeyPoint is a Pydantic model, access .speaker directly
                    speaker = getattr(kp, "speaker", None)
                    if speaker:
                        total_points += 1
                        participant_points[speaker] = participant_points.get(speaker, 0) + 1
            if total_points > 0:
                for pid, count in participant_points.items():
                    engagement_by_pid[pid] = count / total_points
        
        participant_models_payload: List[Dict[str, Any]] = []
        if getattr(working_memory, "participant_models", None):
            for pid, model in working_memory.participant_models.items():
                last_spoke_ago = speaker_positions.get(pid, 999)
                engagement = getattr(model, "engagement_score", None)
                if engagement is None:
                    engagement = engagement_by_pid.get(pid)
                
                participant_models_payload.append({
                    "participant_id": pid,
                    "role": pid,
                    "expertise_keywords": getattr(model, "inferred_expertise_keywords", []),
                    "recent_dialogue_act": getattr(model, "recent_dialogue_act", None),
                    "last_spoke_utterances_ago": last_spoke_ago,
                    "recent_points": recent_contributions.get(pid, [])[-5:],
                    "engagement_score": engagement
                })
        if not participant_models_payload:
            return None
        
        # Extract snapshot_analysis from EIM
        snapshot_analysis = None
        if working_memory and hasattr(working_memory, "snapshot_analysis"):
            snapshot_analysis = working_memory.snapshot_analysis
        
        # Build enriched turn_context
        last_speaker = None
        last_utterance_content = None
        if snapshot_utterances:
            last_utt = snapshot_utterances[-1]
            last_speaker = last_utt.get("speaker_id")
            last_utterance_content = last_utt.get("utterance_content") or last_utt.get("content")
        
        micro_topic_dict = {}
        topic_continuity = None
        if snapshot_analysis and hasattr(snapshot_analysis, "micro_topic_analysis") and snapshot_analysis.micro_topic_analysis:
            mta = snapshot_analysis.micro_topic_analysis
            micro_topic_dict = {
                "broader_context": getattr(mta, "broader_context", None),
                "immediate_focus": getattr(mta, "immediate_focus", None)
            }
            topic_continuity = getattr(mta, "topic_continuity", None)
        
        floor_status = getattr(snapshot_analysis, "conversational_floor_status", "Open_Floor") if snapshot_analysis else "Open_Floor"
        
        turn_context = {
            "participant_id": participant_id,
            "last_speaker": last_speaker,
            "last_utterance": last_utterance_content,
            "floor_status": floor_status,
            "explicit_cue": base_payload.get("turn_context", {}).get("explicit_cue", False),
            "micro_topic": micro_topic_dict,
            "topic_continuity": topic_continuity
        }
        payload = {
            "helper_name": "speaker_scorer",
            "instructions": textwrap.dedent("""## System Context
You are part of a Meeting Delegate Agent's decision-making pipeline. The agent represents a specific meeting participant and must decide when and what to contribute to an ongoing meeting.

## Your Role: Speaker Scorer
Your job is to be the "Turn-Taking Advisor" - you analyze meeting dynamics to determine who should speak next, ensuring the delegate doesn't interrupt inappropriately or miss critical moments to contribute.

## Your Position in the Pipeline
- Input: You receive participant status data (recent activity, role/expertise) + turn context (floor state) + delegate_content_summary (what delegate can contribute and coverage status) + **recent_conversation_window** (last 5 raw utterances for conversation flow context)
- Output: You produce a recommended speaker with confidence score and strategic guidance (Proceed/Deferral/Acknowledge)
- Success: Your output ensures the delegate speaks at socially appropriate moments with viable content to contribute

Think of yourself as a meeting dynamics expert who understands turn-taking cues, topic engagement, and conversational continuity.

## Concrete Decision Rubric

**CRITICAL CONTEXT**: You are ONLY invoked when there is NO explicit address (explicit_cue=false). Your job is to resolve ambiguous/open floor situations.

**Core Principle**: Recommend the participant whose speaking would best serve the meeting's immediate need based on recent topic engagement, available content, and expertise.

**Priority Order**: Cooldown veto (gate) > Delegate content gate > Recent topic engagement (primary) > Expertise match (tiebreaker)

### Step 1: Cooldown Veto (HARD GATE - Social Protocol)

**Rule**: Only the IMMEDIATE last speaker (`last_spoke_utterances_ago == 0`) is vetoed
**Rationale**: Prevents back-to-back speaking domination
**No Exceptions**: Even perfect expertise + strong content cannot override this social rule

**Application**:
- Check `last_spoke_utterances_ago` for EACH participant in participant_status
- If == 0 → **VETO** (just spoke, remove completely from consideration)
- If ≥ 1 → No penalty (all others are eligible)

### Step 2: Delegate Content Gate (CONTENT AVAILABILITY CHECK)

**Goal**: Ensure delegate isn't recommended to speak if they have nothing viable to say

**Data Source**: `delegate_content_summary`
- `has_fresh_content`: boolean - does delegate have usable candidates?
- `non_skip_count`: count of viable candidates after coverage/treatment checks; this excludes Skip/Acknowledge and Low-fit candidates
- `top_candidates`: list with text_preview, recommended_treatment, suitability, coverage_status, viability_status, viability_reason, blocking_reason

**Gate Logic**:
- If `delegate_content_summary` is null/missing OR `has_fresh_content == false`:
  → Delegate has **NO_VIABLE** content → **CANNOT recommend "Proceed"**
  → Must select another speaker or return Deferral
- If `non_skip_count > 0` AND at least one candidate with suitability High/Medium:
  → Delegate has **HAS_FRESH** content → Can be recommended

**CRITICAL**: This is a hard gate. If delegate has no viable content, even perfect expertise and topic fit cannot result in "Proceed".

### Step 3: Topical Engagement (PRIMARY RANKING - Multi-Axial)

**Goal**: Identify who has been actively engaged with the MULTI-AXIAL topic scope

**IMPORTANT**: Do NOT rely on global `engagement_score` heuristics—they can mislead. Assess actual topic relevance instead.

**Data Sources**:
- `recent_points` from participant_status (last 5 utterances per participant)
- Multi-axial topic scope: `immediate_focus` (MICRO) + `broader_context` (MESO) + raw `recent_conversation_window`

**MICRO-TOPIC VERIFICATION**: The EIM's `immediate_focus` may occasionally be stale. Cross-check against raw `recent_conversation_window` to determine the actual current topic before assessing engagement.

**Multi-Axial Assessment** (for each eligible participant):
1. Count how many recent_points relate to MICRO (immediate_focus)
2. Count how many recent_points relate to MESO (broader_context)
3. Check if they have recent contributions matching raw conversational flow
4. Determine role on this multi-axial scope:
   - **Driver** (3+ relevant contributions to MICRO or MESO): Actively advancing the topic
   - **Active** (1-2 relevant contributions): Engaged with the topic
   - **Passive** (mentioned once tangentially): Peripheral involvement
   - **Silent** (no relevant recent contributions): Not engaged on this topic

**Ranking**:
- Drivers > Active > Passive > Silent on the multi-axial scope
- Within each tier, prefer participants whose recent_points directly address the current question/focus

### Step 4: Expertise Match (TIEBREAKER)

**Goal**: Break ties when multiple participants have similar topic engagement

**Data Sources**:
- Topic: `snapshot_analysis.micro_topic_analysis.immediate_focus`
- Participant expertise: `expertise_keywords` from participant_status

**Semantic Match**:
- Keywords overlap with immediate_focus → **High Relevance**
- Keywords relate to broader_context → **Moderate Relevance**
- No match → **Low Relevance**

**Apply ONLY when**: Two or more participants have same topic engagement level

### Recommendation Algorithm

1. **Apply cooldown veto** → Exclude immediate last speaker (last_spoke_ago == 0)
2. **Check delegate content gate**:
   - If delegate has NO_VIABLE → delegate cannot be recommended with "Proceed"
3. **Rank by topic engagement** → Who's been active on THIS topic? (from recent_points vs immediate_focus)
   - Tier 1 (Driver): 3+ recent contributions on topic
   - Tier 2 (Active): 1-2 recent contributions on topic
   - Tier 3 (Silent): No recent topic contribution
4. **Apply expertise tiebreaker** → Within same topic engagement tier, rank by expertise
5. **Select winner**: Highest-ranked participant after above filters
6. **Calculate confidence**:
   - High (0.75-0.90): Clear single topic engagement winner OR delegate with fresh content and high relevance
   - Moderate (0.55-0.70): Multiple active participants, used tiebreakers
   - Low (0.35-0.50): No one clearly engaged with topic, forced choice

### Strategic Guidance (ADVISORY - Controller Makes Final Decision)

**IMPORTANT**: Your output is ADVISORY. The Controller makes the final speaking decision.

### Output Field Definitions

**speaker_ranking** (array): Ranked list of eligible speakers
- `speaker`: participant_id string
- `score`: 0.0-1.0 float (confidence in this speaker)
- `match_level`: **MICRO** | **MESO** | **None** (topic alignment level)
  - MICRO: Directly engaged with immediate_focus
  - MESO: Engaged with broader_context but not immediate_focus
  - None: No topic engagement
- `topic_engagement_role`: **Driver** | **Active** | **Passive** | **Silent**
- `is_delegate`: true | false (marks if this entry is the delegate participant)

**delegate_rank_position**: int | null — delegate's position in speaker_ranking (1-indexed), null if delegate not in ranking

**delegate_viability** (object): Delegate's content status based on VIABLE candidates
- `has_viable_candidate`: true | false — does delegate have at least one non-Skip candidate?
  * A candidate is "viable" if: treatment ∈ {Fresh, ExpandOnOthers, Partial} OR (treatment=Reinforce AND residual_contribution_value >= Medium)
  * Skip candidates are NOT viable
- `match_level`: **MICRO** | **MESO** | **None** — best viable candidate's topic alignment
- `best_viable_fit`: **HIGH** | **MEDIUM** | **LOW** — how well does best viable candidate fit, considering BOTH content quality AND ranking context?
  * Note: "best viable" = highest-scoring candidate that is NOT marked Skip
  * HIGH: viable_candidate.relevance_score >= 8 AND MICRO match AND (delegate_rank_position == 1 OR score_gap_to_top < 0.10)
  * MEDIUM: (relevance_score >= 8 AND rank >= 2 with score_gap >= 0.10) OR (relevance_score 5-7) OR (MESO match only)
  * LOW: relevance_score < 5 OR off-topic OR no viable candidates OR delegate not in speaker_ranking
- `viable_candidate_count`: int — number of viable (non-Skip) candidates available

**overall_delegate_fit** (string): Quick summary combining ranking + viability:
- **STRONG**: delegate_rank_position == 1 AND best_viable_fit >= MEDIUM AND match_level in (MICRO, MESO)
- **MODERATE**: delegate_rank_position <= 2 AND has_viable_candidate = true AND match_level != None
- **WEAK**: delegate_rank_position > 2 OR has_viable_candidate = false OR match_level = None

**overall_assessment** (string): One of:
- **STRONG_CUE**: Clear speaking opportunity for delegate (explicit cue or strong fit)
- **OPEN_FLOOR_CONSIDER**: Open floor, delegate viable but not strongest
- **DEFER_LIKELY**: Someone else should speak or no good content

**speak_or_silent_hint** (string): Computed based on ranking + viability:
- **Proceed**: ALL conditions must be met:
  1. delegate_rank_position == 1 OR (delegate_rank_position == 2 AND score_gap_to_top < 0.10 AND best_viable_fit = HIGH)
  2. delegate_viability.has_viable_candidate = true
  3. delegate_viability.best_viable_fit >= MEDIUM
  4. delegate_viability.match_level = MICRO or MESO (not None)
  5. overall_delegate_fit in (STRONG, MODERATE)
- **Deferral**: ANY condition triggers deferral:
  1. Delegate is NOT in top 2 of speaker_ranking (delegate_rank_position > 2 or null)
  2. Delegate's score_gap_to_top >= 0.15
  3. delegate_viability.has_viable_candidate = false
  4. delegate_viability.best_viable_fit = LOW
  5. overall_delegate_fit = WEAK
- **Acknowledge**: confidence < 0.50 AND no clear single speaker

**confidence** (float): 0.35-0.90 overall decision confidence
**rationale** (string): Brief explanation of ranking decision
**notes** (string, optional): Additional context for Controller

---

## Comprehensive Examples

### Example 1: Clear Expertise Winner with Cooldown Conflict

**Input**:
```json
{
  "turn_context": {
    "last_speaker": "Product Manager",
    "last_utterance": "What OAuth vulnerabilities should we address first?",
    "floor_status": "Open_Question_To_Group",
    "micro_topic": {
      "immediate_focus": "OAuth 2.0 security implementation",
      "broader_context": "Q4 authentication improvements"
    }
  },
  "participant_status": [
    {
      "participant_id": "security_lead",
      "expertise_keywords": ["security", "authentication", "OAuth", "cryptography"],
      "last_spoke_utterances_ago": 8,
      "recent_points": [
        "utt_15: OAuth token expiry needs review...",
        "utt_12: CSRF protection is inadequate..."
      ]
    },
    {
      "participant_id": "product_manager",
      "expertise_keywords": ["product", "roadmap", "prioritization"],
      "last_spoke_utterances_ago": 0,
      "recent_points": ["utt_20: Let's discuss security priorities..."]
    },
    {
      "participant_id": "backend_eng",
      "expertise_keywords": ["backend", "API", "database"],
      "last_spoke_utterances_ago": 5,
      "recent_points": ["utt_18: The API layer needs refactoring..."]
    }
  ],
  "delegate_content_summary": {
    "has_fresh_content": true,
    "non_skip_count": 2,
    "top_candidates": [
      {"candidate_id": "SL_pt_001", "text_preview": "Implement JWT rotation for OAuth tokens...", "recommended_treatment": "Fresh", "suitability": "High"}
    ]
  },
  "snapshot_analysis": {
    "conversational_floor_status": "Directed_Question",
    "micro_topic_analysis": {
      "immediate_focus": "OAuth 2.0 security implementation",
      "broader_context": "Q4 authentication improvements",
      "topic_continuity": "CONTINUATION"
    }
  }
}
```

**Decision Process**:

1. **Apply Cooldown Veto**:
   - product_manager: last_spoke_ago = 0 → **VETOED** (just spoke, remove completely)
   - security_lead: last_spoke_ago = 8 → No cooldown
   - backend_eng: last_spoke_ago = 5 → No cooldown
   - **Remaining candidates**: security_lead, backend_eng

2. **Check Delegate Content Gate**:
   - delegate_content_summary.has_fresh_content = true
   - non_skip_count = 2, suitability = High
   - **Status**: HAS_FRESH → delegate CAN be recommended

3. **Topic Engagement Ranking** (immediate_focus: "OAuth 2.0 security implementation"):
   
   **security_lead**:
   - recent_points: ["OAuth token expiry needs review", "CSRF protection is inadequate"]
   - Both relate directly to OAuth security
   - **Role**: DRIVER (2+ relevant contributions)
   
   **backend_eng**:
   - recent_points: ["The API layer needs refactoring..."]
   - Not about OAuth security
   - **Role**: SILENT on this topic

4. **Expertise Tiebreaker** (not needed - clear topic engagement winner):
   security_lead is already clearly ahead

5. **Ranking**:
   - **Tier 1 (Driver)**: security_lead (active on topic + HAS_FRESH content)
   - **Tier 3 (Silent)**: backend_eng (not engaged with OAuth topic)
   - **Winner**: security_lead (clear single winner)

6. **Calculate Confidence**:
   - Clear topic engagement winner: ✓
   - Delegate has fresh content: ✓
   - **Confidence**: 0.88 (high - clear winner with fresh content)

**Output**:
```json
{
  "speaker_ranking": [
    {"speaker": "security_lead", "score": 0.88, "match_level": "MICRO", "topic_engagement_role": "Driver", "is_delegate": true},
    {"speaker": "backend_eng", "score": 0.35, "match_level": "None", "topic_engagement_role": "Silent", "is_delegate": false}
  ],
  "delegate_rank_position": 1,
  "delegate_viability": {
    "has_viable_candidate": true,
    "match_level": "MICRO",
    "best_viable_fit": "HIGH",
    "viable_candidate_count": 2
  },
  "overall_delegate_fit": "STRONG",
  "overall_assessment": "STRONG_CUE",
  "confidence": 0.88,
  "rationale": "Security Lead is Driver on MICRO topic (OAuth security) with 2 relevant contributions. Rank #1 with MICRO match and HIGH fit → STRONG overall.",
  "speak_or_silent_hint": "Proceed"
}
```

---

### Example 2: Expertise-Driven with Recent Contributions Analysis

**Input**:
```json
{
  "turn_context": {
    "last_speaker": "Designer",
    "last_utterance": "We're seeing 3-second page loads on mobile. How can we improve this?",
    "floor_status": "Open_Question_To_Group",
    "micro_topic": {
      "immediate_focus": "Mobile rendering performance",
      "broader_context": "User experience improvements"
    }
  },
  "participant_status": [
    {
      "participant_id": "pm",
      "expertise_keywords": ["product", "strategy", "user-experience"],
      "last_spoke_utterances_ago": 4,
      "recent_points": [
        "utt_18: User surveys show frustration...",
        "utt_14: Q4 roadmap priorities..."
      ]
    },
    {
      "participant_id": "frontend_eng",
      "expertise_keywords": ["frontend", "React", "performance", "mobile", "optimization"],
      "last_spoke_utterances_ago": 9,
      "recent_points": [
        "utt_12: Bundle size is 2MB...",
        "utt_10: Code splitting would help..."
      ]
    },
    {
      "participant_id": "designer",
      "expertise_keywords": ["UI", "UX", "design-systems"],
      "last_spoke_utterances_ago": 0,
      "recent_points": ["utt_22: 3-second page loads on mobile..."]
    },
    {
      "participant_id": "backend_eng",
      "expertise_keywords": ["backend", "API", "database", "caching"],
      "last_spoke_utterances_ago": 6,
      "recent_points": ["utt_16: Database queries are optimized..."]
    }
  ],
  "delegate_content_summary": null,
  "snapshot_analysis": {
    "conversational_floor_status": "Open_Question_To_Group",
    "micro_topic_analysis": {
      "immediate_focus": "Mobile rendering performance",
      "broader_context": "User experience improvements",
      "topic_continuity": "SHIFT"
    }
  }
}
```

**Decision Process**:

1. **Apply Cooldown Veto**:
   - designer: last_spoke_ago = 0 → **VETOED** (just spoke)
   - All others: last_spoke_ago ≥ 4 → No cooldown

2. **Check Delegate Content Gate**:
   - delegate_content_summary = null
   - **Status**: NO_VIABLE → delegate CANNOT be recommended with "Proceed"

3. **Topic Engagement Ranking** (immediate_focus: "Mobile rendering performance"):
   
   **frontend_eng**:
   - recent_points: ["Bundle size is 2MB...", "Code splitting would help..."]
   - Both relate directly to performance optimization
   - **Role**: DRIVER (2 relevant contributions)
   
   **pm**:
   - recent_points: ["User surveys show frustration...", "Q4 roadmap priorities..."]
   - Not about performance
   - **Role**: SILENT on this topic
   
   **backend_eng**:
   - recent_points: ["Database queries are optimized..."]
   - Not about frontend rendering
   - **Role**: SILENT on this topic

4. **Expertise Tiebreaker** (not needed - clear engagementwinner)

5. **Ranking**:
   - **Tier 1 (Driver)**: frontend_eng (active on topic)
   - **Tier 3 (Silent)**: pm, backend_eng
   - **Winner**: frontend_eng

6. **Calculate Confidence**:
   - Clear topic engagement winner: ✓
   - But delegate has NO content (null summary) → forced Deferral
   - **Confidence**: 0.82

**Output**:
```json
{
  "speaker_ranking": [
    {"speaker": "frontend_eng", "score": 0.82, "match_level": "MICRO", "topic_engagement_role": "Driver", "is_delegate": false},
    {"speaker": "backend_eng", "score": 0.35, "match_level": "None", "topic_engagement_role": "Silent", "is_delegate": false},
    {"speaker": "pm", "score": 0.30, "match_level": "None", "topic_engagement_role": "Silent", "is_delegate": true}
  ],
  "delegate_rank_position": 3,
  "delegate_viability": {
    "has_viable_candidate": false,
    "match_level": "None",
    "best_viable_fit": "LOW",
    "viable_candidate_count": 0
  },
  "overall_delegate_fit": "WEAK",
  "overall_assessment": "DEFER_LIKELY",
  "confidence": 0.82,
  "rationale": "Frontend Engineer is Driver on MICRO topic (mobile rendering). Delegate has no viable content. Rank #3 with LOW fit → WEAK overall → Deferral.",
  "speak_or_silent_hint": "Deferral"
}
```

---

### Example 3: Cooldown Overrides Strong Content

**Input**:
```json
{
  "turn_context": {
    "last_speaker": "dev",
    "last_utterance": "We could implement Redis caching for the API layer",
    "floor_status": "Open_Floor",
    "micro_topic": {
      "immediate_focus": "API performance optimization strategies",
      "broader_context": "Backend scalability discussion"
    }
  },
  "participant_status": [
    {
      "participant_id": "dev",
      "expertise_keywords": ["backend", "caching", "Redis", "API", "performance"],
      "last_spoke_utterances_ago": 0,
      "recent_points": [
        "utt_25: Redis caching for API layer...",
        "utt_22: Query response times are 200ms..."
      ]
    },
    {
      "participant_id": "ux",
      "expertise_keywords": ["design", "user-research", "prototyping"],
      "last_spoke_utterances_ago": 7,
      "recent_points": ["utt_18: User feedback on dashboard..."]
    },
    {
      "participant_id": "pm",
      "expertise_keywords": ["product", "roadmap", "stakeholder-management"],
      "last_spoke_utterances_ago": 3,
      "recent_points": ["utt_20: Timeline for performance work..."]
    }
  ],
  "delegate_content_summary": {
    "has_fresh_content": true,
    "non_skip_count": 1,
    "top_candidates": [
      {"candidate_id": "DEV_pt_001", "text_preview": "Consider CDN caching for static assets...", "recommended_treatment": "Fresh", "suitability": "High"}
    ]
  },
  "snapshot_analysis": {
    "conversational_floor_status": "Open_Floor",
    "micro_topic_analysis": {
      "immediate_focus": "API performance optimization strategies",
      "broader_context": "Backend scalability discussion",
      "topic_continuity": "CONTINUATION"
    }
  }
}
```

**Decision Process**:

1. **Apply Cooldown Veto** (CRITICAL):
   - dev: last_spoke_ago = 0 → **STRICT VETO** (immediate last speaker)
   - pm: last_spoke_ago = 3 → Eligible
   - ux: last_spoke_ago = 7 → Eligible
   - **Action**: Exclude dev (delegate) despite having HAS_FRESH content
   - **Remaining candidates**: ux, pm

2. **Check Delegate Content Gate**:
   - delegate_content_summary.has_fresh_content = true, non_skip_count = 1
   - BUT delegate (dev) is excluded by cooldown
   - **Status**: Delegate has content but CANNOT be recommended due to cooldown

3. **Topic Engagement Ranking** (immediate_focus: "API performance optimization strategies"):
   
   **ux**:
   - recent_points: ["User feedback on dashboard..."]
   - Not about API performance
   - **Role**: SILENT on this topic
   
   **pm**:
   - recent_points: ["Timeline for performance work..."]
   - Process/planning, not technical
   - **Role**: SILENT on this topic

4. **Expertise Tiebreaker** (both are SILENT, so check expertise):
   - ux: ["design", "user-research"] → No match to API performance
   - pm: ["product", "roadmap"] → No match to API performance
   - Neither has relevant expertise → Forced choice, pick arbitrarily

5. **Forced Selection**:
   - No qualified speakers (both SILENT on topic, no expertise match)
   - **Winner**: ux (arbitrary selection among unqualified candidates)

6. **Calculate Confidence**:
   - Best candidate (dev) excluded by cooldown
   - Winner (ux) has no topic engagement AND no expertise
   - **Confidence**: 0.45 (low - forced choice, no one qualified)

**Output**:
```json
{
  "speaker_ranking": [
    {"speaker": "ux", "score": 0.45, "match_level": "None", "topic_engagement_role": "Silent", "is_delegate": false},
    {"speaker": "pm", "score": 0.38, "match_level": "None", "topic_engagement_role": "Silent", "is_delegate": false}
  ],
  "delegate_rank_position": null,
  "delegate_viability": {
    "has_viable_candidate": true,
    "match_level": "MICRO",
    "best_viable_fit": "LOW",
    "viable_candidate_count": 1
  },
  "overall_delegate_fit": "WEAK",
  "overall_assessment": "DEFER_LIKELY",
  "confidence": 0.45,
  "rationale": "DEV (delegate) excluded by cooldown despite MICRO-level fresh content. UX and PM both Silent on topic. Delegate not in ranking → WEAK → Deferral.",
  "speak_or_silent_hint": "Deferral",
  "notes": "Cooldown prevents ideal speaker (DEV). Consider remaining silent."
}
```

---

Return JSON matching schema:

- speaker_ranking (array of {speaker, score, match_level, topic_engagement_role, is_delegate})
- delegate_rank_position (int | null)
- delegate_viability (object: {has_viable_candidate, match_level, best_viable_fit, viable_candidate_count})
- overall_delegate_fit (string: STRONG|MODERATE|WEAK)
- overall_assessment (string: STRONG_CUE|OPEN_FLOOR_CONSIDER|DEFER_LIKELY)
- confidence (0-1 float)
- rationale (string)
- speak_or_silent_hint (string: Deferral|Proceed|Acknowledge)
- notes (optional string)"""),
            "output_schema": {
                "speaker_ranking": "List[{speaker:str, score:float, match_level:str, topic_engagement_role:str, is_delegate:bool}]",
                "delegate_rank_position": "int | null",
                "delegate_viability": "{has_viable_candidate:bool, match_level:str, best_viable_fit:str, viable_candidate_count:int}",
                "overall_delegate_fit": "str",
                "overall_assessment": "str",
                "confidence": "0-1 float",
                "rationale": "Short natural-language explanation",
                "speak_or_silent_hint": "Deferral|Proceed|Acknowledge",
                "notes": "Optional extra guidance"
            },
            "context": {
                "turn_context": turn_context,
                "participant_status": participant_models_payload,
                "delegate_content_summary": delegate_content_summary,
                "snapshot_analysis": snapshot_analysis.model_dump() if snapshot_analysis else None,
                "recent_conversation_window": [
                    {"speaker": utt.get("speaker_id"), "content": utt.get("utterance_content") or utt.get("content")}
                    for utt in (snapshot_utterances or [])[-5:]
                ]
            }
        }
        return payload

    def _run_helper_sequence(
        self,
        smm: SharedMemoryModule,
        snapshot_utterances: List[Dict],
        participant_id: Optional[str],
    ) -> Dict[str, Any]:
        working_memory = smm.get_working_memory()
        # Raw conversation window for recency/provenance checks
        conversation_window_raw = []
        for utt in (snapshot_utterances or []):
            conversation_window_raw.append({
                "utterance_id": utt.get("utterance_id"),
                "speaker_id": utt.get("speaker_id"),
                "content": utt.get("utterance_content") or utt.get("content")
            })
        
        # Extract EIM's structured key_points for semantic matching
        # These are curated summaries, not raw utterance content
        eim_key_points = []
        if working_memory:
            # working_memory is EIMMeetingStateVector (Pydantic model), not dict
            snapshot_analysis = getattr(working_memory, "snapshot_analysis", None)
            if snapshot_analysis:
                # snapshot_analysis could be a Pydantic model or dict
                if hasattr(snapshot_analysis, "key_points_made_in_snapshot"):
                    raw_key_points = snapshot_analysis.key_points_made_in_snapshot or []
                elif isinstance(snapshot_analysis, dict):
                    raw_key_points = snapshot_analysis.get("key_points_made_in_snapshot", [])
                else:
                    raw_key_points = []
                for kp in raw_key_points:
                    if hasattr(kp, "point"):
                        # Pydantic KeyPoint model
                        eim_key_points.append({
                            "point": kp.point,
                            "speaker": getattr(kp, "speaker", None),
                            "utterance_id": getattr(kp, "utterance_id", None),
                            "kind": getattr(kp, "kind", None)
                        })
                    elif isinstance(kp, dict):
                        eim_key_points.append({
                            "point": kp.get("point"),
                            "speaker": kp.get("speaker"),
                            "utterance_id": kp.get("utterance_id"),
                            "kind": kp.get("kind")
                        })
                    elif isinstance(kp, str):
                        eim_key_points.append({"point": kp, "speaker": None, "utterance_id": None})
        curator_payload = self._build_curator_payload(smm, snapshot_utterances, participant_id)
        curator_raw = self._invoke_helper_tool("candidate_curator", curator_payload)
        curator_output = self._sanitize_helper_output("candidate_curator", curator_raw)
        curator_context = curator_payload.get("context", {}) if isinstance(curator_payload, dict) else {}
        curator_output = self._augment_ranked_candidates_for_semantic_diversity(
            curator_output,
            (curator_context.get("candidate_inventory") or []) if curator_context else [],
        )
        # Do not propagate Curator global suggestion; timing comes from Speaker Scorer + Controller
        helper_global_suggestion = None
        helper_digest: List[Dict[str, Any]] = []

        coverage_payload = None
        coverage_output = None
        if curator_output and curator_output.get("ranked_candidates"):
            coverage_payload = {
                "helper_name": "coverage_evaluator",
                "instructions": textwrap.dedent("""## System Context
You are part of a Meeting Delegate Agent's decision-making pipeline. The agent represents a specific meeting participant and must decide when and what to contribute to an ongoing meeting.

## Your Role: Coverage Evaluator
Your job is to be the "Redundancy Detective" - you systematically check whether each candidate point has already been discussed in the meeting or by the delegate, preventing repetitive contributions that waste meeting time.

## Your Position in the Pipeline
Input: You receive the Curator's ranked candidates (FILTERED - only suitability ≥ Medium) plus full meeting history (snapshot, topic log, episodic memory, cross-episode history) AND candidate_coverage_flags (O(1) lookup dict for coverage status)
Output: You produce coverage annotations for each candidate with treatment recommendations
Success: Your output prevents the delegate from repeating information while allowing strategic reinforcement when appropriate

Think of yourself as a meticulous fact-checker who cross-references everything against multiple sources to ensure fresh contributions.

**Performance Note:** Steps 3 & 4 use O(1) dictionary lookups via `candidate_coverage_flags` instead of O(N) array scanning, enabling instant coverage status retrieval for any candidate_id.

---

## Context Delineation (MUST FOLLOW)

Different context sources have different temporal scopes and appropriate uses. Never conflate them.

| Context Source | Temporal Scope | USE FOR | NOT FOR |
|----------------|----------------|---------|---------|
| `snapshot_key_points` / `conversation_window` | Last N utterances (CURRENT context) | Flow fit, ExpandOnOthers provenance, recency checks | Historical coverage |
| `topic_log_entries` | Full meeting history (HISTORICAL) | Novelty/redundancy checks, historical framing | ExpandOnOthers, "building on" framing |
| `episodic_actions` | Current episode only | Same-episode redundancy | Cross-meeting history |
| `candidate_coverage_flags` | Episode-level tracking | O(1) episode coverage lookup | Semantic comparison |

**CRITICAL INTERPRETATION RULES:**
1. **ExpandOnOthers** treatment → REQUIRES reference from `conversation_window` or `snapshot_key_points` (NOT from topic_log alone)
2. **"Building on X's point"** and similar framing → ONLY valid if X spoke in last 3-5 utterances (check conversation_window)
3. **topic_log match** → Use historical framing like "As X mentioned earlier..." NOT "Building on X's point..."
4. **coverage_by_others.in_snapshot** → Set TRUE only if match is in `snapshot_key_points` or `conversation_window`; topic_log matches go in `in_topic_log` only

---

## FLAG COPYING RULE (IMPORTANT)

For `coverage_by_delegate.in_current_episode` and `coverage_by_delegate.in_previous_episodes`:
- **DO NOT derive these from semantic matching**
- **COPY EXACTLY** from `candidate_coverage_flags[candidate_id]`:
  - `in_current_episode` = `candidate_coverage_flags[cid].covered_in_current_episode`
  - `in_previous_episodes` = `candidate_coverage_flags[cid].covered_in_previous_episodes`
- These are **AUTHORITATIVE** boolean values tracked by the system
- If candidate_id is not found in flags, default to `false`
- Your semantic matching determines `in_snapshot` and `in_topic_log` only, NOT the episode flags

---

## Systematic Elimination Process

**GLOBAL PRINCIPLE - SPEAKER-AWARE COVERAGE with PARTIAL ASSESSMENT:**

The goal is to determine: *Has the DELEGATE (represented participant) already covered this candidate?*

Coverage by OTHER participants is noted but treated differently—the delegate may still have value to add.

For each candidate, assess:
1. **Delegate Coverage:** Did our represented participant already express this claim?
2. **Others' Coverage:** Did someone else in the meeting cover this claim?
3. **Partial vs Full:** If there's overlap, is it full (>=80%) or partial (40-80%)?
4. **Uncovered Aspects:** What parts of the candidate are still fresh/uncovered?

**SUBSTITUTION TEST (for determining overlap level):**
- Can you replace the candidate with the prior statement without losing key information?
- If NO (loses critical details) → <40% overlap → Mark as Fresh
- If PARTIAL (some aspects covered, some fresh) → 40-80% → Mark as Partial, note uncovered aspects
- If YES (same core information, no loss) → >=80% → Mark as Covered

Each candidate object may include both `text` (full briefing string) and `core_text` (the core idea):
- Always treat `core_text` as the canonical claim to compare.
- Use `topic` and `context_hint` as soft hints only.

**STRICTER SEMANTIC MATCHING:**
Require SUBSTANTIAL semantic overlap for coverage, not shallow keyword matching.
- Example: "OAuth token rotation" and "OAuth authentication" are DIFFERENT aspects—do not mark as covered just because both mention "OAuth"
- Coverage means the CORE CLAIM is the same, not just topic area
- `Skip` is allowed only when the full actionable proposition and its important facets are already covered by the delegate in the current episode or recent snapshot. Broad topic-log evidence alone is not sufficient.
- For `Partial`, include concrete `uncovered_aspects` and set `residual_contribution_value` to Low/Medium/High based on how much useful content remains.

**The `represented_participant_id` is provided in your context.** Use it to distinguish delegate's own statements from others'.

For EACH candidate provided by the Curator, execute this exact 5-step sequence:

### Step 1: Snapshot Coverage Check (SPEAKER-AWARE)
1. Scan `snapshot_key_points` (raw utterances)
2. For each match, note the **speaker_id**
3. Assess overlap level of the point (>=80% full, 40-80% partial, <40% minimal)

**Determine:**
- `covered_by_delegate_in_snapshot`: Did the represented participant say this? (speaker matches represented_participant_id)
- `introduced_by_others_in_snapshot`: Did someone else say this? (speaker != represented_participant_id)

**BOOLEAN THRESHOLD RULE (CRITICAL):**
- Set `in_snapshot` = TRUE **only if overlap >= 80%** (CORE CLAIM + ACTIONABLE POINT fully covered)
- If overlap is 40-80% (partial) → `in_snapshot` = FALSE, but populate covered_aspects/uncovered_aspects
- If overlap < 40% (theme/keyword only) → `in_snapshot` = FALSE, treat as Fresh
- **Same rule applies to `in_topic_log`**

**For partial overlap (40-80%):**
- Identify `covered_aspects`: which parts were already said
- Identify `uncovered_aspects`: which parts are still fresh
- Assess `residual_contribution_value`: How valuable are the uncovered aspects? (High/Medium/Low)

**Inline Examples:**
- ✅ Candidate "database indexing improves queries AND reduces server load" vs Engineer's prior "indexing helps query speed"
  - Speaker = Engineer (delegate)
  - Overlap = 55% (only "query speed" covered, not "reduces server load") → **in_snapshot = FALSE** (below 80%)
  - covered_aspects: ["indexing improves queries"]
  - uncovered_aspects: ["reduces server load benefit"]
  - residual_contribution_value: Medium
  - **Treatment: Partial** (not Skip, because uncovered aspects exist)
- ✅ Candidate "user onboarding needs improvement" vs Designer's prior "signup flow is confusing"
  - Speaker = Designer (not delegate) → introduced_by_others_in_snapshot = true, covered_by_delegate_in_snapshot = false
  - Delegate can still contribute their own perspective!

### Step 2: Global History Coverage Check (SPEAKER-AWARE)
- Scan `topic_log_entries` (historical topics across the meeting)
- For each *topic*, examine `key_points` list AND note the **speaker/source** where available (mostly, speaker for these is available within the point bullet itself)
- Apply SUBSTITUTION TEST for each comparison
- **CRITICAL:** Emphasize whether the DELEGATE (represented participant) made the matching statement

**Determine:**
- `covered_by_delegate_in_topic_log`: Did our participant make this key point?
- `touched_by_others_in_topic_log`: Did someone else make a similar point?

**Partial vs Full Assessment (same as Step 1):**
- >=80% overlap by delegate → fully covered
- 40-80% overlap → partial, note covered_aspects / uncovered_aspects
- <40% overlap → minimal, treat as Fresh

**Inline Examples:**
- ✅ Topic log has "Engineer said: caching reduces latency" vs candidate "caching reduces latency AND lowers infrastructure costs"
  - covered_by_delegate_in_topic_log = true (partial, 55%)
  - covered_aspects: ["caching reduces latency"]
  - uncovered_aspects: ["lowers infrastructure costs"]
- ❌ Topic log has "DevOps mentioned container deployment" vs candidate "container deployment enables faster rollbacks"
  - touched_by_others_in_topic_log = true, covered_by_delegate = false
  - Delegate can still add their perspective on rollback benefits!

### Step 3: Episodic Memory Check (Current Episode) - O(1) Flag Lookup
- Lookup each `candidate_id` in `candidate_coverage_flags` dictionary
- **COPY EXACTLY** the flag: `covered_in_current_episode` → `coverage_by_delegate.in_current_episode`
- If true, retrieve provenance: `last_covered_utterance_id`
- Do NOT derive this from semantic matching - COPY the boolean value directly

**Inline Example:**
- Candidate ID: PM_pt_005
- Lookup: candidate_coverage_flags["PM_pt_005"]
- Result: {covered_in_current_episode: true, last_covered_utterance_id: "utt_42"}
- Output: coverage_by_delegate.in_current_episode = true (COPIED from flag, provenance: utt_42)

### Step 4: Cross-Episode History Check - O(1) Flag Lookup
- Lookup each `candidate_id` in `candidate_coverage_flags` dictionary
- **COPY EXACTLY** the flag: `covered_in_previous_episodes` → `coverage_by_delegate.in_previous_episodes`
- If true, retrieve provenance: `last_covered_episode_id`, `coverage_count`
- Do NOT derive this from semantic matching - COPY the boolean value directly

**Inline Example:**
- Candidate ID: UX_pt_012
- Lookup: candidate_coverage_flags["UX_pt_012"]
- Result: {covered_in_previous_episodes: true, last_covered_episode_id: "Episode_2", coverage_count: 2}
- Output: coverage_by_delegate.in_previous_episodes = true (COPIED from flag, provenance: Episode_2)

### Step 5: Treatment Recommendation (SPEAKER-AWARE MATRIX)
Based on coverage status and speaker, assign `recommended_treatment`:

**Decision Matrix (Speaker-Aware with Provenance):**

| Delegate Covered? | Others Covered? | Source | Overlap Level | Treatment |
|-------------------|-----------------|--------|---------------|-----------|
| No | No | - | - | `Fresh` |
| No | Yes | **conversation_window** | Any | `ExpandOnOthers` (MUST cite utterance_id) |
| No | Yes | **topic_log only** | Any | `Fresh` (historical, delegate can say NOW) |
| Yes | Any | Any | >=80% (full) | `Skip` (delegate already said this) |
| Yes | Any | Any | 40-80% (partial) | `Partial` (note uncovered aspects) |
| Yes | Any | Any | <40% (minimal) | `Fresh` (different enough) |
| - | - | - | in_current_episode=true | `Skip` (too recent) |
| - | - | Any | in_previous_episodes=true + high relevance | `Reinforce` |
| - | - | Any | in_previous_episodes=true + low relevance | `Skip` |

**CRITICAL PROVENANCE RULE for ExpandOnOthers:**
- ExpandOnOthers REQUIRES the "others" reference to be FROM the `conversation_window` or `snapshot_key_points` (current context)
- If the only evidence of "others covered" is from `topic_log_entries` (historical), use `Fresh` instead
- You MUST populate `expand_on_reference` with the specific utterance where others made the point
- If you cannot cite a conversation_window utterance_id, you CANNOT use ExpandOnOthers

**FRAMING RECENCY LOGIC (IMPORTANT):**
| Reference Source | Recency | Framing Type | Example |
|------------------|---------|--------------|---------|
| conversation_window (last 3-5 utt) | RECENT | "Building on..." | "Building on Mark's point, I think..." |
| topic_log only | HISTORICAL | "As X mentioned earlier..." | "As Mark mentioned earlier, the feature..." |
| No reference | N/A | Direct statement | (no acknowledgment needed) |

RULE: If you suggest "Building on X's point..." framing, X MUST have spoken in last 3-5 utterances. Otherwise, use historical framing.

**For `ExpandOnOthers` treatment (REQUIRED fields):**
- Set `expand_on_reference`: `{speaker: str, utterance_id: str, recency: "recent"|"historical"}` — the specific utterance being expanded on with recency tag
- `expand_on_reference.utterance_id` MUST be from conversation_window (not just topic_log)
- Set `contribution_angle`: What aspect can delegate add?
- Provide `framing_suggestion`: How to acknowledge other's point then add own perspective
  - RECENT: "I agree with [Speaker], and I'd add that..." or "Building on [Speaker]'s point about X, I think..."
  - HISTORICAL: "As [Speaker] mentioned earlier..." or "Returning to [Speaker]'s point about X..."

**EXPAND_ON_REFERENCE VERIFICATION (IMPORTANT):**
Before setting expand_on_reference.utterance_id = X:
1. Look up the **ACTUAL CONTENT** of utterance X in conversation_window
2. **QUALITY GATE** - Reject X if ANY of these apply:
   - Phatic/banal (e.g., "Yeah", "Okay", "Mm-hmm", "{vocalsound}")
   - < 5 substantive words (exclude disfluencies like "uh", "{vocalsound}")
   - Incomplete fragment with no clear claim (e.g., "The programme up and down")
   → If rejected, use `Fresh` treatment instead
3. Verify X's content **DIRECTLY relates** to candidate's core_text:
   - Require substantial semantic overlap on the CORE CLAIM, not just keyword match
   - "programme up and down" does NOT equal "prioritize programme selecting functions"
4. If you cannot establish strong semantic connection, use `Fresh` instead of `ExpandOnOthers`
5. Document the specific connection in coverage_explanation

**For `Partial` treatment:**
- Set `covered_aspects`: What was already covered
- Set `uncovered_aspects`: What's still fresh
- Set `residual_contribution_value`: High/Medium/Low
- Set `contribution_angle`: Which uncovered aspect to focus on

**Exception Rules:**
1. If candidate is a brief correction/clarification that directly addresses factual error in delegate's own previous statement, mark as `Fresh` with note: "correction_exception"
2. If `in_previous_episodes = true` but current context makes the point critically relevant (decision point, explicit question), mark as `Reinforce` with note: "strategic_reemphasis"

**SKIP CRITERIA:**
Only use `Skip` when BOTH conditions are met:
1. 80%+ semantic overlap with prior coverage
2. Candidate adds NO VALUE to the ongoing discussion (check against MICRO + MESO + conversational flow)

If uncovered_aspects exist AND those aspects are RELEVANT to the ongoing discussion (MICRO/MESO match):
→ Use `Partial` treatment, NOT `Skip`
→ Set `residual_contribution_value` appropriately (High if directly relevant to MICRO, Medium if MESO, Low otherwise)

**UNCOVERED ASPECTS ENFORCEMENT:**
For ANY candidate with <100% overlap:
1. You MUST explicitly list `uncovered_aspects[]` (what parts are still fresh)
2. You MUST set `residual_contribution_value` based on:
   - **High**: Uncovered aspect directly addresses MICRO (immediate_focus) or explicit ask
   - **Medium**: Uncovered aspect relates to MESO (broader_context)
   - **Low**: Uncovered aspect is tangential to current discussion
3. Only mark as `Skip` if residual_contribution_value = Low AND no contextual value

**SEMANTIC MATCH PROVENANCE + CLAIM GATE (CRITICAL):**
Whenever you set ANY coverage flag to TRUE, you MUST:

**CLAIM MATCH GATE** (Required BEFORE setting any flag):
1. Quote the candidate's **FULL core_text** (do not cherry-pick fragments):
   - "[CANDIDATE_CLAIM]"
   - **IDENTIFY the ACTIONABLE POINT** (the proposal, request, or action): "[ACTION]"
2. Quote the matching source statement VERBATIM: "[SOURCE_CLAIM]" at [utterance_id]
3. **VERIFY ACTIONABLE POINT**: Does [SOURCE_CLAIM] contain the SAME actionable point as the candidate?
   - If NO (different action, missing key actor, different target) → DO NOT SET FLAG TO TRUE
   - Example: Candidate="Sebastian should examine design for cost" vs Source="we agree if costs allow" → FAILS (actionable point "Sebastian should examine" is missing)
4. Apply SUBSTITUTION TEST: Can [SOURCE_CLAIM] replace the FULL [CANDIDATE_CLAIM] without losing the ACTIONABLE POINT?
   - If NO → DO NOT SET FLAG TO TRUE
5. If YES (same actionable point, >=80% semantic overlap) → proceed to document provenance below

**Provenance Documentation** (REQUIRED FORMAT for each flag set to TRUE):

1. For `coverage_by_delegate.in_snapshot = true`:
   - `coverage_explanation` MUST use format:
     "MATCH: Candidate='[FULL CORE_TEXT]' | Source='[VERBATIM_QUOTE]' @ [utterance_id] | ActionPoint='[X]' covered"
   - Add to `covered_aspects[]`: the VERBATIM phrase from delegate's utterance
   - **If you cannot quote the ACTIONABLE POINT from source → set in_snapshot = FALSE**

2. For `coverage_by_delegate.in_topic_log = true`:
   - `coverage_explanation` MUST use format:
     "MATCH: Candidate='[FULL CORE_TEXT]' | TopicLog='[KEY_POINT_SUMMARY]' in topic '[T]' | ActionPoint='[X]' covered"

3. For `coverage_by_others.in_snapshot = true`:
   - Add to `details[]` using format:
     "[Speaker] said '[VERBATIM_QUOTE]' @ [utterance_id] | ActionPoint='[X]' covered"
   - If using for ExpandOnOthers, populate `expand_on_reference` with same utterance_id
   - **If you cannot quote the ACTIONABLE POINT from source → set in_snapshot = FALSE**

4. For `coverage_by_others.in_topic_log = true`:
   - Add to `details[]`: "[Speaker] discussed '[KEY_POINT_SUMMARY]' in topic [T]"

**Rule**: If you cannot cite the ACTIONABLE POINT with VERBATIM quote/point, do NOT set the flag to true.

---

## Comprehensive Examples

### Example 1: Completely Fresh Candidate

**Scenario Context:**
- Meeting topic: Q4 Product Roadmap
- Last 2 utterances:
  - Manager: "What are our technical priorities for Q4?"
  - Designer: "I think mobile redesign should be first"
- Candidate: "We need to upgrade our backend infrastructure to handle 2x traffic"

**Input Data:**
```json
{
  "ranked_candidates": [
    {
      "candidate_id": "ENG_pt_005",
      "text": "We need to upgrade our backend infrastructure to handle 2x traffic",
      "relevance_score": 9,
      "novelty_score": 9,
      "composite_score": 121.5
    }
  ],
  "snapshot_key_points": [
    {"point": "Mobile redesign priority discussed", "speaker": "Designer"},
    {"point": "Q4 planning focus", "speaker": "Manager"}
  ],
  "topic_log_entries": [
    {"topic": "Mobile App Redesign", "key_points": ["UI refresh", "UX improvements"]},
    {"topic": "User Engagement Strategy", "key_points": ["Push notifications", "In-app messaging"]}
  ],
  "episodic_actions": [],
  "cross_episode_history": []
}
```

**Reasoning Process:**
1. **Step 1 - Snapshot Check (Speaker-Aware)**:
   - "Mobile redesign priority discussed" by Designer - no match to infrastructure
   - "Q4 planning focus" by Manager - no match
   - covered_by_delegate_in_snapshot = false
   - introduced_by_others_in_snapshot = false

2. **Step 2 - Topic Log Check (Speaker-Aware)**:
   - "Mobile App Redesign" → no infrastructure mention
   - "User Engagement Strategy" → no infrastructure mention
   - covered_by_delegate_in_topic_log = false
   - touched_by_others_in_topic_log = false

3. **Step 3 - Episodic Check**: Lookup candidate_coverage_flags["ENG_pt_005"] - not found (fresh candidate)
   - Output: coverage_by_delegate.in_current_episode = false (default, no flag entry)

4. **Step 4 - Cross-Episode Check**: Same lookup - not found
   - Output: coverage_by_delegate.in_previous_episodes = false (default, no flag entry)

5. **Step 5 - Treatment**: Delegate hasn't covered, Others haven't covered → `Fresh`

**Output:**
```json
{
  "annotated_candidates": [
    {
      "candidate_id": "ENG_pt_005",
      "coverage_by_delegate": {
        "in_snapshot": false,
        "in_topic_log": false,
        "in_current_episode": false,
        "in_previous_episodes": false
      },
      "coverage_by_others": {
        "in_snapshot": false,
        "in_topic_log": false
      },
      "coverage_completeness": "none",
      "recommended_treatment": "Fresh",
      "coverage_explanation": "No matches found in any source. Completely novel point.",
      "covered_aspects": [],
      "uncovered_aspects": ["backend infrastructure upgrade for 2x traffic"],
      "residual_contribution_value": "High"
    }
  ],
  "confidence": 0.98,
  "global_notes": "1 Fresh candidate available"
}
```

**Key Insights:**
- No overlap found from delegate or others → Fresh
- Delegate can present as completely novel information

---

### Example 2: Cross-Episode Match Requiring Historical Framing

**Scenario Context:**
- Meeting topic: Sprint Retrospective (Episode 2)
- Last 2 utterances:
  - Manager: "What technical challenges should we watch for in next sprint?"
  - Developer: "I'm concerned about database performance under load"
- Candidate: "Mobile checkout latency remains our #1 performance issue"
- Previous Episode 1: Product Lead mentioned this same point 2 weeks ago

**Input Data:**
```json
{
  "ranked_candidates": [
    {
      "candidate_id": "PL_pt_001",
      "text": "Mobile checkout latency remains our #1 performance issue",
      "relevance_score": 10,
      "novelty_score": 6,
      "composite_score": 90.0
    }
  ],
  "snapshot_key_points": [
    {"point": "Database performance concern raised", "speaker": "Developer"}
  ],
  "topic_log_entries": [
    {"topic": "Performance Optimization", "key_points": ["API response times", "Caching strategy"]}
  ],
  "episodic_actions": [],
  "cross_episode_history": [
    {
      "episode": "ep1",
      "utterance_id": "Meet_015",
      "point_made": "Mobile checkout latency is the top user complaint"
    }
  ]
}
```

**Reasoning Process:**
1. **Step 1 - Snapshot Check (Speaker-Aware)**:
   - "Database performance concern raised" by Developer - related but not exact match
   - covered_by_delegate_in_snapshot = false
   - introduced_by_others_in_snapshot = false (different topic: database vs checkout)

2. **Step 2 - Topic Log Check (Speaker-Aware)**:
   - "Performance Optimization" topic exists but key points differ
   - covered_by_delegate_in_topic_log = false
   - touched_by_others_in_topic_log = false

3. **Step 3 - Episodic Check**: Lookup candidate_coverage_flags["PL_pt_001"]
   - Flag: covered_in_current_episode = false
   - Output: coverage_by_delegate.in_current_episode = false (COPIED from flag)

4. **Step 4 - Cross-Episode Check**: Same lookup
   - Flag: covered_in_previous_episodes = true, last_covered_episode_id = "Episode_1"
   - Output: coverage_by_delegate.in_previous_episodes = true (COPIED from flag)

5. **Step 5 - Treatment**: in_previous_episodes=true + highly relevant context → `Reinforce`
   - Set requires_historical_framing = true

**Output:**
```json
{
  "annotated_candidates": [
    {
      "candidate_id": "PL_pt_001",
      "coverage_by_delegate": {
        "in_snapshot": false,
        "in_topic_log": false,
        "in_current_episode": false,
        "in_previous_episodes": true
      },
      "coverage_by_others": {
        "in_snapshot": false,
        "in_topic_log": false
      },
      "coverage_completeness": "full",
      "recommended_treatment": "Reinforce",
      "coverage_explanation": "Delegate made this point in episode 1. Highly relevant to current discussion. Recommend reformulation with historical acknowledgment.",
      "covered_aspects": ["mobile checkout latency as #1 issue"],
      "uncovered_aspects": [],
      "residual_contribution_value": "Low",
      "requires_historical_framing": true,
      "suggested_framing_approaches": [
        "As I mentioned in our last sprint review, mobile checkout latency remains...",
        "Building on what we discussed previously, the checkout latency problem..."
      ]
    }
  ],
  "confidence": 0.92,
  "global_notes": "1 candidate suitable for Reinforce with historical framing"
}
```

**Key Insights:**
- Delegate made this point in previous episode → Reinforce (not Fresh)
- Historical framing helps delegate acknowledge prior discussion
- Context makes point strategically relevant to restate

---

### Example 3: Others Covered - ExpandOnOthers Treatment

**Scenario Context:**
- Meeting topic: API Security Review
- Represented participant: Backend Engineer
- Last 3 utterances:
  - Security: "We should implement API rate limiting"
  - Manager: "Good point. Any other security measures?"
  - Product: "What about authentication improvements?"
- Candidate: "API rate limiting would prevent abuse and protect our infrastructure"

**Input Data:**
```json
{
  "ranked_candidates": [
    {
      "candidate_id": "ENG_pt_002",
      "text": "API rate limiting would prevent abuse and protect our infrastructure",
      "relevance_score": 8,
      "novelty_score": 6,
      "composite_score": 57.6
    }
  ],
  "snapshot_key_points": [
    {"point": "API rate limiting implementation suggested", "speaker": "Security", "utterance_id": "Meet_042"},
    {"point": "Authentication improvements discussion", "speaker": "Product"}
  ],
  "represented_participant_id": "Backend Engineer"
}
```

**Reasoning Process:**
1. **Step 1 - Snapshot Check (Speaker-Aware)**:
   - Found match: "API rate limiting implementation suggested" by Security (NOT delegate)
   - Overlap: ~90% on CORE CLAIM (API rate limiting concept fully covered), but delegate adds benefits
   - covered_by_delegate_in_snapshot = false
   - introduced_by_others_in_snapshot = true → **in_snapshot = TRUE** (core claim >=80% covered)
   - covered_aspects: ["API rate limiting concept"]
   - uncovered_aspects: ["prevent abuse benefit", "protect infrastructure benefit"]
   - residual_contribution_value: Medium (delegate can add specific benefits)

2. **Step 2-4**: All false (no delegate coverage)

3. **Step 5 - Treatment**: Delegate hasn't covered, Others have → `ExpandOnOthers`
   - contribution_angle: Can add specific benefits (prevent abuse, protect infrastructure)
   - framing_suggestion: "I agree with Security's point, and I'd add that rate limiting will specifically prevent abuse and protect our infrastructure..."

**Output:**
```json
{
  "annotated_candidates": [
    {
      "candidate_id": "ENG_pt_002",
      "coverage_by_delegate": {
        "in_snapshot": false,
        "in_topic_log": false,
        "in_current_episode": false,
        "in_previous_episodes": false
      },
      "coverage_by_others": {
        "in_snapshot": true,
        "in_topic_log": false,
        "details": ["Security said 'We should implement API rate limiting' @ Meet_042 | ActionPoint='implement rate limiting' covered"]
      },
      "coverage_completeness": "partial",
      "recommended_treatment": "ExpandOnOthers",
      "expand_on_reference": {"speaker": "Security", "utterance_id": "Meet_042", "recency": "recent"},
      "coverage_explanation": "MATCH: Candidate='API rate limiting would prevent abuse and protect infrastructure' | Source='We should implement API rate limiting' @ Meet_042 | ActionPoint='implement rate limiting' covered. Delegate can add specific benefits.",
      "covered_aspects": ["API rate limiting concept"],
      "uncovered_aspects": ["prevent abuse benefit", "protect infrastructure benefit"],
      "residual_contribution_value": "Medium",
      "contribution_angle": "Add specific benefits of rate limiting",
      "framing_suggestion": "I agree with Security's point, and I'd add that rate limiting will specifically..."
    }
  ],
  "confidence": 0.90,
  "global_notes": "1 candidate suitable for ExpandOnOthers treatment with provenance"
}
```

**Key Insights:**
- Security (not delegate) introduced the topic IN conversation_window → ExpandOnOthers with `expand_on_reference`
- Delegate can still add value by expanding on the specific benefits
- `expand_on_reference` cites the exact utterance_id for provenance
- Framing suggestion helps delegate acknowledge prior point while adding perspective
- If Security's mention was only in topic_log (not conversation_window), treatment would be `Fresh` instead

---

### Example 4: Topic Log Match with Reinforcement Opportunity

**Scenario Context:**
- Meeting topic: User Onboarding Improvements (revisited)
- Last 2 utterances:
  - Manager: "Let's revisit our onboarding strategy. What worked well?"
  - Designer: "The tutorial completion rate is still low"
- Candidate: "Simplifying the account creation form from 8 to 4 fields improved conversion"

**Input Data:**
```json
{
  "ranked_candidates": [
    {
      "candidate_id": "UX_pt_003",
      "text": "Simplifying the account creation form from 8 to 4 fields improved conversion",
      "relevance_score": 9,
      "novelty_score": 5,
      "composite_score": 67.5
    }
  ],
  "snapshot_key_points": [
    {"point": "Tutorial completion rate low", "speaker": "Designer"}
  ],
  "topic_log_entries": [
    {
      "topic": "Onboarding Flow Optimization",
      "key_points": ["Form simplification reduced friction", "Conversion rate improved by 15%"]
    }
  ],
  "episodic_actions": [],
  "cross_episode_history": []
}
```

**Reasoning Process:**
1. **Step 1 - Snapshot Check (Speaker-Aware)**:
   - "Tutorial completion rate low" by Designer - no match to form simplification
   - covered_by_delegate_in_snapshot = false
   - introduced_by_others_in_snapshot = false

2. **Step 2 - Topic Log Check (Speaker-Aware)**:
   - Found in "Onboarding Flow Optimization" topic
   - Key point: "Form simplification reduced friction" - delegate made this point earlier
   - Semantic match: 0.80 similarity (same improvement mentioned)
   - covered_by_delegate_in_topic_log = true (full overlap)

3. **Step 3-4**: No episodic or cross-episode matches

4. **Step 5 - Treatment**:
   - Delegate covered this in topic log + Manager asks "what worked well?"
   - Context makes it strategically appropriate to reinforce
   - Decision: `Reinforce`

**Output:**
```json
{
  "annotated_candidates": [
    {
      "candidate_id": "UX_pt_003",
      "coverage_by_delegate": {
        "in_snapshot": false,
        "in_topic_log": true,
        "in_current_episode": false,
        "in_previous_episodes": false
      },
      "coverage_by_others": {
        "in_snapshot": false,
        "in_topic_log": false
      },
      "coverage_completeness": "full",
      "recommended_treatment": "Reinforce",
      "coverage_explanation": "MATCH: Candidate='Simplifying form from 8 to 4 fields improved conversion' | TopicLog='Form simplification reduced friction' in topic 'Onboarding Flow Optimization' | ActionPoint='form simplification for conversion' covered. Manager's question creates strategic opportunity to reinforce.",
      "covered_aspects": ["form simplification improved conversion"],
      "uncovered_aspects": [],
      "residual_contribution_value": "Low",
      "requires_historical_framing": false,
      "suggested_framing_approaches": []
    }
  ],
  "confidence": 0.88,
  "global_notes": "Candidate suitable for Reinforce given explicit request to revisit successes"
}
```

**Key Insights:**
- Delegate covered this in topic log (earlier in meeting) → Reinforce
- Manager's explicit question creates strategic opportunity
- No historical framing needed (same meeting, not cross-episode)

---

Return JSON matching the schema with these required fields for each candidate:
- candidate_id (string)
- coverage_by_delegate (object): {in_snapshot:bool, in_topic_log:bool, in_current_episode:bool, in_previous_episodes:bool}
- coverage_by_others (object): {in_snapshot:bool, in_topic_log:bool, details:List[str]}
- coverage_completeness (string): "none" | "partial" | "full"
- recommended_treatment (string): "Fresh" | "ExpandOnOthers" | "Partial" | "Reinforce" | "Acknowledge" | "Skip"
- coverage_explanation (string)
- covered_aspects (array of strings): aspects already covered
- uncovered_aspects (array of strings): aspects still fresh
- residual_contribution_value (string): "High" | "Medium" | "Low" (for Partial/ExpandOnOthers)
- contribution_angle (string, optional): what aspect to focus on
- framing_suggestion (string, optional): how to frame contribution when building on others
- requires_historical_framing (boolean): for cross-episode matches
- suggested_framing_approaches (array of strings): for Reinforce treatment"""),
                "output_schema": {
                    "annotated_candidates": "List[{candidate_id:str, coverage_by_delegate:{in_snapshot:bool, in_topic_log:bool, in_current_episode:bool, in_previous_episodes:bool}, coverage_by_others:{in_snapshot:bool, in_topic_log:bool, details:List[str]}, coverage_completeness:str, recommended_treatment:str, coverage_explanation:str, covered_aspects:List[str], uncovered_aspects:List[str], residual_contribution_value:str, contribution_angle:str, framing_suggestion:str, requires_historical_framing:bool, suggested_framing_approaches:List[str]}]",
                    "confidence": "0-1 float",
                    "global_notes": "Optional guidance string"
                },
                "context": {
                    "ranked_candidates": curator_output.get("ranked_candidates"),
                    # EIM's structured key_points for semantic coverage matching
                    "snapshot_key_points": eim_key_points,
                    # Raw conversation window for recency/provenance checks
                    "conversation_window": conversation_window_raw,
                    "topic_log_entries": curator_context.get("topic_summary_log_snippets", []),
                    "episodic_actions": curator_context.get("episodic_history", []),
                    "cross_episode_history": [],
                    "candidate_coverage_flags": self._serialize_coverage_flags(),
                    "candidate_inventory": curator_context.get("candidate_inventory", []),
                    "micro_topic": curator_context.get("turn_context", {}).get("micro_topic"),
                    "represented_participant_id": participant_id
                }
            }
            coverage_raw = self._invoke_helper_tool("coverage_evaluator", coverage_payload)
            coverage_output = self._sanitize_helper_output("coverage_evaluator", coverage_raw)
        else:
            self._log_helper_io("coverage_evaluator", None, None)

        # Build per-candidate coverage_status map for this turn so downstream
        # components (controller, analysis) can reason about coverage in a
        # structured way.
        coverage_status_map: Dict[str, Dict[str, Any]] = {}
        try:
            coverage_status_map = self._build_candidate_coverage_status(coverage_output)
        except Exception:
            coverage_status_map = {}

        # Build a lightweight map of inventory_tier (primary vs supplementary)
        # for each candidate_id, using the curator's candidate_inventory context.
        inventory_tier_map: Dict[str, str] = {}
        inventory_by_id: Dict[str, Dict[str, Any]] = {}
        try:
            inv_list = (curator_context.get("candidate_inventory") or []) if curator_context else []
            for item in inv_list:
                if not isinstance(item, dict):
                    continue
                cid = item.get("candidate_id")
                if cid:
                    inventory_by_id[str(cid)] = item
                tier = item.get("inventory_tier")
                if cid and tier:
                    inventory_tier_map[str(cid)] = str(tier)
        except Exception:
            inventory_tier_map = {}
            inventory_by_id = {}

        if curator_output:
            annotations_map = {}
            if coverage_output and coverage_output.get("annotated_candidates"):
                annotations_map = {
                    ann.get("candidate_id"): ann
                    for ann in coverage_output.get("annotated_candidates", [])
                    if isinstance(ann, dict) and ann.get("candidate_id")
                }
            ranked_list = curator_output.get("ranked_candidates") or []
            for candidate in ranked_list:
                if not isinstance(candidate, dict):
                    continue
                cid = candidate.get("candidate_id")
                ann = annotations_map.get(cid) if cid else None
                if ann:
                    candidate["recommended_treatment"] = ann.get("recommended_treatment")
                    candidate["coverage"] = ann
                else:
                    candidate.setdefault("recommended_treatment", "Fresh")
                if not candidate.get("helper_notes"):
                    candidate["helper_notes"] = candidate.get("notes")

            for candidate in ranked_list:
                if not isinstance(candidate, dict):
                    continue
                if not candidate.get("candidate_id") or not candidate.get("text"):
                    continue
                cid = candidate.get("candidate_id")
                # Attach a compact coverage_status view for this turn
                status = coverage_status_map.get(cid, {
                    "covered_in_current_episode": False,
                    "delegate_covered_snapshot": False,
                    "covered_in_previous_episodes": False,
                    "ban_reason": None,
                })
                # Infer inventory_tier from inventory context, falling back to
                # source if needed.
                tier = inventory_tier_map.get(cid)
                if not tier:
                    src = candidate.get("source")
                    tier = "primary" if src == "JIT" else "supplementary"
                source_item = inventory_by_id.get(str(cid)) or {}
                digest_entry = {
                    "candidate_id": cid,
                    "text": candidate.get("text"),
                    "core_text": source_item.get("core_text") or candidate.get("text"),
                    "source": candidate.get("source"),
                    "inventory_tier": tier,
                    "suitability": candidate.get("suitability"),
                    "stance_function": (
                        self._normalize_stance_function(candidate.get("stance_function"))
                        or self._normalize_stance_function(source_item.get("stance_function"))
                        or self._infer_stance_function(source_item.get("core_text") or candidate.get("text"))
                    ),
                    "recommended_treatment": candidate.get("recommended_treatment"),
                    "suggested_usage": candidate.get("suggested_usage"),
                    "relevance_score": candidate.get("relevance_score"),
                    "novelty_score": candidate.get("novelty_score"),
                    "composite_score": candidate.get("composite_score"),
                    "helper_notes": candidate.get("helper_notes"),
                    "coverage_status": status,
                    "anchor_phrases": source_item.get("anchor_phrases") or [],
                    "framing": source_item.get("framing"),
                }
                digest_entry["residual_contribution_value"] = (
                    (candidate.get("coverage") or {}).get("residual_contribution_value")
                    if isinstance(candidate.get("coverage"), dict)
                    else None
                )
                digest_entry.update(self._candidate_viability_details(digest_entry))
                helper_digest.append(digest_entry)

        speaker_payload = None
        speaker_output = None
        turn_context = curator_context.get("turn_context", {}) if curator_context else {}
        floor_status = turn_context.get("floor_status")
        explicit_cue = turn_context.get("explicit_cue")
        if floor_status and not explicit_cue and working_memory:
            # Build compact delegate content summary from helper_digest (already has coverage merged)
            delegate_content_summary = self._build_delegate_content_summary(helper_digest)
            speaker_payload = self._build_speaker_payload(working_memory, participant_id, delegate_content_summary, curator_payload, snapshot_utterances)
            if speaker_payload:
                speaker_raw = self._invoke_helper_tool("speaker_scorer", speaker_payload)
                speaker_output = self._sanitize_helper_output("speaker_scorer", speaker_raw)
            else:
                self._log_helper_io("speaker_scorer", None, None)

        helper_context: Dict[str, Any] = {
            "curator_payload": curator_payload,
            "curator_output": curator_output,
            "coverage_payload": coverage_payload,
            "coverage_output": coverage_output,
            "speaker_payload": speaker_payload,
            "speaker_output": speaker_output,
            "helper_candidates_digest": helper_digest,
            "helper_global_suggestion": helper_global_suggestion
        }
        # Compute per-step banned candidate ids with reasons for traceability
        try:
            banned_ids, ban_reasons = self._compute_banned_candidates(coverage_output)
            helper_context["banned_candidate_ids"] = list(banned_ids)
            helper_context["ban_reasons"] = ban_reasons  # Maps cid -> reason for ban
            # Explicitly capture ban info in artifacts for traceability
            if self.artifact_tracker and (banned_ids or ban_reasons):
                try:
                    self.artifact_tracker.capture_helper_io(
                        "ban_computation",
                        {"banned_candidate_ids": list(banned_ids), "ban_reasons": ban_reasons},
                        None
                    )
                except Exception:
                    pass
        except Exception:
            helper_context["banned_candidate_ids"] = []
            helper_context["ban_reasons"] = {}

        return helper_context
    def _format_utterances_for_prompt(self, snapshot_utterances: List[Dict]) -> str:
        """
        Format utterances for prompt inclusion.

        Args:
            snapshot_utterances: List of utterance dictionaries

        Returns:
            Formatted string for prompt use
        """
        formatted_lines = []
        for utt in snapshot_utterances:
            speaker = utt.get('speaker_id', 'Unknown')
            content = utt.get('utterance_content', '')
            utt_id = utt.get('utterance_id', 'Unknown')
            formatted_lines.append(f"({speaker}): {content} ({utt_id})")
        return "\n".join(formatted_lines)

    def _build_delegate_judge_payload(
        self,
        controller_output: ControllerOutput,
        helper_digest_data: Optional[List[Any]],
        participant_id: Optional[str] = None,
        snapshot_utterances: Optional[List[Dict]] = None,
        helper_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a minimal payload for the delegate judge to avoid helper instrumentation noise.

        Enrich with represented_user, a tiny evaluation_context (floor/cue hints), and a short
        pre‑cue context slice for better floor/cue and redundancy judgments. This keeps
        the delegate prompt compact while adding high-signal metadata.
        """
        payload: Dict[str, Any] = {
            "action_details": controller_output.action_details.model_dump(),
            "content_plan": (
                controller_output.content_plan.model_dump(exclude_none=True)
                if controller_output.content_plan
                else None
            ),
            "selected_point": controller_output.selected_point,
            "selection_reason": controller_output.selection_reason,
            "active_candidate_id": controller_output.active_candidate_id,
        }

        # Represented user (who the agent simulates)
        if participant_id:
            payload["represented_user"] = participant_id

        include_helper_priors = True
        if isinstance(helper_context, dict) and helper_context.get("uses_helper_priors_in_judge") is False:
            include_helper_priors = False
        if isinstance(helper_context, dict):
            delegate_variant = helper_context.get("delegate_variant")
            if delegate_variant:
                payload["delegate_variant"] = delegate_variant

        active_candidate_text: Optional[str] = None
        active_candidate_source: Optional[str] = None
        if controller_output.active_candidate_id and helper_digest_data:
            for item in helper_digest_data:
                candidate_id = (
                    item.get('candidate_id') if isinstance(item, dict) else getattr(item, 'candidate_id', None)
                )
                if candidate_id == controller_output.active_candidate_id:
                    active_candidate_text = (
                        item.get('text') if isinstance(item, dict) else getattr(item, 'text', None)
                    )
                    active_candidate_source = (
                        item.get('source') if isinstance(item, dict) else getattr(item, 'source', None)
                    )
                    break

        if active_candidate_text:
            candidate_payload = {
                "candidate_id": controller_output.active_candidate_id,
                "text": active_candidate_text,
            }
            if active_candidate_source:
                candidate_payload["source"] = active_candidate_source
            payload["active_candidate"] = candidate_payload

        # Optional: speaker recommendation signals if present in ControllerOutput
        if include_helper_priors and getattr(controller_output, "speaker_recommendation", None):
            payload["speaker_recommendation"] = controller_output.speaker_recommendation
        if include_helper_priors and getattr(controller_output, "speaker_recommendation_confidence", None) is not None:
            payload["speaker_recommendation_confidence"] = controller_output.speaker_recommendation_confidence

        # Evaluation context (floor/cue hints) derived from available helper context
        eval_ctx: Dict[str, Any] = {}
        tc = None
        if helper_context:
            direct_eval_ctx = helper_context.get("evaluation_context") if isinstance(helper_context, dict) else None
            if isinstance(direct_eval_ctx, dict):
                eval_ctx.update(direct_eval_ctx)
            if include_helper_priors:
                # Prefer turn_context from curator payload if present; otherwise speaker payload
                tc = (
                    helper_context.get("curator_payload", {}).get("turn_context")
                    if isinstance(helper_context.get("curator_payload"), dict)
                    else None
                )
                if not tc:
                    tc = (
                        helper_context.get("speaker_payload", {}).get("turn_context")
                        if isinstance(helper_context.get("speaker_payload"), dict)
                        else None
                    )
        if isinstance(tc, dict):
            if tc.get("floor_status") is not None:
                eval_ctx["floor_status"] = tc.get("floor_status")
            if tc.get("explicit_cue") is not None:
                eval_ctx["explicit_cue"] = tc.get("explicit_cue")
            # If a richer last_utterance structure exists under micro_topic, surface its target
            micro = tc.get("micro_topic") if isinstance(tc.get("micro_topic"), dict) else None
            if micro and isinstance(micro.get("last_utterance"), dict):
                target = micro["last_utterance"].get("target_addressed")
                if target:
                    eval_ctx["explicit_cue_target"] = target

        # Add cue anchors from the last observed turn
        last_utt = None
        if snapshot_utterances:
            last_utt = snapshot_utterances[-1] if len(snapshot_utterances) > 0 else None
        if isinstance(last_utt, dict):
            if last_utt.get("utterance_id"):
                eval_ctx["cue_utterance_id"] = last_utt.get("utterance_id")
            if last_utt.get("speaker_id"):
                eval_ctx["cue_speaker"] = last_utt.get("speaker_id")
            if last_utt.get("utterance_content") or last_utt.get("content"):
                eval_ctx["cue_text"] = last_utt.get("utterance_content") or last_utt.get("content")

        # Attach coverage assessment for the active candidate (if any)
        try:
            coverage_output = (helper_context or {}).get("coverage_output") if include_helper_priors and isinstance(helper_context, dict) else None
            active_cid = controller_output.active_candidate_id
            if coverage_output and active_cid:
                anns = coverage_output.get("annotated_candidates") or []
                selected_ann = None
                for ann in anns:
                    if isinstance(ann, dict) and ann.get("candidate_id") == active_cid:
                        selected_ann = ann
                        break
                if selected_ann:
                    # Build compact assessment object with truncation/limits
                    assessment: Dict[str, Any] = {}
                    # selected candidate synopsis
                    sel: Dict[str, Any] = {"candidate_id": active_cid}
                    if active_candidate_text:
                        txt = (active_candidate_text or "").strip()
                        if len(txt) > 180:
                            txt = txt[:177] + "..."
                        sel["text"] = txt
                    if active_candidate_source:
                        sel["source"] = active_candidate_source
                    assessment["selected_candidate"] = sel
                    # coverage evaluator signals
                    cov_ev: Dict[str, Any] = {}
                    rt = selected_ann.get("recommended_treatment")
                    if rt:
                        cov_ev["recommended_treatment"] = rt
                    expl = selected_ann.get("coverage_explanation")
                    if isinstance(expl, str) and expl:
                        if len(expl) > 160:
                            expl = expl[:157] + "..."
                        cov_ev["coverage_explanation"] = expl
                    if selected_ann.get("requires_historical_framing") is not None:
                        cov_ev["requires_historical_framing"] = bool(selected_ann.get("requires_historical_framing"))
                    sfa = selected_ann.get("suggested_framing_approaches") or []
                    if isinstance(sfa, list) and sfa:
                        cov_ev["suggested_framing_approaches"] = [str(x) for x in sfa[:2]]
                    if cov_ev:
                        assessment["coverage_evaluator"] = cov_ev
                    # coverage sources (trim each list)
                    srcs = (selected_ann.get("coverage_sources") or {}) if isinstance(selected_ann.get("coverage_sources"), dict) else {}
                    if srcs:
                        assessment["coverage_sources"] = {
                            "snapshot_matches": list((srcs.get("snapshot_matches") or [])[:2]),
                            "topic_log_matches": list((srcs.get("topic_log_matches") or [])[:2]),
                            "episodic_matches": list((srcs.get("episodic_matches") or [])[:2]),
                            "cross_episode_matches": list((srcs.get("cross_episode_matches") or [])[:2]),
                        }
                        # Derive simple counts for judge convenience
                        win_ct = len(srcs.get("snapshot_matches") or []) + len(srcs.get("topic_log_matches") or []) + len(srcs.get("episodic_matches") or [])
                        earlier_ct = len(srcs.get("episodic_matches") or [])
                        cross_ct = len(srcs.get("cross_episode_matches") or [])
                        assessment["seen_counts"] = {
                            "window": win_ct,
                            "earlier": earlier_ct,
                            "cross": cross_ct,
                            "total": win_ct + cross_ct,
                        }
                    if assessment:
                        eval_ctx["coverage_assessment"] = assessment
        except Exception:
            # Non-fatal; keep payload compact if anything goes wrong here
            pass

        if eval_ctx:
            payload["evaluation_context"] = eval_ctx

        # Short raw context slice (t−N..t)
        if snapshot_utterances:
            try:
                n = 6  # conservative default
                window = snapshot_utterances[-n:]
                # Render as lines to keep prompt simple
                rendered = self._format_utterances_for_prompt(window)
                payload["context_last_n_lines"] = rendered.split("\n") if rendered else []
            except Exception:
                pass

        # Optional: include brief JIT provenance when active candidate comes from JIT
        try:
            if payload.get("active_candidate") and payload.get("active_candidate", {}).get("source"):
                if str(payload["active_candidate"].get("source")).upper() == "JIT":
                    briefing_points = []
                    # Extract JIT points from candidate_inventory in helper_context
                    if helper_context and isinstance(helper_context.get("curator_payload"), dict):
                        curator_ctx = helper_context["curator_payload"].get("context", {})
                        candidate_inv = curator_ctx.get("candidate_inventory", [])
                        briefing_points = [
                            c.get("text") or c.get("core_text", "")
                            for c in candidate_inv
                            if isinstance(c, dict) and c.get("source") == "JIT"
                        ][:5]  # Cap at 5 JIT points
                    # Fallback to using the candidate text if no briefing list is available
                    if not briefing_points and payload["active_candidate"].get("text"):
                        briefing_points = [payload["active_candidate"]["text"]]
                    # Trim to at most 2 short bullets (~140 chars)
                    trimmed = []
                    for s in briefing_points[:2]:
                        s = (s or "").strip()
                        if len(s) > 140:
                            s = s[:137] + "..."
                        if s:
                            trimmed.append(s)
                    if trimmed:
                        payload["briefing_support"] = trimmed
        except Exception:
            pass

        # Remove None values to keep the payload compact
        return {key: value for key, value in payload.items() if value is not None}

    def controller_phase(
        self, 
        smm: SharedMemoryModule, 
        snapshot_utterances: List[Dict],
        participant_id: Optional[str] = None,
        helper_context: Optional[Dict[str, Any]] = None
    ) -> Optional[ControllerOutput]:
        """
        Execute the Controller phase: strategic decision making.
        
        Args:
            smm: SharedMemoryModule with current agent state
            snapshot_utterances: Current conversation snapshot
            participant_id: Participant identifier for persona-specific prompts
            helper_context: Structured outputs from helper tools (may be None during scaffolding)

        Returns:
            ControllerOutput with strategic decision or None if failed
        """
        print("  [UM] Controller invoked...")

        # 1. Gather inputs from SMM
        profile = smm.get_user_profile()
        working_memory = smm.get_working_memory()
        # Note: briefing and episodic_memory not fetched for Controller
        # (helpers already process these via Curator/Coverage)

        if not profile or not working_memory:
            print("  [UM] WARNING: Controller failed: Missing user profile or working memory.")
            return None

        # 2. Format the inputs for the prompt
        profile_str = json.dumps(profile.model_dump(), indent=2)
        # Note: JIT briefing and episodic memory are NOT passed to Controller.
        # Helpers (Curator, Coverage) already process these and surface relevant info
        # via helper_digest (with coverage flags like ban_reason, requires_historical_framing).
        # Build compact EIM view (drops topic_summary_log and full participant_models)
        eim_view = self._build_controller_eim_view(working_memory)
        wm_str = json.dumps(eim_view, indent=2)
        snapshot_str = self._format_utterances_for_prompt(snapshot_utterances)

        helper_context = helper_context or {}
        curator_output = helper_context.get("curator_output")
        coverage_output = helper_context.get("coverage_output")
        speaker_output = helper_context.get("speaker_output")
        helper_digest_data = helper_context.get("helper_candidates_digest")
        banned_ids: Set[str] = set(helper_context.get("banned_candidate_ids") or [])
        # No helper_global_suggestion fed into Controller; Controller decides timing

        def _safe_json(data: Any) -> str:
            if data is None:
                return "null"
            try:
                return json.dumps(data, indent=2, ensure_ascii=False)
            except Exception:
                return "null"

        curated_json = _safe_json(curator_output)
        coverage_json = _safe_json(coverage_output)
        speaker_json = _safe_json(speaker_output)
        helper_digest_json = _safe_json(helper_digest_data)
        # Do not surface curator/global suggestion; Controller is the authority on timing

        # Build unified helper summary
        helper_summary = self._build_helper_summary(
            curator_output=curator_output,
            coverage_output=coverage_output,
            speaker_output=speaker_output,
            helper_digest_data=helper_digest_data,
            banned_candidate_ids=banned_ids
        )
        helper_summary_json = _safe_json(helper_summary)

        # Build recalibration guidance section (if available)
        guidance_section = ""
        if smm.recalibration_guidance:
            g = smm.recalibration_guidance
            lines = [
                "## Recalibration Advisory",
                "",
                "**What this is**: Backward-looking feedback from the recalibrator analyzing how prior turn(s) went, synthesized from GT alignment, delegate errors, and snapshot analysis.",
                "**How to use it**: Treat as soft steering signals—factor them into your decision alongside helper signals, but never override floor rules, explicit cues, or social appropriateness.",
                "",
                "### Field Meanings:",
                "- **contribution_pressure**: dial-up = seek contribution opportunities | dial-down = lean toward yielding if nothing significant to contribute | neutral = follow normal helper signals",
                "- **contribution_balance**: under-contributing = you've been too quiet, look for openings | over-contributing = you've dominated, yield more | balanced = appropriate level",
                "- **post_contribution_cooldown**: true = you just contributed recently, consider yielding for 1-2 turns",
                "- **last_action_outcome**: How your last action was evaluated (matched, missed_opportunity, off_topic, appropriate_silence, etc.)",
                "- **error_correction**: Specific correction if you made an error (e.g., off-topic content, floor violation)",
                "- **strategy_adjustment**: Tactical advice for upcoming turns",
                "",
                "### Current Signals:",
            ]
            
            pressure = g.get("contribution_pressure")
            if pressure:
                hint_map = {
                    "dial-up": "opportunities remain, be alert for contribution openings",
                    "dial-down": "lean toward yielding unless explicitly cued",
                    "neutral": "follow normal helper signals"
                }
                lines.append(f"- **Contribution Pressure**: {pressure} ({hint_map.get(pressure, '')})")
            
            if g.get("post_contribution_cooldown"):
                lines.append("- **Cooldown**: Active (recently contributed, consider yielding)")
            
            balance = g.get("contribution_balance")
            if balance:
                balance_map = {
                    "under-contributing": "you've been too quiet, look harder for viable opportunities",
                    "over-contributing": "you've dominated the conversation, yield more",
                    "balanced": "appropriate participation level"
                }
                lines.append(f"- **Balance**: {balance} ({balance_map.get(balance, '')})")
            
            if g.get("error_correction"):
                lines.append(f"- **Correction**: {g['error_correction']}")
            
            if g.get("strategy_adjustment"):
                lines.append(f"- **Strategy**: {g['strategy_adjustment']}")
            
            if g.get("last_action_outcome"):
                lines.append(f"- **Last Outcome**: {g['last_action_outcome']}")
            
            if g.get("guidance_summary"):
                lines.append(f"")
                lines.append(f"**Summary**: {g['guidance_summary']}")
            
            guidance_section = "\n".join(lines) + "\n\n"

        filled_prompt = self.controller_user_prompt_template.format(
            person_name=participant_id or profile.inferred_role_and_goals.primary_role,
            profile_string=profile_str,
            meeting_state_vector_string=wm_str,
            snapshot_string=snapshot_str,
            helper_summary_json=helper_summary_json
        )
        
        # Append guidance section after the template content
        if guidance_section:
            filled_prompt = filled_prompt + "\n\n" + guidance_section



        # 3. Call the LLM
        messages = [
            {"role": "system", "content": self.controller_system_prompt},
            {"role": "user", "content": filled_prompt}
        ]

        try:
            # Get JSON response using structured output (critical for GPT-5-mini)
            # Use ControllerLLMOutput (reduced schema) to avoid forcing LLM to output programmatic fields
            json_response = self.client.call_with_retry(
                model=self.model_name,
                messages=messages,
                max_tokens=4000,
                temperature=0.1,
                json_required=True,
                response_model=ControllerLLMOutput,  # Reduced schema for LLM output only
                telemetry_label="controller",
                telemetry_phase="floor",
            )
            
            if not json_response:
                print("  [UM] WARNING: No JSON response from controller phase")
                return None
            
            # Map LLM output to full ControllerOutput
            if hasattr(json_response, 'model_dump'):
                # Structured output - map ControllerLLMOutput to ControllerOutput
                controller_output = ControllerOutput(
                    action_details=json_response.action_details,
                    content_plan=json_response.content_plan,
                    active_candidate_id=json_response.active_candidate_id,
                    candidate_decision_trace=json_response.candidate_decision_trace,
                    # Programmatic fields (selected_point, selection_reason, helper_candidates_digest, etc.)
                    # will be populated below in the existing code
                )
            else:
                # Fallback: raw JSON, validate against full ControllerOutput
                controller_output = self.client.validate_json_response(json_response, ControllerOutput)
            if controller_output:
                inconsistent, best_candidate, veto_reason = self._controller_silence_inconsistent(
                    controller_output,
                    helper_summary=helper_summary,
                    snapshot_utterances=snapshot_utterances,
                    participant_id=participant_id,
                )
                if inconsistent and best_candidate:
                    try:
                        repair_prompt = (
                            "Controller consistency check: you chose Remain_Silent, but helper outputs contain "
                            "a viable candidate and no real silence veto was found. Re-evaluate once. If you still "
                            "remain silent, cite a concrete veto such as represented-user just spoke, floor directed "
                            "elsewhere, all candidates low fit, or strong speaker-scorer deferral.\n\n"
                            f"Best viable candidate:\n{json.dumps(best_candidate, ensure_ascii=False, indent=2)}"
                        )
                        repair_response = self.client.call_with_retry(
                            model=self.model_name,
                            messages=messages + [{"role": "user", "content": repair_prompt}],
                            max_tokens=2200,
                            temperature=0.1,
                            json_required=True,
                            response_model=ControllerLLMOutput,
                            telemetry_label="controller.consistency_retry",
                            telemetry_phase="floor",
                        )
                        if repair_response:
                            if hasattr(repair_response, "model_dump"):
                                repaired_output = ControllerOutput(
                                    action_details=repair_response.action_details,
                                    content_plan=repair_response.content_plan,
                                    active_candidate_id=repair_response.active_candidate_id,
                                    candidate_decision_trace=repair_response.candidate_decision_trace,
                                )
                            else:
                                repaired_output = self.client.validate_json_response(repair_response, ControllerOutput)
                            repaired_inconsistent, _, repaired_veto = self._controller_silence_inconsistent(
                                repaired_output,
                                helper_summary=helper_summary,
                                snapshot_utterances=snapshot_utterances,
                                participant_id=participant_id,
                            )
                            if not repaired_inconsistent:
                                controller_output = repaired_output
                                if repaired_veto:
                                    note = f"Controller consistency retry retained silence with veto: {repaired_veto}."
                                    controller_output.candidate_decision_trace = (
                                        f"{controller_output.candidate_decision_trace} {note}".strip()
                                        if controller_output.candidate_decision_trace
                                        else note
                                    )
                    except Exception as err:
                        print(f"  [UM] Controller consistency retry failed: {err}")
                inconsistent, best_candidate, veto_reason = self._controller_silence_inconsistent(
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
                # Hard guard: do not allow banned candidates (snapshot / same-episode) to drive actions
                try:
                    if controller_output.active_candidate_id and controller_output.active_candidate_id in banned_ids:
                        note = "Hard override: candidate was already covered in current snapshot/episode."
                        self._record_controller_override(
                            controller_output,
                            reason="candidate_already_covered",
                            action_name="Remain_Silent",
                            justification="Final action: Remain_Silent. Programmatic guard suppressed speech because the selected candidate was already covered in the current snapshot or episode.",
                            content_plan=None,
                            active_candidate_id=None,
                            selected_point=None,
                            trace_note=note,
                        )
                except Exception:
                    pass
                try:
                    self._apply_participant_stance_consistency_guard(
                        controller_output,
                        helper_summary=helper_summary,
                    )
                except Exception:
                    pass
                def _build_digest_models(raw_list):
                    models = []
                    for item in raw_list or []:
                        if isinstance(item, ControllerOutput.CandidateDigestItem):
                            models.append(item)
                        elif isinstance(item, dict):
                            filtered = {
                                'candidate_id': item.get('candidate_id'),
                                'text': item.get('text'),
                                'source': item.get('source'),
                                'suitability': item.get('suitability'),
                                'stance_function': item.get('stance_function'),
                                'recommended_treatment': item.get('recommended_treatment'),
                                'relevance_score': item.get('relevance_score'),
                                'novelty_score': item.get('novelty_score'),
                                'helper_notes': item.get('helper_notes'),
                                'suggested_usage': item.get('suggested_usage'),
                                'anchor_phrases': item.get('anchor_phrases') or [],
                                'residual_contribution_value': (
                                    item.get('residual_contribution_value')
                                    or (item.get('coverage_status') or {}).get('residual_contribution_value')
                                ),
                                'viability_status': item.get('viability_status'),
                                'viability_reason': item.get('viability_reason'),
                                'blocking_reason': item.get('blocking_reason'),
                            }
                            if filtered['candidate_id'] and filtered['text']:
                                try:
                                    models.append(ControllerOutput.CandidateDigestItem(**filtered))
                                except Exception:
                                    continue
                    return models

                helper_digest_models = _build_digest_models(helper_digest_data)
                if helper_digest_models and not controller_output.helper_candidates_digest:
                    controller_output.helper_candidates_digest = helper_digest_models
                elif controller_output.helper_candidates_digest:
                    existing_models = _build_digest_models(controller_output.helper_candidates_digest)
                    if existing_models:
                        controller_output.helper_candidates_digest = existing_models
                # Do not copy helper_global_suggestion into controller_output
                if speaker_output:
                    recommended = speaker_output.get('recommended_speaker')
                    if recommended and not controller_output.speaker_recommendation:
                        controller_output.speaker_recommendation = recommended
                    confidence = speaker_output.get('confidence')
                    if confidence is not None and controller_output.speaker_recommendation_confidence is None:
                        try:
                            controller_output.speaker_recommendation_confidence = float(confidence)
                        except (TypeError, ValueError):
                            pass
                if (not controller_output.active_candidate_id and controller_output.selected_point and helper_digest_data):
                    selected_normalized = controller_output.selected_point.strip().lower()
                    for item in helper_digest_data:
                        if isinstance(item, dict):
                            candidate_text = (item.get('text') or '').strip().lower()
                            candidate_id = item.get('candidate_id')
                        else:
                            candidate_text = str(item.text or '').strip().lower()
                            candidate_id = item.candidate_id
                        if candidate_text and candidate_text == selected_normalized and candidate_id:
                            # Respect per-step banned ids even when inferring by text
                            if candidate_id not in banned_ids:
                                controller_output.active_candidate_id = candidate_id
                            break
                print("  [UM] Controller decision received.")
                
                # CAPTURE CONTROLLER OUTPUT ARTIFACT
                if self.artifact_tracker:
                    self.artifact_tracker.capture_controller_output(controller_output)
                
                # UPDATE COVERAGE FLAGS (Phase 3 Enhancement)
                # Track which candidate was used to enable O(1) coverage lookups in future turns
                if controller_output.active_candidate_id:
                    candidate_text = None
                    candidate_source = None
                    # Extract candidate text from helper digest
                    for item in helper_digest_data:
                        item_id = item.get('candidate_id') if isinstance(item, dict) else getattr(item, 'candidate_id', None)
                        if item_id == controller_output.active_candidate_id:
                            candidate_text = item.get('text') if isinstance(item, dict) else getattr(item, 'text', None)
                            candidate_source = item.get('source') if isinstance(item, dict) else getattr(item, 'source', None)
                            break
                    if candidate_text:
                        self._update_candidate_coverage(
                            candidate_id=controller_output.active_candidate_id,
                            candidate_text=candidate_text,
                            utterance_id=working_memory.last_processed_utterance_id if working_memory else "unknown",
                            source=candidate_source
                        )
                
                # Ensure helper_global_suggestion reflects the Controller's final action
                try:
                    act = (controller_output.action_details.action_name or "").strip()
                    if not controller_output.helper_global_suggestion:
                        if act == "Remain_Silent":
                            controller_output.helper_global_suggestion = "Remain_Silent"
                        elif act == "Acknowledge":
                            controller_output.helper_global_suggestion = "Acknowledge"
                        else:
                            controller_output.helper_global_suggestion = "Speak"
                except Exception:
                    pass
                return controller_output
            else:
                print("  [UM] WARNING: Controller JSON validation failed")
                return None
        except (ValidationError, Exception) as e:
            print(f"  [UM] WARNING: Controller phase failed: {e}")
            return None

    def generator_phase(
        self, 
        controller_output: ControllerOutput,
        profile, 
        snapshot_utterances: List[Dict]
    ) -> Optional[GeneratorOutput]:
        """
        Execute the Generator phase: response generation.
        
        Args:
            controller_output: Strategic decision from Controller
            profile: User profile for style information
            snapshot_utterances: Current conversation context
            
        Returns:
            GeneratorOutput with final response or None if failed
        """
        print("  [UM] Generator invoked...")

        # Ensure content_plan exists before accessing its fields
        if not controller_output.content_plan:
            print("  [UM] WARNING: Generator failed: content_plan is null for a non-silent action.")
            return None

        snapshot_str_for_generator = self._format_utterances_for_prompt(snapshot_utterances)

        filled_prompt = self.generator_user_prompt_template.format(
            semantic_intent=controller_output.content_plan.semantic_intent,
            knowledge_to_convey=controller_output.content_plan.knowledge_to_convey or "N/A",
            concepts_to_frame=controller_output.content_plan.concepts_to_frame or "N/A",
            formality=profile.communication_style.formality,
            verbosity=profile.communication_style.verbosity,
            typical_tone=profile.communication_style.typical_tone,
            jargon_usage=profile.communication_style.jargon_usage,
            common_fillers=profile.communication_style.common_fillers_catchphrases or "None",
            snapshot_string=snapshot_str_for_generator,
            scope=(controller_output.action_details.scope or "Concise")
        )

        messages = [
            {"role": "system", "content": self.generator_system_prompt},
            {"role": "user", "content": filled_prompt}
        ]

        try:
            # Get JSON response using structured output (critical for GPT-5-mini)
            json_response = self.client.call_with_retry(
                model=self.model_name,
                messages=messages,
                max_tokens=2000,  # Increased for GPT-5-mini reasoning overhead (was 500)
                temperature=0.3,
                json_required=True,
                response_model=GeneratorOutput,  # Structured output for GPT-5-mini/Gemini
                telemetry_label="generator",
                telemetry_phase="floor",
            )
            
            if not json_response:
                print("  [UM] WARNING: No JSON response from generator phase")
                return None
            
            # Handle both Pydantic model (structured output) and raw JSON (fallback)
            if hasattr(json_response, 'model_dump'):
                generator_output = json_response  # Already a GeneratorOutput
            else:
                # Manually validate the JSON against the generator output schema (fallback)
                generator_output = self.client.validate_json_response(json_response, GeneratorOutput)
            
            if generator_output:
                if self._suppress_unsupported_numeric_generation(
                    controller_output,
                    generator_output,
                    profile,
                    snapshot_utterances=snapshot_utterances,
                ):
                    print("  [UM] Generator utterance suppressed by numeric grounding guard.")
                else:
                    print("  [UM] Generator produced utterance.")
                return generator_output
            else:
                print("  [UM] WARNING: Generator JSON validation failed")
                return None
        except (ValidationError, Exception) as e:
            print(f"  [UM] WARNING: Generator phase failed: {e}")
            return None

    def run_turn(
        self, 
        smm: SharedMemoryModule, 
        snapshot_utterances: List[Dict],
        participant_id: Optional[str] = None,
        gt_next_window: Optional[List[Dict]] = None,
        episode: Optional[object] = None,
        delta_index: int = 0,  # Delta step within episode for cooldown tracking
    ) -> Optional[DelegateResponse]:
        """
        Execute a complete turn using the two-tier architecture.
        
        This is the main method that orchestrates the Controller and Generator phases
        to produce a complete agent response.
        
        Args:
            smm: SharedMemoryModule containing agent state
            snapshot_utterances: Current conversation snapshot to respond to
            
        Returns:
            DelegateResponse with complete turn results or None if failed
        """
        print("  [UM] Starting two-tier turn execution...")
        floor_path_started = time.perf_counter()
        self.last_controller_output = None
        self.last_generator_output = None
        self.last_delegate_judge_payload = None
        self.last_helper_context = None
        self.last_helper_digest_data = None

        # Handle CAPA episode boundary for coverage flags (if episode info is provided)
        prev_episode = self._current_episode_id
        self._update_episode_boundary(episode)
        # Reset episode-level tracking if new episode started
        if episode and self._current_episode_id != prev_episode and prev_episode is not None:
            smm.reset_episode_tracking()
            smm.recalibration_guidance = None  # Clear stale guidance from prior episode

        # Refresh snapshot_analysis for current window (CAPA mode), leak-safe
        # Skip when recalibrator drives state to avoid duplication
        if self.capa_enabled and snapshot_utterances and not getattr(self, 'capa_recalibrator_drives_state', False):
            try:
                wm0 = smm.get_working_memory() if smm else None
                # Skip redundant refresh exactly once at episode start when WM was just seeded for cue
                skip_once = False
                try:
                    seeded_now = smm.consume_wm_seeded_flag()
                    if seeded_now and wm0 and snapshot_utterances:
                        last_id = snapshot_utterances[-1].get('utterance_id')
                        if last_id and getattr(wm0, 'last_processed_utterance_id', None) == last_id:
                            skip_once = True
                except Exception:
                    skip_once = False

                if skip_once:
                    print("  [UM] Snapshot refresh skipped at cue (already up-to-date).")
                else:
                    eim_step = EnvironmentInterpretationModule(self.client)
                    if wm0:
                        # Convert Pydantic objects to dictionaries before creating new EIMMasterState
                        topic_log_dict = {}
                        if hasattr(wm0, 'topic_summary_log') and wm0.topic_summary_log:
                            topic_log_dict = {k: v.model_dump() if hasattr(v, 'model_dump') else v 
                                            for k, v in wm0.topic_summary_log.items()}
                        
                        participant_models_dict = {}
                        if hasattr(wm0, 'participant_models') and wm0.participant_models:
                            participant_models_dict = {k: v.model_dump() if hasattr(v, 'model_dump') else v 
                                                     for k, v in wm0.participant_models.items()}
                        
                        eim_step.master_state = EIMMasterState(
                            topic_summary_log=topic_log_dict,
                            participant_models=participant_models_dict,
                            identity_aliases=getattr(wm0, 'identity_aliases', {}) or {}
                        )
                    fresh_vec = eim_step.analyze_snapshot(snapshot_utterances)
                    if fresh_vec:
                        smm.update_working_memory(fresh_vec)
            except Exception as e:
                print(f"  [UM] Warning: snapshot refresh failed: {e}")

        # Hard yield-after-self gate: if last speaker was the represented user,
        # with no explicit cue, we will force a silent action. We still want
        # CAPA judges and the recalibrator to run, so we construct a synthetic
        # controller decision rather than returning early.
        forced_yield = False
        yield_reason = "Yield-after-self rule: last speaker was me and no explicit cue exists for me now; remaining silent."
        try:
            if participant_id and snapshot_utterances:
                if self._should_yield_after_self(smm, participant_id, snapshot_utterances):
                    forced_yield = True
        except Exception:
            # Fail-safe: if gate computation fails, continue with normal flow
            forced_yield = False

        # Phase 1: Controller - Strategic Decision Making
        helper_context: Dict[str, Any] = {}
        helper_digest_data = None

        if not forced_yield:
            helper_context = self._run_helper_sequence(
                smm,
                snapshot_utterances,
                participant_id,
            )
            helper_digest_data = helper_context.get("helper_candidates_digest") if helper_context else None
            controller_output = self.controller_phase(smm, snapshot_utterances, participant_id, helper_context)
            if not controller_output:
                self.last_helper_context = helper_context
                self.last_helper_digest_data = helper_digest_data
                return DelegateResponse(thoughts="Controller phase failed.", response=None)
        else:
            # Construct a minimal synthetic controller output that encodes a
            # forced silent decision due to yield-after-self. This allows CAPA
            # judges and the recalibrator to evaluate "Remain_Silent" vs GT.
            controller_output = ControllerOutput(
                content_plan=None,
                action_details=ActionDetails(
                    action_name="Remain_Silent",
                    scope="Concise",
                    justification=yield_reason,
                ),
            )
        self.last_helper_context = helper_context
        self.last_helper_digest_data = helper_digest_data
        self.last_controller_output = controller_output

        # Phase 2: Generator - Stylistic Response Generation
        generator_output = None
        if controller_output.action_details.action_name != "Remain_Silent":
            profile = smm.get_user_profile()
            if profile:
                generator_output = self.generator_phase(controller_output, profile, snapshot_utterances)
            else:
                print("  [UM] WARNING: Cannot run Generator: User profile is missing.")
        else:
            # If the action is to remain silent, create a dummy generator output
            generator_output = GeneratorOutput(utterance="")
        self.last_generator_output = generator_output
        record_span = getattr(self.client, "record_telemetry_span", None)
        if callable(record_span):
            record_span(
                "floor_path",
                time.perf_counter() - floor_path_started,
                phase="floor",
                extra={
                    "action_name": controller_output.action_details.action_name,
                    "spoke": bool(generator_output and str(generator_output.utterance or "").strip()),
                },
            )

        # Capture step context + delegate utterance for eval artifacts (before finalize in driver)
        if self.artifact_tracker:
            try:
                if gt_next_window:
                    self.artifact_tracker.capture_step_context(gt_next_window, participant_id)
                self.artifact_tracker.capture_delegate_utterance(generator_output.utterance if generator_output else "")
            except Exception:
                pass

        # Phase 3: CAPA evaluation (optional, perception-first)
        # Assumption: EIM has already analyzed the snapshot externally and SMM.working_memory reflects the current turn (perception-first order).
        delegate_judge_payload = None
        if controller_output:
            delegate_judge_payload = self._build_delegate_judge_payload(
                controller_output,
                helper_digest_data,
                participant_id=participant_id,
                snapshot_utterances=snapshot_utterances,
                helper_context=helper_context,
            )
        self.last_delegate_judge_payload = delegate_judge_payload

        if self.capa_enabled and smm.get_working_memory() and snapshot_utterances:
            try:
                # Prepare inputs for CAPA driver
                eim_state = smm.get_working_memory()
                last_utterances = snapshot_utterances
                current_gt_turn = (gt_next_window[0] if gt_next_window else None)

                def apply_patch_fn(diff: EIMMasterState, current_id: str = None, new_snapshot_analysis=None, recompute_snapshot: bool = False) -> bool:
                    # Merge diff into a local EIM instance to reuse merge logic & guards
                    eim_local = EnvironmentInterpretationModule(self.client)
                    # Seed master_state from current working memory
                    wm = smm.get_working_memory()
                    
                    # Convert Pydantic objects to dictionaries before creating new EIMMasterState
                    topic_log_dict = {}
                    if hasattr(wm, 'topic_summary_log') and wm.topic_summary_log:
                        topic_log_dict = {k: v.model_dump() if hasattr(v, 'model_dump') else v 
                                        for k, v in wm.topic_summary_log.items()}
                    
                    participant_models_dict = {}
                    if hasattr(wm, 'participant_models') and wm.participant_models:
                        participant_models_dict = {k: v.model_dump() if hasattr(v, 'model_dump') else v 
                                                 for k, v in wm.participant_models.items()}
                    
                    eim_local.master_state = EIMMasterState(
                        topic_summary_log=topic_log_dict,
                        participant_models=participant_models_dict,
                        identity_aliases=getattr(wm, 'identity_aliases', {}) or {}
                    )
                    ok = eim_local.apply_patch(diff, current_turn_utterance_id=current_id)
                    if not ok:
                        return False
                    # Build updated full state vector (reuse existing snapshot_analysis)
                    # Determine new snapshot_analysis
                    snapshot_analysis_dict = None
                    if new_snapshot_analysis is not None:
                        try:
                            snapshot_analysis_dict = new_snapshot_analysis.model_dump() if hasattr(new_snapshot_analysis, 'model_dump') else new_snapshot_analysis
                        except Exception:
                            snapshot_analysis_dict = None
                    if snapshot_analysis_dict is None and recompute_snapshot:
                        try:
                            # Recompute on fallback using current snapshot + observed GT turn
                            analysis_snapshot = list(snapshot_utterances)
                            if gt_next_window and isinstance(gt_next_window[0], dict):
                                analysis_snapshot = analysis_snapshot + [gt_next_window[0]]
                            eim_step = EnvironmentInterpretationModule(self.client)
                            # Seed to reuse master_state for analysis
                            eim_step.master_state = eim_local.master_state
                            vec = eim_step.analyze_snapshot(analysis_snapshot)
                            if vec is not None:
                                snapshot_analysis_dict = vec.snapshot_analysis.model_dump() if hasattr(vec.snapshot_analysis, 'model_dump') else vec.snapshot_analysis
                        except Exception:
                            snapshot_analysis_dict = None
                    if snapshot_analysis_dict is None:
                        snapshot_analysis_dict = wm.snapshot_analysis
                        if hasattr(wm.snapshot_analysis, 'model_dump'):
                            snapshot_analysis_dict = wm.snapshot_analysis.model_dump()
                    
                    updated_vector = EIMMeetingStateVector(
                        last_processed_utterance_id=(current_id or wm.last_processed_utterance_id),
                        snapshot_analysis=snapshot_analysis_dict,
                        topic_summary_log=eim_local.master_state.topic_summary_log,
                        participant_models=eim_local.master_state.participant_models,
                        identity_aliases=eim_local.master_state.identity_aliases or {}
                    )
                    smm.update_working_memory(updated_vector)
                    return True

                # Compute tracking info for guidance
                idea_target_count = 0
                if episode and hasattr(episode, 'idea_units') and episode.idea_units:
                    idea_target_count = len(episode.idea_units)
                matched_count, _ = smm.get_idea_unit_coverage(idea_target_count)
                # Get current step index (use utterance count as proxy for step-based tracking)
                current_step = len(snapshot_utterances) if snapshot_utterances else 0
                # Use delta-based cooldown for proper expiration within episodes
                is_cooldown = smm.is_in_cooldown_delta(delta_index)
                contribution_count = smm.episode_contribution_count

                # Invoke CAPA driver with controller/generator outputs for delegate judge
                capa_diff = self._capa_driver.step(
                    eim_state=eim_state,
                    last_utterances=last_utterances,
                    current_gt_turn=current_gt_turn,
                    apply_patch_fn=apply_patch_fn,
                    episode=episode,
                    controller_output=delegate_judge_payload,
                    generator_output=generator_output,
                    gt_window=gt_next_window,
                    artifact_tracker=self.artifact_tracker,
                    apply_guidance_fn=(lambda bundle: setattr(smm, 'recalibration_guidance', bundle.model_dump() if hasattr(bundle, 'model_dump') else bundle)),
                    matched_unit_count=matched_count,
                    total_unit_count=idea_target_count,
                    is_in_cooldown=is_cooldown,
                    episode_contribution_count=contribution_count,
                    record_match_fn=lambda unit_id: smm.record_idea_unit_match(unit_id),
                    participant_id=participant_id,
                    matched_idea_unit_ids=smm.matched_idea_units,  # For anchor coverage check
                )
                if capa_diff:
                    print("  [UM] CAPA patch merged into Working Memory.")
                
                # Record delegate contribution for cooldown tracking only if MATCHED
                # (not just if spoke - cooldown should mean "just made a valuable contribution")
                new_matched_count, _ = smm.get_idea_unit_coverage(idea_target_count)
                if new_matched_count > matched_count:
                    # A new idea unit was matched this step - record with delta_index for cooldown
                    smm.record_delegate_contribution(current_step, delta_index=delta_index)
            except Exception as e:
                print(f"  [UM] WARNING: CAPA step failed: {e}")

        # Phase 4: Final Assembly and Logging
        final_response = generator_output.utterance if generator_output else None

        delegate_response = DelegateResponse(
            thoughts=controller_output.action_details.justification,
            response=final_response
        )
        
        # Log the action to episodic memory (anchor to the current decision point)
        if snapshot_utterances:
            action_anchor_utt_id = None
            if gt_next_window and len(gt_next_window) > 0:
                action_anchor_utt_id = gt_next_window[0].get("utterance_id")
            if not action_anchor_utt_id and snapshot_utterances:
                # Fallback: last utterance in the snapshot window
                action_anchor_utt_id = snapshot_utterances[-1].get("utterance_id", "unknown_cue")

            log_entry = DelegateActionLogEntry(
                utterance_id_of_action=action_anchor_utt_id or "unknown_cue",
                action_taken=controller_output.action_details.action_name,
                point_made_summary=final_response
            )

            ## Experimental: not logging Silence decisions/actions
            if log_entry.action_taken != "Remain_Silent":
              smm.log_delegate_action(log_entry)
            print(f"  [UM] Action '{log_entry.action_taken}' logged against utterance '{log_entry.utterance_id_of_action}'.")

        print("  [UM] Turn execution complete.")
        return delegate_response
