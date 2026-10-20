"""JIT Briefing Generation Module for Meeting Delegate System

This module handles generating just-in-time briefing points from meeting transcript segments.
"""
import json
import time
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import re
import hashlib
from pydantic import ValidationError

# Add current directory to path for robust importing
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from llm_client import LLMClient
from schemas import JITBriefingResponse, JITBriefingPointEvidence


class JITBriefingGenerator:
    """
    JIT briefing generation - extracted from cells 32-35
    Generates session-specific talking points from meeting transcript analysis
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def generate_jit_briefing_for_participant(
        self,
        person_id: str,
        transcript_segment_string: str,
        max_attempts: int = 3,
        base_delay: float = 2.0
    ) -> Optional[JITBriefingResponse]:
        """
        Calls an LLM to extract a concise, user-specific briefing from a transcript segment.
        
        Args:
            person_id: The ID/name of the participant to generate briefing for
            transcript_segment_string: Formatted transcript segment
            max_attempts: Maximum retry attempts for generation
            base_delay: Base delay between retries
            
        Returns:
            Validated JITBriefingResponse or None if failed
        """
        system_prompt = """You are an expert AI analyst. Your task is to create a pre-meeting briefing for a delegate agent that will represent a specific participant.
First, analyze the provided <Transcript Segment> to extract the all of the most important, novel, and substantive contributions (opinions, stances, suggestions, questions, proposals, enquiries, comments, requests, ...) made by that participant.
Second, rephrase these contributions into forward-looking "talking points" or instructions for the delegate agent.

The goal is to produce an objective list of "key talking points" that the specified person introduced or heavily focused on within this specific segment. Essentially, we want a list of actionable talking points that the delegate can use if it were representing this particular participant in this meeting.

## On the provided <Transcript Segment>
- Each utterance is formatted as: (<speaker_id>): <utterance_content> (<utterance_id>)
- You must focus *only* on utterances from the person specified in the user prompt. Use other speakers' utterances only for context.

## Instructions for Extraction:
1.  **Identify Substantive Contributions:** Scan for utterances where the specified person is sharing new facts, stating a strong opinion with reasoning, making a concrete proposal, or asking a significant, on-topic question.
2.  **Filter Out Non-Essential Dialogue:** Ignore simple acknowledgments (e.g., "Okay," "Mm-hmm", ...), greetings, or minor conversational fillers.
3.  **Synthesize contiguous paraphrases only:** If multiple consecutive utterances express the same claim, synthesize into one point. For distinct intents/topics or separated moments, follow the Split/Merge rules.
4.  **Be Objective:** Your task is to extract *what was said*, not to analyze the user's style or overall role. The output should be a list of their key contributions (in a forward-looking manner).

## Participant-Only Content and Specifics (No Leakage)
- Base bullets ONLY on the specified participant's intent and content within the provided segment.
- Use only the specified participant's own utterances for factual content. Participant-owned specifics are allowed only if the participant stated them.
- Do not import facts introduced solely by other speakers unless the participant explicitly re-states them with added stance or role-aligned specifics.
- No clairvoyance: do not infer or invent details not present in this segment.
- Cue hints must be generic (phase/state/topic) and never mention speakers or utterance IDs.
- Topic tags must be 1-3 words with no names/IDs.
- Why-clauses must paraphrase the participant's stated rationale; no new specifics.
- Keep each bullet fully self-contained so the delegate would not misinterpret or misuse it in the wrong context.
- **ANAPHORA RESOLUTION**: If the source utterance contains vague references ("it", "that", "this approach", "the feature"), consult prior utterances and resolve to the SPECIFIC referent.
  - BAD: "the feature should be implemented soon" (which feature?)
  - GOOD: "The OAuth token rotation feature should be implemented soon" (specific, self-contained)
- Names are allowed only if the participant explicitly assigned/mentioned that person (e.g., delegating to Alex); never introduce names that do not appear in the participant's own utterances.

## Must-Capture Categories (if present in this participant's lines)
- Questions the participant intends to ask.
- Proposals/requests the participant intends to make.
- Decisions/commitments the participant owns or initiates.
- Metrics/criteria the participant themselves introduces or enforces (participant-owned specifics allowed).
- Role-aligned stances/assessments with concrete rationale tied to objectives (e.g., user value, risk, feasibility, timeline, ...).

## Topic-Nuclei Coverage (participant lines only)
- Definition: a "topic nucleus" is a distinct theme or claim the participant clearly touches in this segment (e.g., overall stance; interaction ergonomics; visibility/legibility; reliability/latency; power/connectivity), expressed in their own words.
- Identification: find nuclei by scanning only this participant's utterances. Do not infer nuclei that are not clearly present in these lines.
- Coverage rule: include at least one bullet per nucleus present in this participant's lines. Do not include nuclei that are absent.

## Prioritization and Limits
- Minimum coverage rule: include at least one bullet from each Must-Capture Category that actually appears in this participant's own lines (e.g., if they asked a question, include an "Ask" bullet; if they stated a commitment, include a commitment bullet).
- **Cap: Maximum 7-9 important bullets per window.**. If more than 9 distinct items exist, you MUST drop the least critical ones. Include all Must-Capture Category items first, then fill remaining slots with highest-value contributions.
- Exclude phatic content. Keep stances only when tied to a concrete rationale.

## Granularity (Split vs Merge)
- Split into separate bullets when ANY holds:
  - Different intent type: question vs proposal vs decision/commitment vs metric/criteria vs risk/constraint vs result/update vs clarification.
  - Different topical nucleus (distinct theme/goal/feature/metric/risk).
  - General stance vs specifics: keep a general stance bullet if it expresses an independent claim with rationale, and emit separate bullets for distinct specifics. Do not merge stance and specifics unless the stance is a trivial restatement.
  - Distinct risk/constraint nuclei (e.g., different kinds of constraints) must remain separate; do not merge across different nuclei.
  - Temporal/context break (>= 2 intervening speakers OR clearly separate moments in the segment).
  - Combining would exceed two concise sentences or bundles more than one independent claim/action.
- Merge only when ALL hold:
  - Same intent type and the same concrete claim (later lines are paraphrases/minor refinements).
  - Same topical nucleus and evidence (participant's own utterance(s)).
  - Merging does not hide distinct actionable items (owners, numbers, timelines).
- One bullet = one actionable claim. Prefer several precise bullets over a broad catch-all.

## Instructions for Rephrasing into Briefing Points:
**Mandatory format for every bullet (single line):**  
`<forward-looking action and rationale {**only if it's relevant, makes sense and grounded (i.e., covered by the participant in this points context); otherwise, rationale is optional}>. (topic: <1-3 words>[; context: <short generalized pre-move context>])`

1.  **Convert Past Contributions into Future Actions:** Rephrase each extracted point as a concise, actionable instruction for the delegate agent, written as a forward-looking action the delegate could take.
2.  **Use Imperative Verbs:** Start the main sentence with an action verb (e.g., "Ask," "Propose," "Commit," "Document," "Send," "Coordinate," "Affirm," "Highlight," "Summarize"). Do **not** start the main sentence with "If" or "When".
3.  **Frame as Instructions:** The main sentence should sound like a pre-meeting brief (e.g., "Propose that Sebastian should examine the technical design because he can assess feasibility and cost implications.") rather than "The user said X".
4.  **Maintain Core Information:** Keep only participant-owned content; no specifics from other speakers. The main sentence must capture the participant's core idea and stated rationale (the "because" clause) without inventing new facts.
5.  **Topic Tag (required):** End with a compact tag in parentheses: `(topic: <1-3 words>)` with no names/IDs. This tag summarises the main topical focus of the point.
6.  **Context Hint (optional but recommended):** After the topic tag, you may append `; context: <short generalized pre-move context>` inside the same parentheses, for example: `(topic: delegation; context: when the team has agreed on a function but is unsure about technical feasibility and cost)`. This context describes **when** the point is appropriate to use, not the content of the contribution itself.
7.  **Specific & Self-Contained:** Be precise and faithful; include qualifiers needed to avoid ambiguity or misleading the delegate; no hedging or invented context. The main sentence plus the topic/context suffix should be enough for the delegate to understand what to say and roughly when it is relevant.
8.  **Preserve Participant-Owned Specifics:** Keep the concrete feature/metric/task names the participant used (e.g., "sound feature", "dual-channel watch", "teletext"), rather than vague stand-ins like "unique feature" or "advanced option".
9.  **Full Context from Participant:** Include enough participant-stated context (what/which/how/constraint) in the main sentence so the delegate can place the point correctly; never invent or borrow specifics from others.
10. **Length/Shape:** One main sentence for the action, followed by the `(topic: ...[; context: ...])` suffix. Single intent per bullet, <= 200 characters for the main sentence; keep the context hint short (a short clause, not a paragraph).
11. **Optional \"because\" clause (strictly grounded from participant only):**
    - Include a \"because ...\" rationale ONLY when the represented participant explicitly or implicitly stated the reason in their OWN utterances (support_utt_ids).
    - The rationale must be derivable solely from what the participant said, not from other speakers' responses or later utterances.
    - **PROHIBITION**: Never borrow, infer, or backfill rationale from other participants' utterances (context_utt_ids or any other source). If the participant only asked a question or made a statement without stating why, the bullet should have NO \"because\" clause.
    - A simple action without rationale is always better than an attributed rationale that the participant never stated.
    - Example: If participant asks \"Should we add feature X?\" without stating a reason, emit \"Ask whether feature X should be included. (topic: ...)\" — NOT \"Ask whether feature X should be included because it improves Y\" (where Y came from another speaker's response).
- **Names rule:** You may include a name in the main sentence only if the participant themselves used it to assign/acknowledge an owner; otherwise avoid names/IDs.
- Order bullets by transcript order.

## Redundancy & Provenance
- Drop near-duplicates: if two candidate bullets share >80% meaning, keep the one with more specifics (numbers, owners, timelines).
- Use only the represented participant's own utterances for factual content. If another role introduced a fact, include it only if the participant re-states it with stance or adds specifics tied to their role.
- Internal evidence check: include a bullet only if it can be supported by at least one utterance from this participant in the segment.
- Do not drop a nucleus because a related bullet exists; drop only true near-duplicates of the same claim and evidence.
- When deduping, ignore cue hints / why clause / topic tag; dedupe on the core claim and keep the most informative version.
- Maintain fidelity: do not soften, generalize, or add ambiguity that could mislead the delegate about scope, ownership, or timing.

## Example (Output Shape)
Return JSON in the following shape (points_evidence is optional, but preferred when available):
```json
{
  "thoughts": "Engineer_A contributes three nuclei: 30% perf gain; backfill bottleneck + parallelize; ownership of redesign. Each bullet follows the action+rationale+topic/context template. Note: rationale included ONLY where Engineer_A explicitly stated it.",
  "jit_briefing_points": [
    "State that the new DB is consistently 30% faster on read-heavy queries. (topic: performance; context: when asked about performance test results)",
    "Highlight that the backfill script is single-threaded and takes over 12 hours, and propose parallelizing it because it exceeds the maintenance window. (topic: backfill risk; context: when asked about concerns or issues with the migration)",
    "Confirm you will take on redesigning the backfill script. (topic: ownership; context: when asked to take on the backfill work)"
  ],
  "points_evidence": [
    {
      "text": "State that the new DB is consistently 30% faster on read-heavy queries. (topic: performance; context: when asked about performance test results)",
      "support_utt_ids": ["Meeting4_153"],
      "intent_type": "metric",
      "topical_tags": ["performance"],
      "anchor_phrases": ["30% faster", "read-heavy queries"],
      "framing": "introduces",
      "context_utt_ids": ["Meeting4_152"]
    },
    {
      "text": "Highlight that the backfill script is single-threaded and takes over 12 hours, and propose parallelizing it because it exceeds the maintenance window. (topic: backfill risk; context: when asked about concerns or issues with the migration)",
      "support_utt_ids": ["Meeting4_155", "Meeting4_157"],
      "intent_type": "proposal",
      "topical_tags": ["backfill", "maintenance window"],
      "anchor_phrases": ["single-threaded", "12 hours", "exceeds our maintenance window"],
      "framing": "introduces",
      "context_utt_ids": ["Meeting4_154", "Meeting4_156"]
    },
    {
      "text": "Confirm you will take on redesigning the backfill script. (topic: ownership; context: when asked to take on the backfill work)",
      "support_utt_ids": ["Meeting4_159"],
      "intent_type": "decision",
      "topical_tags": ["ownership"],
      "anchor_phrases": ["Yep"],
      "framing": "introduces",
      "context_utt_ids": ["Meeting4_158"]
    }
  ]
}
```
## Illustrative Mini-Scenarios (Generalized)
Use these to understand participant-only specifics, must-capture categories, and phrasing rules. Each includes a short snippet (not tied to any particular meeting) and "Do/Don't" guidance.
### Scenario A - Question vs Borrowed Specifics
Snippet:
(Facilitator): "Before we close, what should our success criteria be?" (T1)
(TargetParticipant): "We should clarify success metrics for each phase." (T2)
(Other): "Let's track completion rate, time-to-first-value, and satisfaction." (T3)

Do: "Propose clarifying success metrics for each phase. (topic: success metrics; context: when success criteria for the work are being discussed)"
Don't: "Propose clarifying metrics including completion rate, time-to-first-value, satisfaction" (those specifics came only from T3).
### Scenario B - Participant-Owned Specifics vs Others' Numbers
Snippet:
(TargetParticipant): "A phased roll-out will reduce operational risk and increase flexibility." (U1)
(Other): "The big-bang approach failed 40% of the time in past projects." (U2)

Do: "Highlight that a phased roll-out reduces operational risk and increases flexibility. (topic: rollout strategy; context: when rollout approaches are being debated)"
Don't: "Cite the 40% failure rate" (that number was introduced only by U2).
### Scenario C - Stance/Assessment vs Phatic
Snippet:
(TargetParticipant): "This plan directly addresses our key user pain points and should improve adoption." (V1)

Do: "Affirm that the plan addresses key user pain points because that drives adoption. (topic: adoption rationale; context: when adoption or user uptake is being discussed)"
Don't: "Say 'Sounds great!'" (phatic; low informational value)
### Scenario D - Question Without Rationale (No Backfilled "Because")
Snippet:
(TargetParticipant): "Should we include the advanced input mode?" (W1)
(Other): "Yes, it would help users navigate more quickly." (W2)
(Other2): "I agree, it improves efficiency significantly." (W3)

Do: "Ask whether the advanced input mode should be included. (topic: feature consideration; context: when discussing input options or feature scope)"
Don't: "Ask whether the advanced input mode should be included because it helps users navigate quickly" (that rationale came from W2/W3, not from TargetParticipant).

**Principle**: If the participant asked a question or made a statement without explicitly stating a reason, the bullet must NOT include a "because" clause backfilled from other speakers' subsequent responses. A clean action without rationale is always preferable to misattributed rationale.

## Testing-Segment and Stance Rules
- Stance Allowed: include the participant's opinions/assessments/decisions on topics introduced by others in this segment when clearly stated in the participant's own utterances.
- No Borrowed Specifics: do not import numbers/lists/details that appear only in other speakers' lines unless the participant restated them.
- Anchor Phrases for Matchability: where helpful, include 1-2 short phrases (2-4 words) the participant actually used inside the bullet text; avoid copying full sentences.
- Consistency: bullets must remain imperative and ≤ 2 sentences; do not insert IDs or tags into the bullet text.

## Contribution Framing and Attribution
- Frame each bullet according to the participant's role in the idea:
  - Introduces (participant clearly originates a new idea in this segment): start with "Propose …" or "Introduce …".
  - Builds on / Refines (participant extends or sharpens another role's idea): start with "Build on <Participant/Role>'s <concept> by …" or "Refine <Participant/Role>'s idea by …".
  - Supports / Acknowledges (participant agrees and adds rationale): start with "Support <Participant/Role>'s <concept> by …" or "Acknowledge … and add …".
  - Questions / Probes (participant asks for clarification or tests assumptions): start with "Question …" or "Probe …".
  - Counters (participant pushes back or recommends against): start with "Push back on …" or "Recommend against …".
  - Summarizes/Synthesizes (participant integrates prior points): start with "Summarize …" or "Synthesize …".
- Do not misattribute ownership: if the participant did not originate the idea in this segment, do not use "Propose …"—use a build/support/question/counter verb instead.

## Uniqueness and Non-Repetition
- Do not emit duplicate bullets. Each bullet must be unique within your output.
- If two candidates express the same claim with minor wording changes, keep only the more informative version (based solely on the participant's own words).

## Redundancy Control (Generalized Canonicalization)
- Internally reduce each candidate to a generalized claim_key before deciding to emit it:
  - Strip lead-in verbs and hedges (e.g., "provide data showing", "highlight that", "emphasize that", "affirm that", "state that", "suggest/propose/advocate", "confirm", "summarize/synthesize").
  - Drop filler like a leading "that" and keep the shortest faithful core (subject + predicate) of the claim.
  - Treat different wordings that assert the same core fact/stance/action as one claim_key.
- Emit at most one bullet per claim_key. If several candidates map to the same claim_key, keep the most informative formulation and fold secondary benefits/contrasts into that bullet (e.g., by adding a short clause), rather than emitting an additional bullet.
- Use the Verb Palette to pick one canonical starter verb per claim_key and framing. Do not vary verbs (e.g., highlight/emphasize/affirm) to produce multiple bullets for the same claim_key.

## Claim Extraction and Consolidation (Generalized, Two-Step within One Response)
- Step 1 — Claims (internal planning): identify a set of unique claims from the participant's lines. For each claim record:
  - canonical_claim: shortest faithful core (subject + predicate), independent of lead-ins and synonyms.
  - subject/action/object: compact tuple capturing the core. Use normalized English lemmas (e.g., subject="single_button_control", action="integrate", object="volume_and_channel").
  - framing and origin: framing ∈ {introduces, builds_on, supports, questions, counters, summarizes}; origin_role ∈ {self, other:<Role>}.
  - **support_utt_ids** (required): The represented participant's utterance IDs that form the basis for this JIT point's content. This is the GT evidence from which the bullet's action and any rationale ("because" clause) are derived. Content and rationale must be traceable to these utterances.
  - **context_utt_ids** (optional): Utterance IDs that form the CONVERSATIONAL CHAIN leading to this point. Include:
    - **Direct trigger**: The immediate utterance(s) that prompted the response (e.g., a question or request from another speaker).
    - **Setup utterances**: Earlier utterances that initiated the topic thread leading to the trigger—especially if the represented participant raised the topic that prompted others' questions.
    - **TEMPORAL RULE**: All context_utt_ids must come BEFORE support_utt_ids in the transcript. They capture what led TO the contribution, not reactions.
    - **CHAIN SCOPE**: Include utterances with clear causal/topical link to this point; stop when the link becomes weak background.
    - Use context_utt_ids ONLY to derive the `context:` hint, NEVER for the "because" clause or main content.
    - Leave empty for proactive ideas with no specific prior trigger.
  - benefits/qualifiers: short list of modifiers to fold into the bullet (e.g., "appeals to young", "appeals to old").
- Step 2 - Bullet synthesis: emit at most one bullet per canonical_claim using the Verb Palette based on framing; fold any benefits/qualifiers into that single bullet; do not produce synonym variants.

## Cue / Trigger Design (Pre-Move Context for `context:`)
- The `(topic: ...; context: ...)` suffix is where you encode a short, **generalized description of the pre-move situation** in which the point is appropriate.
- The `context:` fragment MUST describe a situation the delegate can recognise **before** speaking, based on the observable context in the testing segment.
- Use the following rules:
  - Treat support_utt_ids as evidence for the claim itself (the participant's utterances where the idea is expressed). Do NOT base the `context:` hint on these; they describe the *realisation* of the idea.
  - Treat context_utt_ids as evidence for the **pre-idea context**. Derive the `context:` hint from the **dialogue-act pattern** of these utterances, not just topic keywords:
    - If context contains a **question/request** → use "when asked to [action] [topic]" (e.g., "when asked to list key device functions").
    - If context contains a **challenge/pushback** → use "when [topic] is questioned" or "when there's pushback on [topic]".
    - If context shows the **participant initiated** the topic that led to others' questions → incorporate this (e.g., "when following up on own request for [topic] decisions").
    - If context contains a **proposal from others** → use "when [topic] is proposed" or "when someone suggests [action]".
  - For introduces/framing=self (proactive idea), derive the `context:` from recognizable situational cues (decision points, meeting phase, open floor) rather than abstract topic labels.
  - Prefer **dialogue-act language** ("when asked to...", "when challenged on...", "when confirming...") over passive topic keywords ("when determining...", "when considering...").
- Explicitly avoid self-referential or hindsight contexts:
  - Do NOT write contexts like "when you propose delegating to Sam" or "when I say that latency is an issue".
  - Do NOT encode details that only become knowable *after* the participant speaks.
  - Do NOT mention utterance IDs or quote full sentences verbatim in the `context:`; use short, generic descriptors of the situation.

### Proactive / Self-Initiated Contributions (No context_utt_ids)
- Some ideas are introduced proactively by the participant without any specific prompt from other participants (no clear context_utt_ids). For these:
  - Derive the `context:` hint from the local conversation around the participant's support_utt_ids (e.g., the last few utterances in the testing segment) and the broader topic/phase of the meeting.
  - Make the `context:` describe when this proactive idea would be appropriate, for example:
    - "when the group is exploring potential reliability risks and no one has mentioned thermal safety yet"
    - "when the floor is open and the team is considering options for remote layout"
  - Even for proactive ideas, ensure the context encodes a **recognisable situation** where the idea is relevant; do not use generic desire-based contexts like "when I want to mention ..." or "when I feel like ...".

## Output Format
- The output MUST be a single, valid JSON object with keys:
    - "thoughts": (string) Your reasoning for selecting the points and how you will rephrase them as forward-looking instructions. Briefly list the participant-owned nuclei you identified and confirm each has at least one bullet (only include nuclei actually present in the participant's lines).
    - "jit_briefing_points": (list of strings) A list of the rephrased, forward-looking key points. If no substantive contributions were made, return an empty list ([]).
    - "points_evidence" (optional): array of objects, one per bullet you emitted, each with:
        - text: exact bullet text
        - support_utt_ids: list of the represented participant's utterance IDs that form the GT evidence for this bullet (non-empty). The bullet's action and any "because" clause must be traceable to these utterances ONLY.
        - intent_type (optional): one of question|proposal|decision|metric|stance|risk|request|clarification
        - topical_tags (optional): 1-3 short tags
        - anchor_phrases (optional): 1-2 short phrases (2-4 words) the participant actually used
        - framing (optional): one of introduces|builds_on|supports|questions|counters|summarizes
        - context_utt_ids (optional): Utterance IDs forming the conversational chain that led to this point (trigger + setup utterances). Include both the direct trigger (e.g., question that prompted the response) and earlier setup utterances (e.g., if participant initiated the topic). Used ONLY for deriving the `context:` hint. Must come BEFORE support_utt_ids in transcript order. Leave empty for proactive ideas.
        - canonical_claim (optional): shortest faithful core of the claim, independent of lead-ins/synonyms
        - subject (optional): normalized subject token(s) for the claim (e.g., "single_button_control")
        - action (optional): normalized action (e.g., "integrate", "appeal_to", "deprecate")
        - object (optional): normalized object token(s) (e.g., "volume_and_channel", "young_users")
        - origin_role (optional): "self" if originated by the participant; or "other:<Role>" if building on another role's idea

---

## Example
Suppose the user asks you to analyze a transcript for "Engineer_A".
<Transcript Segment>
(Manager): Okay, team, let's get a status update on the database migration. Engineer_B, where are we with the schema? (Meeting4_150)
(Engineer_B): I've finished the schema mapping. It looks good, no major blockers there. (Meeting4_151)
(Manager): Good. And Engineer_A, what about the performance tests? (Meeting4_152)
(Engineer_A): I've been running the benchmarks all week. The new database instance is consistently 30% faster on read-heavy queries. (Meeting4_153)
(Manager): Excellent. Any concerns? (Meeting4_154)
(Engineer_A): Yeah, one issue. The data backfill script is still single-threaded. (Meeting4_155)
(Engineer_B): How long does it take to run? (Meeting4_156)
(Engineer_A): Right now it's taking over 12 hours, which exceeds our maintenance window. We'll need to parallelize it. (Meeting4_157)
(Manager): Can you take that on? (Meeting4_158)
(Engineer_A): Yep. (Meeting4_159)
(Manager): Alright, let's move on to... (Meeting4_160)
</Transcript Segment>

**Expected Output**:
```json
{
  "thoughts": "Engineer_A contributes three nuclei: 30% perf gain; backfill bottleneck + parallelize; ownership of redesign. Each bullet follows the action+rationale+topic/context template. Note: rationale included ONLY where Engineer_A explicitly stated it.",
  "jit_briefing_points": [
    "State that the new DB is consistently 30% faster on read-heavy queries. (topic: performance; context: when asked about performance test results)",
    "Highlight that the backfill script is single-threaded and takes over 12 hours, and propose parallelizing it because it exceeds the maintenance window. (topic: backfill risk; context: when asked about concerns or issues with the migration)",
    "Confirm you will take on redesigning the backfill script. (topic: ownership; context: when asked to take on the backfill work)"
  ],
  "points_evidence": [
    {
      "text": "State that the new DB is consistently 30% faster on read-heavy queries. (topic: performance; context: when asked about performance test results)",
      "support_utt_ids": ["Meeting4_153"],
      "intent_type": "metric",
      "topical_tags": ["performance"],
      "anchor_phrases": ["30% faster", "read-heavy queries"],
      "framing": "introduces",
      "context_utt_ids": ["Meeting4_152"]
    },
    {
      "text": "Highlight that the backfill script is single-threaded and takes over 12 hours, and propose parallelizing it because it exceeds the maintenance window. (topic: backfill risk; context: when asked about concerns or issues with the migration)",
      "support_utt_ids": ["Meeting4_155", "Meeting4_157"],
      "intent_type": "proposal",
      "topical_tags": ["backfill", "maintenance window"],
      "anchor_phrases": ["single-threaded", "12 hours", "exceeds our maintenance window"],
      "framing": "introduces",
      "context_utt_ids": ["Meeting4_154", "Meeting4_156"]
    },
    {
      "text": "Confirm you will take on redesigning the backfill script. (topic: ownership; context: when asked to take on the backfill work)",
      "support_utt_ids": ["Meeting4_159"],
      "intent_type": "decision",
      "topical_tags": ["commitment", "ownership"],
      "anchor_phrases": ["Yep"],
      "framing": "introduces",
      "context_utt_ids": ["Meeting4_158"]
    }
  ]
}
```
"""

        user_prompt_template = """Please analyze the following transcript segment for the participant '{person_id}'. Extract their key talking points and rephrase them as forward-looking instructions for their delegate as per your system guidelines.
        <Transcript Segment>
        {transcript_segment_string}
        </Transcript Segment>
        """

        # Escape curly braces in the transcript content to prevent format string issues
        escaped_transcript = transcript_segment_string.replace('{', '{{').replace('}', '}}')
        
        user_prompt = user_prompt_template.format(
            person_id=person_id,
            transcript_segment_string=escaped_transcript
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        for attempt in range(max_attempts):
            try:
                # Get JSON response using structured output (critical for GPT-5-mini)
                # Note: GPT-5-mini uses internal reasoning tokens that count against output budget
                json_response = self.llm_client.call_with_retry(
                    messages=messages,
                    max_tokens=8000,  # Increased for GPT-5-mini reasoning overhead
                    temperature=0.1,
                    json_required=True,  # Fallback for models without structured output
                    response_model=JITBriefingResponse,  # Structured output for GPT-5-mini/Gemini
                    telemetry_label="prep.jit_briefing",
                    telemetry_phase="prep",
                )
                
                if not json_response:
                    print(f"No JSON response for JIT briefing {person_id} (attempt {attempt + 1}/{max_attempts})")
                    continue
                
                # Handle both Pydantic model (structured output) and raw JSON (fallback)
                if hasattr(json_response, 'model_dump'):
                    return json_response  # Already a JITBriefingResponse
                
                # Manually validate the JSON against the JIT briefing schema (fallback)
                response = self.llm_client.validate_json_response(json_response, JITBriefingResponse)
                if response:
                    return response
                else:
                    print(f"JSON validation failed for JIT briefing {person_id} (attempt {attempt + 1}/{max_attempts})")
                    
            except (ValidationError, json.JSONDecodeError) as e:
                print(f"Error generating JIT briefing for {person_id} (attempt {attempt + 1}/{max_attempts}): {e}")
                time.sleep(base_delay * (2 ** attempt))
            except Exception as e:
                print(f"An unexpected error occurred while generating JIT briefing for {person_id}: {e}")
                break  # Exit on non-retryable errors

        print(f"Failed to generate valid JIT briefing for {person_id} after {max_attempts} attempts.")
        return None

    # ---- Helpers for improved flows (downstream-safe) ----
    def _norm_text(self, text: str) -> str:
        """Normalize bullet text for simple deduplication without embeddings."""
        if not isinstance(text, str):
            return ""
        t = text.strip().lower()
        # Strip standardized topic/context suffix so dedupe operates on the core idea
        # Example full form: "<action and rationale>. (topic: delegation; context: when ... )"
        # We conservatively drop everything from the first "(topic:" onward.
        topic_split = re.split(r"\(topic:", t, maxsplit=1)
        if topic_split:
            t = topic_split[0]
        t = re.sub(r"[^\w\s]", " ", t)
        t = re.sub(r"\s+", " ", t)
        return t.strip()

    # def _dedupe_key(self, text: str) -> str:
    #     """Canonicalize text to collapse near-identical bullets differing only by lead-in verbs.

    #     This is conservative: it strips common lead-in verb phrases (e.g., 'highlight that',
    #     'emphasize the importance of', 'provide data showing', 'propose', 'suggest', etc.) and
    #     returns a normalized claim body for dedupe comparison. It does NOT rewrite the final
    #     bullet; it is only used to detect duplicates.
    #     """
    #     if not isinstance(text, str):
    #         return ""
    #     t = text.lower().strip()
    #     # remove punctuation for keying
    #     t = re.sub(r"[^a-z0-9\s]", " ", t)
    #     t = re.sub(r"\s+", " ", t)
    #     # drop common lead-in verb phrases
    #     leadins = [
    #         "provide data showing", "show data that", "show that", "highlight the importance of",
    #         "highlight that", "emphasize the importance of", "emphasize that", "affirm that",
    #         "state that", "suggest that", "suggest ", "propose that", "propose ", "advocate that",
    #         "advocate ", "build on ", "refine ", "support ", "acknowledge ", "question ", "probe ",
    #         "push back on ", "recommend against ", "summarize ", "synthesize ", "affirm ", "highlight ",
    #         "emphasize ", "stress ", "confirm ", "assert that", "assert ", "note that", "note ",
    #     ]
    #     for li in leadins:
    #         if t.startswith(li):
    #             t = t[len(li):].lstrip()
    #             break
    #     # drop leading 'that' if present after removal
    #     if t.startswith("that "):
    #         t = t[5:]
    #     # normalize whitespace
    #     t = re.sub(r"\s+", " ", t).strip()
    #     return t

    def format_utterances_for_prompt(self, utterance_list: List[Dict]) -> str:
        """Formats a list of utterance objects into a single string for the LLM prompt."""
        return "\\n".join([
            f"({utt['speaker_id']}): {utt['utterance_content']} ({utt['utterance_id']})" 
            for utt in utterance_list
        ])

    def _clean_points_llm(
        self,
        participant_id: str,
        meeting_id: str,
        bullets: List[str],
        points_evidence: Optional[List[JITBriefingPointEvidence]] = None,
        max_attempts: int = 2,
        base_delay: float = 1.5,
    ) -> Optional[JITBriefingResponse]:
        """Final, single-pass LLM cleanup for deduping/merging bullets and aligning evidence.

        Rules:
        - Do not invent new bullets; only merge duplicates/near-duplicates or drop exact/synonym repeats.
        - Prefer the most informative wording among paraphrases.
        - If merging, union and de-duplicate support_utt_ids. Do not fabricate IDs.
        - Keep output shape strictly as JITBriefingResponse (downstream-safe).
        """
        if not bullets:
            return None

        sys_prompt = (
            "You are an experienced editor refining briefing bullets.\n"
            "Task: remove duplicates/near-duplicates (merge or drop), keep the most informative wording, "
            "realign evidence, and preserve the required structure and specifics. Do not add new bullets, and do not eliminate unique bullet points.\n\n"
            "## Schema (must validate):\n"
            "- JITBriefingResponse: {\n"
            "    \"jit_briefing_points\": array of strings,\n"
            "    \"thoughts\": string,\n"
            "    \"points_evidence\": optional array of JITBriefingPointEvidence\n"
            "  }\n"
            "- JITBriefingPointEvidence: {\n"
            "    \"text\": string,                // exact bullet text as in jit_briefing_points\n"
            "    \"support_utt_ids\": array[string], // non-empty if present\n"
            "    \"intent_type\": optional string in {question, proposal, decision, metric, stance, risk, request, clarification},\n"
            "    \"topical_tags\": optional array[string],\n"
            "    \"anchor_phrases\": optional array[string],\n"
            "    \"framing\": optional string in {introduces, builds_on, supports, questions, counters, summarizes},\n"
            "    \"context_utt_ids\": optional array[string]\n"
            "  }\n\n"
            "## Mandatory bullet template (must preserve or repair):\n"
            "<forward-looking action[ with optional grounded 'because ...' clause]>. (topic: <1-3 words>[; context: <short generalized pre-move context>])\n"
            "- Preserve the main action and any participant-stated rationale that is present; do not drop or rewrite it into vague forms when it exists.\n"
            "- Preserve participant-owned specifics (feature/metric/task names like \"sound feature\", \"dual-channel watch\", \"teletext\", as an example); do NOT replace with generic placeholders.\n"
            "- Topic tag stays at end in parentheses, 1-3 words, no names/IDs.\n"
            "- If a `context:` hint is present inside the parentheses (e.g., '; context: when cost concerns about the base station are raised'), keep it short and faithful to the original generalized pre-move situation.\n"
            "- One main sentence (action + rationale) per bullet, followed by the `(topic: ...[; context: ...])` suffix.\n\n"
            "## Scope of Edits (Do NOT invent new context)\n"
            "- Your job is to clean up redundancy and align evidence, not to change when a point is appropriate to use.\n"
            "- Do NOT invent new cue situations or change the underlying context in which a point is relevant.\n"
            "- Avoid rewriting `context:` descriptions to reference post-hoc or hindsight conditions; keep them consistent with the original intent.\n"
            "- Only make small repairs to obviously self-referential or malformed `context:` hints when you can do so without introducing new, speculative context.\n\n"
            "## Allowed starters (Verb Palette): Introduce, Build on, Support, Question, Counter, Summarize, Acknowledge.\n"
            "If a bullet starts with a non-canonical synonym, remap only the verb; keep cue/why/topic and specifics intact.\n\n"
            "## Normalization rules:\n"
            "- When merging paraphrases, keep a single best bullet; union support_utt_ids from merged items; drop evidence for deleted bullets.\n"
            "- Never fabricate new support_utt_ids. If none are available for a surviving bullet, omit that evidence entry.\n"
            "- If a given utterance ID appears in both support_utt_ids and context_utt_ids for the same point, keep it only in context_utt_ids and remove it from support_utt_ids (to avoid mixing participant and non-participant evidence in the same slot)."
        )

        ev_payload = []
        if points_evidence:
            for ep in points_evidence:
                try:
                    ev_payload.append({
                        'text': ep.text,
                        'support_utt_ids': ep.support_utt_ids,
                        'intent_type': getattr(ep, 'intent_type', None),
                        'topical_tags': getattr(ep, 'topical_tags', None),
                        'anchor_phrases': getattr(ep, 'anchor_phrases', None),
                        'framing': getattr(ep, 'framing', None),
                        'context_utt_ids': getattr(ep, 'context_utt_ids', None),
                    })
                except Exception:
                    continue

        payload = {
            'participant_id': participant_id,
            'meeting_id': meeting_id,
            'bullets': bullets,
            'evidence': ev_payload,
            'instructions': {
                'non_invention': True,
                'dedupe_policy': 'Merge paraphrases; drop synonym repeats; keep most informative wording.',
                'evidence_rules': 'If merging, union support_utt_ids; never invent IDs; keep framing/origin if present.',
                'verb_palette': ['Introduce','Build on','Support','Question','Counter','Summarize','Acknowledge'],
                'template_required': True,
                'preserve_cue_why_topic': True,
                'preserve_participant_specifics': True
            }
        }

        user_prompt = (
            f"Participant: {participant_id} | Meeting: {meeting_id}\n"
            "Return only a single JSON object that validates against the schema.\n\n"
            "<INPUT>\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n</INPUT>\n"
            "<REQUIREMENTS>\n"
            "- Do not add new bullets not present in input.\n"
            "- Remove duplicates/near-duplicates; keep one best bullet per claim.\n"
            "- Align points_evidence to the final bullets (1:1 by text); union support_utt_ids when merging; never invent IDs.\n"
            "- **STRICT CAP**: The final list must contain at most 8 bullets. Prioritize the most critical/high-impact points. Drop less important ones if needed to stay within 8.\n"
            "</REQUIREMENTS>"
        )

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(max_attempts):
            try:
                raw = self.llm_client.call_with_retry(
                    messages=messages,
                    max_tokens=8000,  # Increased for GPT-5-mini reasoning overhead
                    temperature=0.1,
                    json_required=True,
                    telemetry_label="prep.jit_cleanup",
                    telemetry_phase="prep",
                )
                if not raw:
                    time.sleep(base_delay * (attempt + 1))
                    continue
                cleaned = self.llm_client.validate_json_response(raw, JITBriefingResponse)
                if cleaned:
                    return cleaned
            except Exception as e:
                print(f"[JIT-CLEANUP] Error: {e}")
                time.sleep(base_delay * (attempt + 1))
        return None

    def generate_briefings_for_meeting(
        self,
        transcript_path: str,
        output_directory: str,
        split_ratio: float = 0.6,
        meeting_id: str = None,
        target_participant: str = None,
        # Optional sliding-window enrichment (downstream-safe; defaults preserve legacy behavior)
        enable_sliding_windows: bool = False,
        window_size: int = 50,
        window_stride: int = 30,
        emit_evidence_sidecar: bool = True,
        enable_llm_cleanup: bool = False,
    ) -> Dict[str, str]:
        """
        Generate JIT briefings for participants in a meeting transcript.
        
        Args:
            transcript_path: Path to parsed transcript JSON file
            output_directory: Directory to save briefing files
            split_ratio: Ratio for profile generation vs briefing generation (default 0.6)
            meeting_id: Meeting ID to use for naming files (if None, extracts from filename)
            target_participant: If specified, only generate briefings for this participant
            
        Returns:
            Dictionary mapping participant_id to output file path
        """
        # Load the full preprocessed transcript
        if not os.path.exists(transcript_path):
            print(f"Error: Preprocessed transcript '{transcript_path}' not found.")
            return {}

        with open(transcript_path, 'r', encoding='utf-8') as f:
            full_transcript = json.load(f)

        if not full_transcript:
            print(f"Error: Transcript file '{transcript_path}' is empty.")
            return {}

        # Split the transcript into Part A (profiling source) and Part B (briefing source)
        split_index = int(len(full_transcript) * split_ratio)
        testing_segment = full_transcript[split_index:]  # We only need Part B for this task

        if not testing_segment:
            print(f"Error: Testing segment (Part B) of the transcript is empty.")
            return {}

        # Identify all participants who spoke in the testing segment
        participants_in_testing_segment = sorted(list(set(utt['speaker_id'] for utt in testing_segment)))

        # Filter participants if target_participant is specified
        if target_participant:
            if target_participant in participants_in_testing_segment:
                participants_in_testing_segment = [target_participant]
                print(f"Found participants in testing segment to profile: {participants_in_testing_segment} (filtered to target: {target_participant})")
            else:
                print(f"❌ Target participant '{target_participant}' not found in testing segment: {participants_in_testing_segment}")
                return {}
        else:
            print(f"Found participants in testing segment to profile: {participants_in_testing_segment}")

        testing_segment_string = self.format_utterances_for_prompt(testing_segment)

        # Ensure output directory exists
        os.makedirs(output_directory, exist_ok=True)

        # Generate briefings for each participant
        generated_briefings = {}
        
        # Use provided meeting_id or extract from filename as fallback
        if meeting_id:
            meeting_id_base = meeting_id
        else:
            meeting_id_base = os.path.basename(transcript_path).replace('_parsed.json', '')

        for participant_id in participants_in_testing_segment:
            print(f"\\n{'='*20}\\nGenerating JIT Briefing for: {participant_id}\\n{'='*20}")

            if not enable_sliding_windows:
                # Legacy single-pass behavior (unchanged output structure)
                briefing_response = self.generate_jit_briefing_for_participant(
                    person_id=participant_id,
                    transcript_segment_string=testing_segment_string
                )

                if briefing_response:
            # Intra-response dedupe by normalized text
                    seen_norms = set()
                    deduped_points: List[str] = []
                    for b in (briefing_response.jit_briefing_points or []):
                        n = self._norm_text(b)
                        if n and n not in seen_norms:
                            deduped_points.append(b)
                            seen_norms.add(n)
                    # Filter optional evidence to match deduped bullets
                    pe = getattr(briefing_response, 'points_evidence', None)
                    if pe:
                        keep_texts = set(deduped_points)
                        pe = [e for e in pe if getattr(e, 'text', None) in keep_texts]
                        payload = JITBriefingResponse(jit_briefing_points=deduped_points,
                                                      thoughts=briefing_response.thoughts,
                                                      points_evidence=pe)
                    else:
                        payload = JITBriefingResponse(jit_briefing_points=deduped_points,
                                                      thoughts=briefing_response.thoughts)
                    # Optional LLM cleanup pass before writing
                    if enable_llm_cleanup:
                        before_n = len(payload.jit_briefing_points or [])
                        cleaned = self._clean_points_llm(
                            participant_id=participant_id,
                            meeting_id=meeting_id_base,
                            bullets=list(payload.jit_briefing_points or []),
                            points_evidence=list(payload.points_evidence or []),
                        )
                        if cleaned and (cleaned.jit_briefing_points or []):
                            payload = cleaned
                            after_n = len(payload.jit_briefing_points or [])
                            print(f"[JIT-CLEANUP] {participant_id}: bullets {before_n} -> {after_n}")

                    briefing_filename = f"{participant_id.replace(' ', '_')}_briefing_from_{meeting_id_base}_partB.json"
                    briefing_path = os.path.join(output_directory, briefing_filename)
                    with open(briefing_path, 'w', encoding='utf-8') as f:
                        f.write(payload.model_dump_json(indent=2))
                    print(f"--> SUCCESS: Saved JIT briefing for {participant_id} to '{briefing_path}'")
                    generated_briefings[participant_id] = briefing_path
                    # Optional sidecar directly from returned evidence (no extra LLM calls)
                    if emit_evidence_sidecar and getattr(payload, 'points_evidence', None):
                        sidecar = {
                            "points": [
                                {
                                    "point_id": hashlib.sha256((re.sub(r"\s+", " ", (ep.text or '').lower()) + '|' + (ep.support_utt_ids[0] if ep.support_utt_ids else '')).encode('utf-8')).hexdigest()[:16],
                                    "text": ep.text,
                                    "support_utt_ids": ep.support_utt_ids,
                                    "intent_type": ep.intent_type,
                                    "topical_tags": ep.topical_tags or [],
                                    "anchor_phrases": ep.anchor_phrases or [],
                                }
                                for ep in (payload.points_evidence or [])
                                if isinstance(ep, JITBriefingPointEvidence)
                            ],
                            "meta": {"participant_id": participant_id, "meeting_id": meeting_id_base, "segment": "testing"},
                        }
                        sidecar_filename = f"{participant_id.replace(' ', '_')}_briefing_from_{meeting_id_base}_partB.evidence.json"
                        with open(os.path.join(output_directory, sidecar_filename), 'w', encoding='utf-8') as f:
                            json.dump(sidecar, f, indent=2, ensure_ascii=False)
                        print(f"    (evidence) Wrote sidecar -> {os.path.join(output_directory, sidecar_filename)}")
                else:
                    print(f"--> FAILED: Could not generate a valid briefing for {participant_id}.")
                continue

            # Sliding-window enrichment with a single master list (downstream-safe)
            master_points: List[str] = []
            master_norms: set = set()
            saw_valid_response = False
            for start in range(0, len(testing_segment), max(1, window_stride)):
                end = min(len(testing_segment), start + max(1, window_size))
                if start >= end:
                    break
                window_utts = testing_segment[start:end]
                if not window_utts:
                    continue
                win_str = self.format_utterances_for_prompt(window_utts)
                # Provide an "already included" hint to help the LLM avoid duplicates and close paraphrases
                hint = (
                    "\n\n---\nAlready included (avoid duplicates and close paraphrases):\n"
                    + "\n".join([f"- {t}" for t in master_points])
                    + "\n- If a candidate is semantically equivalent or a minor rewording of any item above, skip it.\n"
                    + "- Only add it if it introduces a genuinely new claim; otherwise keep the more informative earlier version.\n"
                    + "- Use the Verb Palette; do not vary verbs for the same claim.\n"
                ) if master_points else ""
                resp = self.generate_jit_briefing_for_participant(
                    person_id=participant_id,
                    transcript_segment_string=(win_str + hint)
                )
                if not resp:
                    continue
                saw_valid_response = True
                # Intra-response dedupe for this window
                resp_seen = set()
                for b in resp.jit_briefing_points or []:
                    n = self._norm_text(b)
                    if not n or n in resp_seen:
                        continue
                    resp_seen.add(n)
                    if n and n not in master_norms:
                        master_points.append(b)
                        master_norms.add(n)
                # Merge any evidence returned for bullets in this window
                if getattr(resp, 'points_evidence', None):
                    if 'evidence_map' not in locals():
                        evidence_map = {}
                    for pe in resp.points_evidence or []:
                        if not isinstance(pe, JITBriefingPointEvidence):
                            continue
                        key = self._norm_text(pe.text or '')
                        if not key:
                            continue
                        rec = evidence_map.get(key) or {
                            'text': pe.text,
                            'support_utt_ids': [],
                            'intent_type': pe.intent_type,
                            'topical_tags': pe.topical_tags or [],
                            'anchor_phrases': pe.anchor_phrases or [],
                            'framing': pe.framing if hasattr(pe, 'framing') else None,
                            'context_utt_ids': list(pe.context_utt_ids) if getattr(pe, 'context_utt_ids', None) else [],
                        }
                        # union supports/anchors/tags
                        rec['support_utt_ids'] = sorted(list({*rec['support_utt_ids'], *(pe.support_utt_ids or [])}))
                        if pe.intent_type and not rec.get('intent_type'):
                            rec['intent_type'] = pe.intent_type
                        rec['topical_tags'] = sorted(list({*(rec.get('topical_tags') or []), *(pe.topical_tags or [])}))
                        # keep up to 2 anchors
                        anchors = list(rec.get('anchor_phrases') or [])
                        for a in (pe.anchor_phrases or []):
                            if a and a not in anchors and len(anchors) < 2:
                                anchors.append(a)
                        rec['anchor_phrases'] = anchors[:2]
                        # framing/context merge: prefer earliest non-empty framing; union context ids
                        if getattr(pe, 'framing', None) and not rec.get('framing'):
                            rec['framing'] = pe.framing
                        if getattr(pe, 'context_utt_ids', None):
                            rec['context_utt_ids'] = sorted(list({*(rec.get('context_utt_ids') or []), *pe.context_utt_ids}))
                        evidence_map[key] = rec

            # Fallback to single-pass if needed
            if not master_points:
                print("[JIT] Sliding-window produced no points; falling back to single-pass.")
                resp = self.generate_jit_briefing_for_participant(
                    person_id=participant_id,
                    transcript_segment_string=testing_segment_string
                )
                if resp:
                    saw_valid_response = True
                    master_points = list(resp.jit_briefing_points or [])

            semantic_empty = False
            if not master_points:
                if not saw_valid_response:
                    print(f"--> FAILED: Could not generate a valid briefing for {participant_id}.")
                    continue
                semantic_empty = True

            # Optional LLM cleanup for sliding-window master_points
            jr: Optional[JITBriefingResponse]
            jr = None
            if semantic_empty:
                jr = JITBriefingResponse(
                    jit_briefing_points=[],
                    thoughts="No meaningful participant-owned briefing points were found in the testing segment.",
                )
            elif enable_llm_cleanup:
                ev_list = []
                if 'evidence_map' in locals() and evidence_map:
                    for rec in evidence_map.values():
                        try:
                            ev_list.append(JITBriefingPointEvidence(**{
                                'text': rec.get('text'),
                                'support_utt_ids': rec.get('support_utt_ids') or [],
                                'intent_type': rec.get('intent_type'),
                                'topical_tags': rec.get('topical_tags') or [],
                                'anchor_phrases': rec.get('anchor_phrases') or [],
                                'framing': rec.get('framing'),
                                'context_utt_ids': rec.get('context_utt_ids') or [],
                            }))
                        except Exception:
                            continue
                before_n = len(master_points)
                cleaned = self._clean_points_llm(
                    participant_id=participant_id,
                    meeting_id=meeting_id_base,
                    bullets=list(master_points),
                    points_evidence=ev_list if ev_list else None,
                )
                if cleaned and (cleaned.jit_briefing_points or []):
                    jr = cleaned
                    after_n = len(jr.jit_briefing_points or [])
                    print(f"[JIT-CLEANUP] {participant_id}: bullets {before_n} -> {after_n}")
            if jr is None:
                thoughts = f"Compiled {len(master_points)} testing-segment points via sliding windows."
                jr = JITBriefingResponse(jit_briefing_points=master_points, thoughts=thoughts)
            briefing_filename = f"{participant_id.replace(' ', '_')}_briefing_from_{meeting_id_base}_partB.json"
            briefing_path = os.path.join(output_directory, briefing_filename)
            with open(briefing_path, 'w', encoding='utf-8') as f:
                f.write(jr.model_dump_json(indent=2))
            generated_briefings[participant_id] = briefing_path
            if semantic_empty:
                print(f"--> SUCCESS: Saved empty JIT briefing (semantic no-point result) for {participant_id} to '{briefing_path}'")
            else:
                print(f"--> SUCCESS: Saved JIT briefing (sliding-window) for {participant_id} to '{briefing_path}'")

            # Optional evidence sidecar (prefer cleaned evidence if present)
            if emit_evidence_sidecar:
                sidecar_points = []
                if getattr(jr, 'points_evidence', None):
                    for ep in jr.points_evidence or []:
                        try:
                            pid_source = re.sub(r"\s+", " ", (ep.text or '').lower()) + '|' + (ep.support_utt_ids[0] if ep.support_utt_ids else '')
                            pid = hashlib.sha256(pid_source.encode('utf-8')).hexdigest()[:16]
                            sidecar_points.append({
                                'point_id': pid,
                                'text': ep.text,
                                'support_utt_ids': ep.support_utt_ids,
                                'intent_type': ep.intent_type,
                                'topical_tags': ep.topical_tags or [],
                                'anchor_phrases': ep.anchor_phrases or [],
                                'framing': getattr(ep, 'framing', None),
                                'context_utt_ids': getattr(ep, 'context_utt_ids', []) or [],
                            })
                        except Exception:
                            continue
                elif 'evidence_map' in locals() and evidence_map:
                    for rec in evidence_map.values():
                        pid_source = re.sub(r"\s+", " ", (rec.get('text') or '').lower()) + '|' + (rec.get('support_utt_ids')[0] if rec.get('support_utt_ids') else '')
                        pid = hashlib.sha256(pid_source.encode('utf-8')).hexdigest()[:16]
                        sidecar_points.append({
                            'point_id': pid,
                            'text': rec.get('text'),
                            'support_utt_ids': rec.get('support_utt_ids') or [],
                            'intent_type': rec.get('intent_type'),
                            'topical_tags': rec.get('topical_tags') or [],
                            'anchor_phrases': rec.get('anchor_phrases') or [],
                            'framing': rec.get('framing'),
                            'context_utt_ids': rec.get('context_utt_ids') or [],
                        })
                if sidecar_points:
                    sidecar = {'points': sidecar_points, 'meta': {'participant_id': participant_id, 'meeting_id': meeting_id_base, 'segment': 'testing'}}
                    sidecar_filename = f"{participant_id.replace(' ', '_')}_briefing_from_{meeting_id_base}_partB.evidence.json"
                    sidecar_path = os.path.join(output_directory, sidecar_filename)
                    with open(sidecar_path, 'w', encoding='utf-8') as f:
                        json.dump(sidecar, f, indent=2, ensure_ascii=False)
                    print(f"    (evidence) Wrote sidecar -> {sidecar_path}")

        return generated_briefings

    def format_jit_briefing_for_prompt(self, jit_briefing_points: List[str]) -> str:
        """Formats the JIT briefing list into a string for the prompt."""
        if not jit_briefing_points:
            return "No specific briefing points provided for this session."
        return "\\n".join([f"- {point}" for point in jit_briefing_points])

    def format_transcript_for_prompt(self, transcript: List[Dict]) -> str:
        """
        Format transcript for inclusion in LLM prompts.
        Helper method for testing and external use.
        
        Args:
            transcript: List of utterance dictionaries
            
        Returns:
            Formatted string for prompt inclusion
        """
        formatted_lines = []
        for utt in transcript:
            speaker = utt.get('speaker_id', 'Unknown')
            content = utt.get('utterance_content', '')
            utt_id = utt.get('utterance_id', 'Unknown')
            formatted_lines.append(f"({speaker}): {content} ({utt_id})")
        return "\n".join(formatted_lines)

    def format_profile_for_prompt(self, profile: Dict) -> str:
        """
        Format user profile for inclusion in LLM prompts.
        Helper method for testing and external use.
        
        Args:
            profile: User profile dictionary
            
        Returns:
            Formatted string representation of the profile
        """
        if isinstance(profile, dict):
            import json
            return json.dumps(profile, indent=2)
        return str(profile)

    # def _map_bullets_to_evidence(
    #     self,
    #     participant_id: str,
    #     meeting_id: str,
    #     bullets: List[str],
    #     testing_segment_string: str,
    #     max_attempts: int = 2,
    # ) -> Optional[Dict[str, Any]]:
    #     """Map bullets to participant support_utt_ids and short anchor phrases (sidecar-only)."""
    #     if not bullets:
    #         return None
    #     sys_prompt = (
    #         "Map each briefing bullet to the specified participant's own utterances in the testing segment. "
    #         "Rules: Participant-Only; Stance Allowed; No Borrowed Specifics; Evidence Required; Anchor Phrases (short). "
    #         "Return JSON with keys: points (array of {text, support_utt_ids, intent_type, topical_tags, anchor_phrases}) and meta."
    #     )
    #     bullets_block = "\n".join([f"- {b}" for b in bullets])
    #     user_prompt = (
    #         f"Participant: {participant_id}\n"
    #         f"Bullets (map to your own utterance ids in this segment):\n{bullets_block}\n\n"
    #         f"<TestingSegment>\n{testing_segment_string}\n</TestingSegment>\n"
    #         "Output JSON only."
    #     )
    #     messages = [
    #         {"role": "system", "content": sys_prompt},
    #         {"role": "user", "content": user_prompt},
    #     ]
    #     for attempt in range(max_attempts):
    #         try:
    #             json_response = self.llm_client.call_with_retry(
    #                 messages=messages,
    #                 max_tokens=2200,
    #                 temperature=0.1,
    #                 json_required=True,
    #             )
    #             if not json_response:
    #                 continue
    #             parsed = json.loads(json_response)
    #             pts = parsed.get('points') if isinstance(parsed, dict) else None
    #             if not isinstance(pts, list):
    #                 continue
    #             cleaned: List[Dict[str, Any]] = []
    #             for p in pts:
    #                 if not isinstance(p, dict):
    #                     continue
    #                 text = (p.get('text') or '').strip()
    #                 su = [s for s in (p.get('support_utt_ids') or []) if isinstance(s, str) and s.strip()]
    #                 if not text or not su:
    #                     continue
    #                 norm = re.sub(r"\s+", " ", text.lower())
    #                 pid_source = norm + '|' + su[0]
    #                 pid = hashlib.sha256(pid_source.encode('utf-8')).hexdigest()[:16]
    #                 cleaned.append({
    #                     'point_id': pid,
    #                     'text': text,
    #                     'support_utt_ids': su,
    #                     'intent_type': p.get('intent_type') or 'stance',
    #                     'topical_tags': p.get('topical_tags') or [],
    #                     'anchor_phrases': p.get('anchor_phrases') or [],
    #                 })
    #             return {
    #                 'points': cleaned,
    #                 'meta': {
    #                     'participant_id': participant_id,
    #                     'meeting_id': meeting_id,
    #                     'segment': 'testing',
    #                 }
    #             }
    #         except Exception as e:
    #             print(f"[JIT] Evidence mapping error (attempt {attempt+1}): {e}")
    #     return None
