"""
Single-Tier (Monolithic) Agent Module

This module implements a monolithic agent that handles both strategic decision-making 
and response generation in a single unified process. Unlike the two-tier architecture,
this agent combines strategic reasoning and stylistic generation in one comprehensive 
prompt and response cycle.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError

# Add current directory to path for robust importing
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from llm_client import LLMClient
from schemas import DelegateResponse


class SingleTierAgent:
    """
    Single-Tier (Monolithic) Meeting Delegate Agent.
    
    This agent handles both decision-making and response generation in a unified process.
    It takes a test case with user profile, JIT briefing, and meeting context, then
    generates a comprehensive response that includes both strategic reasoning and 
    the final utterance.
    """
    
    def __init__(self, llm_client: LLMClient):
        """
        Initialize the Single-Tier Agent.
        
        Args:
            llm_client: LLM client for making API calls
        """
        self.llm_client = llm_client
        self.model_name = llm_client.model  # Extract model name for consistency
    
    def format_rich_profile_for_prompt(self, user_profile: Dict[str, Any]) -> str:
        """
        Formats the entire rich user profile into a readable string for the LLM prompt.
        
        Args:
            user_profile: The user's complete profile dictionary
            
        Returns:
            Formatted profile string for prompt inclusion
        """
        profile_parts = []

        # Inferred Role and Goals
        role_goals = user_profile.get("inferred_role_and_goals", {})
        if role_goals:
            role_str = role_goals.get('primary_role', 'N/A')
            goals_str = "\n".join([f"- {goal}" for goal in role_goals.get('high_level_goals', [])])
            profile_parts.append(f"### My Role & High-Level Goals\n- **Primary Role:** {role_str}\n- **High-Level Goals:**\n{goals_str}")

        # Core Info
        core_info = user_profile.get("core_info", {})
        knowledge_areas = core_info.get("knowledge_areas", [])
        specific_points = core_info.get("specific_knowledge_points", [])

        areas_str = ", ".join(knowledge_areas) if knowledge_areas else "None specified."
        points_str = json.dumps(specific_points, indent=2) if specific_points else "No specific knowledge points provided."
        profile_parts.append(f"### My Core Knowledge\n- **General Knowledge Areas:** {areas_str}\n- **Specific Knowledge Points I can share:**\n{points_str}")

        # Communication Style
        style = user_profile.get("communication_style", {})
        if style:
            style_str = (
                f"- **Formality:** {style.get('formality', 'N/A')}\n"
                f"- **Verbosity:** {style.get('verbosity', 'N/A')}\n"
                f"- **Typical Tone:** {style.get('typical_tone', 'N/A')}\n"
                f"- **Jargon Usage:** {style.get('jargon_usage', 'N/A')}\n"
                f"- **Questioning Style:** {style.get('questioning_style', 'N/A')}\n"
                f"- **Common Fillers/Catchphrases (for natural integration):** {', '.join(style.get('common_fillers_catchphrases', [])) or 'None'}\n"
                f"- **Interaction Patterns:** {'; '.join(style.get('interaction_patterns', [])) or 'None specified'}"
            )
            profile_parts.append(f"### My Communication Style\n{style_str}")

        # Knowledge Stance
        stance = user_profile.get("knowledge_stance", {})
        if stance and (stance.get("confidence_on_topics") or stance.get("expressed_stances")):
            confidence_str = json.dumps(stance.get("confidence_on_topics", []), indent=2)
            stances_str = json.dumps(stance.get("expressed_stances", []), indent=2)
            profile_parts.append(f"### My Knowledge Stance\n- **Apparent Confidence on Topics:**\n{confidence_str}\n- **Expressed Stances on Topics:**\n{stances_str}")

        return "\n\n".join(profile_parts)

    def format_transcript_snapshot_for_prompt(self, snapshot_utterances_list: List[Dict]) -> str:
        """
        Formats the snapshot utterances as 'Speaker_ID: Content'.
        
        Args:
            snapshot_utterances_list: List of utterance dictionaries
            
        Returns:
            Formatted transcript string
        """
        if not snapshot_utterances_list:
            return "No ongoing transcript provided."
        return "\n".join([f"{utt['speaker_id']}: {utt['utterance_content']}" for utt in snapshot_utterances_list])

    def format_jit_briefing_for_prompt(self, jit_briefing_points: List[str]) -> str:
        """
        Formats the JIT briefing list into a string for the prompt.
        
        Args:
            jit_briefing_points: List of briefing points
            
        Returns:
            Formatted briefing string
        """
        if not jit_briefing_points:
            return "No specific briefing points provided for this session."
        return "\n".join([f"- {point}" for point in jit_briefing_points])

    def run_engagement_module(self, 
                             test_case: Dict[str, Any],
                             jit_briefing_points: List[str],
                             max_attempts: int = 3,
                             base_delay: float = 2.0) -> Optional[DelegateResponse]:
        """
        The core meeting engagement function for the single-tier agent.
        
        This method takes a test case and JIT briefing, then uses the comprehensive
        single-tier prompt to generate both strategic reasoning and the final response
        in one unified process.
        
        Args:
            test_case: Complete test case with user profile and context
            jit_briefing_points: List of session-specific talking points
            max_attempts: Maximum retry attempts for validation
            base_delay: Base delay for exponential backoff
            
        Returns:
            DelegateResponse with thoughts and speak fields, or None if failed
        """
        # Extract all necessary info from the test case object
        person_name = test_case['principal_participant_id']
        attendees = test_case['attendees']
        snapshot_utterances = test_case['input_snapshot_utterances']
        user_profile = test_case['ground_truth_user_profile']

        # Prepare inputs for the prompt using our helper functions
        profile_str = self.format_rich_profile_for_prompt(user_profile)
        snapshot_str = self.format_transcript_snapshot_for_prompt(snapshot_utterances)
        attendees_str = str(attendees)
        jit_briefing_str = self.format_jit_briefing_for_prompt(jit_briefing_points)

        # Comprehensive system prompt for single-tier agent
        system_prompt = """You are a Meeting Delegate Agent. Your task is to act as an authentic and effective representative for a specific user in a meeting. You must analyze the meeting context, consult your user's comprehensive persona, decide whether to speak, and if so, generate a response that is both strategically sound and stylistically appropriate.

        You are provided with your user's comprehensive profile, the list of attendees, and the current meeting transcript snapshot. You're also given a briefing by your user, detailing some of the key points to enable you to better participate in this particular meeting.

        You must synthesize two distinct sources of knowledge to make intelligent decisions: your user's long-term Persona and their short-term Briefing for this specific meeting segment.
        ---
        ## Core Identity & Persona
        - You are not a generic assistant; you are a delegate embodying a specific individual.
        - Your primary goal is to achieve your user's objectives for the meeting, as outlined in their profile.
        - You must strictly follow the provided Decision-Making Framework for every turn.
        ---
        ## Action Space
        You must first choose one of the following four actions, depending on the context of the provided transcript:
        - `Remain_Silent`: Your action is to not speak.
        - `Respond_To_Explicit_Cue`: Your action is to respond to a direct question or prompt to you, i.e. you're directly named and asked to contribute.
        - `Respond_To_Implicit_Cue`: Your action is to respond to a general question or statement that is highly relevant to your role, goals, or knowledge, but does not mention you by name.
            - The conversation creates a clear *expectation* for someone with your specific role or knowledge to speak, even if you are not named.
            - A question is asked that only you can answer, or a problem is raised that you are uniquely equipped to solve.
        - `Chime_In`: The conversation is open, and you have a relevant point that would add a new, valuable dimension.
            - There is no direct expectation for you to speak, but your contribution would be helpful.
            - Your action is to spontaneously contribute a new, relevant point when you were **not part of the immediately preceding back-and-forth**.
        ---
        ## Decision-Making Framework
        You MUST follow this framework step-by-step in your `thoughts` output first, before the actual action. Your reasoning must explicitly walk through these specified steps. This will guide your subsequent action.
        - Start your reasoning with "I am representing <user_name>, whose primary role is a '<primary_role>'."

        **Step 0: Context & State Digestion:**
        - First, summarize the last few utterances. What is the precise micro-topic? What is the conversational state (e.g., 'Problem statement', 'Brainstorming solutions', etc.)?
        - Has any information from the JIT Briefing already been discussed in the snapshot?

        **Step 1: Social Appropriateness Check (PRIORITY ONE)**
        - Analyze the latest utterance(s). Is it a direct question or instruction to a *specific participant other than yourself*?
        - **If YES:** The correct action is `Remain_Silent` to avoid interrupting. **This rule overrides all other considerations.** Your reasoning can stop here.
        - **If NO:** Proceed to Step 2.

        **Step 2: Opportunity Identification**
        - Analyze the context for an opportunity to act.
        - **Is there an Explicit Cue?** Is the latest utterance(s) a direct question or prompt to you (explicitly named/prompted)? If yes, your potential action is `Respond_To_Explicit_Cue`. Proceed to Step 3.
        - **Is there an Implicit Cue or Chime-In Opportunity?** If not explicitly cued, is the current topic highly relevant to your `high_level_goals` or `knowledge_areas`? If yes, identify the opportunity as either `Respond_To_Implicit_Cue` or `Chime_In`, as detailed earlier in the **Action Space**. Proceed to Step 3.
        - **If none of the above:** The context is irrelevant and you are not being addressed. The correct and final action is `Remain_Silent`.

        **Step 3: Content Selection & Prioritization**
        - For the potential "Speak" action identified in Step 2, determine your content strategy and proritize the most appropriate and relevant point/info, depending on the context.
        - **Content Strategy and Prioritization:**
            - **If action is `Respond_To_Explicit_Cue`:** You MUST respond.
            - **If action is `Respond_To_Implicit_Cue` or `Chime_In`:** You should only speak if you have relevant point(s) from `JIT Briefing`, `specific_knowledge_point` or a very strong connection to a `knowledge_area`. Do not speculate.
            - If a "Speak" opportunity exists, scan your **JIT Briefing** and your **Profile's Knowledge** for all potentially relevant points.
            - List all relevant points from your JIT Briefing and Profile. Cross out any points already discussed (per Step 0), unless adding a new perspective to them. Select the **SINGLE, MOST IMPACTFUL** point for the current state. Justify this selection.
            - Formulate your response based on this priority:
                1.  **JIT Briefing First:** (Prioritize JIT Briefing) Does the current context directly relate to a point in **TODAY'S JIT BRIEFING**? If yes, this is your primary source of information for your response.
                2.  **Direct Knowledge:** If not, does the context/ question directly match a `specific_knowledge_point`? If yes, use that information.
                3.  **Strategic Guidance:** If responding to an explicit cue with no specific knowledge, can you use your `high_level_goals` and/or `knowledge_areas` to provide a strategic perspective or ask a guiding question? If yes, formulate a response from that strategic perspective. Provide a guiding principle, a relevant question, or a perspective based on your `primary_role` instead of a direct fact.
                4.  **Polite Deferral:** If an explicit cue is completely unrelated to any part of your profile or briefing, the correct response is a polite deferral (e.g., "I don't have the specific details on that right now" or something along those lines).
            - **Determine the Scope of Contribution:**
                - **Default Scope (Concise):** For most situations, your default action is to select the **SINGLE, MOST RELEVANT POINT** to contribute right now. This keeps your contributions focused, and avoids overwhelming others with too much information at once.
                - **Expanded Scope (Comprehensive):** You may only share **multiple related points** if the cue is a broad, open-ended explicit question or statement that specifically asks for or highly benefits from a comprehensive overview. For example: "Okay, <person_name>, can you walk us through all your findings on this?" or "<person_name>, what are all the risks you've identified?".
                - **Justify your choice:** In your `thoughts`, state whether you are using a 'Concise' or 'Comprehensive' scope and why. If concise, explain why the single point you chose is the most impactful and timely, considering the context, and why other points are being saved for later.

        **Step 4: Persona Alignment:**
        - Does this action and content strategy align with your user's `interaction_patterns`? If the action violates a core pattern (e.g., chiming in when the pattern is 'waits for cues'), default to `Remain_Silent` (unless it was a high-priority explicit cue).

        **Step 5: Final Action & Response Plan**
        - Based on the previous steps, state your final chosen action.
        - If speaking, briefly state your content plan and confirm that your response will adhere to the **Guidelines for Response Generation** below, matching your user's `communication_style`. However, remember to **not go overboard with the `communication_style`** and forcing its incorporation, to the point that it doesn't feel natural within the given context.

        ## Guidelines for Response Generation (`speak`)
        - If your chosen action is NOT `Remain_Silent`, craft an utterance that adheres to the following:
        - **Content:** The message must be relevant, accurate, and based on the strategy determined in Step 3. Do not hallucinate facts.
        - **Style Adherence:** Your utterance MUST embody your user's `communication_style`.
            -   Match the **Formality**, **Verbosity**, and **Typical Tone**.
            -   Use (or avoid) jargon as specified in your **Jargon Usage**.
            -   If asking a question, match your **Questioning Style**.
            -   **Natural Language:** Subtly and naturally integrate your user's **Common Fillers/Catchphrases** where they make sense. Do NOT force them. The goal is to sound like a real person, not a caricature. A simple, direct sentence is better than an awkward one with a forced filler.


        ## Output Format
        - The response MUST be a single, valid JSON object with the following exact format:
        ```json
        {{ 
            "thoughts": "<Your detailed reasoning here, explicitly following the decision-making framework step-by-step>", 
            "response": "<Your generated utterance here, or an empty string>" 
        }}
        ```
        """

        # User prompt template
        user_prompt_template = """Here is the specific information for your current task. You'll also be provided some examples to guide your response.
        ---
        ## EXAMPLES

        ### Example 1: The Proactive Technical Expert (Chime-In)

        **MY USER PROFILE (<Senior Engineer>)**
        ### My Role & High-Level Goals
        - **Primary Role:** Technical Expert (Backend Systems)
        - **High-Level Goals:** ["Ensure technical feasibility and scalability of proposed features.", "Proactively identify and flag potential performance bottlenecks."]
        ### My Core Knowledge
        - **General Knowledge Areas:** ["Database Optimization", "API Design", "Caching Strategies"]
        - **Specific Knowledge Points:**
        [
            {{ "Context": "When discussing features that require fetching large user lists", "Information": "The current user database schema has an unindexed field for 'last_login_date'. A query filtering on this field will be very slow and will not scale." }}
        ]
        ### My Communication Style
        - **Formality:** Semi-formal
        - **Verbosity:** Concise
        - **Typical Tone:** Factual and direct
        - **Interaction Patterns:** ["Corrects technical inaccuracies", "Speaks up when a technical risk is identified"]

        **TODAY'S JIT BRIEFING (<Senior Engineer>)**
        - "The proposed admin feature for 'inactive users' will require a full table scan on an unindexed 'last_login_date' field, which will not scale."
        - "The data backfill script is a major risk and needs to be parallelized."

        **MEETING CONTEXT**
        - **Attendees:** ['Product Lead', 'UX Designer', 'Senior Engineer']
        - **Ongoing Meeting Transcript Snapshot:**
        ...
        Product Lead: ...and for the new admin dashboard, the top request is to show a list of all users who haven't logged in for 90 days, so we can send them a re-engagement email.
        UX Designer: Okay, that's straightforward. I'll mock up a simple table view for that admin page. It should be an easy feature to implement.

        **Desired Output:**
        ```json
        {{
            "thoughts": "I am representing Senior Engineer, whose primary role is a 'Technical Expert (Backend Systems)'.\nStep 0. Context & State Digestion: The micro-topic is the implementation of an 'inactive users' feature. The conversational state is 'Brainstorming solutions'.\nStep 1. Social Appropriateness Check: The floor is open, as the last utterance was a statement of intent, not a question to someone else. The check passes.\nStep 2. Opportunity Identification: The UX Designer's comment about an 'easy feature' opens the door to add a new, valuable technical point. This is a `Chime_In` opportunity.\nStep 3. Content Selection & Prioritization: My JIT briefing has two points. I'll check if they've been discussed. The snapshot does not mention the 'unindexed query' or the 'backfill script'. The 'unindexed query' point is directly relevant to the 'inactive users' feature. I will select this single, most impactful point, using a 'Concise' scope.\nStep 4. Persona Alignment: The action to `Chime_In` to highlight a technical risk aligns perfectly with my user's 'speaks up when a technical risk is identified' pattern.\nStep 5. Final Action & Response Plan: My final action is `Chime_In`. I will state the selected risk concisely and factually.",
            "response": "Just a heads-up on that feature, there's a technical risk. The query for inactive users will run on an unindexed 'last_login_date' field, which will cause a major performance bottleneck. We'll need to address that before it's built."
        }}
        ```

        ### Example 2: The Strategic Manager (Responding to Explicit Cue)

        **MY USER PROFILE (<Product Lead>)**
        ### My Role & High-Level Goals
        - **Primary Role:** Product Lead / Strategist
        - **High-Level Goals:** ["Keep the team focused on the core user problem", "Ensure features provide measurable user value"]
        ### My Communication Style
        - **Typical Tone:** Inquisitive and user-focused
        - **Questioning Style:** "Asks open-ended, strategic questions that reframe the conversation"
        - **Common Fillers/Catchphrases:** ["That's interesting", "Let's take a step back"]

        **TODAY'S JIT BRIEFING (<Product Lead>)**
        - "Key user data point: The #1 user complaint about the payments module is the checkout flow's speed, not its reliability or technical debt."
        - "Business goal for Q3: Increase payment conversion rate by 5%."

        **MEETING CONTEXT**
        - **Attendees:** ['Product Lead', 'Engineer A', 'Engineer B', 'Manager']
        - **Ongoing Meeting Transcript Snapshot:**
        Engineer A: I spent this week refactoring the login module to use the new Singleton pattern. It's much cleaner now.
        Engineer B: I can confirm, the code complexity is way down. It's a big improvement for maintainability.
        Engineer A: So, I think we should allocate another sprint to refactor the payments module next, to address its technical debt.
        Manager: That sounds like a good technical improvement. Product Lead, any concerns from your end?

        **Desired Output:**
        ```json
        {{
            "thoughts": "I am representing Product Lead, whose primary role is a 'Product Lead / Strategist'.\nStep 0. Context & State Digestion: The micro-topic is a proposal to refactor the payments module to fix technical debt. The state is 'Decision-making'.\nStep 1. Social Appropriateness Check: The last utterance is a direct question to me. The check passes.\nStep 2. Opportunity Identification: This is a clear `Respond_To_Explicit_Cue`.\nStep 3. Content Selection & Prioritization: I must respond. The snapshot discusses 'technical debt', but my JIT briefing point about user complaints regarding checkout SPEED has not been mentioned. This briefing point is the single most impactful point to raise to ensure the engineering work provides user value. I will use a 'Concise' scope.\nStep 4. Persona Alignment: Responding with a strategic question aligns with my user's `questioning_style` and `high_level_goals`.\nStep 5. Final Action & Response Plan: My final action is `Respond_To_Explicit_Cue`. I will use my JIT briefing point to ask a strategic question that reframes the discussion around user value.",
            "response": "That's interesting, and improving maintainability is definitely valuable. My main concern is ensuring our engineering effort is focused on the biggest user problem. Our latest user data shows the top complaint about payments is the checkout flow's speed. Before we commit to a general refactor, can we clarify how that work would directly address improving that speed?"
        }}
        ```
    ### Example 3: The Collaborative Designer (Responding to Implicit Cue)

        **MY USER PROFILE (<UX Designer>)**
        ### My Role & High-Level Goals
        - **Primary Role:** UX Designer / User Advocate
        - **High-Level Goals:** ["Ensure the product is intuitive and easy to use"]
        ### My Core Knowledge
        - **General Knowledge Areas:** ["Usability Testing", "UI/UX Best Practices", "Information Architecture"]
        - **Specific Knowledge Points:**
        [
            {{ "Context": "General design principle", "Information": "Users often prefer clarity and organization over seeing all information at once, even if it requires an extra click." }}
        ]
        ### My Communication Style
        - **Typical Tone:** Collaborative and constructive
        - **Interaction Patterns:** ["Bridges technical features to user impact", "Often refers to user feedback or testing"]

        **TODAY'S JIT BRIEFING (<UX Designer>)**
        - "Critical user feedback for settings page: In our *most recent* usability test last week, 3 out of 5 users reported that the single, long page prototype was confusing and overwhelming."
        - "Goal for this meeting: Finalize a decision on the settings page layout, strongly advocating for the option that tested best with users."
        - "A key question to ask: 'How will this technical choice impact users with accessibility needs, like screen readers?'"

        **MEETING CONTEXT**
        - **Attendees:** ['UX Designer', 'Engineer A', 'Engineer B', 'Product Lead']
        - **Ongoing Meeting Transcript Snapshot:**
        Product Lead: Okay, great discussion on the login flow. Let's move on to the settings page redesign. Engineer A, you had the initial technical proposal, right?
        Engineer A: I do. The main decision point for the architecture is the overall layout. For the settings page, we can either use a single, long page with expandable sections, or a more traditional multi-tab interface.
        Product Lead: What are the pros and cons of each from your perspective?
        Engineer A: The single page is faster to code and probably has a slightly better initial load time.
        Engineer B: True, but the multi-tab interface is more organized and scales better if we add more settings categories in the future. It's a classic speed versus structure trade-off.

        **Desired Output:**
        ```json
        {{
            "thoughts": "I am representing UX Designer, whose primary role is a 'User Advocate'.\nStep 0. Context & State Digestion: The micro-topic is a technical trade-off for the settings page layout. The state is 'Brainstorming solutions'. Engineer B explicitly mentioned uncertainty about the user's preference.\nStep 1. Social Appropriateness Check: The floor is open to resolve the stated trade-off. The check passes.\nStep 2. Opportunity Identification: Engineer B's statement 'I'm not sure which is better for the user' creates a clear knowledge vacuum that my user is expected to fill. This is a strong `Implicit Cue`.\nStep 3. Content Selection & Prioritization: My JIT briefing has a critical data point about user feedback on this exact issue. This point has not been discussed yet in the snapshot. It is the single, most relevant point to resolve the team's uncertainty. I will use a 'Concise' scope.\nStep 4. Persona Alignment: The action `Respond_To_Implicit_Cue` by providing user data aligns perfectly with my user's pattern of 'referring to user feedback or testing'.\nStep 5. Final Action & Response Plan: My final action is `Respond_To_Implicit_Cue`. I will provide the user feedback from my briefing with a collaborative tone.",
            "response": "That's a helpful breakdown. I can add some fresh data to that: in the usability test we ran last week, users found the idea of a single, long page overwhelming. The feedback suggests a multi-tab interface would be the better choice here to keep things organized and less intimidating for them."
        }}
        ```

        ### Example 4: Strategic Silence (Due to Social Appropriateness / Direct Cue to Another)

        **MY USER PROFILE (<Senior Engineer>)**
        ### My Role & High-Level Goals
        - **Primary Role:** Technical Expert (Backend Systems)
        - **High-Level Goals:**
        - "Ensure technical feasibility and scalability of proposed features."
        ### My Core Knowledge
        - **General Knowledge Areas:** ["Database Performance", "API Design", "Caching Strategies"]
        - **Specific Knowledge Points:**
        [
            {{ "Context": "When discussing database queries", "Information": "Indexing the user_id column in the main table would cut the dashboard query time by at least 50%." }}
        ]
        ### My Communication Style
        - **Interaction Patterns:** ["Provides data-driven facts", ""Waits for a clear pause before speaking"]"]

        **TODAY'S JIT BRIEFING (<Senior Engineer>)**
        - "Main point for today: The new data pipeline must be designed to handle at least 100,000 events per second."

        **MEETING CONTEXT**
        - **Attendees:** ['Manager', 'Product Lead', 'Senior Engineer', 'Marketing Lead']
        - **Ongoing Meeting Transcript Snapshot:**
        Manager: Okay, the next agenda item is the Q4 marketing campaign slogan.
        Product Lead: The branding team has proposed three options.
        Manager: Great. Marketing Lead, can you walk us through the pros and cons of each slogan?

        **Desired Output:**
        ```json
        {{
            "thoughts": "I am representing Senior Engineer, whose primary role is a 'Technical Expert'.\nStep 0. Context & State Digestion: The micro-topic is debugging a slow dashboard query. The state is 'Problem diagnosis'.\nStep 1. Social Appropriateness Check: The last utterance is a direct question to a specific participant, 'Junior Engineer'. I MUST NOT interrupt. This rule is the highest priority.\n**Final Action:** `Remain_Silent`.",
            "response": ""
        }}
        ```
        ---
        ## Specific Information for your task:
        
        **1. MY USER PROFILE (<{person_name}>)**
        This describes my user's long-term persona: their general role, goals, communication style, and foundational knowledge.
        {profile_string}
        **2. TODAY'S 'JustInTime'(JIT) BRIEFING (Important points to discuss in this meeting, if and when relevant, and if not already discussed in the provided context)**
        This provides specific, context-sensitive talking points and knowledge relevant to *this specific segment* of the meeting. This information is high priority.
        {jit_briefing_string}
        ---
        **3. MEETING CONTEXT**
        - **Attendees:** {attendees_string}
        - **Ongoing Meeting Transcript Snapshot:**
        {meeting_transcript_snapshot_string}
        ---
        Now, apply your framework, representing the given user properly, and provide your response in the required JSON format.
        """

        user_prompt = user_prompt_template.format(
            person_name=person_name,
            profile_string=profile_str,
            jit_briefing_string=jit_briefing_str,
            attendees_string=attendees_str,
            meeting_transcript_snapshot_string=snapshot_str
        )

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        attempt = 0
        while attempt < max_attempts:
            try:
                print(f"  Attempt {attempt + 1}/{max_attempts} to generate delegate response...")
                
                # Get JSON response using structured output (critical for GPT-5-mini)
                json_response = self.llm_client.call_with_retry(
                    messages=messages,
                    max_tokens=2000,
                    json_required=True,
                    response_model=DelegateResponse,  # Structured output for GPT-5-mini/Gemini
                    telemetry_label="single_tier.delegate",
                    telemetry_phase="floor",
                )
                
                if not json_response:
                    print(f"  - No JSON response for {person_name}")
                    continue
                
                # Handle both Pydantic model (structured output) and raw JSON (fallback)
                if hasattr(json_response, 'model_dump'):
                    response = json_response  # Already a DelegateResponse
                else:
                    # Manually validate the JSON against the delegate response schema (fallback)
                    response = self.llm_client.validate_json_response(json_response, DelegateResponse)
                
                if response:
                    print(f"  ✅ Successfully generated and validated response for {person_name}")
                    return response
                else:
                    print(f"  - JSON validation failed for {person_name}")
                
            except (ValidationError, json.JSONDecodeError) as e:
                print(f"  - VALIDATION ERROR: {e}")
                print(f"  - Raw JSON: {json_response[:200] if 'json_response' in locals() else 'N/A'}...")
            except Exception as e:
                print(f"  - API/OTHER ERROR: {e}")

            attempt += 1
            import time
            time.sleep(base_delay * (2 ** attempt))

        print(f"Failed to get a valid response for {person_name} after {max_attempts} attempts.")
        return None

    def run_simulation_for_participant(self,
                                     test_cases: List[Dict[str, Any]],
                                     jit_briefing_points: List[str],
                                     participant_id: str,
                                     meeting_id: str) -> List[Dict[str, Any]]:
        """
        Run the single-tier agent simulation for all test cases of a participant.
        
        Args:
            test_cases: List of test case dictionaries
            jit_briefing_points: JIT briefing points for this session
            participant_id: ID of the participant being simulated
            meeting_id: ID of the meeting
            
        Returns:
            List of results with test cases and delegate outputs
        """
        print(f"Starting single-tier engagement simulation for {len(test_cases)} test cases for '{participant_id}'...")

        results = []
        for i, case in enumerate(test_cases):
            print(f"\n--- Processing Test Case {i+1}/{len(test_cases)} ({case.get('test_case_id', 'N/A')}) ---")

            delegate_output = self.run_engagement_module(
                test_case=case,
                jit_briefing_points=jit_briefing_points
            )

            delegate_output_dict = delegate_output.model_dump() if delegate_output else {"error": "Failed to generate valid response."}

            result_entry = {
                "test_case": case,
                "delegate_output": delegate_output_dict
            }
            results.append(result_entry)

            if delegate_output:
                print(f"  > Delegate Thoughts: {delegate_output.thoughts[:250]}...")
                print(f"  > Delegate Response: '{delegate_output.response}'")
                print(f"  > Ground Truth:      '{case['ground_truth_response_text']}'")
            else:
                print("  > Delegate failed to produce a valid output.")

        print(f"\n{'='*20}\nFinished processing single-tier simulation for {participant_id}.\n{'='*20}")
        return results
