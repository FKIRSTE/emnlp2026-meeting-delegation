"""
User Profile Generation Module for Meeting Delegate System

This module handles generating rich user profiles from meeting transcript segments.
Extracted from cells 11-12 in the original notebook.
"""

import json
import time
import sys
from pathlib import Path
from typing import Optional, List, Dict
from pydantic import ValidationError

# Add current directory to path for robust importing
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from llm_client import LLMClient
from schemas import RichProfileGenerationResponse, UserProfile


class UserProfileGenerator:
    """
    User profile generation - extracted from cells 11-12
    Generates comprehensive user profiles from meeting transcript analysis
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        
    def generate_rich_user_profile(
        self,
        participant_speaker_id: str,
        meeting_transcript_segment_string: str,
        max_attempts: int = 3,
        base_delay: float = 2.0
    ) -> Optional[RichProfileGenerationResponse]:
        """
        Calls an LLM with a detailed prompt to generate a comprehensive user profile
        from a provided transcript segment.
        
        Args:
            participant_speaker_id: The ID/name of the participant to profile
            meeting_transcript_segment_string: Formatted transcript segment
            max_attempts: Maximum retry attempts for generation
            base_delay: Base delay between retries
            
        Returns:
            Validated RichProfileGenerationResponse or None if failed
        """
        
        # System prompt: Role, capabilities, and output format
        system_prompt = """You are an expert AI analyst specializing in organizational psychology and communication. Your task is to synthesize a user's underlying persona from their dialogue to create a comprehensive and **generalizable** user profile. This profile should model the participant's dispositions and role, not just be a literal log of their speech. The goal is to create a comprehensive user profile that an AI delegate can use to represent this participant authentically.

        ## Important Notes About Transcript Format
        - The transcript may contain disfluency markers like {vocalsound}, {gap}, {disfmarker}, etc.,  which indicate non-verbal sounds, pauses, and speech hesitations
        - These markers provide valuable context about meeting dynamics and could be considered in your analysis as well.
        - CRITICAL: In your JSON output, do NOT include any curly braces except for the required JSON structure. If at all you need to reference these markers in your analysis text, use alternative notation like [vocalsound], [gap], [disfmarker].

        ## Core Task: Synthesis from Observation to Abstraction
        Your goal is to infer abstract patterns from concrete examples. The final profile should be a model of "who the user is" that allows an AI delegate to act authentically in new situations.

        ## Output Structure Requirements
        The output MUST be a single, valid JSON object that strictly adheres to the following schema:
        - **"Thoughts"** key: Explain your step-by-step reasoning for each part of the profile. First list your raw observations (key things the user said/did), then explicitly show your reasoning as you synthesize these observations into the higher-level profile fields (like `primary_role`, `high_level_goals`, `interaction_patterns` and so on). This demonstrates your reasoning chain.
        - **"UserProfile"** key: A JSON object containing:
            - **"inferred_role_and_goals"**: (object) The participant's high-level purpose.
                - "primary_role": (string) A summary of their main function (e.g., 'Meeting Facilitator', 'Technical Expert', ...).
                - "high_level_goals": (list of strings) Their overarching objectives in the meeting.
            - **"core_info"**: (object) The substance of their contributions.
                - "knowledge_areas": (list of strings) General topics the participant is knowledgeable about.
                - "specific_knowledge_points": (list of objects) Each object must be `{"Context": "...", "Information": "..."}`.
            - **"communication_style"**: (object) The participant's linguistic and interactive persona.
                - "formality": (string) e.g., 'Formal', 'Semi-formal', 'Informal'.
                - "verbosity": (string) e.g., 'Concise', 'Moderate', 'Verbose'.
                - "typical_tone": (string) e.g., 'Analytical', 'Collaborative', 'Questioning'.
                - "jargon_usage": (string) A description of their use of technical language.
                - "questioning_style": (string) A description of how and how often they ask questions.
                - "common_fillers_catchphrases": (list of strings, can be empty).
                - "interaction_patterns": (list of strings) Observed patterns like 'Builds on others ideas', 'Summarizes discussions'.
            - **"knowledge_stance"**: (object, can be omitted if no evidence)
            - **"identity_aliases"**: (list of strings, optional) Known names or nicknames used in the transcript that clearly refer to this participant (e.g., "Ruud"). Include only when strongly grounded by evidence; otherwise omit or use [].
                - "confidence_on_topics": (list of objects) Each must be `{"Topic_Category": "...", "Apparent_Confidence": "..."}`.
                - "expressed_stances": (list of objects) Each must be `{"Topic": "...", "Stance": "..."}`.

        ## Analysis Guidelines
        1. **Focus Exclusively on Target Participant**: Analyze only the target participant's contributions and style. Use utterances from other speakers only to understand context.
        2. **Prioritize Substantive Contributions**: Focus on utterances where the participant is providing information, asking significant questions, stating opinions, or making proposals. Ignore simple greetings and acknowledgments.
        3.  **Synthesize `inferred_role_and_goals`:** Based on the *pattern* of their substantive contributions,  determine their high-level purpose.
            -   **`primary_role`**: What is their main function or archetype in this meeting? (as an example, 'Technical Expert', 'Strategic Lead', 'Team Facilitator', 'Observer', ...).
            -   **`high_level_goals`**: What are the 2-4 (or more, if relevant) overarching objectives they are trying to achieve with their contributions? (as an example, "Ensure technical accuracy," "Drive the team towards a decision," "Maintain project scope", ...).
        4.   **Extract `core_info`:** This section captures the substance of what the participant knows and discusses.
            -   **`knowledge_areas`**: List the general topics or domains where the participant demonstrates clear expertise or knowledge. This defines the boundaries of what they can credibly talk about.
            -   **`specific_knowledge_points`**: Extract concrete facts, well-reasoned opinions, or specific proposals they shared. For each point, determine the generalized "Context" by analyzing preceding utterances. If a single point is made across a few consecutive utterances, synthesize them into a single, coherent "Information" entry.
        5.  **Characterize `communication_style`:** Based on recurring, observable patterns, describe their communication persona.
            -   Analyze their language for `formality`, `verbosity`, `typical_tone`, and `jargon_usage`.
            -   Analyze their dialogue for their `questioning_style`.
            -   Identify any truly characteristic `common_fillers_catchphrases`.
            -   **Critically, define their `interaction_patterns`**. This describes *how* they engage (e.g., "Waits for a clear pause before speaking," "Summarizes discussions to confirm consensus," "Politely interrupts to correct technical inaccuracies", ...). 
        6.  **Infer `knowledge_stance` (with caution):** Only if there is strong, repeated evidence, infer their apparent confidence on different topics and any clearly expressed stances.
            -   For each item in `expressed_stances`, you MUST use a JSON object with two keys: `"Topic"` and `"Stance"`
        7. **Maintain Strict Attribution**: All profile aspects must derive solely from the target participant's speech and observable interaction style.
        8. **Identity Aliases (if any)**: Include an alias only when there is clear evidence (explicit address followed by the participant's response, self-introduction, or repeated consistent usage). Prefer 1–3 short aliases; normalize capitalization; do not include names of other participants.

        ## Example Output Patterns

        ### Example 1: The Technical Expert ("Senior Data Scientist")
        ```json
        {
        "Thoughts": "Observations: User frequently provides specific data points, corrects technical inaccuracies, speaks formally with domain jargon. Synthesis: Primary role is 'Technical Expert' with goals of ensuring accuracy and providing data-driven insights.",
        "UserProfile": {
            "inferred_role_and_goals": {
            "primary_role": "Technical Expert",
            "high_level_goals": ["Ensure technical accuracy", "Provide data-driven insights"]
            },
            "communication_style": {
            "formality": "Formal",
            "verbosity": "Concise",
            "typical_tone": "Analytical",
            "interaction_patterns": ["Corrects technical inaccuracies", "Provides specific data points"]
            }
        }
        }
        ```

        ### Example 2: The Strategic Lead ("Product Lead")
        ```json
        {
            "Thoughts": "Observations: User consistently refocuses on user needs, summarizes discussions, asks strategic questions. Synthesis: Primary role is 'Strategic Lead' driving user-centric decisions.",
            "UserProfile": {
                "inferred_role_and_goals": {
                "primary_role": "Strategic Lead",
                "high_level_goals": ["Keep team focused on user needs", "Drive strategic decisions"]
                },
                "communication_style": {
                "formality": "Semi-formal",
                "verbosity": "Moderate",
                "typical_tone": "Inquisitive and user-focused",
                "interaction_patterns": ["Refocuses on strategic goals", "Summarizes viewpoints", "Asks strategic questions"]
                }
            }
        }
        ```

        ### Example 3: The Quiet Observer ("Junior Developer")
        ```json
        {
        "Thoughts": "Observations: User spoke minimally, only when directly asked, provided factual responses. Synthesis: Primary role is 'Observer' who listens actively and contributes on-demand.",
        "UserProfile": {
            "inferred_role_and_goals": {
            "primary_role": "Observer/Contributor on-demand",
            "high_level_goals": ["Listen and absorb information", "Provide specific facts when asked"]
            },
            "communication_style": {
            "formality": "Informal",
            "verbosity": "Very Concise",
            "typical_tone": "Neutral and factual",
            "interaction_patterns": ["Speaks only when directly addressed", "Uses short, factual statements"]
            }
        }
        }
        ```
    """

        # User prompt: Specific task with input data
        user_prompt = """Please analyze the meeting transcript segment below for the specific participant: **{participant_speaker_id}**

        ## Input Format
        - Each utterance is formatted as: (speaker_id): utterance_content (utterance_id)
        - All utterances are in chronological order
        - Multiple participants are present, but focus exclusively on {participant_speaker_id}

        ## Your Task
        Create a comprehensive user profile for {participant_speaker_id} based on their contributions in this transcript segment. Follow the analysis guidelines and output the result as a valid JSON object with "Thoughts" and "UserProfile" keys.

        <Meeting_Transcript_Segment>
        {meeting_transcript_segment_string}
        </Meeting_Transcript_Segment>

        Analyze {participant_speaker_id}'s contributions and generate their comprehensive user profile now.
        """

        # Escape curly braces in the transcript content to prevent format string issues
        escaped_transcript = meeting_transcript_segment_string.replace('{', '{{').replace('}', '}}')
        
        filled_user_prompt = user_prompt.format(
            participant_speaker_id=participant_speaker_id, 
            meeting_transcript_segment_string=escaped_transcript
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": filled_user_prompt}
        ]

        attempt = 0
        while attempt < max_attempts:
            try:
                print(f"  Attempt {attempt + 1}/{max_attempts} to generate profile...")
                
                # Use LLM client with structured output for GPT-5-mini compatibility
                # Note: GPT-5-mini uses internal reasoning tokens that count against output budget
                response_content = self.llm_client.call_model(
                    messages=messages,
                    max_tokens=8000,  # Increased for GPT-5-mini reasoning overhead
                    temperature=0.2,
                    json_required=True,
                    response_model=RichProfileGenerationResponse,  # Structured output for GPT-5-mini/Gemini
                )
                
                # Handle both Pydantic model (structured output) and raw JSON (fallback)
                if hasattr(response_content, 'model_dump'):
                    return response_content  # Already a RichProfileGenerationResponse
                
                # Use the robust validation method with auto-fix capabilities (fallback)
                validated_response = self.llm_client.validate_json_response(response_content, RichProfileGenerationResponse)
                return validated_response
                
            except ValidationError as e:
                print(f"  Pydantic validation failed. Errors: {e}")
                print(f"  LLM Raw Output: {response_content}")
            except Exception as e:
                print(f"  An API call or other error occurred: {e}")
                print(f"  Error type: {type(e).__name__}")
                if hasattr(e, 'args') and e.args:
                    print(f"  Error args: {e.args}")
                import traceback
                print(f"  Full traceback: {traceback.format_exc()}")

            attempt += 1
            time.sleep(base_delay * (2 ** attempt))

        print(f"Failed to generate a valid profile for {participant_speaker_id} after {max_attempts} attempts.")
        return None

    def format_utterances_for_prompt(self, utterance_list: List[Dict]) -> str:
        """Formats a list of utterance objects into a single string for the LLM prompt."""
        return "\\n".join([
            f"({utt['speaker_id']}): {utt['utterance_content']} ({utt['utterance_id']})" 
            for utt in utterance_list
        ])

    def generate_profiles_for_meeting(
        self,
        transcript_path: str,
        output_directory: str,
        split_ratio: float = 0.6
    ) -> Dict[str, str]:
        """
        Generate rich user profiles for all participants in a meeting transcript.
        
        Args:
            transcript_path: Path to parsed transcript JSON file
            output_directory: Directory to save profile files
            split_ratio: Ratio for profile generation segment (default 0.6)
            
        Returns:
            Dictionary mapping participant_id to output file path
        """
        import os
        
        # Load the preprocessed transcript
        if not os.path.exists(transcript_path):
            print(f"Error: Transcript file '{transcript_path}' not found.")
            return {}

        with open(transcript_path, 'r', encoding='utf-8') as f:
            full_transcript = json.load(f)

        if not full_transcript:
            print(f"Error: Transcript file '{transcript_path}' is empty.")
            return {}

        # Split the transcript into profile generation segment
        split_index = int(len(full_transcript) * split_ratio)
        profile_segment = full_transcript[:split_index]

        if not profile_segment:
            print(f"Error: Profile segment is empty. Check split ratio or transcript length.")
            return {}

        print(f"Profile segment contains {len(profile_segment)} utterances.")

        # Identify unique participants in the profiling segment
        participants_in_segment = sorted(list(set(utt['speaker_id'] for utt in profile_segment)))
        print(f"Found participants: {participants_in_segment}")

        # Format the profiling segment into a single string for the LLM
        transcript_context_string = self.format_utterances_for_prompt(profile_segment)

        # Ensure output directory exists
        os.makedirs(output_directory, exist_ok=True)

        # Generate profiles for each participant
        generated_profiles = {}
        meeting_id_base = os.path.basename(transcript_path).replace('_parsed.json', '')
        
        for speaker_id in participants_in_segment:
            print(f"\\n{'='*20}\\nGenerating Rich User Profile for: {speaker_id}\\n{'='*20}")

            profile_response = self.generate_rich_user_profile(
                participant_speaker_id=speaker_id,
                meeting_transcript_segment_string=transcript_context_string
            )

            if profile_response:
                # Save the generated profile to a file
                profile_filename = f"{speaker_id.replace(' ', '_')}_profile_from_{meeting_id_base}_partA.json"
                profile_path = os.path.join(output_directory, profile_filename)

                with open(profile_path, 'w', encoding='utf-8') as f:
                    f.write(profile_response.model_dump_json(indent=2))

                print(f"--> SUCCESS: Saved rich profile for {speaker_id} to '{profile_path}'")
                generated_profiles[speaker_id] = profile_path
            else:
                print(f"--> FAILED: Could not generate a valid profile for {speaker_id}.")

        return generated_profiles

    def extract_participant_utterances(self, transcript: List[Dict], speaker_id: str) -> List[Dict]:
        """
        Extract utterances from a specific participant from the transcript.
        Helper method for testing and external use.
        
        Args:
            transcript: List of utterance dictionaries
            speaker_id: The speaker ID to filter for
            
        Returns:
            List of utterances from the specified speaker
        """
        return [utt for utt in transcript if utt.get('speaker_id') == speaker_id]

    def format_utterances_for_prompt(self, utterances: List[Dict]) -> str:
        """
        Format a list of utterances for inclusion in LLM prompts.
        Helper method for testing and external use.
        
        Args:
            utterances: List of utterance dictionaries
            
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
