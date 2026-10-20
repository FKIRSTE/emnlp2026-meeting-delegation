"""
Test Case Generation Module for Meeting Delegate System

This module handles extracting test case instances from meeting transcript segments.
"""

import json
import time
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict
from pydantic import ValidationError

# Add current directory to path for robust importing
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from llm_client import LLMClient
from schemas import TestCaseExtractionResponse, TestCaseInstance


class TestCaseGenerator:
    """
    Test case generation - extracted from cells 15-18
    Generates evaluation instances (cue/response pairs) from meeting transcripts
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def extract_test_case_instances(
        self,
        person_id: str,
        meeting_transcript_segment_string: str,
        max_attempts: int = 3,
        base_delay: float = 2.0
    ) -> Optional[TestCaseExtractionResponse]:
        """
        Calls an LLM with a refined, robust prompt to extract test case instances
        (pointers to cue/response pairs) from a transcript segment.
        
        Args:
            person_id: The ID/name of the participant to generate test cases for
            meeting_transcript_segment_string: Formatted transcript segment
            max_attempts: Maximum retry attempts for generation
            base_delay: Base delay between retries
            
        Returns:
            Validated TestCaseExtractionResponse or None if failed
        """
        
        # System prompt: Role, capabilities, and evaluation framework
        system_prompt = """You are an NLP expert agent specializing in meeting conversation analysis. Your task is to generate evaluation datasets by identifying response instances from specific participants within meeting transcripts. You will categorize these instances as 'Chime In', 'Explicit Cue', or 'Implicit Cue'.

        ## Core Competency
        You excel at identifying substantive conversational responses and accurately categorizing the interaction patterns that trigger them. Your focus is exclusively on meaningful contributions that exclude simple acknowledgments unless they are part of a larger, substantial response.

        ## Data Format Understanding
        - Transcripts use specific speaker identifiers (e.g., Project Manager, Industrial Designer)
        - Each utterance has a unique ID in format (MeetingID_XXX), where MeetingID is the meeting identifier and XXX is the utterance number
        - Example: (Meeting21_007) means utterance 007 from Meeting21

        ## Evaluation Type Definitions
        **'Chime In'**: Target participant spontaneously contributes to the conversation without being directly prompted. This occurs when the participant wasn't actively engaged but chimes in with a relevant comment or question.

        **'Explicit Cue'**: Target participant's speaker ID/name is specifically mentioned in the cueing utterance (ID), and they respond directly to that explicit prompt.

        **'Implicit Cue'**: Target participant's speaker ID/name is NOT mentioned, but a general question or statement strongly implies a response is expected from them (due to their role/expertise), and they respond appropriately.

        ## Output Format Requirements
        Your output MUST be a single, valid JSON object with the below specified structure. The value of "extracted_data" MUST be a JSON list of evaluation instances. Each instance in the list MUST be a JSON object with the following structure:
        ```json
        {
            "extracted_data": [
                {
                "Explanation": "(string) Your reasoning for selecting this evaluation instance and categorizing it",
                "Type": "(string) Must be exactly one of: 'Chime In', 'Explicit Cue', or 'Implicit Cue'",
                "Response_IDs": ["(list of strings) Utterance IDs of the target participant's response. Include all Response_IDs that are relevant to the evaluation instance."],
                "ID": "(string) Single utterance ID of the cue that triggered the response", 
                "Speaker": "(string) Speaker ID of the utterance corresponding to the cue ID"
                }
            ]
        }
        ```

If no suitable instances are found, return: `{"extracted_data": []}`

## Critical Requirements
1. **ID Accuracy**: All utterance IDs must exactly match the format seen in the transcript. Do NOT alter, misspell, or hallucinate IDs.
2. **Chronological Order**: The cue utterance (ID) must chronologically precede all response utterances (Response_IDs).
3. **Strict Attribution**: Response_IDs must belong only to the target participant; ID must be from the speaker who triggered the response.
4. **Comprehensiveness**: Scan the entire transcript and identify ALL suitable evaluation instances.

## Reference Examples

### Example 1: Multi-turn "Chime In" Pattern
**Target Participant**: "Speaker 18"

**Transcript Segment**:
```
(Speaker 19): If you want, I can resend it again. (Meeting1_071)
(Speaker 13): Space tokeniser. <unintelligible/> Yes, so es essentially to answer your question in the email. We have to switch to and we have for the IWSLT. We have to switch to SacreBLEU and SacreBLEU does its own tokenisation before scoring. So there is no-. Let's let's simply forget NLTK bleu score. That is not reliable. (Meeting1_072)
(Speaker 18): Yes, but-. (Meeting1_073)
(Speaker 19): Yes, but we can combine our tokeniser with NLTK. (Meeting1_074)
(Speaker 13): ...Let's let's just use SacreBLEU. (Meeting1_075)
(Speaker 19): Okay. (Meeting1_076)
(Speaker 18): I I have one comment about it. (Meeting1_077)
(Speaker 13): Mhm. Yeah. (Meeting1_078)
(Speaker 18): You sh should use tokeniser before enverse segmenter. (Meeting1_079)
(Speaker 13): Yes, that's it. Yeah. (Meeting1_080)
(Speaker 18): Because it's much better. Because it can rely on the on the dots and commas and question marks and so on. And you can you can check my script which does tokeniser, enverse segmenter and then de-tokeniser... (Meeting1_081)
(Speaker 13): Yeah. (Meeting1_082)
(Speaker 18): And it's it's using the Moses seg tokeniser and detokeniser. (Meeting1_083)
(Speaker 13): Yeah. So [PERSON19], do you do you fo-? Do you understand? (Meeting1_084)
(Speaker 19): "Yeah. (Meeting1_085)
```

**Expected Output**:
```json
{
  "extracted_data": [
    {
      "Explanation": "Speaker 18 spontaneously contributes to the conversation about SacreBLEU without being directly prompted. Their contribution is substantive and spans multiple utterances, interjected with acknowledgments from Speaker 13. This is a Chime In.",
      "Type": "Chime In",
      "Response_IDs": ["Meeting1_077", "Meeting1_079", "Meeting1_081", "Meeting1_083"],
      "ID": "Meeting1_076",
      "Speaker": "Speaker 19"
    }
  ]
}
```

### Example 2: Multi-turn "Explicit Cue" Pattern
**Target Participant**: "Speaker 18"

**Transcript Segment**:
```
(Speaker 13): ...So that we meet the internal deadline on the 8th. Yeah, okay. Great. Uh. So Speaker 18, what what is your progress? (Meeting2_117)
(Speaker 18): Hmhm. Yes, and by reading the papers I found an interesting tool. (Meeting2_118)
(Speaker 13): Mhm. (Meeting2_119)
(Speaker 18): I found out that it's possible to measure out the speech rate by cutting the syllables. And there is one tool...which can detect the gender of speaker and the speech rate. (Meeting2_120)
(Speaker 13): Mhm. (Meeting2_121)
(Speaker 18): And some other characteristics. So we can try it and make a dashboard out of it. (Meeting2_122)
(Speaker 13): Mhm. That's that's useful thing... (Meeting2_123)
(Speaker 18): Yes. (MeetingSample2_utt_124)
(Speaker 13): So we could be reducing reducing their speech mole with a different model. (Meeting2_125)
(Speaker 18): Yes, and there was also speech modes. Like whether it was angry or normal and so on. (Meeting2_126)
(Speaker 13): Mhm. (Meeting2_127)
(Speaker 18): But I have no idea how the tool works in practice... (Meeting2_128)
(Speaker 13): Yeah, uh. (Meeting2_129)
(Speaker 18): So we can try it and make a dashboard out of it. (Meeting2_130)
```

**Expected Output**:
```json
{
  "extracted_data": [
    {
      "Explanation": "Speaker 18's utterances are informative and meaningful for the evaluation dataset. Speaker 18 was directly prompted by Speaker 13 by name with a question about their progress. This is an Explicit Cue. Speaker 18's response is a long, multi-turn explanation about a tool they discovered, spanning several utterances.",
      "Type": "Explicit Cue", 
      "Response_IDs": ["Meeting2_118", "Meeting2_120", "Meeting2_122", "Meeting2_124", "Meeting2_126", "Meeting2_128", "Meeting2_130"],
      "ID": "Meeting2_117",
      "Speaker": "Speaker 13"
    }
  ]
}
```
These examples demonstrate the three interaction patterns you should identify and how to structure the output for multi-turn responses."""

        # User prompt: Specific task with input data
        user_prompt = """Please analyze the meeting transcript below to extract all relevant evaluation instances for the target participant: **{person_id}**

        ## Your Task
        Identify all substantive response instances from {person_id} in the provided transcript. For each instance:
        1. Determine the interaction type (Chime In, Explicit Cue, or Implicit Cue)
        2. Identify the triggering utterance (cue)
        3. Collect all relevant response utterances from {person_id}
        4. Provide clear reasoning for your categorization

        Focus on meaningful contributions that would be valuable for evaluating an AI delegate's ability to represent {person_id} authentically.

        ## Input Transcript
        Each line follows the format: (speaker_id): utterance_content (utterance_id)

        <Transcript>
        {meeting_transcript_segment_string}
        </Transcript>

        Analyze the transcript above and extract all evaluation instances for {person_id}. Return your findings as a JSON object with the "extracted_data" key containing a list of instances."""

        # Escape curly braces in the transcript content to prevent format string issues
        escaped_transcript = meeting_transcript_segment_string.replace('{', '{{').replace('}', '}}')
        
        filled_user_prompt = user_prompt.format(
            person_id=person_id, 
            meeting_transcript_segment_string=escaped_transcript
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": filled_user_prompt}
        ]

        attempt = 0
        while attempt < max_attempts:
            try:
                print(f"  Attempt {attempt + 1}/{max_attempts} to extract test case instances...")
                
                # Use LLM client with structured output for GPT-5-mini compatibility
                response_content = self.llm_client.call_model(
                    messages=messages,
                    max_tokens=3000,  # Large token limit for comprehensive extraction
                    json_required=True,
                    response_model=TestCaseExtractionResponse,  # Structured output for GPT-5-mini/Gemini
                )
                
                # Handle both Pydantic model (structured output) and raw JSON (fallback)
                if hasattr(response_content, 'model_dump'):
                    return response_content  # Already a TestCaseExtractionResponse
                
                # Use the robust validation method with auto-fix capabilities (fallback)
                validated_response = self.llm_client.validate_json_response(response_content, TestCaseExtractionResponse)
                return validated_response
                
            except (ValidationError, json.JSONDecodeError) as e:
                print(f"  Validation/JSON parsing failed for {person_id}. Error: {e}")
                print(f"  LLM Raw Output: {response_content}")
            except Exception as e:
                print(f"  An API call or other error occurred for {person_id}: {e}")

            attempt += 1
            time.sleep(base_delay * (2 ** attempt))

        print(f"Failed to extract valid test cases for {person_id} after {max_attempts} attempts.")
        return None

    def format_utterances_for_prompt(self, utterance_list: List[Dict]) -> str:
        """Formats a list of utterance objects into a single string for the LLM prompt."""
        return "\\n".join([
            f"({utt['speaker_id']}): {utt['utterance_content']} ({utt['utterance_id']})" 
            for utt in utterance_list
        ])

    def generate_test_cases_for_meeting(
        self,
        transcript_path: str,
        user_profiles_directory: str,
        output_directory: str,
        split_ratio: float = 0.6,
        context_window_size: int = 20,
        meeting_id: str = None,
        target_participant: str = None
    ) -> Dict[str, str]:
        """
        Generate test cases for participants with existing profiles in a meeting transcript.
        
        Args:
            transcript_path: Path to parsed transcript JSON file
            user_profiles_directory: Directory containing user profile files
            output_directory: Directory to save test case files
            split_ratio: Ratio for profile generation vs test case generation (default 0.6)
            context_window_size: Size of context window for test case snapshots (default 20)
            meeting_id: Meeting ID to use for finding profiles (if None, extracts from filename)
            target_participant: If specified, only generate test cases for this participant
            
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

        # Create a quick lookup map for utterance index by ID for efficiency
        utterance_index_map = {utt['utterance_id']: i for i, utt in enumerate(full_transcript)}

        # Split the transcript into Part A (profiling source) and Part B (testing source)
        split_index = int(len(full_transcript) * split_ratio)
        testing_segment = full_transcript[split_index:]

        if not testing_segment:
            print(f"Error: Testing segment (Part B) of the transcript is empty.")
            return {}

        # Find participants who have existing profiles generated from Part A of this transcript
        participants_with_profiles = []
        
        # Use provided meeting_id or extract from filename as fallback
        if meeting_id:
            meeting_id_base = meeting_id
        else:
            meeting_id_base = os.path.basename(transcript_path).replace('_parsed.json', '')
        
        for filename in os.listdir(user_profiles_directory):
            if f"_profile_from_{meeting_id_base}_partA.json" in filename:
                speaker_id = filename.split('_profile_from_')[0].replace('_', ' ')
                participants_with_profiles.append(speaker_id)

        if not participants_with_profiles:
            print(f"No user profiles found for '{os.path.basename(transcript_path)}'. Please run the profile generation script first.")
            return {}

        # Filter participants if target_participant is specified
        if target_participant:
            if target_participant in participants_with_profiles:
                participants_with_profiles = [target_participant]
                print(f"Found profiles for participants: {participants_with_profiles} (filtered to target: {target_participant})")
            else:
                print(f"❌ Target participant '{target_participant}' not found in available profiles: {participants_with_profiles}")
                return {}
        else:
            print(f"Found profiles for participants: {participants_with_profiles}")

        # Format the testing segment (Part B) into a single string to be fed to the LLM
        testing_segment_string = self.format_utterances_for_prompt(testing_segment)

        # Ensure output directory exists
        os.makedirs(output_directory, exist_ok=True)

        # Generate test cases for each participant
        generated_test_cases = {}

        for participant_id in participants_with_profiles:
            print(f"\\n{'='*20}\\nExtracting Test Cases for: {participant_id}\\n{'='*20}")

            # Load this participant's specific profile
            profile_filename = f"{participant_id.replace(' ', '_')}_profile_from_{meeting_id_base}_partA.json"
            profile_path = os.path.join(user_profiles_directory, profile_filename)
            
            try:
                with open(profile_path, 'r', encoding='utf-8') as f:
                    user_profile_data = json.load(f)
                    
                    # Handle both legacy and CAPA profile formats
                    if 'UserProfile' in user_profile_data:
                        # Legacy format: {"UserProfile": "..."}
                        user_profile_content = user_profile_data['UserProfile']
                    else:
                        # CAPA format: direct JSON structure or string
                        # Convert the entire profile structure to a formatted string
                        user_profile_content = json.dumps(user_profile_data, indent=2)
                    
            except FileNotFoundError:
                print(f"  Warning: Could not load profile for {participant_id}. Skipping.")
                continue
            except Exception as e:
                print(f"  Warning: Error loading profile for {participant_id}: {e}. Skipping.")
                continue

            # Call the LLM to get the instance pointers from Part B
            response = self.extract_test_case_instances(
                person_id=participant_id,
                meeting_transcript_segment_string=testing_segment_string
            )

            if not response or not response.extracted_data:
                print(f"  No suitable test cases found by the LLM for {participant_id} in Part B.")
                continue

            llm_instances = response.extracted_data
            print(f"  LLM identified {len(llm_instances)} potential instances for {participant_id}.")

            # Assemble and Validate Final Test Cases
            final_test_cases = []
            for i, instance in enumerate(llm_instances):
                cue_id = instance.ID

                # CRITICAL VALIDATION STEP
                # Ensure the cue ID and all response IDs from the LLM actually exist in the full transcript
                if cue_id not in utterance_index_map:
                    print(f"  - WARNING: LLM-provided cue ID '{cue_id}' does not exist in the full transcript. Skipping instance.")
                    continue

                all_response_ids_valid = True
                for resp_id in instance.Response_IDs:
                    if resp_id not in utterance_index_map:
                        print(f"  - WARNING: LLM-provided response ID '{resp_id}' does not exist. Skipping instance.")
                        all_response_ids_valid = False
                        break
                if not all_response_ids_valid:
                    continue

                # Construct snapshot with fixed window
                cue_index = utterance_index_map[cue_id]
                start_index = max(0, cue_index - context_window_size + 1)
                snapshot_utterances = full_transcript[start_index : cue_index + 1]

                # Reconstruct ground truth response text
                response_texts = [full_transcript[utterance_index_map[resp_id]]['utterance_content'] for resp_id in instance.Response_IDs]

                final_test_case = {
                    "test_case_id": f"{meeting_id_base}_{participant_id.replace(' ', '_')}_case_{i+1:03d}",
                    "principal_participant_id": participant_id,
                    "attendees": sorted(list(set(utt['speaker_id'] for utt in full_transcript))),
                    "input_snapshot_utterances": snapshot_utterances,
                    "ground_truth_response_text": " ".join(response_texts),
                    "ground_truth_user_profile": user_profile_content,
                    "scenario_info_from_llm": instance.model_dump()  # Convert Pydantic model to dict for JSON serialization
                }
                final_test_cases.append(final_test_case)

            # Save the assembled test cases to a file
            if final_test_cases:
                output_filename = f"test_cases_for_{participant_id.replace(' ', '_')}_from_{meeting_id_base}.json"
                output_path = os.path.join(output_directory, output_filename)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(final_test_cases, f, indent=2, ensure_ascii=False)
                print(f"--> SUCCESS: Saved {len(final_test_cases)} validated test cases for {participant_id} to '{output_path}'")
                generated_test_cases[participant_id] = output_path
            else:
                print(f"--> No valid test cases generated for {participant_id}.")

        return generated_test_cases

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
            return json.dumps(profile, indent=2)
        return str(profile)
