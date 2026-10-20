"""
Transcript Processing Module for Meeting Delegate System

This module handles parsing and preprocessing of QMSUM meeting transcripts.
"""

import os
import re
import json
import pandas as pd
from typing import List, Dict, Optional, Tuple


class TranscriptProcessor:
    """
    Simple transcript processing - extracted from cells 3-4
    Handles QMSUM transcript parsing and preprocessing pipeline
    """
    
    def parse_and_structure_transcript(self, raw_text: str, meeting_id: str) -> List[Dict]:
        """
        Parses a single QMSUM meeting transcript string, handles multi-line utterances,
        and returns a structured list of utterance objects with unique IDs.
        
        Args:
            raw_text (str): The full text of a single meeting transcript.
            meeting_id (str): A unique ID for the meeting (e.g., 'meetingA', 'ES2004a')
                             used to create globally unique utterance IDs.
        
        Returns:
            A list of dictionaries, where each dictionary represents an utterance.
        """
        utterances = []
        current_speaker = None
        current_content = ""
        
        if not isinstance(raw_text, str) or not raw_text.strip():
            print(f"Warning: Empty or invalid transcript text for {meeting_id}. Skipping.")
            return utterances
        
        lines = raw_text.strip().split('\n')
        utt_counter = 1
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Regex to capture "Speaker ID : Content" at the beginning of a line.
            # This is the primary pattern to identify a new utterance.
            match = re.match(r'^(.*?)\s*:\s*(.*)$', line)
            
            if match:
                # This line marks a new speaker's turn.
                # First, finalize any previous utterance
                if current_speaker and current_content.strip():
                    utterances.append({
                        'utterance_id': f"{meeting_id}_{utt_counter:03d}",
                        'speaker_id': current_speaker.strip(),
                        'utterance_content': current_content.strip()
                    })
                    utt_counter += 1
                
                # Start the new utterance
                current_speaker = match.group(1).strip()
                current_content = match.group(2).strip()
            else:
                # This line has no speaker pattern, so it's a continuation of the current utterance.
                if current_speaker:
                    # Append to the existing content
                    current_content += " " + line
                else:
                    # This line has no speaker, and there's no previous utterance to append to.
                    # This might be header text or a malformed start to the transcript.
                    print(f"Info: Skipping initial line with no speaker pattern in {meeting_id}: '{line}'")
        
        # Don't forget the final utterance
        if current_speaker and current_content.strip():
            utterances.append({
                'utterance_id': f"{meeting_id}_{utt_counter:03d}",
                'speaker_id': current_speaker.strip(),
                'utterance_content': current_content.strip()
            })
        
        return utterances
    
    def preprocess_qmsum_from_csv(
        self, 
        csv_filepath: str, 
        output_directory: str,
        meeting_indices: Optional[List[int]] = None
    ):
        """
        Reads a CSV file containing QMSUM meetings, parses specified meetings by index,
        and saves each parsed meeting as a separate JSON file.
        
        Args:
            csv_filepath (str): Path to the input CSV file.
            output_directory (str): Directory to save the parsed JSON files.
            meeting_indices (List[int], optional): Specific meeting indices to process.
                                                 If None, processes all meetings.
        """
        try:
            df = pd.read_csv(csv_filepath)
        except FileNotFoundError:
            print(f"Error: CSV file not found at {csv_filepath}")
            return
        
        if 'Meeting' not in df.columns:
            print("Error: CSV file must contain a 'Meeting' column.")
            return
        
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
            print(f"Created output directory: {output_directory}")
        
        # Determine which rows to process
        target_df = df
        if meeting_indices is not None:
            valid_indices = [i for i in meeting_indices if i < len(df)]
            if len(valid_indices) != len(meeting_indices):
                print(f"Warning: Some provided indices are out of bounds for the CSV with {len(df)} rows.")
            if not valid_indices:
                print("No valid indices to process.")
                return
            target_df = df.iloc[valid_indices]
        
        print(f"Starting preprocessing for {len(target_df)} selected meetings...")
        
        for index, row in target_df.iterrows():
            meeting_text = row['Meeting']
            # Use a column named 'id' for the meeting name if it exists, otherwise use the row index.
            # This makes the filenames and utterance IDs more meaningful.
            meeting_id = row.get('id', f"Meeting{index}")
            
            print(f"\nProcessing meeting '{meeting_id}' (from CSV row {index})...")
            
            parsed_utterances = self.parse_and_structure_transcript(meeting_text, meeting_id)
            
            if parsed_utterances:
                output_filename = os.path.join(output_directory, f"{meeting_id}_parsed.json")
                try:
                    with open(output_filename, 'w', encoding='utf-8') as f:
                        json.dump(parsed_utterances, f, indent=2, ensure_ascii=False)
                    print(f"  Successfully saved to {output_filename}")
                    print(f"  > Extracted {len(parsed_utterances)} utterances.")
                    print(f"  > First utterance example: {parsed_utterances[0]}")
                except Exception as e:
                    print(f"  Error writing JSON for {meeting_id}: {e}")
            else:
                print(f"  > No utterances extracted for {meeting_id}. No file created.")
        
        print(f"\nFinished preprocessing.")
    
    def split_transcript(self, utterances: List[Dict], split_ratio: float = 0.6) -> Tuple[List[Dict], List[Dict]]:
        """
        Split transcript into two parts for profile generation vs testing.
        
        Args:
            utterances: List of utterance dictionaries
            split_ratio: Ratio for first part (default 0.6 = 60% for profiles, 40% for testing)
            
        Returns:
            Tuple of (profile_segment, testing_segment)
        """
        split_index = int(len(utterances) * split_ratio)
        profile_segment = utterances[:split_index]
        testing_segment = utterances[split_index:]
        return profile_segment, testing_segment
    
    def format_utterances_for_prompt(self, utterance_list: List[Dict]) -> str:
        """
        Format utterances for LLM prompts.
        Format: (speaker_id): utterance_content (utterance_id)
        """
        return "\n".join([
            f"({utt['speaker_id']}): {utt['utterance_content']} ({utt['utterance_id']})" 
            for utt in utterance_list
        ])
    
    def create_utterance_index_map(self, utterances: List[Dict]) -> Dict[str, int]:
        """Create a lookup map from utterance_id to index for efficient processing."""
        return {utt['utterance_id']: i for i, utt in enumerate(utterances)}
    
    def load_parsed_transcript(self, filepath: str) -> List[Dict]:
        """Load a previously parsed transcript from JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: Transcript file not found at {filepath}")
            return []
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in {filepath}: {e}")
            return []
