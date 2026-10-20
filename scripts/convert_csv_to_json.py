#!/usr/bin/env python3
"""
CSV to JSON Converter for Meeting Transcripts

Converts CSV files containing meeting transcripts to the JSON format
required by the Meeting Delegate system.

Usage:
    python scripts/convert_csv_to_json.py --input meetings.csv --output parsed_transcripts/

CSV Requirements:
    - Must have a 'Meeting' column containing the raw meeting text
    - Optional 'id' column for meeting identifiers (otherwise uses row index)
    - Meeting text should have speaker labels (e.g., "Project Manager: Hello everyone...")
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
src_dir = project_root / 'src'
sys.path.insert(0, str(src_dir))

from transcript_processor import TranscriptProcessor


def main():
    parser = argparse.ArgumentParser(
        description="Convert CSV meeting transcripts to JSON format for Meeting Delegate system"
    )
    parser.add_argument(
        "--input", 
        required=True, 
        help="Path to input CSV file with 'Meeting' column"
    )
    parser.add_argument(
        "--output", 
        required=True, 
        help="Output directory for parsed JSON files"
    )
    parser.add_argument(
        "--indices", 
        nargs='+', 
        type=int, 
        help="Specific row indices to process (optional, processes all if not specified)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input).exists():
        print(f"❌ Error: Input file not found: {args.input}")
        sys.exit(1)
    
    if not args.input.lower().endswith('.csv'):
        print(f"❌ Error: Input file must be a CSV file: {args.input}")
        sys.exit(1)
    
    # Initialize processor
    processor = TranscriptProcessor()
    
    print(f"🔄 Converting CSV to JSON format...")
    print(f"   Input: {args.input}")
    print(f"   Output: {args.output}")
    if args.indices:
        print(f"   Processing rows: {args.indices}")
    else:
        print(f"   Processing: All rows")
    
    # Convert CSV to JSON
    try:
        processor.preprocess_qmsum_from_csv(
            csv_filepath=args.input,
            output_directory=args.output,
            meeting_indices=args.indices
        )
        print(f"✅ Conversion completed! Check {args.output} for parsed JSON files.")
        print(f"📋 Next step: Use data_pipeline.py to generate simulation data from these JSON files.")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
