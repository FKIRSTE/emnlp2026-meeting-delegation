#!/usr/bin/env python
"""
AMI Corpus Statistics Script

Analyzes the size of meetings (number of utterances) across train, val, and test sets.
Provides mean, median, min, max, and percentile statistics.

Usage:
    python scripts/analyze_ami_stats.py
"""

import json
import os
from pathlib import Path
from typing import Dict, List
import statistics


def load_meeting_stats(parsed_dir: Path) -> Dict[str, int]:
    """Load all meetings from a parsed directory and count utterances."""
    stats = {}
    
    if not parsed_dir.exists():
        print(f"  [WARN] Directory not found: {parsed_dir}")
        return stats
    
    for json_file in sorted(parsed_dir.glob("*_parsed.json")):
        meeting_id = json_file.stem.replace("_parsed", "")
        
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Count utterances (data is typically a list of utterances)
            if isinstance(data, list):
                utterance_count = len(data)
            elif isinstance(data, dict) and "utterances" in data:
                utterance_count = len(data["utterances"])
            else:
                utterance_count = 0
            
            stats[meeting_id] = utterance_count
        except Exception as e:
            print(f"  [ERROR] Failed to load {json_file.name}: {e}")
    
    return stats


def compute_statistics(counts: List[int]) -> Dict[str, float]:
    """Compute comprehensive statistics for a list of counts."""
    if not counts:
        return {"count": 0}
    
    sorted_counts = sorted(counts)
    n = len(counts)
    
    return {
        "count": n,
        "total": sum(counts),
        "mean": statistics.mean(counts),
        "median": statistics.median(counts),
        "stdev": statistics.stdev(counts) if n > 1 else 0,
        "min": min(counts),
        "max": max(counts),
        "p25": sorted_counts[n // 4] if n >= 4 else sorted_counts[0],
        "p75": sorted_counts[3 * n // 4] if n >= 4 else sorted_counts[-1],
        "p90": sorted_counts[int(n * 0.9)] if n >= 10 else sorted_counts[-1],
    }


def print_statistics(name: str, stats: Dict[str, int], detailed_stats: Dict[str, float]):
    """Print formatted statistics."""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    
    if detailed_stats["count"] == 0:
        print("  No meetings found.")
        return
    
    print(f"  Meetings: {detailed_stats['count']}")
    print(f"  Total Utterances: {detailed_stats['total']:,}")
    print()
    print(f"  Mean:   {detailed_stats['mean']:,.1f} utterances/meeting")
    print(f"  Median: {detailed_stats['median']:,.1f} utterances/meeting")
    print(f"  StdDev: {detailed_stats['stdev']:,.1f}")
    print()
    print(f"  Min: {detailed_stats['min']:,}  |  Max: {detailed_stats['max']:,}")
    print(f"  P25: {detailed_stats['p25']:,}  |  P75: {detailed_stats['p75']:,}  |  P90: {detailed_stats['p90']:,}")
    
    # Print individual meetings
    print()
    print("  Individual Meetings:")
    for meeting_id, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        print(f"    {meeting_id}: {count:,} utterances")


def main():
    print("="*60)
    print("AMI CORPUS UTTERANCE STATISTICS")
    print("="*60)
    
    base_dir = Path("data") / "ami"
    
    # Define splits
    splits = {
        "Train": base_dir / "train",
        "Validation": base_dir / "val",
        "Test": base_dir / "test",
    }
    
    all_stats = {}
    all_detailed = {}
    
    for split_name, split_dir in splits.items():
        print(f"\nLoading {split_name}...")
        stats = load_meeting_stats(split_dir)
        all_stats[split_name] = stats
        
        counts = list(stats.values())
        all_detailed[split_name] = compute_statistics(counts)
    
    # Print statistics for each split
    for split_name in splits.keys():
        print_statistics(split_name, all_stats[split_name], all_detailed[split_name])
    
    # Summary comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"\n{'Split':<12} {'Count':>6} {'Mean':>10} {'Median':>10} {'Min':>8} {'Max':>8}")
    print("-" * 60)
    
    for split_name in splits.keys():
        d = all_detailed[split_name]
        if d["count"] > 0:
            print(f"{split_name:<12} {d['count']:>6} {d['mean']:>10.1f} {d['median']:>10.1f} {d['min']:>8} {d['max']:>8}")
    
    # Overall corpus statistics (detailed)
    all_counts = []
    all_meetings = {}
    for split_name, stats in all_stats.items():
        all_counts.extend(stats.values())
        # Merge all meetings with split prefix for identification
        for meeting_id, count in stats.items():
            all_meetings[f"{split_name[0]}-{meeting_id}"] = count
    
    combined = compute_statistics(all_counts)
    print("-" * 60)
    print(f"{'TOTAL':<12} {combined['count']:>6} {combined['mean']:>10.1f} {combined['median']:>10.1f} {combined['min']:>8} {combined['max']:>8}")
    
    # Detailed overall corpus report
    print_statistics("OVERALL CORPUS (Train + Val + Test)", all_meetings, combined)


if __name__ == "__main__":
    main()
