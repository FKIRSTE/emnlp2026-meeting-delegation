#!/usr/bin/env python
"""
aggregate_corpus.py — Enhanced Corpus Aggregation with Robust Statistics

Aggregates meeting-level metrics across the corpus using scientifically sound practices
for both normal and non-normal distributions (e.g., Likert scales, latency).

Key Features:
- **Primary Reporting**: Mean ± SD with 95% Bootstrap CI (Standard for NLP/ACL).
- **Robustness Check**: Median + IQR with 95% Bootstrap CI (Standard for Skewed/Ordinal data).
- **Dual Output**: 
    - `aggregated_meeting_metrics.csv`: Raw meeting-level data.
    - `corpus_summary.json`: Detailed stats including CIs.
    - `paper_metrics.csv`: Clean, citation-ready table with selected strategies.

Usage:
    python scripts/aggregate_corpus.py \
        --root ami_results_test \
        --bootstrap 10000

Output:
    {root}/aggregated_evals/
        - aggregated_meeting_metrics.csv
        - corpus_summary.json
        - paper_metrics.csv
"""

import argparse
import csv
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Metrics that require Robust Aggregation (Median of Medians) typically due to skew
SKEWED_METRICS = {
    # Only Latency is treated as skewed; all rates/Likert scores use Mean/SD per plan
    "idea_latency_mean",
    "idea_latency_median",
    "idea_latency_max",
    "step_latency_mean",
    "latency_mean_signed",
    "latency_median_signed",
    "median_delegate_timing_delta_speaks_macro",
    "env_timing_delta_mean",
}

# -----------------------------------------------------------------------------
# Data Loading & Helpers
# -----------------------------------------------------------------------------

def read_json(path: Path) -> Dict[str, Any]:
    """Read JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_pipeline_status(meeting_dir: Path) -> Dict[str, Any]:
    status_path = meeting_dir / "pipeline_status.json"
    if not status_path.exists():
        return {}
    try:
        return read_json(status_path)
    except Exception:
        return {}

def is_number(v: Any) -> bool:
    """Check if value is numeric."""
    if v is None or v == "":
        return False
    try:
        float(v)
        return True
    except (ValueError, TypeError):
        return False

def collect_meeting_metrics(root: Path) -> List[Dict[str, Any]]:
    """Collect meeting_metrics.json from all completed meetings."""
    rows = []
    # Pattern: {root}/{meeting_id}/evals/{meeting_id}/meeting_metrics.json
    for metrics_file in root.rglob("evals/*/meeting_metrics.json"):
        try:
            data = read_json(metrics_file)
            data["_source_path"] = str(metrics_file)
            if "meeting_id" not in data:
                data["meeting_id"] = metrics_file.parent.name
            meeting_dir = metrics_file.parents[2]
            pipeline_status = load_pipeline_status(meeting_dir)
            if pipeline_status:
                elapsed_seconds = pipeline_status.get("elapsed_seconds")
                if is_number(elapsed_seconds):
                    elapsed_seconds = float(elapsed_seconds)
                    data["pipeline_elapsed_seconds"] = elapsed_seconds
                    data["pipeline_elapsed_minutes"] = elapsed_seconds / 60.0
                data.setdefault("capa_k", pipeline_status.get("capa_k"))
                data.setdefault("late_window_k", pipeline_status.get("late_window_k"))
                data.setdefault("anchor_inventory_dir", pipeline_status.get("anchor_inventory_dir"))
                data.setdefault("static_prep_source_dir", pipeline_status.get("static_prep_source_dir"))
                data.setdefault("eim_cache_dir", pipeline_status.get("eim_cache_dir"))
                data.setdefault("freeze_anchor_inventory", pipeline_status.get("freeze_anchor_inventory"))
            
            # -------------------------------------------------------------------------
            # Dynamic Derivation: Idea Status Rates (Outcome Distribution)
            # -------------------------------------------------------------------------
            # We derive these from counts to avoid re-running capa_eval on existing data.
            # Total = Strict + LooseOnly + BadAttempt + NeverAttempted
            c_strict = float(data.get("idea_status_hit_strict") or 0)
            c_loose = float(data.get("idea_status_hit_loose_only") or 0)
            c_bad = float(data.get("idea_status_attempted_but_bad") or 0)
            c_never = float(data.get("idea_status_never_attempted") or 0)
            
            total_ideas = c_strict + c_loose + c_bad + c_never
            
            if total_ideas > 0:
                data["idea_loose_only_rate"] = c_loose / total_ideas
                data["idea_bad_attempt_rate"] = c_bad / total_ideas
                data["idea_ignored_rate"] = c_never / total_ideas
                # Optional: Ensure strict rate is consistent if missing
                if "idea_strict_recall_rate" not in data:
                     data["idea_strict_recall_rate"] = c_strict / total_ideas
            else:
                data["idea_loose_only_rate"] = None
                data["idea_bad_attempt_rate"] = None
                data["idea_ignored_rate"] = None
            
            rows.append(data)
        except Exception as e:
            print(f"[WARN] Failed to read {metrics_file}: {e}")
    return rows

# -----------------------------------------------------------------------------
# Robust Statistics Engine
# -----------------------------------------------------------------------------

def bootstrap_ci(
    values: List[float], 
    estimator: Callable[[Any], float], 
    n_samples: int = 10000, 
    alpha: float = 0.05,
    seed: int = 37
) -> Dict[str, float]:
    """
    Compute Bootstrap Confidence Interval using vectorized NumPy.
    Standard Percentile Method.
    """
    # 0. Reproducibility
    rng = np.random.default_rng(seed)
    
    # 1. Finite Check
    clean_values = np.array([v for v in values if np.isfinite(v)], dtype=float)
    n = len(clean_values)
    
    if n < 2:
        return {"ci_low": None, "ci_high": None}
    
    # 2. Vectorized Bootstrap
    # Generate (n_samples, n) array of indices
    indices = rng.integers(0, n, size=(n_samples, n))
    
    # Create (n_samples, n) matrix of samples
    samples = clean_values[indices]
    
    # Apply estimator along axis 1 (rows)
    # Note: estimator must accept axis argument or we map
    if estimator == np.mean:
        stats = np.mean(samples, axis=1)
    elif estimator == np.median:
        stats = np.median(samples, axis=1)
    else:
        # Fallback for custom estimators (slower loop)
        stats = np.array([estimator(row) for row in samples])
    
    # 3. Percentiles
    lower_p = 100 * (alpha / 2)
    upper_p = 100 * (1 - alpha / 2)
    
    ci_low, ci_high = np.percentile(stats, [lower_p, upper_p])
    
    return {
        "ci_low": float(ci_low),
        "ci_high": float(ci_high)
    }

def compute_iqr(values: List[float]) -> float:
    """Compute Interquartile Range (Q3 - Q1). Filters NaNs."""
    clean = [v for v in values if np.isfinite(v)]
    if len(clean) < 2:
        return 0.0
    q75, q25 = np.percentile(clean, [75, 25])
    return float(q75 - q25)

def aggregate_metrics(
    rows: List[Dict[str, Any]],
    n_bootstrap: int = 10000,
    seed: int = 37
) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate all numeric metrics with robust stats.
    """
    # 1. Collect values
    metric_values: Dict[str, List[float]] = {}
    for row in rows:
        for key, value in row.items():
            if key.startswith("_"):
                continue
            if is_number(value):
                val_float = float(value)
                if np.isfinite(val_float):  # Filter NaNs at source
                    if key not in metric_values:
                        metric_values[key] = []
                    metric_values[key].append(val_float)
    
    # 2. Compute Statistics
    results = {}
    sorted_metrics = sorted(metric_values.keys()) # Sort to ensure deterministic iteration order if we ever use stateful RNG
    
    for metric in sorted_metrics:
        values = metric_values[metric]
        if not values:
            continue
            
        # Basic Descriptive
        n = len(values)
        val_mean = float(np.mean(values))
        val_std = float(np.std(values, ddof=1)) if n > 1 else 0.0
        val_median = float(np.median(values))
        val_iqr = compute_iqr(values)
        
        # Bootstrap CIs (Mean & Median)
        # Using vectorized bootstrap_ci with threaded seed
        ci_mean = bootstrap_ci(values, np.mean, n_samples=n_bootstrap, seed=seed) if n_bootstrap > 0 else {"ci_low": None, "ci_high": None}
        ci_median = bootstrap_ci(values, np.median, n_samples=n_bootstrap, seed=seed) if n_bootstrap > 0 else {"ci_low": None, "ci_high": None}
        
        results[metric] = {
            "mean": val_mean,
            "std": val_std,
            "mean_ci_95": [ci_mean["ci_low"], ci_mean["ci_high"]],
            
            "median": val_median,
            "iqr": val_iqr,
            "median_ci_95": [ci_median["ci_low"], ci_median["ci_high"]],
            
            "n": n
        }
    return results

# -----------------------------------------------------------------------------
# Formatting & Reports
# -----------------------------------------------------------------------------

def format_primary(stats: Dict[str, Any]) -> str:
    """Format: Mean ± SD [CI] (Standard Academic)."""
    m = stats["mean"]
    sd = stats["std"]
    ci = stats["mean_ci_95"]
    if ci[0] is not None:
        return f"{m:.3f} ± {sd:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
    return f"{m:.3f} ± {sd:.3f}"

def format_robust(stats: Dict[str, Any]) -> str:
    """Format: Median (IQR) [CI] (Robust/Skewed)."""
    m = stats["median"]
    iqr = stats["iqr"]
    ci = stats["median_ci_95"]
    if ci[0] is not None:
        return f"{m:.3f} (IQR={iqr:.3f}) [{ci[0]:.3f}, {ci[1]:.3f}]"
    return f"{m:.3f} (IQR={iqr:.3f})"

def write_paper_report(aggregated: Dict[str, Any], out_dir: Path):
    """
    Generate a citation-ready CSV report that explicitly lists the chosen strategy.
    
    Columns:
    - Metric
    - Aggregation_Strategy (Mean or Median)
    - Value (The aggregate value)
    - CI_95_Low
    - CI_95_High
    - Spread_Measure (SD or IQR)
    - Spread_Value
    - N (Sample size)
    """
    report_rows = []
    
    # Sort metrics for readability
    for metric in sorted(aggregated.keys()):
        stats = aggregated[metric]
        
        # Determine Strategy
        if metric in SKEWED_METRICS:
            strategy = "Median (Robust)"
            val = stats["median"]
            ci = stats["median_ci_95"]
            spread_type = "IQR"
            spread_val = stats["iqr"]
        else:
            strategy = "Mean (Standard)"
            val = stats["mean"]
            ci = stats["mean_ci_95"]
            spread_type = "SD"
            spread_val = stats["std"]
            
        report_rows.append({
            "Metric": metric,
            "Strategy": strategy,
            "Value": f"{val:.4f}",
            "CI_95_Low": f"{ci[0]:.4f}" if ci[0] is not None else "",
            "CI_95_High": f"{ci[1]:.4f}" if ci[1] is not None else "",
            "Spread_Measure": spread_type,
            "Spread_Value": f"{spread_val:.4f}",
            "N": stats["n"]
        })
        
    out_path = out_dir / "paper_metrics.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["Metric", "Strategy", "Value", "CI_95_Low", "CI_95_High", "Spread_Measure", "Spread_Value", "N"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(report_rows)
    print(f"[OK] Written: {out_path}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate CAPA eval metrics using robust statistics (Mean/SD + Median/IQR)."
    )
    parser.add_argument("--root", required=True, help="Root directory containing per-meeting folders")
    parser.add_argument("--out", default=None, help="Output directory (default: <root>/aggregated_evals)")
    parser.add_argument("--bootstrap", type=int, default=10000, help="Number of bootstrap samples (default: 10000)")
    parser.add_argument("--seed", type=int, default=37, help="Random seed for bootstrap reproducibility (default: 37)")
    
    args = parser.parse_args()
    
    root = Path(args.root).resolve()
    out_dir = Path(args.out).resolve() if args.out else root / "aggregated_evals"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Scanning: {root}")
    meeting_rows = collect_meeting_metrics(root)
    
    if not meeting_rows:
        print(f"[ERROR] No meeting metrics found in {root}")
        return
    
    print(f"[INFO] Found {len(meeting_rows)} completed meetings")
    
    # 1. Write Raw Aggregate CSV
    all_fields = list({k for r in meeting_rows for k in r.keys()})
    priority_fields = ["meeting_id"]
    other_fields = sorted([f for f in all_fields if f not in priority_fields and not f.startswith("_")])
    fields = priority_fields + other_fields
    
    csv_path = out_dir / "aggregated_meeting_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(meeting_rows)
    print(f"[OK] Written: {csv_path}")
    
    # 2. Compute Robust Stats
    print(f"[INFO] Computing statistics (B={args.bootstrap} samples, seed={args.seed})...")
    aggregated = aggregate_metrics(meeting_rows, n_bootstrap=args.bootstrap, seed=args.seed)
    
    # 3. Write Summary JSON
    summary = {
        "root": str(root),
        "n_meetings": len(meeting_rows),
        "bootstrap_samples": args.bootstrap,
        "seed": args.seed,
        "metrics": aggregated
    }
    summary_path = out_dir / "corpus_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[OK] Written: {summary_path}")
    
    # 4. Write Paper Report (Citation Ready)
    write_paper_report(aggregated, out_dir)
    
    # 5. Console Report
    primary_metrics = [
        "gt_idea_action_f1_micro",
        "idea_strict_recall_rate",
        "idea_loose_recall_rate",
        "idea_loose_only_rate",
        "idea_bad_attempt_rate",
        "idea_ignored_rate",
        "idea_match_synchronized_rate",
        "idea_match_proactive_rate",
        "delegate_decision_appropriateness_mean",
        "delegate_context_appropriateness_mean",
        "delegate_relevance_mean",
        "delegate_grounding_mean",
        "hallucination_rate_speaks",
        "off_topic_rate_speaks",
        "redundancy_rate_speaks",
    ]
    # Filter skewed metrics to those that exist in aggregated
    found_skewed = [m for m in SKEWED_METRICS if m in aggregated]

    print(f"\n{'=' * 80}")
    print(f"CORPUS SUMMARY ({len(meeting_rows)} meetings)")
    print(f"{'=' * 80}")
    print(f"{'METRIC':<45} | {'MEAN [95% CI]':<30}")
    print(f"{'-' * 80}")
    
    for metric in primary_metrics:
        if metric in aggregated:
            print(f"{metric:<45} | {format_primary(aggregated[metric])}")
            
    if found_skewed:
        print(f"\n{'-' * 80}")
        print(f"{'ROBUST METRICS (Skewed)':<45} | {'MEDIAN (IQR) [95% CI]':<30}")
        print(f"{'-' * 80}")
        for metric in sorted(found_skewed):
             print(f"{metric:<45} | {format_robust(aggregated[metric])}")

    print(f"{'=' * 80}\n")

if __name__ == "__main__":
    main()
