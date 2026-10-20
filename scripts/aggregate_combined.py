#!/usr/bin/env python
"""
aggregate_combined.py — Combine Meeting Metrics from Multiple Subsets

Combines already-aggregated meeting-level metrics from multiple experiment roots
(e.g., ami_results_train, ami_results_val, ami_results_test) and produces
a single paper_metrics.csv with proper corpus-level statistics.

Statistically Sound Approach:
- Each meeting is the unit of observation (Macro-averaging).
- We concatenate ALL meeting-level rows from all subsets.
- We then compute corpus-level Mean/SD/Bootstrap-CI (or Median/IQR for skewed metrics)
  on the combined N meetings.

Usage:
    python scripts/aggregate_combined.py \
        --roots ami_results_train ami_results_val ami_results_test \
        --out combined_corpus_results \
        --bootstrap 10000 \
        --seed 37
"""

import argparse
import csv
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional

# -----------------------------------------------------------------------------
# Configuration (Mirrored from aggregate_corpus.py)
# -----------------------------------------------------------------------------

SKEWED_METRICS = {
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

def is_number(v: Any) -> bool:
    """Check if value is numeric."""
    if v is None or v == "":
        return False
    try:
        float(v)
        return True
    except (ValueError, TypeError):
        return False

def load_meeting_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Load meeting-level metrics from an aggregated CSV."""
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric strings to floats
            parsed = {}
            for k, v in row.items():
                if is_number(v):
                    parsed[k] = float(v)
                else:
                    parsed[k] = v
            rows.append(parsed)
    return rows

def collect_all_meetings(roots: List[Path]) -> List[Dict[str, Any]]:
    """Collect meeting rows from all aggregated CSVs in the given roots."""
    all_rows = []
    for root in roots:
        csv_path = root / "aggregated_evals" / "aggregated_meeting_metrics.csv"
        if not csv_path.exists():
            print(f"[WARN] No aggregated CSV found at {csv_path}, skipping.")
            continue
        rows = load_meeting_csv(csv_path)
        print(f"[INFO] Loaded {len(rows)} meetings from {csv_path}")
        
        # Tag source for debugging/auditing
        for r in rows:
            r["_source"] = str(root.name)
        all_rows.extend(rows)
    return all_rows

# -----------------------------------------------------------------------------
# Derive Status Rates (Same logic as aggregate_corpus.py)
# -----------------------------------------------------------------------------

def derive_status_rates(row: Dict[str, Any]) -> None:
    """Derive idea status rates from counts if not already present."""
    c_strict = float(row.get("idea_status_hit_strict") or 0)
    c_loose = float(row.get("idea_status_hit_loose_only") or 0)
    c_bad = float(row.get("idea_status_attempted_but_bad") or 0)
    c_never = float(row.get("idea_status_never_attempted") or 0)
    
    total = c_strict + c_loose + c_bad + c_never
    
    if total > 0:
        if "idea_loose_only_rate" not in row or row.get("idea_loose_only_rate") is None:
            row["idea_loose_only_rate"] = c_loose / total
        if "idea_bad_attempt_rate" not in row or row.get("idea_bad_attempt_rate") is None:
            row["idea_bad_attempt_rate"] = c_bad / total
        if "idea_ignored_rate" not in row or row.get("idea_ignored_rate") is None:
            row["idea_ignored_rate"] = c_never / total

# -----------------------------------------------------------------------------
# Robust Statistics Engine (Copied from aggregate_corpus.py)
# -----------------------------------------------------------------------------

def bootstrap_ci(
    values: List[float],
    estimator: Callable[[Any], float],
    n_samples: int = 10000,
    alpha: float = 0.05,
    seed: int = 37

    
) -> Dict[str, Optional[float]]:
    """Compute Bootstrap CI using vectorized NumPy."""
    rng = np.random.default_rng(seed)
    clean_values = np.array([v for v in values if np.isfinite(v)], dtype=float)
    n = len(clean_values)
    
    if n < 2:
        return {"ci_low": None, "ci_high": None}
    
    indices = rng.integers(0, n, size=(n_samples, n))
    samples = clean_values[indices]
    
    if estimator == np.mean:
        stats = np.mean(samples, axis=1)
    elif estimator == np.median:
        stats = np.median(samples, axis=1)
    else:
        stats = np.array([estimator(row) for row in samples])
    
    lower_p = 100 * (alpha / 2)
    upper_p = 100 * (1 - alpha / 2)
    ci_low, ci_high = np.percentile(stats, [lower_p, upper_p])
    
    return {"ci_low": float(ci_low), "ci_high": float(ci_high)}

def compute_iqr(values: List[float]) -> float:
    """Compute IQR, filtering NaNs."""
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
    """Aggregate meeting-level metrics to corpus-level statistics."""
    # Derive status rates for each row if missing
    for row in rows:
        derive_status_rates(row)
    
    # Collect values per metric
    metric_values: Dict[str, List[float]] = {}
    for row in rows:
        for key, value in row.items():
            if key.startswith("_"):
                continue
            if is_number(value):
                val_float = float(value)
                if np.isfinite(val_float):
                    if key not in metric_values:
                        metric_values[key] = []
                    metric_values[key].append(val_float)
    
    # Compute statistics
    results = {}
    for metric in sorted(metric_values.keys()):
        values = metric_values[metric]
        if not values:
            continue
        
        n = len(values)
        val_mean = float(np.mean(values))
        val_std = float(np.std(values, ddof=1)) if n > 1 else 0.0
        val_median = float(np.median(values))
        val_iqr = compute_iqr(values)
        
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
# Reporting (Copied from aggregate_corpus.py)
# -----------------------------------------------------------------------------

def write_paper_report(aggregated: Dict[str, Any], out_dir: Path) -> None:
    """Generate citation-ready CSV with strategy labels."""
    report_rows = []
    
    for metric in sorted(aggregated.keys()):
        stats = aggregated[metric]
        
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

def format_primary(stats: Dict[str, Any]) -> str:
    m = stats["mean"]
    sd = stats["std"]
    ci = stats["mean_ci_95"]
    if ci[0] is not None:
        return f"{m:.3f} ± {sd:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
    return f"{m:.3f} ± {sd:.3f}"

def format_robust(stats: Dict[str, Any]) -> str:
    m = stats["median"]
    iqr = stats["iqr"]
    ci = stats["median_ci_95"]
    if ci[0] is not None:
        return f"{m:.3f} (IQR={iqr:.3f}) [{ci[0]:.3f}, {ci[1]:.3f}]"
    return f"{m:.3f} (IQR={iqr:.3f})"

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Combine meeting metrics from multiple subsets and produce corpus-level paper_metrics.csv"
    )
    parser.add_argument("--roots", nargs="+", required=True, help="Root directories containing aggregated_evals subfolders")
    parser.add_argument("--out", required=True, help="Output directory for combined results")
    parser.add_argument("--bootstrap", type=int, default=10000, help="Number of bootstrap samples (default: 10000)")
    parser.add_argument("--seed", type=int, default=37, help="Random seed for reproducibility (default: 37)")
    
    args = parser.parse_args()
    
    roots = [Path(r).resolve() for r in args.roots]
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Collect all meetings
    print(f"[INFO] Collecting meetings from {len(roots)} roots...")
    all_meetings = collect_all_meetings(roots)
    
    if not all_meetings:
        print("[ERROR] No meeting data found.")
        return
    
    print(f"[INFO] Total meetings: {len(all_meetings)}")
    
    # 2. Write combined raw CSV
    all_fields = list({k for r in all_meetings for k in r.keys()})
    priority_fields = ["meeting_id", "_source"]
    other_fields = sorted([f for f in all_fields if f not in priority_fields and not f.startswith("_")])
    fields = priority_fields + other_fields
    
    csv_path = out_dir / "combined_meeting_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_meetings)
    print(f"[OK] Written: {csv_path}")
    
    # 3. Compute aggregated stats
    print(f"[INFO] Computing statistics (B={args.bootstrap}, seed={args.seed})...")
    aggregated = aggregate_metrics(all_meetings, n_bootstrap=args.bootstrap, seed=args.seed)
    
    # 4. Write JSON summary
    summary = {
        "roots": [str(r) for r in roots],
        "n_meetings": len(all_meetings),
        "bootstrap_samples": args.bootstrap,
        "seed": args.seed,
        "metrics": aggregated
    }
    summary_path = out_dir / "corpus_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[OK] Written: {summary_path}")
    
    # 5. Write paper report
    write_paper_report(aggregated, out_dir)
    
    # 6. Console report
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
    found_skewed = [m for m in SKEWED_METRICS if m in aggregated]
    
    print(f"\n{'=' * 80}")
    print(f"COMBINED CORPUS SUMMARY ({len(all_meetings)} meetings from {len(roots)} subsets)")
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
