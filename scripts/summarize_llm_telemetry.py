#!/usr/bin/env python3
"""Aggregate MeetingDelegate LLM telemetry into module and decision summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


def percentile(values: Iterable[float], percentile_value: float) -> Optional[float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    rank = max(0, min(len(ordered) - 1, math.ceil(percentile_value * len(ordered)) - 1))
    return ordered[rank]


def iter_paths(inputs: List[str]) -> Iterable[Path]:
    seen = set()
    for raw in inputs:
        path = Path(raw)
        candidates = path.rglob("*.jsonl") if path.is_dir() else [path]
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen or not candidate.exists():
                continue
            seen.add(resolved)
            yield candidate


def load_events(inputs: List[str]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for path in iter_paths(inputs):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
                event["_source_file"] = str(path)
                events.append(event)
    return events


def cost_usd(
    event: Dict[str, Any],
    *,
    input_rate: float,
    cached_input_rate: float,
    output_rate: float,
) -> float:
    input_tokens = int(event.get("input_tokens") or 0)
    cached_tokens = min(input_tokens, int(event.get("cached_input_tokens") or 0))
    uncached_tokens = input_tokens - cached_tokens
    output_tokens = int(event.get("output_tokens") or 0)
    return (
        uncached_tokens * input_rate
        + cached_tokens * cached_input_rate
        + output_tokens * output_rate
    ) / 1_000_000.0


def aggregate(
    events: List[Dict[str, Any]],
    *,
    input_rate: float,
    cached_input_rate: float,
    output_rate: float,
) -> Dict[str, Any]:
    attempts = [event for event in events if event.get("event_type") == "llm_call_attempt"]
    spans = [event for event in events if event.get("event_type") == "span"]

    calls: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for event in attempts:
        calls[str(event.get("call_id") or f"missing::{id(event)}")].append(event)
    final_calls = [max(call_events, key=lambda event: int(event.get("attempt") or 0)) for call_events in calls.values()]

    decision_ids = sorted({str(event["decision_id"]) for event in events if event.get("decision_id")})
    decision_count = len(decision_ids)
    modules: Dict[str, Dict[str, Any]] = {}

    module_attempts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    module_calls: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for event in attempts:
        module_attempts[str(event.get("module") or "unlabeled")].append(event)
    for event in final_calls:
        module_calls[str(event.get("module") or "unlabeled")].append(event)

    for module in sorted(module_attempts):
        attempt_rows = module_attempts[module]
        call_rows = module_calls.get(module, [])
        successful = [row for row in call_rows if row.get("status") in {"success", "empty_response"}]
        billable = successful
        logical_latencies = [float(row.get("logical_elapsed_seconds") or 0.0) for row in call_rows]
        provider_latencies = [float(row.get("provider_elapsed_seconds") or 0.0) for row in successful]
        total_input = sum(int(row.get("input_tokens") or 0) for row in billable)
        total_output = sum(int(row.get("output_tokens") or 0) for row in billable)
        total_cached = sum(int(row.get("cached_input_tokens") or 0) for row in billable)
        total_retry_sleep = sum(float(row.get("retry_sleep_seconds") or 0.0) for row in attempt_rows)
        total_post_attempt_delay = sum(float(row.get("post_attempt_delay_seconds") or 0.0) for row in attempt_rows)
        total_cost = sum(
            cost_usd(
                row,
                input_rate=input_rate,
                cached_input_rate=cached_input_rate,
                output_rate=output_rate,
            )
            for row in billable
        )
        module_decisions = {str(row["decision_id"]) for row in call_rows if row.get("decision_id")}
        denominator = decision_count or len(module_decisions) or len(successful) or 1
        modules[module] = {
            "phase": next((row.get("phase") for row in call_rows if row.get("phase")), None),
            "logical_calls": len(call_rows),
            "attempts": len(attempt_rows),
            "successful_calls": len(successful),
            "failed_calls": sum(1 for row in call_rows if row.get("status") == "error"),
            "retries": max(0, len(attempt_rows) - len(call_rows)),
            "decisions_observed": len(module_decisions),
            "calls_per_decision": len(call_rows) / denominator,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "cached_input_tokens": total_cached,
            "input_tokens_per_decision": total_input / denominator,
            "output_tokens_per_decision": total_output / denominator,
            "mean_provider_latency_seconds": mean(provider_latencies) if provider_latencies else None,
            "p95_provider_latency_seconds": percentile(provider_latencies, 0.95),
            "mean_logical_latency_seconds": mean(logical_latencies) if logical_latencies else None,
            "p95_logical_latency_seconds": percentile(logical_latencies, 0.95),
            "retry_sleep_seconds": total_retry_sleep,
            "post_attempt_delay_seconds": total_post_attempt_delay,
            "post_attempt_delay_seconds_per_call": total_post_attempt_delay / (len(call_rows) or 1),
            "cost_usd": total_cost,
            "cost_usd_per_decision": total_cost / denominator,
            "provider_usage_calls": sum(1 for row in billable if str(row.get("usage_source", "")).startswith("provider")),
            "estimated_usage_calls": sum(1 for row in billable if not str(row.get("usage_source", "")).startswith("provider")),
        }

    span_summary: Dict[str, Dict[str, Any]] = {}
    spans_by_name: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for event in spans:
        spans_by_name[str(event.get("span_name") or "unknown")].append(event)
    for name, rows in sorted(spans_by_name.items()):
        values = [float(row.get("elapsed_seconds") or 0.0) for row in rows]
        span_summary[name] = {
            "count": len(values),
            "mean_seconds": mean(values) if values else None,
            "p50_seconds": percentile(values, 0.50),
            "p95_seconds": percentile(values, 0.95),
            "max_seconds": max(values) if values else None,
        }

    successful_final_calls = [row for row in final_calls if row.get("status") in {"success", "empty_response"}]
    total_input = sum(int(row.get("input_tokens") or 0) for row in successful_final_calls)
    total_output = sum(int(row.get("output_tokens") or 0) for row in successful_final_calls)
    total_cost = sum(
        cost_usd(
            row,
            input_rate=input_rate,
            cached_input_rate=cached_input_rate,
            output_rate=output_rate,
        )
        for row in successful_final_calls
    )
    denominator = decision_count or 1
    return {
        "pricing": {
            "input_usd_per_million": input_rate,
            "cached_input_usd_per_million": cached_input_rate,
            "output_usd_per_million": output_rate,
        },
        "source_files": sorted({event.get("_source_file") for event in events if event.get("_source_file")}),
        "events": len(events),
        "logical_calls": len(final_calls),
        "attempts": len(attempts),
        "decisions": decision_count,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "input_tokens_per_decision": total_input / denominator,
        "output_tokens_per_decision": total_output / denominator,
        "cost_usd": total_cost,
        "cost_usd_per_decision": total_cost / denominator,
        "modules": modules,
        "spans": span_summary,
    }


def write_csv(path: Path, modules: Dict[str, Dict[str, Any]]) -> None:
    rows = [{"module": module, **summary} for module, summary in modules.items()]
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["module"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: Dict[str, Any]) -> None:
    print(
        f"decisions={summary['decisions']} calls={summary['logical_calls']} attempts={summary['attempts']} "
        f"input_tokens={summary['input_tokens']} output_tokens={summary['output_tokens']} "
        f"cost_usd=${summary['cost_usd']:.4f} cost_per_decision=${summary['cost_usd_per_decision']:.4f}"
    )
    print()
    print(
        f"{'module':40} {'calls/dec':>10} {'in/dec':>10} {'out/dec':>10} "
        f"{'mean_s':>9} {'p95_s':>9} {'$/dec':>9}"
    )
    for module, row in summary["modules"].items():
        mean_latency = row.get("mean_logical_latency_seconds")
        p95_latency = row.get("p95_logical_latency_seconds")
        print(
            f"{module[:40]:40} {row['calls_per_decision']:10.3f} "
            f"{row['input_tokens_per_decision']:10.1f} {row['output_tokens_per_decision']:10.1f} "
            f"{(mean_latency or 0):9.3f} {(p95_latency or 0):9.3f} "
            f"{row['cost_usd_per_decision']:9.4f}"
        )
    if summary["spans"]:
        print("\nspans:")
        for name, row in summary["spans"].items():
            print(
                f"  {name}: n={row['count']} mean={row['mean_seconds']:.3f}s "
                f"p95={row['p95_seconds']:.3f}s max={row['max_seconds']:.3f}s"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Telemetry JSONL file(s) or directories")
    parser.add_argument("--input-usd-per-million", type=float, default=2.50)
    parser.add_argument("--cached-input-usd-per-million", type=float, default=1.25)
    parser.add_argument("--output-usd-per-million", type=float, default=10.00)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()

    events = load_events(args.inputs)
    summary = aggregate(
        events,
        input_rate=args.input_usd_per_million,
        cached_input_rate=args.cached_input_usd_per_million,
        output_rate=args.output_usd_per_million,
    )
    print_summary(summary)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.output_csv:
        write_csv(args.output_csv, summary["modules"])


if __name__ == "__main__":
    main()
