#!/usr/bin/env python
"""
k_window_study.py — Utilities for CAPA k-window sensitivity experiments.

Supports:
- planning manual 5-meeting shards
- emitting shard-wise batch commands for the study
- validating canonical prep before the batch sweep
- comparing frozen anchor inventories across seed runs
- summarizing per-k aggregated results into a paper-ready table and figure
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from capa.anchor_inventory import jaccard_overlap, load_anchor_inventory_records


SUMMARY_METRICS = [
    "idea_strict_recall_rate",
    "idea_loose_recall_rate",
    "gt_idea_action_f1_micro",
    "idea_ignored_rate",
    "matched_unit_rate",
    "idea_latency_mean",
    "delegate_context_appropriateness_mean",
    "delegate_relevance_mean",
    "delegate_grounding_mean",
    "off_topic_rate_speaks",
    "hallucination_rate_speaks",
    "redundancy_rate_speaks",
    "preferred_cue_retained_rate",
    "self_initiation_fallback_rate",
    "anchor_drop_rate",
    "frozen_anchor_count",
    "kept_episode_count",
    "dropped_anchor_count",
    "mean_preferred_cue_gap",
    "mean_effective_cue_gap",
    "total_evaluated_steps",
    "pipeline_elapsed_minutes",
]


def discover_meeting_ids(input_dir: Path, meeting_filter: Optional[Sequence[str]] = None) -> List[str]:
    meeting_ids = sorted(
        path.stem[:-7] if path.stem.endswith("_parsed") else path.stem
        for path in input_dir.glob("*_parsed.json")
    )
    if meeting_filter:
        filter_set = {meeting_id.strip() for meeting_id in meeting_filter if meeting_id.strip()}
        meeting_ids = [meeting_id for meeting_id in meeting_ids if meeting_id in filter_set]
    return meeting_ids


def select_meeting_ids(
    input_dir: Path,
    meetings: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[str]:
    meeting_filter = [item.strip() for item in meetings.split(",")] if meetings else None
    meeting_ids = discover_meeting_ids(input_dir.resolve(), meeting_filter)
    if limit:
        meeting_ids = meeting_ids[:limit]
    return meeting_ids


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_condition_arg(raw: str) -> Tuple[int, Path]:
    if "=" not in raw:
        raise ValueError(f"Expected condition in the form K=/path/to/root, got: {raw}")
    raw_k, raw_path = raw.split("=", 1)
    return int(raw_k.strip()), Path(raw_path.strip()).resolve()


def mean_numeric(rows: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    values: List[float] = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            numeric = float(value)
        except Exception:
            continue
        if math.isfinite(numeric):
            values.append(numeric)
    if not values:
        return None
    return sum(values) / len(values)


def sum_numeric(rows: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    values: List[float] = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            numeric = float(value)
        except Exception:
            continue
        if math.isfinite(numeric):
            values.append(numeric)
    if not values:
        return None
    return sum(values)


def resolve_aggregated_dir(root: Path) -> Path:
    aggregated = root / "aggregated_evals"
    return aggregated if aggregated.exists() else root


def load_condition_payload(root: Path) -> Dict[str, Any]:
    aggregated_dir = resolve_aggregated_dir(root)
    csv_path = aggregated_dir / "aggregated_meeting_metrics.csv"
    summary_path = aggregated_dir / "corpus_summary.json"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing aggregated meeting metrics: {csv_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing corpus summary: {summary_path}")
    return {
        "aggregated_dir": aggregated_dir,
        "rows": read_csv_rows(csv_path),
        "summary": read_json(summary_path),
    }


def summarize_metric(summary: Dict[str, Any], metric: str) -> Dict[str, Optional[float]]:
    metric_stats = ((summary or {}).get("metrics") or {}).get(metric) or {}
    ci = metric_stats.get("mean_ci_95") or [None, None]
    return {
        "value": metric_stats.get("mean"),
        "ci_low": ci[0] if len(ci) > 0 else None,
        "ci_high": ci[1] if len(ci) > 1 else None,
        "n": metric_stats.get("n"),
    }


def format_stat(value: Optional[float], ci_low: Optional[float], ci_high: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "NA"
    if ci_low is None or ci_high is None:
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} [{ci_low:.{digits}f}, {ci_high:.{digits}f}]"


def discover_inventory_records(root: Path, meeting_ids: Optional[Sequence[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
    filter_set = {meeting_id.strip() for meeting_id in meeting_ids or [] if meeting_id.strip()}
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for inventory_path in sorted(root.rglob("anchor_inventory_for_*_from_*.json")):
        try:
            records = load_anchor_inventory_records(inventory_path)
        except Exception:
            continue
        for record in records:
            meeting_id = str(record.get("meeting_id") or "").strip()
            if not meeting_id:
                continue
            if filter_set and meeting_id not in filter_set:
                continue
            grouped.setdefault(meeting_id, []).append(record)
    return grouped


def dir_has_matching_files(root: Path, pattern: str) -> bool:
    if not root.exists() or not root.is_dir():
        return False
    return any(root.glob(pattern))


def dir_has_any_files(root: Path) -> bool:
    if not root.exists() or not root.is_dir():
        return False
    return any(path.is_file() for path in root.rglob("*"))


def command_plan_shards(args: argparse.Namespace) -> int:
    meeting_ids = select_meeting_ids(
        Path(args.input_dir),
        meetings=args.meetings,
        limit=args.limit,
    )
    shards = [
        meeting_ids[index : index + args.shard_size]
        for index in range(0, len(meeting_ids), args.shard_size)
    ]
    payload = {
        "input_dir": str(Path(args.input_dir).resolve()),
        "total_meetings": len(meeting_ids),
        "shard_size": args.shard_size,
        "shards": [
            {
                "index": idx + 1,
                "meeting_ids": shard,
                "meetings_arg": ",".join(shard),
            }
            for idx, shard in enumerate(shards)
        ],
    }
    if args.output:
        write_json(Path(args.output).resolve(), payload)
    for shard in payload["shards"]:
        print(f"shard_{shard['index']}={shard['meetings_arg']}")
    return 0


def command_emit_batch_commands(args: argparse.Namespace) -> int:
    meeting_ids = select_meeting_ids(
        Path(args.input_dir),
        meetings=args.meetings,
        limit=args.limit,
    )
    shards = [
        meeting_ids[index : index + args.shard_size]
        for index in range(0, len(meeting_ids), args.shard_size)
    ]

    output_lines: List[str] = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
    ]
    k_values = [int(item.strip()) for item in args.k_values.split(",") if item.strip()]
    for k_value in k_values:
        for index, shard in enumerate(shards, start=1):
            meetings_arg = ",".join(shard)
            cmd = [
                args.python_bin,
                "scripts/run_batch_meetings.py",
                "--input-dir", str(Path(args.input_dir).resolve()),
                "--output-dir", str((Path(args.output_root).resolve() / f"k{k_value}")),
                "--shared-prep-root", str((Path(args.output_root).resolve() / f"k{k_value}" / "_prep")),
                "--meetings", meetings_arg,
                "--participants", args.participants,
                "--delegate-variant", args.delegate_variant,
                "--model", args.model,
                "--capa-N", str(args.capa_N),
                "--capa-k", str(k_value),
                "--anchor-inventory-root", str(Path(args.canonical_prep_root).resolve()),
                "--static-prep-root", str(Path(args.canonical_prep_root).resolve()),
                "--eim-cache-root", str(Path(args.shared_eim_cache_root).resolve()),
                "--cap-meeting", str(args.cap_meeting),
                "--cap-per-participant", str(args.cap_per_participant),
                "--min-gap", str(args.min_gap),
                "--ctx-strict", str(args.ctx_strict),
                "--rel-strict", str(args.rel_strict),
                "--ctx-loose", str(args.ctx_loose),
                "--rel-loose", str(args.rel_loose),
                "--resume",
            ]
            output_lines.append(f"# k={k_value} shard={index}")
            output_lines.append(shlex.join(cmd))

    rendered = "\n".join(output_lines) + ("\n" if output_lines else "")
    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


def command_validate_canonical_prep(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir).resolve()
    canonical_prep_root = Path(args.canonical_prep_root).resolve()
    if not canonical_prep_root.exists():
        print(f"[FAIL] Canonical prep root does not exist: {canonical_prep_root}", file=sys.stderr)
        return 1

    meeting_ids = select_meeting_ids(input_dir, meetings=args.meetings, limit=args.limit)
    per_meeting: List[Dict[str, Any]] = []
    hard_failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    for meeting_id in meeting_ids:
        meeting_root = canonical_prep_root / meeting_id
        anchor_dir = meeting_root / "capa_anchor_inventory"
        profiles_dir = meeting_root / "user_profiles"
        jit_dir = meeting_root / "jit_briefings"

        missing_required: List[str] = []
        warning_issues: List[str] = []

        if not dir_has_matching_files(anchor_dir, "anchor_inventory_for_*_from_*.json"):
            missing_required.append("capa_anchor_inventory")
        if not dir_has_any_files(profiles_dir):
            warning_issues.append("user_profiles")
        if not dir_has_any_files(jit_dir):
            warning_issues.append("jit_briefings")

        status = "ok"
        if missing_required:
            status = "hard_failure"
        elif warning_issues:
            status = "warning"

        record = {
            "meeting_id": meeting_id,
            "meeting_root": str(meeting_root),
            "status": status,
            "missing_required": missing_required,
            "warnings": warning_issues,
        }
        per_meeting.append(record)
        if missing_required:
            hard_failures.append(record)
        elif warning_issues:
            warnings.append(record)

    payload = {
        "input_dir": str(input_dir),
        "canonical_prep_root": str(canonical_prep_root),
        "selected_meetings": meeting_ids,
        "meeting_count": len(meeting_ids),
        "ok_count": sum(1 for record in per_meeting if record["status"] == "ok"),
        "warning_count": len(warnings),
        "hard_failure_count": len(hard_failures),
        "hard_failures": hard_failures,
        "warnings": warnings,
        "per_meeting": per_meeting,
    }

    if args.output_json:
        write_json(Path(args.output_json).resolve(), payload)

    print(f"[INFO] Selected meetings: {len(meeting_ids)}")
    print(f"[INFO] Canonical prep root: {canonical_prep_root}")
    print(f"[INFO] OK meetings: {payload['ok_count']}")
    print(f"[INFO] Warning-only meetings: {payload['warning_count']}")
    print(f"[INFO] Hard-failure meetings: {payload['hard_failure_count']}")

    for record in warnings:
        issues = ", ".join(record["warnings"])
        print(f"[WARN] {record['meeting_id']}: missing reusable static prep -> {issues}")

    for record in hard_failures:
        issues = ", ".join(record["missing_required"])
        print(f"[FAIL] {record['meeting_id']}: missing required canonical prep -> {issues}", file=sys.stderr)

    if hard_failures:
        return 1
    return 0


def command_compare_inventories(args: argparse.Namespace) -> int:
    meeting_filter = [item.strip() for item in args.meetings.split(",")] if args.meetings else None
    left_records = discover_inventory_records(Path(args.left_root).resolve(), meeting_filter)
    right_records = discover_inventory_records(Path(args.right_root).resolve(), meeting_filter)
    meeting_ids = sorted(set(left_records) | set(right_records))

    per_meeting: List[Dict[str, Any]] = []
    combined_left: List[Dict[str, Any]] = []
    combined_right: List[Dict[str, Any]] = []
    for meeting_id in meeting_ids:
        left = left_records.get(meeting_id, [])
        right = right_records.get(meeting_id, [])
        overlap = jaccard_overlap(left, right)
        overlap["meeting_id"] = meeting_id
        per_meeting.append(overlap)
        combined_left.extend(left)
        combined_right.extend(right)

    overall = jaccard_overlap(combined_left, combined_right)
    payload = {
        "left_root": str(Path(args.left_root).resolve()),
        "right_root": str(Path(args.right_root).resolve()),
        "meeting_count": len(meeting_ids),
        "overall": overall,
        "per_meeting": per_meeting,
    }

    if args.output_json:
        write_json(Path(args.output_json).resolve(), payload)
    if args.output_csv:
        output_csv = Path(args.output_csv).resolve()
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["meeting_id", "left_count", "right_count", "intersection_count", "union_count", "jaccard"],
            )
            writer.writeheader()
            writer.writerows(per_meeting)

    print(json.dumps(payload["overall"], indent=2))
    if args.min_jaccard is not None and float(overall.get("jaccard") or 0.0) < float(args.min_jaccard):
        print(
            f"[FAIL] Overall Jaccard {float(overall.get('jaccard') or 0.0):.3f} < minimum {args.min_jaccard:.3f}",
            file=sys.stderr,
        )
        return 1
    return 0


def command_summarize(args: argparse.Namespace) -> int:
    conditions = sorted(parse_condition_arg(raw) for raw in args.condition)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    chart_rows: List[Dict[str, Any]] = []
    for k_value, root in conditions:
        payload = load_condition_payload(root)
        rows = payload["rows"]
        summary = payload["summary"]
        record: Dict[str, Any] = {
            "k": k_value,
            "source_root": str(root),
            "aggregated_dir": str(payload["aggregated_dir"]),
            "n_meetings": len(rows),
            "bootstrap_samples": summary.get("bootstrap_samples"),
        }
        for metric in SUMMARY_METRICS:
            stats = summarize_metric(summary, metric)
            record[metric] = stats["value"]
            record[f"{metric}_ci_low"] = stats["ci_low"]
            record[f"{metric}_ci_high"] = stats["ci_high"]
            if record[metric] is None:
                record[metric] = mean_numeric(rows, metric)
        total_runtime_minutes = sum_numeric(rows, "pipeline_elapsed_minutes")
        if total_runtime_minutes is not None:
            record["pipeline_elapsed_minutes_total"] = total_runtime_minutes
        chart_rows.append({
            "k": k_value,
            "idea_loose_recall_rate": record.get("idea_loose_recall_rate"),
            "gt_idea_action_f1_micro": record.get("gt_idea_action_f1_micro"),
            "runtime_overlay": record.get("pipeline_elapsed_minutes"),
            "steps_overlay": record.get("total_evaluated_steps"),
        })
        summary_rows.append(record)

    summary_csv = out_dir / "k_window_summary.csv"
    fieldnames = sorted({key for row in summary_rows for key in row.keys()}, key=lambda key: (key != "k", key))
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    markdown_lines = [
        "| k | Loose Recall | GT-Idea F1 | Ignored Rate | Matched Unit | Mean Runtime (min) | Fallback Rate | Drop Rate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summary_rows, key=lambda item: item["k"]):
        markdown_lines.append(
            "| {k} | {loose} | {f1} | {ignored} | {matched} | {runtime} | {fallback} | {drop} |".format(
                k=row["k"],
                loose=format_stat(
                    row.get("idea_loose_recall_rate"),
                    row.get("idea_loose_recall_rate_ci_low"),
                    row.get("idea_loose_recall_rate_ci_high"),
                ),
                f1=format_stat(
                    row.get("gt_idea_action_f1_micro"),
                    row.get("gt_idea_action_f1_micro_ci_low"),
                    row.get("gt_idea_action_f1_micro_ci_high"),
                ),
                ignored=format_stat(row.get("idea_ignored_rate"), None, None),
                matched=format_stat(row.get("matched_unit_rate"), None, None),
                runtime=format_stat(row.get("pipeline_elapsed_minutes"), None, None, digits=2),
                fallback=format_stat(row.get("self_initiation_fallback_rate"), None, None),
                drop=format_stat(row.get("anchor_drop_rate"), None, None),
            )
        )
    (out_dir / "k_window_table.md").write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    write_json(out_dir / "k_window_summary.json", {"conditions": summary_rows})

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        chart_rows = sorted(chart_rows, key=lambda row: row["k"])
        xs = [row["k"] for row in chart_rows]
        loose = [row.get("idea_loose_recall_rate") for row in chart_rows]
        f1 = [row.get("gt_idea_action_f1_micro") for row in chart_rows]
        runtime = [row.get("runtime_overlay") for row in chart_rows]
        overlay_label = "Runtime / meeting (min)"
        if not any(value is not None for value in runtime):
            runtime = [row.get("steps_overlay") for row in chart_rows]
            overlay_label = "Evaluated steps / meeting"

        fig, ax_left = plt.subplots(figsize=(7.2, 4.2))
        ax_left.plot(xs, loose, marker="o", linewidth=2.0, label="Loose Recall")
        ax_left.plot(xs, f1, marker="s", linewidth=2.0, label="GT-Idea F1")
        ax_left.set_xlabel("k")
        ax_left.set_ylabel("Performance")
        ax_left.set_ylim(bottom=0.0)
        ax_left.grid(alpha=0.25, linestyle=":")

        ax_right = ax_left.twinx()
        ax_right.plot(xs, runtime, marker="^", linewidth=1.8, linestyle="--", color="#444444", label=overlay_label)
        ax_right.set_ylabel(overlay_label)

        left_handles, left_labels = ax_left.get_legend_handles_labels()
        right_handles, right_labels = ax_right.get_legend_handles_labels()
        ax_left.legend(left_handles + right_handles, left_labels + right_labels, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / "k_window_tradeoff.png", dpi=220)
        plt.close(fig)
    except Exception as exc:
        print(f"[WARN] Could not render figure: {exc}", file=sys.stderr)

    print(f"[OK] Summary written to {summary_csv}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Utilities for CAPA k-window sensitivity experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan-shards", help="Emit comma-separated manual batch shards")
    plan.add_argument("--input-dir", required=True, help="Directory containing *_parsed.json files")
    plan.add_argument("--meetings", default=None, help="Optional comma-separated meeting filter")
    plan.add_argument("--limit", type=int, default=None, help="Optional cap on number of meetings")
    plan.add_argument("--shard-size", type=int, default=5, help="Meetings per shard")
    plan.add_argument("--output", default=None, help="Optional JSON output path")
    plan.set_defaults(func=command_plan_shards)

    emit = subparsers.add_parser("emit-batch-commands", help="Emit shard-wise batch commands for the k-study")
    emit.add_argument("--input-dir", required=True, help="Directory containing *_parsed.json files")
    emit.add_argument("--output-root", required=True, help="Base output root for per-k runs")
    emit.add_argument("--canonical-prep-root", required=True, help="Canonical seed prep root reused for frozen anchors and static prep")
    emit.add_argument("--shared-eim-cache-root", required=True, help="Study-level root for meeting-scoped shared EIM caches")
    emit.add_argument("--model", required=True, help="Model name to pass through to the batch runner")
    emit.add_argument("--meetings", default=None, help="Optional comma-separated meeting filter")
    emit.add_argument("--limit", type=int, default=None, help="Optional cap on number of meetings")
    emit.add_argument("--shard-size", type=int, default=5, help="Meetings per shard")
    emit.add_argument("--k-values", default="3,5,7,9", help="Comma-separated k values to emit")
    emit.add_argument("--participants", default="all", help="Participants argument for the batch runner")
    emit.add_argument("--delegate-variant", default="capa", choices=["capa"], help="Delegate variant for the study")
    emit.add_argument("--capa-N", type=int, default=20, help="CAPA context window N")
    emit.add_argument("--cap-meeting", type=int, default=20, help="Cap per meeting")
    emit.add_argument("--cap-per-participant", type=int, default=5, help="Cap per participant")
    emit.add_argument("--min-gap", type=int, default=0, help="Minimum gap")
    emit.add_argument("--ctx-strict", type=float, default=0.75, help="Strict context threshold")
    emit.add_argument("--rel-strict", type=float, default=0.75, help="Strict relevance threshold")
    emit.add_argument("--ctx-loose", type=float, default=0.60, help="Loose context threshold")
    emit.add_argument("--rel-loose", type=float, default=0.60, help="Loose relevance threshold")
    emit.add_argument("--python-bin", default=sys.executable, help="Python executable to use in emitted commands")
    emit.add_argument("--output", default=None, help="Optional shell script output path")
    emit.set_defaults(func=command_emit_batch_commands)

    validate = subparsers.add_parser("validate-canonical-prep", help="Validate canonical prep availability before running the k-study batch sweep")
    validate.add_argument("--input-dir", required=True, help="Directory containing *_parsed.json files")
    validate.add_argument("--canonical-prep-root", required=True, help="Canonical seed prep root reused for frozen anchors and static prep")
    validate.add_argument("--meetings", default=None, help="Optional comma-separated meeting filter")
    validate.add_argument("--limit", type=int, default=None, help="Optional cap on number of meetings")
    validate.add_argument("--output-json", default=None, help="Optional JSON report path")
    validate.set_defaults(func=command_validate_canonical_prep)

    compare = subparsers.add_parser("compare-inventories", help="Compute Jaccard overlap across frozen anchor inventories")
    compare.add_argument("--left-root", required=True, help="Prep root for the left inventory set")
    compare.add_argument("--right-root", required=True, help="Prep root for the right inventory set")
    compare.add_argument("--meetings", default=None, help="Optional comma-separated meeting filter")
    compare.add_argument("--min-jaccard", type=float, default=None, help="Fail if overall Jaccard is below this threshold")
    compare.add_argument("--output-json", default=None, help="Optional JSON output path")
    compare.add_argument("--output-csv", default=None, help="Optional per-meeting CSV output path")
    compare.set_defaults(func=command_compare_inventories)

    summarize = subparsers.add_parser("summarize", help="Build the k-window comparison table and figure")
    summarize.add_argument("--condition", action="append", required=True, help="Condition in the form K=/path/to/variant_or_aggregated_root")
    summarize.add_argument("--out-dir", required=True, help="Directory for the summary outputs")
    summarize.set_defaults(func=command_summarize)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
