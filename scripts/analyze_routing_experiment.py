#!/usr/bin/env python3
"""Analyze helpdesk routing experiment results with rich visualizations."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Ensure we use the local cat-experiments source (path dependency)
CAT_SRC = PROJECT_ROOT.parent / "cat-experiments" / "src"
if CAT_SRC.exists() and str(CAT_SRC) not in sys.path:
    sys.path.insert(0, str(CAT_SRC))

from rich import box
from rich.console import Console
from rich.table import Table

# Keep department ordering consistent with the notebook
DEPARTMENTS = ["IT", "HR", "Facilities", "Finance", "Legal", "Security", "None"]

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a helpdesk routing experiment run via cat-experiments and render metrics with rich. "
            "Supports fetching results from CAT Cafe or local storage."
        )
    )
    parser.add_argument(
        "--target",
        choices=["local", "cat-cafe", "phoenix"],
        default="cat-cafe",
        help="Where the experiment is stored (default: cat-cafe).",
    )
    parser.add_argument(
        "--experiment-id",
        help=(
            "Experiment ID or name to analyze. Defaults to the most recent experiment."
        ),
    )
    parser.add_argument(
        "--storage-dir",
        default="eval_results",
        help="Directory containing local experiment outputs (default: eval_results).",
    )
    parser.add_argument(
        "--cat-base-url",
        default="http://localhost:8000",
        help="CAT Cafe base URL (for --target cat-cafe).",
    )
    parser.add_argument(
        "--phoenix-base-url",
        help="Phoenix base URL (for --target phoenix).",
    )
    parser.add_argument(
        "--phoenix-project",
        help="Phoenix project name (optional, for --target phoenix).",
    )
    parser.add_argument(
        "--show-normalized",
        action="store_true",
        help="Additionally render a row-normalized confusion matrix (percentages).",
    )
    return parser.parse_args()


def _latest_local_experiment(storage_dir: Path) -> str:
    """Return the most recent local experiment directory.

    This script is designed to analyze the output produced by:
        cat-experiments run ... --backend local --output-dir eval_results

    That layout creates directories like:
        eval_results/<experiment_id>/runs.jsonl
    """

    def looks_like_experiment_dir(path: Path) -> bool:
        return (
            path.is_dir()
            and (path / "runs.jsonl").exists()
            and (path / "summary.json").exists()
        )

    experiments = sorted(
        (p for p in storage_dir.iterdir() if looks_like_experiment_dir(p)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not experiments:
        raise SystemExit(
            f"No experiments found under {storage_dir}. Expected directories containing runs.jsonl and summary.json."
        )
    return experiments[0].name


def _as_actual_output(raw_actual: Any) -> dict[str, Any]:
    if isinstance(raw_actual, dict):
        return raw_actual
    if isinstance(raw_actual, str):
        try:
            return json.loads(raw_actual)
        except json.JSONDecodeError:
            return {}
    if isinstance(raw_actual, Mapping):
        return dict(raw_actual)
    return {}


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _extract_departments(run: Any) -> tuple[str, str]:
    """Extract expected vs. predicted department from a run record.

    For local evals, cat-experiments writes runs like:
        {
          "output": {"department": "HR"},
          "actual_output": {"department": "Finance", "tool_calls": [...]}
        }

    We prefer the explicit output/actual_output fields, and fall back to any
    tool_correctness metadata if present.
    """

    expected = "None"
    predicted = "None"

    output = _get_field(run, "output", {}) or {}
    if isinstance(output, dict):
        expected = output.get("department") or expected

    actual_output = _as_actual_output(_get_field(run, "actual_output"))
    predicted = actual_output.get("department") or predicted

    # If the model didn't return a direct department, try to infer from tool calls.
    if predicted == "None":
        tool_calls: Iterable[dict[str, Any]] = actual_output.get("tool_calls") or []
        for tc in tool_calls:
            predicted = tc.get("args", {}).get("department", "None") or "None"
            break

    # Fallback: some backends attach tool_correctness evaluator metadata.
    if expected == "None" or predicted == "None":
        tc_meta = (_get_field(run, "evaluator_metadata", {}) or {}).get(
            "tool_correctness"
        ) or {}
        matches: list[dict[str, Any]] = tc_meta.get("matches") or []
        if matches:
            match = matches[0]
            expected = (
                match.get("expected", {}).get("args", {}).get("department", expected)
                or expected
            )
            matched = match.get("matched") or {}
            predicted = (
                matched.get("args", {}).get("department", predicted) or predicted
            )
        elif tc_meta.get("missing_tools") and expected == "None":
            missing = tc_meta["missing_tools"][0]
            expected = missing.get("args", {}).get("department", expected) or expected

    # Fallback: Cat Cafe stores departments in evaluation metadata.
    if predicted == "None":
        evaluations = _get_field(run, "evaluations", []) or []
        for ev in evaluations:
            ev_meta = _get_field(ev, "metadata", {}) or {}
            if ev_meta.get("actual_department"):
                predicted = ev_meta["actual_department"]
                break

    return expected or "None", predicted or "None"


def calculate_department_metrics(runs: list[Any]) -> dict[str, dict[str, float]]:
    stats = {dept: {"tp": 0, "fp": 0, "fn": 0} for dept in DEPARTMENTS}

    for run in runs:
        expected, predicted = _extract_departments(run)
        for dept in DEPARTMENTS:
            if expected == dept and predicted == dept:
                stats[dept]["tp"] += 1
            elif expected != dept and predicted == dept:
                stats[dept]["fp"] += 1
            elif expected == dept and predicted != dept:
                stats[dept]["fn"] += 1

    metrics: dict[str, dict[str, float]] = {}
    for dept, s in stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        metrics[dept] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }
    return metrics


def build_confusion_matrix(runs: list[Any]) -> list[list[int]]:
    size = len(DEPARTMENTS)
    idx = {dept: i for i, dept in enumerate(DEPARTMENTS)}
    matrix = [[0 for _ in range(size)] for _ in range(size)]

    for run in runs:
        expected, predicted = _extract_departments(run)
        if expected in idx and predicted in idx:
            matrix[idx[expected]][idx[predicted]] += 1
    return matrix


def _overall_tool_correctness(summary: Any, runs: list[Any]) -> float:
    """Compute overall routing accuracy.

    Historically this script reported "tool correctness" for a tool-matching evaluator.
    For this workshop repo, the primary metric is routing accuracy:
        expected_department == predicted_department
    """

    if runs:
        correct = 0
        total = 0
        for run in runs:
            expected, predicted = _extract_departments(run)
            total += 1
            if expected == predicted:
                correct += 1
        return correct / total if total else 0.0

    # If we somehow have no runs, fall back to a summary average if present.
    if summary and _get_field(summary, "average_scores"):
        summary_scores = _get_field(summary, "average_scores", {}) or {}
        return summary_scores.get("tool_correctness", 0.0) or 0.0
    return 0.0


def render_summary(summary: Any, experiment_id: str, runs: list[Any]) -> None:
    total = _get_field(summary, "total_examples") or len(runs)
    duration_ms = _get_field(summary, "total_execution_time_ms", 0)
    routing_accuracy = _overall_tool_correctness(summary, runs)

    console.print(f"[bold]Experiment:[/bold] {experiment_id}")
    console.print(
        f"[bold]Total Examples:[/bold] {total} | "
        f"[bold]Routing Accuracy:[/bold] {routing_accuracy:.1%} | "
        f"[bold]Duration:[/bold] {duration_ms / 1000:.1f}s"
    )
    console.print()


def render_department_table(metrics: dict[str, dict[str, float]]) -> None:
    table = Table(
        title="Precision/Recall by Department",
        box=box.SIMPLE_HEAVY,
        header_style="bold",
    )
    table.add_column("Department")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("Support", justify="right")

    for dept in DEPARTMENTS:
        m = metrics.get(dept, {})
        table.add_row(
            dept,
            f"{m.get('precision', 0):.1%}",
            f"{m.get('recall', 0):.1%}",
            f"{m.get('f1', 0):.1%}",
            str(m.get("support", 0)),
        )
    console.print(table)
    console.print()


def render_confusion_matrix(matrix: list[list[int]], normalized: bool = False) -> None:
    title = (
        "Confusion Matrix (raw counts)"
        if not normalized
        else "Confusion Matrix (row-normalized %)"
    )
    table = Table(title=title, box=box.SIMPLE_HEAVY, header_style="bold")
    table.add_column("True \\ Pred", style="bold")
    for dept in DEPARTMENTS:
        table.add_column(dept, justify="right")

    for i, true_dept in enumerate(DEPARTMENTS):
        row_total = sum(matrix[i])
        row_cells = []
        for j in range(len(DEPARTMENTS)):
            val = matrix[i][j]
            if normalized:
                cell = 0.0 if row_total == 0 else val / row_total
                row_cells.append(f"{cell:.1%}")
            else:
                row_cells.append(str(val))
        table.add_row(true_dept, *row_cells)

    console.print(table)
    console.print()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _fetch_json(url: str) -> Any:
    """Fetch JSON from a URL."""
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise SystemExit(f"Failed to fetch {url}: {e}")


def _fetch_cat_cafe_experiments(base_url: str) -> list[dict[str, Any]]:
    """Fetch list of experiments from CAT Cafe."""
    data = _fetch_json(f"{base_url}/api/experiments")
    return data.get("experiments", [])


def _fetch_cat_cafe_runs(base_url: str, experiment_id: str) -> list[dict[str, Any]]:
    """Fetch runs for an experiment from CAT Cafe."""
    data = _fetch_json(f"{base_url}/api/experiments/{experiment_id}/runs")
    return data.get("runs", [])


def _latest_cat_cafe_experiment(base_url: str) -> dict[str, Any]:
    """Return the most recent experiment from CAT Cafe."""
    experiments = _fetch_cat_cafe_experiments(base_url)
    if not experiments:
        raise SystemExit(
            f"No experiments found in CAT Cafe at {base_url}. "
            "Run an experiment first with: uv run cat-experiments run ..."
        )
    # Sort by created_at descending and return the most recent
    experiments.sort(key=lambda e: e.get("created_at", ""), reverse=True)
    return experiments[0]


def fetch_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Fetch experiment summary + runs.

    Local mode reads the directory written by:
        cat-experiments run ... --backend local --output-dir eval_results

    CAT Cafe mode fetches from the CAT Cafe API.
    """

    target = args.target

    if target == "local":
        storage_dir = Path(args.storage_dir)
        exp_id = args.experiment_id or _latest_local_experiment(storage_dir)
        exp_dir = storage_dir / exp_id

        summary_path = exp_dir / "summary.json"
        runs_path = exp_dir / "runs.jsonl"
        if not summary_path.exists() or not runs_path.exists():
            raise SystemExit(
                f"Experiment not found or incomplete: {exp_dir}\n"
                f"Expected: {summary_path} and {runs_path}"
            )

        summary = _read_json(summary_path)
        runs = _read_jsonl(runs_path)
        return {"experiment_id": exp_id, "summary": summary, "results": runs}

    if target == "cat-cafe":
        base_url = args.cat_base_url.rstrip("/")

        if args.experiment_id:
            # Try to find experiment by ID or name
            experiments = _fetch_cat_cafe_experiments(base_url)
            experiment = None
            for exp in experiments:
                if (
                    exp.get("experiment_id") == args.experiment_id
                    or exp.get("name") == args.experiment_id
                ):
                    experiment = exp
                    break
            if not experiment:
                raise SystemExit(
                    f"Experiment '{args.experiment_id}' not found in CAT Cafe.\n"
                    f"Available experiments: {[e.get('name') or e.get('experiment_id') for e in experiments]}"
                )
        else:
            experiment = _latest_cat_cafe_experiment(base_url)

        exp_id = experiment.get("experiment_id", "")
        exp_name = experiment.get("name") or exp_id
        runs = _fetch_cat_cafe_runs(base_url, exp_id)

        # Build summary from experiment data
        summary = experiment.get("summary", {})
        summary["total_examples"] = summary.get("total_examples", len(runs))
        summary["total_execution_time_ms"] = sum(
            (r.get("metadata", {}).get("execution_time_ms", 0) or 0) for r in runs
        )

        return {"experiment_id": exp_name, "summary": summary, "results": runs}

    if target == "phoenix":
        raise SystemExit(
            "Phoenix target is not yet supported. Use --target cat-cafe or --target local."
        )

    raise SystemExit(f"Unsupported target: {target}")


def main() -> None:
    args = parse_args()
    plan = fetch_plan(args)
    summary = plan.get("summary")
    runs = plan.get("results") or []
    experiment_id = plan.get("experiment_id") or (args.experiment_id or "unknown")

    render_summary(summary, experiment_id, runs)
    metrics = calculate_department_metrics(runs)
    render_department_table(metrics)

    matrix = build_confusion_matrix(runs)
    render_confusion_matrix(matrix, normalized=False)
    if args.show_normalized:
        render_confusion_matrix(matrix, normalized=True)


if __name__ == "__main__":
    main()
