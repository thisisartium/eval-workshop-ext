#!/usr/bin/env python3
"""Load a JSONL dataset into Cat Cafe or Phoenix.

Expects dataset files in DatasetExample format:
    {"id": "...", "input": {...}, "output": {...}, "metadata": {...}}
"""

import argparse
import json
import httpx
from pathlib import Path
from typing import Any, Optional


def _get_nested_value(obj: dict[str, Any], path: str) -> Any:
    """Get a value from a nested dict using dot notation (e.g., 'output.department')."""
    keys = path.split(".")
    value = obj
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return None
    return value


def read_dataset(
    path: Path,
    limit: Optional[int] = None,
    filter_path: Optional[str] = None,
    filter_value: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Read JSONL dataset file in DatasetExample format."""
    data: list[dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            if line.strip():
                example = json.loads(line)
                # Apply filter if specified
                if filter_path and filter_value is not None:
                    actual_value = _get_nested_value(example, filter_path)
                    if str(actual_value) != filter_value:
                        continue
                data.append(example)
    if limit is not None:
        data = data[:limit]
    return data


# --------------------------------------------------------------------------- #
# Cat Cafe
# --------------------------------------------------------------------------- #


def ensure_cat_cafe_dataset(
    *,
    dataset_name: str,
    examples: list[dict[str, Any]],
    base_url: Optional[str],
) -> tuple[str, Optional[str]]:
    try:
        from cat.cafe.client import CATCafeClient
        from cat.cafe.client.client import DatasetExample as CafeDatasetExample
        from cat.cafe.client.client import DatasetImport as CafeDatasetImport
    except ImportError as exc:  # pragma: no cover - optional dependency
        print(
            "cat-cafe-client is required. Install with `pip install cat-cafe-client`."
        )
        raise

    client = CATCafeClient(base_url=base_url) if base_url else CATCafeClient()

    dataset_id, dataset_version_id = _cat_cafe_find_dataset(client, dataset_name)
    if dataset_id:
        # Dataset exists; assume examples already present to avoid duplicates.
        return dataset_id, dataset_version_id

    cafe_examples = [
        CafeDatasetExample(
            input=ex.get("input", {}),
            output=ex.get("output", {}),
            metadata=dict(ex.get("metadata", {})),
            id=ex.get("id"),
        )
        for ex in examples
    ]

    payload = CafeDatasetImport(
        name=dataset_name,
        description=f"Helpdesk dataset {dataset_name}",
        metadata={"source": "helpdesk-agent"},
        examples=cafe_examples,
    )

    result = client.import_dataset(payload)
    dataset_id = result.get("id") or result.get("dataset_id") or dataset_name
    dataset_version_id = result.get("version") or result.get("version_id")
    return dataset_id, dataset_version_id


def _cat_cafe_find_dataset(
    client: Any, dataset_name: str
) -> tuple[Optional[str], Optional[str]]:
    # Prefer SDK method that already filters by name
    find_by_name = getattr(client, "find_dataset_by_name", None)
    if callable(find_by_name):
        try:
            result = find_by_name(dataset_name)
        except httpx.HTTPStatusError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                result = None
            else:
                raise
        if result:
            if isinstance(result, dict):
                return (
                    result.get("id") or result.get("dataset_id") or dataset_name,
                    result.get("version") or result.get("version_id"),
                )
            return str(result), None

    # Fallback: list and search
    for method in ("list_datasets",):
        fn = getattr(client, method, None)
        if callable(fn):
            try:
                items = fn()
            except Exception:
                items = []
            for item in items or []:
                if item.get("name") == dataset_name:
                    return (
                        item.get("id") or item.get("dataset_id") or dataset_name,
                        item.get("version") or item.get("version_id"),
                    )

    return None, None


# --------------------------------------------------------------------------- #
# Phoenix
# --------------------------------------------------------------------------- #


def ensure_phoenix_dataset(
    *,
    dataset_name: str,
    examples: list[dict[str, Any]],
    project_name: Optional[str],
) -> tuple[str, Optional[str]]:
    try:
        from phoenix.client import Client as PhoenixClient
    except ImportError:
        print("phoenix-client is required. Install with `pip install phoenix-client`.")
        raise

    client = (
        PhoenixClient(project_name=project_name) if project_name else PhoenixClient()
    )

    # Normalize examples to the minimal shape Phoenix expects
    upload_examples = [
        {
            "input": ex.get("input", {}),
            "output": ex.get("output", {}),
            "metadata": dict(ex.get("metadata", {})),
        }
        for ex in examples
    ]

    try:
        existing = client.datasets.get_dataset(dataset=dataset_name)
        # Dataset exists; reuse it without re-uploading to avoid duplicates
        return existing.id, existing.version_id
    except Exception:
        pass

    created = client.datasets.create_dataset(
        name=dataset_name,
        examples=upload_examples,
        dataset_description=f"Helpdesk dataset {dataset_name}",
    )
    return created.id, created.version_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register a helpdesk JSONL dataset with Cat Cafe or Phoenix"
    )
    parser.add_argument("dataset_file", help="Path to helpdesk dataset JSONL file")
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of records to import (for smoke tests)",
    )
    parser.add_argument(
        "--target",
        choices=["cat-cafe", "phoenix"],
        default="cat-cafe",
        help="Where to register the dataset",
    )
    parser.add_argument(
        "--name",
        help="Dataset name (defaults to dataset file stem)",
    )
    parser.add_argument(
        "--cat-base-url",
        default="http://localhost:8000",
        help="CAT Cafe base URL",
    )
    parser.add_argument(
        "--phoenix-project",
        help="Phoenix project name (optional)",
    )
    parser.add_argument(
        "--filter",
        metavar="PATH=VALUE",
        help="Filter examples by JSON path (e.g., --filter output.department=HR)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.dataset_file)
    if not path.exists():
        print(f"Dataset file not found: {path}")
        raise SystemExit(1)

    dataset_name = args.name or path.stem

    # Parse filter argument
    filter_path = None
    filter_value = None
    if args.filter:
        if "=" not in args.filter:
            print("Filter must be in format PATH=VALUE (e.g., output.department=HR)")
            raise SystemExit(1)
        filter_path, filter_value = args.filter.split("=", 1)

    examples = read_dataset(
        path, limit=args.limit, filter_path=filter_path, filter_value=filter_value
    )

    if args.target == "cat-cafe":
        dataset_id, dataset_version = ensure_cat_cafe_dataset(
            dataset_name=dataset_name, examples=examples, base_url=args.cat_base_url
        )
        print(f"Registered dataset '{dataset_name}' with Cat Cafe.")
        print(f"Dataset ID: {dataset_id}")
        if dataset_version:
            print(f"Dataset version: {dataset_version}")
    else:
        dataset_id, dataset_version = ensure_phoenix_dataset(
            dataset_name=dataset_name,
            examples=examples,
            project_name=args.phoenix_project,
        )
        print(f"Registered dataset '{dataset_name}' with Phoenix.")
        print(f"Dataset ID: {dataset_id}")
        if dataset_version:
            print(f"Dataset version: {dataset_version}")


if __name__ == "__main__":
    main()
