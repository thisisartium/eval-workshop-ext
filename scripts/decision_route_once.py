#!/usr/bin/env python3
"""Smoke-test the Jev router on a single request.

Usage:
    uv run python scripts/jev_route_once.py "My laptop won't turn on"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from helpdesk.jev_router import route_result_as_dict, route_with_jev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route one request with Jev")
    parser.add_argument(
        "request",
        nargs="?",
        default="My laptop won't turn on. I've tried holding the power button.",
        help="Helpdesk request text",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    result = await route_with_jev(args.request)
    print(json.dumps(route_result_as_dict(result), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
