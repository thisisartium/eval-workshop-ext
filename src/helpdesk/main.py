"""CLI entry point for the helpdesk agent.

Thin rendering layer over the platform's AgentClient. Supports two modes:

    Single-shot: ``python -m helpdesk.main "How many vacation days?"``
    Interactive: ``python -m helpdesk.main`` or ``python -m helpdesk.main -i``

All platform interaction goes through AgentClient.send().
"""

import argparse
import asyncio
import os
import sys
from collections.abc import AsyncGenerator

from agent_platform.client import AgentClient
from agent_platform.events import (
    AgentEnd,
    AgentStart,
    Event,
    TextDelta,
    ToolCallEnd,
    ToolCallStart,
)
from helpdesk.app import create_app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Argument list to parse. Defaults to sys.argv[1:].

    Returns:
        Parsed namespace with message, session, interactive, config,
        model, prompt_file, and tracing fields.
    """
    parser = argparse.ArgumentParser(
        prog="helpdesk",
        description="Helpdesk agent CLI — ask questions, get routed answers.",
    )
    parser.add_argument(
        "message",
        nargs="?",
        default=None,
        help="Message to send (single-shot mode). Omit for interactive.",
    )
    parser.add_argument(
        "--session",
        "-s",
        default=None,
        help="Session ID to resume or create.",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive REPL mode.",
    )
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="System config file (e.g. configs/tuned.yaml or just baseline.yaml). Default: system.yaml.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model for all agents (e.g. claude-sonnet-4-6, gpt-4o).",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Override routing prompt file (relative to helpdesk package).",
    )
    parser.add_argument(
        "--tracing",
        action="store_true",
        help="Enable OTel tracing (exports to OTLP collector).",
    )
    return parser.parse_args(argv)


async def render_events(events: AsyncGenerator[Event, None]) -> str:
    """Render agent events to the terminal.

    Status information (agent lifecycle, tool calls) goes to stderr.
    Response text goes to stdout. This separation allows piping the
    response: ``python -m helpdesk.main "question" > answer.txt``

    Args:
        events: Async generator of events from AgentClient.send().

    Returns:
        The concatenated response text.
    """
    text_parts: list[str] = []
    async for event in events:
        match event:
            case AgentStart(agent_name=name):
                print(f"[{name}] started", file=sys.stderr)
            case ToolCallStart(tool_name=name):
                print(f"  Calling {name}...", file=sys.stderr)
            case ToolCallEnd(tool_name=name, duration=dur):
                print(f"  {name} done ({dur:.1f}s)", file=sys.stderr)
            case TextDelta(text=text):
                print(text, end="", flush=True)
                text_parts.append(text)
            case AgentEnd():
                pass
    print()  # trailing newline after response text
    return "".join(text_parts)


async def run_single(client: AgentClient, message: str, session_id: str | None) -> None:
    """Run a single message through the agent and exit.

    Args:
        client: The platform client.
        message: User message to send.
        session_id: Optional session ID for persistence.
    """
    events = client.send(message, session_id=session_id)
    await render_events(events)


async def run_interactive(client: AgentClient, session_id: str | None) -> None:
    """Run an interactive conversation loop.

    Reads messages from stdin, sends each through the agent, and
    renders the response. Type 'quit' or 'exit' or Ctrl-C to stop.

    Args:
        client: The platform client.
        session_id: Optional session ID for persistence.
    """
    if session_id:
        print(f"Session: {session_id}", file=sys.stderr)
    else:
        from agent_platform.session import Session

        session_id = Session.create().session_id
        print(f"New session: {session_id}", file=sys.stderr)

    print("Type 'quit' to exit.\n", file=sys.stderr)

    try:
        while True:
            try:
                user_input = input("> ")
            except EOFError:
                break

            user_input = user_input.strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                break

            events = client.send(user_input, session_id=session_id)
            await render_events(events)
            print(file=sys.stderr)  # blank line between turns
    except KeyboardInterrupt:
        print("\n", file=sys.stderr)
    finally:
        print("Goodbye!", file=sys.stderr)


def main() -> None:
    """CLI entry point for the helpdesk agent."""
    from dotenv import load_dotenv

    load_dotenv(override=True)
    args = parse_args()

    processor = None
    if args.tracing or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        from agent_platform.tracing import configure_tracing

        processor = configure_tracing(service_name="helpdesk-agent")

    client = create_app(
        config=args.config,
        model=args.model,
        prompt_file=args.prompt_file,
        processor=processor,
    )

    if args.interactive or args.message is None:
        asyncio.run(run_interactive(client, args.session))
    else:
        asyncio.run(run_single(client, args.message, args.session))


if __name__ == "__main__":
    main()
