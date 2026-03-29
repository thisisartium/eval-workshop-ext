"""Session management with JSONL persistence.

A Session holds the conversation history (user/assistant messages) for a
single multi-turn interaction. Sessions persist to JSONL files so they
survive process restarts. SessionStore manages a directory of session files.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Session:
    """A conversation session with message history.

    Messages are stored as dicts with ``role`` and ``content`` keys,
    matching the format expected by the OpenAI Responses API's
    ``EasyInputMessageParam``.

    Attributes:
        session_id: Unique identifier for this session.
        messages: Ordered list of conversation messages.
        metadata: Arbitrary key-value pairs for application use.
        created_at: ISO 8601 timestamp of session creation.
    """

    session_id: str
    messages: list[dict[str, str]]
    metadata: dict[str, Any]
    created_at: str

    @classmethod
    def create(cls, session_id: str | None = None) -> "Session":
        """Create a new empty session.

        Args:
            session_id: Explicit ID to use. If None, a UUID is generated.

        Returns:
            A new Session with no messages.
        """
        return cls(
            session_id=session_id or uuid.uuid4().hex,
            messages=[],
            metadata={},
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def add_message(self, role: str, content: str) -> None:
        """Append a message to the conversation history.

        Args:
            role: The message role (``"user"`` or ``"assistant"``).
            content: The message text.
        """
        self.messages.append({"role": role, "content": content})

    def save(self, path: Path) -> None:
        """Write this session to a JSONL file.

        The first line is a session header with metadata. Each subsequent
        line is a message. The file is overwritten on each save.

        Args:
            path: File path to write to.
        """
        with open(path, "w") as f:
            header = {
                "type": "session",
                "session_id": self.session_id,
                "created_at": self.created_at,
                "metadata": self.metadata,
            }
            f.write(json.dumps(header) + "\n")
            for msg in self.messages:
                line = {"type": "message", "role": msg["role"], "content": msg["content"]}
                f.write(json.dumps(line) + "\n")

    @classmethod
    def load(cls, path: Path) -> "Session":
        """Load a session from a JSONL file.

        Args:
            path: File path to read from.

        Returns:
            The deserialized Session.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        with open(path) as f:
            lines = f.read().strip().split("\n")

        header = json.loads(lines[0])
        messages = [
            {"role": obj["role"], "content": obj["content"]}
            for obj in (json.loads(line) for line in lines[1:])
            if obj.get("type") == "message"
        ]

        return cls(
            session_id=header["session_id"],
            messages=messages,
            metadata=header.get("metadata", {}),
            created_at=header["created_at"],
        )


class SessionStore:
    """Manages a directory of session JSONL files.

    Each session is stored as ``{session_id}.jsonl`` in the configured
    directory. The store handles creation, loading, and listing.

    Attributes:
        directory: Path to the directory where session files are stored.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path_for(self, session_id: str) -> Path:
        """Return the file path for a given session ID."""
        return self.directory / f"{session_id}.jsonl"

    def get_or_create(self, session_id: str) -> Session:
        """Load an existing session or create a new one.

        Args:
            session_id: The session ID to look up or create.

        Returns:
            The existing session if found, otherwise a new empty session.
        """
        path = self._path_for(session_id)
        if path.exists():
            return Session.load(path)
        return Session.create(session_id=session_id)

    def get(self, session_id: str) -> Session | None:
        """Load a session by ID, or return None if it doesn't exist.

        Args:
            session_id: The session ID to look up.

        Returns:
            The session if found, otherwise None.
        """
        path = self._path_for(session_id)
        if path.exists():
            return Session.load(path)
        return None

    def save(self, session: Session) -> None:
        """Persist a session to disk.

        Args:
            session: The session to save.
        """
        session.save(self._path_for(session.session_id))

    def list_sessions(self) -> list[str]:
        """List all session IDs in the store.

        Returns:
            A list of session ID strings (without file extensions).
        """
        return [p.stem for p in self.directory.glob("*.jsonl")]
