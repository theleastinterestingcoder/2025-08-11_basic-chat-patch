"""
ConversationContext - Data structure for conversation state in ReasoningNode template method.

This class provides a clean abstraction for conversation data that gets passed
to specialized processing methods in ReasoningNode subclasses.
"""

from dataclasses import dataclass
from typing import Any, List, Optional

from cartesia.agents.events import EventInstance, UserTranscriptionReceived


@dataclass
class ConversationContext:
    """
    Encapsulates conversation state for ReasoningNode template method pattern.

    This standardizes how conversation data is passed between the template method
    (ReasoningNode.generate) and specialized processing (_process_context).

    Attributes:
        events: List of conversation events
        system_prompt: The system prompt for this reasoning node
        metadata: Additional context data for specialized processing
    """

    events: List[EventInstance]
    system_prompt: str
    metadata: dict = None

    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}

    def format_events(self, max_messages: int = None) -> str:
        """
        Format conversation messages as a string for LLM prompts.

        Args:
            max_messages: Maximum number of recent messages to include

        Returns:
            Formatted conversation string
        """
        events = self.events
        if max_messages is not None:
            events = events[-max_messages:]

        return "\n".join(f"{type(event)}: {event}" for event in events)

    def get_latest_user_transcript_message(self) -> Optional[str]:
        """Get the most recent user message content."""
        for msg in reversed(self.events):
            if isinstance(msg, UserTranscriptionReceived):
                return msg.content
        return None

    def get_event_count(self) -> int:
        """Get total number of messages in context."""
        return len(self.events)

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata for specialized processing."""
        self.metadata[key] = value
