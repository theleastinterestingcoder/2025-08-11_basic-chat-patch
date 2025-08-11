from typing import AsyncGenerator
from uuid import uuid4

from loguru import logger

from cartesia.agents.bus import BusMessage
from cartesia.agents.events import EventType


class Node:
    """A base class for all nodes.

    Nodes are the building blocks of the agentic system. They are responsible for:
    - Maintaining state
    - Generating responses
    - Handling tool calls
    - Interrupting the generation process

    Nodes are stateful, and can be used to build multi-agent workflows.

    All nodes have an `id` that is used to identify them.
    When a :class:`Bridge` is created from a node, the node's `id` is used to identify the node in the bridge.
    It can be used when filtering by `source` or `target`.
    We do not require that nodes have a unique `id`.
    """

    def __init__(self, node_id: str = None):
        self.id = node_id or uuid4().hex

    async def init(self):
        """Initialize the node."""
        pass

    def __str__(self):
        return f"{type(self).__name__}(id={self.id})"

    async def cleanup(self):
        """Clean up the node."""
        logger.debug(f"{self} cleanup completed")

    def on_interrupt_generate(self, message: BusMessage) -> None:
        """Handle interrupt event.

        Args:
            message: The interrupt message.
        """
        logger.debug(f"{self} interrupt received.")

    async def generate(self, message: BusMessage) -> AsyncGenerator[EventType, None]:
        """Generate a response to the message."""
        pass
