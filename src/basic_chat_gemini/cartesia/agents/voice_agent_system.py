"""
VoiceAgentSystem

A utility for building voice agent systems.

- Manages bus and harness setup automatically.
- Lets you add and customize components easily.
"""

from typing import Optional

from loguru import logger

from cartesia.agents.bridge import Bridge
from cartesia.agents.bus import Bus, BusMessage
from cartesia.agents.events import (
    AgentResponse,
    UserStartedSpeaking,
    UserStoppedSpeaking,
    UserTranscriptionReceived,
)
from cartesia.agents.harness import ConversationHarness
from cartesia.agents.nodes.reasoning import ReasoningNode
from cartesia.agents.user_bridge import create_user_bridge


class VoiceAgentSystem:
    """
    System builder for voice agent applications.

    Automatically manages Bus and ConversationHarness lifecycle
    while providing fluent API for adding bridges and components.
    """

    def __init__(self, websocket):
        """
        Create voice agent system with websocket.

        Args:
            websocket: FastAPI WebSocket connection for voice communication.
        """
        self.websocket = websocket
        self.bus = Bus()
        self.harness = ConversationHarness(websocket, self.bus.shutdown_event)
        self.bridges: dict[str, Bridge] = {}
        self.components = {}  # Store nodes and other components.
        self.authorized_node = None
        self.main_node: ReasoningNode | None = None  # Track the main reasoning node.

        self._add_logging_bridge()

    def _add_logging_bridge(self):
        """Add a bridge that logs all messages to the console."""
        bridge = Bridge("logging")

        def log_message(msg):
            logger.debug(f"Logger Bridge:\n\n{msg}")

        bridge.on("*").map(log_message)
        self.bridges["logging"] = bridge

    def _setup_authorized_infrastructure(self, authorized_node: str):
        """Setup user bridge for the authorized agent."""

        # Auto-add user bridge for WebSocket communication.
        user_bridge = create_user_bridge(self.harness, authorized_node)
        self.bridges["user"] = user_bridge

    def with_node(self, node: ReasoningNode, bridge: Optional[Bridge] = None):
        """
        Add reasoning node and optional bridge.

        Args:
            name: Node identifier.
            node: Reasoning node instance.
            bridge: Optional NodeBridge instance for the node.

        Returns:
            Self for method chaining.
        """
        self.components[node.id] = node

        # Track the main/authorized node.
        if self.authorized_node == node.id:
            self.main_node = node
        if bridge:
            self.with_bridge(node.id, bridge)
        return self

    def with_speaking_node(self, node: ReasoningNode, bridge: Bridge) -> "VoiceAgentSystem":
        """
        Add the speaking reasoning node and optional bridge, setting it as the authorized agent.

        Args:
            node: Reasoning node instance.
            bridge: Optional NodeBridge instance for the speaking node.

        Returns:
            Self for method chaining.
        """

        bridge.on(UserTranscriptionReceived).map(node.add_event)
        (
            bridge.on(UserStoppedSpeaking)
            # TODO (AD): Add on_exception that triggers when an exception is raised.
            # TODO: this handler is not quite working when we deploy the user code.
            .interrupt_on(UserStartedSpeaking, handler=node.on_interrupt_generate)
            .stream(node.generate)
            .broadcast()
        )

        if self.authorized_node is None:
            self.authorized_node = bridge.node_id
            self._setup_authorized_infrastructure(authorized_node=bridge.node_id)
            logger.info(f"VoiceAgentSystem: Set authorized agent to '{bridge.node_id}'")
        elif self.authorized_node != bridge.node_id:
            logger.warning(
                f"VoiceAgentSystem: Authorized agent already set to '{self.authorized_node}', "
                f"ignoring '{bridge.node_id}'"
            )

        result = self.with_node(node)
        if bridge:
            result = result.with_bridge(bridge.node_id, bridge)
        return result

    def with_main_bridge(self, bridge: Bridge) -> "VoiceAgentSystem":
        """
        Add the main bridge using the authorized agent name.

        Args:
            bridge: Configured NodeBridge instance.

        Returns:
            Self for method chaining.
        """
        if self.authorized_node is None:
            raise ValueError("Must call with_main_node() before with_main_bridge()")
        return self.with_bridge(self.authorized_node, bridge)

    def with_bridge(self, name: str, bridge: Bridge) -> "VoiceAgentSystem":
        """
        Add bridge to the system.

        Args:
            name: Bridge identifier for bus registration.
            bridge: Configured NodeBridge instance.

        Returns:
            Self for method chaining.
        """
        self.bridges[name] = bridge
        return self

    async def start(self):
        """Start the voice agent system."""

        # Register all bridges quietly first
        for name, bridge in self.bridges.items():
            logger.debug(f"VoiceAgentSystem: Registering bridge '{name}'")
            self.bus.register_bridge(name, bridge)

        # Initialize all reasoning nodes quietly
        for name, component in self.components.items():
            if hasattr(component, "init"):
                logger.debug(f"VoiceAgentSystem: Initializing component '{name}'")
                await component.init()

        # Start bus (which will show the summary) then other components
        await self.bus.start()
        await self.harness.start()

        # Start all bridges quietly
        for _, bridge in self.bridges.items():
            await bridge.start()

        logger.info("VoiceAgentSystem: All components started successfully")

    async def cleanup(self):
        """Clean shutdown of the voice agent system."""
        logger.info("VoiceAgentSystem: Starting system cleanup")

        # Stop bridges first.
        logger.debug(f"VoiceAgentSystem: Stopping {len(self.bridges)} bridges")
        for name, bridge in self.bridges.items():
            logger.debug(f"VoiceAgentSystem: Stopping bridge '{name}'")
            try:
                await bridge.stop()
            except Exception as e:
                logger.error(f"VoiceAgentSystem: Error stopping bridge '{name}': {e}")
                raise e

        # Cleanup reasoning nodes.
        logger.debug(f"VoiceAgentSystem: Cleaning up {len(self.components)} components")
        for name, component in self.components.items():
            if hasattr(component, "cleanup"):
                logger.debug(f"VoiceAgentSystem: Cleaning up component '{name}'")
                await component.cleanup()

        # Then stop infrastructure.
        logger.debug("VoiceAgentSystem: Cleaning up ConversationHarness")
        await self.harness.cleanup()

        logger.debug("VoiceAgentSystem: Cleaning up Bus")
        await self.bus.cleanup()

        logger.info("VoiceAgentSystem: System cleanup completed")

    async def wait_for_shutdown(self):
        """Wait until system should shut down (WebSocket disconnect)."""
        await self.bus.shutdown_event.wait()

    async def send_initial_message(self, message: str):
        """
        Send initial message to user and add as a message to the main node's conversation.

        Args:
            message (str): Initial greeting message.
        """
        logger.info(f"VoiceAgentSystem: Sending initial message: '{message[:50]}...'")
        await self.bus.broadcast(
            BusMessage(source=self.authorized_node, event=AgentResponse(content=message))
        )

        # Add to main node conversation history.
        if self.main_node and hasattr(self.main_node, "add_message"):
            logger.debug("VoiceAgentSystem: Adding initial message to main node conversation history")
            self.main_node.add_event(AgentResponse(content=message))
