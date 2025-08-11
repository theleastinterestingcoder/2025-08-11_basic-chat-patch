"""
ReasoningNode

A base class for agent reasoning using the template method pattern.

- Handles conversation history and tool calls.
- Defines a standard flow for generating agent responses.
- Subclasses implement `_process_context()` to provide custom reasoning.

This class simplifies building agents that need both conversation management and tool integration.
"""

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, AsyncGenerator, List, Union

from loguru import logger

from cartesia.agents.bus import BusMessage
from cartesia.agents.events import (
    AgentGenerationComplete,
    AgentResponse,
    EventInstance,
    EventType,
    ToolCall,
    ToolResult,
    UserTranscriptionReceived,
)
from cartesia.agents.nodes.base import Node
from cartesia.agents.nodes.conversation_context import ConversationContext

if TYPE_CHECKING:
    pass


class ReasoningNode(Node):
    """
    Template method pattern for reasoning functionality.

    Manages conversation context, tool handling, and defines the generation flow.
    Subclasses implement _process_context() to provide specialized reasoning logic
    while inheriting conversation management and tool capabilities.

    Template Method Flow:
    1. generate() - Template method (defines the flow)
    2. _build_conversation_context() - Standard context building
    3. _process_context() - Subclass-specific processing (abstract)
    4. Tool handling - Automatic for NodeToolCall results
    """

    def __init__(
        self,
        system_prompt: str,
        max_context_length: int = 100,
        node_id: str = None,
    ):
        """
        Initialize the reasoning node

        Args:
            system_prompt: System prompt for the LLM
            max_context_length: Maximum number of conversation turns to keep
            node_id: Unique identifier for the node. Defaults to uuid4().
        """

        super().__init__(node_id=node_id)
        self.system_prompt = system_prompt
        self.max_context_length = max_context_length

        # Keep track of the conversation history, including user messages,
        # assistant messages, and tool calls.
        # This is a list of the events.
        self.conversation_events: List[Any] = []

        logger.info(f"{self} initialized")

    def on_interrupt_generate(self, message: BusMessage) -> None:
        """Handle interrupt event."""
        super().on_interrupt_generate(message)

        # Clear the conversation history until the last user message.
        # TODO: This is not the right way to do this.
        # while self.conversation_events[-1]["role"] != "user":
        #     self.conversation_events.pop()

    async def generate(
        self, message: BusMessage
    ) -> AsyncGenerator[Union[AgentResponse, ToolCall, ToolResult, EventType], None]:
        """Run the generation flow for all ReasoningNode subclasses.

        Users should implement :method:`_process_context` to provide specialized reasoning logic.

        Flow:
            1. Check for conversation messages
            2. Build conversation context. To override, implement :method:`_build_conversation_context`
            3. Call subclass-specific _process_context() method
            4. Yield all events (e.g. AgentResponse, ToolCall, ToolResult, etc.) for observability

        This method expects :method:`_process_context` to yield AgentResponse | ToolCall | ToolResult.
        All events are yielded to the bus. But different events are handled differently.

        - AgentResponse:
          - The text response from the LM.
          - If this is the speaking node, this will be sent to the user.
        - ToolCall: Record that the LM requested a tool call.
        - ToolResult: Record that the tool call was executed and the result.
          - Does not necessarily correspond to a previous ToolCall if the user decided not to yield one.
          - This is common for tool calls that are sync or run very quickly.
        - EventType: Custom result types (e.g., FormProcessingResult)

        Yields:
            AgentResponse: Text responses.
            ToolCall: Tool execution requests.
            ToolResult: Tool execution results.
            EventType: Custom result types (e.g., FormProcessingResult)
        """
        if not self.conversation_events:
            return

        # 1. Build standardized conversation context.
        ctx = self._build_conversation_context()

        # 2. Let subclass do specialized processing
        logger.info(f"💬 Processing context: {ctx.events}")
        async for chunk in self._process_context(ctx):
            # Save the event to the conversation history.
            self.add_event(chunk)

            # Yield the event to the user.
            yield chunk

        yield AgentGenerationComplete()

    @abstractmethod
    async def _process_context(
        self, context: ConversationContext
    ) -> AsyncGenerator[AgentResponse | ToolCall, None]:
        """
        Abstract method for subclass-specific processing logic.

        This is where subclasses implement their specialized reasoning:
        - Voice agents: Stream LLM responses
        - Form fillers: Extract structured data
        - RAG agents: Query knowledge bases
        - Chat agents: Generate conversational responses

        Args:
            context: Standardized conversation context with messages, tools, and metadata

        Yields:
            AgentResponse: Text content for the user
            ToolCall: Tool execution requests
            Custom types: Subclass-specific results (will be yielded directly)
        """
        # This is an abstract async generator - subclasses must implement
        raise NotImplementedError("Subclasses must implement _process_context")
        yield  # This makes it a generator function (unreachable)

    def _build_conversation_context(self) -> ConversationContext:
        """
        Build standardized conversation context for processing.

        This method creates a ConversationContext with recent messages, system prompt,
        and available tools. Used by the template method to provide consistent
        context to all subclasses.

        Returns:
            ConversationContext: Standardized context for _process_context()
        """
        # Use recent messages based on max_context_length
        recent_messages = self.conversation_events
        if len(recent_messages) > self.max_context_length:
            recent_messages = recent_messages[-self.max_context_length :]

        return ConversationContext(
            events=recent_messages,
            system_prompt=self.system_prompt,
            metadata={
                "max_context_length": self.max_context_length,
                "total_messages": len(self.conversation_events),
            },
        )

    def add_event(self, event: EventInstance):
        """
        Add an event to `self.conversation_events`.

        Events of type AgentResponse and UserTranscriptionReceived are merged if they are consecutive.
        This is useful to avoid having to merge the context of these events (i.e. the text) when we
        are building the conversation context.

        Args:
            event: The event to add to the conversation events.
        """
        # This is a utility because sometimes we get a BusMessage instead of an EventInstance.
        if isinstance(event, BusMessage):
            event = event.event

        if len(self.conversation_events) == 0:
            self.conversation_events.append(event)
            return

        # Merge the content of the same consecutive events for AgentResponse and UserTranscriptionReceived.
        # This allows us to easily build and send the conversation context to the LM.
        mergeable_events = (AgentResponse, UserTranscriptionReceived)
        for event_type in mergeable_events:
            if isinstance(event, event_type) and isinstance(self.conversation_events[-1], event_type):
                self.conversation_events[-1] = event_type(
                    content=self.conversation_events[-1].content + event.content
                )
                return

        self.conversation_events.append(event)
