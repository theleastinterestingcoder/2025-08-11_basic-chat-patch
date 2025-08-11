"""
GeminiReasoningNode - Voice-optimized ReasoningNode implementation using proven Gemini logic
"""

from typing import AsyncGenerator

from config import DEFAULT_MODEL_ID, DEFAULT_TEMPERATURE
from google.genai import types as gemini_types
from loguru import logger

from cartesia.agents.events import AgentResponse, EndCall
from cartesia.agents.nodes.conversation_context import ConversationContext
from cartesia.agents.nodes.reasoning import ReasoningNode
from cartesia.agents.tools.system_tools import EndCallArgs, EndCallTool, end_call
from cartesia.agents.utils.gemini_utils import convert_messages_to_gemini


class ChatNode(ReasoningNode):
    """
    Voice-optimized ReasoningNode using template method pattern with Gemini streaming.
    - Uses ReasoningNode's template method generate() for consistent flow
    - Implements _process_context() for Gemini streaming
    - Integrates with end_call tool
    """

    def __init__(
        self,
        system_prompt: str,
        gemini_client,
        model_id: str = DEFAULT_MODEL_ID,
        temperature: float = DEFAULT_TEMPERATURE,
        max_context_length: int = 100,
        max_output_tokens: int = 1000,
    ):
        """
        Initialize the Voice reasoning node with proven Gemini configuration

        Args:
            system_prompt: System prompt for the LLM
            gemini_client: Google Gemini client instance
            model_id: Gemini model ID to use
            temperature: Temperature for generation
            max_context_length: Maximum number of conversation turns to keep
        """
        super().__init__(system_prompt=system_prompt, max_context_length=max_context_length)

        self.client = gemini_client
        self.model_id = model_id
        self.temperature = temperature

        # Interruption support
        self.stop_generation_event = None

        # Create generation config using utility function
        self.generation_config = gemini_types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            temperature=self.temperature,
            tools=[EndCallTool.to_gemini_tool()],
            max_output_tokens=max_output_tokens,
            thinking_config=gemini_types.ThinkingConfig(thinking_budget=0),
        )

        logger.info(f"GeminiNode initialized with model: {model_id}")

    async def _process_context(
        self, context: ConversationContext
    ) -> AsyncGenerator[AgentResponse | EndCall, None]:
        """
        Process the conversation context and yield responses from Gemini.

        Yields:
            AgentResponse: Text chunks from Gemini
            AgentEndCall: end_call Event
        """
        if not context.events:
            logger.info("No messages to process")
            return

        messages = convert_messages_to_gemini(context.events)

        user_message = context.get_latest_user_transcript_message()
        if user_message:
            logger.info(f'🧠 Processing user message: "{user_message}"')

        full_response = ""
        stream: AsyncGenerator[
            gemini_types.GenerateContentResponse
        ] = await self.client.aio.models.generate_content_stream(
            model=self.model_id,
            contents=messages,
            config=self.generation_config,
        )

        async for msg in stream:
            if msg.text:
                full_response += msg.text
                yield AgentResponse(content=msg.text)

            if msg.function_calls:
                for function_call in msg.function_calls:
                    if function_call.name == EndCallTool.name():
                        goodbye_message = function_call.args.get("goodbye_message", "Goodbye!")
                        args = EndCallArgs(goodbye_message=goodbye_message)
                        logger.info(
                            f"🤖 End call tool called. Ending conversation with goodbye message: "
                            f"{args.goodbye_message}"
                        )
                        async for item in end_call(args):
                            yield item

        if full_response:
            logger.info(f'🤖 Agent response: "{full_response}" ({len(full_response)} chars)')
