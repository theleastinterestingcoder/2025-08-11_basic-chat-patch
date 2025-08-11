"""System tool definitions for Cartesia Voice Agents SDK."""

from typing import AsyncGenerator

from google.genai import types as gemini_types
from pydantic import BaseModel, Field

from cartesia.agents.events import AgentResponse, EndCall
from cartesia.agents.tools.tool_types import ToolDefinition


class EndCallArgs(BaseModel):
    """Arguments for the end_call tool."""

    goodbye_message: str = Field(description="The final message to say before ending the call")


class EndCallTool(ToolDefinition):
    """End call system tool definition.

    Usage example (Gemini):
    ```python
    self.generation_config = GenerateContentConfig(
        ...
        tools=[EndCallTool.to_gemini_tool()],
    )

    async def _process_context(
        self, context: ConversationContext
    ) -> AsyncGenerator[AgentResponse | EndCall, None]:
        ...
        function_call = <LLM function call request>
        if function_call.name == EndCallTool.name():
            goodbye_message = function_call.args.get("goodbye_message", "Goodbye!")
            args = EndCallArgs(goodbye_message=goodbye_message)
            async for item in end_call(args):
                yield item
    """

    @classmethod
    def name(cls) -> str:
        return "end_call"

    @classmethod
    def description(cls) -> str:
        return (
            "End the conversation with a goodbye message. "
            "Call this when the user says something 'goodbye' or something similar indicating they are ready "
            "to end the call."
            "Before calling this tool, do not send any text back, just use the goodbye_message field."
        )

    @classmethod
    def to_gemini_tool(cls) -> gemini_types.Tool:
        """Convert to Gemini tool format"""

        return gemini_types.Tool(
            function_declarations=[
                gemini_types.FunctionDeclaration(
                    name=cls.name(),
                    description=cls.description(),
                    parameters={
                        "type": "object",
                        "properties": {
                            "goodbye_message": {
                                "type": "string",
                                "description": EndCallArgs.model_fields["goodbye_message"].description,
                            }
                        },
                        "required": ["goodbye_message"],
                    },
                )
            ]
        )

    @classmethod
    def to_openai_tool(cls) -> dict[str, object]:
        """Convert to OpenAI tool format for Responses API.

        Note: This returns the format expected by OpenAI's Responses API,
        not the Chat Completions API format.
        """
        return {
            "type": "function",
            "name": cls.name(),
            "description": cls.description(),
            "parameters": {
                "type": "object",
                "properties": {
                    "goodbye_message": {
                        "type": "string",
                        "description": EndCallArgs.model_fields["goodbye_message"].description,
                    }
                },
                "required": ["goodbye_message"],
                "additionalProperties": False,
            },
            "strict": True,
        }


async def end_call(
    args: EndCallArgs,
) -> AsyncGenerator[AgentResponse | EndCall, None]:
    """
    End the call with a goodbye message.

    Yields:
        AgentResponse: The goodbye message to be spoken to the user
        EndCall: Event to end the call
    """
    # Send the goodbye message
    yield AgentResponse(content=args.goodbye_message)

    # End the call
    yield EndCall()
