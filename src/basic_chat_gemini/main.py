import os

from chat_node import ChatNode
from config import SYSTEM_PROMPT
from google import genai

from cartesia.agents import ChatRequest
from cartesia.agents.bridge import Bridge
from cartesia.agents.voice_agent_app import VoiceAgentApp
from cartesia.agents.voice_agent_system import VoiceAgentSystem

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


async def handle_new_call(system: VoiceAgentSystem, chat_request: ChatRequest):
    # Main conversation node
    conversation_node = ChatNode(
        system_prompt=SYSTEM_PROMPT,
        gemini_client=gemini_client,
    )
    conversation_bridge = Bridge(conversation_node)
    system.with_speaking_node(conversation_node, bridge=conversation_bridge)

    await system.start()
    await system.send_initial_message("Hi there! I am a voice agent powered by Cartesia.")
    await system.wait_for_shutdown()


app = VoiceAgentApp(handle_new_call)

if __name__ == "__main__":
    app.run()
