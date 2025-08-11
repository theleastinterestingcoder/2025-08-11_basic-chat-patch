import time

from cartesia.agents.nodes.reasoning import ReasoningNode
from cartesia.agents.utils.gemini_utils import (
    convert_messages_to_gemini,
    log_gemini_messages,
)


class SessionLogger:
    """
    Used for logging the conversation history before and after LLM generation.
    """

    def __init__(self, node: ReasoningNode):
        self.node = node
        self.start_time = None
        self.first_byte_time = None
        self.end_time = None
        self.latency = None

    def log_start(self):
        """
        Logs the state of the session (the conversation_history)
        in the reasoning node at the start of the session.
        """
        self.start_time = time.perf_counter()

        gemini_messages = convert_messages_to_gemini(self.node.conversation_events)
        log_gemini_messages("conversation history pre-llm generation", gemini_messages)

    def note_if_first(self):
        """
        Records the time of the first byte of the first message sent to the LLM.

        Might be called multiple times but only notes the first call.
        """
        if self.first_byte_time is None:
            self.first_byte_time = time.perf_counter()

            if self.start_time is not None:
                self.latency = self.first_byte_time - self.start_time

    def log_end(self):
        """
        Logs the state of the session (the conversation_history)
        in the reasoning node at the end of the session
        """
        self.end_time = time.perf_counter()
        total_duration = self.end_time - self.start_time
        time_to_first_byte = self.first_byte_time - self.start_time

        gemini_messages = convert_messages_to_gemini(self.node.conversation_events)
        log_gemini_messages(
            "conversation history post-llm generation",
            gemini_messages,
            statistics=(
                f"llm_time_to_first_byte={time_to_first_byte:.3f}s, total_duration={total_duration:.3f}s"
            ),
        )
