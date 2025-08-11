#!/usr/bin/env python3
"""
Simple CLI chat interface for testing voice agent templates.

This creates a command-line chat session that connects to voice agent templates
by calling their /chats endpoint to get a WebSocket URL, then connecting via WebSocket.
"""

import argparse
import asyncio
from collections import deque
import curses
from datetime import datetime
import json
import logging
import os
import sys
import textwrap
import time

import httpx
import websockets

# Passed as query parameter to /chats and websocket URL
AGENT_CALL_ID = "ac_" + datetime.now().strftime("%Y%m%d%H%M%S")

# Set up logging to file to avoid interfering with curses
logging.basicConfig(
    level=logging.DEBUG,
    filename="chat_debug.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Message:
    """Represents a single chat message."""

    def __init__(
        self,
        content: str,
        is_user: bool,
        timestamp: datetime | None = None,
        latency_ms: int | None = None,
        is_tool_call: bool = False,
        second_turn_latency_ms: int | None = None,
    ):
        self.content = content
        self.is_user = is_user
        self.is_tool_call = is_tool_call
        self.timestamp = timestamp or datetime.now()
        self.is_complete = is_user  # User messages are always complete
        self.latency_ms = latency_ms  # Time to first response in milliseconds
        self.second_turn_latency_ms = second_turn_latency_ms  # Time from first to second agent turn
        self._word_count = 0  # Cache word count for performance

    def add_chunk(self, chunk: str):
        """Add a chunk to the message content."""
        # Simply concatenate chunks without adding spaces
        self.content += chunk
        # Update word count cache
        self._word_count = len(self.content.split()) if self.content else 0

    def get_word_count(self) -> int:
        """Get the current word count."""
        if self._word_count == 0 and self.content:
            self._word_count = len(self.content.split())
        return self._word_count


class ChatInterface:
    """Terminal-based chat interface using curses."""

    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.messages: deque[Message] = deque(maxlen=1000)
        self.input_buffer = ""
        self.cursor_pos = 0
        self.scroll_offset = 0
        self.agent_typing = False
        self.connected = False
        self.status_message = "Initializing..."
        self.auto_scroll = True  # Whether to auto-scroll to new messages

        # Setup colors
        curses.start_color()
        curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)  # User messages
        curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Agent messages
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Status
        curses.init_pair(4, curses.COLOR_RED, curses.COLOR_BLACK)  # Errors
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLACK)  # Input
        curses.init_pair(6, curses.COLOR_BLUE, curses.COLOR_BLACK)  # Header

        # Configure window
        self.height, self.width = stdscr.getmaxyx()
        curses.curs_set(1)  # Show cursor
        stdscr.nodelay(True)  # Non-blocking input
        stdscr.keypad(True)  # Enable special keys

    def draw(self):
        """Draw the entire interface."""
        self.stdscr.erase()

        # Draw header
        self._draw_header()

        # Draw messages
        self._draw_messages()

        # Draw status line
        self._draw_status()

        # Draw input area
        self._draw_input()

        self.stdscr.noutrefresh()
        curses.doupdate()

    def _draw_header(self):
        """Draw the header."""
        header = "🤖 Voice Agent Chat Interface"
        help_text = "↑↓ Scroll | PgUp/PgDn Page | Home/End Top/Bottom | Ctrl+C Exit"

        self.stdscr.attron(curses.color_pair(6))
        self.stdscr.addstr(0, 0, header.center(self.width))
        self.stdscr.attroff(curses.color_pair(6))

        # Show help text if terminal is wide enough
        if len(help_text) <= self.width:
            try:
                self.stdscr.attron(curses.color_pair(3))  # Yellow for help text
                self.stdscr.addstr(1, 0, help_text.center(self.width))
                self.stdscr.attroff(curses.color_pair(3))
                self.stdscr.addstr(2, 0, "─" * self.width)
            except curses.error:
                # Fallback if can't display help text
                self.stdscr.addstr(1, 0, "─" * self.width)
        else:
            self.stdscr.addstr(1, 0, "─" * self.width)

    def _draw_messages(self):
        """Draw the message history."""
        # Calculate available space for messages
        header_lines = 3  # Title, help text, separator
        status_lines = 1
        input_lines = 3
        available_lines = self.height - header_lines - status_lines - input_lines

        if available_lines <= 0:
            return

        # Get visible messages
        visible_messages = self._get_visible_messages(available_lines)

        # Draw messages
        y = header_lines
        for msg_lines in visible_messages:
            for line in msg_lines:
                if y >= self.height - status_lines - input_lines:
                    break

                # Color different message types
                if line.startswith("💬 You:"):
                    color = curses.color_pair(1)  # User messages in cyan
                elif line.startswith("🔧 Tool Call:"):
                    color = curses.color_pair(3)  # Tool calls in yellow
                else:
                    color = 0  # Default color for agent messages

                # Truncate line if too long (accounting for display width)
                # Don't truncate - rely on word wrapping instead
                line = line[: self.width] if len(line) > self.width else line

                try:
                    if color:
                        self.stdscr.attron(color)
                        self.stdscr.addstr(y, 0, line)
                        self.stdscr.attroff(color)
                    else:
                        self.stdscr.addstr(y, 0, line)
                except curses.error:
                    pass  # Ignore if we can't draw

                y += 1

    def _get_visible_messages(self, available_lines: int) -> list[list[str]]:
        """Get messages that should be visible in the given space."""
        all_lines = []

        # Convert messages to lines with word wrapping
        for msg in self.messages:
            if msg.is_user:
                prefix = "💬 You: "
            elif msg.is_tool_call:
                prefix = "🔧 Tool Call: "
            else:
                # Include latency metrics for agent messages
                parts = ["🤖 Agent"]
                if msg.latency_ms is not None:
                    parts.append(f"{msg.latency_ms}ms")

                if msg.second_turn_latency_ms is not None:
                    parts.append(f"{msg.second_turn_latency_ms}ms")

                if len(parts) > 1:
                    prefix = f"{parts[0]} ({' '.join(parts[1:])}): "
                else:
                    prefix = f"{parts[0]}: "
            # Use left margin for continuation lines, not prefix indentation
            max_content_width = self.width - 2  # Leave margin

            formatted_lines = []

            # Handle empty messages (for streaming)
            if not msg.content:
                # Add typing indicator for incomplete empty agent messages
                if not msg.is_user and not msg.is_complete and self.agent_typing:
                    formatted_lines.append(prefix + "▋")
                else:
                    formatted_lines.append(prefix)
            else:
                # Split by newlines first
                paragraphs = msg.content.split("\n")

                for i, paragraph in enumerate(paragraphs):
                    # Word wrap each paragraph
                    if i == 0:
                        # First paragraph gets the prefix
                        # Calculate available width after prefix for first line
                        first_line_width = max_content_width - len(prefix)
                        wrapped = self._wrap_text(paragraph, first_line_width)
                        if wrapped:
                            formatted_lines.append(prefix + wrapped[0])
                            # Continuation lines use the remaining text and wrap to full width
                            if len(wrapped) > 1:
                                remaining_text = " ".join(wrapped[1:])
                                continuation_wrapped = self._wrap_text(remaining_text, max_content_width)
                                formatted_lines.extend(continuation_wrapped)
                        else:
                            formatted_lines.append(prefix)
                    else:
                        # Subsequent paragraphs use full width and align left
                        wrapped = self._wrap_text(paragraph, max_content_width)
                        formatted_lines.extend(wrapped)

                # Add typing indicator for incomplete agent messages
                if not msg.is_user and not msg.is_complete and self.agent_typing:
                    if formatted_lines:
                        formatted_lines[-1] += " ▋"

            all_lines.append(formatted_lines)

        # Apply scroll offset
        if self.auto_scroll:
            # Auto-scroll: show the most recent messages
            visible_lines = []
            total_lines = 0

            # Start from the most recent messages
            for msg_lines in reversed(all_lines):
                lines_needed = len(msg_lines)
                if total_lines + lines_needed <= available_lines:
                    visible_lines.insert(0, msg_lines)
                    total_lines += lines_needed
                else:
                    # Partial message display
                    remaining = available_lines - total_lines
                    if remaining > 0:
                        visible_lines.insert(0, msg_lines[-remaining:])
                    break
        else:
            # Manual scroll: respect scroll_offset
            # Flatten all lines for easier scrolling
            flat_lines = []
            for msg_lines in all_lines:
                flat_lines.extend(msg_lines)

            # Apply scroll offset
            start_line = max(0, len(flat_lines) - available_lines - self.scroll_offset)
            end_line = start_line + available_lines

            # Get the visible lines
            visible_flat_lines = flat_lines[start_line:end_line]

            # Convert back to message groups for display consistency
            visible_lines = []
            current_group = []
            for line in visible_flat_lines:
                if line.startswith(("💬 You:", "🤖 Agent", "🔧 Tool Call:")):
                    # New message starts
                    if current_group:
                        visible_lines.append(current_group)
                    current_group = [line]
                else:
                    # Continuation line
                    if current_group:
                        current_group.append(line)
                    else:
                        # Orphaned continuation line, create a new group
                        current_group = [line]

            if current_group:
                visible_lines.append(current_group)

        return visible_lines

    def _wrap_text(self, text: str, width: int) -> list[str]:
        """Wrap text to fit within the given width."""
        if not text:
            return []

        # Use textwrap to handle word wrapping
        wrapped = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
        return wrapped if wrapped else [""]

    def _draw_status(self):
        """Draw the status line."""
        y = self.height - 4
        self.stdscr.addstr(y, 0, "─" * self.width)

        # Connection status
        if self.connected:
            status = "✅ Connected"
            color = curses.color_pair(2)
        else:
            status = "❌ Disconnected"
            color = curses.color_pair(4)

        # Add scroll indicator
        scroll_indicator = ""
        if not self.auto_scroll:
            scroll_indicator = f" | ↕ Manual scroll (offset: {self.scroll_offset})"
        elif self.messages:
            scroll_indicator = " | 🔄 Auto-scroll"

        # Status message
        status_text = f"{status} | {self.status_message}{scroll_indicator}"
        if len(status_text) > self.width - 2:
            status_text = status_text[: self.width - 5] + "..."

        self.stdscr.attron(color)
        self.stdscr.addstr(y + 1, 1, status_text)
        self.stdscr.attroff(color)

    def _draw_input(self):
        """Draw the input area."""
        y = self.height - 2

        # Input prompt
        prompt = "› "
        self.stdscr.attron(curses.color_pair(5))
        self.stdscr.addstr(y, 0, prompt)

        # Input text
        visible_input = self.input_buffer
        max_input_width = self.width - len(prompt) - 1

        # Handle long input with scrolling
        if len(visible_input) > max_input_width:
            if self.cursor_pos > max_input_width - 10:
                # Scroll to show cursor
                start = self.cursor_pos - max_input_width + 10
                visible_input = visible_input[start : start + max_input_width]
                cursor_x = len(prompt) + (self.cursor_pos - start)
            else:
                visible_input = visible_input[:max_input_width]
                cursor_x = len(prompt) + self.cursor_pos
        else:
            cursor_x = len(prompt) + self.cursor_pos

        self.stdscr.addstr(y, len(prompt), visible_input)
        self.stdscr.attroff(curses.color_pair(5))

        # Position cursor
        try:
            self.stdscr.move(y, cursor_x)
        except curses.error:
            pass

    def add_message(self, content: str, is_user: bool) -> Message | None:
        """Add a new message to the chat."""
        # Allow empty content for messages that will be filled via streaming
        if is_user and not content.strip():
            return None

        msg = Message(content, is_user)
        self.messages.append(msg)

        # Auto-scroll to new messages if not manually scrolling
        if self.auto_scroll:
            self.scroll_offset = 0

        self.draw()
        return msg

    def add_tool_call_message(self, tool_call_content: str) -> Message:
        """Add a tool call message to the chat."""
        msg = Message(tool_call_content, is_user=False, is_tool_call=True)
        msg.is_complete = True
        self.messages.append(msg)
        self.draw()
        return msg

    def get_current_agent_message(self) -> Message | None:
        """Get the current incomplete agent message."""
        if self.messages and not self.messages[-1].is_user and not self.messages[-1].is_complete:
            return self.messages[-1]
        return None

    def handle_input(self, key: int) -> str | None:
        """Handle keyboard input. Returns message to send if Enter pressed."""
        if key == curses.KEY_ENTER or key == ord("\n"):
            # Send message
            if self.input_buffer.strip():
                message = self.input_buffer
                self.input_buffer = ""
                self.cursor_pos = 0
                # Enable auto-scroll when sending a new message
                self.auto_scroll = True
                self.scroll_offset = 0
                return message

        elif key == curses.KEY_BACKSPACE or key == 127:
            # Delete character before cursor
            if self.cursor_pos > 0:
                self.input_buffer = (
                    self.input_buffer[: self.cursor_pos - 1] + self.input_buffer[self.cursor_pos :]
                )
                self.cursor_pos -= 1

        elif key == curses.KEY_DC:
            # Delete character at cursor
            if self.cursor_pos < len(self.input_buffer):
                self.input_buffer = (
                    self.input_buffer[: self.cursor_pos] + self.input_buffer[self.cursor_pos + 1 :]
                )

        elif key == curses.KEY_LEFT:
            # Move cursor left (only when typing, not scrolling)
            if self.input_buffer and self.cursor_pos > 0:
                self.cursor_pos -= 1

        elif key == curses.KEY_RIGHT:
            # Move cursor right (only when typing, not scrolling)
            if self.input_buffer and self.cursor_pos < len(self.input_buffer):
                self.cursor_pos += 1

        elif key == curses.KEY_UP:
            # Scroll up through messages
            self.auto_scroll = False
            self.scroll_offset += 1
            # Limit scroll offset to prevent scrolling beyond available messages
            max_offset = self._get_max_scroll_offset()
            if self.scroll_offset > max_offset:
                self.scroll_offset = max_offset

        elif key == curses.KEY_DOWN:
            # Scroll down through messages
            if self.scroll_offset > 0:
                self.scroll_offset -= 1
            else:
                # Re-enable auto-scroll when scrolled to bottom
                self.auto_scroll = True
                self.scroll_offset = 0

        elif key == curses.KEY_PPAGE:  # Page Up
            # Scroll up by a page
            self.auto_scroll = False
            page_size = max(1, (self.height - 6) // 2)  # Half screen height
            self.scroll_offset += page_size
            max_offset = self._get_max_scroll_offset()
            if self.scroll_offset > max_offset:
                self.scroll_offset = max_offset

        elif key == curses.KEY_NPAGE:  # Page Down
            # Scroll down by a page
            page_size = max(1, (self.height - 6) // 2)  # Half screen height
            if self.scroll_offset > page_size:
                self.scroll_offset -= page_size
            else:
                # Re-enable auto-scroll when scrolled to bottom
                self.auto_scroll = True
                self.scroll_offset = 0

        elif key == curses.KEY_HOME:
            # Move to beginning of input or scroll to top
            if self.input_buffer:
                self.cursor_pos = 0
            else:
                # Scroll to top of messages
                self.auto_scroll = False
                self.scroll_offset = self._get_max_scroll_offset()

        elif key == curses.KEY_END:
            # Move to end of input or scroll to bottom
            if self.input_buffer:
                self.cursor_pos = len(self.input_buffer)
            else:
                # Scroll to bottom (auto-scroll)
                self.auto_scroll = True
                self.scroll_offset = 0

        elif 32 <= key <= 126:
            # Printable character
            self.input_buffer = (
                self.input_buffer[: self.cursor_pos] + chr(key) + self.input_buffer[self.cursor_pos :]
            )
            self.cursor_pos += 1

        return None

    def _get_max_scroll_offset(self) -> int:
        """Calculate the maximum scroll offset based on available messages."""
        if not self.messages:
            return 0

        # Calculate available space for messages
        header_lines = 3  # Title, help text, separator
        status_lines = 1
        input_lines = 3
        available_lines = self.height - header_lines - status_lines - input_lines

        if available_lines <= 0:
            return 0

        # Calculate total lines from all messages
        total_lines = 0
        for msg in self.messages:
            # Estimate lines per message (same logic as _get_visible_messages)
            if msg.is_user:
                prefix = "💬 You: "
            elif msg.is_tool_call:
                prefix = "🔧 Tool Call: "
            else:
                prefix = "🤖 Agent: "

            max_content_width = self.width - 2
            if not msg.content:
                total_lines += 1
            else:
                paragraphs = msg.content.split("\n")
                for i, paragraph in enumerate(paragraphs):
                    if i == 0:
                        first_line_width = max_content_width - len(prefix)
                        wrapped = self._wrap_text(paragraph, first_line_width)
                        if wrapped:
                            total_lines += 1  # First line with prefix
                            if len(wrapped) > 1:
                                remaining_text = " ".join(wrapped[1:])
                                continuation_wrapped = self._wrap_text(remaining_text, max_content_width)
                                total_lines += len(continuation_wrapped)
                    else:
                        wrapped = self._wrap_text(paragraph, max_content_width)
                        total_lines += len(wrapped)

        # Maximum scroll offset is total lines minus available screen space
        return max(0, total_lines - available_lines)

    def set_status(self, message: str):
        """Update the status message."""
        self.status_message = message
        self.draw()

    def set_connected(self, connected: bool):
        """Update connection status."""
        self.connected = connected
        self.draw()

    def set_agent_typing(self, typing: bool):
        """Update agent typing status."""
        self.agent_typing = typing
        self.draw()


class ChatSession:
    """WebSocket chat session handler."""

    def __init__(self, url: str, interface: ChatInterface):
        self.url = url
        self.interface = interface
        self.ws = None
        self.running = True
        self.message_sent_time = None  # Track when user message was sent
        self.first_agent_response_time = None  # Track when first agent response was received

    async def connect(self):
        """Connect to the voice agent WebSocket."""
        try:
            # Always call /chats endpoint to get WebSocket URL
            self.interface.set_status(f"Getting WebSocket URL from {self.url}/chats...")
            websocket_url = await get_websocket_url_from_http(self.url)

            self.interface.set_status(f"Connecting to {websocket_url}...")
            self.ws = await websockets.connect(websocket_url)
            self.interface.set_connected(True)
            self.interface.set_status("Ready. Type your message and press Enter. Ctrl+C to exit.")
            logger.info(f"Connected to {websocket_url}")
            return True
        except Exception as e:
            self.interface.set_status(f"Connection failed: {str(e)}")
            logger.error(f"Failed to connect: {e}")
            return False

    async def disconnect(self):
        """Disconnect from the voice agent."""
        if self.ws:
            await self.ws.close()
            self.interface.set_connected(False)
            self.interface.set_status("Disconnected")
            logger.info("Disconnected from voice agent")

    async def send_message(self, message: str):
        """Send a message to the voice agent."""
        if not self.ws:
            self.interface.set_status("Not connected!")
            return

        try:
            # Complete any outstanding agent message before adding user message
            current_agent_msg = self.interface.get_current_agent_message()
            if current_agent_msg:
                current_agent_msg.is_complete = True
                self.interface.set_agent_typing(False)

            # Add user message to interface
            self.interface.add_message(message, is_user=True)

            # Record time when message was sent and reset first response time
            self.message_sent_time = time.time()
            self.first_agent_response_time = None  # Reset for new conversation

            # Send user_state idle to trigger response
            await self.ws.send(json.dumps({"type": "user_state", "value": "speaking"}))

            # Send the message
            await self.ws.send(json.dumps({"type": "message", "content": message}))

            # Send user_state idle to trigger response
            await self.ws.send(json.dumps({"type": "user_state", "value": "idle"}))

            logger.info(f"Sent message: {message}")

        except Exception as e:
            self.interface.set_status(f"Error sending message: {str(e)}")
            logger.error(f"Error sending message: {e}")

    async def listen_for_responses(self):
        """Listen for responses from the voice agent."""
        try:
            async for message in self.ws:
                if not self.running:
                    break

                try:
                    data = json.loads(message)
                    logger.debug(f"Received: {data}")

                    if data.get("type") == "message" and "content" in data:
                        content = data.get("content", "")

                        # Get or create current agent message
                        current_msg = self.interface.get_current_agent_message()
                        if not current_msg:
                            # Calculate latency based on whether this is first or second turn
                            latency_ms = None
                            second_turn_latency_ms = None

                            current_time = time.time()

                            if self.message_sent_time:
                                # This is the first agent response
                                latency_ms = int((current_time - self.message_sent_time) * 1000)
                                self.first_agent_response_time = current_time
                                self.message_sent_time = None  # Reset for next message
                            elif self.first_agent_response_time:
                                # This is a second agent turn (supervisor response)
                                second_turn_latency_ms = int(
                                    (current_time - self.first_agent_response_time) * 1000
                                )
                                self.first_agent_response_time = None  # Reset for next conversation

                            # Create new message - start with empty content for streaming
                            current_msg = Message(
                                "",
                                is_user=False,
                                latency_ms=latency_ms,
                                second_turn_latency_ms=second_turn_latency_ms,
                            )
                            current_msg.is_complete = False
                            self.interface.messages.append(current_msg)
                            self.interface.set_agent_typing(True)
                            self.interface.draw()

                        # Always append content to current message
                        if current_msg and content:
                            current_msg.add_chunk(content)
                            # Force immediate update
                            self.interface.draw()
                            curses.doupdate()

                    elif data.get("type") == "message_end":
                        # Mark current message as complete
                        current_msg = self.interface.get_current_agent_message()
                        if current_msg:
                            current_msg.is_complete = True
                            self.interface.set_agent_typing(False)
                            self.interface.draw()

                    elif data.get("type") == "tool_calls":
                        # Display tool call information
                        tool_calls = data.get("content", [])
                        if tool_calls:
                            tool_call_text = json.dumps(tool_calls, indent=2)
                            self.interface.add_tool_call_message(tool_call_text)

                    elif data.get("type") == "error":
                        error_msg = f"Agent Error: {data.get('content', 'Unknown error')}"
                        self.interface.add_message(error_msg, is_user=False)
                        self.interface.set_agent_typing(False)

                except json.JSONDecodeError:
                    logger.warning(f"Received invalid JSON: {message}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")

        except websockets.exceptions.ConnectionClosed:
            self.interface.set_connected(False)
            self.interface.set_status("Connection lost")
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error in message listener: {e}")
            self.interface.set_connected(False)
            self.interface.set_status(f"Connection error: {str(e)}")


async def input_handler(interface: ChatInterface, send_queue: asyncio.Queue):
    """Handle user input in a separate thread."""
    while True:
        try:
            key = interface.stdscr.getch()
            if key != -1:  # Key was pressed
                message = interface.handle_input(key)
                if message:
                    await send_queue.put(message)
                interface.draw()

            await asyncio.sleep(0.01)  # Small delay to prevent CPU spinning

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error in input handler: {e}")


async def main_async(stdscr):
    """Main async function."""
    # Create interface
    interface = ChatInterface(stdscr)
    interface.draw()

    # Get agent URL
    agent_port = int(os.getenv("PORT", 8000))
    agent_url = f"http://localhost:{agent_port}"

    # Check for custom URL
    if len(sys.argv) > 1:
        agent_url = sys.argv[1]

    # Create chat session
    chat = ChatSession(agent_url, interface)

    # Connect to agent
    if not await chat.connect():
        await asyncio.sleep(3)  # Show error for a moment
        return

    # Create message queue
    send_queue = asyncio.Queue()

    # Start tasks
    listener_task = asyncio.create_task(chat.listen_for_responses())
    input_task = asyncio.create_task(input_handler(interface, send_queue))

    try:
        # Main loop
        while chat.running:
            try:
                # Check for messages to send
                message = await asyncio.wait_for(send_queue.get(), timeout=0.1)
                await chat.send_message(message)
            except asyncio.TimeoutError:
                pass
            except KeyboardInterrupt:
                interface.set_status("Shutting down...")
                break

    finally:
        # Clean up
        chat.running = False
        listener_task.cancel()
        input_task.cancel()

        try:
            await listener_task
        except asyncio.CancelledError:
            pass

        try:
            await input_task
        except asyncio.CancelledError:
            pass

        await chat.disconnect()


async def simple_chat(url: str, prompt: str):
    """Simple non-interactive single turn chat for command line testing."""
    print(f"🔗 Connecting to {url}...")

    try:
        # Always call /chats endpoint to get WebSocket URL
        print(f"📞 Getting WebSocket URL from {url}/chats...")
        websocket_url = await get_websocket_url_from_http(url)
        print(f"🔗 Using WebSocket URL: {websocket_url}")

        # Connect to WebSocket
        async with websockets.connect(websocket_url) as websocket:
            print("✅ Connected to voice agent")
            print(f"💬 Sending: {prompt}")

            # Send the message
            await websocket.send(json.dumps({"type": "message", "content": prompt}))

            # Send user_state idle to trigger response
            await websocket.send(json.dumps({"type": "user_state", "value": "idle"}))

            response_content = ""
            start_time = time.time()
            first_token_time = None
            response_started = False
            agent_header_printed = False

            # Listen for responses with timeout
            async def listen_with_timeout():
                async for message in websocket:
                    try:
                        data = json.loads(message)

                        if data.get("type") == "message" and "content" in data:
                            content = data.get("content", "")
                            if content:
                                nonlocal \
                                    response_content, \
                                    response_started, \
                                    first_token_time, \
                                    agent_header_printed
                                if not response_started:
                                    first_token_time = time.time()
                                    first_token_latency_ms = int((first_token_time - start_time) * 1000)
                                    print(f"🤖 Agent response ({first_token_latency_ms} ms):")
                                    agent_header_printed = True
                                    response_started = True
                                response_content += content
                                print(content, end="", flush=True)

                        elif data.get("type") == "message_end":
                            print()  # New line after complete response
                            return

                        elif data.get("type") == "tool_calls":
                            tool_calls = data.get("content", [])
                            if tool_calls:
                                print(f"\n🔧 Tool calls: {json.dumps(tool_calls, indent=2)}")

                        elif data.get("type") == "error":
                            print(f"\n❌ Error: {data.get('content', 'Unknown error')}")
                            return

                        elif data.get("type") == "end_call":
                            print("\n📞 Call ended")
                            return

                    except json.JSONDecodeError:
                        logger.warning(f"Received invalid JSON: {message}")
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")

            try:
                # Wait for response with 2 second timeout
                await asyncio.wait_for(listen_with_timeout(), timeout=2.0)
            except asyncio.TimeoutError:
                print("\n✅ Session completed!")

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return 1

    return 0


async def get_websocket_url_from_http(http_url: str) -> str:
    """Get WebSocket URL by calling /chats endpoint on HTTP URL."""
    if not (http_url.startswith("http://") or http_url.startswith("https://")):
        raise ValueError("URL must start with http:// or https://")

    # Remove trailing slash if present
    base_url = http_url.rstrip("/")
    chats_url = f"{base_url}/chats?agent_call_id={AGENT_CALL_ID}"

    # Create empty payload as mentioned in the requirements
    payload = {}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(chats_url, json=payload)
            response.raise_for_status()

            data = response.json()
            websocket_url = data.get("websocket_url")

            if not websocket_url:
                raise ValueError("No websocket_url in response")

            # Convert relative WebSocket URL to absolute
            if websocket_url.startswith("/"):
                # Convert http/https to ws/wss + websocket_url
                if http_url.startswith("https://"):
                    host = http_url[8:]  # Remove "https://"
                    protocol = "wss"
                else:
                    host = http_url[7:]  # Remove "http://"
                    protocol = "ws"

                if host.endswith("/"):
                    host = host[:-1]  # Remove trailing slash
                websocket_url = f"{protocol}://{host}{websocket_url}"

            if "?" not in websocket_url:
                websocket_url = f"{websocket_url}?agent_call_id={AGENT_CALL_ID}"
            else:
                websocket_url = f"{websocket_url}&agent_call_id={AGENT_CALL_ID}"

            logger.info(f"Got WebSocket URL from /chats: {websocket_url}")
            return websocket_url

    except Exception as e:
        logger.error(f"Failed to get WebSocket URL from {chats_url}: {e}")
        raise


def run_interactive(stdscr):
    """Main function to run with curses."""
    # Run the async main
    asyncio.run(main_async(stdscr))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat interface for voice agent templates")
    parser.add_argument("url", nargs="?", help="HTTP URL (default: http://localhost:8000)")
    parser.add_argument("-p", "--prompt", help="Single prompt for non-interactive testing")

    args = parser.parse_args()

    # Determine the URL
    if args.url:
        agent_url = args.url
    else:
        agent_port = int(os.getenv("PORT", 8000))
        agent_url = f"http://localhost:{agent_port}"

    if not agent_url.startswith("http"):
        agent_url = f"https://{agent_url}"

    try:
        if args.prompt:
            # Non-interactive mode with single prompt
            exit_code = asyncio.run(simple_chat(agent_url, args.prompt))
            sys.exit(exit_code)
        else:
            # Interactive curses mode
            # Check if running in a terminal
            if not sys.stdout.isatty():
                print("This chat interface requires a terminal. Please run in a terminal.")
                print("Or use -p option for non-interactive testing.")
                sys.exit(1)

            # Run with curses
            curses.wrapper(run_interactive)

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\n❌ Unexpected error: {e}")
        print("Check chat_debug.log for details.")
