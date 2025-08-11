"""
Generic MCP Client for connecting to any MCP server
Supports multiple tool servers and generic query processing
"""

import asyncio
from contextlib import AsyncExitStack
from typing import Any, Dict, List

from loguru import logger
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from cartesia.agents.events import ToolResult


class MCPToolClient:
    """Generic MCP Client that can connect to any MCP-compatible tool server"""

    def __init__(self, server_name: str):
        """
        Initialize MCP client for a specific server

        Args:
            server_name: Name/identifier for this server instance
        """
        self.server_name = server_name
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self.available_tools = []
        self.is_connected = False

    async def connect_to_server(
        self,
        command: str,
        args: List[str],
        env: Dict[str, str] = None,
        descriptions: Dict[str, str] = None,
    ):
        """
        Connect to an MCP server

        Args:
            command: Command to run (e.g., "uv", "python")
            args: Arguments for the command (e.g., ["tool", "run", "mcp-server-qdrant"])
            env: Environment variables for the server
            descriptions: Optional override descriptions for specific tools (keyed by tool name)
        """
        server_params = StdioServerParameters(command=command, args=args, env=env or {})

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools

        # Convert MCP tools to standard format
        self.available_tools = []
        for tool in tools:
            # Use override description if provided, otherwise use original
            description = tool.description
            if descriptions and tool.name in descriptions:
                description = descriptions[tool.name]

            self.available_tools.append(
                {
                    "name": tool.name,
                    "description": description,
                    "inputSchema": tool.inputSchema,
                }
            )

        self.is_connected = True
        logger.info(
            f"✅ Connected to {self.server_name} with tools: "
            f"{[tool['name'] for tool in self.available_tools]}"
        )

    async def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> ToolResult:
        """
        Call a tool on the connected server

        # TODO(@voice-team): I am not certain this interface for tools is complete.

        Args:
            tool_name: Name of the tool to call
            tool_args: Arguments for the tool

        Returns:
            Tool result as dictionary with 'success', 'content', and 'error' fields
        """
        if not self.is_connected:
            return ToolResult(
                error="Client not connected to server",
                tool_name=tool_name,
                tool_args=tool_args,
            )

        try:
            result = await self.session.call_tool(tool_name, tool_args)

            # Extract text content from result
            content_text = ""
            for content_item in result.content:
                if hasattr(content_item, "text"):
                    content_text += content_item.text

            return ToolResult(
                result=None,  # MCP returns text, so we use result_str
                result_str=content_text,
                error=content_text if result.isError else None,
                tool_name=tool_name,
                tool_args=tool_args,
            )

        except Exception as e:
            error_msg = str(e)

            # Check for connection-related errors and mark as disconnected
            if any(
                indicator in error_msg.lower()
                for indicator in [
                    "connection closed",
                    "broken pipe",
                    "connection reset",
                    "eof",
                    "stream ended",
                ]
            ):
                print(f"⚠️ MCP connection lost for {self.server_name}: {error_msg}")
                self.is_connected = False
                return ToolResult(
                    error=f"Connection to {self.server_name} was lost. Please restart the service.",
                    tool_name=tool_name,
                    tool_args=tool_args,
                )

            return ToolResult(
                error=f"Error calling tool {tool_name}: {error_msg}",
                tool_name=tool_name,
                tool_args=tool_args,
            )

    async def cleanup(self):
        """Clean up resources properly"""
        self.is_connected = False

        # Clear references first
        exit_stack = self.exit_stack

        self.session = None
        self.stdio = None
        self.write = None

        # Close exit stack (this will terminate the subprocess)
        if exit_stack:
            try:
                await exit_stack.aclose()
            except (asyncio.CancelledError, Exception):
                # Ignore all cleanup errors including cancellation
                pass


class MCPToolManager:
    """Manages multiple MCP tool clients"""

    def __init__(self, clients: Dict[str, MCPToolClient] = None):
        self.clients: Dict[str, MCPToolClient] = clients or {}

    def get_client(self, name: str) -> MCPToolClient | None:
        """Get a client by name"""
        return self.clients.get(name)

    async def add_server(
        self,
        name: str,
        command: str,
        args: List[str],
        env: Dict[str, str] = None,
        descriptions: Dict[str, str] = None,
    ) -> MCPToolClient:
        """
        Add and connect to an MCP server if does not exist already

        Args:
            name: Name for this client instance
            command: Command to run the server
            args: Arguments for the command
            env: Environment variables
            descriptions: Optional override descriptions for specific tools (keyed by tool name)

        Returns:
            Connected MCPToolClient instance

        if a client with the same name already exists, we will not initialize a client to to that server
        """

        if name in self.clients:
            logger.info(f"MCP client {name=} already exists, skipping.")
            return self.clients[name]

        client = MCPToolClient(name)
        self.clients[name] = client
        await client.connect_to_server(command, args, env, descriptions)
        return client

    def get_all_tools(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all available tools from all connected clients"""
        all_tools = {}
        for name, client in self.clients.items():
            if client.is_connected:
                all_tools[name] = client.available_tools
        return all_tools

    async def execute_tool_call(
        self, tool_name: str, tool_args: Dict[str, Any], client_name: str = None
    ) -> ToolResult:
        """
        Execute a tool call, automatically finding the client if not specified

        Args:
            tool_name: Name of the tool to call
            tool_args: Arguments for the tool
            client_name: Optional specific client name

        Returns:
            Tool result with success, content, and error fields.
        """
        # If client_name not specified, try to find the tool
        client = None
        if not client_name:
            tools_by_client = self.get_all_tools()
            for name, tools in tools_by_client.items():
                for tool in tools:
                    if tool["name"] == tool_name:
                        client = self.get_client(name)
                        break
                if client:
                    break

        if not client:
            return ToolResult(
                error=f"Tool '{tool_name}' not found in any connected client",
                tool_name=tool_name,
                tool_args=tool_args,
            )

        return await client.call_tool(tool_name, tool_args)

    async def cleanup_all(self):
        """Clean up all clients properly"""
        # Clean up clients sequentially to avoid cancel scope issues
        for client in list(self.clients.values()):
            try:
                await client.cleanup()
            except (asyncio.CancelledError, Exception) as e:
                # Log but continue cleanup - ignore cancellation errors
                if not isinstance(e, asyncio.CancelledError):
                    print(f"Warning: Error cleaning up client {client.server_name}: {e}")

        self.clients.clear()

        # Give a moment for subprocess cleanup without cancellation
        try:
            await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass
