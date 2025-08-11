"""
Simplified RAG Demo using ReasoningAgent architecture
"""

import asyncio
import threading
import time
from typing import Any, Dict

from config import (
    RAG_MCP_TOOLS,
)
from loguru import logger

from cartesia.agents.tools.mcp_client import MCPToolClient
from cartesia.utils.logging import context_log


class ThreadSafeMCPManager:
    """
    Wraps MCPToolClient in its own event loop to allow it to be passed between event loops.

    We do this because modal calls the following seperately:
    - initialize
    - fastapi_endpoint

    Since we pass the univcorn app, we need to make sure that the MCP clients are initialized.
    """

    def __init__(self):
        self.thread = None
        self.loop = None
        self.clients = {}
        self.initialized = False
        self._shutdown_event = threading.Event()

    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initializes threads and clients"""
        if self.initialized:
            return True

        try:
            self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
            self.thread.start()

            time.sleep(0.1)

            # Initialize clients
            self._run_coro(self._initialize_clients(config))
            self.initialized = True
            return True
        except Exception as e:
            logger.exception(f"Error initializing MCP clients: {e}")
            return False

    def _run_event_loop(self):
        """Run the event loop in the dedicated thread"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            # Keep the loop running until shutdown
            self.loop.run_until_complete(self._run_forever())
        finally:
            self.loop.close()

    async def _run_forever(self):
        """Keep the event loop running"""
        while not self._shutdown_event.is_set():
            await asyncio.sleep(0.1)

    def _run_coro(self, coro):
        """Run a coroutine in the dedicated event loop"""
        if self.loop is None:
            raise RuntimeError("MCP manager not started")

        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result(timeout=30)

    async def _initialize_clients(self, config: Dict[str, Any]):
        """Actually initialize the MCP clients"""
        for tool_name, tool_config in config.items():
            client = MCPToolClient(tool_name)

            with context_log(f"Initializing MCP client {tool_name}"):
                await client.connect_to_server(
                    tool_config["command"],
                    tool_config["args"],
                    tool_config["env"],
                    tool_config["descriptions"],
                )
                self.clients[tool_name] = client

    def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the MCP clients"""
        if not self.initialized:
            return {"success": False, "error": "MCP manager not initialized"}

        try:
            return self._run_coro(self._do_call_tool(tool_name, tool_args))
        except Exception as e:
            logger.exception(f"Error calling tool {tool_name}: {e}")
            return {"success": False, "error": str(e)}

    async def _do_call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Actually call the tool"""
        # Find the client that has this tool
        for _, client in self.clients.items():
            if client.is_connected:
                for tool in client.available_tools:
                    if tool["name"] == tool_name:
                        return await client.call_tool(tool_name, tool_args)

        return {"success": False, "error": f"Tool '{tool_name}' not found"}

    def get_available_tools(self) -> Dict[str, Any]:
        """Get available tools from all clients"""
        if not self.initialized:
            return {}

        try:
            return self._run_coro(self._do_get_tools())
        except Exception as e:
            logger.exception(f"Error getting tools: {e}")
            return {}

    async def _do_get_tools(self) -> Dict[str, Any]:
        """Actually get the tools"""
        all_tools = {}
        for client_name, client in self.clients.items():
            if client.is_connected:
                all_tools[client_name] = client.available_tools
        return all_tools


class MCPClientProxy:
    """
    Proxy that makes SimpleMCPManager look like the original MCPToolClient
    """

    def __init__(self, manager: ThreadSafeMCPManager, server_name: str):
        self.manager = manager
        self.server_name = server_name

    async def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool (async interface)"""
        # Run the sync call in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.manager.call_tool, tool_name, tool_args)

    @property
    def is_connected(self):
        return self.manager.initialized

    @property
    def available_tools(self):
        # Get tools for this specific server
        all_tools = self.manager.get_available_tools()
        return all_tools.get(self.server_name, [])

    async def cleanup(self):
        """Cleanup is handled by the manager"""
        pass


class MCPClientDict:
    """Dictionary-like interface for MCP clients"""

    def __init__(self, manager: ThreadSafeMCPManager):
        self.manager = manager

    def __getitem__(self, key):
        return MCPClientProxy(self.manager, key)

    def __contains__(self, key):
        return key in RAG_MCP_TOOLS

    def keys(self):
        return RAG_MCP_TOOLS.keys()

    def values(self):
        return [MCPClientProxy(self.manager, key) for key in RAG_MCP_TOOLS.keys()]

    def items(self):
        return [(key, MCPClientProxy(self.manager, key)) for key in RAG_MCP_TOOLS.keys()]

    def get(self, key, default=None):
        if key in RAG_MCP_TOOLS:
            return MCPClientProxy(self.manager, key)
        return default

    def clear(self):
        pass
