from datetime import datetime, timezone
import json
import logging
import os
from typing import Awaitable, Callable, Optional
from urllib.parse import urlencode

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
import uvicorn

from cartesia.agents.call_request import CallRequest, PreCallResult
from cartesia.agents.voice_agent_system import VoiceAgentSystem

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class VoiceAgentApp:
    """
    VoiceAgentApp (name tbd) abstracts away the HTTP and websocket handling,
    which should be invisible to developers, because this transport may change
    in the future (eg to WebRTC).
    """

    def __init__(
        self,
        call_handler,
        pre_call_handler: Optional[Callable[[CallRequest], Awaitable[Optional[PreCallResult]]]] = None,
    ):
        self.fastapi_app = FastAPI()
        self.call_handler = call_handler
        self.pre_call_handler = pre_call_handler
        self.ws_route = "/ws"

        self.fastapi_app.add_api_route("/chats", self.create_chat_session, methods=["POST"])
        self.fastapi_app.add_api_route("/status", self.get_status, methods=["GET"])
        self.fastapi_app.add_websocket_route(self.ws_route, self.websocket_endpoint)

    async def create_chat_session(self, request: Request) -> dict:
        """Create a new chat session and return the websocket URL."""
        # Parse JSON body
        body = await request.json()

        # Create initial CallRequest
        call_request = CallRequest(
            call_id=body.get("call_id", "unknown"),
            from_=body.get("from", "unknown"),
            to=body.get("to", "unknown"),
            agent_call_id=body.get("agent_call_id", body.get("call_id", "unknown")),
            metadata=body.get("metadata", {}),
        )

        # Run pre-call handler if provided
        config = None
        if self.pre_call_handler:
            try:
                result = await self.pre_call_handler(call_request)
                if result is None:
                    raise HTTPException(status_code=403, detail="Call rejected")

                # Update call_request metadata with result
                call_request.metadata.update(result.metadata)
                config = result.config

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error in pre_call_handler: {str(e)}")
                raise HTTPException(status_code=500, detail="Server error in call processing") from e

        # Create URL parameters from processed call_request
        url_params = {
            "call_id": call_request.call_id,
            "from": call_request.from_,
            "to": call_request.to,
            "agent_call_id": call_request.agent_call_id,
            "metadata": json.dumps(call_request.metadata),  # JSON encode metadata
        }

        # Build websocket URL with parameters
        query_string = urlencode(url_params)
        websocket_url = f"{self.ws_route}?{query_string}"

        response = {"websocket_url": websocket_url}
        if config:
            response["config"] = config
        return response

    async def get_status(self) -> dict:
        """Status endpoint that returns OK if the server is running."""
        return {
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": "cartesia-line",
        }

    async def websocket_endpoint(self, websocket: WebSocket):
        """Websocket endpoint that manages the complete call lifecycle."""
        await websocket.accept()
        logger.info("Client connected")

        # Parse query parameters from WebSocket URL
        query_params = dict(websocket.query_params)

        # Parse metadata JSON
        metadata = {}
        if "metadata" in query_params:
            try:
                metadata = json.loads(query_params["metadata"])
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Invalid metadata JSON: {query_params['metadata']}")
                metadata = {}

        # Create CallRequest from URL parameters
        call_request = CallRequest(
            call_id=query_params.get("call_id", "unknown"),
            from_=query_params.get("from", "unknown"),
            to=query_params.get("to", "unknown"),
            agent_call_id=query_params.get("agent_call_id", "unknown"),
            metadata=metadata,
        )

        system = VoiceAgentSystem(websocket)

        try:
            # Handler configures nodes and bridges, then starts system
            await self.call_handler(system, call_request)
        except WebSocketDisconnect:
            logger.info("Client disconnected")
        except Exception as e:
            logger.exception(f"Error: {str(e)}")
            try:
                await system.harness.send_error("System has encountered an error, please try again later.")
                await system.harness.end_call()
            except:  # noqa: E722
                pass
        finally:
            await system.cleanup()

    def run(self, host="0.0.0.0", port=None):
        """Run the voice agent server."""
        port = port or int(os.getenv("PORT", 8000))
        uvicorn.run(self.fastapi_app, host=host, port=port)
