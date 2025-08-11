from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class PreCallResult(BaseModel):
    """Result from pre_call_handler containing metadata and config."""

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata to include with the call")
    config: Dict[str, Any] = Field(default_factory=dict, description="Configuration for the call")


class CallRequest(BaseModel):
    """Request body for the /chats endpoint."""

    call_id: str
    from_: str = Field(alias="from")  # Using from_ to avoid Python keyword conflict
    to: str
    agent_call_id: str  # Agent call ID for logging and correlation
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        # Allow both field name (from_) and alias (from) for input
        populate_by_name = True


# Backward compatibility alias
ChatRequest = CallRequest
