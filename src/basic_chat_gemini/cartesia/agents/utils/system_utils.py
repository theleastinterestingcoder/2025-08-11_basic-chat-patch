"""
System utilities for detecting user intent from conversation history.

Provides pattern matching functions to detect when users want to end calls
or transfer to other agents/departments based on their message content.
"""

import re
from typing import Any, Dict, List


def user_wants_to_end_call(message_history: List[Dict[str, Any]]) -> bool:
    """
    Detect if the user wants to end the call based on their last message.

    Analyzes the most recent user message for phrases indicating intent
    to end the conversation.

    Args:
        message_history: List of conversation messages with 'role' and 'content' fields

    Returns:
        bool: True if user appears to want to end the call, False otherwise

    Examples:
        >>> messages = [{"role": "user", "content": "Thanks, goodbye!"}]
        >>> user_wants_to_end_call(messages)
        True

        >>> messages = [{"role": "user", "content": "Can you help me with something?"}]
        >>> user_wants_to_end_call(messages)
        False
    """
    if not message_history:
        return False

    # Find the last user message
    last_user_message = None
    for message in reversed(message_history):
        if message.get("role") == "user":
            last_user_message = message.get("content", "").lower().strip()
            break

    if not last_user_message:
        return False

    # End call patterns - phrases that indicate user wants to end the call
    end_call_patterns = [
        # Direct goodbye phrases
        r"\b(goodbye|good\s*bye|bye|farewell)\b",
        r"\b(see\s+you|talk\s+to\s+you)\s+(later|soon|tomorrow)\b",
        r"\b(have\s+a\s+)?(good|great|nice|wonderful)\s+(day|night|evening|afternoon|morning)\b",
        r"\btake\s+care\b",
        r"\bsee\s+ya\b",
        # Explicit ending statements
        r"\b(that\'s\s+)?(all|everything)\s+(i\s+)?(need|needed|want|wanted)\b",
        r"\b(thanks?|thank\s+you).*?(that\'s\s+)?(all|everything|it)\b",
        r"\bi\'m\s+(all\s+)?(done|finished|good|set)\b",
        r"\bthat\s+(should\s+)?(do\s+it|be\s+it|work)\b",
        r"\bnothing\s+else\b",
        r"\bi\s+(should|need\s+to|have\s+to|better)\s+go\b",
        r"\bi\'ll\s+(let\s+you\s+go|talk\s+to\s+you\s+later)\b",
        # Polite dismissal
        r"\b(thanks?|thank\s+you)\s+(so\s+)?(much|very\s+much)?\s*[.!]*\s*$",
        r"\bappreciate\s+(it|that|your\s+help)\b",
        r"\bthat\s+helps?\b.*thanks?\b",
        r"\bperfect\s*[.!]*\s*(thanks?|thank\s+you)?\s*$",
        r"\bgreat\s*[.!]*\s*(thanks?|thank\s+you)?\s*$",
        # Time-based endings
        r"\bi\s+(gotta|got\s+to|have\s+to|need\s+to)\s+(run|go)\b",
        r"\bi\'m\s+(running\s+)?(late|out\s+of\s+time)\b",
        r"\btalk\s+later\b",
    ]

    # Check if any end call pattern matches
    for pattern in end_call_patterns:
        if re.search(pattern, last_user_message, re.IGNORECASE):
            return True

    return False


def user_wants_to_transfer_call(message_history: List[Dict[str, Any]]) -> bool:
    """
    Detect if the user wants to transfer the call based on their last message.

    Analyzes the most recent user message for phrases indicating intent
    to transfer to another agent, department, or person.

    Args:
        message_history: List of conversation messages with 'role' and 'content' fields

    Returns:
        bool: True if user appears to want to transfer the call, False otherwise

    Examples:
        >>> messages = [{"role": "user", "content": "Can I speak to your manager?"}]
        >>> user_wants_to_transfer_call(messages)
        True

        >>> messages = [{"role": "user", "content": "Transfer me to billing please"}]
        >>> user_wants_to_transfer_call(messages)
        True

        >>> messages = [{"role": "user", "content": "How much does this cost?"}]
        >>> user_wants_to_transfer_call(messages)
        False
    """
    if not message_history:
        return False

    # Find the last user message
    last_user_message = None
    for message in reversed(message_history):
        if message.get("role") == "user":
            last_user_message = message.get("content", "").lower().strip()
            break

    if not last_user_message:
        return False

    # Transfer call patterns - phrases that indicate user wants to be transferred
    transfer_patterns = [
        # Direct transfer requests
        r"\b(transfer|connect|switch)\s+(me\s+)?(to|with)\b",
        r"\bcan\s+(i|you)\s+(talk\s+to|speak\s+(to|with)|be\s+transferred\s+to)\b",
        r"\bi\s+(want|need|would\s+like)\s+to\s+(talk\s+to|speak\s+(to|with))\b",
        r"\bput\s+me\s+through\s+to\b",
        r"\bforward\s+me\s+to\b",
        r"\bhand\s+me\s+(over\s+)?to\b",
        # Specific department/role requests
        r"\b(manager|supervisor|boss|director)\b",
        r"\b(sales|billing|support|technical|tech|customer\s+service|accounting|hr|human\s+resources)\b",
        r"\b(specialist|expert|representative|rep|agent)\b",
        r"\bsomeone\s+(who|that)\s+(can|knows|handles)\b",
        r"\banother\s+(person|agent|representative|rep)\b",
        # Department-specific requests
        r"\bbilling\s+(department|team|office)\b",
        r"\btechnical\s+(support|team|department)\b",
        r"\bcustomer\s+(service|support|care)\b",
        r"\bsales\s+(team|department|rep)\b",
        # Escalation requests
        r"\bi\s+(want|need)\s+to\s+(file\s+a\s+)?(complain|speak\s+to\s+someone\s+else)\b",
        r"\bthis\s+isn\'t\s+(working|helping)\b",
        r"\byou\s+(can\'t|cannot)\s+help\s+me\b",
        r"\bi\s+need\s+(higher|better|more)\s+(level|authority|help)\b",
        r"\blet\s+me\s+talk\s+to\s+someone\s+else\b",
        # Specific person requests
        r"\bis\s+there\s+(someone|anyone)\s+(else|who)\b",
        r"\bwho\s+else\s+can\s+(help|assist)\b",
        r"\banother\s+option\b",
    ]

    # Check if any transfer pattern matches
    for pattern in transfer_patterns:
        if re.search(pattern, last_user_message, re.IGNORECASE):
            return True

    return False
