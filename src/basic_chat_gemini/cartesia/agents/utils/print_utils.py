def truncate_for_printing(message: str, max_length: int = 400) -> str:
    """Truncate a message for printing to the console"""
    if len(message) > max_length:
        return message[: max_length // 2] + "[...]" + message[-max_length // 2 :]
    return message
