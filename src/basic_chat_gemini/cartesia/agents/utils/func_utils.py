import asyncio
from typing import Any, Callable


async def run_sync_or_async(method: Callable, *args, **kwargs) -> Any:
    return await method(*args, **kwargs) if asyncio.iscoroutinefunction(method) else method(*args, **kwargs)
