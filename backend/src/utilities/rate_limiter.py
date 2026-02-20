import asyncio
import time
from typing import Optional

class AsyncRateLimiter:
    """
    An asynchronous rate limiter that uses a token bucket-like strategy 
    to limit requests per minute (RPM).
    """
    def __init__(self, rpm: int = 15):
        self.rpm = rpm
        self.interval = 60.0 / rpm if rpm > 0 else 0
        self.last_request_time = 0.0
        self._lock = asyncio.Lock()

    async def wait(self):
        """
        Wait until it's safe to make the next request.
        """
        if self.rpm <= 0:
            return

        async with self._lock:
            current_time = asyncio.get_event_loop().time()
            elapsed = current_time - self.last_request_time
            
            wait_time = self.interval - elapsed
            if wait_time > 0:
                print(f"[RateLimiter] Throttling: waiting {wait_time:.2f}s...")
                await asyncio.sleep(wait_time)
                self.last_request_time = asyncio.get_event_loop().time()
            else:
                self.last_request_time = current_time

    def __repr__(self):
        return f"AsyncRateLimiter(rpm={self.rpm}, interval={self.interval:.2f}s)"
