"""Tests for concurrent session dispatch (PR B)."""

import asyncio
import pytest

from unittest.mock import AsyncMock, MagicMock, patch


def _make_msg(channel="feishu", chat_id="user1", content="hi"):
    """Create a minimal InboundMessage-like object."""
    msg = MagicMock()
    msg.channel = channel
    msg.chat_id = chat_id
    msg.content = content
    msg.metadata = {}
    msg.message_id = None
    return msg


class TestConcurrentDispatch:
    """Verify _dispatch provides per-session serialization + cross-session concurrency."""

    @pytest.mark.asyncio
    async def test_same_session_serialized(self):
        """Two messages to the same session should not overlap."""
        from nanobot.agent.loop import AgentLoop

        loop = MagicMock(spec=AgentLoop)
        loop._session_locks = {}
        loop._dispatch_sem = asyncio.Semaphore(5)
        loop.bus = MagicMock()
        loop.bus.publish_outbound = AsyncMock()

        overlap_detected = False
        running = asyncio.Event()

        async def slow_process(msg):
            nonlocal overlap_detected
            if running.is_set():
                overlap_detected = True
            running.set()
            await asyncio.sleep(0.05)
            running.clear()
            return None

        loop._process_message = slow_process

        msg1 = _make_msg(chat_id="same")
        msg2 = _make_msg(chat_id="same")

        # Call the real _dispatch
        await asyncio.gather(
            AgentLoop._dispatch(loop, msg1),
            AgentLoop._dispatch(loop, msg2),
        )
        assert not overlap_detected, "Same-session messages should not overlap"

    @pytest.mark.asyncio
    async def test_different_sessions_concurrent(self):
        """Messages to different sessions should run concurrently."""
        from nanobot.agent.loop import AgentLoop

        loop = MagicMock(spec=AgentLoop)
        loop._session_locks = {}
        loop._dispatch_sem = asyncio.Semaphore(5)
        loop.bus = MagicMock()
        loop.bus.publish_outbound = AsyncMock()

        max_concurrent = 0
        active = 0
        lock = asyncio.Lock()

        async def track_process(msg):
            nonlocal max_concurrent, active
            async with lock:
                active += 1
                if active > max_concurrent:
                    max_concurrent = active
            await asyncio.sleep(0.05)
            async with lock:
                active -= 1
            return None

        loop._process_message = track_process

        msgs = [_make_msg(chat_id=f"user{i}") for i in range(3)]
        await asyncio.gather(*(AgentLoop._dispatch(loop, m) for m in msgs))

        assert max_concurrent >= 2, f"Expected concurrent execution, got max_concurrent={max_concurrent}"

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self):
        """Global semaphore should cap concurrent tasks."""
        from nanobot.agent.loop import AgentLoop

        loop = MagicMock(spec=AgentLoop)
        loop._session_locks = {}
        loop._dispatch_sem = asyncio.Semaphore(2)  # cap at 2
        loop.bus = MagicMock()
        loop.bus.publish_outbound = AsyncMock()

        max_concurrent = 0
        active = 0
        lock = asyncio.Lock()

        async def track_process(msg):
            nonlocal max_concurrent, active
            async with lock:
                active += 1
                if active > max_concurrent:
                    max_concurrent = active
            await asyncio.sleep(0.05)
            async with lock:
                active -= 1
            return None

        loop._process_message = track_process

        msgs = [_make_msg(chat_id=f"user{i}") for i in range(5)]
        await asyncio.gather(*(AgentLoop._dispatch(loop, m) for m in msgs))

        assert max_concurrent <= 2, f"Semaphore cap exceeded: max_concurrent={max_concurrent}"

    @pytest.mark.asyncio
    async def test_error_handling_in_dispatch(self):
        """Errors in _process_message should be caught and published."""
        from nanobot.agent.loop import AgentLoop

        loop = MagicMock(spec=AgentLoop)
        loop._session_locks = {}
        loop._dispatch_sem = asyncio.Semaphore(5)
        loop.bus = MagicMock()
        loop.bus.publish_outbound = AsyncMock()

        async def failing_process(msg):
            raise ValueError("boom")

        loop._process_message = failing_process

        msg = _make_msg()
        await AgentLoop._dispatch(loop, msg)

        loop.bus.publish_outbound.assert_called_once()
        error_msg = loop.bus.publish_outbound.call_args[0][0]
        assert "boom" in error_msg.content

    @pytest.mark.asyncio
    async def test_session_lock_pruned_after_use(self):
        """Unused session locks should be cleaned up."""
        from nanobot.agent.loop import AgentLoop

        loop = MagicMock(spec=AgentLoop)
        loop._session_locks = {}
        loop._dispatch_sem = asyncio.Semaphore(5)
        loop.bus = MagicMock()
        loop.bus.publish_outbound = AsyncMock()

        async def noop_process(msg):
            return None

        loop._process_message = noop_process

        msg = _make_msg(chat_id="ephemeral")
        await AgentLoop._dispatch(loop, msg)

        assert "feishu:ephemeral" not in loop._session_locks
