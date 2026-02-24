"""Tests for concurrent session processing."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.task_context import (
    current_channel,
    current_chat_id,
    current_message_id,
    message_sent_in_turn,
)
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.cron import CronTool
from nanobot.bus.events import InboundMessage, OutboundMessage


# ---------------------------------------------------------------------------
# ContextVar isolation tests
# ---------------------------------------------------------------------------

class TestContextVarIsolation:
    """Verify ContextVars are isolated across concurrent asyncio tasks."""

    @pytest.mark.asyncio
    async def test_channel_isolation_across_tasks(self):
        """Two concurrent tasks setting different channels don't interfere."""
        results = {}

        async def task_a():
            current_channel.set("feishu")
            current_chat_id.set("chat_a")
            await asyncio.sleep(0.05)  # yield to task_b
            results["a_channel"] = current_channel.get()
            results["a_chat"] = current_chat_id.get()

        async def task_b():
            current_channel.set("slack")
            current_chat_id.set("chat_b")
            await asyncio.sleep(0.05)
            results["b_channel"] = current_channel.get()
            results["b_chat"] = current_chat_id.get()

        await asyncio.gather(
            asyncio.create_task(task_a()),
            asyncio.create_task(task_b()),
        )

        assert results["a_channel"] == "feishu"
        assert results["a_chat"] == "chat_a"
        assert results["b_channel"] == "slack"
        assert results["b_chat"] == "chat_b"

    @pytest.mark.asyncio
    async def test_sent_in_turn_isolation(self):
        """message_sent_in_turn is isolated per task."""
        results = {}

        async def task_a():
            message_sent_in_turn.set(True)
            await asyncio.sleep(0.05)
            results["a"] = message_sent_in_turn.get()

        async def task_b():
            message_sent_in_turn.set(False)
            await asyncio.sleep(0.05)
            results["b"] = message_sent_in_turn.get()

        await asyncio.gather(
            asyncio.create_task(task_a()),
            asyncio.create_task(task_b()),
        )

        assert results["a"] is True
        assert results["b"] is False

    @pytest.mark.asyncio
    async def test_message_id_isolation(self):
        """current_message_id is isolated per task."""
        results = {}

        async def task_a():
            current_message_id.set("msg_111")
            await asyncio.sleep(0.05)
            results["a"] = current_message_id.get()

        async def task_b():
            current_message_id.set("msg_222")
            await asyncio.sleep(0.05)
            results["b"] = current_message_id.get()

        await asyncio.gather(
            asyncio.create_task(task_a()),
            asyncio.create_task(task_b()),
        )

        assert results["a"] == "msg_111"
        assert results["b"] == "msg_222"


# ---------------------------------------------------------------------------
# MessageTool with ContextVar
# ---------------------------------------------------------------------------

class TestMessageToolContextVar:
    """MessageTool reads from ContextVar instead of instance attributes."""

    @pytest.mark.asyncio
    async def test_set_context_uses_contextvar(self):
        tool = MessageTool(send_callback=AsyncMock())
        tool.set_context("feishu", "chat_123", "msg_456")

        assert current_channel.get() == "feishu"
        assert current_chat_id.get() == "chat_123"
        assert current_message_id.get() == "msg_456"

    @pytest.mark.asyncio
    async def test_execute_reads_contextvar(self):
        callback = AsyncMock()
        tool = MessageTool(send_callback=callback)

        current_channel.set("telegram")
        current_chat_id.set("tg_user_1")

        result = await tool.execute(content="hello")
        assert "telegram:tg_user_1" in result
        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_explicit_channel_overrides_contextvar(self):
        callback = AsyncMock()
        tool = MessageTool(send_callback=callback)

        current_channel.set("feishu")
        current_chat_id.set("chat_default")

        result = await tool.execute(content="hello", channel="slack", chat_id="slack_ch")
        assert "slack:slack_ch" in result

    @pytest.mark.asyncio
    async def test_start_turn_resets_sent_flag(self):
        tool = MessageTool(send_callback=AsyncMock())
        message_sent_in_turn.set(True)
        tool.start_turn()
        assert tool.sent_in_turn is False

    @pytest.mark.asyncio
    async def test_sent_in_turn_property(self):
        callback = AsyncMock()
        tool = MessageTool(send_callback=callback)
        current_channel.set("feishu")
        current_chat_id.set("chat_1")

        tool.start_turn()
        assert tool.sent_in_turn is False

        await tool.execute(content="hi")
        assert tool.sent_in_turn is True


# ---------------------------------------------------------------------------
# SpawnTool with ContextVar
# ---------------------------------------------------------------------------

class TestSpawnToolContextVar:

    @pytest.mark.asyncio
    async def test_execute_reads_contextvar(self):
        manager = MagicMock()
        manager.spawn = AsyncMock(return_value="Spawned task")
        tool = SpawnTool(manager=manager)

        current_channel.set("discord")
        current_chat_id.set("dc_ch_1")

        result = await tool.execute(task="do something")
        assert result == "Spawned task"
        manager.spawn.assert_called_once_with(
            task="do something",
            label=None,
            origin_channel="discord",
            origin_chat_id="dc_ch_1",
        )

    @pytest.mark.asyncio
    async def test_defaults_when_contextvar_unset(self):
        manager = MagicMock()
        manager.spawn = AsyncMock(return_value="ok")
        tool = SpawnTool(manager=manager)

        # Reset contextvars
        current_channel.set("")
        current_chat_id.set("")

        await tool.execute(task="test")
        manager.spawn.assert_called_once_with(
            task="test",
            label=None,
            origin_channel="cli",
            origin_chat_id="direct",
        )


# ---------------------------------------------------------------------------
# CronTool with ContextVar
# ---------------------------------------------------------------------------

class TestCronToolContextVar:

    @pytest.mark.asyncio
    async def test_add_job_reads_contextvar(self):
        cron_service = MagicMock()
        job_mock = MagicMock()
        job_mock.name = "test"
        job_mock.id = "abc123"
        cron_service.add_job.return_value = job_mock
        tool = CronTool(cron_service=cron_service)

        current_channel.set("feishu")
        current_chat_id.set("feishu_chat")

        result = await tool.execute(
            action="add", message="remind me", every_seconds=60
        )
        assert "abc123" in result
        cron_service.add_job.assert_called_once()
        call_kwargs = cron_service.add_job.call_args
        assert call_kwargs.kwargs.get("channel") == "feishu" or call_kwargs[1].get("channel") == "feishu"

    @pytest.mark.asyncio
    async def test_add_job_fails_without_context(self):
        cron_service = MagicMock()
        tool = CronTool(cron_service=cron_service)

        current_channel.set("")
        current_chat_id.set("")

        result = await tool.execute(action="add", message="test", every_seconds=60)
        assert "Error" in result


# ---------------------------------------------------------------------------
# Concurrent dispatch simulation
# ---------------------------------------------------------------------------

class TestConcurrentDispatch:
    """Simulate concurrent message processing with session locking."""

    @pytest.mark.asyncio
    async def test_different_sessions_run_concurrently(self):
        """Messages from different sessions should overlap in time."""
        execution_log = []

        async def fake_process(session_key, delay):
            execution_log.append(f"{session_key}_start")
            await asyncio.sleep(delay)
            execution_log.append(f"{session_key}_end")

        sem = asyncio.Semaphore(5)
        locks: dict[str, asyncio.Lock] = {}

        async def dispatch(session_key, delay):
            lock = locks.setdefault(session_key, asyncio.Lock())
            async with sem:
                async with lock:
                    await fake_process(session_key, delay)

        await asyncio.gather(
            asyncio.create_task(dispatch("feishu:chat_a", 0.1)),
            asyncio.create_task(dispatch("slack:chat_b", 0.1)),
        )

        # Both should start before either ends (concurrent)
        assert execution_log.index("feishu:chat_a_start") < execution_log.index("slack:chat_b_end")
        assert execution_log.index("slack:chat_b_start") < execution_log.index("feishu:chat_a_end")

    @pytest.mark.asyncio
    async def test_same_session_runs_serially(self):
        """Messages from the same session should not overlap."""
        execution_log = []

        async def fake_process(label, delay):
            execution_log.append(f"{label}_start")
            await asyncio.sleep(delay)
            execution_log.append(f"{label}_end")

        lock = asyncio.Lock()
        sem = asyncio.Semaphore(5)

        async def dispatch(label, delay):
            async with sem:
                async with lock:
                    await fake_process(label, delay)

        await asyncio.gather(
            asyncio.create_task(dispatch("msg1", 0.1)),
            asyncio.create_task(dispatch("msg2", 0.1)),
        )

        # msg1 should end before msg2 starts (serial)
        assert execution_log.index("msg1_end") < execution_log.index("msg2_start")

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self):
        """Semaphore should cap the number of concurrent tasks."""
        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def tracked_task(i):
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                if current_concurrent > max_concurrent:
                    max_concurrent = current_concurrent
            await asyncio.sleep(0.05)
            async with lock:
                current_concurrent -= 1

        sem = asyncio.Semaphore(3)

        async def dispatch(i):
            async with sem:
                await tracked_task(i)

        await asyncio.gather(*[
            asyncio.create_task(dispatch(i)) for i in range(10)
        ])

        assert max_concurrent <= 3
