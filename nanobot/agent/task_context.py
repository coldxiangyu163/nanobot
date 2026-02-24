"""Per-task context variables for concurrent session processing.

Uses Python contextvars so each asyncio.create_task() gets an isolated copy.
Tools read from these instead of mutable instance attributes, making them
safe to use across concurrent sessions.
"""

from contextvars import ContextVar

# Routing context — set by _set_tool_context before each _process_message
current_channel: ContextVar[str] = ContextVar("current_channel", default="")
current_chat_id: ContextVar[str] = ContextVar("current_chat_id", default="")
current_message_id: ContextVar[str | None] = ContextVar("current_message_id", default=None)

# Per-turn state for MessageTool
message_sent_in_turn: ContextVar[bool] = ContextVar("message_sent_in_turn", default=False)
