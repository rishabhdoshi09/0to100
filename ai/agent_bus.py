"""Thread-safe message bus for JARVIS multi-agent communication."""
from __future__ import annotations

import queue
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class AgentMessage:
    from_agent: str
    to_agent: str       # "ORCHESTRATOR" | "ALL" | specific agent name
    msg_type: str       # "STATUS" | "RESULT" | "LOG" | "REQUEST"
    content: str
    data: Any = None
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))
    task_id: str = ""


class AgentBus:
    """Publish/consume hub for all JARVIS agents."""

    def __init__(self) -> None:
        self._q: queue.Queue[AgentMessage] = queue.Queue()
        self._log: list[AgentMessage] = []

    def publish(self, msg: AgentMessage) -> None:
        self._q.put(msg)
        self._log.append(msg)

    def drain(self) -> list[AgentMessage]:
        msgs: list[AgentMessage] = []
        try:
            while True:
                msgs.append(self._q.get_nowait())
        except queue.Empty:
            pass
        return msgs

    def get_log(self) -> list[AgentMessage]:
        return list(self._log)

    def clear(self) -> None:
        self._log.clear()
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
