# api/li_memory.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from threading import RLock
from typing import Dict
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore

class _SessionMemory:

    def __init__(self, session_id: str, max_turns: int = 100):
        self.session_id = session_id
        self._turns: List[Tuple[str, str]] = []
        self._lock = RLock()
        self._max_turns = max_turns

    def put_user(self, content: str) -> None:
        self._append(("user", content))

    def append_user(self, content: str) -> None:
        self._append(("user", content))

    def put_assistant(self, content: str) -> None:
        self._append(("assistant", content))

    def append_assistant(self, content: str) -> None:
        self._append(("assistant", content))

    def put(self, msg) -> None:
        """
        Generic fallback: accepts dicts like {"role": "user", "content": "..."}.
        """
        if isinstance(msg, dict):
            role = str(msg.get("role", "")).strip().lower() or "user"
            content = str(msg.get("content", "") or "")
            self._append((role, content))
        elif isinstance(msg, tuple) and len(msg) == 2:
            role, content = msg
            self._append((str(role), str(content)))
        else:

            pass


    def get_turns(self) -> List[Tuple[str, str]]:
        with self._lock:
            return list(self._turns)

    def render(self, max_turns: int = 6, max_chars: int = 1200) -> str:

        with self._lock:
            recent = self._turns[-max_turns:]
        lines: List[str] = []
        for role, content in recent:
            role_up = "USER" if role.lower() == "user" else "ASSISTANT"
            lines.append(f"{role_up}: {content}")
        text = "\n".join(lines).strip()
        return text[:max_chars]


    def _append(self, item: Tuple[str, str]) -> None:
        role, content = item
        role = (role or "").strip().lower() or "user"
        content = (content or "").strip()
        if not content:
            return
        with self._lock:
            self._turns.append((role, content))

            if len(self._turns) > self._max_turns:
                overflow = len(self._turns) - self._max_turns
                del self._turns[0:overflow]

_chat_store = SimpleChatStore()
_memories: Dict[str, ChatMemoryBuffer] = {}

def get_memory(session_id: str) -> ChatMemoryBuffer:
    if session_id not in _memories:
        _memories[session_id] = ChatMemoryBuffer.from_defaults(
            chat_store=_chat_store,
            chat_store_key=session_id,
            token_limit=1500,
        )
    return _memories[session_id]
