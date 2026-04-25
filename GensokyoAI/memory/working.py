"""工作记忆"""

# GensokyoAI\memory\working.py

from typing import Any
from .types import WorkingMemory


class WorkingMemoryManager:
    """工作记忆管理器"""

    def __init__(self, max_turns: int = 20):
        self._memory = WorkingMemory(max_turns=max_turns)

    def add_message(self, role: str, content: str, tool_calls=None, tool_call_id=None):
        msg: dict = {"role": role, "content": content}
        
        if tool_calls:
            msg["tool_calls"] = [
                tc.to_dict() if hasattr(tc, 'to_dict') else tc 
                for tc in tool_calls
            ]
        
        if tool_call_id:
            msg["tool_call_id"] = tool_call_id
        
        self._memory.messages.append(msg)
        
    def get_context(self) -> list[dict[str, Any]]:
        """获取当前上下文"""
        return self._memory.get_context()

    def get_recent(self, n: int) -> list[dict[str, Any]]:
        """获取最近 n 条消息"""
        return self._memory.messages[-n:] if n > 0 else []

    def clear(self) -> None:
        """清空工作记忆"""
        self._memory.clear()

    def __len__(self) -> int:
        return len(self._memory.messages)
