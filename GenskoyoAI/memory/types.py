"""记忆数据类"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Any
from uuid import uuid4, UUID


@dataclass
class MemoryRecord:
    """基础记忆记录"""

    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    role: str = "user"  # user/assistant/system/tool
    timestamp: datetime = field(default_factory=datetime.now)
    character_id: str = "default"
    importance: float = 0.0  # 0-1 重要程度
    metadata: dict = field(default_factory=dict)


@dataclass
class WorkingMemory:
    """工作记忆 - 当前会话的完整对话"""

    messages: List[dict] = field(default_factory=list)
    max_turns: int = 20

    def add(self, role: str, content: str, **kwargs) -> None:
        """添加消息"""
        self.messages.append({"role": role, "content": content, **kwargs})
        self._trim()

    def _trim(self) -> None:
        """裁剪到最大轮数"""
        if len(self.messages) > self.max_turns * 2:  # *2 因为 user+assistant
            self.messages = self.messages[-self.max_turns * 2 :]

    def get_context(self) -> List[dict]:
        """获取上下文"""
        return self.messages.copy()

    def clear(self) -> None:
        """清空"""
        self.messages.clear()


@dataclass
class EpisodicMemory:
    """情景记忆 - 历史摘要"""

    summary: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    message_count: int = 0
    key_events: List[str] = field(default_factory=list)


@dataclass
class SemanticMemory:
    """语义记忆 - 向量化的知识片段"""

    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    embedding: Optional[List[float]] = None
    importance: float = 0.0
    tags: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
