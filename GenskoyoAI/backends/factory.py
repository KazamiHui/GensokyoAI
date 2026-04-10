"""后端工厂"""

from typing import Literal
from .base import BaseBackend
from .console import ConsoleBackend
from ..core.agent import Agent


BackendType = Literal["console"]


class BackendFactory:
    """后端工厂"""

    @staticmethod
    def create(backend_type: BackendType, agent: Agent) -> BaseBackend:
        """创建后端"""
        if backend_type == "console":
            return ConsoleBackend(agent)
        raise ValueError(f"Unknown backend type: {backend_type}")
