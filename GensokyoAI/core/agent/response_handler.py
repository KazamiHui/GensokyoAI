"""响应处理器 - 处理模型响应和工具调用"""

# GensokyoAI/core/agent/response_handler.py

from typing import AsyncIterator, TYPE_CHECKING

from ollama import Message, ChatResponse

from .model_client import StreamChunk
from ...utils.logger import logger
from ...utils.helpers import safe_get
from ...memory.types import MemoryRecord

if TYPE_CHECKING:
    from ...memory.working import WorkingMemoryManager
    from ...memory.episodic import EpisodicMemoryManager
    from ...tools.executor import ToolExecutor
    from ...core.config import AppConfig
    from .model_client import ModelClient
    from .message_builder import MessageBuilder
    from .save_coordinator import SaveCoordinator


class ResponseHandler:
    """
    响应处理器 - 纯响应生成，不操作工作记忆

    魔理沙：只管说话，记东西交给别人DA☆ZE！
    """

    def __init__(
        self,
        config: "AppConfig",
        working_memory: "WorkingMemoryManager",
        episodic_memory: "EpisodicMemoryManager",
        tool_executor: "ToolExecutor",
        model_client: "ModelClient",
        message_builder: "MessageBuilder",
        save_coordinator: "SaveCoordinator",
    ):
        self._config = config
        self._working_memory = working_memory
        self._episodic_memory = episodic_memory
        self._tool_executor = tool_executor
        self._model_client = model_client
        self._message_builder = message_builder
        self._save_coordinator = save_coordinator
        self._shutting_down = False

    def set_shutting_down(self, value: bool) -> None:
        self._shutting_down = value

    @property
    def character_name(self) -> str:
        return safe_get(self._config, "character.name", "default")

    # ==================== 工具处理 ====================

    async def _handle_tool_calls(self, message: Message) -> list[dict] | None:
        if not message.tool_calls:
            return None
        logger.info(f"检测到 {len(message.tool_calls)} 个工具调用")
        if parsed := self._tool_executor.parse_tool_calls(message):
            return await self._tool_executor.execute_batch(parsed)
        return None

    async def _record_tool_results(self, results: list[dict]) -> None:
        for r in results:
            await self._episodic_memory.add_message(
                MemoryRecord(
                    content=r["content"],
                    role="tool",
                    character_id=self.character_name,
                    metadata={"tool_name": r.get("name", "")},
                )
            )

    # ==================== 响应处理 ====================

    async def process_stream(
        self, messages: list[dict[str, str]], tools: list[dict] | None
    ) -> AsyncIterator[StreamChunk]:
        """处理流式响应"""
        tool_calls_message: Message | None = None

        async for chunk in self._model_client.chat_stream(messages, tools):
            if self._shutting_down:
                break
            if chunk.is_tool_call and chunk.tool_info:
                tool_calls_message = chunk.tool_info["message"]
            else:
                yield chunk

        if self._shutting_down:
            return

        if tool_calls_message:
            if tool_results := await self._handle_tool_calls(tool_calls_message):
                await self._record_tool_results(tool_results)
                cont_messages = self._message_builder.build_continuation()
                async for chunk in self._model_client.chat_stream(cont_messages, None):
                    if self._shutting_down:
                        break
                    yield chunk
