"""OpenAI 兼容 Provider 实现

支持所有 OpenAI 兼容 API，包括：
- OpenAI 官方
- Deepseek
- SiliconFlow
- vLLM
- Groq
- 本地 llama.cpp server
- 任何 OpenAI 兼容的第三方服务
"""

# GensokyoAI/core/agent/providers/openai_provider.py

from typing import AsyncIterator, TYPE_CHECKING

from .base import BaseProvider
from ..types import (
    UnifiedResponse,
    UnifiedMessage,
    UnifiedEmbeddingResponse,
    StreamChunk,
    ToolCall,
    ToolCallFunction,
)
from ....utils.logger import logger

if TYPE_CHECKING:
    from ...config import ModelConfig


class OpenAIProvider(BaseProvider):
    """
    OpenAI 兼容 Provider

    使用 openai SDK 调用所有兼容 OpenAI Chat Completions API 的服务。
    通过 base_url 配置可以指向任何兼容端点。
    """

    def __init__(self, config: "ModelConfig"):
        super().__init__(config)
        self._client = self._build_client()
        logger.debug(
            f"OpenAIProvider 初始化完成，base_url: {config.base_url}, model: {config.name}"
        )

    def _build_client(self):
        """构建 OpenAI 异步客户端"""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "使用 OpenAI Provider 需要安装 openai 包: pip install openai\n"
                "或者: pip install gensokyoai[openai]"
            )

        kwargs = {}
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url

        return AsyncOpenAI(**kwargs)

    async def chat(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        options: dict | None = None,
        **kwargs,
    ) -> UnifiedResponse:
        """非流式调用 OpenAI 兼容 API"""
        options = options or {}

        call_kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": options.get("temperature", 0.7),
            "top_p": options.get("top_p", 0.9),
        }

        # max_tokens 映射
        max_tokens = options.get("num_predict") or options.get("max_tokens")
        if max_tokens:
            call_kwargs["max_tokens"] = max_tokens

        # 工具支持
        if tools:
            call_kwargs["tools"] = self._convert_tools_to_openai(tools)

        response = await self._client.chat.completions.create(**call_kwargs)

        return self._convert_response(response)

    async def chat_stream(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        options: dict | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """流式调用 OpenAI 兼容 API"""
        options = options or {}

        call_kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": options.get("temperature", 0.7),
            "top_p": options.get("top_p", 0.9),
            "stream": True,
        }

        max_tokens = options.get("num_predict") or options.get("max_tokens")
        if max_tokens:
            call_kwargs["max_tokens"] = max_tokens

        if tools:
            call_kwargs["tools"] = self._convert_tools_to_openai(tools)

        # 流式工具调用累积器
        tool_calls_acc: dict[int, dict] = {}

        stream = await self._client.chat.completions.create(**call_kwargs)

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue

            # 处理工具调用
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "name": "",
                            "arguments": "",
                        }
                    if tc.function:
                        if tc.function.name:
                            tool_calls_acc[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls_acc[idx]["arguments"] += tc.function.arguments

            # 处理内容
            elif delta.content:
                yield StreamChunk(content=delta.content)

            # 检查结束
            finish_reason = chunk.choices[0].finish_reason if chunk.choices else None
            if finish_reason == "tool_calls" and tool_calls_acc:
                import json

                unified_tool_calls = []
                for _idx, tc_data in sorted(tool_calls_acc.items()):
                    try:
                        args = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
                    except json.JSONDecodeError:
                        args = {}
                    unified_tool_calls.append(
                        ToolCall(
                            function=ToolCallFunction(
                                name=tc_data["name"],
                                arguments=args,
                            )
                        )
                    )

                unified_msg = UnifiedMessage(
                    role="assistant",
                    content="",
                    tool_calls=unified_tool_calls,
                )
                yield StreamChunk(
                    is_tool_call=True,
                    tool_info={"message": unified_msg},
                )

    async def embeddings(
        self,
        model: str,
        prompt: str,
        **kwargs,
    ) -> UnifiedEmbeddingResponse:
        """获取文本向量"""
        response = await self._client.embeddings.create(
            model=model,
            input=prompt,
        )

        return UnifiedEmbeddingResponse(
            embedding=response.data[0].embedding,
            model=model,
        )

    def update_config(self, config: "ModelConfig") -> None:
        """更新配置并重建客户端"""
        super().update_config(config)
        self._client = self._build_client()
        logger.info(f"OpenAIProvider 配置已更新，base_url: {config.base_url}")

    # ==================== 转换工具方法 ====================

    def _convert_response(self, response) -> UnifiedResponse:
        """将 OpenAI ChatCompletion 转换为 UnifiedResponse"""
        choice = response.choices[0] if response.choices else None
        if not choice:
            return UnifiedResponse(model=response.model or "")

        message = choice.message
        tool_calls = None

        if message.tool_calls:
            import json

            tool_calls = []
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(
                    ToolCall(
                        function=ToolCallFunction(
                            name=tc.function.name or "",
                            arguments=args,
                        )
                    )
                )

        thinking = None
        # 部分兼容 API（如 Deepseek）可能有 reasoning_content
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            thinking = message.reasoning_content

        return UnifiedResponse(
            message=UnifiedMessage(
                role=message.role or "assistant",
                content=message.content or "",
                tool_calls=tool_calls,
            ),
            model=response.model or "",
            done=True,
            thinking=thinking,
        )

    @staticmethod
    def _convert_tools_to_openai(tools: list[dict]) -> list[dict]:
        """
        将 Ollama 格式的工具定义转换为 OpenAI 格式

        Ollama 格式:
          {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}

        OpenAI 格式基本一致，但某些字段可能有差异
        """
        openai_tools = []
        for tool in tools:
            # 如果已经是 OpenAI 格式，直接使用
            if "type" in tool and "function" in tool:
                openai_tools.append(tool)
            else:
                # 尝试适配
                openai_tools.append({"type": "function", "function": tool})
        return openai_tools
