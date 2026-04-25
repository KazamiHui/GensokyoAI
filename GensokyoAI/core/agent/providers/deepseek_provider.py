"""DeepSeek Provider 实现

DeepSeek 虽然兼容 OpenAI Chat Completions API，但 thinking mode 对
reasoning_content 的多轮回传有额外要求，尤其是思考模式下发生工具调用后，
后续请求必须保留 assistant 消息中的 reasoning_content。
"""

# GensokyoAI/core/agent/providers/deepseek_provider.py

from typing import AsyncIterator, TYPE_CHECKING

from .openai_provider import OpenAIProvider
from ..types import (
    UnifiedResponse,
    UnifiedMessage,
    StreamChunk,
    ToolCall,
    ToolCallFunction,
)
from ....utils.logger import logger

if TYPE_CHECKING:
    from ...config import ModelConfig


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek 专用 Provider

    - 默认 base_url 使用 https://api.deepseek.com
    - 默认启用 thinking mode
    - 默认 reasoning_effort 为 high
    - 支持流式/非流式捕获并回传 reasoning_content
    """

    DEFAULT_BASE_URL = "https://api.deepseek.com"
    DEFAULT_REASONING_EFFORT = "high"

    def _build_client(self):
        """构建 DeepSeek OpenAI 兼容客户端"""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "使用 DeepSeek Provider 需要安装 openai 包: pip install openai\n"
                "或者: pip install gensokyoai[openai]"
            )

        kwargs = {}
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        kwargs["base_url"] = self.config.base_url or self.DEFAULT_BASE_URL
        return AsyncOpenAI(**kwargs)

    @property
    def _thinking_enabled(self) -> bool:
        if self.config.thinking_enabled is not None:
            return self.config.thinking_enabled
        if self.config.think:
            return True
        return True

    @property
    def _reasoning_effort(self) -> str:
        return self.config.reasoning_effort or self.DEFAULT_REASONING_EFFORT

    def _apply_deepseek_options(self, call_kwargs: dict) -> None:
        """应用 DeepSeek thinking 参数。"""
        if self._thinking_enabled:
            call_kwargs["reasoning_effort"] = self._reasoning_effort
            call_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        else:
            call_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    @staticmethod
    def _prepare_messages(messages: list[dict]) -> list[dict]:
        """复制消息并保留 DeepSeek 支持的 reasoning_content 字段。"""
        prepared: list[dict] = []
        for message in messages:
            copied = dict(message)
            # DeepSeek 要求 tool 调用相关 assistant 消息完整回传 reasoning_content。
            # 非 DeepSeek Provider 不会使用此类，因此不会影响其他 OpenAI-compatible 服务商。
            prepared.append(copied)
        return prepared

    @staticmethod
    def _make_tool_call(tc_data: dict) -> ToolCall:
        return ToolCall(
            id=tc_data.get("id", ""),
            provider="deepseek",
            function=ToolCallFunction(
                name=tc_data.get("name", ""),
                arguments=tc_data.get("arguments", {}),
                provider="deepseek",
            ),
        )

    async def chat(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        options: dict | None = None,
        **kwargs,
    ) -> UnifiedResponse:
        """非流式调用 DeepSeek API。"""
        options = options or {}
        call_kwargs: dict = {
            "model": model,
            "messages": self._prepare_messages(messages),
        }

        max_tokens = (
            options.get("max_completion_tokens")
            or options.get("num_predict")
            or options.get("max_tokens")
        )
        if max_tokens:
            call_kwargs["max_tokens"] = max_tokens

        if not self._thinking_enabled:
            call_kwargs["temperature"] = options.get("temperature", 0.7)
            call_kwargs["top_p"] = options.get("top_p", 0.9)

        self._apply_deepseek_options(call_kwargs)

        if tools:
            call_kwargs["tools"] = self._convert_tools_to_openai(tools)
            if tool_choice := options.get("tool_choice"):
                call_kwargs["tool_choice"] = tool_choice

        response = await self._client.chat.completions.create(**call_kwargs)
        return self._convert_response(response)

    async def chat_stream(  # type: ignore
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        options: dict | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """流式调用 DeepSeek API，捕获 reasoning_content 和 tool_calls。"""
        options = options or {}
        call_kwargs: dict = {
            "model": model,
            "messages": self._prepare_messages(messages),
            "stream": True,
        }

        max_tokens = (
            options.get("max_completion_tokens")
            or options.get("num_predict")
            or options.get("max_tokens")
        )
        if max_tokens:
            call_kwargs["max_tokens"] = max_tokens

        if not self._thinking_enabled:
            call_kwargs["temperature"] = options.get("temperature", 0.7)
            call_kwargs["top_p"] = options.get("top_p", 0.9)

        self._apply_deepseek_options(call_kwargs)

        if tools:
            call_kwargs["tools"] = self._convert_tools_to_openai(tools)
            if tool_choice := options.get("tool_choice"):
                call_kwargs["tool_choice"] = tool_choice

        tool_calls_acc: dict[int, dict] = {}
        content_acc = ""
        reasoning_acc = ""

        stream = await self._client.chat.completions.create(**call_kwargs)

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue

            reasoning_delta = getattr(delta, "reasoning_content", None)
            if reasoning_delta:
                reasoning_acc += reasoning_delta
                yield StreamChunk(reasoning_content=reasoning_delta)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "id": "",
                            "name": "",
                            "arguments": "",
                        }
                    if tc.id:
                        tool_calls_acc[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls_acc[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls_acc[idx]["arguments"] += tc.function.arguments

            if delta.content:
                content_acc += delta.content
                yield StreamChunk(content=delta.content)

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
                        self._make_tool_call(
                            {
                                "id": tc_data.get("id", ""),
                                "name": tc_data.get("name", ""),
                                "arguments": args,
                            }
                        )
                    )

                unified_msg = UnifiedMessage(
                    role="assistant",
                    content=content_acc,
                    tool_calls=unified_tool_calls,
                    reasoning_content=reasoning_acc or None,
                )
                yield StreamChunk(
                    is_tool_call=True,
                    tool_info={"message": unified_msg},
                )

    def _convert_response(self, response) -> UnifiedResponse:
        """将 DeepSeek ChatCompletion 转换为 UnifiedResponse。"""
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
                        id=tc.id or "",
                        provider="deepseek",
                        function=ToolCallFunction(
                            name=tc.function.name or "",
                            arguments=args,
                            provider="deepseek",
                        ),
                    )
                )

        reasoning_content = getattr(message, "reasoning_content", None) or None

        return UnifiedResponse(
            message=UnifiedMessage(
                role=message.role or "assistant",
                content=message.content or "",
                tool_calls=tool_calls,
                reasoning_content=reasoning_content,
            ),
            model=response.model or "",
            done=True,
            thinking=reasoning_content,
        )

    def update_config(self, config: "ModelConfig") -> None:
        """更新配置并重建客户端。"""
        super().update_config(config)
        logger.info(
            f"DeepSeekProvider 配置已更新，base_url: {config.base_url or self.DEFAULT_BASE_URL}"
        )
