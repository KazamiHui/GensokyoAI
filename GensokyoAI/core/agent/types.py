"""统一类型系统 - 消除 ollama 类型耦合"""

# GensokyoAI/core/agent/types.py

from msgspec import Struct, field


class ToolCallFunction(Struct):
    """工具调用函数信息"""

    name: str = ""
    arguments: dict = field(default_factory=dict)


class ToolCall(Struct):
    """工具调用"""

    id: str = ""  # OpenAI tool call ID（如 call_abc123），用于关联 tool 结果消息
    function: ToolCallFunction = field(default_factory=ToolCallFunction)


class UnifiedMessage(Struct):
    """
    统一消息类型 - 替代 ollama.Message

    所有 Provider 返回的消息都转换为此类型
    """

    role: str = "assistant"
    content: str = ""
    tool_calls: list[ToolCall] | None = None


class UnifiedResponse(Struct):
    """
    统一响应类型 - 替代 ollama.ChatResponse

    所有 Provider 的非流式响应都转换为此类型
    """

    message: UnifiedMessage = field(default_factory=UnifiedMessage)
    model: str = ""
    done: bool = True
    thinking: str | None = None


class UnifiedEmbeddingResponse(Struct):
    """
    统一 Embedding 响应类型 - 替代 ollama.EmbeddingsResponse
    """

    embedding: list[float] = field(default_factory=list)
    model: str = ""


class StreamChunk(Struct):
    """
    流式响应块

    替代原来在 model_client.py 中定义的 StreamChunk
    """

    content: str = ""
    is_tool_call: bool = False
    tool_info: dict | None = None
