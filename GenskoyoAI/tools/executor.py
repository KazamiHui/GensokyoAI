"""工具执行器"""

import json
import asyncio
from typing import Optional

from ollama import Message

from .registry import ToolRegistry
from ..utils.logging import logger


class ToolExecutor:
    """工具执行器"""

    def __init__(self, registry: Optional[ToolRegistry] = None):
        self._registry = registry or ToolRegistry()

    def parse_tool_calls(self, message: dict) -> list[dict]:
        """从模型响应中解析工具调用"""
        tool_calls = message.get("tool_calls", [])
        if not tool_calls:
            return []

        parsed = []
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")
            args_str = func.get("arguments", "{}")

            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                args = {}

            parsed.append({"id": tc.get("id", ""), "name": name, "arguments": args})

        return parsed

    def parse_tool_calls_from_message(self, message: Message) -> list[dict]:
        """从 Message 对象中解析工具调用"""
        if not message.tool_calls:
            return []

        parsed = []
        for tc in message.tool_calls:
            # 处理 ollama Message 对象
            if tc.function:
                name = tc.function.name
                args = tc.function.arguments
            elif isinstance(tc, dict):
                func = tc.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments", {})
            else:
                continue

            # 如果 args 是字符串，尝试解析为 JSON
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            parsed.append(
                {
                    "id": getattr(tc, "id", "")
                    if hasattr(tc, "id")
                    else tc.get("id", ""),
                    "name": name,
                    "arguments": args,
                }
            )

        return parsed

    async def execute(self, tool_call: dict) -> dict:
        """执行单个工具调用"""
        name = tool_call.get("name")
        arguments = tool_call.get("arguments", {})
        call_id = tool_call.get("id", "")

        tool_def = self._registry.get(name)  # type: ignore
        if not tool_def:
            error_msg = f"工具 '{name}' 未找到"
            logger.warning(error_msg)
            return {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": f"错误: {error_msg}",
            }

        try:
            logger.debug(f"执行工具: {name}({arguments})")

            if tool_def.is_async:
                result = await tool_def.func(**arguments)
            else:
                # 同步函数在线程池中执行
                result = await asyncio.to_thread(tool_def.func, **arguments)

            # 转换结果为字符串
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False)

            logger.info(f"工具 {name} 执行成功: {result[:100]}...")

            return {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": result,
            }
        except Exception as e:
            error_msg = f"工具执行失败: {e}"
            logger.error(f"工具 {name} 执行错误: {e}")
            return {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": f"错误: {error_msg}",
            }

    async def execute_batch(self, tool_calls: list[dict]) -> list[dict]:
        """批量执行工具调用（并行）"""
        tasks = [self.execute(tc) for tc in tool_calls]
        results = await asyncio.gather(*tasks)
        return results

    def execute_sync(self, tool_call: dict) -> dict:
        """同步执行（兼容非异步环境）"""
        name = tool_call.get("name")
        arguments = tool_call.get("arguments", {})
        call_id = tool_call.get("id", "")

        tool_def = self._registry.get(name)  # type: ignore
        if not tool_def:
            return {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": f"错误: 工具 '{name}' 未找到",
            }

        try:
            result = tool_def.func(**arguments)
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False)
            return {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": result,
            }
        except Exception as e:
            return {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": f"错误: {e}",
            }
