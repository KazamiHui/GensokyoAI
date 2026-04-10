"""系统工具"""

import platform
from ..base import tool
from ..registry import ToolRegistry


@tool(description="获取系统信息")
def get_system_info() -> str:
    """获取操作系统和硬件信息"""
    return f"OS: {platform.system()} {platform.release()}, Python: {platform.python_version()}"


@tool(description="列出当前会话中可用的工具")
def list_available_tools() -> str:
    """列出所有已注册的工具"""
    registry = ToolRegistry()
    tools = registry.list()
    tool_names = [t.name for t in tools]
    return f"可用工具: {', '.join(tool_names)}"
