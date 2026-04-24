"""Provider 工厂 + 注册表"""

# GensokyoAI/core/agent/providers/__init__.py

from typing import TYPE_CHECKING

from .base import BaseProvider
from ....utils.logger import logger

if TYPE_CHECKING:
    from ...config import ModelConfig


class ProviderFactory:
    """
    Provider 工厂 - 根据配置创建对应的 LLM Provider

    支持：
    - 内置 Provider 自动注册
    - 用户自定义 Provider 动态注册

    用法：
        # 使用内置 Provider
        provider = ProviderFactory.create(config)

        # 注册自定义 Provider
        ProviderFactory.register("my_provider", MyProvider)
        provider = ProviderFactory.create(config)  # config.provider = "my_provider"
    """

    _registry: dict[str, type[BaseProvider]] = {}
    _initialized: bool = False

    @classmethod
    def _ensure_builtins(cls) -> None:
        """延迟注册内置 Provider（避免循环导入）"""
        if cls._initialized:
            return

        cls._initialized = True

        # Ollama - 始终注册
        from .ollama_provider import OllamaProvider

        cls._registry["ollama"] = OllamaProvider

        # OpenAI - 尝试注册
        try:
            from .openai_provider import OpenAIProvider

            cls._registry["openai"] = OpenAIProvider
        except ImportError:
            pass

        # Claude - 尝试注册
        try:
            from .claude_provider import ClaudeProvider

            cls._registry["claude"] = ClaudeProvider
        except ImportError:
            pass

        # Gemini - 尝试注册
        try:
            from .gemini_provider import GeminiProvider

            cls._registry["gemini"] = GeminiProvider
        except ImportError:
            pass

    @classmethod
    def register(cls, name: str, provider_cls: type[BaseProvider]) -> None:
        """
        注册自定义 Provider

        Args:
            name: Provider 名称（用于配置文件中的 provider 字段）
            provider_cls: Provider 类（必须继承 BaseProvider）

        Example:
            ProviderFactory.register("my_llm", MyLLMProvider)
        """
        if not issubclass(provider_cls, BaseProvider):
            raise TypeError(f"{provider_cls.__name__} 必须继承 BaseProvider")

        cls._registry[name] = provider_cls
        logger.info(f"注册 Provider: {name} -> {provider_cls.__name__}")

    @classmethod
    def create(cls, config: "ModelConfig", **kwargs) -> BaseProvider:
        """
        根据配置创建 Provider 实例

        Args:
            config: 模型配置
            **kwargs: 额外参数传递给 Provider 构造函数

        Returns:
            BaseProvider: Provider 实例

        Raises:
            ValueError: 未知的 Provider 类型
            ImportError: 对应 SDK 未安装
        """
        cls._ensure_builtins()

        provider_name = config.provider
        provider_cls = cls._registry.get(provider_name)

        if not provider_cls:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"未知的 Provider: '{provider_name}'\n"
                f"可用的 Provider: {available}\n"
                f"请检查配置中的 model.provider 字段"
            )

        logger.info(f"创建 Provider: {provider_name} -> {provider_cls.__name__}")
        return provider_cls(config, **kwargs)

    @classmethod
    def available_providers(cls) -> list[str]:
        """获取所有可用的 Provider 名称"""
        cls._ensure_builtins()
        return list(cls._registry.keys())


__all__ = ["BaseProvider", "ProviderFactory"]
