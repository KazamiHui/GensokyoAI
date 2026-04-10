"""配置管理"""

import os
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field
from enum import Enum
import yaml

from ..utils.logging import setup_logging, logger


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class ModelConfig:
    """模型配置"""

    name: str = "qwen3:14b"
    base_url: str | None = None
    stream: bool = True
    think: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    timeout: int = 60


@dataclass
class MemoryConfig:
    """记忆配置"""

    working_max_turns: int = 20
    episodic_threshold: int = 50
    episodic_summary_model: str = "qwen3:14b"
    episodic_keep_recent: int = 10
    semantic_enabled: bool = True
    semantic_embedding_model: str = "nomic-embed-text"
    semantic_top_k: int = 5
    semantic_similarity_threshold: float = 0.7
    auto_memory_enabled: bool = True
    auto_memory_model: str = "qwen3:14b"


@dataclass
class ToolConfig:
    """工具配置"""

    enabled: bool = True
    builtin_tools: list[str] = field(
        default_factory=lambda: ["time", "moon", "memory", "system"]
    )
    custom_tools_path: Path | None = None


@dataclass
class SessionConfig:
    """会话配置"""

    auto_save: bool = True
    save_path: Path = Path("./sessions")
    max_sessions: int = 100

    def __post_init__(self):
        if isinstance(self.save_path, str):
            self.save_path = Path(self.save_path)


@dataclass
class CharacterConfig:
    """角色配置"""

    name: str
    system_prompt: str
    greeting: str = ""
    example_dialogue: list[dict[str, str]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AppConfig:
    """应用配置"""

    # 日志配置
    log_level: LogLevel = LogLevel.INFO
    log_console: bool = True
    log_file: Path | None = None

    # 子配置
    model: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    tool: ToolConfig = field(default_factory=ToolConfig)
    session: SessionConfig = field(default_factory=SessionConfig)

    # 角色
    character: CharacterConfig | None = None
    character_file: Path | None = None

    def __post_init__(self):
        self.session.save_path.mkdir(parents=True, exist_ok=True)

        # 应用日志配置
        self._apply_logging_config()

    def _apply_logging_config(self) -> None:
        """应用日志配置"""
        setup_logging(
            log_level=self.log_level.value,
            log_console=self.log_console,
            log_file=self.log_file,
        )


class ConfigLoader:
    """配置加载器"""

    def __init__(self):
        self._config: AppConfig | None = None

    def load(self, config_file: Path | None = None) -> AppConfig:
        """加载配置"""
        config = AppConfig()

        # 1. 加载默认配置
        default_file = Path(__file__).parent / "default.yaml"
        if default_file.exists():
            config = self._load_yaml(default_file)

        # 2. 加载用户配置文件
        if config_file and config_file.exists():
            user_config = self._load_yaml(config_file)
            config = self._merge(config, user_config)

        # 3. 环境变量覆盖
        config = self._apply_env(config)

        # 4. 重新应用日志配置（确保使用最终配置）
        config._apply_logging_config()

        self._config = config
        return config

    def _load_yaml(self, path: Path) -> AppConfig:
        """从 YAML 加载配置"""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return self._dict_to_config(data)

    def _dict_to_config(self, data: dict[str, Any]) -> AppConfig:
        """字典转配置对象"""
        config = AppConfig()

        if "log_level" in data:
            config.log_level = LogLevel(data["log_level"])
        if "log_console" in data:
            config.log_console = data["log_console"]
        if "log_file" in data and data["log_file"]:
            config.log_file = Path(data["log_file"])

        if "model" in data:
            config.model = ModelConfig(**data["model"])
        if "memory" in data:
            config.memory = MemoryConfig(**data["memory"])
        if "tool" in data:
            config.tool = ToolConfig(**data["tool"])
        if "session" in data:
            config.session = SessionConfig(**data["session"])

        return config

    def _merge(self, base: AppConfig, override: AppConfig) -> AppConfig:
        """合并配置"""
        result = AppConfig()

        # 日志配置
        result.log_level = (
            override.log_level
            if override.log_level != LogLevel.INFO
            else base.log_level
        )
        result.log_console = (
            override.log_console if not override.log_console else base.log_console
        )
        result.log_file = override.log_file or base.log_file

        # 其他配置
        result.model = override.model or base.model
        result.memory = override.memory or base.memory
        result.tool = override.tool or base.tool
        result.session = override.session or base.session
        result.character = override.character or base.character
        result.character_file = override.character_file or base.character_file

        return result

    def _apply_env(self, config: AppConfig) -> AppConfig:
        """应用环境变量"""
        if os.getenv("GENSKOYAI_MODEL"):
            config.model.name = os.getenv("GENSKOYAI_MODEL")  # type: ignore
        if os.getenv("GENSKOYAI_LOG_LEVEL"):
            config.log_level = LogLevel(os.getenv("GENSKOYAI_LOG_LEVEL"))
        if os.getenv("GENSKOYAI_LOG_CONSOLE"):
            config.log_console = os.getenv("GENSKOYAI_LOG_CONSOLE").lower() == "true"  # type: ignore
        if os.getenv("GENSKOYAI_MEMORY_WORKING_TURNS"):
            config.memory.working_max_turns = int(
                os.getenv("GENSKOYAI_MEMORY_WORKING_TURNS")
            )  # type: ignore
        return config

    def load_character(self, path: Path) -> CharacterConfig:
        """加载角色配置"""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return CharacterConfig(
            name=data["name"],
            system_prompt=data["system_prompt"],
            greeting=data.get("greeting", ""),
            example_dialogue=data.get("example_dialogue"),
            metadata=data.get("metadata", {}),
        )
