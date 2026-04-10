"""会话持久化"""

import json
from pathlib import Path

from .context import SessionContext
from ..utils.logging import logger


class SessionPersistence:
    """会话持久化"""

    def __init__(self, base_path: Path):
        logger.debug(
            f"value the base_path is: {base_path}, type: {type(base_path).__name__}"
        )
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_session_path(self, character_id: str, session_id: str) -> Path:
        """获取会话文件路径"""
        char_path = self.base_path / character_id
        char_path.mkdir(parents=True, exist_ok=True)
        return char_path / f"{session_id}.json"

    def save_session(self, session: SessionContext) -> None:
        """保存会话"""
        path = self._get_session_path(session.character_id, session.session_id)

        # 如果文件已存在，加载现有消息
        existing_messages = []
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                existing_messages = data.get("messages", [])

        data = {"session": session.to_dict(), "messages": existing_messages}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.debug(f"会话已保存: {path}")

    def save_messages(self, session_id: str, messages: list[dict]) -> None:
        """保存消息"""
        saved = False
        # 查找会话文件
        for char_dir in self.base_path.iterdir():
            if char_dir.is_dir():
                session_file = char_dir / f"{session_id}.json"
                if session_file.exists():
                    with open(session_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    data["messages"] = messages
                    # 同时更新会话的 total_turns
                    if "session" in data:
                        data["session"]["total_turns"] = len(messages) // 2
                    with open(session_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    saved = True
                    logger.debug(f"消息已保存: {session_id}, {len(messages)} 条")
                    break

        if not saved:
            logger.warning(f"未找到会话文件: {session_id}")

    def load_messages(self, session_id: str) -> list[dict]:
        """加载消息"""
        for char_dir in self.base_path.iterdir():
            if char_dir.is_dir():
                session_file = char_dir / f"{session_id}.json"
                if session_file.exists():
                    with open(session_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    messages = data.get("messages", [])
                    logger.debug(f"加载消息: {session_id}, {len(messages)} 条")
                    return messages
        return []

    def load_session(self, character_id: str, session_id: str) -> SessionContext | None:
        """加载会话"""
        path = self._get_session_path(character_id, session_id)
        if not path.exists():
            return None

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return SessionContext.from_dict(data["session"])

    def list_sessions(self, character_id: str) -> list[SessionContext]:
        """列出所有会话"""
        sessions = []
        char_path = self.base_path / character_id
        if char_path.exists():
            for file in char_path.glob("*.json"):
                try:
                    with open(file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    sessions.append(SessionContext.from_dict(data["session"]))
                except Exception as e:
                    logger.warning(f"加载会话失败 {file}: {e}")
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        for char_dir in self.base_path.iterdir():
            if char_dir.is_dir():
                session_file = char_dir / f"{session_id}.json"
                if session_file.exists():
                    session_file.unlink()
                    logger.debug(f"会话已删除: {session_id}")
                    return True
        return False
