"""语义记忆 - 支持模型提取和向量检索双模式"""

import json
from typing import Optional
from pathlib import Path
from dataclasses import asdict
from enum import Enum

import ollama
import numpy as np

from .types import SemanticMemory
from ..core.config import MemoryConfig
from ..utils.logging import logger


class SemanticMode(Enum):
    """语义记忆模式"""

    EMBEDDING = "embedding"  # 向量检索模式
    MODEL_EXTRACT = "model_extract"  # 模型提取模式
    DISABLED = "disabled"  # 禁用


class SimpleVectorStore:
    """简单的向量存储（基于 JSON 文件）"""

    def __init__(self, path: Path):
        self.path = path
        self._data: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception as e:
                logger.warning(f"加载向量存储失败: {e}")
                self._data = []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def add(self, item: dict) -> None:
        self._data.append(item)
        self._save()

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """余弦相似度搜索"""
        if not self._data:
            return []

        try:
            query_vec = np.array(query_embedding)

            results = []
            for item in self._data:
                if "embedding" not in item:
                    continue
                item_vec = np.array(item["embedding"])
                similarity = np.dot(query_vec, item_vec) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(item_vec) + 1e-8
                )
                results.append((similarity, item))

            results.sort(key=lambda x: x[0], reverse=True)
            return [item for _, item in results[:top_k]]
        except Exception as e:
            logger.warning(f"向量搜索失败: {e}")
            return []

    def get_all(self) -> list[dict]:
        """获取所有数据"""
        return self._data.copy()

    def delete(self, id: str) -> None:
        self._data = [item for item in self._data if item.get("id") != id]
        self._save()


class SemanticMemoryManager:
    """语义记忆管理器 - 支持自动降级"""

    def __init__(self, config: MemoryConfig, character_id: str, base_path: Path):
        self.config = config
        self.character_id = character_id
        self._store = SimpleVectorStore(base_path / f"{character_id}_semantic.json")
        self._mode = SemanticMode.DISABLED
        self._embedding_checked = False
        self._embedding_available = False

        if config.semantic_enabled:
            self._init_mode()

    def _init_mode(self) -> None:
        """初始化语义记忆模式"""
        # 先尝试检测 embedding 是否可用
        if self._check_embedding_available():
            self._mode = SemanticMode.EMBEDDING
            logger.info(
                f"语义记忆使用向量检索模式: {self.config.semantic_embedding_model}"
            )
        else:
            # 降级到模型提取模式
            self._mode = SemanticMode.MODEL_EXTRACT
            logger.info("语义记忆降级为模型提取模式")

    def _check_embedding_available(self) -> bool:
        """检测 embedding 模型是否可用（只检测一次）"""
        if self._embedding_checked:
            return self._embedding_available

        try:
            # 简单测试
            response = ollama.embeddings(
                model=self.config.semantic_embedding_model, prompt="test"
            )
            self._embedding_available = response is not None
        except Exception as e:
            # 静默处理，不打印警告
            self._embedding_available = False

        self._embedding_checked = True
        return self._embedding_available

    def _get_embedding(self, text: str) -> Optional[list[float]]:
        """获取文本向量"""
        if self._mode != SemanticMode.EMBEDDING:
            return None

        try:
            response = ollama.embeddings(
                model=self.config.semantic_embedding_model, prompt=text
            )
            return response.embedding  # type: ignore
        except Exception as e:
            # 只记录一次错误
            if not hasattr(self, "_embedding_error_logged"):
                logger.debug(f"Embedding 不可用，将使用模型提取模式: {e}")
                self._embedding_error_logged = True
            self._mode = SemanticMode.MODEL_EXTRACT
            return None

    def _extract_key_info_with_model(self, content: str) -> tuple[str, list[str]]:
        """使用模型提取关键信息"""
        prompt = f"""请从以下内容中提取关键信息，用于后续检索。
返回格式为 JSON，包含两个字段：
- summary: 一句话摘要（不超过50字）
- keywords: 关键词列表（3-5个）

内容：
{content}

只返回 JSON，不要其他内容。"""

        try:
            response = ollama.chat(
                model=self.config.auto_memory_model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                options={"temperature": 0.3},
            )

            result_text = response.message.content.strip()  # type: ignore

            # 尝试提取 JSON
            import re

            json_match = re.search(r"\{.*\}", result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result.get("summary", content[:50]), result.get("keywords", [])
        except Exception as e:
            logger.debug(f"模型提取关键信息失败: {e}")

        # 降级：简单截取
        return content[:50], []

    def add(
        self, content: str, importance: float = 0.0, tags: list[str] | None = None
    ) -> Optional[str]:
        """添加语义记忆"""
        if self._mode == SemanticMode.DISABLED:
            return None

        memory = SemanticMemory(
            content=content, embedding=None, importance=importance, tags=tags or []
        )

        if self._mode == SemanticMode.EMBEDDING:
            embedding = self._get_embedding(content)
            if embedding:
                memory.embedding = embedding
            else:
                # 降级后继续
                pass

        # 如果是模型提取模式，添加摘要和关键词到 metadata
        if self._mode == SemanticMode.MODEL_EXTRACT or memory.embedding is None:
            summary, keywords = self._extract_key_info_with_model(content)
            memory_dict = asdict(memory)
            memory_dict["extracted_summary"] = summary
            memory_dict["extracted_keywords"] = keywords
            self._store.add(memory_dict)
            logger.debug(f"添加语义记忆(模型提取): {summary[:30]}...")
        else:
            self._store.add(asdict(memory))
            logger.debug(f"添加语义记忆(向量): {content[:30]}...")

        return memory.id

    def search(self, query: str, top_k: int | None = None) -> list[SemanticMemory]:
        """搜索相关记忆"""
        if self._mode == SemanticMode.DISABLED:
            return []

        top_k = top_k or self.config.semantic_top_k

        if self._mode == SemanticMode.EMBEDDING:
            return self._search_by_embedding(query, top_k)
        else:
            return self._search_by_keywords(query, top_k)

    def _search_by_embedding(self, query: str, top_k: int) -> list[SemanticMemory]:
        """向量检索"""
        embedding = self._get_embedding(query)
        if embedding is None:
            return self._search_by_keywords(query, top_k)

        results = self._store.search(embedding, top_k)

        memories = []
        for item in results:
            try:
                memories.append(SemanticMemory(**item))
            except Exception as e:
                logger.debug(f"解析记忆失败: {e}")

        return memories

    def _search_by_keywords(self, query: str, top_k: int) -> list[SemanticMemory]:
        """关键词匹配检索（降级方案）"""
        all_items = self._store.get_all()

        # 简单的关键词匹配评分
        scored = []
        query_lower = query.lower()

        for item in all_items:
            score = 0
            content = item.get("content", "").lower()

            # 内容匹配
            if query_lower in content:
                score += 5

            # 关键词匹配
            keywords = item.get("extracted_keywords", [])
            for kw in keywords:
                if kw.lower() in query_lower or query_lower in kw.lower():
                    score += 3

            # 摘要匹配
            summary = item.get("extracted_summary", "").lower()
            if query_lower in summary:
                score += 2

            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)

        memories = []
        for _, item in scored[:top_k]:
            try:
                # 移除临时字段
                item_copy = {
                    k: v
                    for k, v in item.items()
                    if k not in ["extracted_summary", "extracted_keywords"]
                }
                memories.append(SemanticMemory(**item_copy))
            except Exception:
                pass

        return memories

    def get_relevant_context(self, query: str, top_k: int = 3) -> list[str]:
        """获取相关上下文"""
        memories = self.search(query, top_k)
        return [m.content for m in memories if m.importance > 0.3]
