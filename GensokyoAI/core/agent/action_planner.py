"""行动规划器 - Agent 的大脑决策区域"""

# GensokyoAI/core/agent/action_planner.py

import json
import re
from typing import Optional, TYPE_CHECKING

from .actions import Action, ActionType, ActionFactory
from ..events import EventBus, Event, SystemEvent, EventPriority
from ...utils.logger import logger

if TYPE_CHECKING:
    from .model_client import ModelClient
    from ...memory.working import WorkingMemoryManager
    from ...memory.semantic import SemanticMemoryManager


class ActionPlanner:
    """
    行动规划器 - Agent 的大脑

    慧音：三思而后行！
    紫：边界要模糊，考虑多种可能！
    灵梦：简单点，能偷懒就偷懒~
    """

    def __init__(
        self,
        character_name: str,
        model_client: "ModelClient",
        working_memory: "WorkingMemoryManager",
        semantic_memory: "SemanticMemoryManager",
        event_bus: EventBus,
    ):
        self.character_name = character_name
        self.model_client = model_client
        self.working_memory = working_memory
        self.semantic_memory = semantic_memory
        self.event_bus = event_bus

        self._last_action: Optional[Action] = None
        self._action_history: list[Action] = []

        self._subscribe_events()
        logger.debug(f"🧠 [ActionPlanner] 初始化完成，角色: {character_name}")

    def _subscribe_events(self) -> None:
        """订阅需要决策的事件"""
        self.event_bus.subscribe(
            SystemEvent.MESSAGE_RECEIVED, self._on_message_received, priority=EventPriority.HIGHEST
        )
        self.event_bus.subscribe(
            SystemEvent.THINK_ENGINE_THOUGHT,
            self._on_thought_generated,
        )
        self.event_bus.subscribe(
            SystemEvent.TOOL_CALL_COMPLETED,
            self._on_tool_completed,
        )

    # ==================== 事件处理 ====================

    async def _on_message_received(self, event: Event) -> None:
        """收到用户消息 - 决定如何回应"""
        user_input = event.data.get("content", "")

        # 空消息不回应
        if not user_input or len(user_input.strip()) <= 1:
            action = ActionFactory.wait(reason="用户输入太短")
        else:
            action = ActionFactory.speak(reason=f"回应: {user_input[:30]}...")

        self._record_action(action)
        self._publish_action(action, trigger_event=event)

    async def _on_thought_generated(self, event: Event) -> None:
        """思考引擎产生想法 - 决定是否主动说话"""
        thought = event.data.get("thought", "")
        topics_detail = event.data.get("topics_detail", [])

        if not thought:
            return

        action = await self._decide_initiative_action(thought, topics_detail)

        if action.type != ActionType.WAIT:
            self._record_action(action)
            self._publish_action(action, trigger_event=event)
            logger.info(f"✨ [ActionPlanner] {self.character_name} 决定主动说话")
        else:
            logger.debug(f"🤫 [ActionPlanner] {self.character_name} 决定不主动说话")

    async def _on_tool_completed(self, event: Event) -> None:
        """工具执行完成 - 继续对话"""
        action = ActionFactory.speak(reason=f"工具执行完成，整合结果")
        self._publish_action(action, trigger_event=event)

    # ==================== 决策核心 ====================

    async def _decide_initiative_action(self, thought: str, topics_detail: list) -> Action:
        """决定是否主动说话"""
        topics_desc = (
            "\n".join(f"- {t.get('name', '')}: {t.get('summary', '')}" for t in topics_detail)
            if topics_detail
            else "无"
        )

        prompt = f"""你是 {self.character_name}，刚刚在静默思考：

思考内容：{thought}
相关话题：{topics_desc}

用 JSON 回答是否想主动说话：
{{"should_speak": true/false, "reason": "...", "message": "..."}}

只输出 JSON。"""

        try:
            response = await self.model_client.chat(
                messages=[{"role": "system", "content": prompt}],
                options={"temperature": 0.7, "num_predict": 200},
            )

            result_text = response.message.content.strip()  # type: ignore
            json_match = re.search(r"\{[^{}]*\}", result_text)

            if json_match:
                data = json.loads(json_match.group())
                if data.get("should_speak", False):
                    message = data.get("message", "").strip()
                    if message:
                        return ActionFactory.initiative_speak(
                            content=message, reason=data.get("reason", "主动想说")
                        )
        except Exception as e:
            logger.error(f"决策主动行动失败: {e}")

        return ActionFactory.wait()

    # ==================== 行动发布 ====================

    def _publish_action(self, action: Action, trigger_event: Optional[Event] = None) -> None:
        """发布行动决策事件"""
        self.event_bus.publish(
            Event(
                type=SystemEvent.ACTION_DECIDED,
                source="action_planner",
                data={
                    "action": action.to_dict(),
                    "trigger_event_id": trigger_event.id if trigger_event else None,
                    "user_input": trigger_event.data.get("content") if trigger_event else None,
                },
            )
        )
        logger.info(f"🧠 [ActionPlanner] 决策: {action.type.name} - {action.reason}")

    def _record_action(self, action: Action) -> None:
        self._last_action = action
        self._action_history.append(action)
        if len(self._action_history) > 50:
            self._action_history = self._action_history[-50:]

    @property
    def last_action(self) -> Optional[Action]:
        return self._last_action
