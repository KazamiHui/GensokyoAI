"""后台任务管理器 - 只负责调度和委托"""

import asyncio
from collections import deque
from typing import Callable, Awaitable

from .types import (
    BackgroundTask,
    TaskResult,
    TaskType,
    TaskPriority,
    MemoryTaskData,
    PersistenceTaskData,
)
from .workers import MemoryWorker, PersistenceWorker
from ..utils.logging import logger


class BackgroundManager:
    """后台任务管理器
    
    职责：
    - 管理任务队列
    - 委托任务给对应的工作器
    - 控制并发数量
    - 不处理具体业务逻辑
    """
    
    def __init__(self, max_workers: int = 3, max_queue_size: int = 100):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        
        # 任务队列（按优先级分组）
        self._queues: dict[TaskPriority, deque[BackgroundTask]] = {
            TaskPriority.LOW: deque(),
            TaskPriority.NORMAL: deque(),
            TaskPriority.HIGH: deque(),
            TaskPriority.CRITICAL: deque(),
        }
        
        # 工作器注册表
        self._workers: dict[TaskType, object] = {}
        
        # 运行状态
        self._running = False
        self._worker_tasks: list[asyncio.Task] = []
        self._result_callbacks: list[Callable[[TaskResult], Awaitable[None]]] = []
        
        # 统计信息
        self._stats = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "timeout": 0,
        }
    
    # ==================== 工作器注册 ====================
    
    def register_worker(self, task_type: TaskType, worker: object) -> "BackgroundManager":
        """注册工作器"""
        self._workers[task_type] = worker
        logger.debug(f"注册工作器: {task_type.name}")
        return self
    
    def register_memory_worker(self, worker: MemoryWorker) -> "BackgroundManager":
        """注册记忆工作器"""
        return self.register_worker(TaskType.MEMORY, worker)
    
    def register_persistence_worker(self, worker: PersistenceWorker) -> "BackgroundManager":
        """注册持久化工作器"""
        return self.register_worker(TaskType.PERSISTENCE, worker)
    
    # ==================== 回调注册 ====================
    
    def on_complete(self, callback: Callable[[TaskResult], Awaitable[None]]) -> "BackgroundManager":
        """注册完成回调"""
        self._result_callbacks.append(callback)
        return self
    
    # ==================== 任务提交 ====================
    
    def submit(self, task: BackgroundTask) -> bool:
        """提交任务到队列"""
        total_tasks = sum(len(q) for q in self._queues.values())
        if total_tasks >= self.max_queue_size:
            logger.warning(f"任务队列已满 ({self.max_queue_size})，丢弃任务: {task.name}")
            return False
        
        self._queues[task.priority].append(task)
        self._stats["submitted"] += 1
        logger.debug(f"提交任务: {task.name} (优先级: {task.priority.name})")
        return True
    
    def submit_memory_task(
        self,
        user_input: str,
        assistant_response: str,
        priority: TaskPriority = TaskPriority.LOW,
        timeout: float = 5.0,
    ) -> bool:
        """提交记忆任务"""
        task = BackgroundTask(
            type=TaskType.MEMORY,
            priority=priority,
            name=f"memory_{len(user_input)}",
            data=MemoryTaskData(
                user_input=user_input,
                assistant_response=assistant_response,
            ),
            timeout=timeout,
        )
        return self.submit(task)
    
    def submit_persistence_task(
        self,
        operation: str,
        data: dict,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: float = 10.0,
    ) -> bool:
        """提交持久化任务"""
        task = BackgroundTask(
            type=TaskType.PERSISTENCE,
            priority=priority,
            name=f"persist_{operation}",
            data=PersistenceTaskData(
                operation=operation,
                data=data,
            ),
            timeout=timeout,
        )
        return self.submit(task)
    
    # ==================== 生命周期 ====================
    
    async def start(self) -> None:
        """启动管理器"""
        if self._running:
            return
        
        self._running = True
        
        # 启动工作协程
        for i in range(self.max_workers):
            task = asyncio.create_task(self._worker_loop(i))
            self._worker_tasks.append(task)
        
        logger.info(f"后台管理器已启动 ({self.max_workers} 个工作器)")
    
    async def stop(self, wait: bool = True) -> None:
        """停止管理器"""
        if not self._running:
            return
        
        self._running = False
        
        if wait:
            # 等待队列清空
            timeout = 5.0
            start = asyncio.get_event_loop().time()
            while any(self._queues.values()):
                if asyncio.get_event_loop().time() - start > timeout:
                    logger.warning(f"等待队列清空超时，剩余任务将被丢弃")
                    break
                await asyncio.sleep(0.1)
        
        # 取消所有工作器
        for task in self._worker_tasks:
            task.cancel()
        
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()
        
        logger.info(
            f"后台管理器已停止 "
            f"(提交: {self._stats['submitted']}, "
            f"完成: {self._stats['completed']}, "
            f"失败: {self._stats['failed']}, "
            f"超时: {self._stats['timeout']})"
        )
    
    # ==================== 工作循环 ====================
    
    async def _worker_loop(self, worker_id: int) -> None:
        """工作器循环"""
        logger.debug(f"工作器 {worker_id} 已启动")
        
        while self._running:
            # 获取下一个任务
            task = self._get_next_task()
            if task is None:
                await asyncio.sleep(0.05)
                continue
            
            # 获取对应的工作器
            worker = self._workers.get(task.type)
            if worker is None:
                logger.warning(f"未找到工作器: {task.type.name}")
                continue
            
            # 执行任务
            try:
                result = await worker.process(task)  # type: ignore
                self._update_stats(result)
                
                # 触发回调
                for callback in self._result_callbacks:
                    try:
                        await callback(result)
                    except Exception as e:
                        logger.debug(f"回调执行失败: {e}")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"任务执行异常: {e}")
        
        logger.debug(f"工作器 {worker_id} 已停止")
    
    def _get_next_task(self) -> BackgroundTask | None:
        """获取下一个任务（按优先级）"""
        for priority in [TaskPriority.CRITICAL, TaskPriority.HIGH, TaskPriority.NORMAL, TaskPriority.LOW]:
            queue = self._queues[priority]
            if queue:
                return queue.popleft()
        return None
    
    def _update_stats(self, result: TaskResult) -> None:
        """更新统计信息"""
        self._stats["completed"] += 1
        if not result.success:
            self._stats["failed"] += 1
            if result.error == "timeout":
                self._stats["timeout"] += 1
    
    # ==================== 状态查询 ====================
    
    @property
    def queue_size(self) -> int:
        """当前队列大小"""
        return sum(len(q) for q in self._queues.values())
    
    @property
    def stats(self) -> dict:
        """获取统计信息"""
        return self._stats.copy()
    
    def clear_queues(self) -> None:
        """清空所有队列"""
        for queue in self._queues.values():
            queue.clear()