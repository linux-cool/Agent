# execution_engine.py
"""
第5章 规划与执行引擎开发 - 执行引擎
实现智能体的任务执行、状态管理、错误处理等功能
"""

import asyncio
import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Awaitable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
import threading
from collections import defaultdict, deque
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExecutionStatus(Enum):
    """执行状态枚举"""
    IDLE = "空闲"
    RUNNING = "运行中"
    PAUSED = "暂停"
    COMPLETED = "已完成"
    FAILED = "失败"
    CANCELLED = "已取消"
    TIMEOUT = "超时"

class ExecutionMode(Enum):
    """执行模式枚举"""
    SEQUENTIAL = "顺序执行"
    PARALLEL = "并行执行"
    PIPELINE = "流水线执行"
    CONDITIONAL = "条件执行"
    LOOP = "循环执行"

class ErrorType(Enum):
    """错误类型枚举"""
    TIMEOUT = "超时错误"
    RESOURCE_ERROR = "资源错误"
    DEPENDENCY_ERROR = "依赖错误"
    VALIDATION_ERROR = "验证错误"
    EXECUTION_ERROR = "执行错误"
    NETWORK_ERROR = "网络错误"
    UNKNOWN_ERROR = "未知错误"

@dataclass
class ExecutionContext:
    """执行上下文数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    agent_id: str = ""
    environment: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "environment": self.environment,
            "variables": self.variables,
            "resources": self.resources,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }

@dataclass
class ExecutionResult:
    """执行结果数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    agent_id: str = ""
    status: ExecutionStatus = ExecutionStatus.COMPLETED
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration: timedelta = field(default_factory=lambda: timedelta(seconds=0))
    output: Any = None
    error: Optional[str] = None
    error_type: Optional[ErrorType] = None
    progress: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration.total_seconds(),
            "output": self.output,
            "error": self.error,
            "error_type": self.error_type.value if self.error_type else None,
            "progress": self.progress,
            "metrics": self.metrics,
            "metadata": self.metadata
        }

@dataclass
class ExecutionStep:
    """执行步骤数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    step_type: str = "action"
    parameters: Dict[str, Any] = field(default_factory=dict)
    conditions: List[str] = field(default_factory=list)
    timeout: Optional[timedelta] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "step_type": self.step_type,
            "parameters": self.parameters,
            "conditions": self.conditions,
            "timeout": self.timeout.total_seconds() if self.timeout else None,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }

class TaskExecutor:
    """任务执行器"""
    
    def __init__(self, executor_id: str, config: Dict[str, Any]):
        self.executor_id = executor_id
        self.config = config
        self.status = ExecutionStatus.IDLE
        self.current_task_id: Optional[str] = None
        self.execution_context: Optional[ExecutionContext] = None
        self.running = False
        self.task_queue = asyncio.Queue()
        self.results: Dict[str, ExecutionResult] = {}
        self.error_handlers: Dict[ErrorType, Callable] = {}
        self.step_handlers: Dict[str, Callable] = {}
    
    async def start(self):
        """启动执行器"""
        self.running = True
        asyncio.create_task(self._execution_loop())
        logger.info(f"Task executor {self.executor_id} started")
    
    async def stop(self):
        """停止执行器"""
        self.running = False
        logger.info(f"Task executor {self.executor_id} stopped")
    
    async def _execution_loop(self):
        """执行循环"""
        while self.running:
            try:
                # 获取任务
                task_data = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                await self._execute_task(task_data)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Execution loop error: {e}")
    
    async def _execute_task(self, task_data: Dict[str, Any]):
        """执行任务"""
        try:
            task_id = task_data.get("task_id")
            task_name = task_data.get("name", "Unknown Task")
            steps = task_data.get("steps", [])
            context = task_data.get("context", {})
            
            # 创建执行上下文
            self.execution_context = ExecutionContext(
                task_id=task_id,
                agent_id=self.executor_id,
                environment=context.get("environment", {}),
                variables=context.get("variables", {}),
                resources=context.get("resources", {})
            )
            
            # 创建执行结果
            result = ExecutionResult(
                task_id=task_id,
                agent_id=self.executor_id,
                status=ExecutionStatus.RUNNING
            )
            
            self.status = ExecutionStatus.RUNNING
            self.current_task_id = task_id
            
            logger.info(f"Starting execution of task: {task_name}")
            
            # 执行步骤
            for step_data in steps:
                step_result = await self._execute_step(step_data)
                if not step_result:
                    result.status = ExecutionStatus.FAILED
                    result.error = f"Step execution failed: {step_data.get('name', 'Unknown')}"
                    break
                
                result.progress = (steps.index(step_data) + 1) / len(steps)
            
            # 完成执行
            if result.status == ExecutionStatus.RUNNING:
                result.status = ExecutionStatus.COMPLETED
                result.progress = 1.0
            
            result.end_time = datetime.now()
            result.duration = result.end_time - result.start_time
            
            self.results[task_id] = result
            self.status = ExecutionStatus.IDLE
            self.current_task_id = None
            
            logger.info(f"Completed execution of task: {task_name}, status: {result.status.value}")
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            if self.current_task_id:
                result = ExecutionResult(
                    task_id=self.current_task_id,
                    agent_id=self.executor_id,
                    status=ExecutionStatus.FAILED,
                    error=str(e),
                    error_type=ErrorType.EXECUTION_ERROR
                )
                result.end_time = datetime.now()
                result.duration = result.end_time - result.start_time
                self.results[self.current_task_id] = result
            
            self.status = ExecutionStatus.IDLE
            self.current_task_id = None
    
    async def _execute_step(self, step_data: Dict[str, Any]) -> bool:
        """执行步骤"""
        try:
            step_name = step_data.get("name", "Unknown Step")
            step_type = step_data.get("type", "action")
            parameters = step_data.get("parameters", {})
            timeout = step_data.get("timeout")
            max_retries = step_data.get("max_retries", 3)
            
            logger.info(f"Executing step: {step_name} (type: {step_type})")
            
            # 检查条件
            conditions = step_data.get("conditions", [])
            if not await self._check_conditions(conditions):
                logger.warning(f"Conditions not met for step: {step_name}")
                return False
            
            # 执行步骤
            retry_count = 0
            while retry_count <= max_retries:
                try:
                    if timeout:
                        result = await asyncio.wait_for(
                            self._run_step_handler(step_type, parameters),
                            timeout=timeout
                        )
                    else:
                        result = await self._run_step_handler(step_type, parameters)
                    
                    if result:
                        logger.info(f"Step completed successfully: {step_name}")
                        return True
                    else:
                        logger.warning(f"Step handler returned False: {step_name}")
                        return False
                        
                except asyncio.TimeoutError:
                    retry_count += 1
                    logger.warning(f"Step timeout (attempt {retry_count}/{max_retries}): {step_name}")
                    if retry_count > max_retries:
                        logger.error(f"Step failed after {max_retries} retries: {step_name}")
                        return False
                
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Step error (attempt {retry_count}/{max_retries}): {step_name}, error: {e}")
                    if retry_count > max_retries:
                        logger.error(f"Step failed after {max_retries} retries: {step_name}")
                        return False
            
            return False
            
        except Exception as e:
            logger.error(f"Step execution failed: {e}")
            return False
    
    async def _check_conditions(self, conditions: List[str]) -> bool:
        """检查条件"""
        try:
            if not conditions:
                return True
            
            for condition in conditions:
                # 简化的条件检查
                if condition.startswith("variable:"):
                    var_name = condition[9:]  # 去掉 "variable:" 前缀
                    if var_name not in self.execution_context.variables:
                        return False
                elif condition.startswith("resource:"):
                    resource_name = condition[9:]  # 去掉 "resource:" 前缀
                    if resource_name not in self.execution_context.resources:
                        return False
                else:
                    # 其他条件检查逻辑
                    pass
            
            return True
            
        except Exception as e:
            logger.error(f"Condition check failed: {e}")
            return False
    
    async def _run_step_handler(self, step_type: str, parameters: Dict[str, Any]) -> bool:
        """运行步骤处理器"""
        try:
            if step_type in self.step_handlers:
                handler = self.step_handlers[step_type]
                return await handler(parameters, self.execution_context)
            else:
                # 默认处理器
                return await self._default_step_handler(step_type, parameters)
                
        except Exception as e:
            logger.error(f"Step handler failed: {e}")
            return False
    
    async def _default_step_handler(self, step_type: str, parameters: Dict[str, Any]) -> bool:
        """默认步骤处理器"""
        try:
            if step_type == "action":
                # 模拟动作执行
                duration = parameters.get("duration", 1.0)
                await asyncio.sleep(duration)
                return True
            
            elif step_type == "calculation":
                # 模拟计算
                expression = parameters.get("expression", "1 + 1")
                result = eval(expression)  # 注意：实际应用中应使用安全的表达式求值
                self.execution_context.variables["result"] = result
                return True
            
            elif step_type == "data_processing":
                # 模拟数据处理
                data_size = parameters.get("data_size", 1000)
                await asyncio.sleep(data_size / 10000)  # 模拟处理时间
                return True
            
            elif step_type == "api_call":
                # 模拟API调用
                url = parameters.get("url", "http://example.com")
                await asyncio.sleep(0.1)  # 模拟网络延迟
                self.execution_context.variables["api_response"] = {"status": "success", "url": url}
                return True
            
            else:
                logger.warning(f"Unknown step type: {step_type}")
                return False
                
        except Exception as e:
            logger.error(f"Default step handler failed: {e}")
            return False
    
    async def submit_task(self, task_data: Dict[str, Any]) -> bool:
        """提交任务"""
        try:
            await self.task_queue.put(task_data)
            logger.info(f"Task submitted to executor {self.executor_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to submit task: {e}")
            return False
    
    def register_step_handler(self, step_type: str, handler: Callable):
        """注册步骤处理器"""
        self.step_handlers[step_type] = handler
        logger.info(f"Registered step handler for type: {step_type}")
    
    def register_error_handler(self, error_type: ErrorType, handler: Callable):
        """注册错误处理器"""
        self.error_handlers[error_type] = handler
        logger.info(f"Registered error handler for type: {error_type.value}")
    
    def get_result(self, task_id: str) -> Optional[ExecutionResult]:
        """获取执行结果"""
        return self.results.get(task_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_results = len(self.results)
        completed_results = len([r for r in self.results.values() if r.status == ExecutionStatus.COMPLETED])
        failed_results = len([r for r in self.results.values() if r.status == ExecutionStatus.FAILED])
        
        return {
            "executor_id": self.executor_id,
            "status": self.status.value,
            "current_task_id": self.current_task_id,
            "total_tasks": total_results,
            "completed_tasks": completed_results,
            "failed_tasks": failed_results,
            "success_rate": completed_results / max(total_results, 1),
            "queue_size": self.task_queue.qsize()
        }

class ExecutionEngine:
    """执行引擎主类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.executors: Dict[str, TaskExecutor] = {}
        self.execution_queue = asyncio.Queue()
        self.running = False
        self.execution_mode = ExecutionMode.SEQUENTIAL
        self.max_concurrent_executions = config.get("max_concurrent_executions", 5)
        self.execution_timeout = config.get("execution_timeout", 300)  # 5分钟
    
    async def start(self):
        """启动执行引擎"""
        self.running = True
        
        # 创建执行器
        num_executors = self.config.get("num_executors", 3)
        for i in range(num_executors):
            executor_id = f"executor_{i+1}"
            executor = TaskExecutor(executor_id, self.config)
            await executor.start()
            self.executors[executor_id] = executor
        
        # 启动执行调度循环
        asyncio.create_task(self._execution_scheduler())
        
        logger.info("Execution engine started")
    
    async def stop(self):
        """停止执行引擎"""
        self.running = False
        
        # 停止所有执行器
        for executor in self.executors.values():
            await executor.stop()
        
        logger.info("Execution engine stopped")
    
    async def _execution_scheduler(self):
        """执行调度器"""
        while self.running:
            try:
                # 获取任务
                task_data = await asyncio.wait_for(self.execution_queue.get(), timeout=1.0)
                
                # 选择执行器
                executor = await self._select_executor()
                if executor:
                    await executor.submit_task(task_data)
                else:
                    logger.warning("No available executor found")
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Execution scheduler error: {e}")
    
    async def _select_executor(self) -> Optional[TaskExecutor]:
        """选择执行器"""
        try:
            # 选择空闲的执行器
            idle_executors = [e for e in self.executors.values() if e.status == ExecutionStatus.IDLE]
            if idle_executors:
                return idle_executors[0]
            
            # 如果没有空闲的，选择负载最小的
            if self.executors:
                return min(self.executors.values(), key=lambda e: e.task_queue.qsize())
            
            return None
            
        except Exception as e:
            logger.error(f"Executor selection failed: {e}")
            return None
    
    async def execute_task(self, task_data: Dict[str, Any]) -> str:
        """执行任务"""
        try:
            task_id = task_data.get("task_id", str(uuid.uuid4()))
            task_data["task_id"] = task_id
            
            await self.execution_queue.put(task_data)
            logger.info(f"Task submitted for execution: {task_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to execute task: {e}")
            return ""
    
    async def get_execution_result(self, task_id: str) -> Optional[ExecutionResult]:
        """获取执行结果"""
        try:
            for executor in self.executors.values():
                result = executor.get_result(task_id)
                if result:
                    return result
            return None
            
        except Exception as e:
            logger.error(f"Failed to get execution result: {e}")
            return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        try:
            # 这里需要实现任务取消逻辑
            # 由于简化实现，这里只是记录日志
            logger.info(f"Task cancellation requested: {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel task: {e}")
            return False
    
    def register_step_handler(self, step_type: str, handler: Callable):
        """注册步骤处理器"""
        for executor in self.executors.values():
            executor.register_step_handler(step_type, handler)
    
    def register_error_handler(self, error_type: ErrorType, handler: Callable):
        """注册错误处理器"""
        for executor in self.executors.values():
            executor.register_error_handler(error_type, handler)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        executor_stats = {}
        for executor_id, executor in self.executors.items():
            executor_stats[executor_id] = executor.get_stats()
        
        total_tasks = sum(stats["total_tasks"] for stats in executor_stats.values())
        completed_tasks = sum(stats["completed_tasks"] for stats in executor_stats.values())
        failed_tasks = sum(stats["failed_tasks"] for stats in executor_stats.values())
        
        return {
            "running": self.running,
            "execution_mode": self.execution_mode.value,
            "num_executors": len(self.executors),
            "queue_size": self.execution_queue.qsize(),
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": completed_tasks / max(total_tasks, 1),
            "executor_stats": executor_stats
        }

# 示例用法
async def main_demo():
    """执行引擎演示"""
    # 创建执行引擎配置
    config = {
        "num_executors": 3,
        "max_concurrent_executions": 5,
        "execution_timeout": 300
    }
    
    # 创建执行引擎
    execution_engine = ExecutionEngine(config)
    await execution_engine.start()
    
    # 创建示例任务
    print("创建示例任务...")
    tasks = [
        {
            "task_id": "task_1",
            "name": "数据收集任务",
            "steps": [
                {
                    "name": "连接数据库",
                    "type": "api_call",
                    "parameters": {"url": "http://database.example.com"},
                    "timeout": 30
                },
                {
                    "name": "查询数据",
                    "type": "data_processing",
                    "parameters": {"data_size": 5000},
                    "timeout": 60
                },
                {
                    "name": "保存结果",
                    "type": "action",
                    "parameters": {"duration": 2.0}
                }
            ],
            "context": {
                "environment": {"database": "production"},
                "variables": {"query_limit": 1000},
                "resources": {"memory": "2GB"}
            }
        },
        {
            "task_id": "task_2",
            "name": "计算任务",
            "steps": [
                {
                    "name": "数据计算",
                    "type": "calculation",
                    "parameters": {"expression": "sum(range(1000))"},
                    "timeout": 10
                },
                {
                    "name": "结果验证",
                    "type": "action",
                    "parameters": {"duration": 1.0}
                }
            ],
            "context": {
                "variables": {"input_data": [1, 2, 3, 4, 5]},
                "resources": {"cpu": "high"}
            }
        }
    ]
    
    # 提交任务
    print("\n提交任务...")
    task_ids = []
    for task in tasks:
        task_id = await execution_engine.execute_task(task)
        task_ids.append(task_id)
        print(f"✓ 提交任务: {task['name']} (ID: {task_id})")
    
    # 等待任务完成
    print("\n等待任务完成...")
    for task_id in task_ids:
        while True:
            result = await execution_engine.get_execution_result(task_id)
            if result:
                print(f"✓ 任务完成: {task_id}")
                print(f"  状态: {result.status.value}")
                print(f"  持续时间: {result.duration.total_seconds():.2f}秒")
                print(f"  进度: {result.progress:.2f}")
                if result.error:
                    print(f"  错误: {result.error}")
                break
            await asyncio.sleep(1)
    
    # 获取统计信息
    print("\n执行引擎统计:")
    stats = execution_engine.get_stats()
    for key, value in stats.items():
        if key != "executor_stats":
            print(f"  {key}: {value}")
    
    # 停止执行引擎
    await execution_engine.stop()
    print("\n执行引擎演示完成")

if __name__ == "__main__":
    asyncio.run(main_demo())
