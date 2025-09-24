# task_scheduler.py
"""
第5章 规划与执行引擎开发 - 任务调度器
实现智能体的任务调度、优先级管理、资源分配等功能
"""

import asyncio
import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Set
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
import heapq
from collections import defaultdict, deque
import threading
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SchedulingPolicy(Enum):
    """调度策略枚举"""
    FIFO = "先进先出"
    PRIORITY = "优先级"
    SHORTEST_JOB_FIRST = "最短作业优先"
    ROUND_ROBIN = "轮询"
    DEADLINE_FIRST = "截止时间优先"
    RESOURCE_BASED = "基于资源"
    LOAD_BALANCING = "负载均衡"

class SchedulingAlgorithm(Enum):
    """调度算法枚举"""
    EDF = "最早截止时间优先"
    RATE_MONOTONIC = "速率单调"
    DEADLINE_MONOTONIC = "截止时间单调"
    LEAST_LAXITY_FIRST = "最小松弛度优先"
    PROPORTIONAL_SHARE = "比例共享"
    WEIGHTED_ROUND_ROBIN = "加权轮询"

class ResourceType(Enum):
    """资源类型枚举"""
    CPU = "CPU"
    MEMORY = "内存"
    STORAGE = "存储"
    NETWORK = "网络"
    GPU = "GPU"
    DISK_IO = "磁盘IO"
    CUSTOM = "自定义"

@dataclass
class ResourceRequirement:
    """资源需求数据结构"""
    resource_type: ResourceType = ResourceType.CPU
    amount: float = 1.0
    unit: str = "cores"
    duration: timedelta = field(default_factory=lambda: timedelta(minutes=1))
    priority: int = 1
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "resource_type": self.resource_type.value,
            "amount": self.amount,
            "unit": self.unit,
            "duration": self.duration.total_seconds(),
            "priority": self.priority,
            "constraints": self.constraints
        }

@dataclass
class ScheduledTask:
    """调度任务数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    priority: int = 1
    deadline: Optional[datetime] = None
    estimated_duration: timedelta = field(default_factory=lambda: timedelta(minutes=1))
    resource_requirements: List[ResourceRequirement] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    assigned_agent: Optional[str] = None
    scheduled_time: Optional[datetime] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: str = "pending"
    progress: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "priority": self.priority,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "estimated_duration": self.estimated_duration.total_seconds(),
            "resource_requirements": [req.to_dict() for req in self.resource_requirements],
            "dependencies": self.dependencies,
            "assigned_agent": self.assigned_agent,
            "scheduled_time": self.scheduled_time.isoformat() if self.scheduled_time else None,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "progress": self.progress,
            "metadata": self.metadata
        }

@dataclass
class ResourcePool:
    """资源池数据结构"""
    resource_type: ResourceType = ResourceType.CPU
    total_amount: float = 10.0
    available_amount: float = 10.0
    allocated_amount: float = 0.0
    unit: str = "cores"
    cost_per_unit: float = 1.0
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "resource_type": self.resource_type.value,
            "total_amount": self.total_amount,
            "available_amount": self.available_amount,
            "allocated_amount": self.allocated_amount,
            "unit": self.unit,
            "cost_per_unit": self.cost_per_unit,
            "constraints": self.constraints
        }

@dataclass
class SchedulingDecision:
    """调度决策数据结构"""
    task_id: str = ""
    agent_id: str = ""
    scheduled_time: datetime = field(default_factory=datetime.now)
    estimated_completion_time: datetime = field(default_factory=datetime.now)
    resource_allocation: Dict[str, float] = field(default_factory=dict)
    cost: float = 0.0
    confidence: float = 1.0
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "scheduled_time": self.scheduled_time.isoformat(),
            "estimated_completion_time": self.estimated_completion_time.isoformat(),
            "resource_allocation": self.resource_allocation,
            "cost": self.cost,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "metadata": self.metadata
        }

class PriorityQueue:
    """优先级队列"""
    
    def __init__(self):
        self.queue = []
        self.task_map = {}
    
    def push(self, task: ScheduledTask):
        """添加任务"""
        # 计算优先级分数（越小优先级越高）
        priority_score = self._calculate_priority_score(task)
        heapq.heappush(self.queue, (priority_score, task.id))
        self.task_map[task.id] = task
    
    def pop(self) -> Optional[ScheduledTask]:
        """弹出最高优先级任务"""
        if not self.queue:
            return None
        
        _, task_id = heapq.heappop(self.queue)
        task = self.task_map.pop(task_id, None)
        return task
    
    def peek(self) -> Optional[ScheduledTask]:
        """查看最高优先级任务"""
        if not self.queue:
            return None
        
        _, task_id = self.queue[0]
        return self.task_map.get(task_id)
    
    def remove(self, task_id: str) -> bool:
        """移除任务"""
        if task_id in self.task_map:
            del self.task_map[task_id]
            # 重建队列
            self.queue = [(self._calculate_priority_score(task), task.id) 
                         for task in self.task_map.values()]
            heapq.heapify(self.queue)
            return True
        return False
    
    def _calculate_priority_score(self, task: ScheduledTask) -> float:
        """计算优先级分数"""
        # 基于优先级、截止时间、持续时间等因素
        base_score = task.priority
        
        # 截止时间因子
        if task.deadline:
            time_to_deadline = (task.deadline - datetime.now()).total_seconds()
            if time_to_deadline < 0:
                time_to_deadline = 0
            deadline_factor = 1.0 / (time_to_deadline + 1)
        else:
            deadline_factor = 0
        
        # 持续时间因子（越短优先级越高）
        duration_factor = task.estimated_duration.total_seconds() / 3600  # 转换为小时
        
        return base_score + deadline_factor * 10 + duration_factor
    
    def size(self) -> int:
        """队列大小"""
        return len(self.queue)
    
    def is_empty(self) -> bool:
        """是否为空"""
        return len(self.queue) == 0

class ResourceManager:
    """资源管理器"""
    
    def __init__(self):
        self.resource_pools: Dict[ResourceType, ResourcePool] = {}
        self.allocations: Dict[str, Dict[ResourceType, float]] = {}
        self.allocation_history: List[Dict[str, Any]] = []
    
    def add_resource_pool(self, resource_pool: ResourcePool):
        """添加资源池"""
        self.resource_pools[resource_pool.resource_type] = resource_pool
        logger.info(f"Added resource pool: {resource_pool.resource_type.value}")
    
    def allocate_resources(self, task_id: str, requirements: List[ResourceRequirement]) -> bool:
        """分配资源"""
        try:
            allocation = {}
            
            for req in requirements:
                resource_type = req.resource_type
                amount = req.amount
                
                if resource_type not in self.resource_pools:
                    logger.error(f"Resource pool not found: {resource_type.value}")
                    return False
                
                pool = self.resource_pools[resource_type]
                
                if pool.available_amount < amount:
                    logger.warning(f"Insufficient {resource_type.value}: required {amount}, available {pool.available_amount}")
                    return False
                
                # 分配资源
                pool.available_amount -= amount
                pool.allocated_amount += amount
                allocation[resource_type] = amount
            
            self.allocations[task_id] = allocation
            
            # 记录分配历史
            self.allocation_history.append({
                "task_id": task_id,
                "timestamp": datetime.now(),
                "allocation": allocation,
                "action": "allocate"
            })
            
            logger.info(f"Allocated resources for task {task_id}: {allocation}")
            return True
            
        except Exception as e:
            logger.error(f"Resource allocation failed: {e}")
            return False
    
    def deallocate_resources(self, task_id: str) -> bool:
        """释放资源"""
        try:
            if task_id not in self.allocations:
                logger.warning(f"No allocation found for task: {task_id}")
                return False
            
            allocation = self.allocations[task_id]
            
            for resource_type, amount in allocation.items():
                if resource_type in self.resource_pools:
                    pool = self.resource_pools[resource_type]
                    pool.available_amount += amount
                    pool.allocated_amount -= amount
            
            del self.allocations[task_id]
            
            # 记录释放历史
            self.allocation_history.append({
                "task_id": task_id,
                "timestamp": datetime.now(),
                "allocation": allocation,
                "action": "deallocate"
            })
            
            logger.info(f"Deallocated resources for task {task_id}: {allocation}")
            return True
            
        except Exception as e:
            logger.error(f"Resource deallocation failed: {e}")
            return False
    
    def check_resource_availability(self, requirements: List[ResourceRequirement]) -> bool:
        """检查资源可用性"""
        try:
            for req in requirements:
                resource_type = req.resource_type
                amount = req.amount
                
                if resource_type not in self.resource_pools:
                    return False
                
                pool = self.resource_pools[resource_type]
                if pool.available_amount < amount:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Resource availability check failed: {e}")
            return False
    
    def get_resource_utilization(self) -> Dict[str, float]:
        """获取资源利用率"""
        utilization = {}
        
        for resource_type, pool in self.resource_pools.items():
            if pool.total_amount > 0:
                utilization[resource_type.value] = pool.allocated_amount / pool.total_amount
            else:
                utilization[resource_type.value] = 0.0
        
        return utilization
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "resource_pools": {rt.value: pool.to_dict() for rt, pool in self.resource_pools.items()},
            "active_allocations": len(self.allocations),
            "allocation_history_size": len(self.allocation_history),
            "resource_utilization": self.get_resource_utilization()
        }

class TaskScheduler:
    """任务调度器主类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scheduling_policy = SchedulingPolicy[config.get("scheduling_policy", "PRIORITY").upper()]
        self.scheduling_algorithm = SchedulingAlgorithm[config.get("scheduling_algorithm", "EDF").upper()]
        self.task_queue = PriorityQueue()
        self.resource_manager = ResourceManager()
        self.scheduled_tasks: Dict[str, ScheduledTask] = {}
        self.scheduling_decisions: Dict[str, SchedulingDecision] = {}
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.running = False
        self.scheduling_interval = config.get("scheduling_interval", 1.0)  # 秒
    
    async def start(self):
        """启动调度器"""
        self.running = True
        
        # 初始化资源池
        await self._initialize_resource_pools()
        
        # 启动调度循环
        asyncio.create_task(self._scheduling_loop())
        
        logger.info("Task scheduler started")
    
    async def stop(self):
        """停止调度器"""
        self.running = False
        logger.info("Task scheduler stopped")
    
    async def _initialize_resource_pools(self):
        """初始化资源池"""
        default_pools = [
            ResourcePool(ResourceType.CPU, 8.0, 8.0, 0.0, "cores", 1.0),
            ResourcePool(ResourceType.MEMORY, 16.0, 16.0, 0.0, "GB", 0.5),
            ResourcePool(ResourceType.STORAGE, 100.0, 100.0, 0.0, "GB", 0.1),
            ResourcePool(ResourceType.NETWORK, 1.0, 1.0, 0.0, "Gbps", 2.0)
        ]
        
        for pool in default_pools:
            self.resource_manager.add_resource_pool(pool)
    
    async def _scheduling_loop(self):
        """调度循环"""
        while self.running:
            try:
                await self._schedule_tasks()
                await asyncio.sleep(self.scheduling_interval)
            except Exception as e:
                logger.error(f"Scheduling loop error: {e}")
    
    async def _schedule_tasks(self):
        """调度任务"""
        try:
            # 获取可调度的任务
            schedulable_tasks = await self._get_schedulable_tasks()
            
            for task in schedulable_tasks:
                # 检查资源可用性
                if self.resource_manager.check_resource_availability(task.resource_requirements):
                    # 选择代理
                    agent = await self._select_agent(task)
                    if agent:
                        # 创建调度决策
                        decision = await self._create_scheduling_decision(task, agent)
                        if decision:
                            # 执行调度
                            await self._execute_scheduling_decision(decision)
                            
        except Exception as e:
            logger.error(f"Task scheduling failed: {e}")
    
    async def _get_schedulable_tasks(self) -> List[ScheduledTask]:
        """获取可调度的任务"""
        schedulable = []
        
        for task in self.scheduled_tasks.values():
            if task.status == "pending":
                # 检查依赖是否满足
                if await self._check_dependencies(task):
                    schedulable.append(task)
        
        return schedulable
    
    async def _check_dependencies(self, task: ScheduledTask) -> bool:
        """检查任务依赖"""
        try:
            for dep_id in task.dependencies:
                if dep_id not in self.scheduled_tasks:
                    return False
                
                dep_task = self.scheduled_tasks[dep_id]
                if dep_task.status != "completed":
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Dependency check failed: {e}")
            return False
    
    async def _select_agent(self, task: ScheduledTask) -> Optional[str]:
        """选择代理"""
        try:
            # 简化的代理选择：选择负载最小的代理
            if not self.agents:
                return "default_agent"
            
            # 计算每个代理的负载
            agent_loads = {}
            for agent_id, agent_info in self.agents.items():
                # 计算代理当前任务数
                current_tasks = len([t for t in self.scheduled_tasks.values() 
                                  if t.assigned_agent == agent_id and t.status == "running"])
                agent_loads[agent_id] = current_tasks
            
            # 选择负载最小的代理
            if agent_loads:
                selected_agent = min(agent_loads.keys(), key=lambda k: agent_loads[k])
                return selected_agent
            
            return None
            
        except Exception as e:
            logger.error(f"Agent selection failed: {e}")
            return None
    
    async def _create_scheduling_decision(self, task: ScheduledTask, agent_id: str) -> Optional[SchedulingDecision]:
        """创建调度决策"""
        try:
            # 计算调度时间
            scheduled_time = datetime.now()
            
            # 计算预计完成时间
            estimated_completion_time = scheduled_time + task.estimated_duration
            
            # 计算资源分配
            resource_allocation = {}
            for req in task.resource_requirements:
                resource_allocation[req.resource_type.value] = req.amount
            
            # 计算成本
            cost = 0.0
            for req in task.resource_requirements:
                if req.resource_type in self.resource_manager.resource_pools:
                    pool = self.resource_manager.resource_pools[req.resource_type]
                    cost += req.amount * pool.cost_per_unit * req.duration.total_seconds() / 3600
            
            # 创建决策
            decision = SchedulingDecision(
                task_id=task.id,
                agent_id=agent_id,
                scheduled_time=scheduled_time,
                estimated_completion_time=estimated_completion_time,
                resource_allocation=resource_allocation,
                cost=cost,
                confidence=0.9,
                reasoning=f"Scheduled using {self.scheduling_policy.value} policy"
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Scheduling decision creation failed: {e}")
            return None
    
    async def _execute_scheduling_decision(self, decision: SchedulingDecision):
        """执行调度决策"""
        try:
            task_id = decision.task_id
            
            # 分配资源
            task = self.scheduled_tasks[task_id]
            if not self.resource_manager.allocate_resources(task_id, task.resource_requirements):
                logger.error(f"Failed to allocate resources for task {task_id}")
                return
            
            # 更新任务状态
            task.assigned_agent = decision.agent_id
            task.scheduled_time = decision.scheduled_time
            task.status = "scheduled"
            
            # 保存决策
            self.scheduling_decisions[task_id] = decision
            
            logger.info(f"Task {task_id} scheduled to agent {decision.agent_id}")
            
        except Exception as e:
            logger.error(f"Scheduling decision execution failed: {e}")
    
    async def submit_task(self, task: ScheduledTask) -> bool:
        """提交任务"""
        try:
            self.scheduled_tasks[task.id] = task
            self.task_queue.push(task)
            logger.info(f"Task submitted: {task.name} ({task.id})")
            return True
            
        except Exception as e:
            logger.error(f"Task submission failed: {e}")
            return False
    
    async def update_task_status(self, task_id: str, status: str, progress: float = None):
        """更新任务状态"""
        try:
            if task_id in self.scheduled_tasks:
                task = self.scheduled_tasks[task_id]
                task.status = status
                
                if progress is not None:
                    task.progress = progress
                
                if status == "running" and task.start_time is None:
                    task.start_time = datetime.now()
                elif status == "completed":
                    task.end_time = datetime.now()
                    # 释放资源
                    self.resource_manager.deallocate_resources(task_id)
                
                logger.info(f"Updated task {task_id} status to {status}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Task status update failed: {e}")
            return False
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        try:
            if task_id in self.scheduled_tasks:
                task = self.scheduled_tasks[task_id]
                task.status = "cancelled"
                
                # 释放资源
                self.resource_manager.deallocate_resources(task_id)
                
                # 从队列中移除
                self.task_queue.remove(task_id)
                
                logger.info(f"Task cancelled: {task_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Task cancellation failed: {e}")
            return False
    
    def register_agent(self, agent_id: str, agent_info: Dict[str, Any]):
        """注册代理"""
        self.agents[agent_id] = agent_info
        logger.info(f"Agent registered: {agent_id}")
    
    def unregister_agent(self, agent_id: str):
        """注销代理"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Agent unregistered: {agent_id}")
    
    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """获取任务"""
        return self.scheduled_tasks.get(task_id)
    
    def get_scheduling_decision(self, task_id: str) -> Optional[SchedulingDecision]:
        """获取调度决策"""
        return self.scheduling_decisions.get(task_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_tasks = len(self.scheduled_tasks)
        pending_tasks = len([t for t in self.scheduled_tasks.values() if t.status == "pending"])
        scheduled_tasks = len([t for t in self.scheduled_tasks.values() if t.status == "scheduled"])
        running_tasks = len([t for t in self.scheduled_tasks.values() if t.status == "running"])
        completed_tasks = len([t for t in self.scheduled_tasks.values() if t.status == "completed"])
        failed_tasks = len([t for t in self.scheduled_tasks.values() if t.status == "failed"])
        
        return {
            "scheduling_policy": self.scheduling_policy.value,
            "scheduling_algorithm": self.scheduling_algorithm.value,
            "total_tasks": total_tasks,
            "pending_tasks": pending_tasks,
            "scheduled_tasks": scheduled_tasks,
            "running_tasks": running_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "queue_size": self.task_queue.size(),
            "registered_agents": len(self.agents),
            "resource_stats": self.resource_manager.get_stats()
        }

# 示例用法
async def main_demo():
    """任务调度器演示"""
    # 创建调度器配置
    config = {
        "scheduling_policy": "priority",
        "scheduling_algorithm": "EDF",
        "scheduling_interval": 1.0
    }
    
    # 创建任务调度器
    scheduler = TaskScheduler(config)
    await scheduler.start()
    
    # 注册代理
    print("注册代理...")
    agents = [
        {"id": "agent_1", "capabilities": ["data_processing", "calculation"], "max_tasks": 3},
        {"id": "agent_2", "capabilities": ["api_call", "data_processing"], "max_tasks": 2},
        {"id": "agent_3", "capabilities": ["calculation", "reporting"], "max_tasks": 4}
    ]
    
    for agent in agents:
        scheduler.register_agent(agent["id"], agent)
        print(f"✓ 注册代理: {agent['id']}")
    
    # 创建示例任务
    print("\n创建示例任务...")
    tasks = [
        ScheduledTask(
            name="数据处理任务",
            description="处理大量数据",
            priority=3,
            deadline=datetime.now() + timedelta(minutes=30),
            estimated_duration=timedelta(minutes=10),
            resource_requirements=[
                ResourceRequirement(ResourceType.CPU, 2.0, "cores", timedelta(minutes=10)),
                ResourceRequirement(ResourceType.MEMORY, 4.0, "GB", timedelta(minutes=10))
            ]
        ),
        ScheduledTask(
            name="计算任务",
            description="执行复杂计算",
            priority=2,
            deadline=datetime.now() + timedelta(minutes=20),
            estimated_duration=timedelta(minutes=5),
            resource_requirements=[
                ResourceRequirement(ResourceType.CPU, 1.0, "cores", timedelta(minutes=5)),
                ResourceRequirement(ResourceType.MEMORY, 2.0, "GB", timedelta(minutes=5))
            ]
        ),
        ScheduledTask(
            name="API调用任务",
            description="调用外部API",
            priority=1,
            deadline=datetime.now() + timedelta(minutes=15),
            estimated_duration=timedelta(minutes=3),
            resource_requirements=[
                ResourceRequirement(ResourceType.NETWORK, 0.1, "Gbps", timedelta(minutes=3))
            ]
        )
    ]
    
    # 提交任务
    print("\n提交任务...")
    for task in tasks:
        await scheduler.submit_task(task)
        print(f"✓ 提交任务: {task.name} (优先级: {task.priority})")
    
    # 等待调度
    print("\n等待任务调度...")
    await asyncio.sleep(5)
    
    # 检查任务状态
    print("\n任务状态:")
    for task in tasks:
        current_task = scheduler.get_task(task.id)
        if current_task:
            print(f"  {current_task.name}: {current_task.status}")
            if current_task.assigned_agent:
                print(f"    分配代理: {current_task.assigned_agent}")
            if current_task.scheduled_time:
                print(f"    调度时间: {current_task.scheduled_time}")
    
    # 模拟任务执行
    print("\n模拟任务执行...")
    for task in tasks:
        await scheduler.update_task_status(task.id, "running", 0.5)
        await asyncio.sleep(1)
        await scheduler.update_task_status(task.id, "completed", 1.0)
        print(f"✓ 任务完成: {task.name}")
    
    # 获取统计信息
    print("\n调度器统计:")
    stats = scheduler.get_stats()
    for key, value in stats.items():
        if key != "resource_stats":
            print(f"  {key}: {value}")
    
    # 停止调度器
    await scheduler.stop()
    print("\n任务调度器演示完成")

if __name__ == "__main__":
    asyncio.run(main_demo())
