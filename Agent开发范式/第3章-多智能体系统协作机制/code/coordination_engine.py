# coordination_engine.py
"""
协调引擎实现
提供多智能体系统的协调和调度功能
"""

import asyncio
import logging
import random
import time
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import uuid
import heapq
from collections import defaultdict, deque
import statistics
import json

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoordinationType(Enum):
    """协调类型枚举"""
    CENTRALIZED = "centralized"
    DISTRIBUTED = "distributed"
    HYBRID = "hybrid"
    HIERARCHICAL = "hierarchical"
    PEER_TO_PEER = "peer_to_peer"

class SchedulingPolicy(Enum):
    """调度策略枚举"""
    FIFO = "fifo"
    PRIORITY = "priority"
    ROUND_ROBIN = "round_robin"
    SHORTEST_JOB_FIRST = "shortest_job_first"
    DEADLINE_FIRST = "deadline_first"
    RESOURCE_AWARE = "resource_aware"
    ADAPTIVE = "adaptive"

class ConflictResolutionStrategy(Enum):
    """冲突解决策略枚举"""
    FIRST_COME_FIRST_SERVED = "first_come_first_served"
    PRIORITY_BASED = "priority_based"
    NEGOTIATION = "negotiation"
    AUCTION = "auction"
    CONSENSUS = "consensus"
    RANDOM = "random"

@dataclass
class CoordinationTask:
    """协调任务数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    priority: int = 1
    deadline: Optional[datetime] = None
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    assigned_agents: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, assigned, in_progress, completed, failed
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentRole:
    """智能体角色数据结构"""
    agent_id: str = ""
    role: str = ""
    capabilities: Set[str] = field(default_factory=set)
    responsibilities: List[str] = field(default_factory=list)
    authority_level: int = 1  # 1-5, 5为最高
    assigned_tasks: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class CoordinationEvent:
    """协调事件数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    source_agent: str = ""
    target_agents: List[str] = field(default_factory=list)
    content: Any = None
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 1
    status: str = "pending"  # pending, processed, failed
    result: Optional[Any] = None

class TaskScheduler:
    """任务调度器"""
    
    def __init__(self, policy: SchedulingPolicy = SchedulingPolicy.PRIORITY):
        self.policy = policy
        self.task_queue: List[CoordinationTask] = []
        self.scheduled_tasks: Dict[str, CoordinationTask] = {}
        self.completed_tasks: List[CoordinationTask] = []
        self.agent_workloads: Dict[str, float] = defaultdict(float)
        self.scheduling_history: List[Dict[str, Any]] = []
    
    def add_task(self, task: CoordinationTask):
        """添加任务到调度队列"""
        self.task_queue.append(task)
        logger.info(f"Task {task.id} added to scheduler")
    
    def remove_task(self, task_id: str) -> bool:
        """从调度队列移除任务"""
        for i, task in enumerate(self.task_queue):
            if task.id == task_id:
                del self.task_queue[i]
                logger.info(f"Task {task_id} removed from scheduler")
                return True
        return False
    
    def schedule_tasks(self, available_agents: List[str]) -> List[CoordinationTask]:
        """调度任务"""
        if not self.task_queue or not available_agents:
            return []
        
        # 根据策略排序任务
        sorted_tasks = self._sort_tasks_by_policy()
        
        scheduled = []
        for task in sorted_tasks:
            # 选择最适合的智能体
            best_agent = self._select_best_agent(task, available_agents)
            
            if best_agent:
                # 分配任务
                task.assigned_agents = [best_agent]
                task.status = "assigned"
                task.started_at = datetime.now()
                
                self.scheduled_tasks[task.id] = task
                self.agent_workloads[best_agent] += 0.1  # 增加工作负载
                
                scheduled.append(task)
                
                # 从队列中移除
                self.task_queue.remove(task)
                
                logger.info(f"Task {task.id} scheduled to agent {best_agent}")
        
        # 记录调度历史
        self.scheduling_history.append({
            "timestamp": datetime.now(),
            "scheduled_count": len(scheduled),
            "queue_size": len(self.task_queue),
            "policy": self.policy.value
        })
        
        return scheduled
    
    def _sort_tasks_by_policy(self) -> List[CoordinationTask]:
        """根据策略排序任务"""
        if self.policy == SchedulingPolicy.FIFO:
            return sorted(self.task_queue, key=lambda t: t.created_at)
        elif self.policy == SchedulingPolicy.PRIORITY:
            return sorted(self.task_queue, key=lambda t: t.priority, reverse=True)
        elif self.policy == SchedulingPolicy.ROUND_ROBIN:
            return self.task_queue  # 保持原有顺序
        elif self.policy == SchedulingPolicy.SHORTEST_JOB_FIRST:
            return sorted(self.task_queue, key=lambda t: self._estimate_task_duration(t))
        elif self.policy == SchedulingPolicy.DEADLINE_FIRST:
            return sorted(self.task_queue, key=lambda t: t.deadline or datetime.max)
        elif self.policy == SchedulingPolicy.RESOURCE_AWARE:
            return sorted(self.task_queue, key=lambda t: sum(t.resource_requirements.values()))
        elif self.policy == SchedulingPolicy.ADAPTIVE:
            return self._adaptive_sort()
        else:
            return self.task_queue
    
    def _adaptive_sort(self) -> List[CoordinationTask]:
        """自适应排序"""
        # 结合多种因素进行排序
        def sort_key(task):
            priority_score = task.priority
            deadline_score = 0
            if task.deadline:
                time_remaining = (task.deadline - datetime.now()).total_seconds()
                deadline_score = 1 / (time_remaining + 1)
            
            resource_score = sum(task.resource_requirements.values())
            
            return priority_score + deadline_score - resource_score
        
        return sorted(self.task_queue, key=sort_key, reverse=True)
    
    def _select_best_agent(self, task: CoordinationTask, available_agents: List[str]) -> Optional[str]:
        """选择最适合的智能体"""
        if not available_agents:
            return None
        
        # 选择工作负载最低的智能体
        best_agent = min(available_agents, key=lambda aid: self.agent_workloads[aid])
        return best_agent
    
    def _estimate_task_duration(self, task: CoordinationTask) -> float:
        """估算任务持续时间"""
        # 简单的持续时间估算
        base_duration = 1.0
        priority_factor = task.priority * 0.1
        resource_factor = sum(task.resource_requirements.values()) * 0.5
        
        return base_duration + priority_factor + resource_factor
    
    def complete_task(self, task_id: str, result: Any = None):
        """完成任务"""
        if task_id in self.scheduled_tasks:
            task = self.scheduled_tasks[task_id]
            task.status = "completed"
            task.completed_at = datetime.now()
            task.result = result
            
            # 减少工作负载
            for agent_id in task.assigned_agents:
                self.agent_workloads[agent_id] = max(0, self.agent_workloads[agent_id] - 0.1)
            
            # 移动到完成列表
            self.completed_tasks.append(task)
            del self.scheduled_tasks[task_id]
            
            logger.info(f"Task {task_id} completed")
    
    def fail_task(self, task_id: str, error: str = None):
        """任务失败"""
        if task_id in self.scheduled_tasks:
            task = self.scheduled_tasks[task_id]
            task.status = "failed"
            task.completed_at = datetime.now()
            
            # 减少工作负载
            for agent_id in task.assigned_agents:
                self.agent_workloads[agent_id] = max(0, self.agent_workloads[agent_id] - 0.1)
            
            # 移动到完成列表
            self.completed_tasks.append(task)
            del self.scheduled_tasks[task_id]
            
            logger.info(f"Task {task_id} failed: {error}")
    
    def get_scheduling_statistics(self) -> Dict[str, Any]:
        """获取调度统计"""
        return {
            "queue_size": len(self.task_queue),
            "scheduled_count": len(self.scheduled_tasks),
            "completed_count": len(self.completed_tasks),
            "policy": self.policy.value,
            "agent_workloads": dict(self.agent_workloads),
            "completion_rate": len(self.completed_tasks) / (len(self.completed_tasks) + len(self.scheduled_tasks)) if (len(self.completed_tasks) + len(self.scheduled_tasks)) > 0 else 0
        }

class ConflictResolver:
    """冲突解决器"""
    
    def __init__(self, strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.PRIORITY_BASED):
        self.strategy = strategy
        self.conflict_history: List[Dict[str, Any]] = []
        self.agent_priorities: Dict[str, int] = {}
        self.resource_locks: Dict[str, str] = {}  # resource -> agent_id
    
    def set_agent_priority(self, agent_id: str, priority: int):
        """设置智能体优先级"""
        self.agent_priorities[agent_id] = priority
    
    async def resolve_conflict(self, conflict_type: str, conflicting_agents: List[str], 
                             resource: str = None) -> str:
        """解决冲突"""
        logger.info(f"Resolving {conflict_type} conflict between agents: {conflicting_agents}")
        
        if self.strategy == ConflictResolutionStrategy.FIRST_COME_FIRST_SERVED:
            winner = self._first_come_first_served(conflicting_agents)
        elif self.strategy == ConflictResolutionStrategy.PRIORITY_BASED:
            winner = self._priority_based(conflicting_agents)
        elif self.strategy == ConflictResolutionStrategy.NEGOTIATION:
            winner = await self._negotiation_based(conflicting_agents)
        elif self.strategy == ConflictResolutionStrategy.AUCTION:
            winner = await self._auction_based(conflicting_agents)
        elif self.strategy == ConflictResolutionStrategy.CONSENSUS:
            winner = await self._consensus_based(conflicting_agents)
        elif self.strategy == ConflictResolutionStrategy.RANDOM:
            winner = self._random_selection(conflicting_agents)
        else:
            winner = conflicting_agents[0]
        
        # 记录冲突解决历史
        self.conflict_history.append({
            "timestamp": datetime.now(),
            "conflict_type": conflict_type,
            "conflicting_agents": conflicting_agents,
            "winner": winner,
            "strategy": self.strategy.value,
            "resource": resource
        })
        
        # 更新资源锁
        if resource:
            self.resource_locks[resource] = winner
        
        logger.info(f"Conflict resolved: {winner} wins")
        return winner
    
    def _first_come_first_served(self, agents: List[str]) -> str:
        """先到先服务"""
        return agents[0]
    
    def _priority_based(self, agents: List[str]) -> str:
        """基于优先级"""
        return max(agents, key=lambda aid: self.agent_priorities.get(aid, 1))
    
    async def _negotiation_based(self, agents: List[str]) -> str:
        """基于协商"""
        # 模拟协商过程
        offers = {}
        for agent_id in agents:
            # 模拟智能体出价
            offer = random.uniform(0, 1)
            offers[agent_id] = offer
        
        # 选择出价最高的智能体
        return max(offers.keys(), key=lambda k: offers[k])
    
    async def _auction_based(self, agents: List[str]) -> str:
        """基于拍卖"""
        # 模拟拍卖过程
        bids = {}
        for agent_id in agents:
            bid = random.uniform(0, 1)
            bids[agent_id] = bid
        
        # 选择出价最高的智能体
        return max(bids.keys(), key=lambda k: bids[k])
    
    async def _consensus_based(self, agents: List[str]) -> str:
        """基于共识"""
        # 模拟共识过程
        votes = {}
        for agent_id in agents:
            votes[agent_id] = random.choice([True, False])
        
        # 选择得票最多的智能体
        vote_counts = defaultdict(int)
        for agent_id, vote in votes.items():
            if vote:
                vote_counts[agent_id] += 1
        
        if vote_counts:
            return max(vote_counts.keys(), key=lambda k: vote_counts[k])
        else:
            return agents[0]
    
    def _random_selection(self, agents: List[str]) -> str:
        """随机选择"""
        return random.choice(agents)
    
    def release_resource(self, resource: str, agent_id: str):
        """释放资源"""
        if resource in self.resource_locks and self.resource_locks[resource] == agent_id:
            del self.resource_locks[resource]
            logger.info(f"Resource {resource} released by agent {agent_id}")
    
    def get_conflict_statistics(self) -> Dict[str, Any]:
        """获取冲突统计"""
        if not self.conflict_history:
            return {}
        
        total_conflicts = len(self.conflict_history)
        strategy_counts = defaultdict(int)
        agent_wins = defaultdict(int)
        
        for conflict in self.conflict_history:
            strategy_counts[conflict["strategy"]] += 1
            agent_wins[conflict["winner"]] += 1
        
        return {
            "total_conflicts": total_conflicts,
            "strategy_distribution": dict(strategy_counts),
            "agent_win_counts": dict(agent_wins),
            "locked_resources": len(self.resource_locks)
        }

class CoordinationEngine:
    """协调引擎"""
    
    def __init__(self, coordination_type: CoordinationType = CoordinationType.CENTRALIZED):
        self.coordination_type = coordination_type
        self.task_scheduler = TaskScheduler()
        self.conflict_resolver = ConflictResolver()
        self.agent_roles: Dict[str, AgentRole] = {}
        self.coordination_events: List[CoordinationEvent] = []
        self.active_coordinations: Dict[str, Dict[str, Any]] = {}
        self.coordination_history: List[Dict[str, Any]] = []
        self.running = False
    
    def register_agent_role(self, agent_id: str, role: str, capabilities: Set[str], 
                          authority_level: int = 1):
        """注册智能体角色"""
        agent_role = AgentRole(
            agent_id=agent_id,
            role=role,
            capabilities=capabilities,
            authority_level=authority_level
        )
        
        self.agent_roles[agent_id] = agent_role
        self.conflict_resolver.set_agent_priority(agent_id, authority_level)
        
        logger.info(f"Agent {agent_id} registered with role {role}")
    
    def unregister_agent_role(self, agent_id: str):
        """注销智能体角色"""
        if agent_id in self.agent_roles:
            del self.agent_roles[agent_id]
            logger.info(f"Agent {agent_id} unregistered")
    
    async def coordinate_task(self, task: CoordinationTask) -> str:
        """协调任务"""
        logger.info(f"Coordinating task {task.id}")
        
        # 根据协调类型处理任务
        if self.coordination_type == CoordinationType.CENTRALIZED:
            return await self._centralized_coordination(task)
        elif self.coordination_type == CoordinationType.DISTRIBUTED:
            return await self._distributed_coordination(task)
        elif self.coordination_type == CoordinationType.HYBRID:
            return await self._hybrid_coordination(task)
        elif self.coordination_type == CoordinationType.HIERARCHICAL:
            return await self._hierarchical_coordination(task)
        elif self.coordination_type == CoordinationType.PEER_TO_PEER:
            return await self._peer_to_peer_coordination(task)
        else:
            return await self._centralized_coordination(task)
    
    async def _centralized_coordination(self, task: CoordinationTask) -> str:
        """集中式协调"""
        # 添加到调度器
        self.task_scheduler.add_task(task)
        
        # 获取可用智能体
        available_agents = [aid for aid, role in self.agent_roles.items() 
                          if task.name in role.capabilities or not task.name]
        
        # 调度任务
        scheduled_tasks = self.task_scheduler.schedule_tasks(available_agents)
        
        if scheduled_tasks:
            scheduled_task = scheduled_tasks[0]
            return scheduled_task.id
        else:
            return task.id
    
    async def _distributed_coordination(self, task: CoordinationTask) -> str:
        """分布式协调"""
        # 广播任务给所有智能体
        available_agents = list(self.agent_roles.keys())
        
        # 模拟智能体自主选择
        interested_agents = []
        for agent_id in available_agents:
            if random.random() > 0.5:  # 模拟智能体兴趣
                interested_agents.append(agent_id)
        
        if interested_agents:
            # 解决冲突
            winner = await self.conflict_resolver.resolve_conflict(
                "task_assignment", interested_agents
            )
            
            task.assigned_agents = [winner]
            task.status = "assigned"
            task.started_at = datetime.now()
            
            return task.id
        
        return task.id
    
    async def _hybrid_coordination(self, task: CoordinationTask) -> str:
        """混合式协调"""
        # 结合集中式和分布式策略
        if task.priority >= 5:
            return await self._centralized_coordination(task)
        else:
            return await self._distributed_coordination(task)
    
    async def _hierarchical_coordination(self, task: CoordinationTask) -> str:
        """层次化协调"""
        # 根据任务优先级选择协调层级
        if task.priority >= 5:
            # 高层协调
            return await self._centralized_coordination(task)
        else:
            # 低层协调
            return await self._distributed_coordination(task)
    
    async def _peer_to_peer_coordination(self, task: CoordinationTask) -> str:
        """点对点协调"""
        # 模拟点对点协商
        available_agents = list(self.agent_roles.keys())
        
        if len(available_agents) >= 2:
            # 选择两个智能体进行协商
            agent1, agent2 = random.sample(available_agents, 2)
            
            # 模拟协商过程
            winner = await self.conflict_resolver.resolve_conflict(
                "peer_negotiation", [agent1, agent2]
            )
            
            task.assigned_agents = [winner]
            task.status = "assigned"
            task.started_at = datetime.now()
        
        return task.id
    
    async def handle_coordination_event(self, event: CoordinationEvent) -> Any:
        """处理协调事件"""
        logger.info(f"Handling coordination event {event.id}")
        
        # 根据事件类型处理
        if event.event_type == "task_completion":
            return await self._handle_task_completion(event)
        elif event.event_type == "task_failure":
            return await self._handle_task_failure(event)
        elif event.event_type == "resource_conflict":
            return await self._handle_resource_conflict(event)
        elif event.event_type == "agent_failure":
            return await self._handle_agent_failure(event)
        else:
            return await self._handle_generic_event(event)
    
    async def _handle_task_completion(self, event: CoordinationEvent) -> Any:
        """处理任务完成事件"""
        task_id = event.content.get("task_id")
        result = event.content.get("result")
        
        if task_id:
            self.task_scheduler.complete_task(task_id, result)
        
        return f"Task {task_id} completion handled"
    
    async def _handle_task_failure(self, event: CoordinationEvent) -> Any:
        """处理任务失败事件"""
        task_id = event.content.get("task_id")
        error = event.content.get("error")
        
        if task_id:
            self.task_scheduler.fail_task(task_id, error)
        
        return f"Task {task_id} failure handled"
    
    async def _handle_resource_conflict(self, event: CoordinationEvent) -> Any:
        """处理资源冲突事件"""
        resource = event.content.get("resource")
        conflicting_agents = event.content.get("conflicting_agents", [])
        
        if conflicting_agents:
            winner = await self.conflict_resolver.resolve_conflict(
                "resource_conflict", conflicting_agents, resource
            )
            return f"Resource {resource} conflict resolved: {winner} wins"
        
        return "No conflicting agents found"
    
    async def _handle_agent_failure(self, event: CoordinationEvent) -> Any:
        """处理智能体失败事件"""
        failed_agent = event.content.get("agent_id")
        
        if failed_agent in self.agent_roles:
            # 重新分配该智能体的任务
            self.unregister_agent_role(failed_agent)
            return f"Agent {failed_agent} failure handled"
        
        return f"Agent {failed_agent} not found"
    
    async def _handle_generic_event(self, event: CoordinationEvent) -> Any:
        """处理通用事件"""
        return f"Generic event {event.id} handled"
    
    def get_coordination_statistics(self) -> Dict[str, Any]:
        """获取协调统计"""
        scheduling_stats = self.task_scheduler.get_scheduling_statistics()
        conflict_stats = self.conflict_resolver.get_conflict_statistics()
        
        return {
            "coordination_type": self.coordination_type.value,
            "registered_agents": len(self.agent_roles),
            "total_events": len(self.coordination_events),
            "active_coordinations": len(self.active_coordinations),
            "scheduling_statistics": scheduling_stats,
            "conflict_statistics": conflict_stats
        }

# 使用示例
async def main():
    """主函数示例"""
    # 创建协调引擎
    engine = CoordinationEngine(coordination_type=CoordinationType.CENTRALIZED)
    
    # 注册智能体角色
    engine.register_agent_role("agent_1", "researcher", {"research", "analysis"}, 3)
    engine.register_agent_role("agent_2", "analyst", {"analysis", "reporting"}, 2)
    engine.register_agent_role("agent_3", "coordinator", {"coordination", "management"}, 4)
    
    # 创建协调任务
    task1 = CoordinationTask(
        name="research",
        description="Conduct research on AI trends",
        priority=5,
        resource_requirements={"cpu": 0.5, "memory": 0.3}
    )
    
    task2 = CoordinationTask(
        name="analysis",
        description="Analyze research data",
        priority=3,
        resource_requirements={"cpu": 0.3, "memory": 0.2}
    )
    
    task3 = CoordinationTask(
        name="coordination",
        description="Coordinate team activities",
        priority=4,
        resource_requirements={"cpu": 0.2, "memory": 0.1}
    )
    
    # 协调任务
    print("Coordinating tasks:")
    task1_id = await engine.coordinate_task(task1)
    print(f"Task 1 coordinated: {task1_id}")
    
    task2_id = await engine.coordinate_task(task2)
    print(f"Task 2 coordinated: {task2_id}")
    
    task3_id = await engine.coordinate_task(task3)
    print(f"Task 3 coordinated: {task3_id}")
    
    # 处理协调事件
    print("\nHandling coordination events:")
    
    # 任务完成事件
    completion_event = CoordinationEvent(
        event_type="task_completion",
        source_agent="agent_1",
        content={"task_id": task1_id, "result": "Research completed"}
    )
    
    result = await engine.handle_coordination_event(completion_event)
    print(f"Completion event handled: {result}")
    
    # 资源冲突事件
    conflict_event = CoordinationEvent(
        event_type="resource_conflict",
        source_agent="agent_2",
        content={
            "resource": "database",
            "conflicting_agents": ["agent_1", "agent_2"]
        }
    )
    
    result = await engine.handle_coordination_event(conflict_event)
    print(f"Conflict event handled: {result}")
    
    # 获取协调统计
    stats = engine.get_coordination_statistics()
    print(f"\nCoordination Statistics: {json.dumps(stats, indent=2, default=str)}")

if __name__ == "__main__":
    asyncio.run(main())
