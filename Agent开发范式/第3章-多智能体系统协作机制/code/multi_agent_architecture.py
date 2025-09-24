# multi_agent_architecture.py
"""
多智能体系统架构实现
提供完整的多智能体系统架构设计和实现
"""

import asyncio
import logging
import uuid
import time
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import json
import threading
from concurrent.futures import ThreadPoolExecutor

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(Enum):
    """智能体状态枚举"""
    IDLE = "idle"
    BUSY = "busy"
    COMMUNICATING = "communicating"
    COORDINATING = "coordinating"
    ERROR = "error"
    OFFLINE = "offline"

class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class CoordinationType(Enum):
    """协调类型枚举"""
    CENTRALIZED = "centralized"
    DISTRIBUTED = "distributed"
    HYBRID = "hybrid"
    HIERARCHICAL = "hierarchical"

@dataclass
class AgentCapability:
    """智能体能力描述"""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    performance_score: float = 1.0
    resource_requirements: Dict[str, float] = field(default_factory=dict)

@dataclass
class Task:
    """任务数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    priority: int = 1
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    required_capabilities: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    deadline: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Message:
    """消息数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    receiver: str = ""
    message_type: str = ""
    content: Any = None
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 1
    reply_to: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentInfo:
    """智能体信息"""
    id: str
    name: str
    state: AgentState
    capabilities: List[AgentCapability]
    current_load: float = 0.0
    max_load: float = 1.0
    last_heartbeat: datetime = field(default_factory=datetime.now)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class CommunicationProtocol(ABC):
    """通信协议抽象基类"""
    
    @abstractmethod
    async def send_message(self, message: Message) -> bool:
        """发送消息"""
        pass
    
    @abstractmethod
    async def receive_message(self, agent_id: str) -> List[Message]:
        """接收消息"""
        pass
    
    @abstractmethod
    async def broadcast_message(self, sender: str, message_type: str, content: Any) -> bool:
        """广播消息"""
        pass

class InMemoryCommunicationProtocol(CommunicationProtocol):
    """内存通信协议实现"""
    
    def __init__(self):
        self.message_queues: Dict[str, List[Message]] = {}
        self.broadcast_listeners: Set[str] = set()
        self.lock = asyncio.Lock()
    
    async def send_message(self, message: Message) -> bool:
        """发送消息"""
        async with self.lock:
            if message.receiver not in self.message_queues:
                self.message_queues[message.receiver] = []
            
            self.message_queues[message.receiver].append(message)
            logger.debug(f"Message sent from {message.sender} to {message.receiver}")
            return True
    
    async def receive_message(self, agent_id: str) -> List[Message]:
        """接收消息"""
        async with self.lock:
            messages = self.message_queues.get(agent_id, [])
            self.message_queues[agent_id] = []
            return messages
    
    async def broadcast_message(self, sender: str, message_type: str, content: Any) -> bool:
        """广播消息"""
        async with self.lock:
            message = Message(
                sender=sender,
                receiver="broadcast",
                message_type=message_type,
                content=content
            )
            
            for listener in self.broadcast_listeners:
                if listener != sender:
                    if listener not in self.message_queues:
                        self.message_queues[listener] = []
                    self.message_queues[listener].append(message)
            
            logger.debug(f"Broadcast message sent from {sender}")
            return True
    
    async def register_broadcast_listener(self, agent_id: str):
        """注册广播监听器"""
        async with self.lock:
            self.broadcast_listeners.add(agent_id)
    
    async def unregister_broadcast_listener(self, agent_id: str):
        """取消注册广播监听器"""
        async with self.lock:
            self.broadcast_listeners.discard(agent_id)

class TaskAllocator:
    """任务分配器"""
    
    def __init__(self):
        self.allocation_strategies = {
            "round_robin": self._round_robin_allocation,
            "load_balanced": self._load_balanced_allocation,
            "capability_based": self._capability_based_allocation,
            "performance_based": self._performance_based_allocation
        }
    
    async def allocate_tasks(self, tasks: List[Task], agents: Dict[str, AgentInfo], 
                           strategy: str = "load_balanced") -> Dict[str, List[Task]]:
        """分配任务给智能体"""
        if strategy not in self.allocation_strategies:
            strategy = "load_balanced"
        
        allocator = self.allocation_strategies[strategy]
        return await allocator(tasks, agents)
    
    async def _round_robin_allocation(self, tasks: List[Task], agents: Dict[str, AgentInfo]) -> Dict[str, List[Task]]:
        """轮询分配策略"""
        allocation = {agent_id: [] for agent_id in agents.keys()}
        agent_ids = list(agents.keys())
        
        for i, task in enumerate(tasks):
            agent_id = agent_ids[i % len(agent_ids)]
            allocation[agent_id].append(task)
        
        return allocation
    
    async def _load_balanced_allocation(self, tasks: List[Task], agents: Dict[str, AgentInfo]) -> Dict[str, List[Task]]:
        """负载均衡分配策略"""
        allocation = {agent_id: [] for agent_id in agents.keys()}
        
        for task in tasks:
            # 选择负载最低的智能体
            best_agent = min(agents.keys(), key=lambda aid: agents[aid].current_load)
            allocation[best_agent].append(task)
            agents[best_agent].current_load += 0.1  # 增加负载
        
        return allocation
    
    async def _capability_based_allocation(self, tasks: List[Task], agents: Dict[str, AgentInfo]) -> Dict[str, List[Task]]:
        """基于能力的分配策略"""
        allocation = {agent_id: [] for agent_id in agents.keys()}
        
        for task in tasks:
            best_agent = None
            best_score = -1
            
            for agent_id, agent_info in agents.items():
                if agent_info.current_load >= agent_info.max_load:
                    continue
                
                # 计算能力匹配度
                capability_score = self._calculate_capability_score(task, agent_info)
                if capability_score > best_score:
                    best_score = capability_score
                    best_agent = agent_id
            
            if best_agent:
                allocation[best_agent].append(task)
                agents[best_agent].current_load += 0.1
        
        return allocation
    
    async def _performance_based_allocation(self, tasks: List[Task], agents: Dict[str, AgentInfo]) -> Dict[str, List[Task]]:
        """基于性能的分配策略"""
        allocation = {agent_id: [] for agent_id in agents.keys()}
        
        for task in tasks:
            best_agent = None
            best_performance = -1
            
            for agent_id, agent_info in agents.items():
                if agent_info.current_load >= agent_info.max_load:
                    continue
                
                # 计算综合性能得分
                performance_score = self._calculate_performance_score(agent_info)
                if performance_score > best_performance:
                    best_performance = performance_score
                    best_agent = agent_id
            
            if best_agent:
                allocation[best_agent].append(task)
                agents[best_agent].current_load += 0.1
        
        return allocation
    
    def _calculate_capability_score(self, task: Task, agent_info: AgentInfo) -> float:
        """计算能力匹配度"""
        if not task.required_capabilities:
            return 1.0
        
        agent_capabilities = {cap.name for cap in agent_info.capabilities}
        required_capabilities = set(task.required_capabilities)
        
        if not required_capabilities.issubset(agent_capabilities):
            return 0.0
        
        # 计算能力匹配度
        matching_capabilities = required_capabilities.intersection(agent_capabilities)
        return len(matching_capabilities) / len(required_capabilities)
    
    def _calculate_performance_score(self, agent_info: AgentInfo) -> float:
        """计算综合性能得分"""
        if not agent_info.performance_metrics:
            return 1.0
        
        # 计算平均性能得分
        scores = list(agent_info.performance_metrics.values())
        return sum(scores) / len(scores) if scores else 1.0

class CollaborationEngine:
    """协作引擎"""
    
    def __init__(self, communication_protocol: CommunicationProtocol):
        self.communication_protocol = communication_protocol
        self.collaboration_strategies = {
            "sequential": self._sequential_collaboration,
            "parallel": self._parallel_collaboration,
            "pipeline": self._pipeline_collaboration,
            "consensus": self._consensus_collaboration
        }
    
    async def execute_collaboration(self, task_allocation: Dict[str, List[Task]], 
                                  strategy: str = "parallel") -> Dict[str, Any]:
        """执行协作"""
        if strategy not in self.collaboration_strategies:
            strategy = "parallel"
        
        collaborator = self.collaboration_strategies[strategy]
        return await collaborator(task_allocation)
    
    async def _sequential_collaboration(self, task_allocation: Dict[str, List[Task]]) -> Dict[str, Any]:
        """顺序协作策略"""
        results = {}
        
        for agent_id, tasks in task_allocation.items():
            if not tasks:
                continue
            
            agent_results = []
            for task in tasks:
                # 模拟任务执行
                result = await self._execute_task(task)
                agent_results.append(result)
            
            results[agent_id] = agent_results
        
        return results
    
    async def _parallel_collaboration(self, task_allocation: Dict[str, List[Task]]) -> Dict[str, Any]:
        """并行协作策略"""
        results = {}
        
        # 创建并行任务
        tasks = []
        for agent_id, agent_tasks in task_allocation.items():
            if agent_tasks:
                task = asyncio.create_task(self._execute_agent_tasks(agent_id, agent_tasks))
                tasks.append((agent_id, task))
        
        # 等待所有任务完成
        for agent_id, task in tasks:
            results[agent_id] = await task
        
        return results
    
    async def _pipeline_collaboration(self, task_allocation: Dict[str, List[Task]]) -> Dict[str, Any]:
        """流水线协作策略"""
        results = {}
        
        # 按优先级排序任务
        all_tasks = []
        for agent_id, tasks in task_allocation.items():
            for task in tasks:
                all_tasks.append((agent_id, task))
        
        all_tasks.sort(key=lambda x: x[1].priority, reverse=True)
        
        # 流水线执行
        for agent_id, task in all_tasks:
            result = await self._execute_task(task)
            if agent_id not in results:
                results[agent_id] = []
            results[agent_id].append(result)
        
        return results
    
    async def _consensus_collaboration(self, task_allocation: Dict[str, List[Task]]) -> Dict[str, Any]:
        """共识协作策略"""
        results = {}
        
        # 收集所有智能体的意见
        opinions = {}
        for agent_id, tasks in task_allocation.items():
            if tasks:
                opinion = await self._get_agent_opinion(agent_id, tasks)
                opinions[agent_id] = opinion
        
        # 达成共识
        consensus = await self._reach_consensus(opinions)
        
        # 执行共识结果
        for agent_id, tasks in task_allocation.items():
            if tasks:
                result = await self._execute_consensus_task(agent_id, tasks, consensus)
                results[agent_id] = result
        
        return results
    
    async def _execute_agent_tasks(self, agent_id: str, tasks: List[Task]) -> List[Any]:
        """执行智能体的任务"""
        results = []
        for task in tasks:
            result = await self._execute_task(task)
            results.append(result)
        return results
    
    async def _execute_task(self, task: Task) -> Any:
        """执行单个任务"""
        # 模拟任务执行
        await asyncio.sleep(0.1)
        return f"Task {task.id} completed by agent"
    
    async def _get_agent_opinion(self, agent_id: str, tasks: List[Task]) -> Any:
        """获取智能体意见"""
        # 模拟智能体意见
        return f"Agent {agent_id} opinion on {len(tasks)} tasks"
    
    async def _reach_consensus(self, opinions: Dict[str, Any]) -> Any:
        """达成共识"""
        # 模拟共识达成
        return "Consensus reached"
    
    async def _execute_consensus_task(self, agent_id: str, tasks: List[Task], consensus: Any) -> Any:
        """执行共识任务"""
        # 模拟基于共识的任务执行
        return f"Consensus task executed by {agent_id}"

class Coordinator:
    """协调器"""
    
    def __init__(self, coordination_type: CoordinationType = CoordinationType.CENTRALIZED):
        self.coordination_type = coordination_type
        self.agents: Dict[str, AgentInfo] = {}
        self.tasks: Dict[str, Task] = {}
        self.task_allocator = TaskAllocator()
        self.collaboration_engine: Optional[CollaborationEngine] = None
        self.coordination_history: List[Dict[str, Any]] = []
    
    def set_collaboration_engine(self, engine: CollaborationEngine):
        """设置协作引擎"""
        self.collaboration_engine = engine
    
    async def register_agent(self, agent_info: AgentInfo):
        """注册智能体"""
        self.agents[agent_info.id] = agent_info
        logger.info(f"Agent {agent_info.id} registered")
    
    async def unregister_agent(self, agent_id: str):
        """注销智能体"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Agent {agent_id} unregistered")
    
    async def submit_task(self, task: Task) -> str:
        """提交任务"""
        self.tasks[task.id] = task
        logger.info(f"Task {task.id} submitted")
        
        # 根据协调类型处理任务
        if self.coordination_type == CoordinationType.CENTRALIZED:
            await self._handle_centralized_task(task)
        elif self.coordination_type == CoordinationType.DISTRIBUTED:
            await self._handle_distributed_task(task)
        elif self.coordination_type == CoordinationType.HYBRID:
            await self._handle_hybrid_task(task)
        elif self.coordination_type == CoordinationType.HIERARCHICAL:
            await self._handle_hierarchical_task(task)
        
        return task.id
    
    async def _handle_centralized_task(self, task: Task):
        """处理集中式任务"""
        # 任务分解
        subtasks = await self._decompose_task(task)
        
        # 任务分配
        allocation = await self.task_allocator.allocate_tasks(subtasks, self.agents)
        
        # 执行协作
        if self.collaboration_engine:
            results = await self.collaboration_engine.execute_collaboration(allocation)
            task.result = results
            task.status = TaskStatus.COMPLETED
    
    async def _handle_distributed_task(self, task: Task):
        """处理分布式任务"""
        # 广播任务给所有智能体
        for agent_id in self.agents:
            message = Message(
                sender="coordinator",
                receiver=agent_id,
                message_type="task_broadcast",
                content=task
            )
            if self.collaboration_engine:
                await self.collaboration_engine.communication_protocol.send_message(message)
    
    async def _handle_hybrid_task(self, task: Task):
        """处理混合式任务"""
        # 结合集中式和分布式策略
        await self._handle_centralized_task(task)
        await self._handle_distributed_task(task)
    
    async def _handle_hierarchical_task(self, task: Task):
        """处理层次式任务"""
        # 分层处理任务
        if task.priority >= 5:
            await self._handle_centralized_task(task)
        else:
            await self._handle_distributed_task(task)
    
    async def _decompose_task(self, task: Task) -> List[Task]:
        """分解任务"""
        # 简单的任务分解逻辑
        subtasks = []
        for i in range(3):  # 分解为3个子任务
            subtask = Task(
                name=f"{task.name}_subtask_{i}",
                description=f"Subtask {i} of {task.name}",
                priority=task.priority,
                required_capabilities=task.required_capabilities,
                resource_requirements=task.resource_requirements
            )
            subtasks.append(subtask)
        
        return subtasks
    
    async def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "coordination_type": self.coordination_type.value,
            "total_agents": len(self.agents),
            "total_tasks": len(self.tasks),
            "active_tasks": len([t for t in self.tasks.values() if t.status == TaskStatus.IN_PROGRESS]),
            "completed_tasks": len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]),
            "failed_tasks": len([t for t in self.tasks.values() if t.status == TaskStatus.FAILED]),
            "agent_states": {aid: info.state.value for aid, info in self.agents.items()},
            "system_load": sum(info.current_load for info in self.agents.values()) / len(self.agents) if self.agents else 0
        }

class MultiAgentSystem:
    """多智能体系统核心类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agents: Dict[str, AgentInfo] = {}
        self.coordinator = Coordinator(
            coordination_type=CoordinationType(config.get("coordination_type", "centralized"))
        )
        self.communication_protocol = InMemoryCommunicationProtocol()
        self.collaboration_engine = CollaborationEngine(self.communication_protocol)
        self.coordinator.set_collaboration_engine(self.collaboration_engine)
        self.running = False
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None
    
    async def start(self):
        """启动系统"""
        self.running = True
        self.event_loop = asyncio.get_event_loop()
        logger.info("Multi-agent system started")
    
    async def stop(self):
        """停止系统"""
        self.running = False
        logger.info("Multi-agent system stopped")
    
    async def add_agent(self, agent_info: AgentInfo):
        """添加智能体"""
        self.agents[agent_info.id] = agent_info
        await self.coordinator.register_agent(agent_info)
        await self.communication_protocol.register_broadcast_listener(agent_info.id)
        logger.info(f"Agent {agent_info.id} added to system")
    
    async def remove_agent(self, agent_id: str):
        """移除智能体"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            await self.coordinator.unregister_agent(agent_id)
            await self.communication_protocol.unregister_broadcast_listener(agent_id)
            logger.info(f"Agent {agent_id} removed from system")
    
    async def submit_task(self, task: Task) -> str:
        """提交任务"""
        return await self.coordinator.submit_task(task)
    
    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """获取任务状态"""
        if task_id in self.coordinator.tasks:
            return self.coordinator.tasks[task_id].status
        return None
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统指标"""
        status = await self.coordinator.get_system_status()
        status.update({
            "communication_protocol": "InMemoryCommunicationProtocol",
            "collaboration_strategies": list(self.collaboration_engine.collaboration_strategies.keys()),
            "allocation_strategies": list(self.task_allocator.allocation_strategies.keys())
        })
        return status
    
    async def run_heartbeat_monitor(self):
        """运行心跳监控"""
        while self.running:
            try:
                current_time = datetime.now()
                for agent_id, agent_info in self.agents.items():
                    # 检查心跳超时
                    time_diff = (current_time - agent_info.last_heartbeat).total_seconds()
                    if time_diff > 30:  # 30秒超时
                        agent_info.state = AgentState.OFFLINE
                        logger.warning(f"Agent {agent_id} is offline")
                
                await asyncio.sleep(5)  # 每5秒检查一次
            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")
                await asyncio.sleep(5)

# 使用示例
async def main():
    """主函数示例"""
    # 创建系统配置
    config = {
        "coordination_type": "centralized",
        "max_agents": 10,
        "heartbeat_interval": 5
    }
    
    # 创建多智能体系统
    system = MultiAgentSystem(config)
    await system.start()
    
    # 创建智能体
    agent1 = AgentInfo(
        id="agent_1",
        name="Research Agent",
        state=AgentState.IDLE,
        capabilities=[
            AgentCapability("research", "Conduct research", {"domain": "AI"}),
            AgentCapability("analysis", "Analyze data", {"accuracy": 0.95})
        ]
    )
    
    agent2 = AgentInfo(
        id="agent_2",
        name="Analysis Agent",
        state=AgentState.IDLE,
        capabilities=[
            AgentCapability("analysis", "Analyze data", {"accuracy": 0.90}),
            AgentCapability("reporting", "Generate reports", {"format": "PDF"})
        ]
    )
    
    # 添加智能体
    await system.add_agent(agent1)
    await system.add_agent(agent2)
    
    # 创建任务
    task = Task(
        name="Research Analysis",
        description="Conduct research and analysis on AI trends",
        priority=5,
        required_capabilities=["research", "analysis"],
        resource_requirements={"cpu": 0.5, "memory": 0.3}
    )
    
    # 提交任务
    task_id = await system.submit_task(task)
    print(f"Task submitted: {task_id}")
    
    # 等待任务完成
    await asyncio.sleep(2)
    
    # 获取系统状态
    metrics = await system.get_system_metrics()
    print(f"System metrics: {json.dumps(metrics, indent=2, default=str)}")
    
    # 停止系统
    await system.stop()

if __name__ == "__main__":
    asyncio.run(main())
