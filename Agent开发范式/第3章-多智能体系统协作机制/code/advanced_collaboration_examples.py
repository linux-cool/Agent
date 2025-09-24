#!/usr/bin/env python3
"""
高级多智能体协作示例 - 展示复杂协作场景的实现

本模块提供了多智能体系统的高级协作示例，包括：
- 复杂任务分解与分配
- 智能体间的协商与共识
- 动态负载均衡
- 故障恢复与容错
- 性能优化策略
"""

import asyncio
import logging
import json
import time
import random
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import heapq
from collections import defaultdict, deque

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(Enum):
    """智能体状态"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class TaskPriority(Enum):
    """任务优先级"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4

class MessageType(Enum):
    """消息类型"""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    HEARTBEAT = "heartbeat"
    STATUS_UPDATE = "status_update"
    NEGOTIATION = "negotiation"
    CONSENSUS = "consensus"
    ERROR = "error"
    BROADCAST = "broadcast"

@dataclass
class Task:
    """任务定义"""
    id: str
    name: str
    description: str
    priority: TaskPriority
    estimated_duration: int  # 秒
    required_skills: List[str]
    dependencies: List[str] = field(default_factory=list)
    deadline: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"
    assigned_agent: Optional[str] = None
    progress: float = 0.0
    result: Optional[Any] = None
    error: Optional[str] = None

@dataclass
class Agent:
    """智能体定义"""
    id: str
    name: str
    skills: List[str]
    capacity: int = 10  # 最大并发任务数
    current_tasks: List[str] = field(default_factory=list)
    state: AgentState = AgentState.IDLE
    performance_score: float = 1.0
    last_heartbeat: datetime = field(default_factory=datetime.now)
    location: str = "default"
    cost_per_hour: float = 1.0

@dataclass
class Message:
    """消息定义"""
    id: str
    sender: str
    receiver: str
    message_type: MessageType
    content: Any
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 1
    ttl: int = 300  # 生存时间（秒）

class AdvancedTaskAllocator:
    """高级任务分配器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.task_queue = []
        self.allocation_history = []
        self.performance_metrics = defaultdict(list)
        
    def register_agent(self, agent: Agent) -> bool:
        """注册智能体"""
        try:
            self.agents[agent.id] = agent
            logger.info(f"Agent {agent.name} registered successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to register agent {agent.name}: {e}")
            return False
    
    def submit_task(self, task: Task) -> bool:
        """提交任务"""
        try:
            self.tasks[task.id] = task
            heapq.heappush(self.task_queue, (-task.priority.value, task.created_at, task))
            logger.info(f"Task {task.name} submitted successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to submit task {task.name}: {e}")
            return False
    
    async def allocate_tasks(self) -> Dict[str, Any]:
        """分配任务"""
        try:
            allocations = []
            unallocated_tasks = []
            
            # 按优先级处理任务
            while self.task_queue:
                priority, created_at, task = heapq.heappop(self.task_queue)
                
                # 检查任务依赖
                if not self._check_dependencies(task):
                    heapq.heappush(self.task_queue, (priority, created_at, task))
                    break
                
                # 寻找最佳智能体
                best_agent = await self._find_best_agent(task)
                
                if best_agent:
                    # 分配任务
                    allocation = await self._allocate_task(task, best_agent)
                    allocations.append(allocation)
                else:
                    unallocated_tasks.append(task)
            
            # 记录分配历史
            self.allocation_history.append({
                "timestamp": datetime.now(),
                "allocations": len(allocations),
                "unallocated": len(unallocated_tasks)
            })
            
            return {
                "allocations": allocations,
                "unallocated_tasks": unallocated_tasks,
                "total_agents": len(self.agents),
                "total_tasks": len(self.tasks)
            }
            
        except Exception as e:
            logger.error(f"Task allocation failed: {e}")
            return {"error": str(e)}
    
    def _check_dependencies(self, task: Task) -> bool:
        """检查任务依赖"""
        for dep_id in task.dependencies:
            if dep_id not in self.tasks:
                return False
            if self.tasks[dep_id].status != "completed":
                return False
        return True
    
    async def _find_best_agent(self, task: Task) -> Optional[Agent]:
        """寻找最佳智能体"""
        try:
            candidates = []
            
            for agent in self.agents.values():
                # 检查智能体状态
                if agent.state != AgentState.IDLE:
                    continue
                
                # 检查技能匹配
                if not self._check_skills_match(task, agent):
                    continue
                
                # 检查容量
                if len(agent.current_tasks) >= agent.capacity:
                    continue
                
                # 计算匹配分数
                score = self._calculate_match_score(task, agent)
                candidates.append((score, agent))
            
            if not candidates:
                return None
            
            # 返回分数最高的智能体
            candidates.sort(reverse=True)
            return candidates[0][1]
            
        except Exception as e:
            logger.error(f"Failed to find best agent: {e}")
            return None
    
    def _check_skills_match(self, task: Task, agent: Agent) -> bool:
        """检查技能匹配"""
        return all(skill in agent.skills for skill in task.required_skills)
    
    def _calculate_match_score(self, task: Task, agent: Agent) -> float:
        """计算匹配分数"""
        try:
            # 基础分数
            base_score = 0.0
            
            # 技能匹配分数
            skill_match_ratio = len(set(task.required_skills) & set(agent.skills)) / len(task.required_skills)
            base_score += skill_match_ratio * 0.4
            
            # 性能分数
            base_score += agent.performance_score * 0.3
            
            # 负载分数（负载越轻分数越高）
            load_ratio = len(agent.current_tasks) / agent.capacity
            base_score += (1 - load_ratio) * 0.2
            
            # 成本分数（成本越低分数越高）
            cost_score = 1.0 / (agent.cost_per_hour + 0.1)
            base_score += cost_score * 0.1
            
            return base_score
            
        except Exception as e:
            logger.error(f"Failed to calculate match score: {e}")
            return 0.0
    
    async def _allocate_task(self, task: Task, agent: Agent) -> Dict[str, Any]:
        """分配任务"""
        try:
            # 更新任务状态
            task.assigned_agent = agent.id
            task.status = "assigned"
            
            # 更新智能体状态
            agent.current_tasks.append(task.id)
            if agent.state == AgentState.IDLE:
                agent.state = AgentState.BUSY
            
            allocation = {
                "task_id": task.id,
                "agent_id": agent.id,
                "allocated_at": datetime.now(),
                "estimated_completion": datetime.now() + timedelta(seconds=task.estimated_duration)
            }
            
            logger.info(f"Task {task.name} allocated to agent {agent.name}")
            return allocation
            
        except Exception as e:
            logger.error(f"Failed to allocate task: {e}")
            return {"error": str(e)}

class AdvancedNegotiationEngine:
    """高级协商引擎"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.negotiations = {}
        self.negotiation_history = []
        
    async def start_negotiation(self, initiator: str, participants: List[str], 
                              topic: str, initial_proposal: Any) -> str:
        """开始协商"""
        try:
            negotiation_id = f"neg_{int(time.time())}"
            
            negotiation = {
                "id": negotiation_id,
                "initiator": initiator,
                "participants": participants,
                "topic": topic,
                "proposals": [initial_proposal],
                "votes": {},
                "status": "active",
                "start_time": datetime.now(),
                "round": 1
            }
            
            self.negotiations[negotiation_id] = negotiation
            
            # 通知参与者
            await self._notify_participants(negotiation_id, "negotiation_started")
            
            logger.info(f"Negotiation {negotiation_id} started by {initiator}")
            return negotiation_id
            
        except Exception as e:
            logger.error(f"Failed to start negotiation: {e}")
            return None
    
    async def submit_proposal(self, negotiation_id: str, agent_id: str, 
                            proposal: Any) -> bool:
        """提交提案"""
        try:
            if negotiation_id not in self.negotiations:
                return False
            
            negotiation = self.negotiations[negotiation_id]
            
            if agent_id not in negotiation["participants"]:
                return False
            
            # 添加提案
            negotiation["proposals"].append({
                "agent_id": agent_id,
                "proposal": proposal,
                "timestamp": datetime.now()
            })
            
            logger.info(f"Proposal submitted by {agent_id} in negotiation {negotiation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit proposal: {e}")
            return False
    
    async def vote_on_proposal(self, negotiation_id: str, agent_id: str, 
                             proposal_index: int, vote: bool) -> bool:
        """对提案投票"""
        try:
            if negotiation_id not in self.negotiations:
                return False
            
            negotiation = self.negotiations[negotiation_id]
            
            if agent_id not in negotiation["participants"]:
                return False
            
            if proposal_index >= len(negotiation["proposals"]):
                return False
            
            # 记录投票
            if negotiation_id not in self.negotiations[negotiation_id]["votes"]:
                self.negotiations[negotiation_id]["votes"] = {}
            
            self.negotiations[negotiation_id]["votes"][agent_id] = {
                "proposal_index": proposal_index,
                "vote": vote,
                "timestamp": datetime.now()
            }
            
            # 检查是否所有参与者都已投票
            if len(self.negotiations[negotiation_id]["votes"]) == len(negotiation["participants"]):
                await self._finalize_negotiation(negotiation_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to vote on proposal: {e}")
            return False
    
    async def _finalize_negotiation(self, negotiation_id: str):
        """完成协商"""
        try:
            negotiation = self.negotiations[negotiation_id]
            
            # 统计投票结果
            vote_counts = defaultdict(int)
            for vote_data in negotiation["votes"].values():
                if vote_data["vote"]:
                    vote_counts[vote_data["proposal_index"]] += 1
            
            # 找到得票最多的提案
            if vote_counts:
                winning_proposal_index = max(vote_counts, key=vote_counts.get)
                winning_proposal = negotiation["proposals"][winning_proposal_index]
                
                negotiation["result"] = {
                    "winning_proposal": winning_proposal,
                    "votes_received": vote_counts[winning_proposal_index],
                    "total_participants": len(negotiation["participants"])
                }
            else:
                negotiation["result"] = {
                    "winning_proposal": None,
                    "votes_received": 0,
                    "total_participants": len(negotiation["participants"])
                }
            
            negotiation["status"] = "completed"
            negotiation["end_time"] = datetime.now()
            
            # 记录协商历史
            self.negotiation_history.append(negotiation)
            
            # 通知参与者
            await self._notify_participants(negotiation_id, "negotiation_completed")
            
            logger.info(f"Negotiation {negotiation_id} completed")
            
        except Exception as e:
            logger.error(f"Failed to finalize negotiation: {e}")
    
    async def _notify_participants(self, negotiation_id: str, event: str):
        """通知参与者"""
        # 这里可以实现具体的通知逻辑
        logger.info(f"Notifying participants of {event} in negotiation {negotiation_id}")

class AdvancedLoadBalancer:
    """高级负载均衡器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agents: Dict[str, Agent] = {}
        self.load_history = defaultdict(list)
        self.balancing_strategies = {
            "round_robin": self._round_robin_balance,
            "least_connections": self._least_connections_balance,
            "weighted_round_robin": self._weighted_round_robin_balance,
            "adaptive": self._adaptive_balance
        }
        self.current_strategy = "adaptive"
        
    def register_agent(self, agent: Agent) -> bool:
        """注册智能体"""
        try:
            self.agents[agent.id] = agent
            self.load_history[agent.id] = []
            logger.info(f"Agent {agent.name} registered for load balancing")
            return True
        except Exception as e:
            logger.error(f"Failed to register agent for load balancing: {e}")
            return False
    
    async def balance_load(self, task: Task) -> Optional[str]:
        """负载均衡"""
        try:
            if not self.agents:
                return None
            
            # 更新负载历史
            self._update_load_history()
            
            # 选择均衡策略
            strategy = self.balancing_strategies.get(self.current_strategy)
            if not strategy:
                strategy = self._adaptive_balance
            
            # 执行负载均衡
            selected_agent_id = await strategy(task)
            
            if selected_agent_id:
                logger.info(f"Task {task.name} balanced to agent {selected_agent_id}")
            
            return selected_agent_id
            
        except Exception as e:
            logger.error(f"Load balancing failed: {e}")
            return None
    
    def _update_load_history(self):
        """更新负载历史"""
        current_time = datetime.now()
        for agent_id, agent in self.agents.items():
            load = len(agent.current_tasks) / agent.capacity if agent.capacity > 0 else 0
            self.load_history[agent_id].append({
                "timestamp": current_time,
                "load": load
            })
            
            # 保留最近1小时的历史
            cutoff_time = current_time - timedelta(hours=1)
            self.load_history[agent_id] = [
                entry for entry in self.load_history[agent_id]
                if entry["timestamp"] > cutoff_time
            ]
    
    async def _round_robin_balance(self, task: Task) -> Optional[str]:
        """轮询负载均衡"""
        available_agents = [
            agent for agent in self.agents.values()
            if agent.state == AgentState.IDLE and len(agent.current_tasks) < agent.capacity
        ]
        
        if not available_agents:
            return None
        
        # 简单的轮询选择
        agent_ids = list(available_agents.keys())
        return random.choice(agent_ids)
    
    async def _least_connections_balance(self, task: Task) -> Optional[str]:
        """最少连接负载均衡"""
        available_agents = [
            agent for agent in self.agents.values()
            if agent.state == AgentState.IDLE and len(agent.current_tasks) < agent.capacity
        ]
        
        if not available_agents:
            return None
        
        # 选择当前任务最少的智能体
        min_tasks = min(len(agent.current_tasks) for agent in available_agents)
        candidates = [agent for agent in available_agents if len(agent.current_tasks) == min_tasks]
        
        return random.choice(candidates).id if candidates else None
    
    async def _weighted_round_robin_balance(self, task: Task) -> Optional[str]:
        """加权轮询负载均衡"""
        available_agents = [
            agent for agent in self.agents.values()
            if agent.state == AgentState.IDLE and len(agent.current_tasks) < agent.capacity
        ]
        
        if not available_agents:
            return None
        
        # 根据性能分数和当前负载计算权重
        weights = []
        for agent in available_agents:
            load_ratio = len(agent.current_tasks) / agent.capacity if agent.capacity > 0 else 0
            weight = agent.performance_score * (1 - load_ratio)
            weights.append(weight)
        
        # 加权随机选择
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(available_agents).id
        
        rand = random.uniform(0, total_weight)
        cumulative = 0
        for i, weight in enumerate(weights):
            cumulative += weight
            if rand <= cumulative:
                return available_agents[i].id
        
        return available_agents[-1].id
    
    async def _adaptive_balance(self, task: Task) -> Optional[str]:
        """自适应负载均衡"""
        available_agents = [
            agent for agent in self.agents.values()
            if agent.state == AgentState.IDLE and len(agent.current_tasks) < agent.capacity
        ]
        
        if not available_agents:
            return None
        
        # 综合考虑多个因素
        scores = []
        for agent in available_agents:
            score = 0.0
            
            # 性能分数
            score += agent.performance_score * 0.3
            
            # 负载分数（负载越轻分数越高）
            load_ratio = len(agent.current_tasks) / agent.capacity if agent.capacity > 0 else 0
            score += (1 - load_ratio) * 0.3
            
            # 历史负载趋势
            if agent.id in self.load_history and len(self.load_history[agent.id]) > 1:
                recent_loads = [entry["load"] for entry in self.load_history[agent.id][-5:]]
                avg_load = sum(recent_loads) / len(recent_loads)
                score += (1 - avg_load) * 0.2
            
            # 成本分数
            cost_score = 1.0 / (agent.cost_per_hour + 0.1)
            score += cost_score * 0.2
            
            scores.append((score, agent.id))
        
        # 选择分数最高的智能体
        scores.sort(reverse=True)
        return scores[0][1] if scores else None

class AdvancedFaultTolerance:
    """高级容错系统"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agents: Dict[str, Agent] = {}
        self.health_checks = {}
        self.failure_history = defaultdict(list)
        self.recovery_strategies = {
            "restart": self._restart_agent,
            "failover": self._failover_tasks,
            "replicate": self._replicate_tasks,
            "degrade": self._degrade_service
        }
        
    def register_agent(self, agent: Agent) -> bool:
        """注册智能体"""
        try:
            self.agents[agent.id] = agent
            self.health_checks[agent.id] = {
                "last_check": datetime.now(),
                "status": "healthy",
                "consecutive_failures": 0
            }
            logger.info(f"Agent {agent.name} registered for fault tolerance")
            return True
        except Exception as e:
            logger.error(f"Failed to register agent for fault tolerance: {e}")
            return False
    
    async def start_health_monitoring(self):
        """开始健康监控"""
        try:
            while True:
                await self._perform_health_checks()
                await asyncio.sleep(self.config.get("health_check_interval", 30))
        except Exception as e:
            logger.error(f"Health monitoring failed: {e}")
    
    async def _perform_health_checks(self):
        """执行健康检查"""
        try:
            current_time = datetime.now()
            
            for agent_id, agent in self.agents.items():
                # 检查心跳
                time_since_heartbeat = (current_time - agent.last_heartbeat).total_seconds()
                heartbeat_timeout = self.config.get("heartbeat_timeout", 60)
                
                if time_since_heartbeat > heartbeat_timeout:
                    await self._handle_agent_failure(agent_id, "heartbeat_timeout")
                else:
                    # 更新健康状态
                    health_check = self.health_checks[agent_id]
                    health_check["last_check"] = current_time
                    health_check["status"] = "healthy"
                    health_check["consecutive_failures"] = 0
                    
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    async def _handle_agent_failure(self, agent_id: str, reason: str):
        """处理智能体故障"""
        try:
            agent = self.agents[agent_id]
            health_check = self.health_checks[agent_id]
            
            # 记录故障
            failure_record = {
                "timestamp": datetime.now(),
                "reason": reason,
                "agent_id": agent_id,
                "agent_name": agent.name
            }
            self.failure_history[agent_id].append(failure_record)
            
            # 更新健康状态
            health_check["status"] = "unhealthy"
            health_check["consecutive_failures"] += 1
            
            # 选择恢复策略
            recovery_strategy = self._select_recovery_strategy(agent_id, reason)
            
            if recovery_strategy in self.recovery_strategies:
                await self.recovery_strategies[recovery_strategy](agent_id)
            
            logger.warning(f"Agent {agent.name} failure handled with strategy {recovery_strategy}")
            
        except Exception as e:
            logger.error(f"Failed to handle agent failure: {e}")
    
    def _select_recovery_strategy(self, agent_id: str, reason: str) -> str:
        """选择恢复策略"""
        health_check = self.health_checks[agent_id]
        consecutive_failures = health_check["consecutive_failures"]
        
        if consecutive_failures <= 2:
            return "restart"
        elif consecutive_failures <= 5:
            return "failover"
        elif consecutive_failures <= 10:
            return "replicate"
        else:
            return "degrade"
    
    async def _restart_agent(self, agent_id: str):
        """重启智能体"""
        try:
            agent = self.agents[agent_id]
            agent.state = AgentState.MAINTENANCE
            
            # 模拟重启过程
            await asyncio.sleep(5)
            
            agent.state = AgentState.IDLE
            agent.last_heartbeat = datetime.now()
            
            logger.info(f"Agent {agent.name} restarted successfully")
            
        except Exception as e:
            logger.error(f"Failed to restart agent: {e}")
    
    async def _failover_tasks(self, agent_id: str):
        """任务故障转移"""
        try:
            agent = self.agents[agent_id]
            
            # 将任务转移到其他智能体
            for task_id in agent.current_tasks.copy():
                await self._transfer_task(task_id, agent_id)
            
            logger.info(f"Tasks failed over from agent {agent.name}")
            
        except Exception as e:
            logger.error(f"Failed to failover tasks: {e}")
    
    async def _replicate_tasks(self, agent_id: str):
        """任务复制"""
        try:
            agent = self.agents[agent_id]
            
            # 复制关键任务到其他智能体
            for task_id in agent.current_tasks:
                await self._replicate_task(task_id, agent_id)
            
            logger.info(f"Tasks replicated from agent {agent.name}")
            
        except Exception as e:
            logger.error(f"Failed to replicate tasks: {e}")
    
    async def _degrade_service(self, agent_id: str):
        """服务降级"""
        try:
            agent = self.agents[agent_id]
            agent.state = AgentState.ERROR
            
            # 停止接受新任务
            # 完成当前任务后停止服务
            
            logger.warning(f"Service degraded for agent {agent.name}")
            
        except Exception as e:
            logger.error(f"Failed to degrade service: {e}")
    
    async def _transfer_task(self, task_id: str, from_agent_id: str):
        """转移任务"""
        # 实现任务转移逻辑
        logger.info(f"Task {task_id} transferred from agent {from_agent_id}")
    
    async def _replicate_task(self, task_id: str, from_agent_id: str):
        """复制任务"""
        # 实现任务复制逻辑
        logger.info(f"Task {task_id} replicated from agent {from_agent_id}")

# 示例使用
async def main():
    """主函数示例"""
    config = {
        "heartbeat_timeout": 60,
        "health_check_interval": 30,
        "max_negotiation_rounds": 10
    }
    
    # 创建智能体
    agents = [
        Agent(
            id="agent_1",
            name="数据分析师",
            skills=["data_analysis", "python", "statistics"],
            capacity=5,
            performance_score=0.9
        ),
        Agent(
            id="agent_2",
            name="机器学习工程师",
            skills=["ml", "python", "tensorflow"],
            capacity=3,
            performance_score=0.95
        ),
        Agent(
            id="agent_3",
            name="后端开发工程师",
            skills=["backend", "python", "api"],
            capacity=8,
            performance_score=0.85
        )
    ]
    
    # 创建任务
    tasks = [
        Task(
            id="task_1",
            name="数据分析任务",
            description="分析用户行为数据",
            priority=TaskPriority.HIGH,
            estimated_duration=3600,
            required_skills=["data_analysis", "python"]
        ),
        Task(
            id="task_2",
            name="模型训练任务",
            description="训练推荐模型",
            priority=TaskPriority.MEDIUM,
            estimated_duration=7200,
            required_skills=["ml", "python", "tensorflow"]
        ),
        Task(
            id="task_3",
            name="API开发任务",
            description="开发用户API",
            priority=TaskPriority.LOW,
            estimated_duration=1800,
            required_skills=["backend", "python", "api"]
        )
    ]
    
    # 创建高级任务分配器
    allocator = AdvancedTaskAllocator(config)
    
    # 注册智能体
    for agent in agents:
        allocator.register_agent(agent)
    
    # 提交任务
    for task in tasks:
        allocator.submit_task(task)
    
    # 执行任务分配
    result = await allocator.allocate_tasks()
    print("任务分配结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    
    # 创建协商引擎
    negotiation_engine = AdvancedNegotiationEngine(config)
    
    # 开始协商
    negotiation_id = await negotiation_engine.start_negotiation(
        initiator="agent_1",
        participants=["agent_1", "agent_2", "agent_3"],
        topic="任务优先级调整",
        initial_proposal={"priority": "high", "reason": "紧急任务"}
    )
    
    print(f"\n协商开始: {negotiation_id}")
    
    # 创建负载均衡器
    load_balancer = AdvancedLoadBalancer(config)
    
    for agent in agents:
        load_balancer.register_agent(agent)
    
    # 测试负载均衡
    test_task = Task(
        id="test_task",
        name="测试任务",
        description="负载均衡测试",
        priority=TaskPriority.MEDIUM,
        estimated_duration=600,
        required_skills=["python"]
    )
    
    balanced_agent = await load_balancer.balance_load(test_task)
    print(f"\n负载均衡结果: 任务分配给智能体 {balanced_agent}")
    
    # 创建容错系统
    fault_tolerance = AdvancedFaultTolerance(config)
    
    for agent in agents:
        fault_tolerance.register_agent(agent)
    
    print("\n容错系统已启动")

if __name__ == "__main__":
    asyncio.run(main())
