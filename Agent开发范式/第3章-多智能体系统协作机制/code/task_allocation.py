# task_allocation.py
"""
任务分配与负载均衡实现
提供多种任务分配策略和负载均衡算法
"""

import asyncio
import logging
import heapq
import random
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import uuid
import statistics
from collections import defaultdict, deque

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AllocationStrategy(Enum):
    """分配策略枚举"""
    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    CAPABILITY_BASED = "capability_based"
    PERFORMANCE_BASED = "performance_based"
    COST_OPTIMIZED = "cost_optimized"
    DEADLINE_AWARE = "deadline_aware"
    RESOURCE_AWARE = "resource_aware"
    ADAPTIVE = "adaptive"

class LoadBalancingAlgorithm(Enum):
    """负载均衡算法枚举"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    CONSISTENT_HASH = "consistent_hash"
    LEAST_LOAD = "least_load"
    ADAPTIVE = "adaptive"

@dataclass
class ResourceRequirement:
    """资源需求"""
    cpu: float = 0.0
    memory: float = 0.0
    disk: float = 0.0
    network: float = 0.0
    gpu: float = 0.0
    custom_resources: Dict[str, float] = field(default_factory=dict)

@dataclass
class ResourceCapacity:
    """资源容量"""
    cpu: float = 1.0
    memory: float = 1.0
    disk: float = 1.0
    network: float = 1.0
    gpu: float = 0.0
    custom_resources: Dict[str, float] = field(default_factory=dict)

@dataclass
class AgentMetrics:
    """智能体指标"""
    agent_id: str
    current_load: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    response_time: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    availability: float = 1.0
    last_update: datetime = field(default_factory=datetime.now)
    performance_score: float = 1.0
    cost_per_task: float = 1.0

@dataclass
class TaskAllocation:
    """任务分配结果"""
    task_id: str
    agent_id: str
    allocation_time: datetime = field(default_factory=datetime.now)
    estimated_completion_time: Optional[datetime] = None
    priority_score: float = 0.0
    resource_utilization: float = 0.0
    cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class TaskDecomposer:
    """任务分解器"""
    
    def __init__(self):
        self.decomposition_strategies = {
            "hierarchical": self._hierarchical_decomposition,
            "dependency": self._dependency_decomposition,
            "resource": self._resource_decomposition,
            "parallel": self._parallel_decomposition
        }
    
    async def decompose_task(self, task: Any, strategy: str = "hierarchical") -> List[Any]:
        """分解任务"""
        if strategy not in self.decomposition_strategies:
            strategy = "hierarchical"
        
        decomposer = self.decomposition_strategies[strategy]
        return await decomposer(task)
    
    async def _hierarchical_decomposition(self, task: Any) -> List[Any]:
        """层次化分解"""
        # 模拟层次化分解
        subtasks = []
        for i in range(3):  # 分解为3个子任务
            subtask = {
                "id": f"{task.get('id', 'task')}_sub_{i}",
                "name": f"{task.get('name', 'Task')} Subtask {i}",
                "priority": task.get("priority", 1),
                "resource_requirements": task.get("resource_requirements", {}),
                "dependencies": [f"{task.get('id', 'task')}_sub_{i-1}"] if i > 0 else []
            }
            subtasks.append(subtask)
        
        return subtasks
    
    async def _dependency_decomposition(self, task: Any) -> List[Any]:
        """依赖关系分解"""
        # 模拟依赖关系分解
        subtasks = []
        phases = ["prepare", "process", "finalize"]
        
        for i, phase in enumerate(phases):
            subtask = {
                "id": f"{task.get('id', 'task')}_{phase}",
                "name": f"{task.get('name', 'Task')} {phase.title()}",
                "priority": task.get("priority", 1),
                "resource_requirements": task.get("resource_requirements", {}),
                "dependencies": [f"{task.get('id', 'task')}_{phases[i-1]}"] if i > 0 else []
            }
            subtasks.append(subtask)
        
        return subtasks
    
    async def _resource_decomposition(self, task: Any) -> List[Any]:
        """资源分解"""
        # 模拟资源分解
        subtasks = []
        resource_types = ["cpu_intensive", "memory_intensive", "io_intensive"]
        
        for resource_type in resource_types:
            subtask = {
                "id": f"{task.get('id', 'task')}_{resource_type}",
                "name": f"{task.get('name', 'Task')} {resource_type.replace('_', ' ').title()}",
                "priority": task.get("priority", 1),
                "resource_requirements": {resource_type: 0.8},
                "dependencies": []
            }
            subtasks.append(subtask)
        
        return subtasks
    
    async def _parallel_decomposition(self, task: Any) -> List[Any]:
        """并行分解"""
        # 模拟并行分解
        subtasks = []
        parallel_count = 4  # 分解为4个并行任务
        
        for i in range(parallel_count):
            subtask = {
                "id": f"{task.get('id', 'task')}_parallel_{i}",
                "name": f"{task.get('name', 'Task')} Parallel {i}",
                "priority": task.get("priority", 1),
                "resource_requirements": task.get("resource_requirements", {}),
                "dependencies": []  # 并行任务无依赖
            }
            subtasks.append(subtask)
        
        return subtasks

class LoadBalancer:
    """负载均衡器"""
    
    def __init__(self, algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.LEAST_LOAD):
        self.algorithm = algorithm
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        self.round_robin_index = 0
        self.consistent_hash_ring = {}
        self.adaptive_weights: Dict[str, float] = {}
        self.history_window = deque(maxlen=100)
    
    def update_agent_metrics(self, agent_id: str, metrics: AgentMetrics):
        """更新智能体指标"""
        self.agent_metrics[agent_id] = metrics
        self.history_window.append((agent_id, metrics, datetime.now()))
        
        # 更新自适应权重
        self._update_adaptive_weights(agent_id, metrics)
    
    def select_agent(self, available_agents: List[str]) -> Optional[str]:
        """选择智能体"""
        if not available_agents:
            return None
        
        if self.algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
            return self._round_robin_selection(available_agents)
        elif self.algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
            return self._least_connections_selection(available_agents)
        elif self.algorithm == LoadBalancingAlgorithm.LEAST_RESPONSE_TIME:
            return self._least_response_time_selection(available_agents)
        elif self.algorithm == LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(available_agents)
        elif self.algorithm == LoadBalancingAlgorithm.CONSISTENT_HASH:
            return self._consistent_hash_selection(available_agents)
        elif self.algorithm == LoadBalancingAlgorithm.LEAST_LOAD:
            return self._least_load_selection(available_agents)
        elif self.algorithm == LoadBalancingAlgorithm.ADAPTIVE:
            return self._adaptive_selection(available_agents)
        else:
            return random.choice(available_agents)
    
    def _round_robin_selection(self, available_agents: List[str]) -> str:
        """轮询选择"""
        agent = available_agents[self.round_robin_index % len(available_agents)]
        self.round_robin_index += 1
        return agent
    
    def _least_connections_selection(self, available_agents: List[str]) -> str:
        """最少连接选择"""
        return min(available_agents, key=lambda aid: self.agent_metrics.get(aid, AgentMetrics(aid)).current_load)
    
    def _least_response_time_selection(self, available_agents: List[str]) -> str:
        """最少响应时间选择"""
        return min(available_agents, key=lambda aid: self.agent_metrics.get(aid, AgentMetrics(aid)).response_time)
    
    def _weighted_round_robin_selection(self, available_agents: List[str]) -> str:
        """加权轮询选择"""
        weights = [self.adaptive_weights.get(aid, 1.0) for aid in available_agents]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return random.choice(available_agents)
        
        # 加权随机选择
        rand = random.uniform(0, total_weight)
        cumulative = 0
        
        for i, weight in enumerate(weights):
            cumulative += weight
            if rand <= cumulative:
                return available_agents[i]
        
        return available_agents[-1]
    
    def _consistent_hash_selection(self, available_agents: List[str]) -> str:
        """一致性哈希选择"""
        # 简单的哈希选择
        hash_value = hash(str(time.time())) % len(available_agents)
        return available_agents[hash_value]
    
    def _least_load_selection(self, available_agents: List[str]) -> str:
        """最少负载选择"""
        return min(available_agents, key=lambda aid: self.agent_metrics.get(aid, AgentMetrics(aid)).current_load)
    
    def _adaptive_selection(self, available_agents: List[str]) -> str:
        """自适应选择"""
        # 基于多个指标的综合选择
        scores = {}
        
        for agent_id in available_agents:
            metrics = self.agent_metrics.get(agent_id, AgentMetrics(agent_id))
            
            # 计算综合得分
            score = (
                metrics.performance_score * 0.3 +
                (1 - metrics.current_load) * 0.25 +
                (1 - metrics.error_rate) * 0.2 +
                metrics.availability * 0.15 +
                (1 / (metrics.response_time + 0.001)) * 0.1
            )
            
            scores[agent_id] = score
        
        return max(scores.keys(), key=lambda k: scores[k])
    
    def _update_adaptive_weights(self, agent_id: str, metrics: AgentMetrics):
        """更新自适应权重"""
        # 基于性能指标调整权重
        base_weight = 1.0
        
        # 性能调整
        performance_factor = metrics.performance_score
        
        # 负载调整
        load_factor = 1 - metrics.current_load
        
        # 错误率调整
        error_factor = 1 - metrics.error_rate
        
        # 可用性调整
        availability_factor = metrics.availability
        
        # 综合权重
        weight = base_weight * performance_factor * load_factor * error_factor * availability_factor
        self.adaptive_weights[agent_id] = max(0.1, min(2.0, weight))  # 限制权重范围
    
    def get_load_distribution(self) -> Dict[str, float]:
        """获取负载分布"""
        return {aid: metrics.current_load for aid, metrics in self.agent_metrics.items()}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.agent_metrics:
            return {}
        
        loads = [metrics.current_load for metrics in self.agent_metrics.values()]
        response_times = [metrics.response_time for metrics in self.agent_metrics.values()]
        error_rates = [metrics.error_rate for metrics in self.agent_metrics.values()]
        
        return {
            "total_agents": len(self.agent_metrics),
            "average_load": statistics.mean(loads),
            "load_std": statistics.stdev(loads) if len(loads) > 1 else 0,
            "average_response_time": statistics.mean(response_times),
            "average_error_rate": statistics.mean(error_rates),
            "load_distribution": self.get_load_distribution()
        }

class TaskAllocator:
    """任务分配器"""
    
    def __init__(self, strategy: AllocationStrategy = AllocationStrategy.LOAD_BALANCED):
        self.strategy = strategy
        self.task_decomposer = TaskDecomposer()
        self.load_balancer = LoadBalancer()
        self.allocation_history: List[TaskAllocation] = []
        self.agent_capabilities: Dict[str, Set[str]] = {}
        self.resource_capacities: Dict[str, ResourceCapacity] = {}
        self.cost_matrix: Dict[str, float] = {}
    
    def register_agent_capabilities(self, agent_id: str, capabilities: Set[str]):
        """注册智能体能力"""
        self.agent_capabilities[agent_id] = capabilities
    
    def register_agent_resources(self, agent_id: str, capacity: ResourceCapacity):
        """注册智能体资源容量"""
        self.resource_capacities[agent_id] = capacity
    
    def set_agent_cost(self, agent_id: str, cost: float):
        """设置智能体成本"""
        self.cost_matrix[agent_id] = cost
    
    async def allocate_tasks(self, tasks: List[Any], available_agents: List[str]) -> List[TaskAllocation]:
        """分配任务"""
        allocations = []
        
        for task in tasks:
            allocation = await self._allocate_single_task(task, available_agents)
            if allocation:
                allocations.append(allocation)
                self.allocation_history.append(allocation)
        
        return allocations
    
    async def _allocate_single_task(self, task: Any, available_agents: List[str]) -> Optional[TaskAllocation]:
        """分配单个任务"""
        if not available_agents:
            return None
        
        # 根据策略选择智能体
        if self.strategy == AllocationStrategy.ROUND_ROBIN:
            agent_id = self._round_robin_allocation(available_agents)
        elif self.strategy == AllocationStrategy.LOAD_BALANCED:
            agent_id = self._load_balanced_allocation(task, available_agents)
        elif self.strategy == AllocationStrategy.CAPABILITY_BASED:
            agent_id = self._capability_based_allocation(task, available_agents)
        elif self.strategy == AllocationStrategy.PERFORMANCE_BASED:
            agent_id = self._performance_based_allocation(task, available_agents)
        elif self.strategy == AllocationStrategy.COST_OPTIMIZED:
            agent_id = self._cost_optimized_allocation(task, available_agents)
        elif self.strategy == AllocationStrategy.DEADLINE_AWARE:
            agent_id = self._deadline_aware_allocation(task, available_agents)
        elif self.strategy == AllocationStrategy.RESOURCE_AWARE:
            agent_id = self._resource_aware_allocation(task, available_agents)
        elif self.strategy == AllocationStrategy.ADAPTIVE:
            agent_id = self._adaptive_allocation(task, available_agents)
        else:
            agent_id = random.choice(available_agents)
        
        if not agent_id:
            return None
        
        # 创建分配结果
        allocation = TaskAllocation(
            task_id=task.get("id", str(uuid.uuid4())),
            agent_id=agent_id,
            priority_score=self._calculate_priority_score(task),
            resource_utilization=self._calculate_resource_utilization(task, agent_id),
            cost=self.cost_matrix.get(agent_id, 1.0)
        )
        
        return allocation
    
    def _round_robin_allocation(self, available_agents: List[str]) -> str:
        """轮询分配"""
        return self.load_balancer._round_robin_selection(available_agents)
    
    def _load_balanced_allocation(self, task: Any, available_agents: List[str]) -> str:
        """负载均衡分配"""
        return self.load_balancer.select_agent(available_agents)
    
    def _capability_based_allocation(self, task: Any, available_agents: List[str]) -> str:
        """基于能力的分配"""
        required_capabilities = set(task.get("required_capabilities", []))
        
        # 过滤有能力的智能体
        capable_agents = []
        for agent_id in available_agents:
            agent_capabilities = self.agent_capabilities.get(agent_id, set())
            if required_capabilities.issubset(agent_capabilities):
                capable_agents.append(agent_id)
        
        if not capable_agents:
            return None
        
        # 在有能力智能体中选择负载最低的
        return self.load_balancer.select_agent(capable_agents)
    
    def _performance_based_allocation(self, task: Any, available_agents: List[str]) -> str:
        """基于性能的分配"""
        best_agent = None
        best_score = -1
        
        for agent_id in available_agents:
            metrics = self.load_balancer.agent_metrics.get(agent_id, AgentMetrics(agent_id))
            score = metrics.performance_score
            
            if score > best_score:
                best_score = score
                best_agent = agent_id
        
        return best_agent
    
    def _cost_optimized_allocation(self, task: Any, available_agents: List[str]) -> str:
        """成本优化分配"""
        best_agent = None
        best_cost = float('inf')
        
        for agent_id in available_agents:
            cost = self.cost_matrix.get(agent_id, 1.0)
            
            if cost < best_cost:
                best_cost = cost
                best_agent = agent_id
        
        return best_agent
    
    def _deadline_aware_allocation(self, task: Any, available_agents: List[str]) -> str:
        """截止时间感知分配"""
        deadline = task.get("deadline")
        if not deadline:
            return self._load_balanced_allocation(task, available_agents)
        
        # 选择响应时间最短的智能体
        return min(available_agents, key=lambda aid: self.load_balancer.agent_metrics.get(aid, AgentMetrics(aid)).response_time)
    
    def _resource_aware_allocation(self, task: Any, available_agents: List[str]) -> str:
        """资源感知分配"""
        task_requirements = task.get("resource_requirements", {})
        
        suitable_agents = []
        for agent_id in available_agents:
            capacity = self.resource_capacities.get(agent_id, ResourceCapacity())
            metrics = self.load_balancer.agent_metrics.get(agent_id, AgentMetrics(agent_id))
            
            # 检查资源是否足够
            if self._check_resource_availability(task_requirements, capacity, metrics):
                suitable_agents.append(agent_id)
        
        if not suitable_agents:
            return None
        
        return self.load_balancer.select_agent(suitable_agents)
    
    def _adaptive_allocation(self, task: Any, available_agents: List[str]) -> str:
        """自适应分配"""
        # 结合多种策略
        strategies = [
            self._capability_based_allocation,
            self._resource_aware_allocation,
            self._load_balanced_allocation
        ]
        
        for strategy in strategies:
            agent_id = strategy(task, available_agents)
            if agent_id:
                return agent_id
        
        return random.choice(available_agents)
    
    def _calculate_priority_score(self, task: Any) -> float:
        """计算优先级得分"""
        priority = task.get("priority", 1)
        deadline = task.get("deadline")
        
        score = priority
        
        if deadline:
            time_remaining = (deadline - datetime.now()).total_seconds()
            if time_remaining > 0:
                score += 1 / (time_remaining + 1)  # 时间越紧得分越高
        
        return score
    
    def _calculate_resource_utilization(self, task: Any, agent_id: str) -> float:
        """计算资源利用率"""
        requirements = task.get("resource_requirements", {})
        capacity = self.resource_capacities.get(agent_id, ResourceCapacity())
        
        if not requirements or not capacity:
            return 0.0
        
        utilization = 0.0
        resource_count = 0
        
        for resource, amount in requirements.items():
            if hasattr(capacity, resource):
                capacity_value = getattr(capacity, resource)
                if capacity_value > 0:
                    utilization += amount / capacity_value
                    resource_count += 1
        
        return utilization / resource_count if resource_count > 0 else 0.0
    
    def _check_resource_availability(self, requirements: Dict[str, float], 
                                   capacity: ResourceCapacity, metrics: AgentMetrics) -> bool:
        """检查资源可用性"""
        for resource, amount in requirements.items():
            if hasattr(capacity, resource):
                capacity_value = getattr(capacity, resource)
                current_usage = getattr(metrics, f"{resource}_usage", 0)
                
                if current_usage + amount > capacity_value:
                    return False
        
        return True
    
    def get_allocation_statistics(self) -> Dict[str, Any]:
        """获取分配统计"""
        if not self.allocation_history:
            return {}
        
        agent_counts = defaultdict(int)
        total_cost = 0
        total_utilization = 0
        
        for allocation in self.allocation_history:
            agent_counts[allocation.agent_id] += 1
            total_cost += allocation.cost
            total_utilization += allocation.resource_utilization
        
        return {
            "total_allocations": len(self.allocation_history),
            "agent_distribution": dict(agent_counts),
            "average_cost": total_cost / len(self.allocation_history),
            "average_utilization": total_utilization / len(self.allocation_history),
            "strategy": self.strategy.value
        }

# 使用示例
async def main():
    """主函数示例"""
    # 创建任务分配器
    allocator = TaskAllocator(strategy=AllocationStrategy.LOAD_BALANCED)
    
    # 注册智能体能力
    allocator.register_agent_capabilities("agent_1", {"research", "analysis"})
    allocator.register_agent_capabilities("agent_2", {"analysis", "reporting"})
    allocator.register_agent_capabilities("agent_3", {"research", "reporting"})
    
    # 注册智能体资源
    allocator.register_agent_resources("agent_1", ResourceCapacity(cpu=1.0, memory=2.0))
    allocator.register_agent_resources("agent_2", ResourceCapacity(cpu=1.5, memory=1.5))
    allocator.register_agent_resources("agent_3", ResourceCapacity(cpu=0.8, memory=1.0))
    
    # 设置智能体成本
    allocator.set_agent_cost("agent_1", 1.0)
    allocator.set_agent_cost("agent_2", 1.2)
    allocator.set_agent_cost("agent_3", 0.8)
    
    # 更新智能体指标
    allocator.load_balancer.update_agent_metrics("agent_1", AgentMetrics("agent_1", current_load=0.3, performance_score=0.9))
    allocator.load_balancer.update_agent_metrics("agent_2", AgentMetrics("agent_2", current_load=0.5, performance_score=0.8))
    allocator.load_balancer.update_agent_metrics("agent_3", AgentMetrics("agent_3", current_load=0.2, performance_score=0.95))
    
    # 创建任务
    tasks = [
        {
            "id": "task_1",
            "name": "Research Task",
            "priority": 5,
            "required_capabilities": ["research"],
            "resource_requirements": {"cpu": 0.5, "memory": 0.5}
        },
        {
            "id": "task_2",
            "name": "Analysis Task",
            "priority": 3,
            "required_capabilities": ["analysis"],
            "resource_requirements": {"cpu": 0.3, "memory": 0.3}
        },
        {
            "id": "task_3",
            "name": "Reporting Task",
            "priority": 4,
            "required_capabilities": ["reporting"],
            "resource_requirements": {"cpu": 0.2, "memory": 0.2}
        }
    ]
    
    # 分配任务
    available_agents = ["agent_1", "agent_2", "agent_3"]
    allocations = await allocator.allocate_tasks(tasks, available_agents)
    
    print("Task Allocations:")
    for allocation in allocations:
        print(f"Task {allocation.task_id} -> Agent {allocation.agent_id}")
        print(f"  Priority Score: {allocation.priority_score:.2f}")
        print(f"  Resource Utilization: {allocation.resource_utilization:.2f}")
        print(f"  Cost: {allocation.cost:.2f}")
        print()
    
    # 获取统计信息
    stats = allocator.get_allocation_statistics()
    print("Allocation Statistics:")
    print(f"Total Allocations: {stats['total_allocations']}")
    print(f"Agent Distribution: {stats['agent_distribution']}")
    print(f"Average Cost: {stats['average_cost']:.2f}")
    print(f"Average Utilization: {stats['average_utilization']:.2f}")
    
    # 获取负载均衡信息
    load_dist = allocator.load_balancer.get_load_distribution()
    print(f"\nLoad Distribution: {load_dist}")
    
    perf_summary = allocator.load_balancer.get_performance_summary()
    print(f"Performance Summary: {perf_summary}")

if __name__ == "__main__":
    asyncio.run(main())
