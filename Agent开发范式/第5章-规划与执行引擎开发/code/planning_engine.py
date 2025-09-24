# planning_engine.py
"""
第5章 规划与执行引擎开发 - 规划引擎
实现智能体的任务规划、路径规划、资源规划等功能
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PlanningType(Enum):
    """规划类型枚举"""
    TASK_PLANNING = "任务规划"
    PATH_PLANNING = "路径规划"
    RESOURCE_PLANNING = "资源规划"
    STRATEGIC_PLANNING = "战略规划"
    TACTICAL_PLANNING = "战术规划"
    OPERATIONAL_PLANNING = "操作规划"

class PlanningAlgorithm(Enum):
    """规划算法枚举"""
    A_STAR = "A*算法"
    DIJKSTRA = "Dijkstra算法"
    BFS = "广度优先搜索"
    DFS = "深度优先搜索"
    GENETIC = "遗传算法"
    SIMULATED_ANNEALING = "模拟退火"
    PARTIAL_ORDER = "偏序规划"
    HTN = "层次任务网络"
    PDDL = "PDDL规划"

class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "待执行"
    READY = "就绪"
    RUNNING = "执行中"
    COMPLETED = "已完成"
    FAILED = "失败"
    CANCELLED = "已取消"

class PlanStatus(Enum):
    """计划状态枚举"""
    PENDING = "待规划"
    IN_PROGRESS = "规划中"
    COMPLETED = "已完成"
    FAILED = "失败"
    CANCELLED = "已取消"

class PlanningStrategy(Enum):
    """规划策略枚举"""
    HIERARCHICAL = "分层规划"
    REACTIVE = "反应式规划"
    DELIBERATIVE = "深思熟虑式规划"
    HYBRID = "混合式规划"
    SUSPENDED = "暂停"

class TaskPriority(Enum):
    """任务优先级枚举"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5

@dataclass
class Task:
    """任务数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    task_type: str = "general"
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    resources_required: Dict[str, Any] = field(default_factory=dict)
    estimated_duration: timedelta = field(default_factory=lambda: timedelta(minutes=1))
    deadline: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    assigned_agent: Optional[str] = None
    progress: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "task_type": self.task_type,
            "priority": self.priority.value,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "resources_required": self.resources_required,
            "estimated_duration": self.estimated_duration.total_seconds(),
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "assigned_agent": self.assigned_agent,
            "progress": self.progress,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """从字典创建任务对象"""
        task = cls()
        task.id = data.get("id", str(uuid.uuid4()))
        task.name = data.get("name", "")
        task.description = data.get("description", "")
        task.task_type = data.get("task_type", "general")
        task.priority = TaskPriority(data.get("priority", 2))
        task.status = TaskStatus(data.get("status", "待执行"))
        task.dependencies = data.get("dependencies", [])
        task.resources_required = data.get("resources_required", {})
        task.estimated_duration = timedelta(seconds=data.get("estimated_duration", 60))
        task.deadline = datetime.fromisoformat(data.get("deadline")) if data.get("deadline") else None
        task.created_at = datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        task.started_at = datetime.fromisoformat(data.get("started_at")) if data.get("started_at") else None
        task.completed_at = datetime.fromisoformat(data.get("completed_at")) if data.get("completed_at") else None
        task.assigned_agent = data.get("assigned_agent")
        task.progress = data.get("progress", 0.0)
        task.metadata = data.get("metadata", {})
        return task

@dataclass
class Plan:
    """计划数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    planning_type: PlanningType = PlanningType.TASK_PLANNING
    algorithm: PlanningAlgorithm = PlanningAlgorithm.A_STAR
    tasks: List[Task] = field(default_factory=list)
    execution_order: List[str] = field(default_factory=list)
    estimated_total_duration: timedelta = field(default_factory=lambda: timedelta(minutes=1))
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "created"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "planning_type": self.planning_type.value,
            "algorithm": self.algorithm.value,
            "tasks": [task.to_dict() for task in self.tasks],
            "execution_order": self.execution_order,
            "estimated_total_duration": self.estimated_total_duration.total_seconds(),
            "resource_requirements": self.resource_requirements,
            "constraints": self.constraints,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "metadata": self.metadata
        }

@dataclass
class PlanningResult:
    """规划结果数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    plan: Plan = None
    success: bool = False
    execution_time: timedelta = field(default_factory=lambda: timedelta(seconds=0))
    cost: float = 0.0
    quality_score: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "plan": self.plan.to_dict() if self.plan else None,
            "success": self.success,
            "execution_time": self.execution_time.total_seconds(),
            "cost": self.cost,
            "quality_score": self.quality_score,
            "warnings": self.warnings,
            "errors": self.errors,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }

class TaskGraph:
    """任务图类"""
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.dependencies: Dict[str, List[str]] = defaultdict(list)
    
    def add_task(self, task: Task) -> bool:
        """添加任务到图中"""
        try:
            self.tasks[task.id] = task
            return True
        except Exception as e:
            logger.error(f"Failed to add task: {e}")
            return False
    
    def add_dependency(self, task_id: str, dependency_id: str) -> bool:
        """添加任务依赖"""
        try:
            if task_id in self.tasks and dependency_id in self.tasks:
                self.dependencies[task_id].append(dependency_id)
                self.tasks[task_id].dependencies.append(dependency_id)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to add dependency: {e}")
            return False
    
    def get_execution_order(self) -> List[str]:
        """获取任务执行顺序（简化拓扑排序）"""
        try:
            # 简化的拓扑排序
            in_degree = defaultdict(int)
            for task_id in self.tasks:
                in_degree[task_id] = 0
            
            for task_id, deps in self.dependencies.items():
                for dep in deps:
                    in_degree[task_id] += 1
            
            queue = [task_id for task_id in self.tasks if in_degree[task_id] == 0]
            result = []
            
            while queue:
                task_id = queue.pop(0)
                result.append(task_id)
                
                for dependent_task in self.dependencies[task_id]:
                    in_degree[dependent_task] -= 1
                    if in_degree[dependent_task] == 0:
                        queue.append(dependent_task)
            
            return result
        except Exception as e:
            logger.error(f"Failed to get execution order: {e}")
            return []
    
    def get_critical_path(self) -> List[str]:
        """获取关键路径（简化实现）"""
        try:
            # 简化的关键路径计算
            execution_order = self.get_execution_order()
            return execution_order
        except Exception as e:
            logger.error(f"Failed to get critical path: {e}")
            return []
    
    def get_task_dependencies(self, task_id: str) -> List[str]:
        """获取任务的所有依赖"""
        try:
            return self.dependencies.get(task_id, [])
        except Exception as e:
            logger.error(f"Failed to get task dependencies: {e}")
            return []
    
    def get_task_dependents(self, task_id: str) -> List[str]:
        """获取依赖该任务的所有任务"""
        try:
            dependents = []
            for task_id_dep, deps in self.dependencies.items():
                if task_id in deps:
                    dependents.append(task_id_dep)
            return dependents
        except Exception as e:
            logger.error(f"Failed to get task dependents: {e}")
            return []

class SimplePlanner:
    """简化规划算法实现"""
    
    def __init__(self):
        pass
    
    def plan(self, start: str, goal: str, tasks: Dict[str, Task]) -> Tuple[List[str], float]:
        """简化规划算法"""
        try:
            if start not in tasks or goal not in tasks:
                return [], float('inf')
            
            # 简化：直接返回从start到goal的路径
            if start == goal:
                return [start], 0.0
            
            # 简化：假设存在直接路径
            path = [start, goal]
            cost = 1.0  # 简化成本
            
            return path, cost
            
        except Exception as e:
            logger.error(f"Simple planning failed: {e}")
            return [], float('inf')

class HTNPlanner:
    """层次任务网络规划器"""
    
    def __init__(self):
        self.methods: Dict[str, List[Dict[str, Any]]] = {}
        self.operators: Dict[str, Dict[str, Any]] = {}
    
    def add_method(self, task_name: str, method: Dict[str, Any]):
        """添加方法"""
        if task_name not in self.methods:
            self.methods[task_name] = []
        self.methods[task_name].append(method)
    
    def add_operator(self, operator_name: str, operator: Dict[str, Any]):
        """添加操作符"""
        self.operators[operator_name] = operator
    
    def plan(self, initial_task: str, initial_state: Dict[str, Any]) -> List[str]:
        """HTN规划"""
        try:
            plan = []
            task_stack = [initial_task]
            current_state = initial_state.copy()
            
            while task_stack:
                current_task = task_stack.pop()
                
                if current_task in self.operators:
                    # 原始任务，检查前置条件
                    operator = self.operators[current_task]
                    preconditions = operator.get('preconditions', [])
                    
                    if self._check_preconditions(preconditions, current_state):
                        # 执行操作
                        effects = operator.get('effects', [])
                        self._apply_effects(effects, current_state)
                        plan.append(current_task)
                    else:
                        logger.warning(f"Preconditions not met for {current_task}")
                        return []
                
                elif current_task in self.methods:
                    # 复合任务，选择方法
                    methods = self.methods[current_task]
                    selected_method = self._select_method(methods, current_state)
                    
                    if selected_method:
                        subtasks = selected_method.get('subtasks', [])
                        # 将子任务按相反顺序添加到栈中
                        for subtask in reversed(subtasks):
                            task_stack.append(subtask)
                    else:
                        logger.warning(f"No applicable method for {current_task}")
                        return []
                
                else:
                    logger.warning(f"Unknown task: {current_task}")
                    return []
            
            return plan
            
        except Exception as e:
            logger.error(f"HTN planning failed: {e}")
            return []
    
    def _check_preconditions(self, preconditions: List[str], state: Dict[str, Any]) -> bool:
        """检查前置条件"""
        for condition in preconditions:
            if condition not in state or not state[condition]:
                return False
        return True
    
    def _apply_effects(self, effects: List[str], state: Dict[str, Any]):
        """应用效果"""
        for effect in effects:
            state[effect] = True
    
    def _select_method(self, methods: List[Dict[str, Any]], state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """选择方法"""
        for method in methods:
            conditions = method.get('conditions', [])
            if self._check_preconditions(conditions, state):
                return method
        return None

class PlanningEngine:
    """规划引擎主类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.task_graph = TaskGraph()
        self.plans: Dict[str, Plan] = {}
        self.planning_results: Dict[str, PlanningResult] = {}
        self.simple_planner = SimplePlanner()
        self.htn_planner = HTNPlanner()
        self.running = False
    
    async def start(self):
        """启动规划引擎"""
        self.running = True
        logger.info("Planning engine started")
    
    async def stop(self):
        """停止规划引擎"""
        self.running = False
        logger.info("Planning engine stopped")
    
    async def add_task(self, task: Task) -> bool:
        """添加任务"""
        try:
            result = self.task_graph.add_task(task)
            if result:
                logger.info(f"Added task: {task.name} ({task.id})")
            return result
        except Exception as e:
            logger.error(f"Failed to add task: {e}")
            return False
    
    async def add_task_dependency(self, task_id: str, dependency_id: str) -> bool:
        """添加任务依赖"""
        try:
            result = self.task_graph.add_dependency(task_id, dependency_id)
            if result:
                logger.info(f"Added dependency: {dependency_id} -> {task_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to add task dependency: {e}")
            return False
    
    async def create_plan(self, plan: Plan) -> bool:
        """创建计划"""
        try:
            self.plans[plan.id] = plan
            logger.info(f"Created plan: {plan.name} ({plan.id})")
            return True
        except Exception as e:
            logger.error(f"Failed to create plan: {e}")
            return False
    
    async def generate_plan(self, planning_type: PlanningType, 
                           algorithm: PlanningAlgorithm,
                           tasks: List[Task],
                           constraints: List[Dict[str, Any]] = None) -> PlanningResult:
        """生成计划"""
        try:
            start_time = datetime.now()
            
            # 创建任务图
            task_graph = TaskGraph()
            for task in tasks:
                task_graph.add_task(task)
                for dep_id in task.dependencies:
                    task_graph.add_dependency(task.id, dep_id)
            
            # 根据算法选择规划方法
            if algorithm == PlanningAlgorithm.A_STAR:
                execution_order = await self._astar_planning(task_graph, constraints)
            elif algorithm == PlanningAlgorithm.HTN:
                execution_order = await self._htn_planning(task_graph, constraints)
            else:
                execution_order = task_graph.get_execution_order()
            
            # 创建计划
            plan = Plan(
                name=f"Plan_{len(self.plans) + 1}",
                description=f"Generated plan using {algorithm.value}",
                planning_type=planning_type,
                algorithm=algorithm,
                tasks=tasks,
                execution_order=execution_order,
                constraints=constraints or []
            )
            
            # 计算计划指标
            total_duration = sum(task.estimated_duration.total_seconds() for task in tasks)
            plan.estimated_total_duration = timedelta(seconds=total_duration)
            
            # 创建规划结果
            execution_time = datetime.now() - start_time
            result = PlanningResult(
                plan=plan,
                success=len(execution_order) > 0,
                execution_time=execution_time,
                cost=self._calculate_plan_cost(plan),
                quality_score=self._calculate_plan_quality(plan)
            )
            
            if result.success:
                self.plans[plan.id] = plan
                logger.info(f"Generated plan successfully: {plan.name}")
            else:
                result.errors.append("Failed to generate valid execution order")
                logger.error("Failed to generate valid plan")
            
            self.planning_results[result.id] = result
            return result
            
        except Exception as e:
            logger.error(f"Plan generation failed: {e}")
            return PlanningResult(success=False, errors=[str(e)])
    
    async def _astar_planning(self, task_graph: TaskGraph, constraints: List[Dict[str, Any]]) -> List[str]:
        """A*规划"""
        try:
            # 简化的A*规划：找到从开始到结束的最优路径
            execution_order = task_graph.get_execution_order()
            
            # 如果有约束，应用约束
            if constraints:
                execution_order = self._apply_constraints(execution_order, constraints)
            
            return execution_order
            
        except Exception as e:
            logger.error(f"A* planning failed: {e}")
            return []
    
    async def _htn_planning(self, task_graph: TaskGraph, constraints: List[Dict[str, Any]]) -> List[str]:
        """HTN规划"""
        try:
            # 简化的HTN规划：使用层次分解
            execution_order = task_graph.get_execution_order()
            
            # 如果有约束，应用约束
            if constraints:
                execution_order = self._apply_constraints(execution_order, constraints)
            
            return execution_order
            
        except Exception as e:
            logger.error(f"HTN planning failed: {e}")
            return []
    
    def _apply_constraints(self, execution_order: List[str], constraints: List[Dict[str, Any]]) -> List[str]:
        """应用约束"""
        try:
            # 简化的约束应用
            for constraint in constraints:
                constraint_type = constraint.get('type', '')
                if constraint_type == 'deadline':
                    # 应用截止时间约束
                    deadline_tasks = constraint.get('tasks', [])
                    # 重新排序以优先处理截止时间任务
                    deadline_tasks_in_order = [t for t in execution_order if t in deadline_tasks]
                    non_deadline_tasks = [t for t in execution_order if t not in deadline_tasks]
                    execution_order = deadline_tasks_in_order + non_deadline_tasks
            
            return execution_order
            
        except Exception as e:
            logger.error(f"Constraint application failed: {e}")
            return execution_order
    
    def _calculate_plan_cost(self, plan: Plan) -> float:
        """计算计划成本"""
        try:
            # 简化的成本计算：基于任务优先级和持续时间
            total_cost = 0.0
            for task in plan.tasks:
                priority_weight = task.priority.value
                duration_weight = task.estimated_duration.total_seconds() / 60.0  # 转换为分钟
                task_cost = priority_weight * duration_weight
                total_cost += task_cost
            
            return total_cost
            
        except Exception as e:
            logger.error(f"Cost calculation failed: {e}")
            return 0.0
    
    def _calculate_plan_quality(self, plan: Plan) -> float:
        """计算计划质量"""
        try:
            # 简化的质量计算：基于任务完成度和资源利用率
            if not plan.tasks:
                return 0.0
            
            # 任务完成度
            completion_score = sum(task.progress for task in plan.tasks) / len(plan.tasks)
            
            # 资源利用率（简化）
            resource_score = 0.8  # 假设80%的资源利用率
            
            # 综合质量分数
            quality_score = (completion_score * 0.7 + resource_score * 0.3)
            return min(1.0, max(0.0, quality_score))
            
        except Exception as e:
            logger.error(f"Quality calculation failed: {e}")
            return 0.0
    
    async def get_plan(self, plan_id: str) -> Optional[Plan]:
        """获取计划"""
        return self.plans.get(plan_id)
    
    async def get_planning_result(self, result_id: str) -> Optional[PlanningResult]:
        """获取规划结果"""
        return self.planning_results.get(result_id)
    
    async def update_task_status(self, task_id: str, status: TaskStatus, progress: float = None):
        """更新任务状态"""
        try:
            if task_id in self.task_graph.tasks:
                task = self.task_graph.tasks[task_id]
                task.status = status
                
                if progress is not None:
                    task.progress = progress
                
                if status == TaskStatus.RUNNING and task.started_at is None:
                    task.started_at = datetime.now()
                elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    task.completed_at = datetime.now()
                
                logger.info(f"Updated task {task_id} status to {status.value}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to update task status: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_tasks": len(self.task_graph.tasks),
            "total_plans": len(self.plans),
            "total_planning_results": len(self.planning_results),
            "task_graph_nodes": self.task_graph.graph.number_of_nodes(),
            "task_graph_edges": self.task_graph.graph.number_of_edges(),
            "successful_plans": len([r for r in self.planning_results.values() if r.success]),
            "failed_plans": len([r for r in self.planning_results.values() if not r.success]),
            "average_planning_time": sum(r.execution_time.total_seconds() for r in self.planning_results.values()) / max(len(self.planning_results), 1)
        }

# 示例用法
async def main_demo():
    """规划引擎演示"""
    # 创建规划引擎配置
    config = {
        "max_tasks": 1000,
        "max_plans": 100
    }
    
    # 创建规划引擎
    planning_engine = PlanningEngine(config)
    await planning_engine.start()
    
    # 创建示例任务
    print("创建示例任务...")
    tasks = [
        Task(
            name="数据收集",
            description="收集项目所需的数据",
            task_type="data_collection",
            priority=TaskPriority.HIGH,
            estimated_duration=timedelta(hours=2)
        ),
        Task(
            name="数据预处理",
            description="清洗和预处理收集的数据",
            task_type="data_processing",
            priority=TaskPriority.HIGH,
            estimated_duration=timedelta(hours=1),
            dependencies=["数据收集"]
        ),
        Task(
            name="模型训练",
            description="训练机器学习模型",
            task_type="model_training",
            priority=TaskPriority.NORMAL,
            estimated_duration=timedelta(hours=4),
            dependencies=["数据预处理"]
        ),
        Task(
            name="模型评估",
            description="评估模型性能",
            task_type="model_evaluation",
            priority=TaskPriority.NORMAL,
            estimated_duration=timedelta(hours=1),
            dependencies=["模型训练"]
        ),
        Task(
            name="报告生成",
            description="生成项目报告",
            task_type="report_generation",
            priority=TaskPriority.LOW,
            estimated_duration=timedelta(hours=1),
            dependencies=["模型评估"]
        )
    ]
    
    # 添加任务
    for task in tasks:
        await planning_engine.add_task(task)
    
    # 添加任务依赖
    await planning_engine.add_task_dependency("数据预处理", "数据收集")
    await planning_engine.add_task_dependency("模型训练", "数据预处理")
    await planning_engine.add_task_dependency("模型评估", "模型训练")
    await planning_engine.add_task_dependency("报告生成", "模型评估")
    
    # 生成计划
    print("\n生成计划...")
    constraints = [
        {
            "type": "deadline",
            "tasks": ["数据收集", "数据预处理"],
            "deadline": datetime.now() + timedelta(hours=4)
        }
    ]
    
    result = await planning_engine.generate_plan(
        PlanningType.TASK_PLANNING,
        PlanningAlgorithm.A_STAR,
        tasks,
        constraints
    )
    
    if result.success:
        print(f"✓ 计划生成成功")
        print(f"  计划名称: {result.plan.name}")
        print(f"  执行顺序: {result.plan.execution_order}")
        print(f"  总持续时间: {result.plan.estimated_total_duration}")
        print(f"  计划成本: {result.cost:.2f}")
        print(f"  质量分数: {result.quality_score:.2f}")
        print(f"  规划时间: {result.execution_time.total_seconds():.2f}秒")
    else:
        print(f"✗ 计划生成失败")
        print(f"  错误: {', '.join(result.errors)}")
    
    # 更新任务状态
    print("\n更新任务状态...")
    await planning_engine.update_task_status("数据收集", TaskStatus.COMPLETED, 1.0)
    await planning_engine.update_task_status("数据预处理", TaskStatus.RUNNING, 0.5)
    
    # 获取统计信息
    print("\n规划引擎统计:")
    stats = planning_engine.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 停止规划引擎
    await planning_engine.stop()
    print("\n规划引擎演示完成")

if __name__ == "__main__":
    asyncio.run(main_demo())
