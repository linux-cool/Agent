# business_automation.py
"""
第6章 企业级智能体应用 - 业务流程自动化
实现企业业务流程的自动化，包括工作流引擎、任务调度、规则引擎等
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
import threading
import time
from collections import defaultdict, deque
import re
import yaml
import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProcessStatus(Enum):
    """流程状态枚举"""
    DRAFT = "草稿"
    ACTIVE = "活跃"
    RUNNING = "运行中"
    PAUSED = "暂停"
    COMPLETED = "已完成"
    FAILED = "失败"
    CANCELLED = "已取消"

class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "待执行"
    READY = "就绪"
    RUNNING = "执行中"
    COMPLETED = "已完成"
    FAILED = "失败"
    SKIPPED = "已跳过"
    CANCELLED = "已取消"

class NodeType(Enum):
    """节点类型枚举"""
    START = "开始"
    END = "结束"
    TASK = "任务"
    GATEWAY = "网关"
    EVENT = "事件"
    SUBPROCESS = "子流程"

class GatewayType(Enum):
    """网关类型枚举"""
    EXCLUSIVE = "排他网关"
    PARALLEL = "并行网关"
    INCLUSIVE = "包容网关"
    EVENT_BASED = "基于事件"

class EventType(Enum):
    """事件类型枚举"""
    TIMER = "定时器"
    MESSAGE = "消息"
    SIGNAL = "信号"
    CONDITION = "条件"
    ERROR = "错误"

class RuleType(Enum):
    """规则类型枚举"""
    CONDITION = "条件规则"
    ACTION = "动作规则"
    TRANSFORMATION = "转换规则"
    VALIDATION = "验证规则"

@dataclass
class ProcessDefinition:
    """流程定义"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    version: str = "1.0"
    category: str = ""
    status: ProcessStatus = ProcessStatus.DRAFT
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "category": self.category,
            "status": self.status.value,
            "nodes": self.nodes,
            "edges": self.edges,
            "variables": self.variables,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by
        }

@dataclass
class ProcessInstance:
    """流程实例"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    process_definition_id: str = ""
    name: str = ""
    status: ProcessStatus = ProcessStatus.RUNNING
    variables: Dict[str, Any] = field(default_factory=dict)
    current_node_id: str = ""
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    started_by: str = ""
    assignee: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "process_definition_id": self.process_definition_id,
            "name": self.name,
            "status": self.status.value,
            "variables": self.variables,
            "current_node_id": self.current_node_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "started_by": self.started_by,
            "assignee": self.assignee
        }

@dataclass
class Task:
    """任务"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    process_instance_id: str = ""
    node_id: str = ""
    name: str = ""
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    assignee: Optional[str] = None
    due_date: Optional[datetime] = None
    priority: int = 1
    variables: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "process_instance_id": self.process_instance_id,
            "node_id": self.node_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "assignee": self.assignee,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "priority": self.priority,
            "variables": self.variables,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }

@dataclass
class BusinessRule:
    """业务规则"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    rule_type: RuleType = RuleType.CONDITION
    condition: str = ""
    action: str = ""
    priority: int = 1
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "rule_type": self.rule_type.value,
            "condition": self.condition,
            "action": self.action,
            "priority": self.priority,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

class WorkflowEngine:
    """工作流引擎"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.process_definitions: Dict[str, ProcessDefinition] = {}
        self.process_instances: Dict[str, ProcessInstance] = {}
        self.tasks: Dict[str, Task] = {}
        self.running = False
        self.execution_queue = asyncio.Queue()
        self.task_executors: Dict[str, Callable] = {}
        self._initialize_default_executors()
    
    def _initialize_default_executors(self):
        """初始化默认执行器"""
        self.task_executors = {
            "send_email": self._execute_send_email,
            "create_document": self._execute_create_document,
            "approve_request": self._execute_approve_request,
            "update_database": self._execute_update_database,
            "call_api": self._execute_call_api,
            "wait": self._execute_wait,
            "condition": self._execute_condition
        }
    
    async def start(self):
        """启动工作流引擎"""
        self.running = True
        logger.info("WorkflowEngine started")
    
    async def stop(self):
        """停止工作流引擎"""
        self.running = False
        logger.info("WorkflowEngine stopped")
    
    async def deploy_process(self, process_definition: ProcessDefinition) -> bool:
        """部署流程定义"""
        try:
            # 验证流程定义
            if not self._validate_process_definition(process_definition):
                logger.error(f"Invalid process definition: {process_definition.name}")
                return False
            
            # 部署流程
            process_definition.status = ProcessStatus.ACTIVE
            self.process_definitions[process_definition.id] = process_definition
            
            logger.info(f"Process deployed: {process_definition.name}")
            return True
            
        except Exception as e:
            logger.error(f"Process deployment failed: {e}")
            return False
    
    def _validate_process_definition(self, process_definition: ProcessDefinition) -> bool:
        """验证流程定义"""
        try:
            # 检查是否有开始和结束节点
            start_nodes = [node for node in process_definition.nodes if node.get("type") == "START"]
            end_nodes = [node for node in process_definition.nodes if node.get("type") == "END"]
            
            if not start_nodes:
                logger.error("Process must have at least one start node")
                return False
            
            if not end_nodes:
                logger.error("Process must have at least one end node")
                return False
            
            # 检查节点ID唯一性
            node_ids = [node.get("id") for node in process_definition.nodes]
            if len(node_ids) != len(set(node_ids)):
                logger.error("Node IDs must be unique")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Process validation failed: {e}")
            return False
    
    async def start_process(self, process_definition_id: str, variables: Dict[str, Any] = None, 
                          started_by: str = "system") -> Optional[ProcessInstance]:
        """启动流程实例"""
        try:
            process_definition = self.process_definitions.get(process_definition_id)
            if not process_definition:
                logger.error(f"Process definition not found: {process_definition_id}")
                return None
            
            # 创建流程实例
            instance = ProcessInstance(
                process_definition_id=process_definition_id,
                name=f"{process_definition.name} - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                variables=variables or {},
                started_by=started_by
            )
            
            # 设置开始节点
            start_nodes = [node for node in process_definition.nodes if node.get("type") == "START"]
            if start_nodes:
                instance.current_node_id = start_nodes[0]["id"]
            
            self.process_instances[instance.id] = instance
            
            # 开始执行流程
            await self._execute_process(instance)
            
            logger.info(f"Process instance started: {instance.id}")
            return instance
            
        except Exception as e:
            logger.error(f"Process start failed: {e}")
            return None
    
    async def _execute_process(self, instance: ProcessInstance):
        """执行流程"""
        try:
            process_definition = self.process_definitions[instance.process_definition_id]
            
            while instance.status == ProcessStatus.RUNNING:
                current_node = self._get_node_by_id(process_definition, instance.current_node_id)
                if not current_node:
                    instance.status = ProcessStatus.FAILED
                    break
                
                # 执行当前节点
                next_node_id = await self._execute_node(instance, current_node)
                
                if next_node_id:
                    instance.current_node_id = next_node_id
                else:
                    # 流程结束
                    instance.status = ProcessStatus.COMPLETED
                    instance.completed_at = datetime.now()
                    break
                
                # 检查是否到达结束节点
                if current_node.get("type") == "END":
                    instance.status = ProcessStatus.COMPLETED
                    instance.completed_at = datetime.now()
                    break
            
        except Exception as e:
            logger.error(f"Process execution failed: {e}")
            instance.status = ProcessStatus.FAILED
    
    def _get_node_by_id(self, process_definition: ProcessDefinition, node_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取节点"""
        for node in process_definition.nodes:
            if node.get("id") == node_id:
                return node
        return None
    
    async def _execute_node(self, instance: ProcessInstance, node: Dict[str, Any]) -> Optional[str]:
        """执行节点"""
        try:
            node_type = node.get("type")
            
            if node_type == "START":
                return await self._execute_start_node(instance, node)
            elif node_type == "END":
                return await self._execute_end_node(instance, node)
            elif node_type == "TASK":
                return await self._execute_task_node(instance, node)
            elif node_type == "GATEWAY":
                return await self._execute_gateway_node(instance, node)
            elif node_type == "EVENT":
                return await self._execute_event_node(instance, node)
            else:
                logger.warning(f"Unknown node type: {node_type}")
                return None
                
        except Exception as e:
            logger.error(f"Node execution failed: {e}")
            return None
    
    async def _execute_start_node(self, instance: ProcessInstance, node: Dict[str, Any]) -> Optional[str]:
        """执行开始节点"""
        # 开始节点通常直接进入下一个节点
        return self._get_next_node_id(instance, node)
    
    async def _execute_end_node(self, instance: ProcessInstance, node: Dict[str, Any]) -> Optional[str]:
        """执行结束节点"""
        # 结束节点没有下一个节点
        return None
    
    async def _execute_task_node(self, instance: ProcessInstance, node: Dict[str, Any]) -> Optional[str]:
        """执行任务节点"""
        try:
            task_type = node.get("task_type", "default")
            
            # 创建任务
            task = Task(
                process_instance_id=instance.id,
                node_id=node["id"],
                name=node.get("name", "未命名任务"),
                description=node.get("description", ""),
                variables=node.get("variables", {})
            )
            
            self.tasks[task.id] = task
            
            # 执行任务
            if task_type in self.task_executors:
                result = await self.task_executors[task_type](task, instance)
                task.status = TaskStatus.COMPLETED if result else TaskStatus.FAILED
            else:
                logger.warning(f"Unknown task type: {task_type}")
                task.status = TaskStatus.FAILED
            
            task.completed_at = datetime.now()
            
            return self._get_next_node_id(instance, node)
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return None
    
    async def _execute_gateway_node(self, instance: ProcessInstance, node: Dict[str, Any]) -> Optional[str]:
        """执行网关节点"""
        try:
            gateway_type = node.get("gateway_type", "EXCLUSIVE")
            
            if gateway_type == "EXCLUSIVE":
                return await self._execute_exclusive_gateway(instance, node)
            elif gateway_type == "PARALLEL":
                return await self._execute_parallel_gateway(instance, node)
            else:
                logger.warning(f"Unknown gateway type: {gateway_type}")
                return None
                
        except Exception as e:
            logger.error(f"Gateway execution failed: {e}")
            return None
    
    async def _execute_exclusive_gateway(self, instance: ProcessInstance, node: Dict[str, Any]) -> Optional[str]:
        """执行排他网关"""
        # 简化的排他网关实现
        conditions = node.get("conditions", [])
        for condition in conditions:
            if self._evaluate_condition(condition["expression"], instance.variables):
                return condition["target_node_id"]
        
        # 默认路径
        return node.get("default_path")
    
    async def _execute_parallel_gateway(self, instance: ProcessInstance, node: Dict[str, Any]) -> Optional[str]:
        """执行并行网关"""
        # 简化的并行网关实现
        # 在实际实现中，这里需要处理并行分支的同步
        return self._get_next_node_id(instance, node)
    
    async def _execute_event_node(self, instance: ProcessInstance, node: Dict[str, Any]) -> Optional[str]:
        """执行事件节点"""
        try:
            event_type = node.get("event_type", "TIMER")
            
            if event_type == "TIMER":
                duration = node.get("duration", "PT1H")
                await self._execute_wait(None, instance, duration=duration)
            elif event_type == "MESSAGE":
                # 等待消息事件
                await self._wait_for_message(node.get("message_name"))
            
            return self._get_next_node_id(instance, node)
            
        except Exception as e:
            logger.error(f"Event execution failed: {e}")
            return None
    
    def _get_next_node_id(self, instance: ProcessInstance, node: Dict[str, Any]) -> Optional[str]:
        """获取下一个节点ID"""
        process_definition = self.process_definitions[instance.process_definition_id]
        
        # 查找从当前节点出发的边
        for edge in process_definition.edges:
            if edge.get("source") == node["id"]:
                return edge.get("target")
        
        return None
    
    def _evaluate_condition(self, expression: str, variables: Dict[str, Any]) -> bool:
        """评估条件表达式"""
        try:
            # 简化的条件评估
            # 在实际实现中，这里需要使用更安全的表达式解析器
            for key, value in variables.items():
                expression = expression.replace(f"${{{key}}}", str(value))
            
            # 简单的条件评估
            if "==" in expression:
                left, right = expression.split("==")
                return left.strip() == right.strip()
            elif ">" in expression:
                left, right = expression.split(">")
                return float(left.strip()) > float(right.strip())
            elif "<" in expression:
                left, right = expression.split("<")
                return float(left.strip()) < float(right.strip())
            
            return False
            
        except Exception as e:
            logger.error(f"Condition evaluation failed: {e}")
            return False
    
    # 任务执行器
    async def _execute_send_email(self, task: Task, instance: ProcessInstance) -> bool:
        """执行发送邮件任务"""
        logger.info(f"Executing send email task: {task.name}")
        await asyncio.sleep(0.1)  # 模拟执行时间
        return True
    
    async def _execute_create_document(self, task: Task, instance: ProcessInstance) -> bool:
        """执行创建文档任务"""
        logger.info(f"Executing create document task: {task.name}")
        await asyncio.sleep(0.2)  # 模拟执行时间
        return True
    
    async def _execute_approve_request(self, task: Task, instance: ProcessInstance) -> bool:
        """执行审批任务"""
        logger.info(f"Executing approve request task: {task.name}")
        await asyncio.sleep(0.1)  # 模拟执行时间
        return True
    
    async def _execute_update_database(self, task: Task, instance: ProcessInstance) -> bool:
        """执行更新数据库任务"""
        logger.info(f"Executing update database task: {task.name}")
        await asyncio.sleep(0.1)  # 模拟执行时间
        return True
    
    async def _execute_call_api(self, task: Task, instance: ProcessInstance) -> bool:
        """执行调用API任务"""
        logger.info(f"Executing call API task: {task.name}")
        await asyncio.sleep(0.1)  # 模拟执行时间
        return True
    
    async def _execute_wait(self, task: Task, instance: ProcessInstance, duration: str = "PT1H") -> bool:
        """执行等待任务"""
        logger.info(f"Executing wait task: {task.name if task else 'wait'}")
        # 解析ISO 8601持续时间
        if duration.startswith("PT"):
            duration_str = duration[2:]
            if "H" in duration_str:
                hours = int(duration_str.split("H")[0])
                await asyncio.sleep(hours * 0.01)  # 模拟等待
            elif "M" in duration_str:
                minutes = int(duration_str.split("M")[0])
                await asyncio.sleep(minutes * 0.01)  # 模拟等待
        return True
    
    async def _execute_condition(self, task: Task, instance: ProcessInstance) -> bool:
        """执行条件任务"""
        logger.info(f"Executing condition task: {task.name}")
        return True
    
    async def _wait_for_message(self, message_name: str):
        """等待消息"""
        logger.info(f"Waiting for message: {message_name}")
        await asyncio.sleep(0.1)  # 模拟等待
    
    def get_process_instance(self, instance_id: str) -> Optional[ProcessInstance]:
        """获取流程实例"""
        return self.process_instances.get(instance_id)
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务"""
        return self.tasks.get(task_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_process_definitions": len(self.process_definitions),
            "total_process_instances": len(self.process_instances),
            "active_process_instances": len([i for i in self.process_instances.values() if i.status == ProcessStatus.RUNNING]),
            "completed_process_instances": len([i for i in self.process_instances.values() if i.status == ProcessStatus.COMPLETED]),
            "total_tasks": len(self.tasks),
            "completed_tasks": len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED])
        }

class RuleEngine:
    """规则引擎"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rules: Dict[str, BusinessRule] = {}
        self.rule_cache: Dict[str, List[BusinessRule]] = defaultdict(list)
    
    async def add_rule(self, rule: BusinessRule) -> bool:
        """添加规则"""
        try:
            self.rules[rule.id] = rule
            self._update_rule_cache(rule)
            logger.info(f"Rule added: {rule.name}")
            return True
        except Exception as e:
            logger.error(f"Rule addition failed: {e}")
            return False
    
    def _update_rule_cache(self, rule: BusinessRule):
        """更新规则缓存"""
        # 根据规则类型缓存
        self.rule_cache[rule.rule_type.value].append(rule)
    
    async def evaluate_rules(self, context: Dict[str, Any], rule_type: RuleType = None) -> List[Dict[str, Any]]:
        """评估规则"""
        try:
            results = []
            
            # 获取要评估的规则
            rules_to_evaluate = []
            if rule_type:
                rules_to_evaluate = self.rule_cache.get(rule_type.value, [])
            else:
                rules_to_evaluate = list(self.rules.values())
            
            # 按优先级排序
            rules_to_evaluate.sort(key=lambda r: r.priority, reverse=True)
            
            for rule in rules_to_evaluate:
                if not rule.is_active:
                    continue
                
                # 评估条件
                if await self._evaluate_condition(rule.condition, context):
                    # 执行动作
                    action_result = await self._execute_action(rule.action, context)
                    
                    results.append({
                        "rule_id": rule.id,
                        "rule_name": rule.name,
                        "rule_type": rule.rule_type.value,
                        "condition": rule.condition,
                        "action": rule.action,
                        "action_result": action_result,
                        "priority": rule.priority
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Rule evaluation failed: {e}")
            return []
    
    async def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """评估条件"""
        try:
            # 替换变量
            evaluated_condition = condition
            for key, value in context.items():
                evaluated_condition = evaluated_condition.replace(f"${{{key}}}", str(value))
            
            # 简化的条件评估
            if "==" in evaluated_condition:
                left, right = evaluated_condition.split("==")
                return left.strip() == right.strip()
            elif ">" in evaluated_condition:
                left, right = evaluated_condition.split(">")
                return float(left.strip()) > float(right.strip())
            elif "<" in evaluated_condition:
                left, right = evaluated_condition.split("<")
                return float(left.strip()) < float(right.strip())
            elif "contains" in evaluated_condition:
                # 处理包含条件
                parts = evaluated_condition.split("contains")
                if len(parts) == 2:
                    left = parts[0].strip()
                    right = parts[1].strip()
                    return right in left
            
            return False
            
        except Exception as e:
            logger.error(f"Condition evaluation failed: {e}")
            return False
    
    async def _execute_action(self, action: str, context: Dict[str, Any]) -> Any:
        """执行动作"""
        try:
            # 简化的动作执行
            if action.startswith("set_"):
                # 设置变量
                var_name = action[4:]
                context[var_name] = True
                return f"Set {var_name} = True"
            elif action.startswith("send_"):
                # 发送通知
                notification_type = action[5:]
                return f"Sent {notification_type} notification"
            elif action.startswith("log_"):
                # 记录日志
                message = action[4:]
                logger.info(f"Rule action: {message}")
                return f"Logged: {message}"
            else:
                return f"Executed action: {action}"
                
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return None

class BusinessAutomationSystem:
    """业务流程自动化系统"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.workflow_engine = WorkflowEngine(config.get("workflow", {}))
        self.rule_engine = RuleEngine(config.get("rules", {}))
        self.process_templates = self._load_process_templates()
        self.business_rules = self._load_business_rules()
    
    def _load_process_templates(self) -> Dict[str, ProcessDefinition]:
        """加载流程模板"""
        templates = {}
        
        # 员工入职流程模板
        onboarding_process = ProcessDefinition(
            name="员工入职流程",
            description="新员工入职的完整流程",
            category="HR",
            nodes=[
                {"id": "start", "type": "START", "name": "开始"},
                {"id": "create_account", "type": "TASK", "name": "创建账户", "task_type": "create_document"},
                {"id": "send_welcome_email", "type": "TASK", "name": "发送欢迎邮件", "task_type": "send_email"},
                {"id": "setup_workspace", "type": "TASK", "name": "设置工作空间", "task_type": "update_database"},
                {"id": "schedule_training", "type": "TASK", "name": "安排培训", "task_type": "call_api"},
                {"id": "end", "type": "END", "name": "结束"}
            ],
            edges=[
                {"source": "start", "target": "create_account"},
                {"source": "create_account", "target": "send_welcome_email"},
                {"source": "send_welcome_email", "target": "setup_workspace"},
                {"source": "setup_workspace", "target": "schedule_training"},
                {"source": "schedule_training", "target": "end"}
            ]
        )
        templates["employee_onboarding"] = onboarding_process
        
        # 采购审批流程模板
        purchase_process = ProcessDefinition(
            name="采购审批流程",
            description="采购申请的审批流程",
            category="Finance",
            nodes=[
                {"id": "start", "type": "START", "name": "开始"},
                {"id": "submit_request", "type": "TASK", "name": "提交申请", "task_type": "create_document"},
                {"id": "manager_approval", "type": "TASK", "name": "经理审批", "task_type": "approve_request"},
                {"id": "finance_approval", "type": "TASK", "name": "财务审批", "task_type": "approve_request"},
                {"id": "procurement", "type": "TASK", "name": "执行采购", "task_type": "call_api"},
                {"id": "end", "type": "END", "name": "结束"}
            ],
            edges=[
                {"source": "start", "target": "submit_request"},
                {"source": "submit_request", "target": "manager_approval"},
                {"source": "manager_approval", "target": "finance_approval"},
                {"source": "finance_approval", "target": "procurement"},
                {"source": "procurement", "target": "end"}
            ]
        )
        templates["purchase_approval"] = purchase_process
        
        return templates
    
    def _load_business_rules(self) -> List[BusinessRule]:
        """加载业务规则"""
        rules = [
            BusinessRule(
                name="高金额采购规则",
                description="超过10000元的采购需要额外审批",
                rule_type=RuleType.CONDITION,
                condition="${amount} > 10000",
                action="send_high_value_approval",
                priority=1
            ),
            BusinessRule(
                name="紧急采购规则",
                description="紧急采购可以跳过部分审批步骤",
                rule_type=RuleType.CONDITION,
                condition="${urgency} == 'high'",
                action="set_fast_track",
                priority=2
            ),
            BusinessRule(
                name="部门预算规则",
                description="检查部门预算是否充足",
                rule_type=RuleType.VALIDATION,
                condition="${department_budget} >= ${amount}",
                action="log_budget_check",
                priority=3
            )
        ]
        return rules
    
    async def start(self):
        """启动系统"""
        await self.workflow_engine.start()
        
        # 部署流程模板
        for template in self.process_templates.values():
            await self.workflow_engine.deploy_process(template)
        
        # 添加业务规则
        for rule in self.business_rules:
            await self.rule_engine.add_rule(rule)
        
        logger.info("BusinessAutomationSystem started")
    
    async def stop(self):
        """停止系统"""
        await self.workflow_engine.stop()
        logger.info("BusinessAutomationSystem stopped")
    
    async def start_process(self, process_name: str, variables: Dict[str, Any] = None, 
                          started_by: str = "system") -> Optional[ProcessInstance]:
        """启动流程"""
        try:
            # 查找流程模板
            template = None
            for t in self.process_templates.values():
                if t.name == process_name:
                    template = t
                    break
            
            if not template:
                logger.error(f"Process template not found: {process_name}")
                return None
            
            # 启动流程实例
            instance = await self.workflow_engine.start_process(template.id, variables, started_by)
            
            if instance:
                # 应用业务规则
                await self._apply_business_rules(instance)
            
            return instance
            
        except Exception as e:
            logger.error(f"Process start failed: {e}")
            return None
    
    async def _apply_business_rules(self, instance: ProcessInstance):
        """应用业务规则"""
        try:
            # 评估规则
            rule_results = await self.rule_engine.evaluate_rules(instance.variables)
            
            # 根据规则结果调整流程
            for result in rule_results:
                if result["action"] == "set_fast_track":
                    # 设置快速通道
                    instance.variables["fast_track"] = True
                    logger.info(f"Fast track enabled for instance {instance.id}")
                elif result["action"] == "send_high_value_approval":
                    # 发送高价值审批通知
                    logger.info(f"High value approval required for instance {instance.id}")
            
        except Exception as e:
            logger.error(f"Business rules application failed: {e}")
    
    def get_process_templates(self) -> List[ProcessDefinition]:
        """获取流程模板"""
        return list(self.process_templates.values())
    
    def get_process_instance(self, instance_id: str) -> Optional[ProcessInstance]:
        """获取流程实例"""
        return self.workflow_engine.get_process_instance(instance_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        workflow_stats = self.workflow_engine.get_stats()
        return {
            **workflow_stats,
            "total_process_templates": len(self.process_templates),
            "total_business_rules": len(self.business_rules)
        }

# 示例用法
async def main_demo():
    """业务流程自动化演示"""
    config = {
        "workflow": {},
        "rules": {}
    }
    
    # 创建业务流程自动化系统
    automation_system = BusinessAutomationSystem(config)
    await automation_system.start()
    
    print("🔄 业务流程自动化演示")
    print("=" * 50)
    
    # 1. 显示可用的流程模板
    print("\n1. 可用的流程模板:")
    templates = automation_system.get_process_templates()
    for template in templates:
        print(f"  - {template.name}: {template.description}")
        print(f"    节点数: {len(template.nodes)}, 边数: {len(template.edges)}")
    
    # 2. 启动员工入职流程
    print("\n2. 启动员工入职流程...")
    onboarding_variables = {
        "employee_name": "张三",
        "department": "技术部",
        "position": "软件工程师",
        "start_date": "2025-01-01"
    }
    
    onboarding_instance = await automation_system.start_process(
        "员工入职流程", 
        onboarding_variables, 
        "HR系统"
    )
    
    if onboarding_instance:
        print(f"✓ 入职流程已启动: {onboarding_instance.id}")
        print(f"  状态: {onboarding_instance.status.value}")
        print(f"  当前节点: {onboarding_instance.current_node_id}")
        
        # 等待流程完成
        await asyncio.sleep(0.5)
        
        # 检查流程状态
        updated_instance = automation_system.get_process_instance(onboarding_instance.id)
        if updated_instance:
            print(f"  最终状态: {updated_instance.status.value}")
            if updated_instance.completed_at:
                print(f"  完成时间: {updated_instance.completed_at}")
    
    # 3. 启动采购审批流程
    print("\n3. 启动采购审批流程...")
    purchase_variables = {
        "item": "办公设备",
        "amount": 15000,
        "department": "行政部",
        "urgency": "normal",
        "department_budget": 50000
    }
    
    purchase_instance = await automation_system.start_process(
        "采购审批流程", 
        purchase_variables, 
        "采购系统"
    )
    
    if purchase_instance:
        print(f"✓ 采购流程已启动: {purchase_instance.id}")
        print(f"  状态: {purchase_instance.status.value}")
        
        # 等待流程完成
        await asyncio.sleep(0.5)
        
        # 检查流程状态
        updated_instance = automation_system.get_process_instance(purchase_instance.id)
        if updated_instance:
            print(f"  最终状态: {updated_instance.status.value}")
    
    # 4. 显示系统统计信息
    print("\n4. 系统统计信息:")
    stats = automation_system.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    await automation_system.stop()
    print("\n🎉 业务流程自动化演示完成")

if __name__ == "__main__":
    asyncio.run(main_demo())
