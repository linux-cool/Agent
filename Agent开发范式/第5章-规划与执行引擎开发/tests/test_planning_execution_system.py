# test_planning_execution_system.py
"""
第5章 规划与执行引擎开发 - 测试用例
测试规划引擎、执行引擎、任务调度器、资源管理器和监控系统
"""

import pytest
import asyncio
import json
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

# 导入被测试的模块
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))

from planning_engine import PlanningEngine, Plan, PlanStatus, PlanningStrategy
from execution_engine import ExecutionEngine, ExecutionStatus, ExecutionResult
from task_scheduler import TaskScheduler, ScheduledTask, ResourceRequirement, ResourceType, SchedulingPolicy
from resource_manager import ResourceManager, ResourceRequest, ResourcePool, ResourceSpec, AllocationStrategy
from monitoring_system import MonitoringSystem, AlertRule, AlertSeverity, Metric, MetricType

class TestPlanningEngine:
    """测试规划引擎"""
    
    @pytest.fixture
    def planning_engine(self):
        """创建规划引擎实例"""
        config = {
            "planning_strategy": "DELIBERATIVE",
            "llm_model": "gpt-4",
            "api_key": "test_key"
        }
        return PlanningEngine(config)
    
    @pytest.mark.asyncio
    async def test_plan_creation(self, planning_engine):
        """测试计划创建"""
        plan = Plan("test_plan", "测试目标")
        assert plan.plan_id == "test_plan"
        assert plan.goal == "测试目标"
        assert plan.status == PlanStatus.PENDING
        assert len(plan.steps) == 0
    
    @pytest.mark.asyncio
    async def test_plan_add_step(self, planning_engine):
        """测试添加计划步骤"""
        plan = Plan("test_plan", "测试目标")
        plan.add_step("步骤1", ["tool1"], ["capability1"])
        
        assert len(plan.steps) == 1
        assert plan.steps[0]["description"] == "步骤1"
        assert plan.steps[0]["required_tools"] == ["tool1"]
        assert plan.steps[0]["required_capabilities"] == ["capability1"]
    
    @pytest.mark.asyncio
    async def test_plan_update_step_status(self, planning_engine):
        """测试更新步骤状态"""
        plan = Plan("test_plan", "测试目标")
        plan.add_step("步骤1")
        
        result = plan.update_step_status(plan.steps[0]["step_id"], PlanStatus.COMPLETED, "结果")
        assert result is True
        assert plan.steps[0]["status"] == PlanStatus.COMPLETED.value
        assert plan.steps[0]["result"] == "结果"
    
    @pytest.mark.asyncio
    async def test_generate_plan_deliberative(self, planning_engine):
        """测试深思熟虑式规划"""
        goal = "开发一个简单的Web应用"
        current_state = {"user_needs": "CRUD功能"}
        available_tools = ["code_generator", "db_manager"]
        agent_capabilities = ["software_development"]
        
        plan = await planning_engine.generate_plan(goal, current_state, available_tools, agent_capabilities)
        
        assert plan is not None
        assert plan.goal == goal
        assert plan.strategy == PlanningStrategy.DELIBERATIVE
        assert len(plan.steps) > 0
        assert plan.status == PlanStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_generate_plan_reactive(self, planning_engine):
        """测试反应式规划"""
        goal = "处理紧急问题"
        current_state = {"urgent_issue": "critical_bug"}
        available_tools = ["debugger"]
        agent_capabilities = ["bug_fixing"]
        
        plan = await planning_engine.generate_plan(goal, current_state, available_tools, agent_capabilities, PlanningStrategy.REACTIVE)
        
        assert plan is not None
        assert plan.strategy == PlanningStrategy.REACTIVE
        assert len(plan.steps) > 0
    
    @pytest.mark.asyncio
    async def test_evaluate_plan(self, planning_engine):
        """测试计划评估"""
        plan = Plan("test_plan", "测试目标")
        plan.add_step("步骤1", ["tool1"])
        plan.add_step("步骤2", ["tool2"])
        
        expected_outcome = {"expected_value": "success"}
        evaluation = await planning_engine.evaluate_plan(plan, expected_outcome)
        
        assert "plan_id" in evaluation
        assert "overall_quality" in evaluation
        assert "score" in evaluation
        assert "feedback" in evaluation
        assert "is_feasible" in evaluation
    
    @pytest.mark.asyncio
    async def test_optimize_plan(self, planning_engine):
        """测试计划优化"""
        plan = Plan("test_plan", "测试目标")
        plan.add_step("步骤1", ["tool1"])
        plan.add_step("步骤2")  # 没有工具要求
        plan.add_step("步骤3", ["tool3"])
        
        original_steps = len(plan.steps)
        optimized_plan = await planning_engine.optimize_plan(plan)
        
        assert optimized_plan is not None
        # 优化后应该移除冗余步骤
        assert len(optimized_plan.steps) <= original_steps

class TestExecutionEngine:
    """测试执行引擎"""
    
    @pytest.fixture
    def execution_engine(self):
        """创建执行引擎实例"""
        config = {"execution_strategy": "sequential"}
        return ExecutionEngine(config)
    
    @pytest.mark.asyncio
    async def test_execution_result_creation(self, execution_engine):
        """测试执行结果创建"""
        result = ExecutionResult("step1", ExecutionStatus.COMPLETED, "输出", start_time=0.0, end_time=1.0)
        
        assert result.step_id == "step1"
        assert result.status == ExecutionStatus.COMPLETED
        assert result.output == "输出"
        assert result.duration == 1.0
    
    @pytest.mark.asyncio
    async def test_execute_plan_sequential(self, execution_engine):
        """测试顺序执行计划"""
        # 创建测试计划
        plan = Plan("test_plan", "测试目标")
        plan.add_step("步骤1", ["tool1"])
        plan.add_step("步骤2", ["tool2"])
        
        # 模拟工具管理器
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool = AsyncMock(return_value="工具执行结果")
        execution_engine.tool_manager = mock_tool_manager
        
        result = await execution_engine.execute_plan(plan)
        
        assert result["status"] == "completed"
        assert result["plan_id"] == "test_plan"
        assert len(result["results"]) == 2
    
    @pytest.mark.asyncio
    async def test_execute_plan_parallel(self, execution_engine):
        """测试并行执行计划"""
        config = {"execution_strategy": "parallel"}
        execution_engine = ExecutionEngine(config)
        
        plan = Plan("test_plan", "测试目标")
        plan.add_step("步骤1", ["tool1"])
        plan.add_step("步骤2", ["tool2"])
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool = AsyncMock(return_value="工具执行结果")
        execution_engine.tool_manager = mock_tool_manager
        
        result = await execution_engine.execute_plan(plan)
        
        assert result["status"] == "completed"
        assert len(result["results"]) == 2
    
    @pytest.mark.asyncio
    async def test_pause_resume_execution(self, execution_engine):
        """测试暂停和恢复执行"""
        plan = Plan("test_plan", "测试目标")
        plan.add_step("步骤1", ["tool1"])
        
        # 启动执行
        execution_task = asyncio.create_task(execution_engine.execute_plan(plan))
        await asyncio.sleep(0.1)
        
        # 暂停执行
        await execution_engine.pause_execution(plan.plan_id)
        assert execution_engine.active_executions[plan.plan_id]["overall_status"] == ExecutionStatus.PAUSED
        
        # 恢复执行
        await execution_engine.resume_execution(plan.plan_id)
        assert execution_engine.active_executions[plan.plan_id]["overall_status"] == ExecutionStatus.RUNNING
        
        # 等待执行完成
        await execution_task
    
    @pytest.mark.asyncio
    async def test_cancel_execution(self, execution_engine):
        """测试取消执行"""
        plan = Plan("test_plan", "测试目标")
        plan.add_step("步骤1", ["tool1"])
        
        # 启动执行
        execution_task = asyncio.create_task(execution_engine.execute_plan(plan))
        await asyncio.sleep(0.1)
        
        # 取消执行
        await execution_engine.cancel_execution(plan.plan_id)
        assert plan.plan_id not in execution_engine.active_executions
        
        # 等待任务完成
        await execution_task

class TestTaskScheduler:
    """测试任务调度器"""
    
    @pytest.fixture
    def task_scheduler(self):
        """创建任务调度器实例"""
        config = {
            "scheduling_policy": "priority",
            "scheduling_algorithm": "EDF",
            "scheduling_interval": 1.0
        }
        return TaskScheduler(config)
    
    @pytest.mark.asyncio
    async def test_scheduled_task_creation(self, task_scheduler):
        """测试调度任务创建"""
        task = ScheduledTask(
            name="测试任务",
            description="测试任务描述",
            priority=1,
            estimated_duration=timedelta(minutes=5)
        )
        
        assert task.name == "测试任务"
        assert task.priority == 1
        assert task.status == "pending"
        assert task.progress == 0.0
    
    @pytest.mark.asyncio
    async def test_resource_requirement_creation(self, task_scheduler):
        """测试资源需求创建"""
        req = ResourceRequirement(
            resource_type=ResourceType.CPU,
            amount=2.0,
            unit="cores",
            duration=timedelta(minutes=10)
        )
        
        assert req.resource_type == ResourceType.CPU
        assert req.amount == 2.0
        assert req.unit == "cores"
    
    @pytest.mark.asyncio
    async def test_submit_task(self, task_scheduler):
        """测试提交任务"""
        task = ScheduledTask(
            name="测试任务",
            priority=1,
            estimated_duration=timedelta(minutes=5)
        )
        
        result = await task_scheduler.submit_task(task)
        assert result is True
        assert task.id in task_scheduler.scheduled_tasks
    
    @pytest.mark.asyncio
    async def test_update_task_status(self, task_scheduler):
        """测试更新任务状态"""
        task = ScheduledTask(
            name="测试任务",
            priority=1,
            estimated_duration=timedelta(minutes=5)
        )
        
        await task_scheduler.submit_task(task)
        
        result = await task_scheduler.update_task_status(task.id, "running", 0.5)
        assert result is True
        
        updated_task = task_scheduler.get_task(task.id)
        assert updated_task.status == "running"
        assert updated_task.progress == 0.5
    
    @pytest.mark.asyncio
    async def test_cancel_task(self, task_scheduler):
        """测试取消任务"""
        task = ScheduledTask(
            name="测试任务",
            priority=1,
            estimated_duration=timedelta(minutes=5)
        )
        
        await task_scheduler.submit_task(task)
        
        result = await task_scheduler.cancel_task(task.id)
        assert result is True
        
        updated_task = task_scheduler.get_task(task.id)
        assert updated_task.status == "cancelled"
    
    @pytest.mark.asyncio
    async def test_register_agent(self, task_scheduler):
        """测试注册代理"""
        agent_info = {
            "capabilities": ["data_processing"],
            "max_tasks": 3
        }
        
        task_scheduler.register_agent("agent_1", agent_info)
        assert "agent_1" in task_scheduler.agents
    
    @pytest.mark.asyncio
    async def test_get_stats(self, task_scheduler):
        """测试获取统计信息"""
        stats = task_scheduler.get_stats()
        
        assert "scheduling_policy" in stats
        assert "total_tasks" in stats
        assert "pending_tasks" in stats
        assert "registered_agents" in stats

class TestResourceManager:
    """测试资源管理器"""
    
    @pytest.fixture
    def resource_manager(self):
        """创建资源管理器实例"""
        config = {
            "monitoring_interval": 1.0,
            "optimization_enabled": True
        }
        return ResourceManager(config)
    
    @pytest.mark.asyncio
    async def test_resource_request_creation(self, resource_manager):
        """测试资源请求创建"""
        request = ResourceRequest(
            requester="agent_1",
            resource_type=ResourceType.CPU,
            amount=2.0,
            unit="cores",
            duration=timedelta(minutes=5),
            priority=1
        )
        
        assert request.requester == "agent_1"
        assert request.resource_type == ResourceType.CPU
        assert request.amount == 2.0
        assert request.status == "pending"
    
    @pytest.mark.asyncio
    async def test_resource_pool_creation(self, resource_manager):
        """测试资源池创建"""
        pool = ResourcePool(
            name="CPU池",
            resource_spec=ResourceSpec(ResourceType.CPU, 8.0, "cores", 1.0),
            available_amount=8.0,
            allocation_strategy=AllocationStrategy.FIRST_FIT
        )
        
        assert pool.name == "CPU池"
        assert pool.resource_spec.resource_type == ResourceType.CPU
        assert pool.available_amount == 8.0
    
    @pytest.mark.asyncio
    async def test_request_resources(self, resource_manager):
        """测试请求资源"""
        request = ResourceRequest(
            requester="agent_1",
            resource_type=ResourceType.CPU,
            amount=2.0,
            unit="cores",
            duration=timedelta(minutes=5),
            priority=1
        )
        
        request_id = await resource_manager.request_resources(request)
        assert request_id == request.id
        assert request.id in resource_manager.requests
    
    @pytest.mark.asyncio
    async def test_deallocate_resources(self, resource_manager):
        """测试释放资源"""
        # 先创建一个分配
        allocation = ResourceAllocation(
            resource_type=ResourceType.CPU,
            amount=2.0,
            unit="cores",
            allocated_to="agent_1",
            duration=timedelta(minutes=5)
        )
        
        resource_manager.allocations[allocation.id] = allocation
        
        # 更新资源池
        for pool in resource_manager.resource_pools.values():
            if pool.resource_spec.resource_type == ResourceType.CPU:
                pool.available_amount -= allocation.amount
                pool.allocated_amount += allocation.amount
                break
        
        result = await resource_manager.deallocate_resources(allocation.id)
        assert result is True
        assert allocation.id not in resource_manager.allocations
    
    @pytest.mark.asyncio
    async def test_create_resource_pool(self, resource_manager):
        """测试创建资源池"""
        pool = ResourcePool(
            name="测试池",
            resource_spec=ResourceSpec(ResourceType.CPU, 4.0, "cores", 1.0),
            available_amount=4.0
        )
        
        pool_id = resource_manager.create_resource_pool(pool)
        assert pool_id == pool.id
        assert pool.id in resource_manager.resource_pools
    
    @pytest.mark.asyncio
    async def test_get_resource_utilization(self, resource_manager):
        """测试获取资源利用率"""
        utilization = resource_manager.get_resource_utilization()
        
        assert isinstance(utilization, dict)
        assert "CPU" in utilization
        assert "内存" in utilization
        assert "存储" in utilization
        assert "网络" in utilization

class TestMonitoringSystem:
    """测试监控系统"""
    
    @pytest.fixture
    def monitoring_system(self):
        """创建监控系统实例"""
        config = {
            "collection_interval": 1.0,
            "evaluation_interval": 5.0,
            "db_path": ":memory:"  # 使用内存数据库
        }
        return MonitoringSystem(config)
    
    def test_metric_creation(self, monitoring_system):
        """测试指标创建"""
        metric = Metric(
            name="test_metric",
            value=42.0,
            metric_type=MetricType.GAUGE,
            labels={"host": "localhost"}
        )
        
        assert metric.name == "test_metric"
        assert metric.value == 42.0
        assert metric.metric_type == MetricType.GAUGE
        assert metric.labels["host"] == "localhost"
    
    def test_alert_rule_creation(self, monitoring_system):
        """测试告警规则创建"""
        rule = AlertRule(
            name="测试告警",
            description="测试告警描述",
            metric_name="test_metric",
            condition="value > 80",
            severity=AlertSeverity.WARNING
        )
        
        assert rule.name == "测试告警"
        assert rule.metric_name == "test_metric"
        assert rule.condition == "value > 80"
        assert rule.severity == AlertSeverity.WARNING
    
    def test_create_alert_rule(self, monitoring_system):
        """测试创建告警规则"""
        rule = AlertRule(
            name="测试告警",
            description="测试告警描述",
            metric_name="test_metric",
            condition="value > 80",
            severity=AlertSeverity.WARNING
        )
        
        rule_id = monitoring_system.create_alert_rule(rule)
        assert rule_id == rule.id
        assert rule.id in monitoring_system.alert_manager.rules
    
    def test_get_active_alerts(self, monitoring_system):
        """测试获取活跃告警"""
        alerts = monitoring_system.get_active_alerts()
        assert isinstance(alerts, list)
    
    def test_acknowledge_alert(self, monitoring_system):
        """测试确认告警"""
        # 创建一个测试告警
        alert = Alert(
            rule_id="test_rule",
            name="测试告警",
            description="测试告警描述",
            severity=AlertSeverity.WARNING
        )
        
        monitoring_system.alert_manager.active_alerts[alert.id] = alert
        
        result = monitoring_system.acknowledge_alert(alert.id, "admin")
        assert result is True
        assert alert.status == AlertStatus.ACKNOWLEDGED
        assert alert.acknowledged_by == "admin"
    
    def test_create_dashboard(self, monitoring_system):
        """测试创建仪表板"""
        dashboard = Dashboard(
            name="测试仪表板",
            description="测试仪表板描述",
            widgets=[
                {
                    "type": "line_chart",
                    "title": "测试图表",
                    "metric": "test_metric"
                }
            ]
        )
        
        dashboard_id = monitoring_system.create_dashboard(dashboard)
        assert dashboard_id == dashboard.id
        assert dashboard.id in monitoring_system.dashboards
    
    def test_get_stats(self, monitoring_system):
        """测试获取统计信息"""
        stats = monitoring_system.get_stats()
        
        assert "active_alerts" in stats
        assert "total_dashboards" in stats
        assert "registered_collectors" in stats
        assert "collection_interval" in stats
        assert "evaluation_interval" in stats

class TestIntegration:
    """集成测试"""
    
    @pytest.mark.asyncio
    async def test_planning_execution_integration(self):
        """测试规划与执行集成"""
        # 创建规划引擎
        planning_config = {"planning_strategy": "DELIBERATIVE"}
        planning_engine = PlanningEngine(planning_config)
        
        # 创建执行引擎
        execution_config = {"execution_strategy": "sequential"}
        execution_engine = ExecutionEngine(execution_config)
        
        # 生成计划
        goal = "开发一个简单的应用"
        current_state = {"user_needs": "基本功能"}
        available_tools = ["code_generator"]
        agent_capabilities = ["software_development"]
        
        plan = await planning_engine.generate_plan(goal, current_state, available_tools, agent_capabilities)
        assert plan is not None
        
        # 执行计划
        result = await execution_engine.execute_plan(plan)
        assert result["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_scheduler_resource_manager_integration(self):
        """测试调度器与资源管理器集成"""
        # 创建任务调度器
        scheduler_config = {
            "scheduling_policy": "priority",
            "scheduling_algorithm": "EDF",
            "scheduling_interval": 1.0
        }
        scheduler = TaskScheduler(scheduler_config)
        
        # 创建资源管理器
        resource_config = {
            "monitoring_interval": 1.0,
            "optimization_enabled": True
        }
        resource_manager = ResourceManager(resource_config)
        
        # 创建任务
        task = ScheduledTask(
            name="测试任务",
            priority=1,
            estimated_duration=timedelta(minutes=5),
            resource_requirements=[
                ResourceRequirement(ResourceType.CPU, 2.0, "cores", timedelta(minutes=5))
            ]
        )
        
        # 提交任务
        await scheduler.submit_task(task)
        
        # 检查任务状态
        assert task.id in scheduler.scheduled_tasks
        
        # 检查资源管理器状态
        utilization = resource_manager.get_resource_utilization()
        assert isinstance(utilization, dict)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
