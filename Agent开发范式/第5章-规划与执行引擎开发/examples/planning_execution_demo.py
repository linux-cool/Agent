# planning_execution_demo.py
"""
第5章 规划与执行引擎开发 - 演示程序
演示规划引擎、执行引擎、任务调度器、资源管理器和监控系统的集成使用
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any

# 导入第5章的模块
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))

from planning_engine import PlanningEngine, Plan, PlanStatus, PlanningStrategy
from execution_engine import ExecutionEngine, ExecutionStatus, MockToolManager, MockAgentManager
from task_scheduler import TaskScheduler, ScheduledTask, ResourceRequirement, ResourceType, SchedulingPolicy
from resource_manager import ResourceManager, ResourceRequest, ResourcePool, ResourceSpec, AllocationStrategy
from monitoring_system import MonitoringSystem, AlertRule, AlertSeverity, Dashboard, Metric, MetricType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PlanningExecutionDemo:
    """规划与执行系统演示类"""
    
    def __init__(self):
        self.planning_engine = None
        self.execution_engine = None
        self.task_scheduler = None
        self.resource_manager = None
        self.monitoring_system = None
        self.running = False
    
    async def initialize_systems(self):
        """初始化所有系统"""
        print("🚀 初始化规划与执行系统...")
        
        # 1. 初始化规划引擎
        planning_config = {
            "planning_strategy": "DELIBERATIVE",
            "llm_model": "gpt-4",
            "api_key": "demo_key"
        }
        self.planning_engine = PlanningEngine(planning_config)
        print("✓ 规划引擎初始化完成")
        
        # 2. 初始化执行引擎
        execution_config = {
            "execution_strategy": "sequential"
        }
        mock_tool_manager = MockToolManager()
        mock_agent_manager = MockAgentManager()
        self.execution_engine = ExecutionEngine(execution_config, mock_tool_manager, mock_agent_manager)
        print("✓ 执行引擎初始化完成")
        
        # 3. 初始化任务调度器
        scheduler_config = {
            "scheduling_policy": "priority",
            "scheduling_algorithm": "EDF",
            "scheduling_interval": 1.0
        }
        self.task_scheduler = TaskScheduler(scheduler_config)
        await self.task_scheduler.start()
        print("✓ 任务调度器初始化完成")
        
        # 4. 初始化资源管理器
        resource_config = {
            "monitoring_interval": 1.0,
            "optimization_enabled": True
        }
        self.resource_manager = ResourceManager(resource_config)
        await self.resource_manager.start()
        print("✓ 资源管理器初始化完成")
        
        # 5. 初始化监控系统
        monitoring_config = {
            "collection_interval": 2.0,
            "evaluation_interval": 5.0,
            "db_path": "demo_monitoring.db"
        }
        self.monitoring_system = MonitoringSystem(monitoring_config)
        await self.monitoring_system.start()
        print("✓ 监控系统初始化完成")
        
        print("🎉 所有系统初始化完成！\n")
    
    async def setup_demo_environment(self):
        """设置演示环境"""
        print("🔧 设置演示环境...")
        
        # 注册代理到任务调度器
        agents = [
            {"id": "web_developer", "capabilities": ["web_development", "frontend", "backend"], "max_tasks": 3},
            {"id": "data_analyst", "capabilities": ["data_analysis", "statistics", "visualization"], "max_tasks": 2},
            {"id": "devops_engineer", "capabilities": ["deployment", "monitoring", "infrastructure"], "max_tasks": 4},
            {"id": "qa_tester", "capabilities": ["testing", "quality_assurance", "automation"], "max_tasks": 2}
        ]
        
        for agent in agents:
            self.task_scheduler.register_agent(agent["id"], agent)
            print(f"✓ 注册代理: {agent['id']}")
        
        # 创建告警规则
        alert_rules = [
            AlertRule(
                name="CPU使用率过高",
                description="CPU使用率超过80%",
                metric_name="cpu_usage_percent",
                condition="value > 80",
                severity=AlertSeverity.WARNING
            ),
            AlertRule(
                name="内存使用率过高",
                description="内存使用率超过90%",
                metric_name="memory_usage_percent",
                condition="value > 90",
                severity=AlertSeverity.ERROR
            ),
            AlertRule(
                name="任务执行失败率过高",
                description="任务执行失败率超过10%",
                metric_name="task_failure_rate",
                condition="value > 10",
                severity=AlertSeverity.CRITICAL
            )
        ]
        
        for rule in alert_rules:
            self.monitoring_system.create_alert_rule(rule)
            print(f"✓ 创建告警规则: {rule.name}")
        
        # 创建监控仪表板
        dashboard = Dashboard(
            name="智能体系统监控仪表板",
            description="监控智能体系统的整体运行状态",
            widgets=[
                {
                    "type": "line_chart",
                    "title": "CPU使用率",
                    "metric": "cpu_usage_percent",
                    "position": {"x": 0, "y": 0, "w": 6, "h": 4}
                },
                {
                    "type": "line_chart",
                    "title": "内存使用率",
                    "metric": "memory_usage_percent",
                    "position": {"x": 6, "y": 0, "w": 6, "h": 4}
                },
                {
                    "type": "gauge",
                    "title": "任务完成率",
                    "metric": "task_completion_rate",
                    "position": {"x": 0, "y": 4, "w": 4, "h": 4}
                },
                {
                    "type": "table",
                    "title": "活跃告警",
                    "data_source": "active_alerts",
                    "position": {"x": 4, "y": 4, "w": 8, "h": 4}
                }
            ],
            refresh_interval=30
        )
        
        dashboard_id = self.monitoring_system.create_dashboard(dashboard)
        print(f"✓ 创建监控仪表板: {dashboard.name}")
        
        print("🎯 演示环境设置完成！\n")
    
    async def demo_scenario_1_web_development(self):
        """演示场景1：Web应用开发"""
        print("📱 演示场景1：Web应用开发")
        print("=" * 50)
        
        # 1. 生成开发计划
        print("\n1. 生成Web应用开发计划...")
        goal = "开发一个用户管理系统Web应用"
        current_state = {
            "user_requirements": "用户注册、登录、个人资料管理",
            "tech_stack": "Python Flask + React",
            "timeline": "2周"
        }
        available_tools = ["code_generator", "database_manager", "api_tester", "frontend_builder"]
        agent_capabilities = ["web_development", "database_design", "api_development", "frontend_development"]
        
        plan = await self.planning_engine.generate_plan(goal, current_state, available_tools, agent_capabilities)
        if plan:
            print(f"✓ 生成计划: {plan.plan_id}")
            print(f"  目标: {plan.goal}")
            print(f"  策略: {plan.strategy.value}")
            print(f"  步骤数: {len(plan.steps)}")
            
            for i, step in enumerate(plan.steps):
                print(f"    步骤{i+1}: {step['description']}")
        else:
            print("❌ 计划生成失败")
            return
        
        # 2. 创建任务
        print("\n2. 创建开发任务...")
        tasks = [
            ScheduledTask(
                name="后端API开发",
                description="开发用户管理API",
                priority=3,
                deadline=datetime.now() + timedelta(days=3),
                estimated_duration=timedelta(hours=8),
                resource_requirements=[
                    ResourceRequirement(ResourceType.CPU, 2.0, "cores", timedelta(hours=8)),
                    ResourceRequirement(ResourceType.MEMORY, 4.0, "GB", timedelta(hours=8))
                ],
                dependencies=[]
            ),
            ScheduledTask(
                name="数据库设计",
                description="设计用户数据表结构",
                priority=2,
                deadline=datetime.now() + timedelta(days=1),
                estimated_duration=timedelta(hours=4),
                resource_requirements=[
                    ResourceRequirement(ResourceType.CPU, 1.0, "cores", timedelta(hours=4)),
                    ResourceRequirement(ResourceType.MEMORY, 2.0, "GB", timedelta(hours=4))
                ],
                dependencies=[]
            ),
            ScheduledTask(
                name="前端界面开发",
                description="开发用户界面组件",
                priority=2,
                deadline=datetime.now() + timedelta(days=5),
                estimated_duration=timedelta(hours=12),
                resource_requirements=[
                    ResourceRequirement(ResourceType.CPU, 1.5, "cores", timedelta(hours=12)),
                    ResourceRequirement(ResourceType.MEMORY, 3.0, "GB", timedelta(hours=12))
                ],
                dependencies=["数据库设计"]
            ),
            ScheduledTask(
                name="系统集成测试",
                description="测试整个系统功能",
                priority=1,
                deadline=datetime.now() + timedelta(days=7),
                estimated_duration=timedelta(hours=6),
                resource_requirements=[
                    ResourceRequirement(ResourceType.CPU, 2.0, "cores", timedelta(hours=6)),
                    ResourceRequirement(ResourceType.MEMORY, 4.0, "GB", timedelta(hours=6))
                ],
                dependencies=["后端API开发", "前端界面开发"]
            )
        ]
        
        # 提交任务
        for task in tasks:
            await self.task_scheduler.submit_task(task)
            print(f"✓ 提交任务: {task.name} (优先级: {task.priority})")
        
        # 3. 等待任务调度
        print("\n3. 等待任务调度...")
        await asyncio.sleep(3)
        
        # 4. 检查任务状态
        print("\n4. 检查任务状态:")
        for task in tasks:
            current_task = self.task_scheduler.get_task(task.id)
            if current_task:
                print(f"  {current_task.name}: {current_task.status}")
                if current_task.assigned_agent:
                    print(f"    分配代理: {current_task.assigned_agent}")
                if current_task.scheduled_time:
                    print(f"    调度时间: {current_task.scheduled_time}")
        
        # 5. 执行计划
        print("\n5. 执行开发计划...")
        execution_result = await self.execution_engine.execute_plan(plan)
        print(f"✓ 计划执行结果: {execution_result['status']}")
        
        # 6. 模拟任务执行
        print("\n6. 模拟任务执行...")
        for task in tasks:
            await self.task_scheduler.update_task_status(task.id, "running", 0.3)
            await asyncio.sleep(1)
            await self.task_scheduler.update_task_status(task.id, "completed", 1.0)
            print(f"✓ 任务完成: {task.name}")
        
        print("🎉 Web应用开发场景演示完成！\n")
    
    async def demo_scenario_2_data_analysis(self):
        """演示场景2：数据分析项目"""
        print("📊 演示场景2：数据分析项目")
        print("=" * 50)
        
        # 1. 生成分析计划
        print("\n1. 生成数据分析计划...")
        goal = "分析销售数据并生成可视化报告"
        current_state = {
            "data_source": "销售数据库",
            "analysis_type": "趋势分析",
            "output_format": "PDF报告"
        }
        available_tools = ["data_extractor", "statistical_analyzer", "chart_generator", "report_builder"]
        agent_capabilities = ["data_analysis", "statistics", "visualization", "report_generation"]
        
        plan = await self.planning_engine.generate_plan(goal, current_state, available_tools, agent_capabilities, PlanningStrategy.HIERARCHICAL)
        if plan:
            print(f"✓ 生成分析计划: {plan.plan_id}")
            print(f"  策略: {plan.strategy.value}")
            print(f"  步骤数: {len(plan.steps)}")
        else:
            print("❌ 分析计划生成失败")
            return
        
        # 2. 创建分析任务
        print("\n2. 创建数据分析任务...")
        analysis_tasks = [
            ScheduledTask(
                name="数据提取",
                description="从数据库提取销售数据",
                priority=3,
                deadline=datetime.now() + timedelta(hours=2),
                estimated_duration=timedelta(hours=1),
                resource_requirements=[
                    ResourceRequirement(ResourceType.CPU, 1.0, "cores", timedelta(hours=1)),
                    ResourceRequirement(ResourceType.MEMORY, 2.0, "GB", timedelta(hours=1)),
                    ResourceRequirement(ResourceType.STORAGE, 5.0, "GB", timedelta(hours=1))
                ]
            ),
            ScheduledTask(
                name="数据清洗",
                description="清洗和预处理数据",
                priority=2,
                deadline=datetime.now() + timedelta(hours=3),
                estimated_duration=timedelta(hours=2),
                resource_requirements=[
                    ResourceRequirement(ResourceType.CPU, 2.0, "cores", timedelta(hours=2)),
                    ResourceRequirement(ResourceType.MEMORY, 4.0, "GB", timedelta(hours=2))
                ],
                dependencies=["数据提取"]
            ),
            ScheduledTask(
                name="统计分析",
                description="执行统计分析和趋势计算",
                priority=2,
                deadline=datetime.now() + timedelta(hours=5),
                estimated_duration=timedelta(hours=3),
                resource_requirements=[
                    ResourceRequirement(ResourceType.CPU, 3.0, "cores", timedelta(hours=3)),
                    ResourceRequirement(ResourceType.MEMORY, 6.0, "GB", timedelta(hours=3))
                ],
                dependencies=["数据清洗"]
            ),
            ScheduledTask(
                name="可视化生成",
                description="生成图表和可视化",
                priority=1,
                deadline=datetime.now() + timedelta(hours=6),
                estimated_duration=timedelta(hours=1),
                resource_requirements=[
                    ResourceRequirement(ResourceType.CPU, 1.0, "cores", timedelta(hours=1)),
                    ResourceRequirement(ResourceType.MEMORY, 2.0, "GB", timedelta(hours=1))
                ],
                dependencies=["统计分析"]
            )
        ]
        
        # 提交任务
        for task in analysis_tasks:
            await self.task_scheduler.submit_task(task)
            print(f"✓ 提交分析任务: {task.name}")
        
        # 3. 等待调度和执行
        print("\n3. 等待任务调度和执行...")
        await asyncio.sleep(3)
        
        # 4. 执行分析计划
        print("\n4. 执行数据分析计划...")
        execution_result = await self.execution_engine.execute_plan(plan)
        print(f"✓ 分析计划执行结果: {execution_result['status']}")
        
        # 5. 模拟任务执行
        print("\n5. 模拟分析任务执行...")
        for task in analysis_tasks:
            await self.task_scheduler.update_task_status(task.id, "running", 0.5)
            await asyncio.sleep(1)
            await self.task_scheduler.update_task_status(task.id, "completed", 1.0)
            print(f"✓ 分析任务完成: {task.name}")
        
        print("🎉 数据分析场景演示完成！\n")
    
    async def demo_scenario_3_emergency_response(self):
        """演示场景3：紧急响应"""
        print("🚨 演示场景3：紧急响应")
        print("=" * 50)
        
        # 1. 生成紧急响应计划
        print("\n1. 生成紧急响应计划...")
        goal = "处理系统故障并恢复服务"
        current_state = {
            "incident_type": "系统故障",
            "severity": "高",
            "affected_users": "1000+",
            "estimated_downtime": "30分钟"
        }
        available_tools = ["system_monitor", "log_analyzer", "backup_restorer", "deployment_tool"]
        agent_capabilities = ["incident_response", "system_recovery", "monitoring", "deployment"]
        
        plan = await self.planning_engine.generate_plan(goal, current_state, available_tools, agent_capabilities, PlanningStrategy.REACTIVE)
        if plan:
            print(f"✓ 生成紧急响应计划: {plan.plan_id}")
            print(f"  策略: {plan.strategy.value}")
            print(f"  步骤数: {len(plan.steps)}")
        else:
            print("❌ 紧急响应计划生成失败")
            return
        
        # 2. 创建紧急任务
        print("\n2. 创建紧急响应任务...")
        emergency_tasks = [
            ScheduledTask(
                name="故障诊断",
                description="诊断系统故障原因",
                priority=5,  # 最高优先级
                deadline=datetime.now() + timedelta(minutes=10),
                estimated_duration=timedelta(minutes=5),
                resource_requirements=[
                    ResourceRequirement(ResourceType.CPU, 4.0, "cores", timedelta(minutes=5)),
                    ResourceRequirement(ResourceType.MEMORY, 8.0, "GB", timedelta(minutes=5))
                ]
            ),
            ScheduledTask(
                name="服务恢复",
                description="恢复受影响的服务",
                priority=5,
                deadline=datetime.now() + timedelta(minutes=20),
                estimated_duration=timedelta(minutes=10),
                resource_requirements=[
                    ResourceRequirement(ResourceType.CPU, 2.0, "cores", timedelta(minutes=10)),
                    ResourceRequirement(ResourceType.MEMORY, 4.0, "GB", timedelta(minutes=10))
                ],
                dependencies=["故障诊断"]
            ),
            ScheduledTask(
                name="系统验证",
                description="验证系统功能正常",
                priority=4,
                deadline=datetime.now() + timedelta(minutes=30),
                estimated_duration=timedelta(minutes=5),
                resource_requirements=[
                    ResourceRequirement(ResourceType.CPU, 1.0, "cores", timedelta(minutes=5)),
                    ResourceRequirement(ResourceType.MEMORY, 2.0, "GB", timedelta(minutes=5))
                ],
                dependencies=["服务恢复"]
            )
        ]
        
        # 提交紧急任务
        for task in emergency_tasks:
            await self.task_scheduler.submit_task(task)
            print(f"✓ 提交紧急任务: {task.name} (优先级: {task.priority})")
        
        # 3. 立即执行紧急计划
        print("\n3. 立即执行紧急响应计划...")
        execution_result = await self.execution_engine.execute_plan(plan)
        print(f"✓ 紧急响应计划执行结果: {execution_result['status']}")
        
        # 4. 快速执行紧急任务
        print("\n4. 快速执行紧急任务...")
        for task in emergency_tasks:
            await self.task_scheduler.update_task_status(task.id, "running", 0.8)
            await asyncio.sleep(0.5)  # 快速执行
            await self.task_scheduler.update_task_status(task.id, "completed", 1.0)
            print(f"✓ 紧急任务完成: {task.name}")
        
        print("🎉 紧急响应场景演示完成！\n")
    
    async def demo_monitoring_and_alerts(self):
        """演示监控和告警"""
        print("📈 演示监控和告警系统")
        print("=" * 50)
        
        # 1. 等待指标收集
        print("\n1. 等待系统指标收集...")
        await asyncio.sleep(5)
        
        # 2. 检查收集的指标
        print("\n2. 检查系统指标:")
        metric_names = ["cpu_usage_percent", "memory_usage_percent", "disk_usage_percent"]
        for name in metric_names:
            latest_metric = self.monitoring_system.get_latest_metric(name)
            if latest_metric:
                print(f"  {name}: {latest_metric.value:.2f}")
            else:
                print(f"  {name}: 无数据")
        
        # 3. 检查活跃告警
        print("\n3. 检查活跃告警:")
        active_alerts = self.monitoring_system.get_active_alerts()
        if active_alerts:
            for alert in active_alerts:
                print(f"  🚨 {alert.name}: {alert.severity.value}")
                print(f"     描述: {alert.description}")
                print(f"     触发时间: {alert.triggered_at}")
        else:
            print("  ✅ 无活跃告警")
        
        # 4. 模拟告警确认
        if active_alerts:
            print("\n4. 确认告警...")
            for alert in active_alerts:
                self.monitoring_system.acknowledge_alert(alert.id, "admin")
                print(f"✓ 确认告警: {alert.name}")
        
        # 5. 获取系统统计
        print("\n5. 系统统计信息:")
        stats = self.monitoring_system.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("🎉 监控和告警演示完成！\n")
    
    async def demo_resource_management(self):
        """演示资源管理"""
        print("💾 演示资源管理")
        print("=" * 50)
        
        # 1. 创建资源请求
        print("\n1. 创建资源请求...")
        resource_requests = [
            ResourceRequest(
                requester="web_developer",
                resource_type=ResourceType.CPU,
                amount=3.0,
                unit="cores",
                duration=timedelta(hours=4),
                priority=3
            ),
            ResourceRequest(
                requester="data_analyst",
                resource_type=ResourceType.MEMORY,
                amount=8.0,
                unit="GB",
                duration=timedelta(hours=6),
                priority=2
            ),
            ResourceRequest(
                requester="devops_engineer",
                resource_type=ResourceType.STORAGE,
                amount=20.0,
                unit="GB",
                duration=timedelta(hours=2),
                priority=1
            )
        ]
        
        # 提交资源请求
        for request in resource_requests:
            request_id = await self.resource_manager.request_resources(request)
            print(f"✓ 提交资源请求: {request.requester} -> {request.resource_type.value} ({request.amount} {request.unit})")
        
        # 2. 等待资源分配
        print("\n2. 等待资源分配...")
        await asyncio.sleep(3)
        
        # 3. 检查资源分配
        print("\n3. 检查资源分配:")
        for request in resource_requests:
            current_request = self.resource_manager.get_request(request.id)
            if current_request:
                print(f"  {current_request.requester}: {current_request.status}")
        
        # 4. 检查资源利用率
        print("\n4. 资源利用率:")
        utilization = self.resource_manager.get_resource_utilization()
        for resource_type, rate in utilization.items():
            print(f"  {resource_type}: {rate:.2%}")
        
        # 5. 获取资源管理器统计
        print("\n5. 资源管理器统计:")
        stats = self.resource_manager.get_stats()
        for key, value in stats.items():
            if key != "current_metrics":
                print(f"  {key}: {value}")
        
        print("🎉 资源管理演示完成！\n")
    
    async def demo_system_integration(self):
        """演示系统集成"""
        print("🔗 演示系统集成")
        print("=" * 50)
        
        # 1. 获取各系统统计
        print("\n1. 各系统统计信息:")
        
        # 任务调度器统计
        scheduler_stats = self.task_scheduler.get_stats()
        print("  任务调度器:")
        for key, value in scheduler_stats.items():
            if key not in ["resource_stats"]:
                print(f"    {key}: {value}")
        
        # 资源管理器统计
        resource_stats = self.resource_manager.get_stats()
        print("  资源管理器:")
        for key, value in resource_stats.items():
            if key not in ["current_metrics"]:
                print(f"    {key}: {value}")
        
        # 监控系统统计
        monitoring_stats = self.monitoring_system.get_stats()
        print("  监控系统:")
        for key, value in monitoring_stats.items():
            print(f"    {key}: {value}")
        
        # 2. 系统健康检查
        print("\n2. 系统健康检查:")
        
        # 检查任务调度器
        if scheduler_stats["total_tasks"] > 0:
            completion_rate = scheduler_stats["completed_tasks"] / scheduler_stats["total_tasks"]
            print(f"  任务完成率: {completion_rate:.2%}")
        
        # 检查资源利用率
        utilization = self.resource_manager.get_resource_utilization()
        avg_utilization = sum(utilization.values()) / len(utilization)
        print(f"  平均资源利用率: {avg_utilization:.2%}")
        
        # 检查告警状态
        active_alerts = self.monitoring_system.get_active_alerts()
        print(f"  活跃告警数量: {len(active_alerts)}")
        
        # 3. 性能指标
        print("\n3. 性能指标:")
        print(f"  系统运行时间: {time.time() - start_time:.2f}秒")
        print(f"  处理的任务数: {scheduler_stats['total_tasks']}")
        print(f"  资源池数量: {resource_stats['total_resource_pools']}")
        print(f"  监控指标数: {len(self.monitoring_system.metrics_collector.collectors)}")
        
        print("🎉 系统集成演示完成！\n")
    
    async def cleanup(self):
        """清理资源"""
        print("🧹 清理系统资源...")
        
        if self.task_scheduler:
            await self.task_scheduler.stop()
            print("✓ 任务调度器已停止")
        
        if self.resource_manager:
            await self.resource_manager.stop()
            print("✓ 资源管理器已停止")
        
        if self.monitoring_system:
            await self.monitoring_system.stop()
            print("✓ 监控系统已停止")
        
        # 清理临时文件
        if os.path.exists("demo_monitoring.db"):
            os.remove("demo_monitoring.db")
            print("✓ 临时数据库文件已删除")
        
        print("🎯 资源清理完成！")

async def main():
    """主函数"""
    global start_time
    start_time = time.time()
    
    print("🌟 第5章 规划与执行引擎开发 - 综合演示")
    print("=" * 60)
    print("本演示将展示智能体的规划引擎、执行引擎、任务调度器、")
    print("资源管理器和监控系统的集成使用。")
    print("=" * 60)
    
    demo = PlanningExecutionDemo()
    
    try:
        # 初始化系统
        await demo.initialize_systems()
        
        # 设置演示环境
        await demo.setup_demo_environment()
        
        # 演示场景1：Web应用开发
        await demo.demo_scenario_1_web_development()
        
        # 演示场景2：数据分析项目
        await demo.demo_scenario_2_data_analysis()
        
        # 演示场景3：紧急响应
        await demo.demo_scenario_3_emergency_response()
        
        # 演示监控和告警
        await demo.demo_monitoring_and_alerts()
        
        # 演示资源管理
        await demo.demo_resource_management()
        
        # 演示系统集成
        await demo.demo_system_integration()
        
        print("🎊 所有演示场景完成！")
        print("=" * 60)
        print("感谢观看第5章规划与执行引擎开发的综合演示！")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        print(f"❌ 演示失败: {e}")
    
    finally:
        # 清理资源
        await demo.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
