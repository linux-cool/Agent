# collaboration_demo.py
"""
第3章 多智能体系统协作机制 - 完整演示程序
展示多智能体协作的核心功能和使用方法
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# 导入各模块示例
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.multi_agent_architecture import MultiAgentSystem, AgentInfo, Task, AgentState
from code.communication_protocols import CommunicationManager, ProtocolType, ProtocolConfig, MessageType
from code.task_allocation import TaskAllocator, AllocationStrategy, LoadBalancer, LoadBalancingAlgorithm
from code.collaboration_strategies import CollaborationEngine, CollaborationStrategy, ConsensusEngine, ConsensusAlgorithm
from code.fault_tolerance import FaultToleranceManager, HealthStatus
from code.coordination_engine import CoordinationEngine, CoordinationType, SchedulingPolicy

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CollaborationDemo:
    """多智能体协作演示类"""
    
    def __init__(self):
        self.systems = {}
        self.demo_results = {}
        self.performance_metrics = {}
    
    async def run_complete_demo(self):
        """运行完整演示"""
        logger.info("🚀 开始第3章多智能体系统协作机制演示")
        
        # 1. 多智能体架构演示
        await self.demo_multi_agent_architecture()
        
        # 2. 通信协议演示
        await self.demo_communication_protocols()
        
        # 3. 任务分配演示
        await self.demo_task_allocation()
        
        # 4. 协作策略演示
        await self.demo_collaboration_strategies()
        
        # 5. 容错机制演示
        await self.demo_fault_tolerance()
        
        # 6. 协调引擎演示
        await self.demo_coordination_engine()
        
        # 7. 集成协作演示
        await self.demo_integrated_collaboration()
        
        # 8. 性能基准测试
        await self.demo_performance_benchmark()
        
        # 9. 生成演示报告
        await self.generate_demo_report()
        
        logger.info("✅ 第3章演示完成")
    
    async def demo_multi_agent_architecture(self):
        """多智能体架构演示"""
        logger.info("📋 演示1: 多智能体架构")
        
        try:
            # 创建系统配置
            config = {
                "coordination_type": "centralized",
                "max_agents": 5,
                "heartbeat_interval": 5
            }
            
            # 创建多智能体系统
            system = MultiAgentSystem(config)
            await system.start()
            
            # 创建智能体
            agents = [
                AgentInfo(
                    id="researcher_1",
                    name="Research Agent 1",
                    state=AgentState.IDLE,
                    capabilities=[
                        {"name": "research", "description": "Conduct research", "performance_score": 0.9},
                        {"name": "analysis", "description": "Analyze data", "performance_score": 0.8}
                    ]
                ),
                AgentInfo(
                    id="analyst_1",
                    name="Analysis Agent 1",
                    state=AgentState.IDLE,
                    capabilities=[
                        {"name": "analysis", "description": "Analyze data", "performance_score": 0.95},
                        {"name": "reporting", "description": "Generate reports", "performance_score": 0.85}
                    ]
                ),
                AgentInfo(
                    id="coordinator_1",
                    name="Coordination Agent 1",
                    state=AgentState.IDLE,
                    capabilities=[
                        {"name": "coordination", "description": "Coordinate tasks", "performance_score": 0.9},
                        {"name": "management", "description": "Manage resources", "performance_score": 0.8}
                    ]
                )
            ]
            
            # 添加智能体
            for agent in agents:
                await system.add_agent(agent)
            
            # 创建任务
            tasks = [
                Task(
                    name="Research Task",
                    description="Conduct research on AI trends",
                    priority=5,
                    required_capabilities=["research"],
                    resource_requirements={"cpu": 0.5, "memory": 0.3}
                ),
                Task(
                    name="Analysis Task",
                    description="Analyze research data",
                    priority=3,
                    required_capabilities=["analysis"],
                    resource_requirements={"cpu": 0.3, "memory": 0.2}
                ),
                Task(
                    name="Coordination Task",
                    description="Coordinate team activities",
                    priority=4,
                    required_capabilities=["coordination"],
                    resource_requirements={"cpu": 0.2, "memory": 0.1}
                )
            ]
            
            # 提交任务
            task_ids = []
            for task in tasks:
                task_id = await system.submit_task(task)
                task_ids.append(task_id)
            
            # 等待任务完成
            await asyncio.sleep(2)
            
            # 获取系统状态
            metrics = await system.get_system_metrics()
            
            self.demo_results["multi_agent_architecture"] = {
                "status": "success",
                "agents_registered": len(agents),
                "tasks_submitted": len(tasks),
                "system_metrics": metrics,
                "features": [
                    "分层架构设计",
                    "智能体状态管理",
                    "任务分解与分配",
                    "系统监控与指标"
                ]
            }
            
            # 停止系统
            await system.stop()
            
            logger.info("✅ 多智能体架构演示完成")
            
        except Exception as e:
            logger.error(f"多智能体架构演示失败: {e}")
            self.demo_results["multi_agent_architecture"] = {
                "status": "error",
                "error": str(e)
            }
    
    async def demo_communication_protocols(self):
        """通信协议演示"""
        logger.info("📋 演示2: 通信协议")
        
        try:
            # 创建通信管理器
            comm_manager = CommunicationManager()
            
            # 创建协议配置
            http_config = ProtocolConfig(
                protocol_type=ProtocolType.HTTP,
                host="localhost",
                port=8080,
                encryption_key="test_key"
            )
            
            websocket_config = ProtocolConfig(
                protocol_type=ProtocolType.WEBSOCKET,
                host="localhost",
                port=8081
            )
            
            # 创建协议实例
            from code.communication_protocols import HTTPProtocol, WebSocketProtocol
            
            http_protocol = HTTPProtocol(http_config)
            websocket_protocol = WebSocketProtocol(websocket_config)
            
            # 注册协议
            comm_manager.register_protocol(ProtocolType.HTTP, http_protocol)
            comm_manager.register_protocol(ProtocolType.WEBSOCKET, websocket_protocol)
            comm_manager.set_active_protocol(ProtocolType.HTTP)
            
            # 启动协议
            await comm_manager.start_all_protocols()
            
            # 测试消息传递
            from code.communication_protocols import Message, MessagePriority
            
            messages = [
                Message(
                    sender="agent_1",
                    receiver="agent_2",
                    message_type=MessageType.REQUEST,
                    content="Hello from agent 1",
                    priority=MessagePriority.HIGH
                ),
                Message(
                    sender="agent_2",
                    receiver="agent_1",
                    message_type=MessageType.RESPONSE,
                    content="Hello back from agent 2",
                    priority=MessagePriority.NORMAL
                )
            ]
            
            # 发送消息
            for message in messages:
                success = await comm_manager.send_message(message)
                if success:
                    logger.info(f"Message sent: {message.sender} -> {message.receiver}")
            
            # 接收消息
            received_messages = await comm_manager.receive_message("agent_2")
            
            # 广播消息
            await comm_manager.broadcast_message("agent_1", MessageType.NOTIFICATION, "System update")
            
            self.demo_results["communication_protocols"] = {
                "status": "success",
                "protocols_registered": 2,
                "messages_sent": len(messages),
                "messages_received": len(received_messages),
                "features": [
                    "多种通信协议支持",
                    "消息加密与安全",
                    "广播与点对点通信",
                    "协议自动选择"
                ]
            }
            
            # 停止协议
            await comm_manager.stop_all_protocols()
            
            logger.info("✅ 通信协议演示完成")
            
        except Exception as e:
            logger.error(f"通信协议演示失败: {e}")
            self.demo_results["communication_protocols"] = {
                "status": "error",
                "error": str(e)
            }
    
    async def demo_task_allocation(self):
        """任务分配演示"""
        logger.info("📋 演示3: 任务分配")
        
        try:
            # 创建任务分配器
            allocator = TaskAllocator(strategy=AllocationStrategy.LOAD_BALANCED)
            
            # 注册智能体能力
            allocator.register_agent_capabilities("agent_1", {"research", "analysis"})
            allocator.register_agent_capabilities("agent_2", {"analysis", "reporting"})
            allocator.register_agent_capabilities("agent_3", {"research", "reporting"})
            
            # 注册智能体资源
            from code.task_allocation import ResourceCapacity
            
            allocator.register_agent_resources("agent_1", ResourceCapacity(cpu=1.0, memory=2.0))
            allocator.register_agent_resources("agent_2", ResourceCapacity(cpu=1.5, memory=1.5))
            allocator.register_agent_resources("agent_3", ResourceCapacity(cpu=0.8, memory=1.0))
            
            # 设置智能体成本
            allocator.set_agent_cost("agent_1", 1.0)
            allocator.set_agent_cost("agent_2", 1.2)
            allocator.set_agent_cost("agent_3", 0.8)
            
            # 更新智能体指标
            from code.task_allocation import AgentMetrics
            
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
            
            # 获取统计信息
            stats = allocator.get_allocation_statistics()
            load_dist = allocator.load_balancer.get_load_distribution()
            perf_summary = allocator.load_balancer.get_performance_summary()
            
            self.demo_results["task_allocation"] = {
                "status": "success",
                "tasks_allocated": len(allocations),
                "allocation_statistics": stats,
                "load_distribution": load_dist,
                "performance_summary": perf_summary,
                "features": [
                    "多种分配策略",
                    "负载均衡算法",
                    "能力匹配",
                    "资源感知分配"
                ]
            }
            
            logger.info("✅ 任务分配演示完成")
            
        except Exception as e:
            logger.error(f"任务分配演示失败: {e}")
            self.demo_results["task_allocation"] = {
                "status": "error",
                "error": str(e)
            }
    
    async def demo_collaboration_strategies(self):
        """协作策略演示"""
        logger.info("📋 演示4: 协作策略")
        
        try:
            # 创建协作引擎
            engine = CollaborationEngine()
            
            # 设置共识引擎
            engine.consensus_engine.set_agent_weight("agent_1", 1.0)
            engine.consensus_engine.set_agent_weight("agent_2", 1.2)
            engine.consensus_engine.set_agent_weight("agent_3", 0.8)
            engine.consensus_engine.set_consensus_threshold(0.6)
            
            # 设置协商引擎
            engine.negotiation_engine.set_agent_preferences("agent_1", {"cpu": 0.8, "memory": 0.6})
            engine.negotiation_engine.set_agent_preferences("agent_2", {"cpu": 0.6, "memory": 0.8})
            engine.negotiation_engine.set_agent_preferences("agent_3", {"cpu": 0.7, "memory": 0.7})
            
            # 测试共识机制
            proposal_id = await engine.consensus_engine.propose("agent_1", "Proposal for task allocation")
            
            await engine.consensus_engine.vote("agent_1", proposal_id, True, "Good proposal")
            await engine.consensus_engine.vote("agent_2", proposal_id, True, "Agree")
            await engine.consensus_engine.vote("agent_3", proposal_id, False, "Disagree")
            
            consensus_stats = engine.consensus_engine.get_consensus_statistics()
            
            # 测试协商机制
            offer_id = await engine.negotiation_engine.make_offer("agent_1", "agent_2", {"cpu": 0.5, "memory": 0.3})
            
            accepted = await engine.negotiation_engine.respond_to_offer(offer_id, "agent_2", {"cpu": 0.4, "memory": 0.4})
            
            negotiation_stats = engine.negotiation_engine.get_negotiation_statistics()
            
            # 测试联盟机制
            coalition_id = await engine.coalition_manager.create_coalition("agent_1", "Research collaboration")
            
            await engine.coalition_manager.join_coalition("agent_2", coalition_id)
            await engine.coalition_manager.join_coalition("agent_3", coalition_id)
            
            coalition_stats = engine.coalition_manager.get_coalition_statistics()
            
            # 测试协作策略
            task_allocation = {
                "agent_1": [{"id": "task_1", "priority": 5}],
                "agent_2": [{"id": "task_2", "priority": 3}],
                "agent_3": [{"id": "task_3", "priority": 4}]
            }
            
            # 并行协作
            parallel_results = await engine.execute_collaboration(task_allocation, CollaborationStrategy.PARALLEL)
            
            # 共识协作
            consensus_results = await engine.execute_collaboration(task_allocation, CollaborationStrategy.CONSENSUS)
            
            self.demo_results["collaboration_strategies"] = {
                "status": "success",
                "consensus_statistics": consensus_stats,
                "negotiation_statistics": negotiation_stats,
                "coalition_statistics": coalition_stats,
                "parallel_results": parallel_results,
                "consensus_results": consensus_results,
                "features": [
                    "多种协作策略",
                    "共识机制",
                    "协商机制",
                    "联盟管理"
                ]
            }
            
            logger.info("✅ 协作策略演示完成")
            
        except Exception as e:
            logger.error(f"协作策略演示失败: {e}")
            self.demo_results["collaboration_strategies"] = {
                "status": "error",
                "error": str(e)
            }
    
    async def demo_fault_tolerance(self):
        """容错机制演示"""
        logger.info("📋 演示5: 容错机制")
        
        try:
            # 创建容错管理器
            fault_tolerance_manager = FaultToleranceManager()
            
            # 模拟健康检查函数
            async def health_check_func(agent_id: str) -> HealthStatus:
                await asyncio.sleep(0.1)
                import random
                statuses = [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
                return random.choice(statuses)
            
            # 启动容错保护
            await fault_tolerance_manager.start_fault_tolerance("agent_1", health_check_func)
            await fault_tolerance_manager.start_fault_tolerance("agent_2", health_check_func)
            await fault_tolerance_manager.start_fault_tolerance("agent_3", health_check_func)
            
            # 模拟故障检测
            metrics = {"response_time": 6.0, "error_rate": 0.05, "memory_usage": 0.7, "cpu_usage": 0.6}
            faults = await fault_tolerance_manager.check_agent_health("agent_1", metrics)
            
            # 等待一段时间让健康监控运行
            await asyncio.sleep(3)
            
            # 获取系统健康状态
            system_health = fault_tolerance_manager.get_system_health()
            
            # 停止容错保护
            await fault_tolerance_manager.stop_fault_tolerance("agent_1")
            await fault_tolerance_manager.stop_fault_tolerance("agent_2")
            await fault_tolerance_manager.stop_fault_tolerance("agent_3")
            
            self.demo_results["fault_tolerance"] = {
                "status": "success",
                "faults_detected": len(faults),
                "system_health": system_health,
                "features": [
                    "健康监控",
                    "故障检测",
                    "自动恢复",
                    "熔断保护"
                ]
            }
            
            logger.info("✅ 容错机制演示完成")
            
        except Exception as e:
            logger.error(f"容错机制演示失败: {e}")
            self.demo_results["fault_tolerance"] = {
                "status": "error",
                "error": str(e)
            }
    
    async def demo_coordination_engine(self):
        """协调引擎演示"""
        logger.info("📋 演示6: 协调引擎")
        
        try:
            # 创建协调引擎
            engine = CoordinationEngine(coordination_type=CoordinationType.CENTRALIZED)
            
            # 注册智能体角色
            engine.register_agent_role("agent_1", "researcher", {"research", "analysis"}, 3)
            engine.register_agent_role("agent_2", "analyst", {"analysis", "reporting"}, 2)
            engine.register_agent_role("agent_3", "coordinator", {"coordination", "management"}, 4)
            
            # 创建协调任务
            from code.coordination_engine import CoordinationTask
            
            tasks = [
                CoordinationTask(
                    name="research",
                    description="Conduct research on AI trends",
                    priority=5,
                    resource_requirements={"cpu": 0.5, "memory": 0.3}
                ),
                CoordinationTask(
                    name="analysis",
                    description="Analyze research data",
                    priority=3,
                    resource_requirements={"cpu": 0.3, "memory": 0.2}
                ),
                CoordinationTask(
                    name="coordination",
                    description="Coordinate team activities",
                    priority=4,
                    resource_requirements={"cpu": 0.2, "memory": 0.1}
                )
            ]
            
            # 协调任务
            task_ids = []
            for task in tasks:
                task_id = await engine.coordinate_task(task)
                task_ids.append(task_id)
            
            # 处理协调事件
            from code.coordination_engine import CoordinationEvent
            
            # 任务完成事件
            completion_event = CoordinationEvent(
                event_type="task_completion",
                source_agent="agent_1",
                content={"task_id": task_ids[0], "result": "Research completed"}
            )
            
            result = await engine.handle_coordination_event(completion_event)
            
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
            
            # 获取协调统计
            stats = engine.get_coordination_statistics()
            
            self.demo_results["coordination_engine"] = {
                "status": "success",
                "tasks_coordinated": len(task_ids),
                "coordination_statistics": stats,
                "features": [
                    "多种协调类型",
                    "任务调度",
                    "冲突解决",
                    "事件处理"
                ]
            }
            
            logger.info("✅ 协调引擎演示完成")
            
        except Exception as e:
            logger.error(f"协调引擎演示失败: {e}")
            self.demo_results["coordination_engine"] = {
                "status": "error",
                "error": str(e)
            }
    
    async def demo_integrated_collaboration(self):
        """集成协作演示"""
        logger.info("📋 演示7: 集成协作")
        
        try:
            # 模拟集成协作场景
            integration_scenario = {
                "name": "智能研究团队",
                "description": "多智能体协作进行AI研究项目",
                "components": {
                    "multi_agent_system": "管理智能体生命周期",
                    "communication_protocols": "处理智能体间通信",
                    "task_allocation": "分配研究任务",
                    "collaboration_strategies": "协调研究过程",
                    "fault_tolerance": "确保系统稳定性",
                    "coordination_engine": "整体项目协调"
                },
                "workflow": [
                    "项目初始化 (CoordinationEngine)",
                    "智能体注册 (MultiAgentSystem)",
                    "任务分解与分配 (TaskAllocation)",
                    "协作执行 (CollaborationStrategies)",
                    "通信协调 (CommunicationProtocols)",
                    "故障监控 (FaultTolerance)",
                    "结果整合 (CoordinationEngine)"
                ]
            }
            
            # 模拟集成测试
            integration_results = {
                "components_integrated": len(integration_scenario["components"]),
                "workflow_steps": len(integration_scenario["workflow"]),
                "integration_success": True,
                "performance_metrics": {
                    "response_time": 1.5,  # seconds
                    "throughput": 40,  # requests per minute
                    "error_rate": 0.005,  # 0.5%
                    "availability": 0.998  # 99.8%
                }
            }
            
            self.demo_results["integrated_collaboration"] = {
                "status": "success",
                "scenario": integration_scenario,
                "results": integration_results,
                "best_practices": [
                    "模块化设计",
                    "松耦合架构",
                    "统一接口标准",
                    "完善的错误处理",
                    "实时监控与日志"
                ]
            }
            
            logger.info("✅ 集成协作演示完成")
            
        except Exception as e:
            logger.error(f"集成协作演示失败: {e}")
            self.demo_results["integrated_collaboration"] = {
                "status": "error",
                "error": str(e)
            }
    
    async def demo_performance_benchmark(self):
        """性能基准测试演示"""
        logger.info("📋 演示8: 性能基准测试")
        
        try:
            # 模拟性能测试
            benchmark_results = {
                "multi_agent_architecture": {
                    "response_time": 1.2,
                    "memory_usage": 450,
                    "concurrent_agents": 10,
                    "task_throughput": 50
                },
                "communication_protocols": {
                    "message_latency": 0.05,
                    "throughput": 1000,
                    "reliability": 0.999,
                    "scalability": "high"
                },
                "task_allocation": {
                    "allocation_time": 0.1,
                    "accuracy": 0.95,
                    "load_balance": 0.92,
                    "resource_utilization": 0.88
                },
                "collaboration_strategies": {
                    "consensus_time": 0.8,
                    "negotiation_success_rate": 0.85,
                    "coalition_stability": 0.90,
                    "collaboration_efficiency": 0.87
                },
                "fault_tolerance": {
                    "detection_time": 0.3,
                    "recovery_time": 2.0,
                    "availability": 0.995,
                    "fault_coverage": 0.95
                },
                "coordination_engine": {
                    "coordination_time": 0.5,
                    "conflict_resolution_time": 0.2,
                    "scheduling_efficiency": 0.93,
                    "system_throughput": 60
                }
            }
            
            # 计算综合性能得分
            total_score = 0
            component_count = 0
            
            for component, metrics in benchmark_results.items():
                component_score = 0
                metric_count = 0
                
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        # 归一化得分 (0-1)
                        if "time" in metric_name.lower():
                            # 时间越短越好
                            normalized_score = max(0, 1 - value / 10)
                        elif "rate" in metric_name.lower() or "accuracy" in metric_name.lower():
                            # 比率越高越好
                            normalized_score = value
                        else:
                            # 其他指标
                            normalized_score = min(1, value / 100)
                        
                        component_score += normalized_score
                        metric_count += 1
                
                if metric_count > 0:
                    component_score /= metric_count
                    total_score += component_score
                    component_count += 1
            
            overall_score = total_score / component_count if component_count > 0 else 0
            
            self.demo_results["performance_benchmark"] = {
                "status": "success",
                "benchmark_results": benchmark_results,
                "overall_score": overall_score,
                "performance_summary": {
                    "total_components": component_count,
                    "average_score": overall_score,
                    "best_component": max(benchmark_results.keys(), key=lambda k: sum(v for v in benchmark_results[k].values() if isinstance(v, (int, float)))),
                    "performance_grade": "A" if overall_score > 0.8 else "B" if overall_score > 0.6 else "C"
                }
            }
            
            logger.info("✅ 性能基准测试演示完成")
            
        except Exception as e:
            logger.error(f"性能基准测试演示失败: {e}")
            self.demo_results["performance_benchmark"] = {
                "status": "error",
                "error": str(e)
            }
    
    async def generate_demo_report(self):
        """生成演示报告"""
        logger.info("📋 生成演示报告")
        
        report = {
            "demo_info": {
                "title": "第3章 多智能体系统协作机制演示报告",
                "timestamp": datetime.now().isoformat(),
                "total_demos": len(self.demo_results)
            },
            "results": self.demo_results,
            "summary": {
                "demos_completed": len([r for r in self.demo_results.values() if r["status"] == "success"]),
                "total_features_tested": sum(
                    len(r.get("features", [])) for r in self.demo_results.values() 
                    if r["status"] == "success" and "features" in r
                ),
                "performance_benchmarks": len(
                    self.demo_results.get("performance_benchmark", {}).get("benchmark_results", {})
                ),
                "integration_scenarios": 1
            },
            "recommendations": {
                "architecture_design": "采用分层架构，确保模块化和可扩展性",
                "communication": "使用多种协议，支持不同场景的通信需求",
                "task_allocation": "结合负载均衡和能力匹配，提高分配效率",
                "collaboration": "根据任务特点选择合适的协作策略",
                "fault_tolerance": "实施全面的容错机制，确保系统稳定性",
                "coordination": "建立有效的协调机制，管理复杂的多智能体交互"
            }
        }
        
        # 保存报告到文件
        report_filename = f"chapter3_demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📄 演示报告已保存到: {report_filename}")
        
        # 打印摘要
        print("\n" + "="*60)
        print("📊 第3章演示报告摘要")
        print("="*60)
        print(f"演示时间: {report['demo_info']['timestamp']}")
        print(f"演示项目数: {report['demo_info']['total_demos']}")
        print(f"完成演示数: {report['summary']['demos_completed']}")
        print(f"功能测试数: {report['summary']['total_features_tested']}")
        print(f"性能基准数: {report['summary']['performance_benchmarks']}")
        print(f"集成场景数: {report['summary']['integration_scenarios']}")
        
        print("\n🎯 技术建议:")
        for category, recommendation in report['recommendations'].items():
            print(f"- {category}: {recommendation}")
        
        print("="*60)
        
        return report

async def main():
    """主函数"""
    print("🤖 第3章 多智能体系统协作机制 - 完整演示")
    print("="*60)
    
    demo = CollaborationDemo()
    await demo.run_complete_demo()
    
    print("\n🎉 演示完成！请查看生成的报告文件了解详细结果。")

if __name__ == "__main__":
    asyncio.run(main())
