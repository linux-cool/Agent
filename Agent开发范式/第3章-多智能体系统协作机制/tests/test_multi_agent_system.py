# test_multi_agent_system.py
"""
第3章 多智能体系统协作机制 - 测试文件
测试多智能体系统的各个组件和功能
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any, Optional

# 导入被测试的模块
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.multi_agent_architecture import (
    MultiAgentSystem, AgentInfo, Task, AgentState, 
    CoordinationType, SchedulingPolicy
)
from code.communication_protocols import (
    CommunicationManager, ProtocolType, ProtocolConfig, 
    Message, MessageType, MessagePriority
)
from code.task_allocation import (
    TaskAllocator, AllocationStrategy, LoadBalancer, 
    LoadBalancingAlgorithm, ResourceCapacity, AgentMetrics
)
from code.collaboration_strategies import (
    CollaborationEngine, CollaborationStrategy, 
    ConsensusEngine, ConsensusAlgorithm
)
from code.fault_tolerance import (
    FaultToleranceManager, HealthStatus, 
    FaultToleranceStrategy
)
from code.coordination_engine import (
    CoordinationEngine, CoordinationType, 
    SchedulingPolicy, CoordinationTask, CoordinationEvent
)

class TestMultiAgentSystem:
    """多智能体系统测试类"""
    
    @pytest.fixture
    def system_config(self):
        """系统配置fixture"""
        return {
            "coordination_type": "centralized",
            "max_agents": 10,
            "heartbeat_interval": 5
        }
    
    @pytest.fixture
    def sample_agents(self):
        """示例智能体fixture"""
        return [
            AgentInfo(
                id="agent_1",
                name="Test Agent 1",
                state=AgentState.IDLE,
                capabilities=[
                    {"name": "research", "description": "Conduct research", "performance_score": 0.9},
                    {"name": "analysis", "description": "Analyze data", "performance_score": 0.8}
                ]
            ),
            AgentInfo(
                id="agent_2",
                name="Test Agent 2",
                state=AgentState.IDLE,
                capabilities=[
                    {"name": "analysis", "description": "Analyze data", "performance_score": 0.95},
                    {"name": "reporting", "description": "Generate reports", "performance_score": 0.85}
                ]
            )
        ]
    
    @pytest.fixture
    def sample_tasks(self):
        """示例任务fixture"""
        return [
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
            )
        ]
    
    @pytest.mark.asyncio
    async def test_system_initialization(self, system_config):
        """测试系统初始化"""
        system = MultiAgentSystem(system_config)
        
        assert system.config == system_config
        assert system.coordination_type == CoordinationType.CENTRALIZED
        assert system.max_agents == 10
        assert system.heartbeat_interval == 5
        assert len(system.agents) == 0
        assert len(system.tasks) == 0
    
    @pytest.mark.asyncio
    async def test_agent_registration(self, system_config, sample_agents):
        """测试智能体注册"""
        system = MultiAgentSystem(system_config)
        await system.start()
        
        # 注册智能体
        for agent in sample_agents:
            await system.add_agent(agent)
        
        assert len(system.agents) == 2
        assert "agent_1" in system.agents
        assert "agent_2" in system.agents
        
        # 测试重复注册
        with pytest.raises(ValueError):
            await system.add_agent(sample_agents[0])
        
        await system.stop()
    
    @pytest.mark.asyncio
    async def test_task_submission(self, system_config, sample_tasks):
        """测试任务提交"""
        system = MultiAgentSystem(system_config)
        await system.start()
        
        # 提交任务
        task_ids = []
        for task in sample_tasks:
            task_id = await system.submit_task(task)
            task_ids.append(task_id)
        
        assert len(task_ids) == 2
        assert len(system.tasks) == 2
        
        # 验证任务ID唯一性
        assert len(set(task_ids)) == len(task_ids)
        
        await system.stop()
    
    @pytest.mark.asyncio
    async def test_system_metrics(self, system_config, sample_agents, sample_tasks):
        """测试系统指标"""
        system = MultiAgentSystem(system_config)
        await system.start()
        
        # 添加智能体和任务
        for agent in sample_agents:
            await system.add_agent(agent)
        
        for task in sample_tasks:
            await system.submit_task(task)
        
        # 获取系统指标
        metrics = await system.get_system_metrics()
        
        assert "total_agents" in metrics
        assert "active_agents" in metrics
        assert "total_tasks" in metrics
        assert "pending_tasks" in metrics
        assert "completed_tasks" in metrics
        assert "system_uptime" in metrics
        
        assert metrics["total_agents"] == 2
        assert metrics["total_tasks"] == 2
        
        await system.stop()
    
    @pytest.mark.asyncio
    async def test_agent_state_management(self, system_config, sample_agents):
        """测试智能体状态管理"""
        system = MultiAgentSystem(system_config)
        await system.start()
        
        # 添加智能体
        for agent in sample_agents:
            await system.add_agent(agent)
        
        # 测试状态更新
        await system.update_agent_state("agent_1", AgentState.BUSY)
        assert system.agents["agent_1"].state == AgentState.BUSY
        
        await system.update_agent_state("agent_1", AgentState.IDLE)
        assert system.agents["agent_1"].state == AgentState.IDLE
        
        await system.stop()

class TestCommunicationProtocols:
    """通信协议测试类"""
    
    @pytest.fixture
    def comm_manager(self):
        """通信管理器fixture"""
        return CommunicationManager()
    
    @pytest.fixture
    def http_config(self):
        """HTTP配置fixture"""
        return ProtocolConfig(
            protocol_type=ProtocolType.HTTP,
            host="localhost",
            port=8080,
            encryption_key="test_key"
        )
    
    @pytest.fixture
    def sample_message(self):
        """示例消息fixture"""
        return Message(
            sender="agent_1",
            receiver="agent_2",
            message_type=MessageType.REQUEST,
            content="Test message",
            priority=MessagePriority.NORMAL
        )
    
    @pytest.mark.asyncio
    async def test_protocol_registration(self, comm_manager, http_config):
        """测试协议注册"""
        from code.communication_protocols import HTTPProtocol
        
        http_protocol = HTTPProtocol(http_config)
        comm_manager.register_protocol(ProtocolType.HTTP, http_protocol)
        
        assert ProtocolType.HTTP in comm_manager.protocols
        assert comm_manager.protocols[ProtocolType.HTTP] == http_protocol
    
    @pytest.mark.asyncio
    async def test_message_sending(self, comm_manager, sample_message):
        """测试消息发送"""
        # 模拟协议
        mock_protocol = AsyncMock()
        comm_manager.protocols[ProtocolType.HTTP] = mock_protocol
        comm_manager.active_protocol = ProtocolType.HTTP
        
        # 发送消息
        success = await comm_manager.send_message(sample_message)
        
        assert success is True
        mock_protocol.send_message.assert_called_once_with(sample_message)
    
    @pytest.mark.asyncio
    async def test_message_receiving(self, comm_manager):
        """测试消息接收"""
        # 模拟协议
        mock_protocol = AsyncMock()
        mock_message = Message(
            sender="agent_2",
            receiver="agent_1",
            message_type=MessageType.RESPONSE,
            content="Response message",
            priority=MessagePriority.NORMAL
        )
        mock_protocol.receive_message.return_value = mock_message
        comm_manager.protocols[ProtocolType.HTTP] = mock_protocol
        comm_manager.active_protocol = ProtocolType.HTTP
        
        # 接收消息
        received_message = await comm_manager.receive_message("agent_1")
        
        assert received_message == mock_message
        mock_protocol.receive_message.assert_called_once_with("agent_1")
    
    @pytest.mark.asyncio
    async def test_broadcast_message(self, comm_manager):
        """测试广播消息"""
        # 模拟协议
        mock_protocol = AsyncMock()
        comm_manager.protocols[ProtocolType.HTTP] = mock_protocol
        comm_manager.active_protocol = ProtocolType.HTTP
        
        # 广播消息
        await comm_manager.broadcast_message("agent_1", MessageType.NOTIFICATION, "Broadcast message")
        
        mock_protocol.broadcast_message.assert_called_once()

class TestTaskAllocation:
    """任务分配测试类"""
    
    @pytest.fixture
    def task_allocator(self):
        """任务分配器fixture"""
        return TaskAllocator(strategy=AllocationStrategy.LOAD_BALANCED)
    
    @pytest.fixture
    def sample_tasks(self):
        """示例任务fixture"""
        return [
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
            }
        ]
    
    @pytest.fixture
    def sample_agents(self):
        """示例智能体fixture"""
        return ["agent_1", "agent_2", "agent_3"]
    
    def test_agent_capabilities_registration(self, task_allocator):
        """测试智能体能力注册"""
        capabilities = {"research", "analysis", "reporting"}
        task_allocator.register_agent_capabilities("agent_1", capabilities)
        
        assert "agent_1" in task_allocator.agent_capabilities
        assert task_allocator.agent_capabilities["agent_1"] == capabilities
    
    def test_agent_resources_registration(self, task_allocator):
        """测试智能体资源注册"""
        resources = ResourceCapacity(cpu=1.0, memory=2.0)
        task_allocator.register_agent_resources("agent_1", resources)
        
        assert "agent_1" in task_allocator.agent_resources
        assert task_allocator.agent_resources["agent_1"] == resources
    
    def test_agent_cost_setting(self, task_allocator):
        """测试智能体成本设置"""
        task_allocator.set_agent_cost("agent_1", 1.5)
        
        assert "agent_1" in task_allocator.agent_costs
        assert task_allocator.agent_costs["agent_1"] == 1.5
    
    @pytest.mark.asyncio
    async def test_task_allocation(self, task_allocator, sample_tasks, sample_agents):
        """测试任务分配"""
        # 注册智能体能力
        task_allocator.register_agent_capabilities("agent_1", {"research", "analysis"})
        task_allocator.register_agent_capabilities("agent_2", {"analysis", "reporting"})
        task_allocator.register_agent_capabilities("agent_3", {"research", "reporting"})
        
        # 注册智能体资源
        task_allocator.register_agent_resources("agent_1", ResourceCapacity(cpu=1.0, memory=2.0))
        task_allocator.register_agent_resources("agent_2", ResourceCapacity(cpu=1.5, memory=1.5))
        task_allocator.register_agent_resources("agent_3", ResourceCapacity(cpu=0.8, memory=1.0))
        
        # 设置智能体成本
        task_allocator.set_agent_cost("agent_1", 1.0)
        task_allocator.set_agent_cost("agent_2", 1.2)
        task_allocator.set_agent_cost("agent_3", 0.8)
        
        # 分配任务
        allocations = await task_allocator.allocate_tasks(sample_tasks, sample_agents)
        
        assert len(allocations) == 2
        assert all("agent_id" in allocation for allocation in allocations)
        assert all("task_id" in allocation for allocation in allocations)
    
    def test_allocation_statistics(self, task_allocator):
        """测试分配统计"""
        stats = task_allocator.get_allocation_statistics()
        
        assert "total_allocations" in stats
        assert "successful_allocations" in stats
        assert "failed_allocations" in stats
        assert "average_allocation_time" in stats

class TestCollaborationStrategies:
    """协作策略测试类"""
    
    @pytest.fixture
    def collaboration_engine(self):
        """协作引擎fixture"""
        return CollaborationEngine()
    
    @pytest.fixture
    def sample_task_allocation(self):
        """示例任务分配fixture"""
        return {
            "agent_1": [{"id": "task_1", "priority": 5}],
            "agent_2": [{"id": "task_2", "priority": 3}],
            "agent_3": [{"id": "task_3", "priority": 4}]
        }
    
    def test_consensus_engine_initialization(self, collaboration_engine):
        """测试共识引擎初始化"""
        assert isinstance(collaboration_engine.consensus_engine, ConsensusEngine)
        assert collaboration_engine.consensus_engine.consensus_threshold == 0.5
    
    def test_agent_weight_setting(self, collaboration_engine):
        """测试智能体权重设置"""
        collaboration_engine.consensus_engine.set_agent_weight("agent_1", 1.5)
        
        assert "agent_1" in collaboration_engine.consensus_engine.agent_weights
        assert collaboration_engine.consensus_engine.agent_weights["agent_1"] == 1.5
    
    def test_consensus_threshold_setting(self, collaboration_engine):
        """测试共识阈值设置"""
        collaboration_engine.consensus_engine.set_consensus_threshold(0.7)
        
        assert collaboration_engine.consensus_engine.consensus_threshold == 0.7
    
    @pytest.mark.asyncio
    async def test_consensus_proposal(self, collaboration_engine):
        """测试共识提案"""
        proposal_id = await collaboration_engine.consensus_engine.propose("agent_1", "Test proposal")
        
        assert proposal_id is not None
        assert proposal_id in collaboration_engine.consensus_engine.proposals
    
    @pytest.mark.asyncio
    async def test_consensus_voting(self, collaboration_engine):
        """测试共识投票"""
        proposal_id = await collaboration_engine.consensus_engine.propose("agent_1", "Test proposal")
        
        await collaboration_engine.consensus_engine.vote("agent_1", proposal_id, True, "Agree")
        await collaboration_engine.consensus_engine.vote("agent_2", proposal_id, True, "Agree")
        await collaboration_engine.consensus_engine.vote("agent_3", proposal_id, False, "Disagree")
        
        consensus_stats = collaboration_engine.consensus_engine.get_consensus_statistics()
        
        assert consensus_stats["total_proposals"] == 1
        assert consensus_stats["successful_consensus"] == 1
    
    @pytest.mark.asyncio
    async def test_negotiation_offer(self, collaboration_engine):
        """测试协商提议"""
        offer_id = await collaboration_engine.negotiation_engine.make_offer(
            "agent_1", "agent_2", {"cpu": 0.5, "memory": 0.3}
        )
        
        assert offer_id is not None
        assert offer_id in collaboration_engine.negotiation_engine.offers
    
    @pytest.mark.asyncio
    async def test_negotiation_response(self, collaboration_engine):
        """测试协商响应"""
        offer_id = await collaboration_engine.negotiation_engine.make_offer(
            "agent_1", "agent_2", {"cpu": 0.5, "memory": 0.3}
        )
        
        accepted = await collaboration_engine.negotiation_engine.respond_to_offer(
            offer_id, "agent_2", {"cpu": 0.4, "memory": 0.4}
        )
        
        assert accepted is True
    
    @pytest.mark.asyncio
    async def test_coalition_creation(self, collaboration_engine):
        """测试联盟创建"""
        coalition_id = await collaboration_engine.coalition_manager.create_coalition(
            "agent_1", "Test coalition"
        )
        
        assert coalition_id is not None
        assert coalition_id in collaboration_engine.coalition_manager.coalitions
    
    @pytest.mark.asyncio
    async def test_coalition_joining(self, collaboration_engine):
        """测试联盟加入"""
        coalition_id = await collaboration_engine.coalition_manager.create_coalition(
            "agent_1", "Test coalition"
        )
        
        success = await collaboration_engine.coalition_manager.join_coalition("agent_2", coalition_id)
        
        assert success is True
    
    @pytest.mark.asyncio
    async def test_collaboration_execution(self, collaboration_engine, sample_task_allocation):
        """测试协作执行"""
        # 并行协作
        parallel_results = await collaboration_engine.execute_collaboration(
            sample_task_allocation, CollaborationStrategy.PARALLEL
        )
        
        assert parallel_results is not None
        assert "execution_time" in parallel_results
        assert "success_rate" in parallel_results
        
        # 共识协作
        consensus_results = await collaboration_engine.execute_collaboration(
            sample_task_allocation, CollaborationStrategy.CONSENSUS
        )
        
        assert consensus_results is not None
        assert "execution_time" in consensus_results
        assert "success_rate" in consensus_results

class TestFaultTolerance:
    """容错机制测试类"""
    
    @pytest.fixture
    def fault_tolerance_manager(self):
        """容错管理器fixture"""
        return FaultToleranceManager()
    
    @pytest.fixture
    def mock_health_check(self):
        """模拟健康检查函数"""
        async def health_check_func(agent_id: str) -> HealthStatus:
            await asyncio.sleep(0.1)
            return HealthStatus.HEALTHY
        return health_check_func
    
    @pytest.mark.asyncio
    async def test_fault_tolerance_start(self, fault_tolerance_manager, mock_health_check):
        """测试容错保护启动"""
        await fault_tolerance_manager.start_fault_tolerance("agent_1", mock_health_check)
        
        assert "agent_1" in fault_tolerance_manager.protected_agents
        assert fault_tolerance_manager.protected_agents["agent_1"]["health_check"] == mock_health_check
    
    @pytest.mark.asyncio
    async def test_fault_tolerance_stop(self, fault_tolerance_manager, mock_health_check):
        """测试容错保护停止"""
        await fault_tolerance_manager.start_fault_tolerance("agent_1", mock_health_check)
        await fault_tolerance_manager.stop_fault_tolerance("agent_1")
        
        assert "agent_1" not in fault_tolerance_manager.protected_agents
    
    @pytest.mark.asyncio
    async def test_health_check(self, fault_tolerance_manager, mock_health_check):
        """测试健康检查"""
        await fault_tolerance_manager.start_fault_tolerance("agent_1", mock_health_check)
        
        # 等待健康检查运行
        await asyncio.sleep(1)
        
        health_status = fault_tolerance_manager.get_agent_health("agent_1")
        assert health_status is not None
    
    @pytest.mark.asyncio
    async def test_fault_detection(self, fault_tolerance_manager):
        """测试故障检测"""
        metrics = {
            "response_time": 6.0,
            "error_rate": 0.05,
            "memory_usage": 0.7,
            "cpu_usage": 0.6
        }
        
        faults = await fault_tolerance_manager.check_agent_health("agent_1", metrics)
        
        assert isinstance(faults, list)
        # 根据阈值，应该检测到响应时间故障
        assert len(faults) > 0
        assert any(fault["type"] == "response_time" for fault in faults)
    
    def test_system_health(self, fault_tolerance_manager):
        """测试系统健康状态"""
        system_health = fault_tolerance_manager.get_system_health()
        
        assert "overall_health" in system_health
        assert "agent_health" in system_health
        assert "fault_summary" in system_health

class TestCoordinationEngine:
    """协调引擎测试类"""
    
    @pytest.fixture
    def coordination_engine(self):
        """协调引擎fixture"""
        return CoordinationEngine(coordination_type=CoordinationType.CENTRALIZED)
    
    @pytest.fixture
    def sample_coordination_task(self):
        """示例协调任务fixture"""
        return CoordinationTask(
            name="test_task",
            description="Test coordination task",
            priority=5,
            resource_requirements={"cpu": 0.5, "memory": 0.3}
        )
    
    def test_agent_role_registration(self, coordination_engine):
        """测试智能体角色注册"""
        coordination_engine.register_agent_role(
            "agent_1", "researcher", {"research", "analysis"}, 3
        )
        
        assert "agent_1" in coordination_engine.agent_roles
        assert coordination_engine.agent_roles["agent_1"]["role"] == "researcher"
        assert coordination_engine.agent_roles["agent_1"]["capabilities"] == {"research", "analysis"}
        assert coordination_engine.agent_roles["agent_1"]["priority"] == 3
    
    @pytest.mark.asyncio
    async def test_task_coordination(self, coordination_engine, sample_coordination_task):
        """测试任务协调"""
        task_id = await coordination_engine.coordinate_task(sample_coordination_task)
        
        assert task_id is not None
        assert task_id in coordination_engine.coordination_tasks
    
    @pytest.mark.asyncio
    async def test_coordination_event_handling(self, coordination_engine):
        """测试协调事件处理"""
        event = CoordinationEvent(
            event_type="task_completion",
            source_agent="agent_1",
            content={"task_id": "task_1", "result": "Task completed"}
        )
        
        result = await coordination_engine.handle_coordination_event(event)
        
        assert result is not None
        assert "handled" in result
        assert result["handled"] is True
    
    def test_coordination_statistics(self, coordination_engine):
        """测试协调统计"""
        stats = coordination_engine.get_coordination_statistics()
        
        assert "total_tasks" in stats
        assert "completed_tasks" in stats
        assert "failed_tasks" in stats
        assert "average_coordination_time" in stats

class TestIntegration:
    """集成测试类"""
    
    @pytest.mark.asyncio
    async def test_full_system_integration(self):
        """测试完整系统集成"""
        # 创建系统配置
        config = {
            "coordination_type": "centralized",
            "max_agents": 5,
            "heartbeat_interval": 5
        }
        
        # 创建多智能体系统
        system = MultiAgentSystem(config)
        await system.start()
        
        # 创建通信管理器
        comm_manager = CommunicationManager()
        
        # 创建任务分配器
        task_allocator = TaskAllocator(strategy=AllocationStrategy.LOAD_BALANCED)
        
        # 创建协作引擎
        collaboration_engine = CollaborationEngine()
        
        # 创建容错管理器
        fault_tolerance_manager = FaultToleranceManager()
        
        # 创建协调引擎
        coordination_engine = CoordinationEngine(coordination_type=CoordinationType.CENTRALIZED)
        
        # 注册智能体
        agent = AgentInfo(
            id="test_agent",
            name="Test Agent",
            state=AgentState.IDLE,
            capabilities=[
                {"name": "test", "description": "Test capability", "performance_score": 0.9}
            ]
        )
        await system.add_agent(agent)
        
        # 注册智能体能力
        task_allocator.register_agent_capabilities("test_agent", {"test"})
        task_allocator.register_agent_resources("test_agent", ResourceCapacity(cpu=1.0, memory=1.0))
        
        # 注册智能体角色
        coordination_engine.register_agent_role("test_agent", "tester", {"test"}, 1)
        
        # 创建任务
        task = Task(
            name="Integration Test Task",
            description="Test integration",
            priority=5,
            required_capabilities=["test"],
            resource_requirements={"cpu": 0.5, "memory": 0.5}
        )
        
        # 提交任务
        task_id = await system.submit_task(task)
        
        # 分配任务
        task_dict = {
            "id": task_id,
            "name": task.name,
            "priority": task.priority,
            "required_capabilities": task.required_capabilities,
            "resource_requirements": task.resource_requirements
        }
        allocations = await task_allocator.allocate_tasks([task_dict], ["test_agent"])
        
        # 协调任务
        coordination_task = CoordinationTask(
            name="coordination_test",
            description="Test coordination",
            priority=5,
            resource_requirements={"cpu": 0.5, "memory": 0.5}
        )
        coord_task_id = await coordination_engine.coordinate_task(coordination_task)
        
        # 获取系统指标
        system_metrics = await system.get_system_metrics()
        
        # 验证集成结果
        assert len(system.agents) == 1
        assert len(system.tasks) == 1
        assert len(allocations) == 1
        assert coord_task_id is not None
        assert system_metrics["total_agents"] == 1
        assert system_metrics["total_tasks"] == 1
        
        # 清理
        await system.stop()

# 性能测试
class TestPerformance:
    """性能测试类"""
    
    @pytest.mark.asyncio
    async def test_system_performance(self):
        """测试系统性能"""
        # 创建系统
        config = {
            "coordination_type": "centralized",
            "max_agents": 100,
            "heartbeat_interval": 5
        }
        
        system = MultiAgentSystem(config)
        await system.start()
        
        # 批量添加智能体
        start_time = time.time()
        
        for i in range(50):
            agent = AgentInfo(
                id=f"agent_{i}",
                name=f"Agent {i}",
                state=AgentState.IDLE,
                capabilities=[
                    {"name": "test", "description": "Test capability", "performance_score": 0.9}
                ]
            )
            await system.add_agent(agent)
        
        add_time = time.time() - start_time
        
        # 批量提交任务
        start_time = time.time()
        
        for i in range(100):
            task = Task(
                name=f"Task {i}",
                description=f"Test task {i}",
                priority=5,
                required_capabilities=["test"],
                resource_requirements={"cpu": 0.1, "memory": 0.1}
            )
            await system.submit_task(task)
        
        submit_time = time.time() - start_time
        
        # 获取系统指标
        start_time = time.time()
        metrics = await system.get_system_metrics()
        metrics_time = time.time() - start_time
        
        # 验证性能
        assert add_time < 5.0  # 添加50个智能体应在5秒内完成
        assert submit_time < 10.0  # 提交100个任务应在10秒内完成
        assert metrics_time < 1.0  # 获取指标应在1秒内完成
        
        assert metrics["total_agents"] == 50
        assert metrics["total_tasks"] == 100
        
        await system.stop()

# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
