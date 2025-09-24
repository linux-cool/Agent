# test_base_agent.py
"""
基础智能体测试用例
测试智能体核心功能和六大技术支柱
"""

import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from base_agent import (
    BaseAgent, AgentConfig, AgentState, TaskStatus, Task, Memory,
    TaskPlanner, MemoryManager, ToolManager, AutonomousLoop,
    SecurityController, MultiAgentCoordinator, CalculatorTool, WebSearchTool
)

class TestAgentConfig:
    """智能体配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = AgentConfig()
        
        assert config.name == "BaseAgent"
        assert config.version == "1.0.0"
        assert config.max_iterations == 10
        assert config.timeout == 300
        assert config.temperature == 0.7
        assert config.max_memory_size == 1000
        assert config.enable_security is True
        assert config.enable_monitoring is True
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = AgentConfig(
            name="TestAgent",
            max_iterations=5,
            timeout=60,
            temperature=0.5
        )
        
        assert config.name == "TestAgent"
        assert config.max_iterations == 5
        assert config.timeout == 60
        assert config.temperature == 0.5

class TestTask:
    """任务数据结构测试"""
    
    def test_task_creation(self):
        """测试任务创建"""
        task = Task(
            description="Test task",
            priority=5,
            status=TaskStatus.PENDING
        )
        
        assert task.description == "Test task"
        assert task.priority == 5
        assert task.status == TaskStatus.PENDING
        assert task.id is not None
        assert isinstance(task.created_at, datetime)
        assert isinstance(task.updated_at, datetime)
    
    def test_task_dependencies(self):
        """测试任务依赖"""
        task1 = Task(description="Task 1")
        task2 = Task(description="Task 2")
        
        task2.dependencies.append(task1.id)
        
        assert task1.id in task2.dependencies
        assert len(task2.dependencies) == 1

class TestTaskPlanner:
    """任务规划器测试"""
    
    @pytest.fixture
    def planner(self):
        return TaskPlanner()
    
    @pytest.mark.asyncio
    async def test_hierarchical_decomposition(self, planner):
        """测试层次化分解"""
        task_description = "Complete project"
        tasks = await planner.decompose_task(task_description, "hierarchical")
        
        assert len(tasks) == 4
        assert all(task.description for task in tasks)
        assert all(task.priority > 0 for task in tasks)
        
        # 检查任务类型
        task_types = [task.context.get("type") for task in tasks]
        assert "analysis" in task_types
        assert "planning" in task_types
        assert "execution" in task_types
        assert "review" in task_types
    
    @pytest.mark.asyncio
    async def test_dependency_decomposition(self, planner):
        """测试依赖关系分解"""
        task_description = "Process data"
        tasks = await planner.decompose_task(task_description, "dependency")
        
        assert len(tasks) == 3
        assert all(task.description for task in tasks)
        
        # 检查依赖关系
        for i, task in enumerate(tasks):
            if i > 0:
                assert len(task.dependencies) == 1
                assert task.dependencies[0] == tasks[i-1].id
    
    @pytest.mark.asyncio
    async def test_resource_decomposition(self, planner):
        """测试资源分解"""
        task_description = "Allocate resources"
        tasks = await planner.decompose_task(task_description, "resource")
        
        assert len(tasks) == 3
        assert all(task.description for task in tasks)
        
        # 检查资源相关任务
        task_types = [task.context.get("type") for task in tasks]
        assert "resource_allocation" in task_types
        assert "resource_execution" in task_types
        assert "resource_release" in task_types
    
    @pytest.mark.asyncio
    async def test_prioritize_tasks(self, planner):
        """测试任务优先级排序"""
        tasks = [
            Task("Low priority", priority=1),
            Task("High priority", priority=10),
            Task("Medium priority", priority=5),
        ]
        
        prioritized = await planner.prioritize_tasks(tasks)
        
        assert prioritized[0].priority == 10
        assert prioritized[1].priority == 5
        assert prioritized[2].priority == 1
    
    @pytest.mark.asyncio
    async def test_update_task_status(self, planner):
        """测试任务状态更新"""
        task = Task("Test task")
        planner.tasks.append(task)
        
        await planner.update_task_status(task.id, TaskStatus.COMPLETED, "Success")
        
        assert task.status == TaskStatus.COMPLETED
        assert task.result == "Success"
        assert task.updated_at > task.created_at

class TestMemoryManager:
    """记忆管理器测试"""
    
    @pytest.fixture
    def memory_manager(self):
        return MemoryManager(max_short_term=100, max_process_steps=50)
    
    @pytest.mark.asyncio
    async def test_store_retrieve_short_term(self, memory_manager):
        """测试短期记忆存储和检索"""
        await memory_manager.store_short_term("test_key", "test_value")
        
        context = await memory_manager.retrieve_context("test")
        assert "test_key" in context["short_term"]
        assert context["short_term"]["test_key"]["value"] == "test_value"
    
    @pytest.mark.asyncio
    async def test_store_retrieve_long_term(self, memory_manager):
        """测试长期记忆存储和检索"""
        await memory_manager.store_long_term("knowledge", "AI knowledge")
        
        context = await memory_manager.retrieve_context("AI")
        assert "knowledge" in context["long_term"]
        assert context["long_term"]["knowledge"]["value"] == "AI knowledge"
    
    @pytest.mark.asyncio
    async def test_store_process_memory(self, memory_manager):
        """测试过程记忆存储"""
        step = {"action": "test", "result": "success"}
        await memory_manager.store_process(step)
        
        assert len(memory_manager.memory.process) == 1
        assert memory_manager.memory.process[0]["action"] == "test"
        assert memory_manager.memory.process[0]["result"] == "success"
        assert "timestamp" in memory_manager.memory.process[0]
    
    @pytest.mark.asyncio
    async def test_memory_cleanup(self, memory_manager):
        """测试记忆清理"""
        # 添加超过限制的短期记忆
        for i in range(110):
            await memory_manager.store_short_term(f"key_{i}", f"value_{i}")
        
        # 检查是否清理到合理范围
        assert len(memory_manager.memory.short_term) <= 100
    
    @pytest.mark.asyncio
    async def test_process_memory_limit(self, memory_manager):
        """测试过程记忆限制"""
        # 添加超过限制的过程记忆
        for i in range(60):
            await memory_manager.store_process({"step": i})
        
        # 检查是否保持在限制范围内
        assert len(memory_manager.memory.process) <= 50

class TestToolManager:
    """工具管理器测试"""
    
    @pytest.fixture
    def tool_manager(self):
        return ToolManager()
    
    @pytest.fixture
    def calculator_tool(self):
        return CalculatorTool()
    
    @pytest.mark.asyncio
    async def test_register_tool(self, tool_manager, calculator_tool):
        """测试工具注册"""
        tool_manager.register_tool(calculator_tool)
        
        assert "calculator" in tool_manager.tools
        assert tool_manager.tools["calculator"] == calculator_tool
    
    @pytest.mark.asyncio
    async def test_execute_tool(self, tool_manager, calculator_tool):
        """测试工具执行"""
        tool_manager.register_tool(calculator_tool)
        
        result = await tool_manager.execute_tool("calculator", operation="2+2")
        
        assert result == 4.0
        assert len(tool_manager.execution_history) == 1
        assert tool_manager.execution_history[0]["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self, tool_manager):
        """测试执行不存在的工具"""
        with pytest.raises(ValueError, match="Tool nonexistent not found"):
            await tool_manager.execute_tool("nonexistent", operation="test")
    
    @pytest.mark.asyncio
    async def test_execute_tool_with_error(self, tool_manager, calculator_tool):
        """测试工具执行错误"""
        tool_manager.register_tool(calculator_tool)
        
        with pytest.raises(ValueError):
            await tool_manager.execute_tool("calculator", operation="invalid_operation")
        
        assert len(tool_manager.execution_history) == 1
        assert tool_manager.execution_history[0]["status"] == "failed"
    
    def test_list_tools(self, tool_manager, calculator_tool):
        """测试工具列表"""
        tool_manager.register_tool(calculator_tool)
        
        tools = tool_manager.list_tools()
        
        assert len(tools) == 1
        assert tools[0]["name"] == "calculator"
        assert tools[0]["description"] == "Performs mathematical calculations"
    
    def test_tool_execution_stats(self, tool_manager):
        """测试工具执行统计"""
        stats = tool_manager.get_tool_execution_stats()
        
        assert stats["total_executions"] == 0

class TestSecurityController:
    """安全控制器测试"""
    
    @pytest.fixture
    def security_controller(self):
        return SecurityController()
    
    @pytest.mark.asyncio
    async def test_validate_input_valid(self, security_controller):
        """测试有效输入验证"""
        valid_input = "This is a valid input"
        
        result = await security_controller.validate_input(valid_input)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_validate_input_too_long(self, security_controller):
        """测试过长输入验证"""
        long_input = "x" * 10001
        
        result = await security_controller.validate_input(long_input)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_validate_input_malicious(self, security_controller):
        """测试恶意输入验证"""
        malicious_input = "<script>alert('xss')</script>"
        
        result = await security_controller.validate_input(malicious_input)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_check_rate_limit(self, security_controller):
        """测试速率限制"""
        # 第一次检查应该通过
        result1 = await security_controller.check_rate_limit()
        assert result1 is True
        
        # 快速连续检查应该通过（在限制内）
        for _ in range(10):
            result = await security_controller.check_rate_limit()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_monitor_execution_timeout(self, security_controller):
        """测试执行超时监控"""
        start_time = 0  # 模拟很久以前开始
        
        result = await security_controller.monitor_execution(start_time)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_apply_guardrails(self, security_controller):
        """测试安全护栏应用"""
        sensitive_output = "Credit card: 1234-5678-9012-3456"
        
        filtered_output = await security_controller.apply_guardrails(sensitive_output)
        
        assert "[FILTERED]" in filtered_output
        assert "1234-5678-9012-3456" not in filtered_output

class TestMultiAgentCoordinator:
    """多智能体协调器测试"""
    
    @pytest.fixture
    def coordinator(self):
        return MultiAgentCoordinator()
    
    @pytest.fixture
    def mock_agent(self):
        return Mock()
    
    def test_register_agent(self, coordinator, mock_agent):
        """测试智能体注册"""
        coordinator.register_agent("agent1", mock_agent)
        
        assert "agent1" in coordinator.agents
        assert coordinator.agents["agent1"] == mock_agent
    
    @pytest.mark.asyncio
    async def test_send_message(self, coordinator, mock_agent):
        """测试消息发送"""
        coordinator.register_agent("agent1", mock_agent)
        coordinator.register_agent("agent2", mock_agent)
        
        message = {"type": "test", "content": "hello"}
        result = await coordinator.send_message("agent1", "agent2", message)
        
        assert result is True
        assert len(coordinator.message_queue) == 1
        assert coordinator.message_queue[0]["sender"] == "agent1"
        assert coordinator.message_queue[0]["receiver"] == "agent2"
    
    @pytest.mark.asyncio
    async def test_send_message_to_nonexistent_agent(self, coordinator):
        """测试向不存在的智能体发送消息"""
        message = {"type": "test", "content": "hello"}
        result = await coordinator.send_message("agent1", "nonexistent", message)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_broadcast_message(self, coordinator, mock_agent):
        """测试广播消息"""
        coordinator.register_agent("agent1", mock_agent)
        coordinator.register_agent("agent2", mock_agent)
        coordinator.register_agent("agent3", mock_agent)
        
        message = {"type": "broadcast", "content": "announcement"}
        await coordinator.broadcast_message("agent1", message)
        
        # 应该发送给agent2和agent3，不包括发送者agent1
        assert len(coordinator.message_queue) == 2
    
    @pytest.mark.asyncio
    async def test_centralized_coordination(self, coordinator, mock_agent):
        """测试集中式协调"""
        coordinator.register_agent("agent1", mock_agent)
        coordinator.register_agent("agent2", mock_agent)
        
        task = "Complete project"
        agent_ids = ["agent1", "agent2"]
        
        results = await coordinator.coordinate_task(task, agent_ids, "centralized")
        
        assert len(results) == 2
        assert "agent1" in results
        assert "agent2" in results
    
    @pytest.mark.asyncio
    async def test_distributed_coordination(self, coordinator, mock_agent):
        """测试分布式协调"""
        coordinator.register_agent("agent1", mock_agent)
        coordinator.register_agent("agent2", mock_agent)
        
        task = "Distributed task"
        agent_ids = ["agent1", "agent2"]
        
        results = await coordinator.coordinate_task(task, agent_ids, "distributed")
        
        assert len(results) == 2
        assert "agent1" in results
        assert "agent2" in results

class TestBaseAgent:
    """基础智能体测试"""
    
    @pytest.fixture
    def agent_config(self):
        return AgentConfig(
            name="TestAgent",
            max_iterations=5,
            timeout=60
        )
    
    @pytest.fixture
    def agent(self, agent_config):
        return BaseAgent(agent_config)
    
    @pytest.fixture
    def calculator_tool(self):
        return CalculatorTool()
    
    def test_agent_initialization(self, agent):
        """测试智能体初始化"""
        assert agent.name == "TestAgent"
        assert agent.state == AgentState.IDLE
        assert agent.config.max_iterations == 5
        assert agent.config.timeout == 60
        assert agent.task_planner is not None
        assert agent.memory_manager is not None
        assert agent.tool_manager is not None
        assert agent.autonomous_loop is not None
        assert agent.security_controller is not None
    
    def test_register_tool(self, agent, calculator_tool):
        """测试工具注册"""
        agent.tool_manager.register_tool(calculator_tool)
        
        assert "calculator" in agent.tool_manager.tools
    
    @pytest.mark.asyncio
    async def test_process_input_valid(self, agent, calculator_tool):
        """测试有效输入处理"""
        agent.tool_manager.register_tool(calculator_tool)
        
        result = await agent.process_input("Calculate 2+2")
        
        assert result is not None
        assert "2+2" in result or "Error" in result
    
    @pytest.mark.asyncio
    async def test_process_input_invalid(self, agent):
        """测试无效输入处理"""
        malicious_input = "<script>alert('xss')</script>"
        
        result = await agent.process_input(malicious_input)
        
        assert result == "Input validation failed"
    
    @pytest.mark.asyncio
    async def test_set_coordinator(self, agent):
        """测试设置协调器"""
        coordinator = MultiAgentCoordinator()
        agent.set_coordinator(coordinator)
        
        assert agent.coordinator == coordinator
        assert "TestAgent" in coordinator.agents
    
    def test_get_status(self, agent):
        """测试获取状态"""
        status = agent.get_status()
        
        assert status["name"] == "TestAgent"
        assert status["state"] == AgentState.IDLE.value
        assert "config" in status
        assert "tools" in status
        assert "tool_stats" in status
        assert "memory_stats" in status

class TestCalculatorTool:
    """计算器工具测试"""
    
    @pytest.fixture
    def calculator_tool(self):
        return CalculatorTool()
    
    def test_tool_properties(self, calculator_tool):
        """测试工具属性"""
        assert calculator_tool.name == "calculator"
        assert calculator_tool.description == "Performs mathematical calculations"
        assert "operation" in calculator_tool.parameters
    
    @pytest.mark.asyncio
    async def test_execute_valid_operation(self, calculator_tool):
        """测试有效操作执行"""
        result = await calculator_tool.execute(operation="2+2")
        
        assert result == 4.0
    
    @pytest.mark.asyncio
    async def test_execute_complex_operation(self, calculator_tool):
        """测试复杂操作执行"""
        result = await calculator_tool.execute(operation="(2+3)*4")
        
        assert result == 20.0
    
    @pytest.mark.asyncio
    async def test_execute_invalid_operation(self, calculator_tool):
        """测试无效操作执行"""
        with pytest.raises(ValueError):
            await calculator_tool.execute(operation="import os")
    
    @pytest.mark.asyncio
    async def test_execute_malicious_operation(self, calculator_tool):
        """测试恶意操作执行"""
        with pytest.raises(ValueError):
            await calculator_tool.execute(operation="__import__('os').system('ls')")

class TestWebSearchTool:
    """网络搜索工具测试"""
    
    @pytest.fixture
    def web_search_tool(self):
        return WebSearchTool()
    
    def test_tool_properties(self, web_search_tool):
        """测试工具属性"""
        assert web_search_tool.name == "web_search"
        assert web_search_tool.description == "Searches the web for information"
        assert "query" in web_search_tool.parameters
    
    @pytest.mark.asyncio
    async def test_execute_search(self, web_search_tool):
        """测试搜索执行"""
        result = await web_search_tool.execute(query="AI agents")
        
        assert "AI agents" in result
        assert "Search results for:" in result

class TestAutonomousLoop:
    """自治循环测试"""
    
    @pytest.fixture
    def agent_config(self):
        return AgentConfig(name="TestAgent", max_iterations=3)
    
    @pytest.fixture
    def agent(self, agent_config):
        return BaseAgent(agent_config)
    
    @pytest.fixture
    def autonomous_loop(self, agent):
        return AutonomousLoop(agent)
    
    @pytest.mark.asyncio
    async def test_think_phase(self, autonomous_loop):
        """测试思考阶段"""
        thoughts = await autonomous_loop.think("test input")
        
        assert "test input" in thoughts
        assert autonomous_loop.agent.state == AgentState.THINKING
    
    @pytest.mark.asyncio
    async def test_observe_phase(self, autonomous_loop):
        """测试观察阶段"""
        observation = await autonomous_loop.observe("test result")
        
        assert "test result" in observation
        assert autonomous_loop.agent.state == AgentState.OBSERVING
    
    @pytest.mark.asyncio
    async def test_reflect_phase(self, autonomous_loop):
        """测试反思阶段"""
        reflection = await autonomous_loop.reflect("test observation")
        
        assert "test observation" in reflection
        assert autonomous_loop.agent.state == AgentState.REFLECTING
    
    @pytest.mark.asyncio
    async def test_is_task_complete(self, autonomous_loop):
        """测试任务完成检查"""
        # 测试完成的情况
        complete_reflection = "Task is complete"
        assert await autonomous_loop.is_task_complete(complete_reflection) is True
        
        # 测试未完成的情况
        incomplete_reflection = "Task is in progress"
        assert await autonomous_loop.is_task_complete(incomplete_reflection) is False

# 集成测试
class TestAgentIntegration:
    """智能体集成测试"""
    
    @pytest.fixture
    async def full_agent(self):
        """创建完整的智能体"""
        config = AgentConfig(name="IntegrationAgent", max_iterations=3)
        agent = BaseAgent(config)
        
        # 注册工具
        calculator = CalculatorTool()
        web_search = WebSearchTool()
        agent.tool_manager.register_tool(calculator)
        agent.tool_manager.register_tool(web_search)
        
        return agent
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, full_agent):
        """测试端到端工作流"""
        input_data = "Calculate 3+3 and search for AI"
        
        result = await full_agent.process_input(input_data)
        
        assert result is not None
        assert len(result) > 0
        
        # 验证记忆是否被正确存储
        context = await full_agent.memory_manager.retrieve_context("calculate")
        assert len(context["short_term"]) > 0 or len(context["long_term"]) > 0
    
    @pytest.mark.asyncio
    async def test_multi_agent_coordination(self):
        """测试多智能体协调"""
        coordinator = MultiAgentCoordinator()
        
        # 创建多个智能体
        agent1 = BaseAgent(AgentConfig(name="Agent1"))
        agent2 = BaseAgent(AgentConfig(name="Agent2"))
        
        coordinator.register_agent("agent1", agent1)
        coordinator.register_agent("agent2", agent2)
        
        # 测试任务协调
        task = "Analyze market data"
        agent_ids = ["agent1", "agent2"]
        
        results = await coordinator.coordinate_task(task, agent_ids)
        
        assert len(results) == 2
        assert "agent1" in results
        assert "agent2" in results

# 性能测试
class TestAgentPerformance:
    """智能体性能测试"""
    
    @pytest.fixture
    async def performance_agent(self):
        """创建性能测试智能体"""
        config = AgentConfig(name="PerformanceAgent", max_iterations=5)
        agent = BaseAgent(config)
        
        calculator = CalculatorTool()
        agent.tool_manager.register_tool(calculator)
        
        return agent
    
    @pytest.mark.asyncio
    async def test_response_time(self, performance_agent):
        """测试响应时间"""
        import time
        
        start_time = time.time()
        result = await performance_agent.process_input("Simple test input")
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # 响应时间应该在合理范围内
        assert response_time < 5.0  # 5秒内完成
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, performance_agent):
        """测试并发请求处理"""
        async def make_request(request_id: int):
            return await performance_agent.process_input(f"Request {request_id}")
        
        # 并发发送5个请求
        tasks = [make_request(i) for i in range(5)]
        start_time = time.time()
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 验证所有请求都成功
        assert len(results) == 5
        assert all(result is not None for result in results)
        
        # 并发处理应该比串行处理快
        assert total_time < 10.0  # 5个请求在10秒内完成

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
