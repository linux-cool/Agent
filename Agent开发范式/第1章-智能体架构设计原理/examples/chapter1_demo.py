# chapter1_demo.py
"""
第1章 智能体架构设计原理 - 演示程序
展示六大核心技术支柱的完整实现和使用方法
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any

# 导入基础智能体组件
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.base_agent import (
    BaseAgent, AgentConfig, AgentState, TaskStatus,
    TaskPlanner, MemoryManager, ToolManager, AutonomousLoop,
    SecurityController, MultiAgentCoordinator,
    CalculatorTool, WebSearchTool
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Chapter1Demo:
    """第1章演示类"""
    
    def __init__(self):
        self.agents = {}
        self.coordinator = MultiAgentCoordinator()
        self.demo_results = {}
    
    async def run_complete_demo(self):
        """运行完整演示"""
        logger.info("🚀 开始第1章智能体架构设计原理演示")
        
        # 1. 基础智能体演示
        await self.demo_basic_agent()
        
        # 2. 任务规划演示
        await self.demo_task_planning()
        
        # 3. 记忆管理演示
        await self.demo_memory_management()
        
        # 4. 工具调用演示
        await self.demo_tool_execution()
        
        # 5. 自治循环演示
        await self.demo_autonomous_loop()
        
        # 6. 安全控制演示
        await self.demo_security_control()
        
        # 7. 多智能体协作演示
        await self.demo_multi_agent_coordination()
        
        # 8. 性能测试演示
        await self.demo_performance_testing()
        
        # 9. 生成演示报告
        await self.generate_demo_report()
        
        logger.info("✅ 第1章演示完成")
    
    async def demo_basic_agent(self):
        """基础智能体演示"""
        logger.info("📋 演示1: 基础智能体创建和配置")
        
        # 创建智能体配置
        config = AgentConfig(
            name="DemoAgent",
            version="1.0.0",
            max_iterations=5,
            timeout=60,
            temperature=0.7
        )
        
        # 创建智能体
        agent = BaseAgent(config)
        
        # 注册工具
        calculator = CalculatorTool()
        web_search = WebSearchTool()
        agent.tool_manager.register_tool(calculator)
        agent.tool_manager.register_tool(web_search)
        
        # 测试基本功能
        result = await agent.process_input("Hello, I'm testing the basic agent")
        
        # 获取智能体状态
        status = agent.get_status()
        
        self.demo_results["basic_agent"] = {
            "config": {
                "name": config.name,
                "max_iterations": config.max_iterations,
                "timeout": config.timeout
            },
            "result": result,
            "status": status,
            "tools_registered": len(agent.tool_manager.tools)
        }
        
        logger.info(f"✅ 基础智能体演示完成，处理结果: {result[:50]}...")
    
    async def demo_task_planning(self):
        """任务规划演示"""
        logger.info("📋 演示2: 任务规划系统")
        
        planner = TaskPlanner()
        
        # 测试不同分解策略
        strategies = ["hierarchical", "dependency", "resource"]
        task_description = "Develop a new AI agent system"
        
        planning_results = {}
        
        for strategy in strategies:
            tasks = await planner.decompose_task(task_description, strategy)
            prioritized_tasks = await planner.prioritize_tasks(tasks)
            
            planning_results[strategy] = {
                "task_count": len(tasks),
                "tasks": [
                    {
                        "id": task.id,
                        "description": task.description,
                        "priority": task.priority,
                        "type": task.context.get("type", "unknown")
                    }
                    for task in prioritized_tasks
                ]
            }
        
        self.demo_results["task_planning"] = planning_results
        
        logger.info(f"✅ 任务规划演示完成，测试了{len(strategies)}种分解策略")
    
    async def demo_memory_management(self):
        """记忆管理演示"""
        logger.info("📋 演示3: 记忆管理系统")
        
        memory_manager = MemoryManager()
        
        # 存储不同类型的记忆
        await memory_manager.store_short_term("current_session", "User is asking about AI agents")
        await memory_manager.store_short_term("user_preference", "Prefers detailed explanations")
        
        await memory_manager.store_long_term("domain_knowledge", "AI agents are autonomous software entities")
        await memory_manager.store_long_term("best_practices", "Always validate input before processing")
        
        await memory_manager.store_process({
            "action": "user_query",
            "input": "What are AI agents?",
            "response": "AI agents are autonomous software entities..."
        })
        
        # 检索记忆
        context = await memory_manager.retrieve_context("AI agents")
        
        memory_stats = {
            "short_term_size": len(memory_manager.memory.short_term),
            "long_term_size": len(memory_manager.memory.long_term),
            "process_steps": len(memory_manager.memory.process),
            "retrieved_context": {
                "short_term_matches": len(context["short_term"]),
                "long_term_matches": len(context["long_term"]),
                "recent_process_steps": len(context["recent_process"])
            }
        }
        
        self.demo_results["memory_management"] = memory_stats
        
        logger.info(f"✅ 记忆管理演示完成，存储了{memory_stats['short_term_size']}个短期记忆")
    
    async def demo_tool_execution(self):
        """工具调用演示"""
        logger.info("📋 演示4: 工具调用系统")
        
        tool_manager = ToolManager()
        
        # 注册工具
        calculator = CalculatorTool()
        web_search = WebSearchTool()
        tool_manager.register_tool(calculator)
        tool_manager.register_tool(web_search)
        
        # 执行工具
        tool_results = {}
        
        # 计算器工具
        calc_result = await tool_manager.execute_tool("calculator", operation="(2+3)*4")
        tool_results["calculator"] = {
            "operation": "(2+3)*4",
            "result": calc_result,
            "execution_time": tool_manager.execution_history[-1]["execution_time"]
        }
        
        # 网络搜索工具
        search_result = await tool_manager.execute_tool("web_search", query="AI agent frameworks")
        tool_results["web_search"] = {
            "query": "AI agent frameworks",
            "result": search_result,
            "execution_time": tool_manager.execution_history[-1]["execution_time"]
        }
        
        # 获取工具统计
        tool_stats = tool_manager.get_tool_execution_stats()
        
        self.demo_results["tool_execution"] = {
            "results": tool_results,
            "stats": tool_stats,
            "available_tools": tool_manager.list_tools()
        }
        
        logger.info(f"✅ 工具调用演示完成，执行了{tool_stats['total_executions']}次工具调用")
    
    async def demo_autonomous_loop(self):
        """自治循环演示"""
        logger.info("📋 演示5: 自治循环系统")
        
        # 创建智能体
        config = AgentConfig(name="AutonomousAgent", max_iterations=3)
        agent = BaseAgent(config)
        
        # 注册工具
        calculator = CalculatorTool()
        agent.tool_manager.register_tool(calculator)
        
        # 执行ReAct循环
        start_time = time.time()
        result = await agent.autonomous_loop.react_loop("Calculate 5*6 and analyze the result")
        execution_time = time.time() - start_time
        
        # 获取过程记忆
        process_memory = agent.memory_manager.memory.process
        
        loop_stats = {
            "execution_time": execution_time,
            "result": result,
            "process_steps": len(process_memory),
            "phases_completed": len([step for step in process_memory if "phase" in step]),
            "agent_state": agent.state.value
        }
        
        self.demo_results["autonomous_loop"] = loop_stats
        
        logger.info(f"✅ 自治循环演示完成，执行时间: {execution_time:.2f}秒")
    
    async def demo_security_control(self):
        """安全控制演示"""
        logger.info("📋 演示6: 安全控制系统")
        
        security_controller = SecurityController()
        
        # 测试输入验证
        test_inputs = [
            "This is a valid input",
            "x" * 10001,  # 过长输入
            "<script>alert('xss')</script>",  # 恶意脚本
            "eval('malicious code')",  # 恶意代码
            "Normal user query about AI agents"  # 正常查询
        ]
        
        validation_results = {}
        for i, test_input in enumerate(test_inputs):
            is_valid = await security_controller.validate_input(test_input)
            validation_results[f"test_{i+1}"] = {
                "input": test_input[:50] + "..." if len(test_input) > 50 else test_input,
                "valid": is_valid
            }
        
        # 测试速率限制
        rate_limit_results = []
        for i in range(5):
            is_allowed = await security_controller.check_rate_limit()
            rate_limit_results.append(is_allowed)
        
        # 测试安全护栏
        sensitive_output = "User credit card: 1234-5678-9012-3456"
        filtered_output = await security_controller.apply_guardrails(sensitive_output)
        
        security_stats = {
            "input_validation": validation_results,
            "rate_limiting": {
                "requests_allowed": sum(rate_limit_results),
                "total_requests": len(rate_limit_results)
            },
            "guardrails": {
                "original_output": sensitive_output,
                "filtered_output": filtered_output,
                "filtering_applied": sensitive_output != filtered_output
            }
        }
        
        self.demo_results["security_control"] = security_stats
        
        logger.info(f"✅ 安全控制演示完成，验证了{len(test_inputs)}个输入样本")
    
    async def demo_multi_agent_coordination(self):
        """多智能体协作演示"""
        logger.info("📋 演示7: 多智能体协作系统")
        
        coordinator = MultiAgentCoordinator()
        
        # 创建多个智能体
        agents = {}
        for i in range(3):
            config = AgentConfig(name=f"Agent{i+1}")
            agent = BaseAgent(config)
            agents[f"agent{i+1}"] = agent
            coordinator.register_agent(f"agent{i+1}", agent)
        
        # 测试不同协调策略
        task = "Analyze market trends and generate report"
        agent_ids = ["agent1", "agent2", "agent3"]
        
        coordination_results = {}
        strategies = ["centralized", "distributed", "hybrid"]
        
        for strategy in strategies:
            results = await coordinator.coordinate_task(task, agent_ids, strategy)
            coordination_results[strategy] = {
                "agents_involved": len(results),
                "task_distribution": results
            }
        
        # 测试消息传递
        message_results = []
        for i in range(3):
            success = await coordinator.send_message("agent1", f"agent{i+1}", {
                "type": "coordination",
                "message": f"Hello from agent1 to agent{i+1}"
            })
            message_results.append(success)
        
        coordination_stats = {
            "strategies_tested": coordination_results,
            "message_passing": {
                "messages_sent": len(message_results),
                "successful_messages": sum(message_results)
            },
            "registered_agents": len(coordinator.agents)
        }
        
        self.demo_results["multi_agent_coordination"] = coordination_stats
        
        logger.info(f"✅ 多智能体协作演示完成，测试了{len(strategies)}种协调策略")
    
    async def demo_performance_testing(self):
        """性能测试演示"""
        logger.info("📋 演示8: 性能测试")
        
        # 创建性能测试智能体
        config = AgentConfig(name="PerformanceAgent", max_iterations=3)
        agent = BaseAgent(config)
        
        calculator = CalculatorTool()
        agent.tool_manager.register_tool(calculator)
        
        # 单次请求性能测试
        single_request_times = []
        for i in range(5):
            start_time = time.time()
            result = await agent.process_input(f"Performance test {i+1}")
            end_time = time.time()
            single_request_times.append(end_time - start_time)
        
        # 并发请求性能测试
        async def concurrent_request(request_id: int):
            start_time = time.time()
            result = await agent.process_input(f"Concurrent request {request_id}")
            end_time = time.time()
            return end_time - start_time
        
        concurrent_start = time.time()
        concurrent_tasks = [concurrent_request(i) for i in range(5)]
        concurrent_times = await asyncio.gather(*concurrent_tasks)
        concurrent_total = time.time() - concurrent_start
        
        # 内存使用测试
        memory_stats = {
            "short_term_size": len(agent.memory_manager.memory.short_term),
            "long_term_size": len(agent.memory_manager.memory.long_term),
            "process_steps": len(agent.memory_manager.memory.process)
        }
        
        performance_stats = {
            "single_request": {
                "times": single_request_times,
                "average": sum(single_request_times) / len(single_request_times),
                "min": min(single_request_times),
                "max": max(single_request_times)
            },
            "concurrent_request": {
                "individual_times": concurrent_times,
                "total_time": concurrent_total,
                "average": sum(concurrent_times) / len(concurrent_times)
            },
            "memory_usage": memory_stats,
            "tool_execution_stats": agent.tool_manager.get_tool_execution_stats()
        }
        
        self.demo_results["performance_testing"] = performance_stats
        
        logger.info(f"✅ 性能测试演示完成，平均响应时间: {performance_stats['single_request']['average']:.3f}秒")
    
    async def generate_demo_report(self):
        """生成演示报告"""
        logger.info("📋 生成演示报告")
        
        report = {
            "demo_info": {
                "title": "第1章 智能体架构设计原理演示报告",
                "timestamp": datetime.now().isoformat(),
                "total_demos": len(self.demo_results)
            },
            "results": self.demo_results,
            "summary": {
                "total_agents_created": len(self.agents) + 1,  # 包括演示中创建的智能体
                "total_tools_tested": 2,  # CalculatorTool 和 WebSearchTool
                "total_strategies_tested": 6,  # 3种任务分解策略 + 3种协调策略
                "total_security_tests": 5,  # 5个输入验证测试
                "performance_metrics": {
                    "average_response_time": self.demo_results["performance_testing"]["single_request"]["average"],
                    "concurrent_throughput": len(self.demo_results["performance_testing"]["concurrent_request"]["individual_times"]) / self.demo_results["performance_testing"]["concurrent_request"]["total_time"]
                }
            }
        }
        
        # 保存报告到文件
        report_filename = f"chapter1_demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📄 演示报告已保存到: {report_filename}")
        
        # 打印摘要
        print("\n" + "="*60)
        print("📊 第1章演示报告摘要")
        print("="*60)
        print(f"演示时间: {report['demo_info']['timestamp']}")
        print(f"演示项目数: {report['demo_info']['total_demos']}")
        print(f"创建智能体数: {report['summary']['total_agents_created']}")
        print(f"测试工具数: {report['summary']['total_tools_tested']}")
        print(f"测试策略数: {report['summary']['total_strategies_tested']}")
        print(f"安全测试数: {report['summary']['total_security_tests']}")
        print(f"平均响应时间: {report['summary']['performance_metrics']['average_response_time']:.3f}秒")
        print(f"并发吞吐量: {report['summary']['performance_metrics']['concurrent_throughput']:.2f}请求/秒")
        print("="*60)

async def main():
    """主函数"""
    print("🤖 第1章 智能体架构设计原理 - 完整演示")
    print("="*60)
    
    demo = Chapter1Demo()
    await demo.run_complete_demo()
    
    print("\n🎉 演示完成！请查看生成的报告文件了解详细结果。")

if __name__ == "__main__":
    asyncio.run(main())
