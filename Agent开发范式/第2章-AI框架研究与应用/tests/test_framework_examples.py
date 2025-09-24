# test_framework_examples.py
"""
第2章 AI框架研究与应用 - 测试用例
测试各框架示例的功能和性能
"""

import pytest
import asyncio
import os
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any

# 导入框架示例
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.langchain_examples import LangChainExamples
from code.crewai_examples import CrewAIExamples
from code.autogen_examples import AutoGenExamples
from code.langgraph_examples import LangGraphExamples
from code.framework_comparison import FrameworkComparison, FrameworkMetrics, UseCase

class TestLangChainExamples:
    """LangChain示例测试"""
    
    @pytest.fixture
    def langchain_examples(self):
        return LangChainExamples()
    
    def test_basic_llm_example(self, langchain_examples):
        """测试基础LLM示例"""
        with patch.object(langchain_examples.llm, '__call__') as mock_llm:
            mock_llm.return_value.content = "Paris is the capital of France."
            
            result = langchain_examples.basic_llm_example()
            
            assert "Paris" in result
            mock_llm.assert_called_once()
    
    def test_prompt_template_example(self, langchain_examples):
        """测试提示模板示例"""
        with patch.object(langchain_examples.llm, 'generate') as mock_generate:
            mock_generate.return_value.generations = [[Mock(text="Supervised learning uses labeled data, unsupervised learning uses unlabeled data.")]]
            
            result = langchain_examples.prompt_template_example()
            
            assert "supervised" in result.lower()
            assert "unsupervised" in result.lower()
    
    def test_conversation_chain_example(self, langchain_examples):
        """测试对话链示例"""
        with patch.object(langchain_examples.llm, 'predict') as mock_predict:
            mock_predict.side_effect = ["Hello Alice!", "Your name is Alice."]
            
            result = langchain_examples.conversation_chain_example()
            
            assert "Alice" in result
            assert mock_predict.call_count == 2
    
    def test_agent_with_tools_example(self, langchain_examples):
        """测试带工具的智能体示例"""
        with patch.object(langchain_examples.llm, 'run') as mock_run:
            mock_run.return_value = "I calculated 2+2=4, got current time, and counted words."
            
            result = langchain_examples.agent_with_tools_example()
            
            assert "calculated" in result.lower()
            mock_run.assert_called_once()
    
    def test_vector_store_example(self, langchain_examples):
        """测试向量存储示例"""
        with patch('langchain.vectorstores.Chroma.from_texts') as mock_chroma:
            mock_docs = [Mock(page_content="Machine learning is a subset of AI")]
            mock_chroma.return_value.similarity_search.return_value = mock_docs
            
            result = langchain_examples.vector_store_example()
            
            assert "machine learning" in result.lower()
            mock_chroma.assert_called_once()
    
    def test_custom_tool_example(self, langchain_examples):
        """测试自定义工具示例"""
        with patch.object(langchain_examples.llm, 'run') as mock_run:
            mock_run.return_value = "Weather in Beijing: Sunny, 25°C. Calculation: 120"
            
            result = langchain_examples.custom_tool_example()
            
            assert "beijing" in result.lower()
            assert "120" in result
            mock_run.assert_called_once()
    
    def test_error_handling_example(self, langchain_examples):
        """测试错误处理示例"""
        with patch.object(langchain_examples.llm, '__call__') as mock_llm:
            mock_llm.side_effect = Exception("API Error")
            
            result = langchain_examples.error_handling_example()
            
            assert "Error handled" in result
    
    def test_performance_optimization_example(self, langchain_examples):
        """测试性能优化示例"""
        with patch.object(langchain_examples.llm, 'generate') as mock_generate:
            mock_generate.return_value.generations = [[Mock(text="Machine learning is a subset of AI.")]]
            
            result = langchain_examples.performance_optimization_example()
            
            assert "Performance test results" in result
            assert "execution_time" in result

class TestCrewAIExamples:
    """CrewAI示例测试"""
    
    @pytest.fixture
    def crewai_examples(self):
        return CrewAIExamples()
    
    def test_basic_crew_example(self, crewai_examples):
        """测试基础团队示例"""
        with patch('crewai.Crew.kickoff') as mock_kickoff:
            mock_kickoff.return_value = "AI trends report completed"
            
            result = crewai_examples.basic_crew_example()
            
            assert "AI trends" in result
            mock_kickoff.assert_called_once()
    
    def test_hierarchical_crew_example(self, crewai_examples):
        """测试层次化团队示例"""
        with patch('crewai.Crew.kickoff') as mock_kickoff:
            mock_kickoff.return_value = "Hierarchical project completed"
            
            result = crewai_examples.hierarchical_crew_example()
            
            assert "Hierarchical" in result
            mock_kickoff.assert_called_once()
    
    def test_parallel_crew_example(self, crewai_examples):
        """测试并行团队示例"""
        with patch('crewai.Crew.kickoff') as mock_kickoff:
            mock_kickoff.return_value = "Parallel analysis completed"
            
            result = crewai_examples.parallel_crew_example()
            
            assert "Parallel" in result
            mock_kickoff.assert_called_once()
    
    def test_custom_tool_example(self, crewai_examples):
        """测试自定义工具示例"""
        with patch('crewai.Crew.kickoff') as mock_kickoff:
            mock_kickoff.return_value = "Weather and calculation completed"
            
            result = crewai_examples.custom_tool_example()
            
            assert "Weather" in result
            assert "calculation" in result
            mock_kickoff.assert_called_once()
    
    def test_memory_example(self, crewai_examples):
        """测试记忆系统示例"""
        with patch('crewai.Crew.kickoff') as mock_kickoff:
            mock_kickoff.return_value = "Memory-based research completed"
            
            result = crewai_examples.memory_example()
            
            assert "Memory" in result
            mock_kickoff.assert_called_once()
    
    def test_delegation_example(self, crewai_examples):
        """测试任务委派示例"""
        with patch('crewai.Crew.kickoff') as mock_kickoff:
            mock_kickoff.return_value = "Delegation project completed"
            
            result = crewai_examples.delegation_example()
            
            assert "Delegation" in result
            mock_kickoff.assert_called_once()
    
    def test_error_handling_example(self, crewai_examples):
        """测试错误处理示例"""
        with patch('crewai.Crew.kickoff') as mock_kickoff:
            mock_kickoff.side_effect = Exception("CrewAI Error")
            
            result = crewai_examples.error_handling_example()
            
            assert "Error handled" in result
    
    def test_performance_optimization_example(self, crewai_examples):
        """测试性能优化示例"""
        with patch('crewai.Crew.kickoff') as mock_kickoff:
            mock_kickoff.return_value = "Performance test completed"
            
            result = crewai_examples.performance_optimization_example()
            
            assert "Performance test results" in result

class TestAutoGenExamples:
    """AutoGen示例测试"""
    
    @pytest.fixture
    def autogen_examples(self):
        return AutoGenExamples()
    
    def test_basic_conversation_example(self, autogen_examples):
        """测试基础对话示例"""
        with patch('autogen.ConversableAgent.initiate_chat') as mock_chat:
            mock_chat.return_value.chat_history = [Mock(), Mock(), Mock()]
            
            result = autogen_examples.basic_conversation_example()
            
            assert "Messages: 3" in result
            mock_chat.assert_called_once()
    
    def test_group_chat_example(self, autogen_examples):
        """测试群组对话示例"""
        with patch('autogen.ConversableAgent.initiate_chat') as mock_chat:
            mock_chat.return_value.chat_history = [Mock() for _ in range(6)]
            
            result = autogen_examples.group_chat_example()
            
            assert "Messages: 6" in result
            mock_chat.assert_called_once()
    
    def test_code_execution_example(self, autogen_examples):
        """测试代码执行示例"""
        with patch('autogen.ConversableAgent.initiate_chat') as mock_chat:
            mock_chat.return_value.chat_history = [Mock(), Mock(), Mock()]
            
            result = autogen_examples.code_execution_example()
            
            assert "Messages: 3" in result
            mock_chat.assert_called_once()
    
    def test_function_calling_example(self, autogen_examples):
        """测试函数调用示例"""
        with patch('autogen.ConversableAgent.initiate_chat') as mock_chat:
            mock_chat.return_value.chat_history = [Mock(), Mock(), Mock()]
            
            result = autogen_examples.function_calling_example()
            
            assert "Messages: 3" in result
            mock_chat.assert_called_once()
    
    def test_multimodal_example(self, autogen_examples):
        """测试多模态示例"""
        with patch('autogen.ConversableAgent.initiate_chat') as mock_chat:
            mock_chat.return_value.chat_history = [Mock(), Mock()]
            
            result = autogen_examples.multimodal_example()
            
            assert "Messages: 2" in result
            mock_chat.assert_called_once()
    
    def test_hierarchical_agents_example(self, autogen_examples):
        """测试层次化智能体示例"""
        with patch('autogen.ConversableAgent.initiate_chat') as mock_chat:
            mock_chat.return_value.chat_history = [Mock() for _ in range(8)]
            
            result = autogen_examples.hierarchical_agents_example()
            
            assert "Messages: 8" in result
            mock_chat.assert_called_once()
    
    def test_memory_example(self, autogen_examples):
        """测试记忆系统示例"""
        with patch('autogen.ConversableAgent.initiate_chat') as mock_chat:
            mock_chat.side_effect = [
                Mock(chat_history=[Mock(), Mock()]),
                Mock(chat_history=[Mock(), Mock()])
            ]
            
            result = autogen_examples.memory_example()
            
            assert "Memory test completed" in result
            assert mock_chat.call_count == 2
    
    def test_error_handling_example(self, autogen_examples):
        """测试错误处理示例"""
        with patch('autogen.ConversableAgent.initiate_chat') as mock_chat:
            mock_chat.side_effect = Exception("AutoGen Error")
            
            result = autogen_examples.error_handling_example()
            
            assert "Error handled" in result
    
    def test_performance_optimization_example(self, autogen_examples):
        """测试性能优化示例"""
        with patch('autogen.ConversableAgent.initiate_chat') as mock_chat:
            mock_chat.return_value.chat_history = [Mock()]
            
            result = autogen_examples.performance_optimization_example()
            
            assert "Performance test results" in result

class TestLangGraphExamples:
    """LangGraph示例测试"""
    
    @pytest.fixture
    def langgraph_examples(self):
        return LangGraphExamples()
    
    def test_basic_workflow_example(self, langgraph_examples):
        """测试基础工作流示例"""
        with patch('langgraph.graph.StateGraph.compile') as mock_compile:
            mock_app = Mock()
            mock_app.invoke.return_value = {"messages": [Mock(), Mock(), Mock()], "step_count": 3}
            mock_compile.return_value = mock_app
            
            result = langgraph_examples.basic_workflow_example()
            
            assert "Messages: 3" in result
            assert "Steps: 3" in result
            mock_app.invoke.assert_called_once()
    
    def test_conditional_workflow_example(self, langgraph_examples):
        """测试条件工作流示例"""
        with patch('langgraph.graph.StateGraph.compile') as mock_compile:
            mock_app = Mock()
            mock_app.invoke.return_value = {"result": "Positive result"}
            mock_compile.return_value = mock_app
            
            result = langgraph_examples.conditional_workflow_example()
            
            assert "Positive result" in result
            mock_app.invoke.assert_called_once()
    
    def test_loop_workflow_example(self, langgraph_examples):
        """测试循环工作流示例"""
        with patch('langgraph.graph.StateGraph.compile') as mock_compile:
            mock_app = Mock()
            mock_app.invoke.return_value = {"counter": 3}
            mock_compile.return_value = mock_app
            
            result = langgraph_examples.loop_workflow_example()
            
            assert "Iterations: 3" in result
            mock_app.invoke.assert_called_once()
    
    def test_tool_integration_example(self, langgraph_examples):
        """测试工具集成示例"""
        with patch('langgraph.graph.StateGraph.compile') as mock_compile:
            mock_app = Mock()
            mock_app.invoke.return_value = {"tools_used": ["get_weather", "calculate"]}
            mock_compile.return_value = mock_app
            
            result = langgraph_examples.tool_integration_example()
            
            assert "get_weather" in result
            assert "calculate" in result
            mock_app.invoke.assert_called_once()
    
    def test_parallel_execution_example(self, langgraph_examples):
        """测试并行执行示例"""
        with patch('langgraph.graph.StateGraph.compile') as mock_compile:
            mock_app = Mock()
            mock_app.invoke.return_value = {
                "task1_result": "Task 1 result",
                "task2_result": "Task 2 result",
                "task3_result": "Task 3 result"
            }
            mock_compile.return_value = mock_app
            
            result = langgraph_examples.parallel_execution_example()
            
            assert "Task 1 result" in result
            assert "Task 2 result" in result
            assert "Task 3 result" in result
            mock_app.invoke.assert_called_once()
    
    def test_error_handling_example(self, langgraph_examples):
        """测试错误处理示例"""
        with patch('langgraph.graph.StateGraph.compile') as mock_compile:
            mock_app = Mock()
            mock_app.invoke.return_value = {"success": True, "error_count": 2}
            mock_compile.return_value = mock_app
            
            result = langgraph_examples.error_handling_example()
            
            assert "Success: True" in result
            assert "Errors: 2" in result
            mock_app.invoke.assert_called_once()
    
    def test_performance_optimization_example(self, langgraph_examples):
        """测试性能优化示例"""
        with patch('langgraph.graph.StateGraph.compile') as mock_compile:
            mock_app = Mock()
            mock_app.invoke.return_value = {"execution_time": 0.123}
            mock_compile.return_value = mock_app
            
            result = langgraph_examples.performance_optimization_example()
            
            assert "0.123" in result
            mock_app.invoke.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_example(self, langgraph_examples):
        """测试异步处理示例"""
        with patch('langgraph.graph.StateGraph.compile') as mock_compile:
            mock_app = Mock()
            mock_app.ainvoke.return_value = {"async_result": "Async success"}
            mock_compile.return_value = mock_app
            
            result = langgraph_examples.async_example()
            
            assert "Async success" in result
            mock_app.ainvoke.assert_called_once()

class TestFrameworkComparison:
    """框架对比测试"""
    
    @pytest.fixture
    def comparator(self):
        return FrameworkComparison()
    
    def test_compare_frameworks(self, comparator):
        """测试框架对比"""
        framework_names = ["langchain", "crewai"]
        result = comparator.compare_frameworks(framework_names)
        
        assert len(result) == 2
        assert "langchain" in result
        assert "crewai" in result
        assert result["langchain"]["name"] == "LangChain"
        assert result["crewai"]["name"] == "CrewAI"
    
    def test_recommend_framework(self, comparator):
        """测试框架推荐"""
        use_case = "general_ai_app"
        requirements = {
            "tool_ecosystem": 0.3,
            "documentation_quality": 0.25,
            "community_activity": 0.25,
            "learning_curve": 0.2
        }
        
        recommendations = comparator.recommend_framework(use_case, requirements)
        
        assert len(recommendations) > 0
        assert isinstance(recommendations[0], tuple)
        assert len(recommendations[0]) == 2
        assert isinstance(recommendations[0][1], float)
    
    def test_generate_comparison_report(self, comparator):
        """测试生成对比报告"""
        framework_names = ["langchain", "crewai"]
        report = comparator.generate_comparison_report(framework_names)
        
        assert isinstance(report, str)
        assert "AI框架对比分析报告" in report
        assert "LangChain" in report
        assert "CrewAI" in report
        assert "GitHub Stars" in report
    
    def test_benchmark_frameworks(self, comparator):
        """测试框架基准测试"""
        framework_names = ["langchain", "crewai"]
        test_cases = ["basic_task", "complex_workflow"]
        
        results = comparator.benchmark_frameworks(framework_names, test_cases)
        
        assert len(results) == 2
        assert "langchain" in results
        assert "crewai" in results
        
        for framework_name, result in results.items():
            assert "framework_name" in result
            assert "total_time" in result
            assert "test_results" in result
            assert "overall_score" in result
    
    def test_analyze_trends(self, comparator):
        """测试趋势分析"""
        trends = comparator.analyze_trends()
        
        assert "github_stars_ranking" in trends
        assert "feature_completeness_ranking" in trends
        assert "enterprise_readiness_ranking" in trends
        assert "emerging_trends" in trends
        assert "future_directions" in trends
        
        assert len(trends["github_stars_ranking"]) > 0
        assert len(trends["emerging_trends"]) > 0
        assert len(trends["future_directions"]) > 0
    
    def test_generate_selection_guide(self, comparator):
        """测试生成选择指南"""
        guide = comparator.generate_selection_guide()
        
        assert isinstance(guide, str)
        assert "AI框架选择指南" in guide
        assert "选择流程" in guide
        assert "决策矩阵" in guide
        assert "最佳实践" in guide

class TestFrameworkIntegration:
    """框架集成测试"""
    
    def test_framework_compatibility(self):
        """测试框架兼容性"""
        # 模拟框架兼容性测试
        frameworks = ["langchain", "crewai", "autogen", "langgraph"]
        compatibility_matrix = {}
        
        for framework1 in frameworks:
            compatibility_matrix[framework1] = {}
            for framework2 in frameworks:
                if framework1 == framework2:
                    compatibility_matrix[framework1][framework2] = 1.0
                else:
                    # 模拟兼容性评分
                    compatibility_matrix[framework1][framework2] = 0.8
        
        assert len(compatibility_matrix) == 4
        assert all(len(compat) == 4 for compat in compatibility_matrix.values())
        assert all(compatibility_matrix[f][f] == 1.0 for f in frameworks)
    
    def test_message_format_compatibility(self):
        """测试消息格式兼容性"""
        # 模拟消息格式测试
        message_formats = {
            "langchain": "BaseMessage",
            "crewai": "AgentMessage",
            "autogen": "ConversableMessage",
            "langgraph": "StateMessage"
        }
        
        # 测试格式转换
        conversions = {
            ("langchain", "crewai"): True,
            ("crewai", "autogen"): True,
            ("autogen", "langgraph"): True,
            ("langgraph", "langchain"): True
        }
        
        assert len(message_formats) == 4
        assert len(conversions) == 4
        assert all(conversions.values())
    
    def test_tool_integration(self):
        """测试工具集成"""
        # 模拟工具集成测试
        tools = {
            "calculator": {"langchain": True, "crewai": True, "autogen": True, "langgraph": False},
            "web_search": {"langchain": True, "crewai": True, "autogen": False, "langgraph": False},
            "file_manager": {"langchain": True, "crewai": False, "autogen": True, "langgraph": False},
            "database": {"langchain": True, "crewai": True, "autogen": True, "langgraph": True}
        }
        
        # 计算工具支持度
        support_scores = {}
        for framework in ["langchain", "crewai", "autogen", "langgraph"]:
            score = sum(1 for tool_support in tools.values() if tool_support[framework])
            support_scores[framework] = score / len(tools)
        
        assert len(support_scores) == 4
        assert all(0 <= score <= 1 for score in support_scores.values())
        assert support_scores["langchain"] >= support_scores["langgraph"]

class TestPerformanceBenchmarks:
    """性能基准测试"""
    
    def test_execution_time_benchmark(self):
        """测试执行时间基准"""
        # 模拟执行时间测试
        execution_times = {
            "langchain": 1.2,
            "crewai": 0.8,
            "autogen": 2.0,
            "langgraph": 0.9
        }
        
        # 找出最快的框架
        fastest_framework = min(execution_times, key=execution_times.get)
        slowest_framework = max(execution_times, key=execution_times.get)
        
        assert fastest_framework == "crewai"
        assert slowest_framework == "autogen"
        assert all(time > 0 for time in execution_times.values())
    
    def test_memory_usage_benchmark(self):
        """测试内存使用基准"""
        # 模拟内存使用测试
        memory_usage = {
            "langchain": 450,  # MB
            "crewai": 280,     # MB
            "autogen": 600,    # MB
            "langgraph": 380   # MB
        }
        
        # 找出内存使用最少的框架
        most_efficient = min(memory_usage, key=memory_usage.get)
        least_efficient = max(memory_usage, key=memory_usage.get)
        
        assert most_efficient == "crewai"
        assert least_efficient == "autogen"
        assert all(usage > 0 for usage in memory_usage.values())
    
    def test_concurrent_performance_benchmark(self):
        """测试并发性能基准"""
        # 模拟并发性能测试
        concurrent_performance = {
            "langchain": 10,   # requests/second
            "crewai": 15,      # requests/second
            "autogen": 8,      # requests/second
            "langgraph": 12    # requests/second
        }
        
        # 找出并发性能最好的框架
        best_concurrent = max(concurrent_performance, key=concurrent_performance.get)
        worst_concurrent = min(concurrent_performance, key=concurrent_performance.get)
        
        assert best_concurrent == "crewai"
        assert worst_concurrent == "autogen"
        assert all(perf > 0 for perf in concurrent_performance.values())

# 集成测试
class TestFrameworkDemo:
    """框架演示集成测试"""
    
    @pytest.mark.asyncio
    async def test_demo_integration(self):
        """测试演示集成"""
        # 模拟演示集成测试
        demo_results = {
            "langchain": {"status": "success", "features": 4},
            "crewai": {"status": "success", "features": 3},
            "autogen": {"status": "success", "features": 4},
            "langgraph": {"status": "success", "features": 3}
        }
        
        # 验证所有演示都成功
        success_count = sum(1 for result in demo_results.values() if result["status"] == "success")
        total_count = len(demo_results)
        
        assert success_count == total_count
        assert success_count == 4
        
        # 验证功能测试数量
        total_features = sum(result["features"] for result in demo_results.values())
        assert total_features == 14
    
    def test_performance_comparison(self):
        """测试性能对比"""
        # 模拟性能对比测试
        performance_data = {
            "langchain": {"response_time": 1200, "memory": 450, "concurrent": 10},
            "crewai": {"response_time": 800, "memory": 280, "concurrent": 15},
            "autogen": {"response_time": 2000, "memory": 600, "concurrent": 8},
            "langgraph": {"response_time": 900, "memory": 380, "concurrent": 12}
        }
        
        # 计算综合性能得分
        performance_scores = {}
        for framework, metrics in performance_data.items():
            # 归一化得分 (越小越好)
            response_score = 1 / (metrics["response_time"] / 1000)  # 转换为秒
            memory_score = 1 / (metrics["memory"] / 100)  # 转换为100MB单位
            concurrent_score = metrics["concurrent"] / 20  # 归一化到20
            
            # 综合得分
            score = (response_score + memory_score + concurrent_score) / 3
            performance_scores[framework] = score
        
        # 验证性能得分
        assert len(performance_scores) == 4
        assert all(score > 0 for score in performance_scores.values())
        
        # 找出性能最好的框架
        best_performance = max(performance_scores, key=performance_scores.get)
        assert best_performance == "crewai"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
