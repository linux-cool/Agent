# framework_demo.py
"""
第2章 AI框架研究与应用 - 完整演示程序
展示各框架的核心功能和使用方法
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# 导入各框架示例
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.langchain_examples import LangChainExamples
from code.crewai_examples import CrewAIExamples
from code.autogen_examples import AutoGenExamples
from code.langgraph_examples import LangGraphExamples
from code.framework_comparison import FrameworkComparison

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FrameworkDemo:
    """框架演示类"""
    
    def __init__(self):
        self.frameworks = {}
        self.demo_results = {}
        self.comparator = FrameworkComparison()
    
    async def run_complete_demo(self):
        """运行完整演示"""
        logger.info("🚀 开始第2章AI框架研究与应用演示")
        
        # 1. LangChain演示
        await self.demo_langchain()
        
        # 2. CrewAI演示
        await self.demo_crewai()
        
        # 3. AutoGen演示
        await self.demo_autogen()
        
        # 4. LangGraph演示
        await self.demo_langgraph()
        
        # 5. 框架对比分析
        await self.demo_framework_comparison()
        
        # 6. 性能基准测试
        await self.demo_performance_benchmark()
        
        # 7. 集成示例
        await self.demo_integration_example()
        
        # 8. 生成演示报告
        await self.generate_demo_report()
        
        logger.info("✅ 第2章演示完成")
    
    async def demo_langchain(self):
        """LangChain演示"""
        logger.info("📋 演示1: LangChain框架")
        
        try:
            examples = LangChainExamples()
            
            # 运行基础示例
            basic_result = examples.basic_llm_example()
            
            # 运行智能体示例
            agent_result = examples.agent_with_tools_example()
            
            # 运行向量存储示例
            vector_result = examples.vector_store_example()
            
            self.demo_results["langchain"] = {
                "status": "success",
                "basic_llm": basic_result[:100] + "..." if len(basic_result) > 100 else basic_result,
                "agent_tools": agent_result[:100] + "..." if len(agent_result) > 100 else agent_result,
                "vector_store": vector_result[:100] + "..." if len(vector_result) > 100 else vector_result,
                "features": [
                    "丰富的工具生态",
                    "强大的链式调用",
                    "多种记忆类型",
                    "向量数据库集成"
                ]
            }
            
            logger.info("✅ LangChain演示完成")
            
        except Exception as e:
            logger.error(f"LangChain演示失败: {e}")
            self.demo_results["langchain"] = {
                "status": "error",
                "error": str(e)
            }
    
    async def demo_crewai(self):
        """CrewAI演示"""
        logger.info("📋 演示2: CrewAI框架")
        
        try:
            examples = CrewAIExamples()
            
            # 运行基础团队示例
            basic_crew_result = examples.basic_crew_example()
            
            # 运行层次化团队示例
            hierarchical_result = examples.hierarchical_crew_example()
            
            # 运行并行团队示例
            parallel_result = examples.parallel_crew_example()
            
            self.demo_results["crewai"] = {
                "status": "success",
                "basic_crew": basic_crew_result[:100] + "..." if len(basic_crew_result) > 100 else basic_crew_result,
                "hierarchical": hierarchical_result[:100] + "..." if len(hierarchical_result) > 100 else hierarchical_result,
                "parallel": parallel_result[:100] + "..." if len(parallel_result) > 100 else parallel_result,
                "features": [
                    "专业多智能体协作",
                    "角色分工明确",
                    "任务依赖管理",
                    "团队协作机制"
                ]
            }
            
            logger.info("✅ CrewAI演示完成")
            
        except Exception as e:
            logger.error(f"CrewAI演示失败: {e}")
            self.demo_results["crewai"] = {
                "status": "error",
                "error": str(e)
            }
    
    async def demo_autogen(self):
        """AutoGen演示"""
        logger.info("📋 演示3: AutoGen框架")
        
        try:
            examples = AutoGenExamples()
            
            # 运行基础对话示例
            basic_conversation_result = examples.basic_conversation_example()
            
            # 运行群组对话示例
            group_chat_result = examples.group_chat_example()
            
            # 运行代码执行示例
            code_execution_result = examples.code_execution_example()
            
            self.demo_results["autogen"] = {
                "status": "success",
                "basic_conversation": basic_conversation_result[:100] + "..." if len(basic_conversation_result) > 100 else basic_conversation_result,
                "group_chat": group_chat_result[:100] + "..." if len(group_chat_result) > 100 else group_chat_result,
                "code_execution": code_execution_result[:100] + "..." if len(code_execution_result) > 100 else code_execution_result,
                "features": [
                    "对话式多智能体协作",
                    "代码执行能力",
                    "多轮对话支持",
                    "动态协作调整"
                ]
            }
            
            logger.info("✅ AutoGen演示完成")
            
        except Exception as e:
            logger.error(f"AutoGen演示失败: {e}")
            self.demo_results["autogen"] = {
                "status": "error",
                "error": str(e)
            }
    
    async def demo_langgraph(self):
        """LangGraph演示"""
        logger.info("📋 演示4: LangGraph框架")
        
        try:
            examples = LangGraphExamples()
            
            # 运行基础工作流示例
            basic_workflow_result = examples.basic_workflow_example()
            
            # 运行条件工作流示例
            conditional_result = examples.conditional_workflow_example()
            
            # 运行循环工作流示例
            loop_result = examples.loop_workflow_example()
            
            self.demo_results["langgraph"] = {
                "status": "success",
                "basic_workflow": basic_workflow_result[:100] + "..." if len(basic_workflow_result) > 100 else basic_workflow_result,
                "conditional": conditional_result[:100] + "..." if len(conditional_result) > 100 else conditional_result,
                "loop": loop_result[:100] + "..." if len(loop_result) > 100 else loop_result,
                "features": [
                    "状态机工作流",
                    "条件分支处理",
                    "循环控制",
                    "可视化工作流"
                ]
            }
            
            logger.info("✅ LangGraph演示完成")
            
        except Exception as e:
            logger.error(f"LangGraph演示失败: {e}")
            self.demo_results["langgraph"] = {
                "status": "error",
                "error": str(e)
            }
    
    async def demo_framework_comparison(self):
        """框架对比演示"""
        logger.info("📋 演示5: 框架对比分析")
        
        try:
            # 对比主要框架
            framework_names = ["langchain", "crewai", "autogen", "langgraph"]
            comparison_data = self.comparator.compare_frameworks(framework_names)
            
            # 生成对比报告
            report = self.comparator.generate_comparison_report(framework_names)
            
            # 分析趋势
            trends = self.comparator.analyze_trends()
            
            self.demo_results["framework_comparison"] = {
                "status": "success",
                "frameworks_compared": len(comparison_data),
                "report_length": len(report),
                "trends_analyzed": len(trends["emerging_trends"]),
                "comparison_summary": {
                    "langchain": {
                        "github_stars": comparison_data["langchain"]["github_stars"],
                        "learning_curve": comparison_data["langchain"]["learning_curve"],
                        "tool_ecosystem": comparison_data["langchain"]["tool_ecosystem"]
                    },
                    "crewai": {
                        "github_stars": comparison_data["crewai"]["github_stars"],
                        "learning_curve": comparison_data["crewai"]["learning_curve"],
                        "multi_agent_support": comparison_data["crewai"]["multi_agent_support"]
                    },
                    "autogen": {
                        "github_stars": comparison_data["autogen"]["github_stars"],
                        "learning_curve": comparison_data["autogen"]["learning_curve"],
                        "multi_agent_support": comparison_data["autogen"]["multi_agent_support"]
                    },
                    "langgraph": {
                        "github_stars": comparison_data["langgraph"]["github_stars"],
                        "learning_curve": comparison_data["langgraph"]["learning_curve"],
                        "workflow_management": comparison_data["langgraph"]["workflow_management"]
                    }
                }
            }
            
            logger.info("✅ 框架对比演示完成")
            
        except Exception as e:
            logger.error(f"框架对比演示失败: {e}")
            self.demo_results["framework_comparison"] = {
                "status": "error",
                "error": str(e)
            }
    
    async def demo_performance_benchmark(self):
        """性能基准测试演示"""
        logger.info("📋 演示6: 性能基准测试")
        
        try:
            # 定义测试用例
            test_cases = ["basic_task", "complex_workflow", "multi_agent_collaboration"]
            framework_names = ["langchain", "crewai", "autogen", "langgraph"]
            
            # 运行基准测试
            benchmark_results = self.comparator.benchmark_frameworks(framework_names, test_cases)
            
            # 分析性能数据
            performance_summary = {}
            for framework_name, results in benchmark_results.items():
                performance_summary[framework_name] = {
                    "overall_score": results["overall_score"],
                    "test_count": len(results["test_results"]),
                    "average_execution_time": sum(
                        test["execution_time"] for test in results["test_results"].values()
                    ) / len(results["test_results"])
                }
            
            self.demo_results["performance_benchmark"] = {
                "status": "success",
                "frameworks_tested": len(benchmark_results),
                "test_cases": len(test_cases),
                "performance_summary": performance_summary,
                "ranking": sorted(
                    performance_summary.items(),
                    key=lambda x: x[1]["overall_score"],
                    reverse=True
                )
            }
            
            logger.info("✅ 性能基准测试演示完成")
            
        except Exception as e:
            logger.error(f"性能基准测试演示失败: {e}")
            self.demo_results["performance_benchmark"] = {
                "status": "error",
                "error": str(e)
            }
    
    async def demo_integration_example(self):
        """集成示例演示"""
        logger.info("📋 演示7: 多框架集成示例")
        
        try:
            # 模拟多框架集成场景
            integration_scenario = {
                "name": "智能客服系统",
                "description": "使用多个框架构建的智能客服系统",
                "components": {
                    "langchain": "处理用户查询和工具调用",
                    "crewai": "多智能体协作处理复杂问题",
                    "autogen": "对话管理和上下文保持",
                    "langgraph": "工作流管理和状态控制"
                },
                "workflow": [
                    "用户输入处理 (LangChain)",
                    "问题分类和路由 (CrewAI)",
                    "对话管理 (AutoGen)",
                    "工作流控制 (LangGraph)"
                ]
            }
            
            # 模拟集成测试
            integration_results = {
                "components_integrated": len(integration_scenario["components"]),
                "workflow_steps": len(integration_scenario["workflow"]),
                "integration_success": True,
                "performance_metrics": {
                    "response_time": 1.2,  # seconds
                    "throughput": 50,  # requests per minute
                    "error_rate": 0.01,  # 1%
                    "availability": 0.999  # 99.9%
                }
            }
            
            self.demo_results["integration_example"] = {
                "status": "success",
                "scenario": integration_scenario,
                "results": integration_results,
                "best_practices": [
                    "使用统一的消息格式",
                    "实施错误处理和重试机制",
                    "监控各组件性能",
                    "实现优雅降级"
                ]
            }
            
            logger.info("✅ 集成示例演示完成")
            
        except Exception as e:
            logger.error(f"集成示例演示失败: {e}")
            self.demo_results["integration_example"] = {
                "status": "error",
                "error": str(e)
            }
    
    async def generate_demo_report(self):
        """生成演示报告"""
        logger.info("📋 生成演示报告")
        
        report = {
            "demo_info": {
                "title": "第2章 AI框架研究与应用演示报告",
                "timestamp": datetime.now().isoformat(),
                "total_demos": len(self.demo_results)
            },
            "results": self.demo_results,
            "summary": {
                "frameworks_demoed": len([r for r in self.demo_results.values() if r["status"] == "success"]),
                "total_features_tested": sum(
                    len(r.get("features", [])) for r in self.demo_results.values() 
                    if r["status"] == "success" and "features" in r
                ),
                "performance_benchmarks": len(
                    self.demo_results.get("performance_benchmark", {}).get("performance_summary", {})
                ),
                "integration_scenarios": 1
            },
            "recommendations": {
                "general_ai_app": "LangChain - 生态最丰富",
                "multi_agent_collaboration": "CrewAI - 专业多智能体框架",
                "workflow_automation": "LangGraph - 强大的状态机支持",
                "conversational_ai": "AutoGen - 对话式协作"
            }
        }
        
        # 保存报告到文件
        report_filename = f"chapter2_demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📄 演示报告已保存到: {report_filename}")
        
        # 打印摘要
        print("\n" + "="*60)
        print("📊 第2章演示报告摘要")
        print("="*60)
        print(f"演示时间: {report['demo_info']['timestamp']}")
        print(f"演示项目数: {report['demo_info']['total_demos']}")
        print(f"框架演示数: {report['summary']['frameworks_demoed']}")
        print(f"功能测试数: {report['summary']['total_features_tested']}")
        print(f"性能基准数: {report['summary']['performance_benchmarks']}")
        print(f"集成场景数: {report['summary']['integration_scenarios']}")
        
        print("\n🎯 框架推荐:")
        for use_case, recommendation in report['recommendations'].items():
            print(f"- {use_case}: {recommendation}")
        
        print("="*60)
        
        return report

async def main():
    """主函数"""
    print("🤖 第2章 AI框架研究与应用 - 完整演示")
    print("="*60)
    
    demo = FrameworkDemo()
    await demo.run_complete_demo()
    
    print("\n🎉 演示完成！请查看生成的报告文件了解详细结果。")

if __name__ == "__main__":
    asyncio.run(main())
