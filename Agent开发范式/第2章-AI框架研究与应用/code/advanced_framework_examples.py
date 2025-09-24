#!/usr/bin/env python3
"""
高级AI框架示例 - 展示各框架的高级功能和最佳实践

本模块提供了LangChain、CrewAI、AutoGen、LangGraph等框架的高级使用示例，
包括复杂场景的实现、性能优化、错误处理等。
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json
import time
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LangChain 高级示例
class LangChainAdvancedExamples:
    """LangChain高级功能示例"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_langchain()
    
    def setup_langchain(self):
        """设置LangChain环境"""
        try:
            from langchain.llms import OpenAI
            from langchain.chat_models import ChatOpenAI
            from langchain.agents import initialize_agent, AgentType
            from langchain.tools import Tool
            from langchain.memory import ConversationBufferMemory
            from langchain.chains import ConversationChain
            from langchain.prompts import PromptTemplate
            from langchain.vectorstores import Chroma
            from langchain.embeddings import OpenAIEmbeddings
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain.document_loaders import TextLoader
            
            self.llm = OpenAI(temperature=0.7)
            self.chat_llm = ChatOpenAI(temperature=0.7)
            self.embeddings = OpenAIEmbeddings()
            
        except ImportError as e:
            logger.error(f"Failed to import LangChain: {e}")
            raise
    
    def create_advanced_agent(self) -> Any:
        """创建高级智能体"""
        try:
            # 定义高级工具
            tools = [
                Tool(
                    name="Advanced Calculator",
                    func=self.advanced_calculator,
                    description="Advanced calculator with error handling and validation"
                ),
                Tool(
                    name="Data Analyzer",
                    func=self.data_analyzer,
                    description="Analyze data and generate insights"
                ),
                Tool(
                    name="Code Generator",
                    func=self.code_generator,
                    description="Generate code based on requirements"
                ),
                Tool(
                    name="Document Processor",
                    func=self.document_processor,
                    description="Process and analyze documents"
                )
            ]
            
            # 创建记忆
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            # 创建高级智能体
            agent = initialize_agent(
                tools=tools,
                llm=self.chat_llm,
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                memory=memory,
                verbose=True,
                max_iterations=10,
                early_stopping_method="generate"
            )
            
            return agent
            
        except Exception as e:
            logger.error(f"Failed to create advanced agent: {e}")
            raise
    
    def advanced_calculator(self, expression: str) -> str:
        """高级计算器"""
        try:
            # 安全的表达式评估
            allowed_names = {
                k: v for k, v in __builtins__.items()
                if k in ['abs', 'round', 'min', 'max', 'sum', 'pow']
            }
            allowed_names.update({
                'sin': lambda x: __import__('math').sin(x),
                'cos': lambda x: __import__('math').cos(x),
                'tan': lambda x: __import__('math').tan(x),
                'log': lambda x: __import__('math').log(x),
                'sqrt': lambda x: __import__('math').sqrt(x),
                'pi': __import__('math').pi,
                'e': __import__('math').e
            })
            
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return f"计算结果: {result}"
            
        except Exception as e:
            return f"计算错误: {str(e)}"
    
    def data_analyzer(self, data: str) -> str:
        """数据分析器"""
        try:
            import pandas as pd
            import numpy as np
            
            # 解析数据
            if data.startswith('[') and data.endswith(']'):
                data_list = json.loads(data)
                df = pd.DataFrame(data_list)
            else:
                # 假设是CSV格式
                from io import StringIO
                df = pd.read_csv(StringIO(data))
            
            # 生成分析报告
            analysis = {
                "数据形状": df.shape,
                "数据类型": df.dtypes.to_dict(),
                "缺失值": df.isnull().sum().to_dict(),
                "数值统计": df.describe().to_dict(),
                "相关性": df.corr().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 1 else "无数值列"
            }
            
            return json.dumps(analysis, ensure_ascii=False, indent=2)
            
        except Exception as e:
            return f"数据分析错误: {str(e)}"
    
    def code_generator(self, requirements: str) -> str:
        """代码生成器"""
        try:
            # 根据需求生成代码
            if "函数" in requirements or "function" in requirements.lower():
                code = f'''
def {requirements.split()[0] if requirements.split() else "example_function"}():
    """
    根据需求生成的函数
    需求: {requirements}
    """
    # TODO: 实现具体逻辑
    pass
'''
            elif "类" in requirements or "class" in requirements.lower():
                code = f'''
class {requirements.split()[0] if requirements.split() else "ExampleClass"}:
    """
    根据需求生成的类
    需求: {requirements}
    """
    def __init__(self):
        pass
    
    def example_method(self):
        # TODO: 实现具体逻辑
        pass
'''
            else:
                code = f'''
# 根据需求生成的代码
# 需求: {requirements}

# TODO: 实现具体逻辑
'''
            
            return code
            
        except Exception as e:
            return f"代码生成错误: {str(e)}"
    
    def document_processor(self, document: str) -> str:
        """文档处理器"""
        try:
            # 简单的文档分析
            words = document.split()
            sentences = document.split('.')
            
            analysis = {
                "字符数": len(document),
                "单词数": len(words),
                "句子数": len(sentences),
                "平均单词长度": sum(len(word) for word in words) / len(words) if words else 0,
                "关键词": list(set(word.lower() for word in words if len(word) > 3))[:10]
            }
            
            return json.dumps(analysis, ensure_ascii=False, indent=2)
            
        except Exception as e:
            return f"文档处理错误: {str(e)}"
    
    def create_rag_system(self, documents: List[str]) -> Any:
        """创建RAG系统"""
        try:
            # 加载文档
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            # 创建文档
            docs = []
            for i, doc in enumerate(documents):
                docs.append({
                    "page_content": doc,
                    "metadata": {"source": f"doc_{i}"}
                })
            
            # 创建向量存储
            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings
            )
            
            # 创建检索器
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            
            return retriever
            
        except Exception as e:
            logger.error(f"Failed to create RAG system: {e}")
            raise

# CrewAI 高级示例
class CrewAIAdvancedExamples:
    """CrewAI高级功能示例"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_crewai()
    
    def setup_crewai(self):
        """设置CrewAI环境"""
        try:
            from crewai import Agent, Task, Crew, Process
            from crewai.tools import BaseTool
            from crewai.memory import ShortTermMemory
            
            self.Process = Process
            self.ShortTermMemory = ShortTermMemory
            
        except ImportError as e:
            logger.error(f"Failed to import CrewAI: {e}")
            raise
    
    def create_research_crew(self) -> Any:
        """创建研究团队"""
        try:
            from crewai import Agent, Task, Crew
            
            # 创建研究分析师
            researcher = Agent(
                role='高级研究分析师',
                goal='进行深度研究并生成高质量的分析报告',
                backstory='你是一位经验丰富的研究分析师，擅长数据收集、分析和报告撰写。',
                verbose=True,
                memory=True,
                tools=[self.create_research_tool()]
            )
            
            # 创建数据分析师
            analyst = Agent(
                role='数据分析师',
                goal='分析数据并提供洞察',
                backstory='你是一位专业的数据分析师，擅长统计分析和数据可视化。',
                verbose=True,
                memory=True,
                tools=[self.create_analysis_tool()]
            )
            
            # 创建报告撰写员
            writer = Agent(
                role='技术写作专家',
                goal='将分析结果转化为清晰、专业的报告',
                backstory='你是一位技术写作专家，擅长将复杂的技术内容转化为易懂的报告。',
                verbose=True,
                memory=True,
                tools=[self.create_writing_tool()]
            )
            
            # 创建任务
            research_task = Task(
                description='研究AI智能体的最新发展趋势，包括技术突破、市场动态和未来预测',
                agent=researcher,
                expected_output='详细的研究报告，包含数据、分析和结论'
            )
            
            analysis_task = Task(
                description='分析研究数据，识别关键趋势和模式',
                agent=analyst,
                expected_output='数据分析报告，包含统计结果和可视化图表'
            )
            
            writing_task = Task(
                description='将研究和分析结果整合成最终报告',
                agent=writer,
                expected_output='专业的最终报告，包含执行摘要、详细分析和建议'
            )
            
            # 创建团队
            crew = Crew(
                agents=[researcher, analyst, writer],
                tasks=[research_task, analysis_task, writing_task],
                process=Process.sequential,
                verbose=True,
                memory=True
            )
            
            return crew
            
        except Exception as e:
            logger.error(f"Failed to create research crew: {e}")
            raise
    
    def create_research_tool(self) -> Any:
        """创建研究工具"""
        class ResearchTool:
            def __init__(self):
                self.name = "research_tool"
                self.description = "进行在线研究和数据收集"
            
            def run(self, query: str) -> str:
                # 模拟研究功能
                return f"研究结果: {query}的相关信息已收集"
        
        return ResearchTool()
    
    def create_analysis_tool(self) -> Any:
        """创建分析工具"""
        class AnalysisTool:
            def __init__(self):
                self.name = "analysis_tool"
                self.description = "分析数据和生成洞察"
            
            def run(self, data: str) -> str:
                # 模拟分析功能
                return f"分析结果: 基于{data}的分析报告已生成"
        
        return AnalysisTool()
    
    def create_writing_tool(self) -> Any:
        """创建写作工具"""
        class WritingTool:
            def __init__(self):
                self.name = "writing_tool"
                self.description = "生成专业报告和文档"
            
            def run(self, content: str) -> str:
                # 模拟写作功能
                return f"写作结果: 基于{content}的专业报告已生成"
        
        return WritingTool()

# AutoGen 高级示例
class AutoGenAdvancedExamples:
    """AutoGen高级功能示例"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_autogen()
    
    def setup_autogen(self):
        """设置AutoGen环境"""
        try:
            import autogen
            from autogen import ConversableAgent, GroupChat, GroupChatManager
            
            self.autogen = autogen
            self.ConversableAgent = ConversableAgent
            self.GroupChat = GroupChat
            self.GroupChatManager = GroupChatManager
            
        except ImportError as e:
            logger.error(f"Failed to import AutoGen: {e}")
            raise
    
    def create_multi_agent_system(self) -> Any:
        """创建多智能体系统"""
        try:
            # 配置LLM
            config_list = [
                {
                    "model": "gpt-3.5-turbo",
                    "api_key": self.config.get("openai_api_key", "your-api-key")
                }
            ]
            
            llm_config = {
                "config_list": config_list,
                "temperature": 0.7,
            }
            
            # 创建智能体
            coder = self.ConversableAgent(
                name="Coder",
                system_message="你是一位专业的软件工程师，擅长编写高质量的代码。",
                llm_config=llm_config,
            )
            
            reviewer = self.ConversableAgent(
                name="Reviewer",
                system_message="你是一位代码审查专家，擅长发现代码中的问题和改进建议。",
                llm_config=llm_config,
            )
            
            tester = self.ConversableAgent(
                name="Tester",
                system_message="你是一位测试工程师，擅长编写测试用例和进行测试。",
                llm_config=llm_config,
            )
            
            # 创建群聊
            groupchat = self.GroupChat(
                agents=[coder, reviewer, tester],
                messages=[],
                max_round=10
            )
            
            manager = self.GroupChatManager(
                groupchat=groupchat,
                llm_config=llm_config
            )
            
            return manager
            
        except Exception as e:
            logger.error(f"Failed to create multi-agent system: {e}")
            raise
    
    def create_agent_with_tools(self) -> Any:
        """创建带工具的智能体"""
        try:
            # 定义工具
            def calculator(operation: str) -> str:
                try:
                    result = eval(operation)
                    return f"计算结果: {result}"
                except Exception as e:
                    return f"计算错误: {str(e)}"
            
            def weather_query(city: str) -> str:
                # 模拟天气查询
                return f"{city}的天气: 晴天，温度25°C"
            
            # 创建智能体
            agent = self.ConversableAgent(
                name="Assistant",
                system_message="你是一位有用的助手，可以使用工具来帮助用户。",
                llm_config={
                    "config_list": [{"model": "gpt-3.5-turbo", "api_key": "your-api-key"}],
                    "temperature": 0.7,
                },
                function_map={
                    "calculator": calculator,
                    "weather_query": weather_query
                }
            )
            
            return agent
            
        except Exception as e:
            logger.error(f"Failed to create agent with tools: {e}")
            raise

# LangGraph 高级示例
class LangGraphAdvancedExamples:
    """LangGraph高级功能示例"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_langgraph()
    
    def setup_langgraph(self):
        """设置LangGraph环境"""
        try:
            from langgraph.graph import StateGraph, END
            from langgraph.prebuilt import ToolNode
            from langchain.tools import tool
            
            self.StateGraph = StateGraph
            self.END = END
            self.ToolNode = ToolNode
            self.tool = tool
            
        except ImportError as e:
            logger.error(f"Failed to import LangGraph: {e}")
            raise
    
    def create_workflow_graph(self) -> Any:
        """创建工作流图"""
        try:
            from typing import TypedDict, Annotated
            from langchain_core.messages import BaseMessage
            
            # 定义状态
            class WorkflowState(TypedDict):
                messages: Annotated[list[BaseMessage], "消息列表"]
                current_step: str
                results: dict
                error: Optional[str]
            
            # 定义工具
            @self.tool
            def process_data(data: str) -> str:
                """处理数据"""
                return f"处理结果: {data.upper()}"
            
            @self.tool
            def validate_data(data: str) -> str:
                """验证数据"""
                if len(data) > 0:
                    return "数据验证通过"
                else:
                    return "数据验证失败"
            
            @self.tool
            def save_data(data: str) -> str:
                """保存数据"""
                return f"数据已保存: {data}"
            
            # 定义节点函数
            def start_node(state: WorkflowState) -> WorkflowState:
                """开始节点"""
                state["current_step"] = "start"
                state["results"] = {"status": "started"}
                return state
            
            def process_node(state: WorkflowState) -> WorkflowState:
                """处理节点"""
                state["current_step"] = "processing"
                state["results"]["processed"] = True
                return state
            
            def validate_node(state: WorkflowState) -> WorkflowState:
                """验证节点"""
                state["current_step"] = "validating"
                state["results"]["validated"] = True
                return state
            
            def save_node(state: WorkflowState) -> WorkflowState:
                """保存节点"""
                state["current_step"] = "saving"
                state["results"]["saved"] = True
                return state
            
            def error_node(state: WorkflowState) -> WorkflowState:
                """错误节点"""
                state["current_step"] = "error"
                state["error"] = "处理过程中发生错误"
                return state
            
            # 创建图
            workflow = self.StateGraph(WorkflowState)
            
            # 添加节点
            workflow.add_node("start", start_node)
            workflow.add_node("process", process_node)
            workflow.add_node("validate", validate_node)
            workflow.add_node("save", save_node)
            workflow.add_node("error", error_node)
            
            # 添加边
            workflow.add_edge("start", "process")
            workflow.add_edge("process", "validate")
            workflow.add_edge("validate", "save")
            workflow.add_edge("save", self.END)
            
            # 添加条件边
            workflow.add_conditional_edges(
                "process",
                lambda state: "error" if state.get("error") else "validate"
            )
            
            # 编译图
            app = workflow.compile()
            
            return app
            
        except Exception as e:
            logger.error(f"Failed to create workflow graph: {e}")
            raise

# 框架对比分析器
class FrameworkComparisonAnalyzer:
    """框架对比分析器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.frameworks = {
            "langchain": LangChainAdvancedExamples(config),
            "crewai": CrewAIAdvancedExamples(config),
            "autogen": AutoGenAdvancedExamples(config),
            "langgraph": LangGraphAdvancedExamples(config)
        }
    
    def compare_frameworks(self) -> Dict[str, Any]:
        """对比框架"""
        comparison = {
            "langchain": {
                "优点": [
                    "生态系统丰富",
                    "文档完善",
                    "社区活跃",
                    "易于上手"
                ],
                "缺点": [
                    "性能相对较低",
                    "内存占用较大",
                    "复杂场景支持有限"
                ],
                "适用场景": [
                    "快速原型开发",
                    "简单应用",
                    "学习和实验"
                ],
                "性能评分": 7.5
            },
            "crewai": {
                "优点": [
                    "多智能体协作",
                    "任务分解清晰",
                    "易于扩展",
                    "支持复杂工作流"
                ],
                "缺点": [
                    "学习曲线较陡",
                    "文档相对较少",
                    "社区较小"
                ],
                "适用场景": [
                    "多智能体系统",
                    "复杂任务分解",
                    "团队协作"
                ],
                "性能评分": 8.0
            },
            "autogen": {
                "优点": [
                    "对话能力强",
                    "支持多轮对话",
                    "工具集成简单",
                    "微软支持"
                ],
                "缺点": [
                    "配置复杂",
                    "调试困难",
                    "性能开销大"
                ],
                "适用场景": [
                    "对话系统",
                    "客服应用",
                    "多轮交互"
                ],
                "性能评分": 7.0
            },
            "langgraph": {
                "优点": [
                    "状态管理强大",
                    "工作流清晰",
                    "易于调试",
                    "性能优秀"
                ],
                "缺点": [
                    "相对较新",
                    "文档有限",
                    "学习成本高"
                ],
                "适用场景": [
                    "复杂工作流",
                    "状态机应用",
                    "高性能需求"
                ],
                "性能评分": 8.5
            }
        }
        
        return comparison
    
    def generate_recommendation(self, requirements: Dict[str, Any]) -> str:
        """生成框架推荐"""
        try:
            # 分析需求
            complexity = requirements.get("complexity", "medium")
            performance = requirements.get("performance", "medium")
            team_size = requirements.get("team_size", "small")
            timeline = requirements.get("timeline", "medium")
            
            recommendations = []
            
            if complexity == "high" and performance == "high":
                recommendations.append("推荐使用LangGraph，适合复杂工作流和高性能需求")
            
            if team_size == "large" and complexity == "high":
                recommendations.append("推荐使用CrewAI，适合大型团队和复杂任务分解")
            
            if timeline == "short" and complexity == "low":
                recommendations.append("推荐使用LangChain，适合快速开发和简单应用")
            
            if requirements.get("conversation", False):
                recommendations.append("推荐使用AutoGen，适合对话系统和多轮交互")
            
            return "\n".join(recommendations) if recommendations else "建议根据具体需求选择合适的框架"
            
        except Exception as e:
            logger.error(f"Failed to generate recommendation: {e}")
            return "推荐生成失败"

# 示例使用
async def main():
    """主函数示例"""
    config = {
        "openai_api_key": "your-api-key-here",
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    # 创建框架示例
    langchain_examples = LangChainAdvancedExamples(config)
    crewai_examples = CrewAIAdvancedExamples(config)
    autogen_examples = AutoGenAdvancedExamples(config)
    langgraph_examples = LangGraphAdvancedExamples(config)
    
    # 创建对比分析器
    analyzer = FrameworkComparisonAnalyzer(config)
    
    # 进行框架对比
    comparison = analyzer.compare_frameworks()
    print("框架对比结果:")
    for framework, details in comparison.items():
        print(f"\n{framework.upper()}:")
        print(f"  性能评分: {details['性能评分']}")
        print(f"  优点: {', '.join(details['优点'])}")
        print(f"  缺点: {', '.join(details['缺点'])}")
        print(f"  适用场景: {', '.join(details['适用场景'])}")
    
    # 生成推荐
    requirements = {
        "complexity": "high",
        "performance": "high",
        "team_size": "medium",
        "timeline": "medium",
        "conversation": False
    }
    
    recommendation = analyzer.generate_recommendation(requirements)
    print(f"\n推荐结果: {recommendation}")

if __name__ == "__main__":
    asyncio.run(main())
