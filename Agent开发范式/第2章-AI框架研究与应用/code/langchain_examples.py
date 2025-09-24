# langchain_examples.py
"""
LangChain框架示例代码
展示LangChain的核心功能和使用方法
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# LangChain核心组件
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.agents.agent import AgentExecutor
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.chains import LLMChain, ConversationChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.tools import BaseTool, DuckDuckGoSearchRun
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.callbacks import StreamingStdOutCallbackHandler

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LangChainExamples:
    """LangChain示例集合"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OpenAI API key not found. Some examples may not work.")
        
        self.llm = ChatOpenAI(
            openai_api_key=self.api_key,
            temperature=0.7,
            model_name="gpt-3.5-turbo"
        )
        
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
    
    def basic_llm_example(self) -> str:
        """基础LLM使用示例"""
        logger.info("Running basic LLM example...")
        
        messages = [
            SystemMessage(content="You are a helpful AI assistant."),
            HumanMessage(content="What is the capital of France?")
        ]
        
        response = self.llm(messages)
        return response.content
    
    def prompt_template_example(self) -> str:
        """提示模板示例"""
        logger.info("Running prompt template example...")
        
        template = """
        You are a {role} with expertise in {domain}.
        
        Question: {question}
        
        Please provide a detailed answer:
        """
        
        prompt = PromptTemplate(
            input_variables=["role", "domain", "question"],
            template=template
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        result = chain.run(
            role="Data Scientist",
            domain="Machine Learning",
            question="What is the difference between supervised and unsupervised learning?"
        )
        
        return result
    
    def conversation_chain_example(self) -> str:
        """对话链示例"""
        logger.info("Running conversation chain example...")
        
        # 创建记忆
        memory = ConversationBufferMemory()
        
        # 创建对话链
        conversation = ConversationChain(
            llm=self.llm,
            memory=memory,
            verbose=True
        )
        
        # 进行对话
        response1 = conversation.predict(input="Hi, my name is Alice.")
        response2 = conversation.predict(input="What's my name?")
        
        return f"Response 1: {response1}\nResponse 2: {response2}"
    
    def conversation_summary_example(self) -> str:
        """对话摘要示例"""
        logger.info("Running conversation summary example...")
        
        # 创建摘要记忆
        memory = ConversationSummaryMemory(
            llm=self.llm,
            return_messages=True
        )
        
        # 创建对话链
        conversation = ConversationChain(
            llm=self.llm,
            memory=memory,
            verbose=True
        )
        
        # 进行长对话
        responses = []
        topics = [
            "Tell me about artificial intelligence",
            "What are the main applications of AI?",
            "How does machine learning work?",
            "What is deep learning?"
        ]
        
        for topic in topics:
            response = conversation.predict(input=topic)
            responses.append(response)
        
        # 获取摘要
        summary = memory.moving_summary_buffer
        
        return f"Summary: {summary}\n\nResponses: {responses}"
    
    def agent_with_tools_example(self) -> str:
        """带工具的智能体示例"""
        logger.info("Running agent with tools example...")
        
        # 定义工具
        def calculator(operation: str) -> str:
            """计算器工具"""
            try:
                result = eval(operation)
                return str(result)
            except Exception as e:
                return f"Error: {e}"
        
        def get_current_time() -> str:
            """获取当前时间工具"""
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        def word_count(text: str) -> str:
            """字数统计工具"""
            return f"Word count: {len(text.split())}"
        
        # 创建工具列表
        tools = [
            Tool(
                name="Calculator",
                func=calculator,
                description="Useful for mathematical calculations. Input should be a mathematical expression."
            ),
            Tool(
                name="CurrentTime",
                func=get_current_time,
                description="Useful for getting the current date and time."
            ),
            Tool(
                name="WordCount",
                func=word_count,
                description="Useful for counting words in a text."
            )
        ]
        
        # 创建智能体
        agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )
        
        # 执行任务
        result = agent.run("Calculate 15 * 8, then tell me the current time, and count the words in 'Hello world'")
        
        return result
    
    def search_agent_example(self) -> str:
        """搜索智能体示例"""
        logger.info("Running search agent example...")
        
        # 创建搜索工具
        search = DuckDuckGoSearchRun()
        
        tools = [
            Tool(
                name="Search",
                func=search.run,
                description="Useful for searching the web for current information."
            )
        ]
        
        # 创建智能体
        agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
        # 执行搜索任务
        result = agent.run("Search for the latest news about artificial intelligence")
        
        return result
    
    def vector_store_example(self) -> str:
        """向量存储示例"""
        logger.info("Running vector store example...")
        
        # 创建示例文档
        documents = [
            "Artificial intelligence is the simulation of human intelligence in machines.",
            "Machine learning is a subset of AI that focuses on algorithms.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing deals with human language understanding.",
            "Computer vision enables machines to interpret visual information."
        ]
        
        # 创建向量存储
        vectorstore = Chroma.from_texts(
            texts=documents,
            embedding=self.embeddings
        )
        
        # 相似性搜索
        query = "What is machine learning?"
        docs = vectorstore.similarity_search(query, k=2)
        
        result = f"Query: {query}\n\nSimilar documents:\n"
        for i, doc in enumerate(docs):
            result += f"{i+1}. {doc.page_content}\n"
        
        return result
    
    def streaming_example(self) -> str:
        """流式输出示例"""
        logger.info("Running streaming example...")
        
        # 创建流式回调
        streaming_callback = StreamingStdOutCallbackHandler()
        
        # 创建流式LLM
        streaming_llm = ChatOpenAI(
            openai_api_key=self.api_key,
            temperature=0.7,
            streaming=True,
            callbacks=[streaming_callback]
        )
        
        # 流式生成
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Write a short story about a robot.")
        ]
        
        response = streaming_llm(messages)
        
        return response.content
    
    def custom_tool_example(self) -> str:
        """自定义工具示例"""
        logger.info("Running custom tool example...")
        
        class WeatherTool(BaseTool):
            """天气查询工具"""
            name = "weather"
            description = "Get weather information for a city"
            
            def _run(self, city: str) -> str:
                # 模拟天气API调用
                weather_data = {
                    "Beijing": "Sunny, 25°C",
                    "Shanghai": "Cloudy, 22°C",
                    "Guangzhou": "Rainy, 28°C",
                    "Shenzhen": "Sunny, 30°C"
                }
                return weather_data.get(city, f"Weather data not available for {city}")
            
            async def _arun(self, city: str) -> str:
                return self._run(city)
        
        # 创建自定义工具
        weather_tool = WeatherTool()
        
        # 创建智能体
        agent = initialize_agent(
            tools=[weather_tool],
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
        # 执行任务
        result = agent.run("What's the weather like in Beijing?")
        
        return result
    
    def chain_composition_example(self) -> str:
        """链组合示例"""
        logger.info("Running chain composition example...")
        
        # 创建多个链
        template1 = "Summarize this text: {text}"
        prompt1 = PromptTemplate(input_variables=["text"], template=template1)
        chain1 = LLMChain(llm=self.llm, prompt=prompt1)
        
        template2 = "Translate this to Chinese: {text}"
        prompt2 = PromptTemplate(input_variables=["text"], template=template2)
        chain2 = LLMChain(llm=self.llm, prompt=prompt2)
        
        # 组合链
        from langchain.chains import SimpleSequentialChain
        overall_chain = SimpleSequentialChain(
            chains=[chain1, chain2],
            verbose=True
        )
        
        # 执行组合链
        text = "Artificial intelligence is transforming the world. It has applications in healthcare, finance, transportation, and many other fields."
        result = overall_chain.run(text)
        
        return result
    
    def async_example(self) -> str:
        """异步处理示例"""
        logger.info("Running async example...")
        
        async def async_llm_call():
            messages = [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="Tell me a joke.")
            ]
            
            response = await self.llm.agenerate([messages])
            return response.generations[0][0].text
        
        # 运行异步任务
        result = asyncio.run(async_llm_call())
        
        return result
    
    def error_handling_example(self) -> str:
        """错误处理示例"""
        logger.info("Running error handling example...")
        
        try:
            # 模拟可能出错的操作
            messages = [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="This is a test message.")
            ]
            
            response = self.llm(messages)
            return f"Success: {response.content}"
            
        except Exception as e:
            logger.error(f"Error occurred: {e}")
            return f"Error handled: {str(e)}"
    
    def performance_optimization_example(self) -> str:
        """性能优化示例"""
        logger.info("Running performance optimization example...")
        
        import time
        
        # 测试不同配置的性能
        configs = [
            {"temperature": 0.1, "max_tokens": 100},
            {"temperature": 0.7, "max_tokens": 200},
            {"temperature": 1.0, "max_tokens": 300}
        ]
        
        results = []
        for config in configs:
            start_time = time.time()
            
            llm = ChatOpenAI(
                openai_api_key=self.api_key,
                **config
            )
            
            messages = [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="Explain machine learning in one sentence.")
            ]
            
            response = llm(messages)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            results.append({
                "config": config,
                "execution_time": execution_time,
                "response_length": len(response.content)
            })
        
        return f"Performance test results: {results}"

def run_all_examples():
    """运行所有示例"""
    print("🚀 LangChain框架示例演示")
    print("=" * 50)
    
    # 创建示例实例
    examples = LangChainExamples()
    
    # 定义示例列表
    example_methods = [
        ("基础LLM使用", examples.basic_llm_example),
        ("提示模板", examples.prompt_template_example),
        ("对话链", examples.conversation_chain_example),
        ("对话摘要", examples.conversation_summary_example),
        ("带工具的智能体", examples.agent_with_tools_example),
        ("搜索智能体", examples.search_agent_example),
        ("向量存储", examples.vector_store_example),
        ("流式输出", examples.streaming_example),
        ("自定义工具", examples.custom_tool_example),
        ("链组合", examples.chain_composition_example),
        ("异步处理", examples.async_example),
        ("错误处理", examples.error_handling_example),
        ("性能优化", examples.performance_optimization_example)
    ]
    
    results = {}
    
    for name, method in example_methods:
        try:
            print(f"\n📋 运行示例: {name}")
            print("-" * 30)
            
            result = method()
            results[name] = {
                "status": "success",
                "result": result[:200] + "..." if len(result) > 200 else result
            }
            
            print(f"✅ {name} 完成")
            
        except Exception as e:
            print(f"❌ {name} 失败: {e}")
            results[name] = {
                "status": "error",
                "error": str(e)
            }
    
    # 生成报告
    print("\n📊 示例运行报告")
    print("=" * 50)
    
    success_count = sum(1 for r in results.values() if r["status"] == "success")
    total_count = len(results)
    
    print(f"总示例数: {total_count}")
    print(f"成功数: {success_count}")
    print(f"失败数: {total_count - success_count}")
    print(f"成功率: {success_count/total_count*100:.1f}%")
    
    return results

if __name__ == "__main__":
    run_all_examples()
