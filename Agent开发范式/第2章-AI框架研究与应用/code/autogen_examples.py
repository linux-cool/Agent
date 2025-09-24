# autogen_examples.py
"""
AutoGen框架示例代码
展示AutoGen多智能体对话协作的核心功能
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# AutoGen核心组件
import autogen
from autogen import ConversableAgent, GroupChat, GroupChatManager
from autogen.coding import LocalCommandLineCodeExecutor
from autogen.agentchat.contrib import MultimodalConversableAgent

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoGenExamples:
    """AutoGen示例集合"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OpenAI API key not found. Some examples may not work.")
        
        # 配置LLM
        self.llm_config = {
            "model": "gpt-3.5-turbo",
            "api_key": self.api_key,
            "temperature": 0.7,
            "timeout": 120,
        }
    
    def basic_conversation_example(self) -> str:
        """基础对话示例"""
        logger.info("Running basic conversation example...")
        
        # 创建智能体
        user_proxy = ConversableAgent(
            name="user_proxy",
            system_message="You are a helpful assistant.",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3,
        )
        
        assistant = ConversableAgent(
            name="assistant",
            system_message="You are a helpful AI assistant.",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3,
        )
        
        # 进行对话
        result = user_proxy.initiate_chat(
            assistant,
            message="Hello! Can you tell me about artificial intelligence?",
            max_turns=3
        )
        
        return f"Conversation completed. Messages: {len(result.chat_history)}"
    
    def group_chat_example(self) -> str:
        """群组对话示例"""
        logger.info("Running group chat example...")
        
        # 创建智能体
        user_proxy = ConversableAgent(
            name="user_proxy",
            system_message="You are a user who asks questions.",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=2,
        )
        
        researcher = ConversableAgent(
            name="researcher",
            system_message="You are a research expert who provides detailed information.",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=2,
        )
        
        analyst = ConversableAgent(
            name="analyst",
            system_message="You are an analyst who provides insights and analysis.",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=2,
        )
        
        # 创建群组对话
        groupchat = GroupChat(
            agents=[user_proxy, researcher, analyst],
            messages=[],
            max_round=6,
            speaker_selection_method="auto"
        )
        
        manager = GroupChatManager(
            groupchat=groupchat,
            llm_config=self.llm_config
        )
        
        # 开始群组对话
        result = user_proxy.initiate_chat(
            manager,
            message="Let's discuss the future of artificial intelligence. What are the key trends?",
            max_turns=6
        )
        
        return f"Group chat completed. Messages: {len(result.chat_history)}"
    
    def code_execution_example(self) -> str:
        """代码执行示例"""
        logger.info("Running code execution example...")
        
        # 创建代码执行器
        executor = LocalCommandLineCodeExecutor(
            timeout=10,
            work_dir="temp_code"
        )
        
        # 创建智能体
        user_proxy = ConversableAgent(
            name="user_proxy",
            system_message="You are a user who asks for code solutions.",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=2,
        )
        
        coder = ConversableAgent(
            name="coder",
            system_message="You are a programmer who writes and executes code.",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=2,
            code_execution_config={"executor": executor}
        )
        
        # 进行代码对话
        result = user_proxy.initiate_chat(
            coder,
            message="Write a Python function to calculate the factorial of a number and test it with 5.",
            max_turns=3
        )
        
        return f"Code execution completed. Messages: {len(result.chat_history)}"
    
    def function_calling_example(self) -> str:
        """函数调用示例"""
        logger.info("Running function calling example...")
        
        # 定义函数
        def get_weather(city: str) -> str:
            """获取天气信息"""
            weather_data = {
                "Beijing": "Sunny, 25°C",
                "Shanghai": "Cloudy, 22°C",
                "Guangzhou": "Rainy, 28°C",
                "Shenzhen": "Sunny, 30°C"
            }
            return weather_data.get(city, f"Weather data not available for {city}")
        
        def calculate(expression: str) -> str:
            """执行数学计算"""
            try:
                result = eval(expression)
                return str(result)
            except Exception as e:
                return f"Error: {e}"
        
        # 创建智能体
        user_proxy = ConversableAgent(
            name="user_proxy",
            system_message="You are a user who asks for information and calculations.",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=2,
        )
        
        assistant = ConversableAgent(
            name="assistant",
            system_message="You are a helpful assistant with access to weather and calculation functions.",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=2,
            function_map={
                "get_weather": get_weather,
                "calculate": calculate
            }
        )
        
        # 进行函数调用对话
        result = user_proxy.initiate_chat(
            assistant,
            message="What's the weather in Beijing? Also, calculate 15 * 8.",
            max_turns=3
        )
        
        return f"Function calling completed. Messages: {len(result.chat_history)}"
    
    def multimodal_example(self) -> str:
        """多模态示例"""
        logger.info("Running multimodal example...")
        
        # 创建多模态智能体
        user_proxy = ConversableAgent(
            name="user_proxy",
            system_message="You are a user who can provide text and images.",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=2,
        )
        
        multimodal_agent = MultimodalConversableAgent(
            name="multimodal_agent",
            system_message="You are a multimodal assistant who can understand text and images.",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=2,
        )
        
        # 进行多模态对话
        result = user_proxy.initiate_chat(
            multimodal_agent,
            message="Describe what you see in this image: [This is a text description of an image]",
            max_turns=2
        )
        
        return f"Multimodal conversation completed. Messages: {len(result.chat_history)}"
    
    def hierarchical_agents_example(self) -> str:
        """层次化智能体示例"""
        logger.info("Running hierarchical agents example...")
        
        # 创建管理者智能体
        manager = ConversableAgent(
            name="manager",
            system_message="You are a project manager who coordinates tasks and delegates work.",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=2,
        )
        
        # 创建专业智能体
        developer = ConversableAgent(
            name="developer",
            system_message="You are a software developer who writes code and implements solutions.",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=2,
        )
        
        tester = ConversableAgent(
            name="tester",
            system_message="You are a QA tester who tests software and reports bugs.",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=2,
        )
        
        # 创建群组对话
        groupchat = GroupChat(
            agents=[manager, developer, tester],
            messages=[],
            max_round=8,
            speaker_selection_method="auto"
        )
        
        manager_group = GroupChatManager(
            groupchat=groupchat,
            llm_config=self.llm_config
        )
        
        # 开始层次化对话
        result = manager.initiate_chat(
            manager_group,
            message="We need to develop a simple calculator application. Let's plan the development process.",
            max_turns=8
        )
        
        return f"Hierarchical agents conversation completed. Messages: {len(result.chat_history)}"
    
    def memory_example(self) -> str:
        """记忆系统示例"""
        logger.info("Running memory example...")
        
        # 创建带记忆的智能体
        user_proxy = ConversableAgent(
            name="user_proxy",
            system_message="You are a user who asks questions.",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=2,
        )
        
        assistant = ConversableAgent(
            name="assistant",
            system_message="You are a helpful assistant who remembers previous conversations.",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=2,
        )
        
        # 第一轮对话
        result1 = user_proxy.initiate_chat(
            assistant,
            message="My name is Alice and I'm interested in machine learning.",
            max_turns=2
        )
        
        # 第二轮对话（测试记忆）
        result2 = user_proxy.initiate_chat(
            assistant,
            message="What's my name and what am I interested in?",
            max_turns=2
        )
        
        return f"Memory test completed. First conversation: {len(result1.chat_history)} messages, Second conversation: {len(result2.chat_history)} messages"
    
    def error_handling_example(self) -> str:
        """错误处理示例"""
        logger.info("Running error handling example...")
        
        try:
            # 创建智能体
            user_proxy = ConversableAgent(
                name="user_proxy",
                system_message="You are a user who asks questions.",
                llm_config=self.llm_config,
                human_input_mode="NEVER",
                max_consecutive_auto_reply=2,
            )
            
            assistant = ConversableAgent(
                name="assistant",
                system_message="You are a helpful assistant.",
                llm_config=self.llm_config,
                human_input_mode="NEVER",
                max_consecutive_auto_reply=2,
            )
            
            # 进行对话
            result = user_proxy.initiate_chat(
                assistant,
                message="This is a test message.",
                max_turns=2
            )
            
            return f"Success: {len(result.chat_history)} messages"
            
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
            
            # 创建优化配置的LLM
            llm_config = {
                "model": "gpt-3.5-turbo",
                "api_key": self.api_key,
                **config
            }
            
            # 创建智能体
            user_proxy = ConversableAgent(
                name="user_proxy",
                system_message="You are a user who asks questions.",
                llm_config=llm_config,
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1,
            )
            
            assistant = ConversableAgent(
                name="assistant",
                system_message="You are a helpful assistant.",
                llm_config=llm_config,
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1,
            )
            
            # 进行对话
            result = user_proxy.initiate_chat(
                assistant,
                message="Explain machine learning in one sentence.",
                max_turns=1
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            results.append({
                "config": config,
                "execution_time": execution_time,
                "messages": len(result.chat_history)
            })
        
        return f"Performance test results: {results}"
    
    def async_example(self) -> str:
        """异步处理示例"""
        logger.info("Running async example...")
        
        async def async_conversation():
            # 创建智能体
            user_proxy = ConversableAgent(
                name="user_proxy",
                system_message="You are a user who asks questions.",
                llm_config=self.llm_config,
                human_input_mode="NEVER",
                max_consecutive_auto_reply=2,
            )
            
            assistant = ConversableAgent(
                name="assistant",
                system_message="You are a helpful assistant.",
                llm_config=self.llm_config,
                human_input_mode="NEVER",
                max_consecutive_auto_reply=2,
            )
            
            # 异步进行对话
            result = await user_proxy.a_initiate_chat(
                assistant,
                message="Tell me about artificial intelligence.",
                max_turns=2
            )
            
            return result
        
        # 运行异步任务
        result = asyncio.run(async_conversation())
        
        return f"Async conversation completed. Messages: {len(result.chat_history)}"

def run_all_examples():
    """运行所有示例"""
    print("🚀 AutoGen框架示例演示")
    print("=" * 50)
    
    # 创建示例实例
    examples = AutoGenExamples()
    
    # 定义示例列表
    example_methods = [
        ("基础对话", examples.basic_conversation_example),
        ("群组对话", examples.group_chat_example),
        ("代码执行", examples.code_execution_example),
        ("函数调用", examples.function_calling_example),
        ("多模态", examples.multimodal_example),
        ("层次化智能体", examples.hierarchical_agents_example),
        ("记忆系统", examples.memory_example),
        ("错误处理", examples.error_handling_example),
        ("性能优化", examples.performance_optimization_example),
        ("异步处理", examples.async_example)
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
