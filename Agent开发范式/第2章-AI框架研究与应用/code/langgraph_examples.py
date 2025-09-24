# langgraph_examples.py
"""
LangGraph框架示例代码
展示LangGraph状态机工作流的核心功能
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from datetime import datetime

# LangGraph核心组件
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LangGraphExamples:
    """LangGraph示例集合"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OpenAI API key not found. Some examples may not work.")
        
        # 配置LLM
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=self.api_key,
            temperature=0.7
        )
    
    def basic_workflow_example(self) -> str:
        """基础工作流示例"""
        logger.info("Running basic workflow example...")
        
        # 定义状态
        class WorkflowState(TypedDict):
            messages: Annotated[List[BaseMessage], add_messages]
            step_count: int
        
        # 定义节点函数
        def start_node(state: WorkflowState) -> WorkflowState:
            """开始节点"""
            return {
                "messages": [HumanMessage(content="Workflow started")],
                "step_count": 1
            }
        
        def process_node(state: WorkflowState) -> WorkflowState:
            """处理节点"""
            messages = state["messages"]
            step_count = state["step_count"]
            
            # 添加处理消息
            new_message = AIMessage(content=f"Processing step {step_count}")
            messages.append(new_message)
            
            return {
                "messages": messages,
                "step_count": step_count + 1
            }
        
        def end_node(state: WorkflowState) -> WorkflowState:
            """结束节点"""
            messages = state["messages"]
            messages.append(AIMessage(content="Workflow completed"))
            
            return {
                "messages": messages,
                "step_count": state["step_count"]
            }
        
        # 创建状态图
        workflow = StateGraph(WorkflowState)
        
        # 添加节点
        workflow.add_node("start", start_node)
        workflow.add_node("process", process_node)
        workflow.add_node("end", end_node)
        
        # 添加边
        workflow.add_edge(START, "start")
        workflow.add_edge("start", "process")
        workflow.add_edge("process", "end")
        workflow.add_edge("end", END)
        
        # 编译图
        app = workflow.compile()
        
        # 执行工作流
        initial_state = {"messages": [], "step_count": 0}
        result = app.invoke(initial_state)
        
        return f"Workflow completed. Messages: {len(result['messages'])}, Steps: {result['step_count']}"
    
    def conditional_workflow_example(self) -> str:
        """条件工作流示例"""
        logger.info("Running conditional workflow example...")
        
        # 定义状态
        class ConditionalState(TypedDict):
            messages: Annotated[List[BaseMessage], add_messages]
            condition: str
            result: str
        
        # 定义节点函数
        def input_node(state: ConditionalState) -> ConditionalState:
            """输入节点"""
            return {
                "messages": [HumanMessage(content="Input received")],
                "condition": "positive",
                "result": ""
            }
        
        def positive_node(state: ConditionalState) -> ConditionalState:
            """正数处理节点"""
            messages = state["messages"]
            messages.append(AIMessage(content="Processing positive case"))
            
            return {
                "messages": messages,
                "condition": state["condition"],
                "result": "Positive result"
            }
        
        def negative_node(state: ConditionalState) -> ConditionalState:
            """负数处理节点"""
            messages = state["messages"]
            messages.append(AIMessage(content="Processing negative case"))
            
            return {
                "messages": messages,
                "condition": state["condition"],
                "result": "Negative result"
            }
        
        def end_node(state: ConditionalState) -> ConditionalState:
            """结束节点"""
            messages = state["messages"]
            messages.append(AIMessage(content=f"Final result: {state['result']}"))
            
            return {
                "messages": messages,
                "condition": state["condition"],
                "result": state["result"]
            }
        
        # 定义条件函数
        def should_process_positive(state: ConditionalState) -> str:
            """判断是否处理正数"""
            return "positive" if state["condition"] == "positive" else "negative"
        
        # 创建状态图
        workflow = StateGraph(ConditionalState)
        
        # 添加节点
        workflow.add_node("input", input_node)
        workflow.add_node("positive", positive_node)
        workflow.add_node("negative", negative_node)
        workflow.add_node("end", end_node)
        
        # 添加边
        workflow.add_edge(START, "input")
        workflow.add_conditional_edges(
            "input",
            should_process_positive,
            {
                "positive": "positive",
                "negative": "negative"
            }
        )
        workflow.add_edge("positive", "end")
        workflow.add_edge("negative", "end")
        workflow.add_edge("end", END)
        
        # 编译图
        app = workflow.compile()
        
        # 执行工作流
        initial_state = {"messages": [], "condition": "positive", "result": ""}
        result = app.invoke(initial_state)
        
        return f"Conditional workflow completed. Result: {result['result']}"
    
    def loop_workflow_example(self) -> str:
        """循环工作流示例"""
        logger.info("Running loop workflow example...")
        
        # 定义状态
        class LoopState(TypedDict):
            messages: Annotated[List[BaseMessage], add_messages]
            counter: int
            max_iterations: int
        
        # 定义节点函数
        def start_node(state: LoopState) -> LoopState:
            """开始节点"""
            return {
                "messages": [HumanMessage(content="Loop started")],
                "counter": 0,
                "max_iterations": 3
            }
        
        def loop_node(state: LoopState) -> LoopState:
            """循环节点"""
            messages = state["messages"]
            counter = state["counter"] + 1
            
            messages.append(AIMessage(content=f"Loop iteration {counter}"))
            
            return {
                "messages": messages,
                "counter": counter,
                "max_iterations": state["max_iterations"]
            }
        
        def end_node(state: LoopState) -> LoopState:
            """结束节点"""
            messages = state["messages"]
            messages.append(AIMessage(content="Loop completed"))
            
            return {
                "messages": messages,
                "counter": state["counter"],
                "max_iterations": state["max_iterations"]
            }
        
        # 定义条件函数
        def should_continue(state: LoopState) -> str:
            """判断是否继续循环"""
            return "continue" if state["counter"] < state["max_iterations"] else "end"
        
        # 创建状态图
        workflow = StateGraph(LoopState)
        
        # 添加节点
        workflow.add_node("start", start_node)
        workflow.add_node("loop", loop_node)
        workflow.add_node("end", end_node)
        
        # 添加边
        workflow.add_edge(START, "start")
        workflow.add_edge("start", "loop")
        workflow.add_conditional_edges(
            "loop",
            should_continue,
            {
                "continue": "loop",
                "end": "end"
            }
        )
        workflow.add_edge("end", END)
        
        # 编译图
        app = workflow.compile()
        
        # 执行工作流
        initial_state = {"messages": [], "counter": 0, "max_iterations": 3}
        result = app.invoke(initial_state)
        
        return f"Loop workflow completed. Iterations: {result['counter']}"
    
    def tool_integration_example(self) -> str:
        """工具集成示例"""
        logger.info("Running tool integration example...")
        
        # 定义工具
        @tool
        def get_weather(city: str) -> str:
            """获取天气信息"""
            weather_data = {
                "Beijing": "Sunny, 25°C",
                "Shanghai": "Cloudy, 22°C",
                "Guangzhou": "Rainy, 28°C",
                "Shenzhen": "Sunny, 30°C"
            }
            return weather_data.get(city, f"Weather data not available for {city}")
        
        @tool
        def calculate(expression: str) -> str:
            """执行数学计算"""
            try:
                result = eval(expression)
                return str(result)
            except Exception as e:
                return f"Error: {e}"
        
        # 定义状态
        class ToolState(TypedDict):
            messages: Annotated[List[BaseMessage], add_messages]
            tools_used: List[str]
        
        # 定义节点函数
        def start_node(state: ToolState) -> ToolState:
            """开始节点"""
            return {
                "messages": [HumanMessage(content="Tool integration started")],
                "tools_used": []
            }
        
        def weather_node(state: ToolState) -> ToolState:
            """天气工具节点"""
            messages = state["messages"]
            tools_used = state["tools_used"]
            
            weather_result = get_weather("Beijing")
            messages.append(AIMessage(content=f"Weather in Beijing: {weather_result}"))
            tools_used.append("get_weather")
            
            return {
                "messages": messages,
                "tools_used": tools_used
            }
        
        def calculate_node(state: ToolState) -> ToolState:
            """计算工具节点"""
            messages = state["messages"]
            tools_used = state["tools_used"]
            
            calc_result = calculate("15 * 8")
            messages.append(AIMessage(content=f"Calculation result: {calc_result}"))
            tools_used.append("calculate")
            
            return {
                "messages": messages,
                "tools_used": tools_used
            }
        
        def end_node(state: ToolState) -> ToolState:
            """结束节点"""
            messages = state["messages"]
            tools_used = state["tools_used"]
            
            messages.append(AIMessage(content=f"Tools used: {', '.join(tools_used)}"))
            
            return {
                "messages": messages,
                "tools_used": tools_used
            }
        
        # 创建状态图
        workflow = StateGraph(ToolState)
        
        # 添加节点
        workflow.add_node("start", start_node)
        workflow.add_node("weather", weather_node)
        workflow.add_node("calculate", calculate_node)
        workflow.add_node("end", end_node)
        
        # 添加边
        workflow.add_edge(START, "start")
        workflow.add_edge("start", "weather")
        workflow.add_edge("weather", "calculate")
        workflow.add_edge("calculate", "end")
        workflow.add_edge("end", END)
        
        # 编译图
        app = workflow.compile()
        
        # 执行工作流
        initial_state = {"messages": [], "tools_used": []}
        result = app.invoke(initial_state)
        
        return f"Tool integration completed. Tools used: {result['tools_used']}"
    
    def parallel_execution_example(self) -> str:
        """并行执行示例"""
        logger.info("Running parallel execution example...")
        
        # 定义状态
        class ParallelState(TypedDict):
            messages: Annotated[List[BaseMessage], add_messages]
            task1_result: str
            task2_result: str
            task3_result: str
        
        # 定义节点函数
        def start_node(state: ParallelState) -> ParallelState:
            """开始节点"""
            return {
                "messages": [HumanMessage(content="Parallel execution started")],
                "task1_result": "",
                "task2_result": "",
                "task3_result": ""
            }
        
        def task1_node(state: ParallelState) -> ParallelState:
            """任务1节点"""
            messages = state["messages"]
            messages.append(AIMessage(content="Task 1 completed"))
            
            return {
                "messages": messages,
                "task1_result": "Task 1 result",
                "task2_result": state["task2_result"],
                "task3_result": state["task3_result"]
            }
        
        def task2_node(state: ParallelState) -> ParallelState:
            """任务2节点"""
            messages = state["messages"]
            messages.append(AIMessage(content="Task 2 completed"))
            
            return {
                "messages": messages,
                "task1_result": state["task1_result"],
                "task2_result": "Task 2 result",
                "task3_result": state["task3_result"]
            }
        
        def task3_node(state: ParallelState) -> ParallelState:
            """任务3节点"""
            messages = state["messages"]
            messages.append(AIMessage(content="Task 3 completed"))
            
            return {
                "messages": messages,
                "task1_result": state["task1_result"],
                "task2_result": state["task2_result"],
                "task3_result": "Task 3 result"
            }
        
        def merge_node(state: ParallelState) -> ParallelState:
            """合并节点"""
            messages = state["messages"]
            messages.append(AIMessage(content="All tasks completed and merged"))
            
            return {
                "messages": messages,
                "task1_result": state["task1_result"],
                "task2_result": state["task2_result"],
                "task3_result": state["task3_result"]
            }
        
        # 创建状态图
        workflow = StateGraph(ParallelState)
        
        # 添加节点
        workflow.add_node("start", start_node)
        workflow.add_node("task1", task1_node)
        workflow.add_node("task2", task2_node)
        workflow.add_node("task3", task3_node)
        workflow.add_node("merge", merge_node)
        
        # 添加边
        workflow.add_edge(START, "start")
        workflow.add_edge("start", "task1")
        workflow.add_edge("start", "task2")
        workflow.add_edge("start", "task3")
        workflow.add_edge("task1", "merge")
        workflow.add_edge("task2", "merge")
        workflow.add_edge("task3", "merge")
        workflow.add_edge("merge", END)
        
        # 编译图
        app = workflow.compile()
        
        # 执行工作流
        initial_state = {
            "messages": [],
            "task1_result": "",
            "task2_result": "",
            "task3_result": ""
        }
        result = app.invoke(initial_state)
        
        return f"Parallel execution completed. Results: {result['task1_result']}, {result['task2_result']}, {result['task3_result']}"
    
    def error_handling_example(self) -> str:
        """错误处理示例"""
        logger.info("Running error handling example...")
        
        # 定义状态
        class ErrorState(TypedDict):
            messages: Annotated[List[BaseMessage], add_messages]
            error_count: int
            success: bool
        
        # 定义节点函数
        def start_node(state: ErrorState) -> ErrorState:
            """开始节点"""
            return {
                "messages": [HumanMessage(content="Error handling test started")],
                "error_count": 0,
                "success": False
            }
        
        def risky_node(state: ErrorState) -> ErrorState:
            """风险节点"""
            messages = state["messages"]
            error_count = state["error_count"]
            
            try:
                # 模拟可能出错的操作
                if error_count < 2:
                    raise Exception("Simulated error")
                
                messages.append(AIMessage(content="Risky operation succeeded"))
                return {
                    "messages": messages,
                    "error_count": error_count,
                    "success": True
                }
                
            except Exception as e:
                messages.append(AIMessage(content=f"Error occurred: {e}"))
                return {
                    "messages": messages,
                    "error_count": error_count + 1,
                    "success": False
                }
        
        def retry_node(state: ErrorState) -> ErrorState:
            """重试节点"""
            messages = state["messages"]
            messages.append(AIMessage(content="Retrying operation"))
            
            return {
                "messages": messages,
                "error_count": state["error_count"],
                "success": state["success"]
            }
        
        def end_node(state: ErrorState) -> ErrorState:
            """结束节点"""
            messages = state["messages"]
            messages.append(AIMessage(content="Error handling completed"))
            
            return {
                "messages": messages,
                "error_count": state["error_count"],
                "success": state["success"]
            }
        
        # 定义条件函数
        def should_retry(state: ErrorState) -> str:
            """判断是否重试"""
            if state["success"]:
                return "end"
            elif state["error_count"] < 3:
                return "retry"
            else:
                return "end"
        
        # 创建状态图
        workflow = StateGraph(ErrorState)
        
        # 添加节点
        workflow.add_node("start", start_node)
        workflow.add_node("risky", risky_node)
        workflow.add_node("retry", retry_node)
        workflow.add_node("end", end_node)
        
        # 添加边
        workflow.add_edge(START, "start")
        workflow.add_edge("start", "risky")
        workflow.add_conditional_edges(
            "risky",
            should_retry,
            {
                "retry": "retry",
                "end": "end"
            }
        )
        workflow.add_edge("retry", "risky")
        workflow.add_edge("end", END)
        
        # 编译图
        app = workflow.compile()
        
        # 执行工作流
        initial_state = {"messages": [], "error_count": 0, "success": False}
        result = app.invoke(initial_state)
        
        return f"Error handling completed. Success: {result['success']}, Errors: {result['error_count']}"
    
    def performance_optimization_example(self) -> str:
        """性能优化示例"""
        logger.info("Running performance optimization example...")
        
        import time
        
        # 定义状态
        class PerformanceState(TypedDict):
            messages: Annotated[List[BaseMessage], add_messages]
            execution_time: float
            optimization_level: str
        
        # 定义节点函数
        def start_node(state: PerformanceState) -> PerformanceState:
            """开始节点"""
            return {
                "messages": [HumanMessage(content="Performance optimization started")],
                "execution_time": 0.0,
                "optimization_level": "basic"
            }
        
        def optimize_node(state: PerformanceState) -> PerformanceState:
            """优化节点"""
            messages = state["messages"]
            start_time = time.time()
            
            # 模拟优化操作
            time.sleep(0.1)  # 模拟处理时间
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            messages.append(AIMessage(content=f"Optimization completed in {execution_time:.3f}s"))
            
            return {
                "messages": messages,
                "execution_time": execution_time,
                "optimization_level": "optimized"
            }
        
        def end_node(state: PerformanceState) -> PerformanceState:
            """结束节点"""
            messages = state["messages"]
            messages.append(AIMessage(content="Performance optimization completed"))
            
            return {
                "messages": messages,
                "execution_time": state["execution_time"],
                "optimization_level": state["optimization_level"]
            }
        
        # 创建状态图
        workflow = StateGraph(PerformanceState)
        
        # 添加节点
        workflow.add_node("start", start_node)
        workflow.add_node("optimize", optimize_node)
        workflow.add_node("end", end_node)
        
        # 添加边
        workflow.add_edge(START, "start")
        workflow.add_edge("start", "optimize")
        workflow.add_edge("optimize", "end")
        workflow.add_edge("end", END)
        
        # 编译图
        app = workflow.compile()
        
        # 执行工作流
        initial_state = {"messages": [], "execution_time": 0.0, "optimization_level": "basic"}
        result = app.invoke(initial_state)
        
        return f"Performance optimization completed. Execution time: {result['execution_time']:.3f}s"
    
    def async_example(self) -> str:
        """异步处理示例"""
        logger.info("Running async example...")
        
        # 定义状态
        class AsyncState(TypedDict):
            messages: Annotated[List[BaseMessage], add_messages]
            async_result: str
        
        # 定义异步节点函数
        async def async_node(state: AsyncState) -> AsyncState:
            """异步节点"""
            messages = state["messages"]
            
            # 模拟异步操作
            await asyncio.sleep(0.1)
            
            messages.append(AIMessage(content="Async operation completed"))
            
            return {
                "messages": messages,
                "async_result": "Async success"
            }
        
        def start_node(state: AsyncState) -> AsyncState:
            """开始节点"""
            return {
                "messages": [HumanMessage(content="Async workflow started")],
                "async_result": ""
            }
        
        def end_node(state: AsyncState) -> AsyncState:
            """结束节点"""
            messages = state["messages"]
            messages.append(AIMessage(content="Async workflow completed"))
            
            return {
                "messages": messages,
                "async_result": state["async_result"]
            }
        
        # 创建状态图
        workflow = StateGraph(AsyncState)
        
        # 添加节点
        workflow.add_node("start", start_node)
        workflow.add_node("async", async_node)
        workflow.add_node("end", end_node)
        
        # 添加边
        workflow.add_edge(START, "start")
        workflow.add_edge("start", "async")
        workflow.add_edge("async", "end")
        workflow.add_edge("end", END)
        
        # 编译图
        app = workflow.compile()
        
        # 异步执行工作流
        async def run_async():
            initial_state = {"messages": [], "async_result": ""}
            result = await app.ainvoke(initial_state)
            return result
        
        result = asyncio.run(run_async())
        
        return f"Async workflow completed. Result: {result['async_result']}"

def run_all_examples():
    """运行所有示例"""
    print("🚀 LangGraph框架示例演示")
    print("=" * 50)
    
    # 创建示例实例
    examples = LangGraphExamples()
    
    # 定义示例列表
    example_methods = [
        ("基础工作流", examples.basic_workflow_example),
        ("条件工作流", examples.conditional_workflow_example),
        ("循环工作流", examples.loop_workflow_example),
        ("工具集成", examples.tool_integration_example),
        ("并行执行", examples.parallel_execution_example),
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
