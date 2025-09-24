# agent_template.py
"""
AI智能体开发模板
基于六大核心技术支柱的完整智能体实现
"""

from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
import json
import time
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(Enum):
    """智能体状态枚举"""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    REFLECTING = "reflecting"

@dataclass
class Task:
    """任务数据结构"""
    id: str
    description: str
    priority: int
    status: str
    context: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

@dataclass
class Memory:
    """记忆数据结构"""
    short_term: Dict[str, Any]
    long_term: Dict[str, Any]
    process: List[Dict[str, Any]]

class Tool(ABC):
    """工具基类"""
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """执行工具"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述"""
        pass

class TaskPlanner:
    """任务规划器 - 核心技术支柱1"""
    
    def __init__(self):
        self.tasks = []
        self.task_counter = 0
    
    async def decompose_task(self, task_description: str) -> List[Task]:
        """任务分解"""
        logger.info(f"Decomposing task: {task_description}")
        
        # 简单的任务分解逻辑
        subtasks = [
            f"Analyze: {task_description}",
            f"Plan: {task_description}",
            f"Execute: {task_description}",
            f"Review: {task_description}"
        ]
        
        tasks = []
        for i, subtask in enumerate(subtasks):
            task = Task(
                id=f"task_{self.task_counter}_{i}",
                description=subtask,
                priority=len(subtasks) - i,
                status="pending",
                context={"parent_task": task_description},
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            tasks.append(task)
            self.task_counter += 1
        
        self.tasks.extend(tasks)
        return tasks
    
    async def prioritize_tasks(self, tasks: List[Task]) -> List[Task]:
        """任务优先级排序"""
        return sorted(tasks, key=lambda x: x.priority, reverse=True)
    
    async def update_task_status(self, task_id: str, status: str):
        """更新任务状态"""
        for task in self.tasks:
            if task.id == task_id:
                task.status = status
                task.updated_at = datetime.now()
                break

class MemoryManager:
    """记忆管理器 - 核心技术支柱2"""
    
    def __init__(self):
        self.memory = Memory(
            short_term={},
            long_term={},
            process=[]
        )
        self.max_short_term = 1000
        self.max_process_steps = 100
    
    async def store_short_term(self, key: str, value: Any):
        """存储短期记忆"""
        self.memory.short_term[key] = {
            "value": value,
            "timestamp": datetime.now(),
            "access_count": 0
        }
        
        # 清理过期记忆
        if len(self.memory.short_term) > self.max_short_term:
            await self._cleanup_short_term()
    
    async def store_long_term(self, key: str, value: Any):
        """存储长期记忆"""
        self.memory.long_term[key] = {
            "value": value,
            "timestamp": datetime.now(),
            "importance": 1.0
        }
    
    async def store_process(self, step: Dict[str, Any]):
        """存储过程记忆"""
        step["timestamp"] = datetime.now()
        self.memory.process.append(step)
        
        # 保持过程记忆在合理范围内
        if len(self.memory.process) > self.max_process_steps:
            self.memory.process = self.memory.process[-self.max_process_steps:]
    
    async def retrieve_context(self, query: str) -> Dict[str, Any]:
        """检索相关上下文"""
        context = {
            "short_term": {},
            "long_term": {},
            "recent_process": self.memory.process[-5:] if self.memory.process else []
        }
        
        # 简单的关键词匹配
        query_lower = query.lower()
        for key, value in self.memory.short_term.items():
            if query_lower in str(value["value"]).lower():
                context["short_term"][key] = value
                value["access_count"] += 1
        
        for key, value in self.memory.long_term.items():
            if query_lower in str(value["value"]).lower():
                context["long_term"][key] = value
        
        return context
    
    async def _cleanup_short_term(self):
        """清理短期记忆"""
        # 按访问次数和时间清理
        sorted_items = sorted(
            self.memory.short_term.items(),
            key=lambda x: (x[1]["access_count"], x[1]["timestamp"]),
            reverse=True
        )
        
        # 保留前80%的记忆
        keep_count = int(len(sorted_items) * 0.8)
        self.memory.short_term = dict(sorted_items[:keep_count])

class ToolManager:
    """工具管理器 - 核心技术支柱3"""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
    
    def register_tool(self, tool: Tool):
        """注册工具"""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """执行工具"""
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found")
        
        logger.info(f"Executing tool: {tool_name}")
        start_time = time.time()
        
        try:
            result = await self.tools[tool_name].execute(**kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Tool {tool_name} executed successfully in {execution_time:.2f}s")
            return result
        except Exception as e:
            logger.error(f"Tool {tool_name} execution failed: {e}")
            raise
    
    def list_tools(self) -> List[Dict[str, str]]:
        """列出所有工具"""
        return [
            {
                "name": tool.name,
                "description": tool.description
            }
            for tool in self.tools.values()
        ]

class AutonomousLoop:
    """自治循环模块 - 核心技术支柱4"""
    
    def __init__(self, agent):
        self.agent = agent
        self.max_iterations = 10
        self.reflection_threshold = 0.7
    
    async def react_loop(self, input_data: str) -> str:
        """ReAct循环"""
        logger.info(f"Starting ReAct loop with: {input_data}")
        
        iteration = 0
        result = None
        
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"ReAct iteration {iteration}")
            
            # 思考阶段
            thoughts = await self.think(input_data, result)
            
            # 行动阶段
            action_result = await self.act(thoughts)
            
            # 观察阶段
            observation = await self.observe(action_result)
            
            # 反思阶段
            reflection = await self.reflect(observation)
            
            # 检查是否完成任务
            if await self.is_task_complete(reflection):
                result = action_result
                break
            
            # 更新输入数据用于下一轮
            input_data = reflection
        
        return result or "Task completed with iterations limit reached"
    
    async def think(self, input_data: str, previous_result: Any = None) -> str:
        """思考阶段"""
        self.agent.state = AgentState.THINKING
        logger.info(f"{self.agent.name} is thinking...")
        
        # 检索相关记忆
        context = await self.agent.memory_manager.retrieve_context(input_data)
        
        # 生成思考
        thoughts = f"Processing: {input_data}"
        if previous_result:
            thoughts += f" Previous result: {previous_result}"
        
        await self.agent.memory_manager.store_short_term("current_thoughts", thoughts)
        await self.agent.memory_manager.store_process({
            "phase": "think",
            "input": input_data,
            "thoughts": thoughts,
            "context": context
        })
        
        return thoughts
    
    async def act(self, thoughts: str) -> Any:
        """行动阶段"""
        self.agent.state = AgentState.ACTING
        logger.info(f"{self.agent.name} is acting based on: {thoughts}")
        
        # 简单的行动逻辑
        if "calculate" in thoughts.lower():
            action_result = await self.agent.tool_manager.execute_tool("calculator", operation="2+2")
        elif "search" in thoughts.lower():
            action_result = await self.agent.tool_manager.execute_tool("web_search", query=thoughts)
        else:
            action_result = f"Action result for: {thoughts}"
        
        await self.agent.memory_manager.store_process({
            "phase": "act",
            "thoughts": thoughts,
            "action_result": action_result
        })
        
        return action_result
    
    async def observe(self, result: Any) -> str:
        """观察阶段"""
        self.agent.state = AgentState.OBSERVING
        logger.info(f"{self.agent.name} is observing result...")
        
        observation = f"Observed result: {result}"
        await self.agent.memory_manager.store_short_term("last_observation", observation)
        await self.agent.memory_manager.store_process({
            "phase": "observe",
            "result": result,
            "observation": observation
        })
        
        return observation
    
    async def reflect(self, observation: str) -> str:
        """反思阶段"""
        self.agent.state = AgentState.REFLECTING
        logger.info(f"{self.agent.name} is reflecting...")
        
        reflection = f"Reflection on: {observation}"
        await self.agent.memory_manager.store_long_term("reflection", reflection)
        await self.agent.memory_manager.store_process({
            "phase": "reflect",
            "observation": observation,
            "reflection": reflection
        })
        
        return reflection
    
    async def is_task_complete(self, reflection: str) -> bool:
        """检查任务是否完成"""
        # 简单的完成条件检查
        completion_keywords = ["complete", "finished", "done", "success"]
        return any(keyword in reflection.lower() for keyword in completion_keywords)

class SecurityController:
    """安全控制器 - 核心技术支柱5"""
    
    def __init__(self):
        self.max_iterations = 10
        self.timeout = 300  # seconds
        self.rate_limit = 100  # requests per minute
        self.request_count = 0
        self.last_reset = time.time()
    
    async def validate_input(self, input_data: str) -> bool:
        """输入验证"""
        # 检查输入长度
        if len(input_data) > 10000:
            logger.warning("Input too long")
            return False
        
        # 检查恶意内容
        malicious_patterns = ["<script>", "eval(", "exec("]
        for pattern in malicious_patterns:
            if pattern in input_data.lower():
                logger.warning(f"Malicious pattern detected: {pattern}")
                return False
        
        return True
    
    async def check_rate_limit(self) -> bool:
        """检查速率限制"""
        current_time = time.time()
        if current_time - self.last_reset > 60:  # 重置计数器
            self.request_count = 0
            self.last_reset = current_time
        
        if self.request_count >= self.rate_limit:
            logger.warning("Rate limit exceeded")
            return False
        
        self.request_count += 1
        return True
    
    async def monitor_execution(self, start_time: float) -> bool:
        """监控执行时间"""
        execution_time = time.time() - start_time
        if execution_time > self.timeout:
            logger.warning(f"Execution timeout: {execution_time}s")
            return False
        return True

class MultiAgentCoordinator:
    """多智能体协调器 - 核心技术支柱6"""
    
    def __init__(self):
        self.agents = {}
        self.communication_protocol = "json"
        self.message_queue = []
    
    def register_agent(self, agent_id: str, agent):
        """注册智能体"""
        self.agents[agent_id] = agent
        logger.info(f"Registered agent: {agent_id}")
    
    async def send_message(self, sender_id: str, receiver_id: str, message: Dict[str, Any]):
        """发送消息"""
        if receiver_id not in self.agents:
            logger.error(f"Agent {receiver_id} not found")
            return False
        
        message_data = {
            "id": f"msg_{len(self.message_queue)}",
            "sender": sender_id,
            "receiver": receiver_id,
            "timestamp": datetime.now(),
            "data": message
        }
        
        self.message_queue.append(message_data)
        logger.info(f"Message sent from {sender_id} to {receiver_id}")
        return True
    
    async def broadcast_message(self, sender_id: str, message: Dict[str, Any]):
        """广播消息"""
        for agent_id in self.agents:
            if agent_id != sender_id:
                await self.send_message(sender_id, agent_id, message)
    
    async def coordinate_task(self, task: str, agent_ids: List[str]) -> Dict[str, Any]:
        """协调任务执行"""
        logger.info(f"Coordinating task: {task} with agents: {agent_ids}")
        
        # 任务分解
        subtasks = await self.decompose_coordination_task(task, agent_ids)
        
        # 分配子任务
        results = {}
        for i, agent_id in enumerate(agent_ids):
            if i < len(subtasks):
                subtask = subtasks[i]
                await self.send_message("coordinator", agent_id, {
                    "type": "task_assignment",
                    "task": subtask
                })
                results[agent_id] = subtask
        
        return results
    
    async def decompose_coordination_task(self, task: str, agent_ids: List[str]) -> List[str]:
        """分解协调任务"""
        # 简单的任务分解
        subtasks = [
            f"Analyze {task}",
            f"Plan {task}",
            f"Execute {task}",
            f"Review {task}"
        ]
        
        return subtasks[:len(agent_ids)]

class Agent:
    """智能体核心类"""
    
    def __init__(self, name: str):
        self.name = name
        self.state = AgentState.IDLE
        self.task_planner = TaskPlanner()
        self.memory_manager = MemoryManager()
        self.tool_manager = ToolManager()
        self.autonomous_loop = AutonomousLoop(self)
        self.security_controller = SecurityController()
        self.current_task: Optional[Task] = None
        self.coordinator: Optional[MultiAgentCoordinator] = None
    
    async def process_input(self, input_data: str) -> str:
        """处理输入"""
        logger.info(f"{self.name} processing input: {input_data}")
        
        # 安全检查
        if not await self.security_controller.validate_input(input_data):
            return "Input validation failed"
        
        if not await self.security_controller.check_rate_limit():
            return "Rate limit exceeded"
        
        start_time = time.time()
        
        try:
            # 任务规划
            tasks = await self.task_planner.decompose_task(input_data)
            prioritized_tasks = await self.task_planner.prioritize_tasks(tasks)
            
            # 执行ReAct循环
            result = await self.autonomous_loop.react_loop(input_data)
            
            # 监控执行时间
            if not await self.security_controller.monitor_execution(start_time):
                return "Execution timeout"
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return f"Error: {e}"
    
    def set_coordinator(self, coordinator: MultiAgentCoordinator):
        """设置协调器"""
        self.coordinator = coordinator
        coordinator.register_agent(self.name, self)

# 示例工具实现
class CalculatorTool(Tool):
    """计算器工具"""
    
    @property
    def name(self) -> str:
        return "calculator"
    
    @property
    def description(self) -> str:
        return "Performs mathematical calculations"
    
    async def execute(self, operation: str, **kwargs) -> float:
        """执行计算"""
        try:
            # 安全的计算，只允许基本数学运算
            allowed_chars = set("0123456789+-*/.() ")
            if not all(c in allowed_chars for c in operation):
                raise ValueError("Invalid characters in operation")
            
            result = eval(operation)
            return float(result)
        except Exception as e:
            raise ValueError(f"Invalid operation: {operation}, Error: {e}")

class WebSearchTool(Tool):
    """网络搜索工具"""
    
    @property
    def name(self) -> str:
        return "web_search"
    
    @property
    def description(self) -> str:
        return "Searches the web for information"
    
    async def execute(self, query: str, **kwargs) -> str:
        """执行网络搜索"""
        # 模拟网络搜索
        return f"Search results for: {query}"

# 使用示例
async def main():
    """主函数示例"""
    # 创建智能体
    agent = Agent("MyAgent")
    
    # 注册工具
    calculator = CalculatorTool()
    web_search = WebSearchTool()
    agent.tool_manager.register_tool(calculator)
    agent.tool_manager.register_tool(web_search)
    
    # 创建多智能体协调器
    coordinator = MultiAgentCoordinator()
    agent.set_coordinator(coordinator)
    
    # 执行任务
    result = await agent.process_input("Calculate 2 + 2 and search for AI agents")
    print(f"Result: {result}")
    
    # 列出工具
    tools = agent.tool_manager.list_tools()
    print(f"Available tools: {tools}")

if __name__ == "__main__":
    asyncio.run(main())
