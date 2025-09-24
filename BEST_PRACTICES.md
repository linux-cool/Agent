# AI智能体开发最佳实践指南

> 基于30+个开源智能体框架深度分析的最佳实践总结

## 📋 目录

1. [开发最佳实践](#开发最佳实践)
2. [架构设计原则](#架构设计原则)
3. [性能优化策略](#性能优化策略)
4. [安全防护措施](#安全防护措施)
5. [部署运维指南](#部署运维指南)
6. [测试质量保证](#测试质量保证)
7. [监控告警体系](#监控告警体系)

---

## 开发最佳实践

### ✅ 1. 模块化设计

**原则：**
- 将智能体功能拆分为独立模块
- 使用接口和抽象类定义标准
- 保持模块间的松耦合

**实现示例：**
```python
from abc import ABC, abstractmethod

class MemoryInterface(ABC):
    """记忆接口定义"""
    
    @abstractmethod
    async def store(self, key: str, value: Any) -> bool:
        pass
    
    @abstractmethod
    async def retrieve(self, key: str) -> Any:
        pass

class ShortTermMemory(MemoryInterface):
    """短期记忆实现"""
    
    async def store(self, key: str, value: Any) -> bool:
        # 实现短期记忆存储
        pass
    
    async def retrieve(self, key: str) -> Any:
        # 实现短期记忆检索
        pass
```

### ✅ 2. 错误处理机制

**策略：**
- 实现完善的异常处理机制
- 提供优雅的降级策略
- 记录详细的错误日志

**实现示例：**
```python
import logging
from typing import Optional

class AgentError(Exception):
    """智能体基础异常"""
    pass

class ToolExecutionError(AgentError):
    """工具执行异常"""
    pass

async def safe_tool_execution(tool_name: str, **kwargs) -> Optional[Any]:
    """安全的工具执行"""
    try:
        result = await tool_manager.execute_tool(tool_name, **kwargs)
        return result
    except ToolExecutionError as e:
        logger.error(f"Tool execution failed: {e}")
        # 降级策略
        return await fallback_execution(tool_name, **kwargs)
    except Exception as e:
        logger.critical(f"Unexpected error: {e}")
        raise AgentError(f"Tool execution failed: {e}")
```

### ✅ 3. 配置管理

**原则：**
- 使用环境变量管理敏感信息
- 实现配置热更新
- 分离开发和生产配置

**实现示例：**
```python
from pydantic_settings import BaseSettings
from typing import Optional

class AgentSettings(BaseSettings):
    """智能体配置"""
    
    # 基础配置
    agent_name: str = "MyAgent"
    version: str = "1.0.0"
    
    # API配置
    openai_api_key: str
    anthropic_api_key: Optional[str] = None
    
    # 数据库配置
    database_url: str
    redis_url: str = "redis://localhost:6379"
    
    # 安全配置
    max_iterations: int = 10
    timeout: int = 300
    rate_limit: int = 100
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# 使用配置
settings = AgentSettings()
```

### ✅ 4. 日志记录

**策略：**
- 使用结构化日志
- 实现日志轮转
- 区分不同级别的日志

**实现示例：**
```python
import structlog
import logging
from datetime import datetime

# 配置结构化日志
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# 使用示例
async def process_task(task_id: str, input_data: str):
    logger.info("Processing task", task_id=task_id, input_length=len(input_data))
    
    try:
        result = await agent.process(input_data)
        logger.info("Task completed", task_id=task_id, success=True)
        return result
    except Exception as e:
        logger.error("Task failed", task_id=task_id, error=str(e), exc_info=True)
        raise
```

---

## 架构设计原则

### 🏗️ 1. 单一职责原则

**每个模块只负责一个功能：**

```python
class TaskPlanner:
    """任务规划器 - 只负责任务分解和规划"""
    
    async def decompose_task(self, task: str) -> List[SubTask]:
        pass
    
    async def prioritize_tasks(self, tasks: List[SubTask]) -> List[SubTask]:
        pass

class MemoryManager:
    """记忆管理器 - 只负责记忆存储和检索"""
    
    async def store_memory(self, key: str, value: Any) -> bool:
        pass
    
    async def retrieve_memory(self, key: str) -> Any:
        pass
```

### 🏗️ 2. 开闭原则

**对扩展开放，对修改关闭：**

```python
class ToolRegistry:
    """工具注册器 - 支持动态注册新工具"""
    
    def __init__(self):
        self._tools = {}
    
    def register_tool(self, tool: Tool):
        """注册新工具而不修改现有代码"""
        self._tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

# 扩展新工具
class CustomTool(Tool):
    def execute(self, **kwargs):
        # 新工具实现
        pass

# 注册新工具
registry.register_tool(CustomTool())
```

### 🏗️ 3. 依赖倒置原则

**依赖抽象而不是具体实现：**

```python
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    """LLM提供者抽象接口"""
    
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI实现"""
    
    async def generate(self, prompt: str) -> str:
        # OpenAI API调用
        pass

class AnthropicProvider(LLMProvider):
    """Anthropic实现"""
    
    async def generate(self, prompt: str) -> str:
        # Anthropic API调用
        pass

class Agent:
    """智能体依赖抽象接口"""
    
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider  # 依赖抽象
```

---

## 性能优化策略

### ⚡ 1. 异步编程

**使用异步I/O提高并发性能：**

```python
import asyncio
import aiohttp
from typing import List

async def parallel_tool_execution(tools: List[Tool], inputs: List[str]) -> List[Any]:
    """并行执行多个工具"""
    
    async def execute_tool(tool: Tool, input_data: str) -> Any:
        return await tool.execute(input_data)
    
    # 并行执行所有工具
    tasks = [execute_tool(tool, input_data) for tool, input_data in zip(tools, inputs)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return results

# 使用示例
async def main():
    tools = [calculator_tool, search_tool, file_tool]
    inputs = ["2+2", "AI agents", "read file.txt"]
    
    results = await parallel_tool_execution(tools, inputs)
    print(results)
```

### ⚡ 2. 缓存机制

**实现智能缓存减少重复计算：**

```python
import redis
import json
from typing import Any, Optional
import hashlib

class CacheManager:
    """缓存管理器"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.default_ttl = 3600  # 1小时
    
    def _generate_key(self, prefix: str, data: Any) -> str:
        """生成缓存键"""
        data_str = json.dumps(data, sort_keys=True)
        hash_obj = hashlib.md5(data_str.encode())
        return f"{prefix}:{hash_obj.hexdigest()}"
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        try:
            value = self.redis.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存"""
        try:
            ttl = ttl or self.default_ttl
            self.redis.setex(key, ttl, json.dumps(value))
            return True
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")
            return False
    
    async def cached_execution(self, func, *args, **kwargs) -> Any:
        """缓存函数执行结果"""
        cache_key = self._generate_key(func.__name__, (args, kwargs))
        
        # 尝试从缓存获取
        cached_result = await self.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # 执行函数并缓存结果
        result = await func(*args, **kwargs)
        await self.set(cache_key, result)
        
        return result
```

### ⚡ 3. 连接池管理

**使用连接池提高数据库性能：**

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
import asyncpg
import aioredis

class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, database_url: str):
        # PostgreSQL连接池
        self.pg_pool = None
        self.pg_url = database_url
        
        # Redis连接池
        self.redis_pool = None
    
    async def initialize(self):
        """初始化连接池"""
        # PostgreSQL连接池
        self.pg_pool = await asyncpg.create_pool(
            self.pg_url,
            min_size=5,
            max_size=20,
            command_timeout=60
        )
        
        # Redis连接池
        self.redis_pool = await aioredis.create_redis_pool(
            'redis://localhost:6379',
            minsize=5,
            maxsize=20
        )
    
    async def execute_query(self, query: str, *args) -> List[dict]:
        """执行查询"""
        async with self.pg_pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]
    
    async def cache_set(self, key: str, value: str, ttl: int = 3600):
        """设置Redis缓存"""
        await self.redis_pool.setex(key, ttl, value)
    
    async def cache_get(self, key: str) -> Optional[str]:
        """获取Redis缓存"""
        return await self.redis_pool.get(key)
```

---

## 安全防护措施

### 🛡️ 1. 输入验证

**实现严格的输入验证：**

```python
import re
from typing import Any
from pydantic import BaseModel, validator

class TaskInput(BaseModel):
    """任务输入验证模型"""
    
    task_description: str
    priority: int
    context: dict
    
    @validator('task_description')
    def validate_task_description(cls, v):
        if len(v) > 10000:
            raise ValueError('Task description too long')
        
        # 检查恶意模式
        malicious_patterns = [
            r'<script.*?>.*?</script>',
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__\s*\(',
        ]
        
        for pattern in malicious_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError(f'Malicious pattern detected: {pattern}')
        
        return v
    
    @validator('priority')
    def validate_priority(cls, v):
        if not 1 <= v <= 10:
            raise ValueError('Priority must be between 1 and 10')
        return v

# 使用验证
async def process_task_input(input_data: dict) -> TaskInput:
    """处理并验证任务输入"""
    try:
        validated_input = TaskInput(**input_data)
        return validated_input
    except Exception as e:
        logger.warning(f"Input validation failed: {e}")
        raise ValueError(f"Invalid input: {e}")
```

### 🛡️ 2. 权限控制

**实现细粒度权限控制：**

```python
from enum import Enum
from typing import Set, Dict, Any

class Permission(Enum):
    """权限枚举"""
    READ_MEMORY = "read_memory"
    WRITE_MEMORY = "write_memory"
    EXECUTE_TOOLS = "execute_tools"
    ACCESS_FILES = "access_files"
    NETWORK_ACCESS = "network_access"

class Role:
    """角色定义"""
    
    def __init__(self, name: str, permissions: Set[Permission]):
        self.name = name
        self.permissions = permissions

class SecurityManager:
    """安全管理器"""
    
    def __init__(self):
        self.roles = {
            "admin": Role("admin", set(Permission)),
            "user": Role("user", {Permission.READ_MEMORY, Permission.EXECUTE_TOOLS}),
            "guest": Role("guest", {Permission.READ_MEMORY}),
        }
        self.user_roles: Dict[str, str] = {}
    
    def assign_role(self, user_id: str, role_name: str):
        """分配角色"""
        if role_name not in self.roles:
            raise ValueError(f"Role {role_name} not found")
        self.user_roles[user_id] = role_name
    
    def check_permission(self, user_id: str, permission: Permission) -> bool:
        """检查权限"""
        user_role = self.user_roles.get(user_id)
        if not user_role:
            return False
        
        role = self.roles[user_role]
        return permission in role.permissions
    
    async def authorize_action(self, user_id: str, action: str, **kwargs) -> bool:
        """授权操作"""
        permission_map = {
            "read_memory": Permission.READ_MEMORY,
            "write_memory": Permission.WRITE_MEMORY,
            "execute_tool": Permission.EXECUTE_TOOLS,
            "access_file": Permission.ACCESS_FILES,
            "network_request": Permission.NETWORK_ACCESS,
        }
        
        permission = permission_map.get(action)
        if not permission:
            return False
        
        return self.check_permission(user_id, permission)
```

### 🛡️ 3. 安全护栏

**实现多层安全护栏：**

```python
class SecurityGuardrail:
    """安全护栏"""
    
    def __init__(self):
        self.content_filters = [
            self._filter_toxic_content,
            self._filter_personal_info,
            self._filter_sensitive_data,
        ]
    
    async def validate_output(self, output: str) -> tuple[bool, str]:
        """验证输出内容"""
        for filter_func in self.content_filters:
            is_safe, message = await filter_func(output)
            if not is_safe:
                return False, message
        
        return True, "Content is safe"
    
    async def _filter_toxic_content(self, content: str) -> tuple[bool, str]:
        """过滤有毒内容"""
        toxic_keywords = ["hate", "violence", "harassment"]
        
        content_lower = content.lower()
        for keyword in toxic_keywords:
            if keyword in content_lower:
                return False, f"Toxic content detected: {keyword}"
        
        return True, "No toxic content"
    
    async def _filter_personal_info(self, content: str) -> tuple[bool, str]:
        """过滤个人信息"""
        import re
        
        # 检查邮箱
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        if re.search(email_pattern, content):
            return False, "Personal email detected"
        
        # 检查电话号码
        phone_pattern = r'\b\d{3}-\d{3}-\d{4}\b'
        if re.search(phone_pattern, content):
            return False, "Phone number detected"
        
        return True, "No personal info"
    
    async def _filter_sensitive_data(self, content: str) -> tuple[bool, str]:
        """过滤敏感数据"""
        sensitive_patterns = [
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # 信用卡号
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        ]
        
        for pattern in sensitive_patterns:
            if re.search(pattern, content):
                return False, "Sensitive data detected"
        
        return True, "No sensitive data"
```

---

## 部署运维指南

### 🚀 1. 容器化部署

**使用Docker进行应用容器化：**

```dockerfile
# 多阶段构建优化镜像大小
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim

# 创建非root用户
RUN groupadd -r agent && useradd -r -g agent agent

WORKDIR /app

# 从builder阶段复制依赖
COPY --from=builder /root/.local /home/agent/.local
COPY --chown=agent:agent . .

# 切换到非root用户
USER agent

# 设置环境变量
ENV PATH=/home/agent/.local/bin:$PATH
ENV PYTHONPATH=/app

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

EXPOSE 8000

CMD ["python", "main.py"]
```

### 🚀 2. Kubernetes部署

**使用Kubernetes进行容器编排：**

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-deployment
  labels:
    app: agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent
  template:
    metadata:
      labels:
        app: agent
    spec:
      containers:
      - name: agent
        image: agent:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: database-url
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: agent-service
spec:
  selector:
    app: agent
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agent-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-deployment
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 🚀 3. CI/CD流水线

**使用GitHub Actions实现自动化部署：**

```yaml
# .github/workflows/deploy.yml
name: Deploy Agent

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest --cov=src tests/
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -t agent:${{ github.sha }} .
        docker tag agent:${{ github.sha }} agent:latest
    
    - name: Push to registry
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker push agent:${{ github.sha }}
        docker push agent:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - name: Deploy to Kubernetes
      run: |
        kubectl apply -f k8s/
        kubectl rollout restart deployment/agent-deployment
```

---

## 测试质量保证

### 🧪 1. 单元测试

**编写全面的单元测试：**

```python
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from agent_template import Agent, TaskPlanner, MemoryManager

class TestTaskPlanner:
    """任务规划器测试"""
    
    @pytest.fixture
    def planner(self):
        return TaskPlanner()
    
    @pytest.mark.asyncio
    async def test_decompose_task(self, planner):
        """测试任务分解"""
        task_description = "Analyze market trends"
        tasks = await planner.decompose_task(task_description)
        
        assert len(tasks) == 4
        assert all(task.description for task in tasks)
        assert all(task.priority > 0 for task in tasks)
    
    @pytest.mark.asyncio
    async def test_prioritize_tasks(self, planner):
        """测试任务优先级排序"""
        tasks = [
            Task("1", "Low priority", 1, "pending", {}, datetime.now(), datetime.now()),
            Task("2", "High priority", 10, "pending", {}, datetime.now(), datetime.now()),
            Task("3", "Medium priority", 5, "pending", {}, datetime.now(), datetime.now()),
        ]
        
        prioritized = await planner.prioritize_tasks(tasks)
        
        assert prioritized[0].priority == 10
        assert prioritized[1].priority == 5
        assert prioritized[2].priority == 1

class TestMemoryManager:
    """记忆管理器测试"""
    
    @pytest.fixture
    def memory_manager(self):
        return MemoryManager()
    
    @pytest.mark.asyncio
    async def test_store_retrieve_short_term(self, memory_manager):
        """测试短期记忆存储和检索"""
        await memory_manager.store_short_term("test_key", "test_value")
        
        context = await memory_manager.retrieve_context("test")
        assert "test_key" in context["short_term"]
        assert context["short_term"]["test_key"]["value"] == "test_value"
    
    @pytest.mark.asyncio
    async def test_memory_cleanup(self, memory_manager):
        """测试记忆清理"""
        # 添加超过限制的记忆
        for i in range(1100):
            await memory_manager.store_short_term(f"key_{i}", f"value_{i}")
        
        # 检查是否清理到合理范围
        assert len(memory_manager.memory.short_term) <= 1000

class TestAgent:
    """智能体测试"""
    
    @pytest.fixture
    def agent(self):
        return Agent("TestAgent")
    
    @pytest.mark.asyncio
    async def test_process_input(self, agent):
        """测试输入处理"""
        # Mock工具管理器
        agent.tool_manager.execute_tool = AsyncMock(return_value="test_result")
        
        result = await agent.process_input("Test input")
        
        assert result is not None
        assert "test_result" in result or "Error" in result
    
    @pytest.mark.asyncio
    async def test_security_validation(self, agent):
        """测试安全验证"""
        # 测试恶意输入
        malicious_input = "<script>alert('xss')</script>"
        result = await agent.process_input(malicious_input)
        
        assert "Input validation failed" in result
```

### 🧪 2. 集成测试

**编写集成测试验证模块协作：**

```python
import pytest
import asyncio
from agent_template import Agent, CalculatorTool, WebSearchTool

class TestAgentIntegration:
    """智能体集成测试"""
    
    @pytest.fixture
    async def agent_with_tools(self):
        """创建带工具的智能体"""
        agent = Agent("IntegrationTestAgent")
        
        # 注册工具
        calculator = CalculatorTool()
        web_search = WebSearchTool()
        agent.tool_manager.register_tool(calculator)
        agent.tool_manager.register_tool(web_search)
        
        return agent
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, agent_with_tools):
        """测试端到端工作流"""
        # 模拟完整的智能体工作流
        input_data = "Calculate 2+2 and search for AI agents"
        
        result = await agent_with_tools.process_input(input_data)
        
        # 验证结果
        assert result is not None
        assert len(result) > 0
        
        # 验证记忆是否被正确存储
        context = await agent_with_tools.memory_manager.retrieve_context("calculate")
        assert len(context["short_term"]) > 0 or len(context["long_term"]) > 0
    
    @pytest.mark.asyncio
    async def test_multi_agent_coordination(self):
        """测试多智能体协调"""
        from agent_template import MultiAgentCoordinator
        
        coordinator = MultiAgentCoordinator()
        
        # 创建多个智能体
        agent1 = Agent("Agent1")
        agent2 = Agent("Agent2")
        
        coordinator.register_agent("agent1", agent1)
        coordinator.register_agent("agent2", agent2)
        
        # 测试任务协调
        task = "Analyze market data"
        agent_ids = ["agent1", "agent2"]
        
        results = await coordinator.coordinate_task(task, agent_ids)
        
        assert len(results) == 2
        assert "agent1" in results
        assert "agent2" in results
```

### 🧪 3. 性能测试

**编写性能测试验证系统性能：**

```python
import pytest
import asyncio
import time
from agent_template import Agent

class TestPerformance:
    """性能测试"""
    
    @pytest.fixture
    def agent(self):
        return Agent("PerformanceTestAgent")
    
    @pytest.mark.asyncio
    async def test_response_time(self, agent):
        """测试响应时间"""
        start_time = time.time()
        
        result = await agent.process_input("Simple test input")
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # 响应时间应该在合理范围内
        assert response_time < 5.0  # 5秒内完成
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, agent):
        """测试并发请求处理"""
        async def make_request(request_id: int):
            return await agent.process_input(f"Request {request_id}")
        
        # 并发发送10个请求
        tasks = [make_request(i) for i in range(10)]
        start_time = time.time()
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 验证所有请求都成功
        assert len(results) == 10
        assert all(result is not None for result in results)
        
        # 并发处理应该比串行处理快
        assert total_time < 10.0  # 10个请求在10秒内完成
    
    @pytest.mark.asyncio
    async def test_memory_usage(self, agent):
        """测试内存使用"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # 执行大量操作
        for i in range(100):
            await agent.process_input(f"Memory test {i}")
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # 内存增长应该在合理范围内
        assert memory_increase < 100 * 1024 * 1024  # 100MB以内
```

---

## 监控告警体系

### 📊 1. 指标收集

**使用Prometheus收集关键指标：**

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# 定义监控指标
agent_requests_total = Counter('agent_requests_total', 'Total agent requests', ['agent_name', 'status'])
agent_request_duration = Histogram('agent_request_duration_seconds', 'Agent request duration')
agent_memory_usage = Gauge('agent_memory_usage_bytes', 'Agent memory usage')
agent_active_tasks = Gauge('agent_active_tasks', 'Number of active tasks')
agent_tool_executions = Counter('agent_tool_executions_total', 'Tool executions', ['tool_name', 'status'])

class AgentMonitor:
    """智能体监控器"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
    
    def record_request(self, duration: float, status: str = "success"):
        """记录请求指标"""
        agent_requests_total.labels(
            agent_name=self.agent_name,
            status=status
        ).inc()
        agent_request_duration.observe(duration)
    
    def update_memory_usage(self, usage: int):
        """更新内存使用量"""
        agent_memory_usage.set(usage)
    
    def update_active_tasks(self, count: int):
        """更新活跃任务数"""
        agent_active_tasks.set(count)
    
    def record_tool_execution(self, tool_name: str, status: str = "success"):
        """记录工具执行"""
        agent_tool_executions.labels(
            tool_name=tool_name,
            status=status
        ).inc()

# 启动监控服务器
def start_monitoring(port: int = 9090):
    """启动监控服务器"""
    start_http_server(port)
    print(f"Monitoring server started on port {port}")
```

### 📊 2. 日志聚合

**使用ELK Stack进行日志聚合：**

```python
import logging
import json
from datetime import datetime
from elasticsearch import Elasticsearch

class ElasticsearchHandler(logging.Handler):
    """Elasticsearch日志处理器"""
    
    def __init__(self, es_client: Elasticsearch, index_name: str):
        super().__init__()
        self.es_client = es_client
        self.index_name = index_name
    
    def emit(self, record):
        """发送日志到Elasticsearch"""
        try:
            log_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'level': record.levelname,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno,
            }
            
            # 添加额外字段
            if hasattr(record, 'agent_name'):
                log_entry['agent_name'] = record.agent_name
            if hasattr(record, 'task_id'):
                log_entry['task_id'] = record.task_id
            if hasattr(record, 'user_id'):
                log_entry['user_id'] = record.user_id
            
            # 发送到Elasticsearch
            self.es_client.index(
                index=self.index_name,
                body=log_entry
            )
        except Exception as e:
            print(f"Failed to send log to Elasticsearch: {e}")

# 配置日志
def setup_logging():
    """设置日志配置"""
    es_client = Elasticsearch(['localhost:9200'])
    
    # 创建Elasticsearch处理器
    es_handler = ElasticsearchHandler(es_client, 'agent-logs')
    es_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    es_handler.setFormatter(formatter)
    
    # 配置根日志器
    root_logger = logging.getLogger()
    root_logger.addHandler(es_handler)
    root_logger.setLevel(logging.INFO)
```

### 📊 3. 告警规则

**使用Grafana配置告警规则：**

```yaml
# grafana-alerts.yml
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-alerts
data:
  agent-alerts.yml: |
    groups:
    - name: agent-alerts
      rules:
      - alert: HighErrorRate
        expr: rate(agent_requests_total{status="error"}[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"
      
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(agent_request_duration_seconds_bucket[5m])) > 5
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "High response time detected"
          description: "95th percentile response time is {{ $value }} seconds"
      
      - alert: HighMemoryUsage
        expr: agent_memory_usage_bytes > 1000000000  # 1GB
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is {{ $value }} bytes"
      
      - alert: ToolExecutionFailure
        expr: rate(agent_tool_executions_total{status="error"}[5m]) > 0.05
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "High tool execution failure rate"
          description: "Tool execution failure rate is {{ $value }} failures per second"
```

---

## 🎯 总结

本最佳实践指南基于对30+个开源智能体框架的深度分析，提供了完整的开发、部署和运维指导。通过遵循这些最佳实践，您可以：

1. **提高开发效率**：使用标准化的开发模板和工具链
2. **确保代码质量**：通过完善的测试和代码审查流程
3. **优化系统性能**：使用异步编程、缓存和连接池等技术
4. **保障系统安全**：实施多层安全防护和权限控制
5. **简化部署运维**：使用容器化和自动化部署方案
6. **建立监控体系**：实现全面的监控、日志和告警

选择适合您项目需求的最佳实践，持续优化和改进您的智能体系统。

---

*本指南基于开源智能体项目分析报告，持续更新中。如有问题或建议，欢迎反馈。*
