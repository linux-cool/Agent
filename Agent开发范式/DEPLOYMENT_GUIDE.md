# Agent开发范式 - 综合部署指南

## 📋 目录

1. [环境准备](#环境准备)
2. [基础架构部署](#基础架构部署)
3. [智能体系统部署](#智能体系统部署)
4. [监控与运维](#监控与运维)
5. [安全配置](#安全配置)
6. [性能优化](#性能优化)
7. [故障排除](#故障排除)
8. [扩展与升级](#扩展与升级)

## 🚀 环境准备

### 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| CPU | 4核心 | 8核心+ |
| 内存 | 8GB | 16GB+ |
| 存储 | 100GB SSD | 500GB+ NVMe |
| 网络 | 1Gbps | 10Gbps+ |
| 操作系统 | Ubuntu 20.04+ | Ubuntu 22.04 LTS |

### 软件依赖

```bash
# 基础工具
sudo apt update && sudo apt install -y \
    curl \
    wget \
    git \
    vim \
    htop \
    tree \
    jq \
    unzip

# Python环境
sudo apt install -y python3.11 python3.11-pip python3.11-venv
sudo apt install -y python3.11-dev build-essential

# Docker环境
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Kubernetes (可选)
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
```

### 环境变量配置

```bash
# 创建环境配置文件
cat > .env << EOF
# 基础配置
PROJECT_NAME=agent-development-paradigm
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# 数据库配置
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=agent_db
POSTGRES_USER=agent_user
POSTGRES_PASSWORD=your_secure_password

# Redis配置
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=your_redis_password

# API配置
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# 安全配置
SECRET_KEY=your_super_secret_key_here
JWT_SECRET=your_jwt_secret_here
ENCRYPTION_KEY=your_encryption_key_here

# 外部服务
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
HUGGINGFACE_API_KEY=your_huggingface_api_key

# 监控配置
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
JAEGER_PORT=16686

# 存储配置
S3_BUCKET=agent-storage
S3_REGION=us-west-2
S3_ACCESS_KEY=your_s3_access_key
S3_SECRET_KEY=your_s3_secret_key
EOF
```

## 🏗️ 基础架构部署

### 1. 数据库部署

#### PostgreSQL配置

```yaml
# docker-compose.yml
version: '3.8'
services:
  postgres:
    image: postgres:15
    container_name: agent_postgres
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./config/postgres/postgresql.conf:/etc/postgresql/postgresql.conf
      - ./config/postgres/init.sql:/docker-entrypoint-initdb.d/init.sql
    command: postgres -c config_file=/etc/postgresql/postgresql.conf
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  postgres_data:
```

#### 数据库初始化脚本

```sql
-- config/postgres/init.sql
-- 创建扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- 创建用户表
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'
);

-- 创建对话表
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_id VARCHAR(100) NOT NULL,
    messages JSONB DEFAULT '[]',
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 创建任务表
CREATE TABLE IF NOT EXISTS tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(200) NOT NULL,
    description TEXT,
    status VARCHAR(20) DEFAULT 'pending',
    assigned_agent VARCHAR(100),
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    result JSONB,
    metadata JSONB DEFAULT '{}'
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_assigned_agent ON tasks(assigned_agent);

-- 创建全文搜索索引
CREATE INDEX IF NOT EXISTS idx_conversations_messages_gin ON conversations USING gin(messages);
CREATE INDEX IF NOT EXISTS idx_tasks_description_gin ON tasks USING gin(to_tsvector('english', description));
```

### 2. 缓存系统部署

#### Redis配置

```yaml
# docker-compose.yml (Redis部分)
  redis:
    image: redis:7-alpine
    container_name: agent_redis
    command: redis-server --requirepass ${REDIS_PASSWORD} --appendonly yes
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
      - ./config/redis/redis.conf:/usr/local/etc/redis/redis.conf
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  redis_data:
```

#### Redis配置文件

```conf
# config/redis/redis.conf
# 基础配置
bind 0.0.0.0
port 6379
timeout 300
tcp-keepalive 60

# 内存配置
maxmemory 2gb
maxmemory-policy allkeys-lru

# 持久化配置
save 900 1
save 300 10
save 60 10000
appendonly yes
appendfsync everysec

# 日志配置
loglevel notice
logfile ""

# 安全配置
requirepass your_redis_password
```

### 3. 消息队列部署

#### RabbitMQ配置

```yaml
# docker-compose.yml (RabbitMQ部分)
  rabbitmq:
    image: rabbitmq:3-management
    container_name: agent_rabbitmq
    environment:
      RABBITMQ_DEFAULT_USER: admin
      RABBITMQ_DEFAULT_PASS: your_rabbitmq_password
    ports:
      - "5672:5672"
      - "15672:15672"
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "rabbitmq-diagnostics", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  rabbitmq_data:
```

## 🤖 智能体系统部署

### 1. 核心智能体服务

#### 基础智能体服务

```yaml
# docker-compose.yml (智能体服务部分)
  agent-core:
    build:
      context: ./Agent开发范式/第1章-智能体架构设计原理
      dockerfile: Dockerfile
    container_name: agent_core
    environment:
      - DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    ports:
      - "8001:8000"
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

#### 多智能体协作服务

```yaml
  multi-agent:
    build:
      context: ./Agent开发范式/第3章-多智能体系统协作机制
      dockerfile: Dockerfile
    container_name: multi_agent
    environment:
      - DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
      - RABBITMQ_URL=amqp://admin:${RABBITMQ_PASSWORD}@rabbitmq:5672/
    ports:
      - "8002:8000"
    depends_on:
      - postgres
      - redis
      - rabbitmq
    restart: unless-stopped
```

#### 记忆推理服务

```yaml
  memory-reasoning:
    build:
      context: ./Agent开发范式/第4章-记忆与推理系统构建
      dockerfile: Dockerfile
    container_name: memory_reasoning
    environment:
      - DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
      - VECTOR_DB_URL=chroma://localhost:8000
    ports:
      - "8003:8000"
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
```

### 2. 企业级应用服务

#### 智能客服系统

```yaml
  customer-service:
    build:
      context: ./Agent开发范式/第6章-企业级智能体应用
      dockerfile: Dockerfile.customer-service
    container_name: customer_service
    environment:
      - DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - SECRET_KEY=${SECRET_KEY}
    ports:
      - "8004:8000"
    depends_on:
      - postgres
      - redis
      - agent-core
    restart: unless-stopped
```

#### 代码助手服务

```yaml
  code-assistant:
    build:
      context: ./Agent开发范式/第6章-企业级智能体应用
      dockerfile: Dockerfile.code-assistant
    container_name: code_assistant
    environment:
      - DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - GITHUB_TOKEN=${GITHUB_TOKEN}
    ports:
      - "8005:8000"
    depends_on:
      - postgres
      - redis
      - agent-core
    restart: unless-stopped
```

### 3. API网关配置

#### Nginx配置

```nginx
# config/nginx/nginx.conf
upstream agent_core {
    server agent-core:8000;
}

upstream multi_agent {
    server multi-agent:8000;
}

upstream memory_reasoning {
    server memory-reasoning:8000;
}

upstream customer_service {
    server customer-service:8000;
}

upstream code_assistant {
    server code-assistant:8000;
}

server {
    listen 80;
    server_name your-domain.com;
    
    # 安全头
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    # 限流
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
    
    # 代理配置
    location /api/v1/core/ {
        proxy_pass http://agent_core/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /api/v1/multi-agent/ {
        proxy_pass http://multi_agent/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /api/v1/memory/ {
        proxy_pass http://memory_reasoning/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /api/v1/customer-service/ {
        proxy_pass http://customer_service/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /api/v1/code-assistant/ {
        proxy_pass http://code_assistant/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # 健康检查
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
```

## 📊 监控与运维

### 1. Prometheus监控

#### Prometheus配置

```yaml
# docker-compose.yml (监控部分)
  prometheus:
    image: prom/prometheus:latest
    container_name: agent_prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    restart: unless-stopped

volumes:
  prometheus_data:
```

#### Prometheus配置文件

```yaml
# config/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'agent-core'
    static_configs:
      - targets: ['agent-core:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'multi-agent'
    static_configs:
      - targets: ['multi-agent:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'memory-reasoning'
    static_configs:
      - targets: ['memory-reasoning:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'customer-service'
    static_configs:
      - targets: ['customer-service:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'code-assistant'
    static_configs:
      - targets: ['code-assistant:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'rabbitmq'
    static_configs:
      - targets: ['rabbitmq:15692']
```

### 2. Grafana可视化

#### Grafana配置

```yaml
  grafana:
    image: grafana/grafana:latest
    container_name: agent_grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./config/grafana/provisioning:/etc/grafana/provisioning
      - ./config/grafana/dashboards:/var/lib/grafana/dashboards
    restart: unless-stopped

volumes:
  grafana_data:
```

### 3. 日志管理

#### ELK Stack配置

```yaml
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.8.0
    container_name: agent_elasticsearch
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    restart: unless-stopped

  logstash:
    image: docker.elastic.co/logstash/logstash:8.8.0
    container_name: agent_logstash
    volumes:
      - ./config/logstash/logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    ports:
      - "5044:5044"
    depends_on:
      - elasticsearch
    restart: unless-stopped

  kibana:
    image: docker.elastic.co/kibana/kibana:8.8.0
    container_name: agent_kibana
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch
    restart: unless-stopped

volumes:
  elasticsearch_data:
```

## 🔒 安全配置

### 1. SSL/TLS配置

#### 证书生成

```bash
# 生成自签名证书（生产环境请使用CA签发的证书）
mkdir -p config/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout config/ssl/private.key \
    -out config/ssl/certificate.crt \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=your-domain.com"
```

#### Nginx SSL配置

```nginx
# config/nginx/ssl.conf
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/nginx/ssl/certificate.crt;
    ssl_certificate_key /etc/nginx/ssl/private.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # 其他配置...
}

# HTTP重定向到HTTPS
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}
```

### 2. 防火墙配置

#### UFW配置

```bash
# 启用UFW
sudo ufw enable

# 允许SSH
sudo ufw allow 22

# 允许HTTP和HTTPS
sudo ufw allow 80
sudo ufw allow 443

# 允许内部通信
sudo ufw allow from 10.0.0.0/8
sudo ufw allow from 172.16.0.0/12
sudo ufw allow from 192.168.0.0/16

# 拒绝其他所有连接
sudo ufw default deny incoming
sudo ufw default allow outgoing
```

### 3. 访问控制

#### 用户认证配置

```python
# config/security/auth.py
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta

security = HTTPBearer()

class AuthManager:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
    
    def create_access_token(self, data: dict) -> str:
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> dict:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    auth_manager = AuthManager("your-secret-key")
    payload = auth_manager.verify_token(credentials.credentials)
    return payload
```

## ⚡ 性能优化

### 1. 数据库优化

#### PostgreSQL优化配置

```conf
# config/postgres/postgresql.conf
# 内存配置
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB

# 连接配置
max_connections = 100
shared_preload_libraries = 'pg_stat_statements'

# 日志配置
log_destination = 'stderr'
logging_collector = on
log_directory = 'pg_log'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_rotation_age = 1d
log_rotation_size = 100MB
log_min_duration_statement = 1000
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '

# 查询优化
random_page_cost = 1.1
effective_io_concurrency = 200
```

### 2. Redis优化

#### Redis优化配置

```conf
# config/redis/redis.conf
# 内存优化
maxmemory 2gb
maxmemory-policy allkeys-lru
maxmemory-samples 5

# 网络优化
tcp-keepalive 300
timeout 0

# 持久化优化
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes

# AOF优化
appendonly yes
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
```

### 3. 应用优化

#### Python应用优化

```python
# config/performance/gunicorn.conf.py
bind = "0.0.0.0:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
preload_app = True
keepalive = 2
timeout = 30
graceful_timeout = 30
```

## 🚨 故障排除

### 1. 常见问题

#### 数据库连接问题

```bash
# 检查PostgreSQL状态
docker exec -it agent_postgres pg_isready -U agent_user -d agent_db

# 查看PostgreSQL日志
docker logs agent_postgres

# 连接测试
docker exec -it agent_postgres psql -U agent_user -d agent_db -c "SELECT version();"
```

#### Redis连接问题

```bash
# 检查Redis状态
docker exec -it agent_redis redis-cli ping

# 查看Redis日志
docker logs agent_redis

# 连接测试
docker exec -it agent_redis redis-cli -a your_redis_password ping
```

#### 服务健康检查

```bash
# 检查所有服务状态
docker-compose ps

# 查看服务日志
docker-compose logs -f service_name

# 重启服务
docker-compose restart service_name

# 重新构建并启动
docker-compose up --build -d
```

### 2. 监控告警

#### 告警规则配置

```yaml
# config/prometheus/alert_rules.yml
groups:
  - name: agent_alerts
    rules:
      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service {{ $labels.instance }} is down"
          description: "Service {{ $labels.instance }} has been down for more than 1 minute."
      
      - alert: HighCPUUsage
        expr: 100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage on {{ $labels.instance }}"
          description: "CPU usage is above 80% for more than 5 minutes."
      
      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100 > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage on {{ $labels.instance }}"
          description: "Memory usage is above 80% for more than 5 minutes."
```

## 📈 扩展与升级

### 1. 水平扩展

#### Kubernetes部署

```yaml
# k8s/agent-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-core
  labels:
    app: agent-core
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent-core
  template:
    metadata:
      labels:
        app: agent-core
    spec:
      containers:
      - name: agent-core
        image: agent-core:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: redis-url
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
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
  name: agent-core-service
spec:
  selector:
    app: agent-core
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### 2. 版本升级

#### 滚动升级脚本

```bash
#!/bin/bash
# scripts/upgrade.sh

set -e

echo "开始升级智能体系统..."

# 备份当前版本
echo "备份当前版本..."
docker-compose exec postgres pg_dump -U agent_user agent_db > backup_$(date +%Y%m%d_%H%M%S).sql

# 拉取最新镜像
echo "拉取最新镜像..."
docker-compose pull

# 停止服务
echo "停止服务..."
docker-compose stop

# 启动服务
echo "启动服务..."
docker-compose up -d

# 等待服务就绪
echo "等待服务就绪..."
sleep 30

# 健康检查
echo "执行健康检查..."
curl -f http://localhost:8000/health || exit 1

echo "升级完成！"
```

### 3. 数据迁移

#### 数据库迁移脚本

```python
# scripts/migrate.py
import asyncio
import asyncpg
from datetime import datetime

async def migrate_database():
    """数据库迁移"""
    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        user="agent_user",
        password="your_password",
        database="agent_db"
    )
    
    try:
        # 创建新表
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS new_users (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                hashed_password VARCHAR(255) NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                last_login TIMESTAMP WITH TIME ZONE,
                metadata JSONB DEFAULT '{}',
                version INTEGER DEFAULT 1
            );
        """)
        
        # 迁移数据
        await conn.execute("""
            INSERT INTO new_users (id, username, email, hashed_password, is_active, created_at, last_login, metadata)
            SELECT id, username, email, hashed_password, is_active, created_at, last_login, metadata
            FROM users
            ON CONFLICT (id) DO NOTHING;
        """)
        
        # 重命名表
        await conn.execute("ALTER TABLE users RENAME TO users_old;")
        await conn.execute("ALTER TABLE new_users RENAME TO users;")
        
        print("数据库迁移完成！")
        
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(migrate_database())
```

## 📋 部署检查清单

### 部署前检查

- [ ] 环境变量配置正确
- [ ] 数据库连接测试通过
- [ ] Redis连接测试通过
- [ ] 外部API密钥配置正确
- [ ] SSL证书配置完成
- [ ] 防火墙规则配置正确
- [ ] 监控系统配置完成
- [ ] 日志系统配置完成
- [ ] 备份策略配置完成

### 部署后检查

- [ ] 所有服务正常运行
- [ ] 健康检查通过
- [ ] 监控指标正常
- [ ] 日志记录正常
- [ ] 性能测试通过
- [ ] 安全扫描通过
- [ ] 用户认证正常
- [ ] API接口正常
- [ ] 数据库性能正常

### 运维检查

- [ ] 定期备份数据
- [ ] 监控系统告警
- [ ] 日志分析
- [ ] 性能优化
- [ ] 安全更新
- [ ] 容量规划
- [ ] 故障演练
- [ ] 文档更新

---

## 📞 技术支持

如果在部署过程中遇到问题，请：

1. 查看相关日志文件
2. 检查配置文件
3. 参考故障排除章节
4. 提交Issue到GitHub仓库
5. 联系技术支持团队

---

*本部署指南将根据系统更新持续维护，请定期查看最新版本。*
