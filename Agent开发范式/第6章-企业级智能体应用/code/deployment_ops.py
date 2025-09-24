# deployment_ops.py
"""
第6章 企业级智能体应用 - 部署与运维
实现企业级智能体应用的部署、监控、运维管理等功能
"""

import asyncio
import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Set
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
import threading
import time
from collections import defaultdict, deque
import re
import subprocess
import os
import yaml
import psutil
import socket

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeploymentStatus(Enum):
    """部署状态枚举"""
    PENDING = "待部署"
    DEPLOYING = "部署中"
    DEPLOYED = "已部署"
    FAILED = "部署失败"
    ROLLING_BACK = "回滚中"
    ROLLED_BACK = "已回滚"

class ServiceStatus(Enum):
    """服务状态枚举"""
    HEALTHY = "健康"
    UNHEALTHY = "不健康"
    STARTING = "启动中"
    STOPPING = "停止中"
    STOPPED = "已停止"
    CRASHED = "崩溃"

class AlertLevel(Enum):
    """告警级别枚举"""
    INFO = "信息"
    WARNING = "警告"
    ERROR = "错误"
    CRITICAL = "严重"

class DeploymentStrategy(Enum):
    """部署策略枚举"""
    BLUE_GREEN = "蓝绿部署"
    ROLLING = "滚动部署"
    CANARY = "金丝雀部署"
    RECREATE = "重新创建"

class EnvironmentType(Enum):
    """环境类型枚举"""
    DEVELOPMENT = "开发环境"
    TESTING = "测试环境"
    STAGING = "预发布环境"
    PRODUCTION = "生产环境"

@dataclass
class Application:
    """应用程序"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    version: str = ""
    description: str = ""
    image: str = ""
    ports: List[int] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)
    health_check: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "image": self.image,
            "ports": self.ports,
            "environment_variables": self.environment_variables,
            "resources": self.resources,
            "health_check": self.health_check,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

@dataclass
class Deployment:
    """部署"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    application_id: str = ""
    environment: EnvironmentType = EnvironmentType.DEVELOPMENT
    strategy: DeploymentStrategy = DeploymentStrategy.ROLLING
    status: DeploymentStatus = DeploymentStatus.PENDING
    replicas: int = 1
    config: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    deployed_by: str = ""
    rollback_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "application_id": self.application_id,
            "environment": self.environment.value,
            "strategy": self.strategy.value,
            "status": self.status.value,
            "replicas": self.replicas,
            "config": self.config,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "deployed_by": self.deployed_by,
            "rollback_version": self.rollback_version
        }

@dataclass
class Service:
    """服务"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    application_id: str = ""
    deployment_id: str = ""
    status: ServiceStatus = ServiceStatus.STOPPED
    replicas: int = 0
    running_replicas: int = 0
    ports: List[int] = field(default_factory=list)
    endpoints: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "application_id": self.application_id,
            "deployment_id": self.deployment_id,
            "status": self.status.value,
            "replicas": self.replicas,
            "running_replicas": self.running_replicas,
            "ports": self.ports,
            "endpoints": self.endpoints,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

@dataclass
class Metric:
    """指标"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    service_id: str = ""
    name: str = ""
    value: float = 0.0
    unit: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "service_id": self.service_id,
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels
        }

@dataclass
class Alert:
    """告警"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    service_id: str = ""
    name: str = ""
    level: AlertLevel = AlertLevel.WARNING
    message: str = ""
    status: str = "active"  # active, resolved, acknowledged
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "service_id": self.service_id,
            "name": self.name,
            "level": self.level.value,
            "message": self.message,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "acknowledged_by": self.acknowledged_by
        }

class ContainerManager:
    """容器管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.docker_client = None
        self.kubernetes_client = None
        self._initialize_clients()
    
    def _initialize_clients(self):
        """初始化客户端"""
        # 在实际实现中，这里会初始化Docker和Kubernetes客户端
        logger.info("Container clients initialized")
    
    async def build_image(self, app: Application, build_context: str = ".") -> bool:
        """构建镜像"""
        try:
            logger.info(f"Building image for {app.name}:{app.version}")
            
            # 模拟镜像构建
            await asyncio.sleep(1.0)
            
            # 在实际实现中，这里会调用Docker API构建镜像
            logger.info(f"Image built successfully: {app.image}")
            return True
            
        except Exception as e:
            logger.error(f"Image build failed: {e}")
            return False
    
    async def push_image(self, app: Application, registry: str = "localhost:5000") -> bool:
        """推送镜像"""
        try:
            logger.info(f"Pushing image {app.image} to registry {registry}")
            
            # 模拟镜像推送
            await asyncio.sleep(0.5)
            
            logger.info(f"Image pushed successfully to {registry}")
            return True
            
        except Exception as e:
            logger.error(f"Image push failed: {e}")
            return False
    
    async def deploy_service(self, deployment: Deployment, app: Application) -> bool:
        """部署服务"""
        try:
            logger.info(f"Deploying service for {app.name} using {deployment.strategy.value}")
            
            # 根据部署策略执行部署
            if deployment.strategy == DeploymentStrategy.BLUE_GREEN:
                return await self._deploy_blue_green(deployment, app)
            elif deployment.strategy == DeploymentStrategy.ROLLING:
                return await self._deploy_rolling(deployment, app)
            elif deployment.strategy == DeploymentStrategy.CANARY:
                return await self._deploy_canary(deployment, app)
            else:
                return await self._deploy_recreate(deployment, app)
                
        except Exception as e:
            logger.error(f"Service deployment failed: {e}")
            return False
    
    async def _deploy_blue_green(self, deployment: Deployment, app: Application) -> bool:
        """蓝绿部署"""
        logger.info("Executing blue-green deployment")
        
        # 1. 部署新版本（绿色环境）
        await asyncio.sleep(0.5)
        logger.info("Green environment deployed")
        
        # 2. 健康检查
        await asyncio.sleep(0.2)
        logger.info("Health check passed")
        
        # 3. 切换流量
        await asyncio.sleep(0.3)
        logger.info("Traffic switched to green environment")
        
        # 4. 清理旧版本（蓝色环境）
        await asyncio.sleep(0.2)
        logger.info("Blue environment cleaned up")
        
        return True
    
    async def _deploy_rolling(self, deployment: Deployment, app: Application) -> bool:
        """滚动部署"""
        logger.info("Executing rolling deployment")
        
        # 逐步替换实例
        for i in range(deployment.replicas):
            await asyncio.sleep(0.3)
            logger.info(f"Replaced replica {i+1}/{deployment.replicas}")
        
        return True
    
    async def _deploy_canary(self, deployment: Deployment, app: Application) -> bool:
        """金丝雀部署"""
        logger.info("Executing canary deployment")
        
        # 1. 部署少量实例
        await asyncio.sleep(0.3)
        logger.info("Canary instances deployed")
        
        # 2. 监控指标
        await asyncio.sleep(0.5)
        logger.info("Monitoring canary metrics")
        
        # 3. 逐步扩大范围
        await asyncio.sleep(0.4)
        logger.info("Gradually expanding canary deployment")
        
        # 4. 全量部署
        await asyncio.sleep(0.3)
        logger.info("Full deployment completed")
        
        return True
    
    async def _deploy_recreate(self, deployment: Deployment, app: Application) -> bool:
        """重新创建部署"""
        logger.info("Executing recreate deployment")
        
        # 1. 停止所有实例
        await asyncio.sleep(0.2)
        logger.info("Stopped all instances")
        
        # 2. 部署新实例
        await asyncio.sleep(0.5)
        logger.info("Deployed new instances")
        
        return True
    
    async def rollback_deployment(self, deployment: Deployment, app: Application) -> bool:
        """回滚部署"""
        try:
            logger.info(f"Rolling back deployment {deployment.id}")
            
            # 回滚到指定版本
            await asyncio.sleep(0.5)
            
            logger.info(f"Deployment rolled back to version {deployment.rollback_version}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    async def scale_service(self, service_id: str, replicas: int) -> bool:
        """扩缩容服务"""
        try:
            logger.info(f"Scaling service {service_id} to {replicas} replicas")
            
            # 模拟扩缩容
            await asyncio.sleep(0.3)
            
            logger.info(f"Service scaled successfully to {replicas} replicas")
            return True
            
        except Exception as e:
            logger.error(f"Service scaling failed: {e}")
            return False

class MonitoringSystem:
    """监控系统"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics: Dict[str, List[Metric]] = defaultdict(list)
        self.alerts: Dict[str, List[Alert]] = defaultdict(list)
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.running = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self._initialize_alert_rules()
    
    def _initialize_alert_rules(self):
        """初始化告警规则"""
        self.alert_rules = {
            "high_cpu": {
                "condition": "cpu_usage > 80",
                "level": AlertLevel.WARNING,
                "message": "CPU使用率过高"
            },
            "high_memory": {
                "condition": "memory_usage > 90",
                "level": AlertLevel.ERROR,
                "message": "内存使用率过高"
            },
            "service_down": {
                "condition": "status != 'healthy'",
                "level": AlertLevel.CRITICAL,
                "message": "服务不可用"
            },
            "high_response_time": {
                "condition": "response_time > 5000",
                "level": AlertLevel.WARNING,
                "message": "响应时间过长"
            }
        }
    
    async def start(self):
        """启动监控系统"""
        self.running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("MonitoringSystem started")
    
    async def stop(self):
        """停止监控系统"""
        self.running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("MonitoringSystem stopped")
    
    async def _monitoring_loop(self):
        """监控主循环"""
        while self.running:
            try:
                await self._collect_metrics()
                await self._check_alerts()
                await asyncio.sleep(self.config.get("monitoring_interval", 30))
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)
    
    async def _collect_metrics(self):
        """收集指标"""
        try:
            # 收集系统指标
            cpu_usage = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # 创建系统指标
            system_metrics = [
                Metric(
                    service_id="system",
                    name="cpu_usage",
                    value=cpu_usage,
                    unit="percent"
                ),
                Metric(
                    service_id="system",
                    name="memory_usage",
                    value=memory.percent,
                    unit="percent"
                ),
                Metric(
                    service_id="system",
                    name="disk_usage",
                    value=disk.percent,
                    unit="percent"
                )
            ]
            
            # 存储指标
            for metric in system_metrics:
                self.metrics[metric.service_id].append(metric)
                
                # 保持最近1000个指标
                if len(self.metrics[metric.service_id]) > 1000:
                    self.metrics[metric.service_id] = self.metrics[metric.service_id][-1000:]
            
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
    
    async def _check_alerts(self):
        """检查告警"""
        try:
            for service_id, metrics_list in self.metrics.items():
                if not metrics_list:
                    continue
                
                # 获取最新指标
                latest_metrics = {}
                for metric in metrics_list[-10:]:  # 最近10个指标
                    latest_metrics[metric.name] = metric.value
                
                # 检查告警规则
                for rule_name, rule in self.alert_rules.items():
                    if await self._evaluate_alert_condition(rule["condition"], latest_metrics):
                        await self._create_alert(service_id, rule_name, rule)
                        
        except Exception as e:
            logger.error(f"Alert checking failed: {e}")
    
    async def _evaluate_alert_condition(self, condition: str, metrics: Dict[str, float]) -> bool:
        """评估告警条件"""
        try:
            # 替换指标值
            evaluated_condition = condition
            for metric_name, value in metrics.items():
                evaluated_condition = evaluated_condition.replace(metric_name, str(value))
            
            # 简化的条件评估
            if ">" in evaluated_condition:
                left, right = evaluated_condition.split(">")
                return float(left.strip()) > float(right.strip())
            elif "<" in evaluated_condition:
                left, right = evaluated_condition.split("<")
                return float(left.strip()) < float(right.strip())
            elif "==" in evaluated_condition:
                left, right = evaluated_condition.split("==")
                return left.strip() == right.strip()
            elif "!=" in evaluated_condition:
                left, right = evaluated_condition.split("!=")
                return left.strip() != right.strip()
            
            return False
            
        except Exception as e:
            logger.error(f"Alert condition evaluation failed: {e}")
            return False
    
    async def _create_alert(self, service_id: str, rule_name: str, rule: Dict[str, Any]):
        """创建告警"""
        try:
            # 检查是否已存在活跃告警
            existing_alerts = self.alerts.get(service_id, [])
            active_alert = next(
                (alert for alert in existing_alerts 
                 if alert.name == rule_name and alert.status == "active"), 
                None
            )
            
            if active_alert:
                return  # 告警已存在
            
            # 创建新告警
            alert = Alert(
                service_id=service_id,
                name=rule_name,
                level=rule["level"],
                message=rule["message"]
            )
            
            self.alerts[service_id].append(alert)
            logger.warning(f"Alert created: {alert.name} - {alert.message}")
            
        except Exception as e:
            logger.error(f"Alert creation failed: {e}")
    
    async def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """解决告警"""
        try:
            for service_id, alerts in self.alerts.items():
                for alert in alerts:
                    if alert.id == alert_id:
                        alert.status = "resolved"
                        alert.resolved_at = datetime.now()
                        alert.acknowledged_by = resolved_by
                        logger.info(f"Alert resolved: {alert.name}")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Alert resolution failed: {e}")
            return False
    
    def get_metrics(self, service_id: str, metric_name: str = None, limit: int = 100) -> List[Metric]:
        """获取指标"""
        metrics = self.metrics.get(service_id, [])
        
        if metric_name:
            metrics = [m for m in metrics if m.name == metric_name]
        
        return metrics[-limit:] if limit else metrics
    
    def get_alerts(self, service_id: str = None, status: str = None) -> List[Alert]:
        """获取告警"""
        all_alerts = []
        
        if service_id:
            alerts = self.alerts.get(service_id, [])
        else:
            alerts = []
            for service_alerts in self.alerts.values():
                alerts.extend(service_alerts)
        
        if status:
            alerts = [alert for alert in alerts if alert.status == status]
        
        return alerts

class LogManager:
    """日志管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.log_storage = config.get("log_storage", "file")
        self.log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        self.log_files: Dict[str, str] = {}
    
    async def collect_logs(self, service_id: str, log_level: str = "INFO") -> List[Dict[str, Any]]:
        """收集日志"""
        try:
            # 模拟日志收集
            logs = []
            for i in range(10):
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "level": log_level,
                    "service_id": service_id,
                    "message": f"Sample log message {i+1}",
                    "source": "application"
                }
                logs.append(log_entry)
            
            return logs
            
        except Exception as e:
            logger.error(f"Log collection failed: {e}")
            return []
    
    async def search_logs(self, query: str, service_id: str = None, 
                         start_time: datetime = None, end_time: datetime = None) -> List[Dict[str, Any]]:
        """搜索日志"""
        try:
            # 模拟日志搜索
            logs = []
            for i in range(5):
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "level": "INFO",
                    "service_id": service_id or "unknown",
                    "message": f"Log entry containing '{query}' - {i+1}",
                    "source": "application"
                }
                logs.append(log_entry)
            
            return logs
            
        except Exception as e:
            logger.error(f"Log search failed: {e}")
            return []

class DeploymentOpsSystem:
    """部署运维系统"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.container_manager = ContainerManager(config.get("container", {}))
        self.monitoring_system = MonitoringSystem(config.get("monitoring", {}))
        self.log_manager = LogManager(config.get("logging", {}))
        self.applications: Dict[str, Application] = {}
        self.deployments: Dict[str, Deployment] = {}
        self.services: Dict[str, Service] = {}
    
    async def start(self):
        """启动系统"""
        await self.monitoring_system.start()
        logger.info("DeploymentOpsSystem started")
    
    async def stop(self):
        """停止系统"""
        await self.monitoring_system.stop()
        logger.info("DeploymentOpsSystem stopped")
    
    async def register_application(self, app: Application) -> bool:
        """注册应用程序"""
        try:
            self.applications[app.id] = app
            logger.info(f"Application registered: {app.name}")
            return True
        except Exception as e:
            logger.error(f"Application registration failed: {e}")
            return False
    
    async def deploy_application(self, app_id: str, environment: EnvironmentType, 
                               strategy: DeploymentStrategy = DeploymentStrategy.ROLLING,
                               deployed_by: str = "system") -> Optional[Deployment]:
        """部署应用程序"""
        try:
            app = self.applications.get(app_id)
            if not app:
                logger.error(f"Application not found: {app_id}")
                return None
            
            # 创建部署
            deployment = Deployment(
                application_id=app_id,
                environment=environment,
                strategy=strategy,
                deployed_by=deployed_by
            )
            
            self.deployments[deployment.id] = deployment
            
            # 开始部署
            deployment.status = DeploymentStatus.DEPLOYING
            deployment.started_at = datetime.now()
            
            # 构建镜像
            if not await self.container_manager.build_image(app):
                deployment.status = DeploymentStatus.FAILED
                return deployment
            
            # 推送镜像
            if not await self.container_manager.push_image(app):
                deployment.status = DeploymentStatus.FAILED
                return deployment
            
            # 部署服务
            if await self.container_manager.deploy_service(deployment, app):
                deployment.status = DeploymentStatus.DEPLOYED
                deployment.completed_at = datetime.now()
                
                # 创建服务
                service = Service(
                    name=f"{app.name}-{environment.value.lower()}",
                    application_id=app_id,
                    deployment_id=deployment.id,
                    status=ServiceStatus.HEALTHY,
                    replicas=deployment.replicas,
                    running_replicas=deployment.replicas,
                    ports=app.ports
                )
                self.services[service.id] = service
                
                logger.info(f"Application deployed successfully: {app.name}")
            else:
                deployment.status = DeploymentStatus.FAILED
            
            return deployment
            
        except Exception as e:
            logger.error(f"Application deployment failed: {e}")
            return None
    
    async def rollback_deployment(self, deployment_id: str) -> bool:
        """回滚部署"""
        try:
            deployment = self.deployments.get(deployment_id)
            if not deployment:
                logger.error(f"Deployment not found: {deployment_id}")
                return False
            
            app = self.applications.get(deployment.application_id)
            if not app:
                logger.error(f"Application not found: {deployment.application_id}")
                return False
            
            deployment.status = DeploymentStatus.ROLLING_BACK
            
            if await self.container_manager.rollback_deployment(deployment, app):
                deployment.status = DeploymentStatus.ROLLED_BACK
                logger.info(f"Deployment rolled back: {deployment_id}")
                return True
            else:
                deployment.status = DeploymentStatus.FAILED
                return False
                
        except Exception as e:
            logger.error(f"Deployment rollback failed: {e}")
            return False
    
    async def scale_service(self, service_id: str, replicas: int) -> bool:
        """扩缩容服务"""
        try:
            service = self.services.get(service_id)
            if not service:
                logger.error(f"Service not found: {service_id}")
                return False
            
            if await self.container_manager.scale_service(service_id, replicas):
                service.replicas = replicas
                service.running_replicas = replicas
                service.updated_at = datetime.now()
                logger.info(f"Service scaled: {service_id} to {replicas} replicas")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Service scaling failed: {e}")
            return False
    
    def get_deployment_status(self, deployment_id: str) -> Optional[Deployment]:
        """获取部署状态"""
        return self.deployments.get(deployment_id)
    
    def get_service_status(self, service_id: str) -> Optional[Service]:
        """获取服务状态"""
        return self.services.get(service_id)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统指标"""
        return {
            "total_applications": len(self.applications),
            "total_deployments": len(self.deployments),
            "total_services": len(self.services),
            "active_deployments": len([d for d in self.deployments.values() if d.status == DeploymentStatus.DEPLOYED]),
            "healthy_services": len([s for s in self.services.values() if s.status == ServiceStatus.HEALTHY])
        }
    
    def get_alerts_summary(self) -> Dict[str, Any]:
        """获取告警摘要"""
        all_alerts = []
        for alerts in self.monitoring_system.alerts.values():
            all_alerts.extend(alerts)
        
        active_alerts = [alert for alert in all_alerts if alert.status == "active"]
        
        return {
            "total_alerts": len(all_alerts),
            "active_alerts": len(active_alerts),
            "critical_alerts": len([a for a in active_alerts if a.level == AlertLevel.CRITICAL]),
            "error_alerts": len([a for a in active_alerts if a.level == AlertLevel.ERROR]),
            "warning_alerts": len([a for a in active_alerts if a.level == AlertLevel.WARNING])
        }

# 示例用法
async def main_demo():
    """部署运维演示"""
    config = {
        "container": {},
        "monitoring": {"monitoring_interval": 5},
        "logging": {"log_storage": "file"}
    }
    
    # 创建部署运维系统
    ops_system = DeploymentOpsSystem(config)
    await ops_system.start()
    
    print("🚀 部署运维系统演示")
    print("=" * 50)
    
    # 1. 注册应用程序
    print("\n1. 注册应用程序...")
    app = Application(
        name="智能客服系统",
        version="1.0.0",
        description="基于AI的智能客服系统",
        image="customer-service:1.0.0",
        ports=[8080, 8081],
        environment_variables={
            "ENV": "production",
            "LOG_LEVEL": "INFO"
        },
        resources={
            "cpu": "500m",
            "memory": "1Gi"
        },
        health_check={
            "path": "/health",
            "port": 8080,
            "interval": 30
        }
    )
    
    await ops_system.register_application(app)
    print(f"✓ 应用程序已注册: {app.name} v{app.version}")
    
    # 2. 部署到生产环境
    print("\n2. 部署到生产环境...")
    deployment = await ops_system.deploy_application(
        app.id, 
        EnvironmentType.PRODUCTION, 
        DeploymentStrategy.BLUE_GREEN,
        "admin"
    )
    
    if deployment:
        print(f"✓ 部署已启动: {deployment.id}")
        print(f"  状态: {deployment.status.value}")
        print(f"  策略: {deployment.strategy.value}")
        print(f"  环境: {deployment.environment.value}")
        
        # 等待部署完成
        await asyncio.sleep(1.0)
        
        # 检查部署状态
        updated_deployment = ops_system.get_deployment_status(deployment.id)
        if updated_deployment:
            print(f"  最终状态: {updated_deployment.status.value}")
            if updated_deployment.completed_at:
                print(f"  完成时间: {updated_deployment.completed_at}")
    
    # 3. 监控系统指标
    print("\n3. 监控系统指标...")
    await asyncio.sleep(2.0)  # 等待监控数据收集
    
    metrics = ops_system.monitoring_system.get_metrics("system")
    if metrics:
        print("✓ 系统指标:")
        for metric in metrics[-3:]:  # 显示最近3个指标
            print(f"  {metric.name}: {metric.value} {metric.unit}")
    
    # 4. 检查告警
    print("\n4. 检查告警...")
    alerts = ops_system.monitoring_system.get_alerts(status="active")
    if alerts:
        print("✓ 活跃告警:")
        for alert in alerts:
            print(f"  {alert.level.value}: {alert.message}")
    else:
        print("✓ 无活跃告警")
    
    # 5. 服务扩缩容
    print("\n5. 服务扩缩容...")
    services = list(ops_system.services.values())
    if services:
        service = services[0]
        print(f"  当前副本数: {service.replicas}")
        
        # 扩容到3个副本
        await ops_system.scale_service(service.id, 3)
        updated_service = ops_system.get_service_status(service.id)
        if updated_service:
            print(f"  扩容后副本数: {updated_service.replicas}")
    
    # 6. 系统统计信息
    print("\n6. 系统统计信息:")
    system_stats = ops_system.get_system_metrics()
    for key, value in system_stats.items():
        print(f"  {key}: {value}")
    
    # 7. 告警摘要
    print("\n7. 告警摘要:")
    alert_summary = ops_system.get_alerts_summary()
    for key, value in alert_summary.items():
        print(f"  {key}: {value}")
    
    await ops_system.stop()
    print("\n🎉 部署运维系统演示完成")

if __name__ == "__main__":
    asyncio.run(main_demo())
