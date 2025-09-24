# monitoring_system.py
"""
第5章 规划与执行引擎开发 - 监控系统
实现智能体的监控、指标收集、告警和可视化功能
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
import psutil
import os
import sqlite3
import csv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MetricType(Enum):
    """指标类型枚举"""
    COUNTER = "计数器"
    GAUGE = "仪表"
    HISTOGRAM = "直方图"
    SUMMARY = "摘要"
    CUSTOM = "自定义"

class AlertSeverity(Enum):
    """告警严重程度枚举"""
    INFO = "信息"
    WARNING = "警告"
    ERROR = "错误"
    CRITICAL = "严重"

class AlertStatus(Enum):
    """告警状态枚举"""
    ACTIVE = "活跃"
    RESOLVED = "已解决"
    SUPPRESSED = "已抑制"
    ACKNOWLEDGED = "已确认"

@dataclass
class Metric:
    """指标数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    value: float = 0.0
    metric_type: MetricType = MetricType.GAUGE
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "value": self.value,
            "metric_type": self.metric_type.value,
            "labels": self.labels,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

@dataclass
class AlertRule:
    """告警规则"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    metric_name: str = ""
    condition: str = ""  # 例如: "value > 80"
    severity: AlertSeverity = AlertSeverity.WARNING
    duration: timedelta = field(default_factory=lambda: timedelta(minutes=1))
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "metric_name": self.metric_name,
            "condition": self.condition,
            "severity": self.severity.value,
            "duration": self.duration.total_seconds(),
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }

@dataclass
class Alert:
    """告警"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    rule_id: str = ""
    name: str = ""
    description: str = ""
    severity: AlertSeverity = AlertSeverity.WARNING
    status: AlertStatus = AlertStatus.ACTIVE
    triggered_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    metric_value: float = 0.0
    threshold: float = 0.0
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "severity": self.severity.value,
            "status": self.status.value,
            "triggered_at": self.triggered_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "labels": self.labels,
            "metadata": self.metadata
        }

@dataclass
class Dashboard:
    """仪表板"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    widgets: List[Dict[str, Any]] = field(default_factory=list)
    layout: Dict[str, Any] = field(default_factory=dict)
    refresh_interval: int = 30  # 秒
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "widgets": self.widgets,
            "layout": self.layout,
            "refresh_interval": self.refresh_interval,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }

class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.collectors: Dict[str, Callable] = {}
        self.collecting = False
        self.collection_interval = config.get("collection_interval", 1.0)
    
    def register_collector(self, name: str, collector_func: Callable):
        """注册收集器"""
        self.collectors[name] = collector_func
        logger.info(f"Registered metrics collector: {name}")
    
    async def start_collection(self):
        """开始收集"""
        self.collecting = True
        asyncio.create_task(self._collection_loop())
        logger.info("Metrics collection started")
    
    async def stop_collection(self):
        """停止收集"""
        self.collecting = False
        logger.info("Metrics collection stopped")
    
    async def _collection_loop(self):
        """收集循环"""
        while self.collecting:
            try:
                await self._collect_metrics()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
    
    async def _collect_metrics(self):
        """收集指标"""
        for name, collector_func in self.collectors.items():
            try:
                if asyncio.iscoroutinefunction(collector_func):
                    metrics = await collector_func()
                else:
                    metrics = collector_func()
                
                if isinstance(metrics, list):
                    for metric in metrics:
                        self.metrics[metric.name].append(metric)
                elif isinstance(metrics, Metric):
                    self.metrics[metrics.name].append(metrics)
                    
            except Exception as e:
                logger.error(f"Collector {name} failed: {e}")
    
    def get_metric(self, name: str, limit: int = 100) -> List[Metric]:
        """获取指标"""
        if name in self.metrics:
            return list(self.metrics[name])[-limit:]
        return []
    
    def get_latest_metric(self, name: str) -> Optional[Metric]:
        """获取最新指标"""
        if name in self.metrics and self.metrics[name]:
            return self.metrics[name][-1]
        return None

class AlertManager:
    """告警管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.evaluating = False
        self.evaluation_interval = config.get("evaluation_interval", 5.0)
    
    async def start_evaluation(self):
        """开始告警评估"""
        self.evaluating = True
        asyncio.create_task(self._evaluation_loop())
        logger.info("Alert evaluation started")
    
    async def stop_evaluation(self):
        """停止告警评估"""
        self.evaluating = False
        logger.info("Alert evaluation stopped")
    
    async def _evaluation_loop(self):
        """评估循环"""
        while self.evaluating:
            try:
                await self._evaluate_rules()
                await asyncio.sleep(self.evaluation_interval)
            except Exception as e:
                logger.error(f"Alert evaluation error: {e}")
    
    async def _evaluate_rules(self):
        """评估告警规则"""
        for rule in self.rules.values():
            if rule.enabled:
                await self._evaluate_rule(rule)
    
    async def _evaluate_rule(self, rule: AlertRule):
        """评估单个规则"""
        try:
            # 这里需要从指标收集器获取指标值
            # 简化实现：模拟指标值
            metric_value = np.random.uniform(0, 100)
            
            # 评估条件
            if self._evaluate_condition(metric_value, rule.condition):
                # 触发告警
                await self._trigger_alert(rule, metric_value)
            else:
                # 检查是否需要解决告警
                await self._resolve_alert_if_needed(rule)
                
        except Exception as e:
            logger.error(f"Rule evaluation failed: {e}")
    
    def _evaluate_condition(self, value: float, condition: str) -> bool:
        """评估条件"""
        try:
            # 简化的条件评估
            if ">" in condition:
                threshold = float(condition.split(">")[1].strip())
                return value > threshold
            elif "<" in condition:
                threshold = float(condition.split("<")[1].strip())
                return value < threshold
            elif ">=" in condition:
                threshold = float(condition.split(">=")[1].strip())
                return value >= threshold
            elif "<=" in condition:
                threshold = float(condition.split("<=")[1].strip())
                return value <= threshold
            elif "==" in condition:
                threshold = float(condition.split("==")[1].strip())
                return value == threshold
            else:
                return False
                
        except Exception as e:
            logger.error(f"Condition evaluation failed: {e}")
            return False
    
    async def _trigger_alert(self, rule: AlertRule, metric_value: float):
        """触发告警"""
        try:
            # 检查是否已有活跃告警
            existing_alert = None
            for alert in self.active_alerts.values():
                if alert.rule_id == rule.id and alert.status == AlertStatus.ACTIVE:
                    existing_alert = alert
                    break
            
            if existing_alert:
                # 更新现有告警
                existing_alert.metric_value = metric_value
                existing_alert.triggered_at = datetime.now()
            else:
                # 创建新告警
                alert = Alert(
                    rule_id=rule.id,
                    name=rule.name,
                    description=rule.description,
                    severity=rule.severity,
                    metric_value=metric_value,
                    threshold=self._extract_threshold(rule.condition),
                    labels={}
                )
                
                self.active_alerts[alert.id] = alert
                self.alert_history.append(alert)
                
                logger.warning(f"Alert triggered: {alert.name} (severity: {alert.severity.value})")
                
        except Exception as e:
            logger.error(f"Alert triggering failed: {e}")
    
    async def _resolve_alert_if_needed(self, rule: AlertRule):
        """如果需要则解决告警"""
        try:
            for alert in self.active_alerts.values():
                if alert.rule_id == rule.id and alert.status == AlertStatus.ACTIVE:
                    alert.status = AlertStatus.RESOLVED
                    alert.resolved_at = datetime.now()
                    logger.info(f"Alert resolved: {alert.name}")
                    
        except Exception as e:
            logger.error(f"Alert resolution failed: {e}")
    
    def _extract_threshold(self, condition: str) -> float:
        """提取阈值"""
        try:
            if ">" in condition:
                return float(condition.split(">")[1].strip())
            elif "<" in condition:
                return float(condition.split("<")[1].strip())
            elif ">=" in condition:
                return float(condition.split(">=")[1].strip())
            elif "<=" in condition:
                return float(condition.split("<=")[1].strip())
            elif "==" in condition:
                return float(condition.split("==")[1].strip())
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Threshold extraction failed: {e}")
            return 0.0
    
    def create_rule(self, rule: AlertRule) -> str:
        """创建告警规则"""
        self.rules[rule.id] = rule
        logger.info(f"Created alert rule: {rule.name}")
        return rule.id
    
    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """更新告警规则"""
        if rule_id in self.rules:
            rule = self.rules[rule_id]
            for key, value in updates.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)
            logger.info(f"Updated alert rule: {rule.name}")
            return True
        return False
    
    def delete_rule(self, rule_id: str) -> bool:
        """删除告警规则"""
        if rule_id in self.rules:
            rule = self.rules[rule_id]
            del self.rules[rule_id]
            logger.info(f"Deleted alert rule: {rule.name}")
            return True
        return False
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """确认告警"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now()
            alert.acknowledged_by = acknowledged_by
            logger.info(f"Alert acknowledged: {alert.name}")
            return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        return [alert for alert in self.active_alerts.values() if alert.status == AlertStatus.ACTIVE]
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """获取告警历史"""
        return self.alert_history[-limit:]

class DataStorage:
    """数据存储"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = config.get("db_path", "monitoring.db")
        self.init_database()
    
    def init_database(self):
        """初始化数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 创建指标表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    metric_type TEXT NOT NULL,
                    labels TEXT,
                    timestamp TEXT NOT NULL,
                    metadata TEXT
                )
            ''')
            
            # 创建告警表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    rule_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    triggered_at TEXT NOT NULL,
                    resolved_at TEXT,
                    acknowledged_at TEXT,
                    acknowledged_by TEXT,
                    metric_value REAL,
                    threshold REAL,
                    labels TEXT,
                    metadata TEXT
                )
            ''')
            
            # 创建告警规则表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alert_rules (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    metric_name TEXT NOT NULL,
                    condition TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    duration INTEGER NOT NULL,
                    enabled BOOLEAN NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
    
    def store_metric(self, metric: Metric) -> bool:
        """存储指标"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO metrics (id, name, value, metric_type, labels, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                metric.id,
                metric.name,
                metric.value,
                metric.metric_type.value,
                json.dumps(metric.labels),
                metric.timestamp.isoformat(),
                json.dumps(metric.metadata)
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Metric storage failed: {e}")
            return False
    
    def store_alert(self, alert: Alert) -> bool:
        """存储告警"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO alerts (id, rule_id, name, description, severity, status,
                                 triggered_at, resolved_at, acknowledged_at, acknowledged_by,
                                 metric_value, threshold, labels, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.id,
                alert.rule_id,
                alert.name,
                alert.description,
                alert.severity.value,
                alert.status.value,
                alert.triggered_at.isoformat(),
                alert.resolved_at.isoformat() if alert.resolved_at else None,
                alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                alert.acknowledged_by,
                alert.metric_value,
                alert.threshold,
                json.dumps(alert.labels),
                json.dumps(alert.metadata)
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Alert storage failed: {e}")
            return False
    
    def get_metrics(self, name: str, start_time: datetime, end_time: datetime) -> List[Metric]:
        """获取指标"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, name, value, metric_type, labels, timestamp, metadata
                FROM metrics
                WHERE name = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            ''', (name, start_time.isoformat(), end_time.isoformat()))
            
            rows = cursor.fetchall()
            conn.close()
            
            metrics = []
            for row in rows:
                metric = Metric(
                    id=row[0],
                    name=row[1],
                    value=row[2],
                    metric_type=MetricType(row[3]),
                    labels=json.loads(row[4]) if row[4] else {},
                    timestamp=datetime.fromisoformat(row[5]),
                    metadata=json.loads(row[6]) if row[6] else {}
                )
                metrics.append(metric)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Metrics retrieval failed: {e}")
            return []

class MonitoringSystem:
    """监控系统主类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_collector = MetricsCollector(config)
        self.alert_manager = AlertManager(config)
        self.data_storage = DataStorage(config)
        self.dashboards: Dict[str, Dashboard] = {}
        self.running = False
    
    async def start(self):
        """启动监控系统"""
        self.running = True
        
        # 注册默认收集器
        self._register_default_collectors()
        
        # 启动组件
        await self.metrics_collector.start_collection()
        await self.alert_manager.start_evaluation()
        
        logger.info("Monitoring system started")
    
    async def stop(self):
        """停止监控系统"""
        self.running = False
        
        # 停止组件
        await self.metrics_collector.stop_collection()
        await self.alert_manager.stop_evaluation()
        
        logger.info("Monitoring system stopped")
    
    def _register_default_collectors(self):
        """注册默认收集器"""
        # CPU使用率收集器
        self.metrics_collector.register_collector("cpu", self._collect_cpu_metrics)
        
        # 内存使用率收集器
        self.metrics_collector.register_collector("memory", self._collect_memory_metrics)
        
        # 磁盘使用率收集器
        self.metrics_collector.register_collector("disk", self._collect_disk_metrics)
        
        # 网络使用率收集器
        self.metrics_collector.register_collector("network", self._collect_network_metrics)
    
    def _collect_cpu_metrics(self) -> List[Metric]:
        """收集CPU指标"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            return [
                Metric(
                    name="cpu_usage_percent",
                    value=cpu_percent,
                    metric_type=MetricType.GAUGE,
                    labels={"host": "localhost"}
                )
            ]
        except Exception as e:
            logger.error(f"CPU metrics collection failed: {e}")
            return []
    
    def _collect_memory_metrics(self) -> List[Metric]:
        """收集内存指标"""
        try:
            memory = psutil.virtual_memory()
            return [
                Metric(
                    name="memory_usage_percent",
                    value=memory.percent,
                    metric_type=MetricType.GAUGE,
                    labels={"host": "localhost"}
                ),
                Metric(
                    name="memory_used_bytes",
                    value=memory.used,
                    metric_type=MetricType.GAUGE,
                    labels={"host": "localhost"}
                )
            ]
        except Exception as e:
            logger.error(f"Memory metrics collection failed: {e}")
            return []
    
    def _collect_disk_metrics(self) -> List[Metric]:
        """收集磁盘指标"""
        try:
            disk = psutil.disk_usage('/')
            return [
                Metric(
                    name="disk_usage_percent",
                    value=disk.percent,
                    metric_type=MetricType.GAUGE,
                    labels={"host": "localhost", "mount": "/"}
                ),
                Metric(
                    name="disk_used_bytes",
                    value=disk.used,
                    metric_type=MetricType.GAUGE,
                    labels={"host": "localhost", "mount": "/"}
                )
            ]
        except Exception as e:
            logger.error(f"Disk metrics collection failed: {e}")
            return []
    
    def _collect_network_metrics(self) -> List[Metric]:
        """收集网络指标"""
        try:
            network = psutil.net_io_counters()
            return [
                Metric(
                    name="network_bytes_sent",
                    value=network.bytes_sent,
                    metric_type=MetricType.COUNTER,
                    labels={"host": "localhost"}
                ),
                Metric(
                    name="network_bytes_recv",
                    value=network.bytes_recv,
                    metric_type=MetricType.COUNTER,
                    labels={"host": "localhost"}
                )
            ]
        except Exception as e:
            logger.error(f"Network metrics collection failed: {e}")
            return []
    
    def create_dashboard(self, dashboard: Dashboard) -> str:
        """创建仪表板"""
        self.dashboards[dashboard.id] = dashboard
        logger.info(f"Created dashboard: {dashboard.name}")
        return dashboard.id
    
    def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """获取仪表板"""
        return self.dashboards.get(dashboard_id)
    
    def get_metrics(self, name: str, limit: int = 100) -> List[Metric]:
        """获取指标"""
        return self.metrics_collector.get_metric(name, limit)
    
    def get_latest_metric(self, name: str) -> Optional[Metric]:
        """获取最新指标"""
        return self.metrics_collector.get_latest_metric(name)
    
    def create_alert_rule(self, rule: AlertRule) -> str:
        """创建告警规则"""
        return self.alert_manager.create_rule(rule)
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        return self.alert_manager.get_active_alerts()
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """确认告警"""
        return self.alert_manager.acknowledge_alert(alert_id, acknowledged_by)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "active_alerts": len(self.alert_manager.get_active_alerts()),
            "total_dashboards": len(self.dashboards),
            "registered_collectors": len(self.metrics_collector.collectors),
            "collection_interval": self.metrics_collector.collection_interval,
            "evaluation_interval": self.alert_manager.evaluation_interval
        }

# 示例用法
async def main_demo():
    """监控系统演示"""
    # 创建配置
    config = {
        "collection_interval": 2.0,
        "evaluation_interval": 5.0,
        "db_path": "monitoring_demo.db"
    }
    
    # 创建监控系统
    monitoring_system = MonitoringSystem(config)
    await monitoring_system.start()
    
    print("监控系统演示")
    print("=" * 50)
    
    # 创建告警规则
    print("\n1. 创建告警规则...")
    alert_rules = [
        AlertRule(
            name="CPU使用率过高",
            description="CPU使用率超过80%",
            metric_name="cpu_usage_percent",
            condition="value > 80",
            severity=AlertSeverity.WARNING
        ),
        AlertRule(
            name="内存使用率过高",
            description="内存使用率超过90%",
            metric_name="memory_usage_percent",
            condition="value > 90",
            severity=AlertSeverity.ERROR
        ),
        AlertRule(
            name="磁盘使用率过高",
            description="磁盘使用率超过95%",
            metric_name="disk_usage_percent",
            condition="value > 95",
            severity=AlertSeverity.CRITICAL
        )
    ]
    
    for rule in alert_rules:
        rule_id = monitoring_system.create_alert_rule(rule)
        print(f"✓ 创建告警规则: {rule.name}")
    
    # 创建仪表板
    print("\n2. 创建仪表板...")
    dashboard = Dashboard(
        name="系统监控仪表板",
        description="系统资源监控仪表板",
        widgets=[
            {
                "type": "line_chart",
                "title": "CPU使用率",
                "metric": "cpu_usage_percent",
                "position": {"x": 0, "y": 0, "w": 6, "h": 4}
            },
            {
                "type": "line_chart",
                "title": "内存使用率",
                "metric": "memory_usage_percent",
                "position": {"x": 6, "y": 0, "w": 6, "h": 4}
            },
            {
                "type": "gauge",
                "title": "磁盘使用率",
                "metric": "disk_usage_percent",
                "position": {"x": 0, "y": 4, "w": 4, "h": 4}
            },
            {
                "type": "table",
                "title": "活跃告警",
                "data_source": "active_alerts",
                "position": {"x": 4, "y": 4, "w": 8, "h": 4}
            }
        ],
        refresh_interval=30
    )
    
    dashboard_id = monitoring_system.create_dashboard(dashboard)
    print(f"✓ 创建仪表板: {dashboard.name}")
    
    # 等待收集指标
    print("\n3. 等待指标收集...")
    await asyncio.sleep(10)
    
    # 检查指标
    print("\n4. 检查收集的指标:")
    metric_names = ["cpu_usage_percent", "memory_usage_percent", "disk_usage_percent"]
    for name in metric_names:
        latest_metric = monitoring_system.get_latest_metric(name)
        if latest_metric:
            print(f"  {name}: {latest_metric.value:.2f}")
        else:
            print(f"  {name}: 无数据")
    
    # 检查告警
    print("\n5. 检查活跃告警:")
    active_alerts = monitoring_system.get_active_alerts()
    if active_alerts:
        for alert in active_alerts:
            print(f"  {alert.name}: {alert.severity.value} (值: {alert.metric_value:.2f})")
    else:
        print("  无活跃告警")
    
    # 获取统计信息
    print("\n6. 监控系统统计:")
    stats = monitoring_system.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 停止监控系统
    await monitoring_system.stop()
    print("\n监控系统演示完成")

if __name__ == "__main__":
    asyncio.run(main_demo())
