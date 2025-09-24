# security_monitoring.py
"""
第7章 安全隐私防护体系 - 安全监控与审计系统
实现安全事件监控、异常检测、审计日志等功能
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Callable, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
import threading
from collections import defaultdict, deque
import statistics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SecurityEventType(Enum):
    """安全事件类型"""
    LOGIN_SUCCESS = "登录成功"
    LOGIN_FAILURE = "登录失败"
    PERMISSION_DENIED = "权限拒绝"
    DATA_ACCESS = "数据访问"
    DATA_MODIFICATION = "数据修改"
    SYSTEM_ERROR = "系统错误"
    SUSPICIOUS_ACTIVITY = "可疑活动"
    MALWARE_DETECTED = "恶意软件检测"
    BRUTE_FORCE_ATTACK = "暴力破解攻击"
    DATA_BREACH = "数据泄露"
    UNAUTHORIZED_ACCESS = "未授权访问"
    CONFIGURATION_CHANGE = "配置变更"
    OTHER = "其他"

class SeverityLevel(Enum):
    """严重程度级别"""
    LOW = "低"
    MEDIUM = "中"
    HIGH = "高"
    CRITICAL = "关键"

class AlertStatus(Enum):
    """告警状态"""
    ACTIVE = "活跃"
    ACKNOWLEDGED = "已确认"
    RESOLVED = "已解决"
    SUPPRESSED = "已抑制"

@dataclass
class SecurityEvent:
    """安全事件"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: SecurityEventType = SecurityEventType.OTHER
    severity: SeverityLevel = SeverityLevel.MEDIUM
    timestamp: datetime = field(default_factory=datetime.now)
    source_ip: str = ""
    user_id: str = ""
    resource: str = ""
    description: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "source_ip": self.source_ip,
            "user_id": self.user_id,
            "resource": self.resource,
            "description": self.description,
            "details": self.details,
            "tags": self.tags
        }

@dataclass
class SecurityAlert:
    """安全告警"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_id: str = ""
    alert_type: str = ""
    severity: SeverityLevel = SeverityLevel.MEDIUM
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    message: str = ""
    recommendations: List[str] = field(default_factory=list)
    assigned_to: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "event_id": self.event_id,
            "alert_type": self.alert_type,
            "severity": self.severity.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "message": self.message,
            "recommendations": self.recommendations,
            "assigned_to": self.assigned_to
        }

@dataclass
class AuditLog:
    """审计日志"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: str = ""
    action: str = ""
    resource: str = ""
    result: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: str = ""
    user_agent: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "action": self.action,
            "resource": self.resource,
            "result": self.result,
            "details": self.details,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent
        }

class AnomalyDetector:
    """异常检测器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.baseline_metrics = {}
        self.thresholds = config.get("thresholds", {})
        self.window_size = config.get("window_size", 100)
        self.event_history = deque(maxlen=self.window_size)
    
    def update_baseline(self, metric_name: str, value: float):
        """更新基线指标"""
        if metric_name not in self.baseline_metrics:
            self.baseline_metrics[metric_name] = []
        
        self.baseline_metrics[metric_name].append(value)
        
        # 保持最近的数据
        if len(self.baseline_metrics[metric_name]) > self.window_size:
            self.baseline_metrics[metric_name] = self.baseline_metrics[metric_name][-self.window_size:]
    
    def detect_anomaly(self, metric_name: str, value: float) -> Tuple[bool, float]:
        """检测异常"""
        try:
            if metric_name not in self.baseline_metrics or len(self.baseline_metrics[metric_name]) < 10:
                return False, 0.0
            
            # 计算统计指标
            baseline_values = self.baseline_metrics[metric_name]
            mean_val = statistics.mean(baseline_values)
            std_val = statistics.stdev(baseline_values) if len(baseline_values) > 1 else 0
            
            if std_val == 0:
                return False, 0.0
            
            # 计算Z分数
            z_score = abs((value - mean_val) / std_val)
            
            # 获取阈值
            threshold = self.thresholds.get(metric_name, 3.0)
            
            is_anomaly = z_score > threshold
            anomaly_score = z_score
            
            return is_anomaly, anomaly_score
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return False, 0.0
    
    def detect_pattern_anomaly(self, events: List[SecurityEvent]) -> List[SecurityEvent]:
        """检测模式异常"""
        try:
            anomalies = []
            
            # 检测频繁失败登录
            failed_logins = [e for e in events if e.event_type == SecurityEventType.LOGIN_FAILURE]
            if len(failed_logins) > 5:  # 5分钟内超过5次失败
                anomalies.extend(failed_logins[-5:])
            
            # 检测异常时间访问
            current_hour = datetime.now().hour
            if current_hour < 6 or current_hour > 22:  # 非工作时间
                night_events = [e for e in events if e.timestamp.hour < 6 or e.timestamp.hour > 22]
                anomalies.extend(night_events)
            
            # 检测异常IP
            ip_counts = defaultdict(int)
            for event in events:
                ip_counts[event.source_ip] += 1
            
            for ip, count in ip_counts.items():
                if count > 10:  # 同一IP超过10次访问
                    ip_events = [e for e in events if e.source_ip == ip]
                    anomalies.extend(ip_events[-5:])
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Pattern anomaly detection failed: {e}")
            return []

class SecurityRuleEngine:
    """安全规则引擎"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rules = self._initialize_rules()
    
    def _initialize_rules(self) -> List[Dict[str, Any]]:
        """初始化安全规则"""
        return [
            {
                "name": "多次登录失败",
                "condition": lambda events: len([e for e in events if e.event_type == SecurityEventType.LOGIN_FAILURE]) > 3,
                "severity": SeverityLevel.HIGH,
                "action": "block_ip"
            },
            {
                "name": "非工作时间访问",
                "condition": lambda events: any(e.timestamp.hour < 6 or e.timestamp.hour > 22 for e in events),
                "severity": SeverityLevel.MEDIUM,
                "action": "alert_admin"
            },
            {
                "name": "权限提升尝试",
                "condition": lambda events: len([e for e in events if e.event_type == SecurityEventType.PERMISSION_DENIED]) > 5,
                "severity": SeverityLevel.HIGH,
                "action": "suspend_user"
            },
            {
                "name": "大量数据访问",
                "condition": lambda events: len([e for e in events if e.event_type == SecurityEventType.DATA_ACCESS]) > 20,
                "severity": SeverityLevel.MEDIUM,
                "action": "rate_limit"
            }
        ]
    
    def evaluate_rules(self, events: List[SecurityEvent]) -> List[SecurityAlert]:
        """评估安全规则"""
        try:
            alerts = []
            
            for rule in self.rules:
                if rule["condition"](events):
                    alert = SecurityAlert(
                        alert_type=rule["name"],
                        severity=rule["severity"],
                        message=f"触发安全规则: {rule['name']}",
                        recommendations=[f"建议执行: {rule['action']}"]
                    )
                    alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Rule evaluation failed: {e}")
            return []

class SecurityMonitor:
    """安全监控器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.events: List[SecurityEvent] = []
        self.alerts: List[SecurityAlert] = []
        self.audit_logs: List[AuditLog] = []
        self.anomaly_detector = AnomalyDetector(config.get("anomaly_detection", {}))
        self.rule_engine = SecurityRuleEngine(config.get("rules", {}))
        self.monitoring_active = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """启动监控"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop)
            self.monitor_thread.start()
            logger.info("Security monitoring started")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Security monitoring stopped")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 检测异常
                self._detect_anomalies()
                
                # 评估规则
                self._evaluate_rules()
                
                # 清理旧数据
                self._cleanup_old_data()
                
                time.sleep(5)  # 每5秒检查一次
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(10)
    
    def _detect_anomalies(self):
        """检测异常"""
        try:
            # 获取最近的事件
            recent_events = self.events[-100:] if len(self.events) > 100 else self.events
            
            # 模式异常检测
            pattern_anomalies = self.anomaly_detector.detect_pattern_anomaly(recent_events)
            
            for event in pattern_anomalies:
                if not any(alert.event_id == event.id for alert in self.alerts):
                    alert = SecurityAlert(
                        event_id=event.id,
                        alert_type="模式异常",
                        severity=SeverityLevel.MEDIUM,
                        message=f"检测到异常模式: {event.description}",
                        recommendations=["建议进一步调查"]
                    )
                    self.alerts.append(alert)
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
    
    def _evaluate_rules(self):
        """评估规则"""
        try:
            # 获取最近的事件
            recent_events = self.events[-50:] if len(self.events) > 50 else self.events
            
            # 评估规则
            new_alerts = self.rule_engine.evaluate_rules(recent_events)
            
            for alert in new_alerts:
                if not any(a.alert_type == alert.alert_type and a.status == AlertStatus.ACTIVE for a in self.alerts):
                    self.alerts.append(alert)
            
        except Exception as e:
            logger.error(f"Rule evaluation failed: {e}")
    
    def _cleanup_old_data(self):
        """清理旧数据"""
        try:
            cutoff_time = datetime.now() - timedelta(days=7)
            
            # 清理旧事件
            self.events = [e for e in self.events if e.timestamp > cutoff_time]
            
            # 清理旧审计日志
            self.audit_logs = [a for a in self.audit_logs if a.timestamp > cutoff_time]
            
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
    
    async def log_event(self, event: SecurityEvent):
        """记录安全事件"""
        try:
            self.events.append(event)
            
            # 更新异常检测基线
            if event.event_type == SecurityEventType.LOGIN_SUCCESS:
                self.anomaly_detector.update_baseline("login_success", 1.0)
            elif event.event_type == SecurityEventType.LOGIN_FAILURE:
                self.anomaly_detector.update_baseline("login_failure", 1.0)
            
            logger.info(f"Security event logged: {event.event_type.value}")
            
        except Exception as e:
            logger.error(f"Event logging failed: {e}")
    
    async def log_audit(self, audit_log: AuditLog):
        """记录审计日志"""
        try:
            self.audit_logs.append(audit_log)
            logger.info(f"Audit log recorded: {audit_log.action}")
            
        except Exception as e:
            logger.error(f"Audit logging failed: {e}")
    
    def get_active_alerts(self) -> List[SecurityAlert]:
        """获取活跃告警"""
        return [alert for alert in self.alerts if alert.status == AlertStatus.ACTIVE]
    
    def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """确认告警"""
        try:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.status = AlertStatus.ACKNOWLEDGED
                    alert.acknowledged_at = datetime.now()
                    alert.assigned_to = user_id
                    logger.info(f"Alert {alert_id} acknowledged by {user_id}")
                    return True
            return False
        except Exception as e:
            logger.error(f"Alert acknowledgment failed: {e}")
            return False
    
    def resolve_alert(self, alert_id: str, user_id: str) -> bool:
        """解决告警"""
        try:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.status = AlertStatus.RESOLVED
                    alert.resolved_at = datetime.now()
                    alert.assigned_to = user_id
                    logger.info(f"Alert {alert_id} resolved by {user_id}")
                    return True
            return False
        except Exception as e:
            logger.error(f"Alert resolution failed: {e}")
            return False
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """获取安全指标"""
        try:
            now = datetime.now()
            last_24h = now - timedelta(hours=24)
            last_7d = now - timedelta(days=7)
            
            # 24小时内的事件
            events_24h = [e for e in self.events if e.timestamp > last_24h]
            
            # 7天内的事件
            events_7d = [e for e in self.events if e.timestamp > last_7d]
            
            # 活跃告警
            active_alerts = self.get_active_alerts()
            
            # 按严重程度统计
            severity_counts = defaultdict(int)
            for event in events_24h:
                severity_counts[event.severity.value] += 1
            
            # 按类型统计
            type_counts = defaultdict(int)
            for event in events_24h:
                type_counts[event.event_type.value] += 1
            
            return {
                "total_events_24h": len(events_24h),
                "total_events_7d": len(events_7d),
                "active_alerts": len(active_alerts),
                "severity_distribution": dict(severity_counts),
                "event_type_distribution": dict(type_counts),
                "critical_events_24h": len([e for e in events_24h if e.severity == SeverityLevel.CRITICAL]),
                "high_events_24h": len([e for e in events_24h if e.severity == SeverityLevel.HIGH]),
                "audit_logs_24h": len([a for a in self.audit_logs if a.timestamp > last_24h])
            }
            
        except Exception as e:
            logger.error(f"Security metrics failed: {e}")
            return {}

# 示例用法
async def main_demo():
    """安全监控系统演示"""
    config = {
        "anomaly_detection": {
            "thresholds": {
                "login_failure": 2.0,
                "data_access": 3.0
            },
            "window_size": 100
        },
        "rules": {}
    }
    
    # 创建安全监控器
    monitor = SecurityMonitor(config)
    monitor.start_monitoring()
    
    print("🔒 安全监控与审计系统演示")
    print("=" * 50)
    
    # 模拟安全事件
    test_events = [
        SecurityEvent(
            event_type=SecurityEventType.LOGIN_SUCCESS,
            severity=SeverityLevel.LOW,
            source_ip="192.168.1.100",
            user_id="user1",
            description="用户登录成功"
        ),
        SecurityEvent(
            event_type=SecurityEventType.LOGIN_FAILURE,
            severity=SeverityLevel.MEDIUM,
            source_ip="192.168.1.101",
            user_id="user2",
            description="密码错误"
        ),
        SecurityEvent(
            event_type=SecurityEventType.LOGIN_FAILURE,
            severity=SeverityLevel.MEDIUM,
            source_ip="192.168.1.101",
            user_id="user2",
            description="密码错误"
        ),
        SecurityEvent(
            event_type=SecurityEventType.LOGIN_FAILURE,
            severity=SeverityLevel.MEDIUM,
            source_ip="192.168.1.101",
            user_id="user2",
            description="密码错误"
        ),
        SecurityEvent(
            event_type=SecurityEventType.LOGIN_FAILURE,
            severity=SeverityLevel.MEDIUM,
            source_ip="192.168.1.101",
            user_id="user2",
            description="密码错误"
        ),
        SecurityEvent(
            event_type=SecurityEventType.DATA_ACCESS,
            severity=SeverityLevel.LOW,
            source_ip="192.168.1.100",
            user_id="user1",
            resource="customer_data",
            description="访问客户数据"
        ),
        SecurityEvent(
            event_type=SecurityEventType.PERMISSION_DENIED,
            severity=SeverityLevel.HIGH,
            source_ip="192.168.1.102",
            user_id="user3",
            resource="admin_panel",
            description="尝试访问管理面板"
        )
    ]
    
    print("\n📝 记录安全事件:")
    for i, event in enumerate(test_events, 1):
        print(f"{i}. {event.event_type.value} - {event.description}")
        await monitor.log_event(event)
        await asyncio.sleep(0.1)
    
    # 模拟审计日志
    print(f"\n📋 记录审计日志:")
    audit_logs = [
        AuditLog(
            user_id="user1",
            action="LOGIN",
            resource="system",
            result="SUCCESS",
            ip_address="192.168.1.100"
        ),
        AuditLog(
            user_id="user1",
            action="DATA_ACCESS",
            resource="customer_data",
            result="SUCCESS",
            ip_address="192.168.1.100"
        ),
        AuditLog(
            user_id="user2",
            action="LOGIN",
            resource="system",
            result="FAILURE",
            ip_address="192.168.1.101"
        )
    ]
    
    for i, audit_log in enumerate(audit_logs, 1):
        print(f"{i}. {audit_log.action} - {audit_log.result}")
        await monitor.log_audit(audit_log)
        await asyncio.sleep(0.1)
    
    # 等待监控处理
    await asyncio.sleep(2)
    
    # 显示活跃告警
    print(f"\n🚨 活跃告警:")
    active_alerts = monitor.get_active_alerts()
    if active_alerts:
        for i, alert in enumerate(active_alerts, 1):
            print(f"{i}. {alert.alert_type} - {alert.severity.value}")
            print(f"   消息: {alert.message}")
            print(f"   建议: {', '.join(alert.recommendations)}")
    else:
        print("   暂无活跃告警")
    
    # 显示安全指标
    print(f"\n📊 安全指标:")
    metrics = monitor.get_security_metrics()
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")
    
    # 告警管理演示
    if active_alerts:
        print(f"\n🔧 告警管理演示:")
        alert = active_alerts[0]
        
        # 确认告警
        if monitor.acknowledge_alert(alert.id, "admin"):
            print(f"   告警 {alert.id} 已确认")
        
        # 解决告警
        if monitor.resolve_alert(alert.id, "admin"):
            print(f"   告警 {alert.id} 已解决")
    
    # 停止监控
    monitor.stop_monitoring()
    
    print("\n🎉 安全监控系统演示完成")

if __name__ == "__main__":
    asyncio.run(main_demo())
