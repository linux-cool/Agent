# test_security_system.py
"""
第7章 安全隐私防护体系 - 测试用例
测试安全威胁分析、输入验证、权限控制、隐私保护、安全监控等功能
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import sys
import os

# 添加代码路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))

from threat_analysis import ThreatAnalyzer, Threat, AttackVector, RiskLevel, AgentComponent
from input_validation import InputValidator, ValidationRule, SanitizationAction
from access_control import AccessControlSystem, Permission, ResourceType, PermissionAction, Role, User
from privacy_protection import PrivacyProtectionSystem, DataType, PrivacyLevel, EncryptionService, DataAnonymizer
from security_monitoring import SecurityMonitor, SecurityEvent, SecurityEventType, SeverityLevel, AuditLog, SecurityAlert, AlertStatus

class TestThreatAnalysis:
    """威胁分析测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.analyzer = ThreatAnalyzer()
    
    def test_threat_creation(self):
        """测试威胁创建"""
        threat = Threat(
            id="TEST001",
            name="测试威胁",
            description="这是一个测试威胁",
            attack_vector=AttackVector.PROMPT_INJECTION,
            target_component=AgentComponent.LLM,
            potential_impact="测试影响",
            risk_level=RiskLevel.HIGH
        )
        
        assert threat.id == "TEST001"
        assert threat.name == "测试威胁"
        assert threat.attack_vector == AttackVector.PROMPT_INJECTION
        assert threat.risk_level == RiskLevel.HIGH
    
    def test_threat_to_dict(self):
        """测试威胁转字典"""
        threat = Threat(
            id="TEST001",
            name="测试威胁",
            description="这是一个测试威胁",
            attack_vector=AttackVector.PROMPT_INJECTION,
            target_component=AgentComponent.LLM,
            potential_impact="测试影响",
            risk_level=RiskLevel.HIGH
        )
        
        threat_dict = threat.to_dict()
        assert threat_dict["id"] == "TEST001"
        assert threat_dict["attack_vector"] == "Prompt Injection"
        assert threat_dict["risk_level"] == "高"
    
    @pytest.mark.asyncio
    async def test_add_threat(self):
        """测试添加威胁"""
        threat = Threat(
            id="TEST001",
            name="测试威胁",
            description="这是一个测试威胁",
            attack_vector=AttackVector.PROMPT_INJECTION,
            target_component=AgentComponent.LLM,
            potential_impact="测试影响",
            risk_level=RiskLevel.HIGH
        )
        
        result = await self.analyzer.add_threat(threat)
        assert result is True
        
        # 测试重复添加
        result = await self.analyzer.add_threat(threat)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_threat(self):
        """测试获取威胁"""
        threat = await self.analyzer.get_threat("TI001")
        assert threat is not None
        assert threat.name == "Prompt Injection"
        
        # 测试不存在的威胁
        threat = await self.analyzer.get_threat("NONEXISTENT")
        assert threat is None
    
    @pytest.mark.asyncio
    async def test_analyze_agent_system(self):
        """测试智能体系统分析"""
        architecture = {
            "name": "测试智能体",
            "components": ["LLM", "Memory", "Execution Engine"]
        }
        
        threats = await self.analyzer.analyze_agent_system(architecture)
        assert len(threats) > 0
        
        # 检查是否包含预期的威胁
        threat_names = [t.name for t in threats]
        assert "Prompt Injection" in threat_names
        assert "Data Poisoning" in threat_names
    
    @pytest.mark.asyncio
    async def test_assess_risk(self):
        """测试风险评估"""
        context = {
            "system_sensitivity": 0.8,
            "existing_controls_effectiveness": 0.6
        }
        
        result = await self.analyzer.assess_risk("TI001", context)
        assert result["success"] is True
        assert "risk_score" in result
        assert "assessment" in result
        assert "mitigation_strategy" in result

class TestInputValidation:
    """输入验证测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.validator = InputValidator()
    
    @pytest.mark.asyncio
    async def test_validate_normal_input(self):
        """测试正常输入验证"""
        is_valid, sanitized_value, errors = await self.validator.validate_and_sanitize(
            "user_query", "请问如何查询我的订单状态？"
        )
        
        assert is_valid is True
        assert sanitized_value == "请问如何查询我的订单状态？"
        assert len(errors) == 0
    
    @pytest.mark.asyncio
    async def test_validate_sql_injection(self):
        """测试SQL注入检测"""
        is_valid, sanitized_value, errors = await self.validator.validate_and_sanitize(
            "user_query", "SELECT * FROM users WHERE id = 1"
        )
        
        assert is_valid is False
        assert len(errors) > 0
        assert any("SQL injection" in error for error in errors)
    
    @pytest.mark.asyncio
    async def test_validate_xss(self):
        """测试XSS检测"""
        is_valid, sanitized_value, errors = await self.validator.validate_and_sanitize(
            "user_query", "<script>alert('XSS')</script>"
        )
        
        assert is_valid is False
        assert len(errors) > 0
        assert any("XSS" in error for error in errors)
    
    @pytest.mark.asyncio
    async def test_validate_max_length(self):
        """测试最大长度验证"""
        long_input = "a" * 600
        is_valid, sanitized_value, errors = await self.validator.validate_and_sanitize(
            "user_query", long_input
        )
        
        assert is_valid is False
        assert len(errors) > 0
        assert any("max length" in error for error in errors)
    

class TestAccessControl:
    """权限控制测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.acs = AccessControlSystem()
    
    def test_permission_creation(self):
        """测试权限创建"""
        permission = Permission(
            resource_type=ResourceType.LLM_MODEL,
            resource_id="gpt-4",
            action=PermissionAction.READ
        )
        
        assert permission.resource_type == ResourceType.LLM_MODEL
        assert permission.resource_id == "gpt-4"
        assert permission.action == PermissionAction.READ
    
    def test_role_creation(self):
        """测试角色创建"""
        role = Role(
            role_id="test_role",
            name="测试角色",
            description="这是一个测试角色"
        )
        
        assert role.role_id == "test_role"
        assert role.name == "测试角色"
        
        # 测试权限添加
        permission = Permission(ResourceType.LLM_MODEL, None, PermissionAction.READ)
        role.add_permission(permission)
        assert role.has_permission(permission)
    
    def test_user_creation(self):
        """测试用户创建"""
        user = User(
            user_id="test_user",
            name="测试用户"
        )
        
        assert user.user_id == "test_user"
        assert user.name == "测试用户"
        
        # 测试角色添加
        user.add_role("admin")
        assert "admin" in user.roles
    
    @pytest.mark.asyncio
    async def test_rbac_authorization(self):
        """测试RBAC授权"""
        # 测试管理员权限
        admin_permission = Permission(ResourceType.CONFIGURATION, None, PermissionAction.MANAGE)
        result = await self.acs.authorize("user_alice", admin_permission)
        assert result is True
        
        # 测试普通用户权限
        user_permission = Permission(ResourceType.LLM_MODEL, None, PermissionAction.READ)
        result = await self.acs.authorize("agent_cs", user_permission)
        assert result is False  # 客服智能体没有LLM读取权限
    
    @pytest.mark.asyncio
    async def test_abac_authorization(self):
        """测试ABAC授权"""
        # 测试客服智能体读取客户数据库
        db_read_permission = Permission(ResourceType.DATA_SOURCE, "customer_db", PermissionAction.READ)
        context = {"current_hour": 10}
        
        result = await self.acs.authorize("agent_cs", db_read_permission, context)
        assert result is True
        
        # 测试非工作时间
        context = {"current_hour": 20}
        result = await self.acs.authorize("agent_cs", db_read_permission, context)
        assert result is False

class TestPrivacyProtection:
    """隐私保护测试"""
    
    def setup_method(self):
        """测试前准备"""
        config = {
            "encryption": {"algorithm": "FERNET"},
            "anonymization": {},
            "differential_privacy": {"epsilon": 1.0, "delta": 1e-5},
            "masking": {}
        }
        self.privacy_system = PrivacyProtectionSystem(config)
    
    @pytest.mark.asyncio
    async def test_protect_data(self):
        """测试数据保护"""
        record = await self.privacy_system.protect_data(
            "test@example.com",
            DataType.EMAIL,
            PrivacyLevel.HIGH
        )
        
        assert record.data_type == DataType.EMAIL
        assert record.original_value == "test@example.com"
        assert record.encrypted_value != ""
        assert record.anonymized_value != ""
    
    @pytest.mark.asyncio
    async def test_retrieve_data(self):
        """测试数据检索"""
        # 保护数据
        record = await self.privacy_system.protect_data(
            "test@example.com",
            DataType.EMAIL,
            PrivacyLevel.HIGH
        )
        
        # 普通用户检索
        user_data = await self.privacy_system.retrieve_data(record.id, ["user"])
        assert user_data == record.anonymized_value
        
        # 管理员检索
        admin_data = await self.privacy_system.retrieve_data(record.id, ["admin"])
        assert admin_data == "test@example.com"
    
    @pytest.mark.asyncio
    async def test_add_noise_to_statistics(self):
        """测试差分隐私噪声添加"""
        original_values = [100, 150, 200, 180, 120]
        noisy_values = await self.privacy_system.add_noise_to_statistics(original_values)
        
        assert len(noisy_values) == len(original_values)
        # 噪声值应该与原始值不同
        assert any(abs(noisy - orig) > 0.1 for noisy, orig in zip(noisy_values, original_values))
    
    @pytest.mark.asyncio
    async def test_mask_sensitive_data(self):
        """测试数据脱敏"""
        sensitive_text = "我的邮箱是 user@example.com，电话是 13812345678"
        masked_text = await self.privacy_system.mask_sensitive_data(
            sensitive_text, 
            ["email", "phone"]
        )
        
        assert "user@example.com" not in masked_text
        assert "13812345678" not in masked_text
        assert "*" in masked_text
    
    def test_privacy_stats(self):
        """测试隐私统计"""
        stats = self.privacy_system.get_privacy_stats()
        
        assert "total_records" in stats
        assert "encrypted_records" in stats
        assert "anonymized_records" in stats
        assert "privacy_policies" in stats

class TestSecurityMonitoring:
    """安全监控测试"""
    
    def setup_method(self):
        """测试前准备"""
        config = {
            "anomaly_detection": {
                "thresholds": {"login_failure": 2.0},
                "window_size": 100
            },
            "rules": {}
        }
        self.monitor = SecurityMonitor(config)
    
    def test_security_event_creation(self):
        """测试安全事件创建"""
        event = SecurityEvent(
            event_type=SecurityEventType.LOGIN_SUCCESS,
            severity=SeverityLevel.LOW,
            source_ip="192.168.1.100",
            user_id="user1",
            description="用户登录成功"
        )
        
        assert event.event_type == SecurityEventType.LOGIN_SUCCESS
        assert event.severity == SeverityLevel.LOW
        assert event.source_ip == "192.168.1.100"
        assert event.user_id == "user1"
    
    def test_audit_log_creation(self):
        """测试审计日志创建"""
        audit_log = AuditLog(
            user_id="user1",
            action="LOGIN",
            resource="system",
            result="SUCCESS",
            ip_address="192.168.1.100"
        )
        
        assert audit_log.user_id == "user1"
        assert audit_log.action == "LOGIN"
        assert audit_log.result == "SUCCESS"
    
    @pytest.mark.asyncio
    async def test_log_event(self):
        """测试事件记录"""
        event = SecurityEvent(
            event_type=SecurityEventType.LOGIN_SUCCESS,
            severity=SeverityLevel.LOW,
            description="测试事件"
        )
        
        await self.monitor.log_event(event)
        assert len(self.monitor.events) > 0
        assert self.monitor.events[-1].id == event.id
    
    @pytest.mark.asyncio
    async def test_log_audit(self):
        """测试审计记录"""
        audit_log = AuditLog(
            user_id="user1",
            action="LOGIN",
            resource="system",
            result="SUCCESS"
        )
        
        await self.monitor.log_audit(audit_log)
        assert len(self.monitor.audit_logs) > 0
        assert self.monitor.audit_logs[-1].id == audit_log.id
    
    def test_get_active_alerts(self):
        """测试获取活跃告警"""
        alerts = self.monitor.get_active_alerts()
        assert isinstance(alerts, list)
    
    def test_acknowledge_alert(self):
        """测试确认告警"""
        # 创建一个测试告警
        alert = SecurityAlert(
            alert_type="测试告警",
            severity=SeverityLevel.MEDIUM,
            message="这是一个测试告警"
        )
        self.monitor.alerts.append(alert)
        
        result = self.monitor.acknowledge_alert(alert.id, "admin")
        assert result is True
        
        # 检查告警状态
        updated_alert = next(a for a in self.monitor.alerts if a.id == alert.id)
        assert updated_alert.status == AlertStatus.ACKNOWLEDGED
        assert updated_alert.assigned_to == "admin"
    
    def test_resolve_alert(self):
        """测试解决告警"""
        # 创建一个测试告警
        alert = SecurityAlert(
            alert_type="测试告警",
            severity=SeverityLevel.MEDIUM,
            message="这是一个测试告警"
        )
        self.monitor.alerts.append(alert)
        
        result = self.monitor.resolve_alert(alert.id, "admin")
        assert result is True
        
        # 检查告警状态
        updated_alert = next(a for a in self.monitor.alerts if a.id == alert.id)
        assert updated_alert.status == AlertStatus.RESOLVED
        assert updated_alert.assigned_to == "admin"
    
    def test_security_metrics(self):
        """测试安全指标"""
        metrics = self.monitor.get_security_metrics()
        
        assert "total_events_24h" in metrics
        assert "active_alerts" in metrics
        assert "severity_distribution" in metrics
        assert "event_type_distribution" in metrics

class TestIntegration:
    """集成测试"""
    
    @pytest.mark.asyncio
    async def test_security_workflow(self):
        """测试安全防护工作流程"""
        # 1. 威胁分析
        analyzer = ThreatAnalyzer()
        architecture = {"name": "测试系统", "components": ["LLM", "Memory"]}
        threats = await analyzer.analyze_agent_system(architecture)
        assert len(threats) > 0
        
        # 2. 输入验证
        validator = InputValidator()
        is_valid, sanitized_value, errors = await validator.validate_and_sanitize(
            "user_query", "正常用户查询"
        )
        assert is_valid is True
        
        # 3. 权限控制
        acs = AccessControlSystem()
        permission = Permission(ResourceType.LLM_MODEL, None, PermissionAction.READ)
        authorized = await acs.authorize("user_alice", permission)
        assert authorized is True
        
        # 4. 隐私保护
        config = {"encryption": {"algorithm": "FERNET"}}
        privacy_system = PrivacyProtectionSystem(config)
        record = await privacy_system.protect_data(
            "test@example.com", DataType.EMAIL, PrivacyLevel.HIGH
        )
        assert record.encrypted_value != ""
        
        # 5. 安全监控
        monitor = SecurityMonitor({})
        event = SecurityEvent(
            event_type=SecurityEventType.LOGIN_SUCCESS,
            severity=SeverityLevel.LOW,
            description="集成测试事件"
        )
        await monitor.log_event(event)
        assert len(monitor.events) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
