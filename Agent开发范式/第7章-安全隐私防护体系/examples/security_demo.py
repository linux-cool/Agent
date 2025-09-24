# security_demo.py
"""
第7章 安全隐私防护体系 - 演示程序
展示安全威胁分析、输入验证、权限控制、隐私保护、安全监控等功能
"""

import asyncio
import logging
import yaml
import json
from datetime import datetime, timedelta
import sys
import os

# 添加代码路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))

from threat_analysis import ThreatAnalyzer, Threat, AttackVector, RiskLevel, AgentComponent
from input_validation import InputValidator
from access_control import AccessControlSystem, Permission, ResourceType, PermissionAction, Role, User
from privacy_protection import PrivacyProtectionSystem, DataType, PrivacyLevel, DataMaskingService
from security_monitoring import SecurityMonitor, SecurityEvent, SecurityEventType, SeverityLevel, AuditLog

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SecurityDemo:
    """安全防护系统演示"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), '..', 'config', 'security_config.yaml')
        self.config = self._load_config()
        
        # 初始化各个组件
        self.threat_analyzer = ThreatAnalyzer(self.config.get('threat_analysis', {}))
        self.input_validator = InputValidator(self.config.get('input_validation', {}))
        self.data_masking_service = DataMaskingService(self.config.get('privacy_protection', {}).get('data_masking', {}))
        self.access_control = AccessControlSystem(self.config.get('access_control', {}))
        self.privacy_system = PrivacyProtectionSystem(self.config.get('privacy_protection', {}))
        self.security_monitor = SecurityMonitor(self.config.get('security_monitoring', {}))
        
        # 启动安全监控
        self.security_monitor.start_monitoring()
    
    def _load_config(self) -> dict:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config file: {e}, using default config")
            return {}
    
    async def run_comprehensive_demo(self):
        """运行综合演示"""
        print("🔒 安全隐私防护体系综合演示")
        print("=" * 60)
        
        # 1. 威胁分析演示
        await self._demo_threat_analysis()
        
        # 2. 输入验证演示
        await self._demo_input_validation()
        
        # 3. 权限控制演示
        await self._demo_access_control()
        
        # 4. 隐私保护演示
        await self._demo_privacy_protection()
        
        # 5. 安全监控演示
        await self._demo_security_monitoring()
        
        # 6. 综合安全防护演示
        await self._demo_comprehensive_security()
        
        # 停止监控
        self.security_monitor.stop_monitoring()
        
        print("\n🎉 安全隐私防护体系演示完成")
    
    async def _demo_threat_analysis(self):
        """威胁分析演示"""
        print("\n🔍 1. 安全威胁分析演示")
        print("-" * 40)
        
        # 分析智能体系统架构
        agent_architectures = [
            {
                "name": "智能客服系统",
                "components": ["LLM", "Memory", "Execution Engine", "Tool Manager"],
                "data_sensitivity": "HIGH"
            },
            {
                "name": "代码助手系统",
                "components": ["LLM", "Memory", "Tool Manager"],
                "data_sensitivity": "MEDIUM"
            },
            {
                "name": "数据分析系统",
                "components": ["LLM", "Memory", "Data Storage", "Tool Manager"],
                "data_sensitivity": "CRITICAL"
            }
        ]
        
        for i, architecture in enumerate(agent_architectures, 1):
            print(f"\n{i}. 分析系统: {architecture['name']}")
            print(f"   组件: {', '.join(architecture['components'])}")
            
            # 分析威胁
            threats = await self.threat_analyzer.analyze_agent_system(architecture)
            print(f"   识别到 {len(threats)} 个潜在威胁:")
            
            for threat in threats[:3]:  # 显示前3个威胁
                print(f"     - {threat.name} ({threat.attack_vector.value})")
                print(f"       风险等级: {threat.risk_level.value}")
                print(f"       潜在影响: {threat.potential_impact}")
                
                # 风险评估
                context = {
                    "system_sensitivity": 0.8 if architecture["data_sensitivity"] == "HIGH" else 0.5,
                    "existing_controls_effectiveness": 0.6
                }
                risk_assessment = await self.threat_analyzer.assess_risk(threat.id, context)
                print(f"       风险评估: {risk_assessment['assessment']} (得分: {risk_assessment['risk_score']:.2f})")
                print(f"       缓解策略: {risk_assessment['mitigation_strategy']}")
                print()
            
            await asyncio.sleep(0.5)
    
    async def _demo_input_validation(self):
        """输入验证演示"""
        print("\n🛡️ 2. 输入验证与过滤演示")
        print("-" * 40)
        
        # 测试用例
        test_cases = [
            {
                "input": "请问如何查询我的订单状态？",
                "description": "正常用户查询",
                "expected": "通过"
            },
            {
                "input": "SELECT * FROM users WHERE id = 1; DROP TABLE users;",
                "description": "SQL注入攻击",
                "expected": "拒绝"
            },
            {
                "input": "<script>alert('XSS攻击')</script>你好",
                "description": "XSS攻击",
                "expected": "拒绝"
            },
            {
                "input": "../../etc/passwd",
                "description": "路径遍历攻击",
                "expected": "拒绝"
            },
            {
                "input": "a" * 600,
                "description": "超长输入",
                "expected": "拒绝"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{i}. {test_case['description']}")
            print(f"   输入: {test_case['input'][:50]}{'...' if len(test_case['input']) > 50 else ''}")
            
            # 验证输入
            result = await self.input_validator.validate(test_case["input"])
            is_valid = result.is_valid
            sanitized_value = result.sanitized_data
            errors = result.errors
            
            print(f"   验证结果: {'✅ 通过' if is_valid else '❌ 拒绝'}")
            if errors:
                print(f"   错误信息: {', '.join(errors)}")
            if sanitized_value != test_case["input"]:
                print(f"   净化后: {sanitized_value[:50]}{'...' if len(sanitized_value) > 50 else ''}")
            
            await asyncio.sleep(0.3)
        
        # 敏感数据检测演示
        print(f"\n🔍 敏感数据检测演示:")
        sensitive_text = "我的邮箱是 user@example.com，电话是 13812345678，身份证号是 110101199001011234"
        
        print(f"   原始文本: {sensitive_text}")
        
        masked_text = self.data_masking_service.mask_data(sensitive_text, "email")
        masked_text = self.data_masking_service.mask_data(masked_text, "phone")
        masked_text = self.data_masking_service.mask_data(masked_text, "id_card")
        print(f"   脱敏后: {masked_text}")
    
    async def _demo_access_control(self):
        """权限控制演示"""
        print("\n🔐 3. 权限控制与访问管理演示")
        print("-" * 40)
        
        # 测试用户和权限
        test_scenarios = [
            {
                "user": "user_alice",
                "permission": Permission(ResourceType.CONFIGURATION, None, PermissionAction.MANAGE),
                "description": "管理员管理配置",
                "expected": "允许"
            },
            {
                "user": "agent_cs",
                "permission": Permission(ResourceType.DATA_SOURCE, "customer_db", PermissionAction.READ),
                "description": "客服智能体读取客户数据",
                "expected": "允许"
            },
            {
                "user": "agent_cs",
                "permission": Permission(ResourceType.DATA_SOURCE, "customer_db", PermissionAction.WRITE),
                "description": "客服智能体写入客户数据",
                "expected": "拒绝"
            },
            {
                "user": "agent_data",
                "permission": Permission(ResourceType.TOOL, "search_web", PermissionAction.INVOKE),
                "description": "数据分析智能体调用搜索工具",
                "expected": "拒绝"
            }
        ]
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n{i}. {scenario['description']}")
            print(f"   用户: {scenario['user']}")
            print(f"   权限: {scenario['permission'].to_dict()}")
            
            # RBAC检查
            rbac_result = await self.access_control.authorize(scenario["user"], scenario["permission"])
            print(f"   RBAC结果: {'✅ 允许' if rbac_result else '❌ 拒绝'}")
            
            # ABAC检查（带上下文）
            context = {"current_hour": 10, "department": "CustomerService"}
            abac_result = await self.access_control.authorize(scenario["user"], scenario["permission"], context)
            print(f"   ABAC结果: {'✅ 允许' if abac_result else '❌ 拒绝'}")
            
            await asyncio.sleep(0.3)
        
        # 权限管理演示
        print(f"\n👥 权限管理演示:")
        
        # 创建新角色
        new_role = Role(
            role_id="data_analyst",
            name="数据分析师",
            description="可以分析数据和生成报告"
        )
        new_role.add_permission(Permission(ResourceType.DATA_SOURCE, None, PermissionAction.READ))
        self.access_control.add_role(new_role)
        print(f"   创建新角色: {new_role.name}")
        
        # 创建新用户
        new_user = User(
            user_id="analyst_john",
            name="John Analyst",
            roles={"data_analyst"},
            attributes={"department": "DataAnalytics", "level": "L2"}
        )
        self.access_control.add_user(new_user)
        print(f"   创建新用户: {new_user.name}")
        
        # 测试新用户权限
        test_permission = Permission(ResourceType.DATA_SOURCE, "sales_db", PermissionAction.READ)
        result = await self.access_control.authorize("analyst_john", test_permission)
        print(f"   新用户权限测试: {'✅ 允许' if result else '❌ 拒绝'}")
    
    async def _demo_privacy_protection(self):
        """隐私保护演示"""
        print("\n🔒 4. 隐私保护技术演示")
        print("-" * 40)
        
        # 测试数据
        test_data = [
            {
                "data": "user@example.com",
                "type": DataType.EMAIL,
                "level": PrivacyLevel.HIGH,
                "description": "邮箱地址"
            },
            {
                "data": "13812345678",
                "type": DataType.PHONE,
                "level": PrivacyLevel.HIGH,
                "description": "电话号码"
            },
            {
                "data": "张三",
                "type": DataType.NAME,
                "level": PrivacyLevel.MEDIUM,
                "description": "姓名"
            },
            {
                "data": "110101199001011234",
                "type": DataType.ID_CARD,
                "level": PrivacyLevel.CRITICAL,
                "description": "身份证号"
            }
        ]
        
        protected_records = []
        
        for i, test_case in enumerate(test_data, 1):
            print(f"\n{i}. {test_case['description']}")
            print(f"   原始数据: {test_case['data']}")
            
            # 保护数据
            record = await self.privacy_system.protect_data(
                test_case["data"],
                test_case["type"],
                test_case["level"]
            )
            
            protected_records.append(record)
            
            print(f"   加密数据: {record.encrypted_value[:30] + '...' if record.encrypted_value else '无'}")
            print(f"   匿名数据: {record.anonymized_value}")
            
            await asyncio.sleep(0.3)
        
        # 数据检索演示
        print(f"\n🔍 数据检索演示:")
        
        for i, record in enumerate(protected_records, 1):
            print(f"\n{i}. 检索记录 {record.id}")
            
            # 普通用户检索
            user_data = await self.privacy_system.retrieve_data(record.id, ["user"])
            print(f"   普通用户看到: {user_data}")
            
            # 管理员检索
            admin_data = await self.privacy_system.retrieve_data(record.id, ["admin"])
            print(f"   管理员看到: {admin_data}")
        
        # 差分隐私演示
        print(f"\n📊 差分隐私演示:")
        
        # 原始统计数据
        original_stats = [100, 150, 200, 180, 120, 160, 140, 190, 170, 130]
        print(f"   原始统计: {original_stats}")
        print(f"   平均值: {sum(original_stats) / len(original_stats):.2f}")
        
        # 添加噪声
        noisy_stats = await self.privacy_system.add_noise_to_statistics(original_stats)
        print(f"   添加噪声后: {[round(x, 2) for x in noisy_stats]}")
        print(f"   噪声后平均值: {sum(noisy_stats) / len(noisy_stats):.2f}")
        
        # 数据脱敏演示
        print(f"\n🎭 数据脱敏演示:")
        
        sensitive_text = "我的邮箱是 user@example.com，电话是 13812345678，身份证号是 110101199001011234"
        print(f"   原始文本: {sensitive_text}")
        
        masked_text = await self.privacy_system.mask_sensitive_data(
            sensitive_text, 
            ["email", "phone", "id_card"]
        )
        print(f"   脱敏后: {masked_text}")
    
    async def _demo_security_monitoring(self):
        """安全监控演示"""
        print("\n📊 5. 安全监控与审计演示")
        print("-" * 40)
        
        # 模拟安全事件
        security_events = [
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
            ),
            SecurityEvent(
                event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
                severity=SeverityLevel.HIGH,
                source_ip="192.168.1.103",
                user_id="unknown",
                description="检测到可疑活动"
            )
        ]
        
        print("📝 记录安全事件:")
        for i, event in enumerate(security_events, 1):
            print(f"   {i}. {event.event_type.value} - {event.description}")
            await self.security_monitor.log_event(event)
            await asyncio.sleep(0.1)
        
        # 模拟审计日志
        print(f"\n📋 记录审计日志:")
        audit_logs = [
            AuditLog(
                user_id="user1",
                action="LOGIN",
                resource="system",
                result="SUCCESS",
                ip_address="192.168.1.100",
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
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
            ),
            AuditLog(
                user_id="user3",
                action="ADMIN_ACCESS",
                resource="admin_panel",
                result="FAILURE",
                ip_address="192.168.1.102"
            )
        ]
        
        for i, audit_log in enumerate(audit_logs, 1):
            print(f"   {i}. {audit_log.action} - {audit_log.result}")
            await self.security_monitor.log_audit(audit_log)
            await asyncio.sleep(0.1)
        
        # 等待监控处理
        await asyncio.sleep(2)
        
        # 显示活跃告警
        print(f"\n🚨 活跃告警:")
        active_alerts = self.security_monitor.get_active_alerts()
        if active_alerts:
            for i, alert in enumerate(active_alerts, 1):
                print(f"   {i}. {alert.alert_type} - {alert.severity.value}")
                print(f"      消息: {alert.message}")
                print(f"      建议: {', '.join(alert.recommendations)}")
                print(f"      创建时间: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("   暂无活跃告警")
        
        # 显示安全指标
        print(f"\n📈 安全指标:")
        metrics = self.security_monitor.get_security_metrics()
        for key, value in metrics.items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for sub_key, sub_value in value.items():
                    print(f"     {sub_key}: {sub_value}")
            else:
                print(f"   {key}: {value}")
        
        # 告警管理演示
        if active_alerts:
            print(f"\n🔧 告警管理演示:")
            alert = active_alerts[0]
            
            # 确认告警
            if self.security_monitor.acknowledge_alert(alert.id, "admin"):
                print(f"   ✅ 告警 {alert.id} 已确认")
            
            # 解决告警
            if self.security_monitor.resolve_alert(alert.id, "admin"):
                print(f"   ✅ 告警 {alert.id} 已解决")
    
    async def _demo_comprehensive_security(self):
        """综合安全防护演示"""
        print("\n🛡️ 6. 综合安全防护演示")
        print("-" * 40)
        
        # 模拟一个完整的用户请求处理流程
        print("模拟用户请求处理流程:")
        
        # 1. 用户输入
        user_input = "请帮我查询用户 user@example.com 的订单信息"
        print(f"1. 用户输入: {user_input}")
        
        # 2. 输入验证
        result = await self.input_validator.validate(user_input)
        is_valid = result.is_valid
        sanitized_input = result.sanitized_data
        errors = result.errors
        print(f"2. 输入验证: {'✅ 通过' if is_valid else '❌ 拒绝'}")
        if errors:
            print(f"   错误: {', '.join(errors)}")
        
        if not is_valid:
            print("   请求被拒绝，流程终止")
            return
        
        # 3. 权限检查
        permission = Permission(ResourceType.DATA_SOURCE, "order_db", PermissionAction.READ)
        authorized = await self.access_control.authorize("user1", permission)
        print(f"3. 权限检查: {'✅ 允许' if authorized else '❌ 拒绝'}")
        
        if not authorized:
            print("   权限不足，请求被拒绝")
            return
        
        # 4. 隐私保护
        email_data = "user@example.com"
        protected_record = await self.privacy_system.protect_data(
            email_data, DataType.EMAIL, PrivacyLevel.HIGH
        )
        print(f"4. 隐私保护: 数据已加密和匿名化")
        
        # 5. 记录安全事件
        event = SecurityEvent(
            event_type=SecurityEventType.DATA_ACCESS,
            severity=SeverityLevel.LOW,
            source_ip="192.168.1.100",
            user_id="user1",
            resource="order_db",
            description="查询订单信息"
        )
        await self.security_monitor.log_event(event)
        print(f"5. 安全监控: 事件已记录")
        
        # 6. 记录审计日志
        audit_log = AuditLog(
            user_id="user1",
            action="DATA_QUERY",
            resource="order_db",
            result="SUCCESS",
            ip_address="192.168.1.100"
        )
        await self.security_monitor.log_audit(audit_log)
        print(f"6. 审计日志: 操作已记录")
        
        print("✅ 请求处理完成，所有安全措施已应用")
        
        # 显示最终统计
        print(f"\n📊 系统安全统计:")
        
        # 隐私保护统计
        privacy_stats = self.privacy_system.get_privacy_stats()
        print(f"   隐私保护:")
        print(f"     总记录数: {privacy_stats.get('total_records', 0)}")
        print(f"     加密记录数: {privacy_stats.get('encrypted_records', 0)}")
        print(f"     匿名记录数: {privacy_stats.get('anonymized_records', 0)}")
        
        # 安全监控统计
        security_metrics = self.security_monitor.get_security_metrics()
        print(f"   安全监控:")
        print(f"     24小时事件数: {security_metrics.get('total_events_24h', 0)}")
        print(f"     活跃告警数: {security_metrics.get('active_alerts', 0)}")
        print(f"     关键事件数: {security_metrics.get('critical_events_24h', 0)}")

async def main():
    """主函数"""
    try:
        # 创建演示实例
        demo = SecurityDemo()
        
        # 运行综合演示
        await demo.run_comprehensive_demo()
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ 演示失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())
