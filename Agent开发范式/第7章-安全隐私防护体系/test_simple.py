#!/usr/bin/env python3
"""
第7章 安全隐私防护体系 - 简单测试脚本
"""

import asyncio
import sys
import os

# 添加代码路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))

from threat_analysis import ThreatAnalyzer, Threat, AttackVector, RiskLevel, AgentComponent
from access_control import AccessControlSystem, Permission, ResourceType, PermissionAction, Role, User
from privacy_protection import PrivacyProtectionSystem, DataType, PrivacyLevel

async def test_threat_analysis():
    """测试威胁分析功能"""
    print("🔍 测试威胁分析功能...")
    
    analyzer = ThreatAnalyzer()
    
    # 测试威胁创建
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
    print("   ✅ 威胁创建测试通过")
    
    # 测试系统分析
    architecture = {"name": "测试系统", "components": ["LLM", "Memory"]}
    threats = await analyzer.analyze_agent_system(architecture)
    assert len(threats) > 0
    print("   ✅ 系统分析测试通过")
    
    # 测试风险评估
    context = {"system_sensitivity": 0.8, "existing_controls_effectiveness": 0.6}
    result = await analyzer.assess_risk("TI001", context)
    assert result["success"] is True
    assert "risk_score" in result
    print("   ✅ 风险评估测试通过")

async def test_access_control():
    """测试权限控制功能"""
    print("🔐 测试权限控制功能...")
    
    acs = AccessControlSystem()
    
    # 测试权限创建
    permission = Permission(
        resource_type=ResourceType.LLM_MODEL,
        resource_id="gpt-4",
        action=PermissionAction.READ
    )
    assert permission.resource_type == ResourceType.LLM_MODEL
    print("   ✅ 权限创建测试通过")
    
    # 测试角色创建
    role = Role("test_role", "测试角色", "这是一个测试角色")
    assert role.role_id == "test_role"
    print("   ✅ 角色创建测试通过")
    
    # 测试用户创建
    user = User("test_user", "测试用户")
    assert user.user_id == "test_user"
    print("   ✅ 用户创建测试通过")
    
    # 测试授权
    admin_permission = Permission(ResourceType.CONFIGURATION, None, PermissionAction.MANAGE)
    result = await acs.authorize("user_alice", admin_permission)
    assert result is True
    print("   ✅ 授权测试通过")

async def test_privacy_protection():
    """测试隐私保护功能"""
    print("🔒 测试隐私保护功能...")
    
    config = {"encryption": {"algorithm": "FERNET"}}
    privacy_system = PrivacyProtectionSystem(config)
    
    # 测试数据保护
    record = await privacy_system.protect_data(
        "test@example.com",
        DataType.EMAIL,
        PrivacyLevel.HIGH
    )
    assert record.data_type == DataType.EMAIL
    assert record.original_value == "test@example.com"
    assert record.encrypted_value != ""
    assert record.anonymized_value != ""
    print("   ✅ 数据保护测试通过")
    
    # 测试数据检索
    user_data = await privacy_system.retrieve_data(record.id, ["user"])
    assert user_data == record.anonymized_value
    print("   ✅ 数据检索测试通过")
    
    # 测试差分隐私
    original_values = [100, 150, 200, 180, 120]
    noisy_values = await privacy_system.add_noise_to_statistics(original_values)
    assert len(noisy_values) == len(original_values)
    print("   ✅ 差分隐私测试通过")
    
    # 测试隐私统计
    stats = privacy_system.get_privacy_stats()
    assert "total_records" in stats
    print("   ✅ 隐私统计测试通过")

async def main():
    """主测试函数"""
    print("🧪 第7章 安全隐私防护体系测试")
    print("=" * 50)
    
    try:
        await test_threat_analysis()
        await test_access_control()
        await test_privacy_protection()
        
        print("\n🎉 所有测试通过！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
