#!/usr/bin/env python3
"""
第7章 安全隐私防护体系 - 快速测试脚本
"""

import asyncio
import sys
import os

# 添加代码路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))

from threat_analysis import ThreatAnalyzer
from access_control import AccessControlSystem, Permission, ResourceType, PermissionAction
from privacy_protection import PrivacyProtectionSystem, DataType, PrivacyLevel

async def quick_demo():
    print('🔒 安全隐私防护体系快速演示')
    print('=' * 40)
    
    # 威胁分析演示
    print('\n1. 威胁分析演示')
    analyzer = ThreatAnalyzer()
    architecture = {'name': '测试系统', 'components': ['LLM', 'Memory']}
    threats = await analyzer.analyze_agent_system(architecture)
    print(f'   识别到 {len(threats)} 个威胁')
    for threat in threats[:2]:
        print(f'   - {threat.name}: {threat.risk_level.value}')
    
    # 权限控制演示
    print('\n2. 权限控制演示')
    acs = AccessControlSystem()
    permission = Permission(ResourceType.LLM_MODEL, None, PermissionAction.READ)
    result = await acs.authorize('user_alice', permission)
    print(f'   管理员权限检查: {"✅ 允许" if result else "❌ 拒绝"}')
    
    # 隐私保护演示
    print('\n3. 隐私保护演示')
    privacy_system = PrivacyProtectionSystem({})
    record = await privacy_system.protect_data("test@example.com", DataType.EMAIL, PrivacyLevel.HIGH)
    print(f'   数据保护: 原始={record.original_value}, 匿名={record.anonymized_value}')
    
    print('\n✅ 演示完成')

if __name__ == "__main__":
    asyncio.run(quick_demo())
