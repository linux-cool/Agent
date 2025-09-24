# privacy_protection.py
"""
第7章 安全隐私防护体系 - 隐私保护技术
实现数据加密、脱敏、差分隐私等隐私保护功能
"""

import asyncio
import logging
import hashlib
import secrets
import base64
import re
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import uuid
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataType(Enum):
    """数据类型枚举"""
    TEXT = "文本"
    EMAIL = "邮箱"
    PHONE = "电话"
    ID_CARD = "身份证"
    CREDIT_CARD = "信用卡"
    IP_ADDRESS = "IP地址"
    NAME = "姓名"
    ADDRESS = "地址"

class PrivacyLevel(Enum):
    """隐私级别枚举"""
    LOW = "低"
    MEDIUM = "中"
    HIGH = "高"
    CRITICAL = "关键"

class EncryptionAlgorithm(Enum):
    """加密算法枚举"""
    AES = "AES"
    RSA = "RSA"
    FERNET = "FERNET"

@dataclass
class PrivacyPolicy:
    """隐私策略"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    data_type: DataType = DataType.TEXT
    privacy_level: PrivacyLevel = PrivacyLevel.MEDIUM
    encryption_required: bool = True
    anonymization_required: bool = False
    retention_days: int = 365
    access_control: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "data_type": self.data_type.value,
            "privacy_level": self.privacy_level.value,
            "encryption_required": self.encryption_required,
            "anonymization_required": self.anonymization_required,
            "retention_days": self.retention_days,
            "access_control": self.access_control,
            "created_at": self.created_at.isoformat()
        }

@dataclass
class DataRecord:
    """数据记录"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data_type: DataType = DataType.TEXT
    original_value: str = ""
    encrypted_value: str = ""
    anonymized_value: str = ""
    privacy_level: PrivacyLevel = PrivacyLevel.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "data_type": self.data_type.value,
            "original_value": self.original_value,
            "encrypted_value": self.encrypted_value,
            "anonymized_value": self.anonymized_value,
            "privacy_level": self.privacy_level.value,
            "created_at": self.created_at.isoformat()
        }

class EncryptionService:
    """加密服务"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.algorithm = EncryptionAlgorithm(config.get("algorithm", "FERNET"))
        self.key = self._generate_key()
    
    def _generate_key(self) -> bytes:
        """生成加密密钥"""
        try:
            return secrets.token_bytes(32)
        except Exception as e:
            logger.error(f"Key generation failed: {e}")
            raise
    
    def encrypt(self, data: str) -> str:
        """加密数据"""
        try:
            # 简化的加密实现
            return base64.b64encode(data.encode('utf-8')).decode('utf-8')
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt(self, encrypted_data: str) -> str:
        """解密数据"""
        try:
            # 简化的解密实现
            return base64.b64decode(encrypted_data.encode('utf-8')).decode('utf-8')
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise

class DataAnonymizer:
    """数据匿名化器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.anonymization_rules = self._initialize_rules()
    
    def _initialize_rules(self) -> Dict[DataType, Callable]:
        """初始化匿名化规则"""
        return {
            DataType.EMAIL: self._anonymize_email,
            DataType.PHONE: self._anonymize_phone,
            DataType.ID_CARD: self._anonymize_id_card,
            DataType.CREDIT_CARD: self._anonymize_credit_card,
            DataType.IP_ADDRESS: self._anonymize_ip,
            DataType.NAME: self._anonymize_name,
            DataType.ADDRESS: self._anonymize_address,
            DataType.TEXT: self._anonymize_text
        }
    
    def anonymize(self, data: str, data_type: DataType) -> str:
        """匿名化数据"""
        try:
            anonymizer = self.anonymization_rules.get(data_type, self._anonymize_text)
            return anonymizer(data)
        except Exception as e:
            logger.error(f"Anonymization failed: {e}")
            return data
    
    def _anonymize_email(self, email: str) -> str:
        """匿名化邮箱"""
        if '@' in email:
            local, domain = email.split('@', 1)
            return f"{local[0]}***@{domain}"
        return "***@***"
    
    def _anonymize_phone(self, phone: str) -> str:
        """匿名化电话"""
        digits = re.sub(r'\D', '', phone)
        if len(digits) >= 4:
            return f"{digits[:2]}***{digits[-2:]}"
        return "***"
    
    def _anonymize_id_card(self, id_card: str) -> str:
        """匿名化身份证"""
        if len(id_card) == 18:
            return f"{id_card[:6]}********{id_card[-4:]}"
        return "***"
    
    def _anonymize_credit_card(self, card: str) -> str:
        """匿名化信用卡"""
        digits = re.sub(r'\D', '', card)
        if len(digits) >= 4:
            return f"****-****-****-{digits[-4:]}"
        return "****"
    
    def _anonymize_ip(self, ip: str) -> str:
        """匿名化IP地址"""
        parts = ip.split('.')
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.***.***"
        return "***"
    
    def _anonymize_name(self, name: str) -> str:
        """匿名化姓名"""
        if len(name) >= 2:
            return f"{name[0]}***"
        return "***"
    
    def _anonymize_address(self, address: str) -> str:
        """匿名化地址"""
        return "***"
    
    def _anonymize_text(self, text: str) -> str:
        """匿名化文本"""
        return "***"

class DifferentialPrivacyEngine:
    """差分隐私引擎"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.epsilon = config.get("epsilon", 1.0)  # 隐私预算
        self.delta = config.get("delta", 1e-5)  # 失败概率
    
    def add_noise(self, value: float, sensitivity: float = 1.0) -> float:
        """添加拉普拉斯噪声"""
        try:
            import numpy as np
            
            # 计算噪声规模
            scale = sensitivity / self.epsilon
            
            # 生成拉普拉斯噪声
            noise = np.random.laplace(0, scale)
            
            return value + noise
        except Exception as e:
            logger.error(f"Noise addition failed: {e}")
            return value
    
    def calculate_sensitivity(self, data: List[float]) -> float:
        """计算敏感度"""
        try:
            if len(data) < 2:
                return 0.0
            
            # 简化的敏感度计算
            return max(data) - min(data)
        except Exception as e:
            logger.error(f"Sensitivity calculation failed: {e}")
            return 1.0

class DataMaskingService:
    """数据脱敏服务"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.masking_rules = self._initialize_masking_rules()
    
    def _initialize_masking_rules(self) -> Dict[str, str]:
        """初始化脱敏规则"""
        return {
            "email": r"([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
            "phone": r"(\d{3})\d{4}(\d{4})",
            "id_card": r"(\d{6})\d{8}(\d{4})",
            "credit_card": r"(\d{4})\d{8}(\d{4})",
            "ip": r"(\d{1,3}\.\d{1,3})\.\d{1,3}\.\d{1,3}"
        }
    
    def mask_data(self, data: str, data_type: str) -> str:
        """脱敏数据"""
        try:
            if data_type in self.masking_rules:
                pattern = self.masking_rules[data_type]
                if data_type == "email":
                    return re.sub(pattern, r"\1***@\2", data)
                elif data_type == "phone":
                    return re.sub(pattern, r"\1****\2", data)
                elif data_type == "id_card":
                    return re.sub(pattern, r"\1********\2", data)
                elif data_type == "credit_card":
                    return re.sub(pattern, r"\1********\2", data)
                elif data_type == "ip":
                    return re.sub(pattern, r"\1.***.***", data)
            
            return data
        except Exception as e:
            logger.error(f"Data masking failed: {e}")
            return data

class PrivacyProtectionSystem:
    """隐私保护系统主类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.encryption_service = EncryptionService(config.get("encryption", {}))
        self.anonymizer = DataAnonymizer(config.get("anonymization", {}))
        self.dp_engine = DifferentialPrivacyEngine(config.get("differential_privacy", {}))
        self.masking_service = DataMaskingService(config.get("masking", {}))
        self.privacy_policies: Dict[str, PrivacyPolicy] = {}
        self.data_records: Dict[str, DataRecord] = {}
        self._initialize_default_policies()
    
    def _initialize_default_policies(self):
        """初始化默认隐私策略"""
        # 邮箱隐私策略
        email_policy = PrivacyPolicy(
            name="邮箱隐私策略",
            data_type=DataType.EMAIL,
            privacy_level=PrivacyLevel.HIGH,
            encryption_required=True,
            anonymization_required=True,
            retention_days=365
        )
        self.privacy_policies[email_policy.id] = email_policy
        
        # 电话隐私策略
        phone_policy = PrivacyPolicy(
            name="电话隐私策略",
            data_type=DataType.PHONE,
            privacy_level=PrivacyLevel.HIGH,
            encryption_required=True,
            anonymization_required=True,
            retention_days=180
        )
        self.privacy_policies[phone_policy.id] = phone_policy
    
    async def protect_data(self, data: str, data_type: DataType, 
                          privacy_level: PrivacyLevel = PrivacyLevel.MEDIUM) -> DataRecord:
        """保护数据"""
        try:
            # 创建数据记录
            record = DataRecord(
                data_type=data_type,
                original_value=data,
                privacy_level=privacy_level
            )
            
            # 根据隐私级别决定保护措施
            if privacy_level in [PrivacyLevel.HIGH, PrivacyLevel.CRITICAL]:
                # 加密
                record.encrypted_value = self.encryption_service.encrypt(data)
                
                # 匿名化
                record.anonymized_value = self.anonymizer.anonymize(data, data_type)
            
            # 存储记录
            self.data_records[record.id] = record
            
            return record
            
        except Exception as e:
            logger.error(f"Data protection failed: {e}")
            return DataRecord(original_value=data, data_type=data_type)
    
    async def retrieve_data(self, record_id: str, user_permissions: List[str] = None) -> Optional[str]:
        """检索数据"""
        try:
            record = self.data_records.get(record_id)
            if not record:
                return None
            
            # 检查权限
            if user_permissions and "admin" not in user_permissions:
                # 非管理员只能看到匿名化数据
                return record.anonymized_value
            
            # 管理员可以访问原始数据
            if record.encrypted_value:
                return self.encryption_service.decrypt(record.encrypted_value)
            else:
                return record.original_value
                
        except Exception as e:
            logger.error(f"Data retrieval failed: {e}")
            return None
    
    async def add_noise_to_statistics(self, values: List[float], sensitivity: float = None) -> List[float]:
        """为统计数据添加噪声"""
        try:
            if sensitivity is None:
                sensitivity = self.dp_engine.calculate_sensitivity(values)
            
            noisy_values = []
            for value in values:
                noisy_value = self.dp_engine.add_noise(value, sensitivity)
                noisy_values.append(noisy_value)
            
            return noisy_values
            
        except Exception as e:
            logger.error(f"Noise addition failed: {e}")
            return values
    
    async def mask_sensitive_data(self, data: str, data_types: List[str]) -> str:
        """脱敏敏感数据"""
        try:
            masked_data = data
            
            for data_type in data_types:
                masked_data = self.masking_service.mask_data(masked_data, data_type)
            
            return masked_data
            
        except Exception as e:
            logger.error(f"Data masking failed: {e}")
            return data
    
    def get_privacy_stats(self) -> Dict[str, Any]:
        """获取隐私保护统计"""
        try:
            total_records = len(self.data_records)
            encrypted_records = len([r for r in self.data_records.values() if r.encrypted_value])
            anonymized_records = len([r for r in self.data_records.values() if r.anonymized_value])
            
            return {
                "total_records": total_records,
                "encrypted_records": encrypted_records,
                "anonymized_records": anonymized_records,
                "privacy_policies": len(self.privacy_policies),
                "encryption_rate": encrypted_records / total_records if total_records > 0 else 0,
                "anonymization_rate": anonymized_records / total_records if total_records > 0 else 0
            }
        except Exception as e:
            logger.error(f"Privacy stats failed: {e}")
            return {}

# 示例用法
async def main_demo():
    """隐私保护系统演示"""
    config = {
        "encryption": {
            "algorithm": "FERNET"
        },
        "anonymization": {},
        "differential_privacy": {
            "epsilon": 1.0,
            "delta": 1e-5
        },
        "masking": {}
    }
    
    # 创建隐私保护系统
    privacy_system = PrivacyProtectionSystem(config)
    
    print("🔒 隐私保护系统演示")
    print("=" * 50)
    
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
            "data": "Hello, this is a normal text",
            "type": DataType.TEXT,
            "level": PrivacyLevel.LOW,
            "description": "普通文本"
        }
    ]
    
    print("\n🔐 数据保护演示:")
    protected_records = []
    
    for i, test_case in enumerate(test_data, 1):
        print(f"\n{i}. {test_case['description']}")
        print(f"   原始数据: {test_case['data']}")
        
        # 保护数据
        record = await privacy_system.protect_data(
            test_case["data"],
            test_case["type"],
            test_case["level"]
        )
        
        protected_records.append(record)
        
        print(f"   加密数据: {record.encrypted_value[:50] + '...' if record.encrypted_value else '无'}")
        print(f"   匿名数据: {record.anonymized_value}")
        
        # 模拟处理间隔
        await asyncio.sleep(0.3)
    
    # 数据检索演示
    print(f"\n🔍 数据检索演示:")
    
    for i, record in enumerate(protected_records, 1):
        print(f"\n{i}. 检索记录 {record.id}")
        
        # 普通用户检索
        user_data = await privacy_system.retrieve_data(record.id, ["user"])
        print(f"   普通用户看到: {user_data}")
        
        # 管理员检索
        admin_data = await privacy_system.retrieve_data(record.id, ["admin"])
        print(f"   管理员看到: {admin_data}")
    
    # 差分隐私演示
    print(f"\n📊 差分隐私演示:")
    
    # 原始统计数据
    original_stats = [100, 150, 200, 180, 120]
    print(f"   原始统计: {original_stats}")
    
    # 添加噪声
    noisy_stats = await privacy_system.add_noise_to_statistics(original_stats)
    print(f"   添加噪声后: {[round(x, 2) for x in noisy_stats]}")
    
    # 数据脱敏演示
    print(f"\n🎭 数据脱敏演示:")
    
    sensitive_text = "我的邮箱是 user@example.com，电话是 13812345678"
    print(f"   原始文本: {sensitive_text}")
    
    masked_text = await privacy_system.mask_sensitive_data(
        sensitive_text, 
        ["email", "phone"]
    )
    print(f"   脱敏后: {masked_text}")
    
    # 显示统计信息
    print(f"\n📈 隐私保护统计:")
    stats = privacy_system.get_privacy_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value}")
    
    print("\n🎉 隐私保护系统演示完成")

if __name__ == "__main__":
    asyncio.run(main_demo())
