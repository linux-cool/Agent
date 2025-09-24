# input_validation.py
"""
第7章 安全隐私防护体系 - 输入验证与过滤框架
实现智能体系统的输入验证、过滤和清理功能
"""

import asyncio
import logging
import re
import html
import json
from typing import Dict, List, Any, Optional, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import uuid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ValidationType(Enum):
    """验证类型枚举"""
    TEXT = "文本"
    EMAIL = "邮箱"
    URL = "URL"
    PHONE = "电话"
    NUMBER = "数字"
    JSON = "JSON"
    SQL = "SQL"
    HTML = "HTML"
    SCRIPT = "脚本"

class FilterType(Enum):
    """过滤类型枚举"""
    SANITIZE = "清理"
    ESCAPE = "转义"
    REMOVE = "移除"
    REPLACE = "替换"
    TRUNCATE = "截断"

@dataclass
class ValidationRule:
    """验证规则"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    validation_type: ValidationType = ValidationType.TEXT
    pattern: str = ""
    min_length: int = 0
    max_length: int = 1000
    required: bool = True
    custom_validator: Optional[Callable] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "validation_type": self.validation_type.value,
            "pattern": self.pattern,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "required": self.required
        }

@dataclass
class FilterRule:
    """过滤规则"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    filter_type: FilterType = FilterType.SANITIZE
    pattern: str = ""
    replacement: str = ""
    max_length: int = 1000
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "filter_type": self.filter_type.value,
            "pattern": self.pattern,
            "replacement": self.replacement,
            "max_length": self.max_length
        }

@dataclass
class ValidationResult:
    """验证结果"""
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sanitized_data: str = ""
    original_data: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "sanitized_data": self.sanitized_data,
            "original_data": self.original_data
        }

class InputValidator:
    """输入验证器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validation_rules: List[ValidationRule] = []
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """初始化默认验证规则"""
        # 文本验证规则
        text_rule = ValidationRule(
            name="文本验证",
            validation_type=ValidationType.TEXT,
            min_length=1,
            max_length=1000
        )
        self.validation_rules.append(text_rule)
        
        # 邮箱验证规则
        email_rule = ValidationRule(
            name="邮箱验证",
            validation_type=ValidationType.EMAIL,
            pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            min_length=5,
            max_length=254
        )
        self.validation_rules.append(email_rule)
        
        # URL验证规则
        url_rule = ValidationRule(
            name="URL验证",
            validation_type=ValidationType.URL,
            pattern=r'^https?://[^\s/$.?#].[^\s]*$',
            min_length=10,
            max_length=2048
        )
        self.validation_rules.append(url_rule)
    
    async def validate(self, data: str, validation_type: ValidationType = ValidationType.TEXT) -> ValidationResult:
        """验证输入数据"""
        try:
            result = ValidationResult(original_data=data, sanitized_data=data)
            
            # 查找对应的验证规则
            rule = next((r for r in self.validation_rules if r.validation_type == validation_type), None)
            if not rule:
                result.errors.append(f"未找到验证规则: {validation_type.value}")
                result.valid = False
                return result
            
            # 基本验证
            if rule.required and not data.strip():
                result.errors.append("输入不能为空")
                result.valid = False
            
            if len(data) < rule.min_length:
                result.errors.append(f"输入长度不能少于 {rule.min_length} 字符")
                result.valid = False
            
            if len(data) > rule.max_length:
                result.errors.append(f"输入长度不能超过 {rule.max_length} 字符")
                result.valid = False
            
            # 模式验证
            if rule.pattern and not re.match(rule.pattern, data):
                result.errors.append(f"输入格式不符合 {rule.name} 要求")
                result.valid = False
            
            # 自定义验证
            if rule.custom_validator:
                try:
                    custom_result = rule.custom_validator(data)
                    if not custom_result:
                        result.errors.append("自定义验证失败")
                        result.valid = False
                except Exception as e:
                    result.errors.append(f"自定义验证错误: {str(e)}")
                    result.valid = False
            
            return result
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return ValidationResult(valid=False, errors=[str(e)], original_data=data)

class InputFilter:
    """输入过滤器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.filter_rules: List[FilterRule] = []
        self._initialize_default_filters()
    
    def _initialize_default_filters(self):
        """初始化默认过滤规则"""
        # HTML标签过滤
        html_filter = FilterRule(
            name="HTML标签过滤",
            filter_type=FilterType.REMOVE,
            pattern=r'<[^>]+>'
        )
        self.filter_rules.append(html_filter)
        
        # 脚本标签过滤
        script_filter = FilterRule(
            name="脚本标签过滤",
            filter_type=FilterType.REMOVE,
            pattern=r'<script[^>]*>.*?</script>'
        )
        self.filter_rules.append(script_filter)
        
        # SQL注入过滤
        sql_filter = FilterRule(
            name="SQL注入过滤",
            filter_type=FilterType.REPLACE,
            pattern=r"([';]|--|/\*|\*/|xp_|sp_)",
            replacement=""
        )
        self.filter_rules.append(sql_filter)
        
        # 长度截断
        length_filter = FilterRule(
            name="长度截断",
            filter_type=FilterType.TRUNCATE,
            max_length=1000
        )
        self.filter_rules.append(length_filter)
    
    async def filter(self, data: str, filter_types: List[FilterType] = None) -> str:
        """过滤输入数据"""
        try:
            filtered_data = data
            
            # 应用过滤规则
            for rule in self.filter_rules:
                if filter_types and rule.filter_type not in filter_types:
                    continue
                
                if rule.filter_type == FilterType.REMOVE:
                    filtered_data = re.sub(rule.pattern, '', filtered_data, flags=re.IGNORECASE | re.DOTALL)
                elif rule.filter_type == FilterType.REPLACE:
                    filtered_data = re.sub(rule.pattern, rule.replacement, filtered_data, flags=re.IGNORECASE)
                elif rule.filter_type == FilterType.ESCAPE:
                    filtered_data = html.escape(filtered_data)
                elif rule.filter_type == FilterType.TRUNCATE:
                    if len(filtered_data) > rule.max_length:
                        filtered_data = filtered_data[:rule.max_length]
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"Filtering failed: {e}")
            return data

class InputSanitizer:
    """输入清理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validator = InputValidator(config.get("validation", {}))
        self.filter = InputFilter(config.get("filtering", {}))
    
    async def sanitize(self, data: str, validation_type: ValidationType = ValidationType.TEXT) -> ValidationResult:
        """清理输入数据"""
        try:
            # 首先过滤
            filtered_data = await self.filter.filter(data)
            
            # 然后验证
            result = await self.validator.validate(filtered_data, validation_type)
            
            # 更新清理后的数据
            result.sanitized_data = filtered_data
            
            return result
            
        except Exception as e:
            logger.error(f"Sanitization failed: {e}")
            return ValidationResult(valid=False, errors=[str(e)], original_data=data)

class InputValidationSystem:
    """输入验证系统主类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sanitizer = InputSanitizer(config)
        self.validation_stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "filtered_inputs": 0
        }
    
    async def process_input(self, data: str, validation_type: ValidationType = ValidationType.TEXT) -> ValidationResult:
        """处理输入数据"""
        try:
            self.validation_stats["total_validations"] += 1
            
            result = await self.sanitizer.sanitize(data, validation_type)
            
            if result.valid:
                self.validation_stats["successful_validations"] += 1
            else:
                self.validation_stats["failed_validations"] += 1
            
            if result.sanitized_data != result.original_data:
                self.validation_stats["filtered_inputs"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Input processing failed: {e}")
            return ValidationResult(valid=False, errors=[str(e)], original_data=data)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.validation_stats.copy()

# 示例用法
async def main_demo():
    """输入验证系统演示"""
    config = {
        "validation": {},
        "filtering": {}
    }
    
    # 创建输入验证系统
    validation_system = InputValidationSystem(config)
    
    print("🛡️ 输入验证与过滤系统演示")
    print("=" * 50)
    
    # 测试用例
    test_cases = [
        {
            "input": "Hello, world!",
            "type": ValidationType.TEXT,
            "description": "正常文本"
        },
        {
            "input": "<script>alert('XSS')</script>Hello",
            "type": ValidationType.TEXT,
            "description": "包含脚本标签"
        },
        {
            "input": "user@example.com",
            "type": ValidationType.EMAIL,
            "description": "有效邮箱"
        },
        {
            "input": "invalid-email",
            "type": ValidationType.EMAIL,
            "description": "无效邮箱"
        },
        {
            "input": "https://www.example.com",
            "type": ValidationType.URL,
            "description": "有效URL"
        },
        {
            "input": "SELECT * FROM users WHERE id = 1 OR 1=1",
            "type": ValidationType.TEXT,
            "description": "SQL注入尝试"
        }
    ]
    
    print("\n🔍 输入验证演示:")
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['description']}")
        print(f"   输入: {test_case['input']}")
        
        result = await validation_system.process_input(
            test_case["input"], 
            test_case["type"]
        )
        
        print(f"   🛡️ 验证结果:")
        print(f"     有效: {'✅ 是' if result.valid else '❌ 否'}")
        print(f"     清理后: {result.sanitized_data}")
        
        if result.errors:
            print(f"     错误: {', '.join(result.errors)}")
        
        if result.warnings:
            print(f"     警告: {', '.join(result.warnings)}")
        
        # 模拟处理间隔
        await asyncio.sleep(0.3)
    
    # 显示统计信息
    print(f"\n📊 验证统计:")
    stats = validation_system.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n🎉 输入验证系统演示完成")

if __name__ == "__main__":
    asyncio.run(main_demo())
