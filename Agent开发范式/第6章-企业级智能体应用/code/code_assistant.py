# code_assistant.py
"""
第6章 企业级智能体应用 - 代码助手
实现基于AI的代码助手，包括代码生成、代码审查、文档生成、测试生成等功能
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
import ast
import tokenize
import io

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodeLanguage(Enum):
    """编程语言枚举"""
    PYTHON = "Python"
    JAVASCRIPT = "JavaScript"
    JAVA = "Java"
    CPP = "C++"
    CSHARP = "C#"
    GO = "Go"
    RUST = "Rust"
    TYPESCRIPT = "TypeScript"

class CodeQuality(Enum):
    """代码质量等级"""
    EXCELLENT = "优秀"
    GOOD = "良好"
    FAIR = "一般"
    POOR = "较差"
    CRITICAL = "严重"

class ComplexityLevel(Enum):
    """复杂度等级"""
    LOW = "低"
    MEDIUM = "中"
    HIGH = "高"
    VERY_HIGH = "很高"

class TestType(Enum):
    """测试类型"""
    UNIT = "单元测试"
    INTEGRATION = "集成测试"
    FUNCTIONAL = "功能测试"
    PERFORMANCE = "性能测试"
    SECURITY = "安全测试"

class DocumentationType(Enum):
    """文档类型"""
    API_DOCS = "API文档"
    README = "README"
    CODE_COMMENTS = "代码注释"
    ARCHITECTURE = "架构文档"
    USER_GUIDE = "用户指南"

@dataclass
class CodeFile:
    """代码文件"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    path: str = ""
    language: CodeLanguage = CodeLanguage.PYTHON
    content: str = ""
    size: int = 0
    lines_of_code: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "path": self.path,
            "language": self.language.value,
            "content": self.content,
            "size": self.size,
            "lines_of_code": self.lines_of_code,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }

@dataclass
class CodeIssue:
    """代码问题"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    file_id: str = ""
    line_number: int = 0
    column_number: int = 0
    issue_type: str = ""
    severity: str = ""
    message: str = ""
    suggestion: str = ""
    rule_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "file_id": self.file_id,
            "line_number": self.line_number,
            "column_number": self.column_number,
            "issue_type": self.issue_type,
            "severity": self.severity,
            "message": self.message,
            "suggestion": self.suggestion,
            "rule_id": self.rule_id,
            "created_at": self.created_at.isoformat()
        }

@dataclass
class CodeMetrics:
    """代码指标"""
    file_id: str = ""
    cyclomatic_complexity: int = 0
    cognitive_complexity: int = 0
    maintainability_index: float = 0.0
    technical_debt: float = 0.0
    code_smells: int = 0
    duplications: int = 0
    test_coverage: float = 0.0
    security_hotspots: int = 0
    bugs: int = 0
    vulnerabilities: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "file_id": self.file_id,
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "cognitive_complexity": self.cognitive_complexity,
            "maintainability_index": self.maintainability_index,
            "technical_debt": self.technical_debt,
            "code_smells": self.code_smells,
            "duplications": self.duplications,
            "test_coverage": self.test_coverage,
            "security_hotspots": self.security_hotspots,
            "bugs": self.bugs,
            "vulnerabilities": self.vulnerabilities
        }

@dataclass
class TestCase:
    """测试用例"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    test_type: TestType = TestType.UNIT
    code: str = ""
    expected_result: str = ""
    setup_code: str = ""
    teardown_code: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "test_type": self.test_type.value,
            "code": self.code,
            "expected_result": self.expected_result,
            "setup_code": self.setup_code,
            "teardown_code": self.teardown_code,
            "parameters": self.parameters,
            "created_at": self.created_at.isoformat()
        }

@dataclass
class Documentation:
    """文档"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    content: str = ""
    doc_type: DocumentationType = DocumentationType.CODE_COMMENTS
    language: str = "markdown"
    file_id: Optional[str] = None
    function_name: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "doc_type": self.doc_type.value,
            "language": self.language,
            "file_id": self.file_id,
            "function_name": self.function_name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

class CodeAnalyzer:
    """代码分析器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.analysis_rules = self._load_analysis_rules()
    
    def _load_analysis_rules(self) -> Dict[str, Dict[str, Any]]:
        """加载分析规则"""
        return {
            "python": {
                "style_rules": [
                    {"rule": "line_length", "max": 120, "severity": "warning"},
                    {"rule": "function_length", "max": 50, "severity": "warning"},
                    {"rule": "class_length", "max": 200, "severity": "warning"},
                    {"rule": "complexity", "max": 10, "severity": "error"}
                ],
                "security_rules": [
                    {"rule": "sql_injection", "pattern": r"execute\s*\(.*%.*\)", "severity": "critical"},
                    {"rule": "eval_usage", "pattern": r"eval\s*\(", "severity": "critical"},
                    {"rule": "exec_usage", "pattern": r"exec\s*\(", "severity": "critical"}
                ]
            },
            "javascript": {
                "style_rules": [
                    {"rule": "line_length", "max": 120, "severity": "warning"},
                    {"rule": "function_length", "max": 50, "severity": "warning"},
                    {"rule": "complexity", "max": 10, "severity": "error"}
                ],
                "security_rules": [
                    {"rule": "eval_usage", "pattern": r"eval\s*\(", "severity": "critical"},
                    {"rule": "innerHTML", "pattern": r"innerHTML\s*=", "severity": "warning"}
                ]
            }
        }
    
    async def analyze_code(self, code_file: CodeFile) -> Tuple[List[CodeIssue], CodeMetrics]:
        """分析代码"""
        try:
            issues = []
            metrics = CodeMetrics(file_id=code_file.id)
            
            # 基础分析
            await self._analyze_basic_metrics(code_file, metrics)
            
            # 代码质量分析
            await self._analyze_code_quality(code_file, issues, metrics)
            
            # 安全检查
            await self._analyze_security(code_file, issues)
            
            # 复杂度分析
            await self._analyze_complexity(code_file, metrics)
            
            return issues, metrics
            
        except Exception as e:
            logger.error(f"Code analysis failed: {e}")
            return [], CodeMetrics(file_id=code_file.id)
    
    async def _analyze_basic_metrics(self, code_file: CodeFile, metrics: CodeMetrics):
        """分析基础指标"""
        try:
            lines = code_file.content.split('\n')
            code_file.lines_of_code = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
            code_file.size = len(code_file.content.encode('utf-8'))
            
            # 计算代码行数
            metrics.lines_of_code = code_file.lines_of_code
            
        except Exception as e:
            logger.error(f"Basic metrics analysis failed: {e}")
    
    async def _analyze_code_quality(self, code_file: CodeFile, issues: List[CodeIssue], metrics: CodeMetrics):
        """分析代码质量"""
        try:
            rules = self.analysis_rules.get(code_file.language.value.lower(), {})
            style_rules = rules.get("style_rules", [])
            
            lines = code_file.content.split('\n')
            
            for i, line in enumerate(lines, 1):
                # 检查行长度
                for rule in style_rules:
                    if rule["rule"] == "line_length" and len(line) > rule["max"]:
                        issue = CodeIssue(
                            file_id=code_file.id,
                            line_number=i,
                            issue_type="style",
                            severity=rule["severity"],
                            message=f"行长度超过{rule['max']}字符",
                            suggestion="考虑将长行拆分为多行",
                            rule_id=rule["rule"]
                        )
                        issues.append(issue)
                
                # 检查TODO注释
                if "TODO" in line.upper() or "FIXME" in line.upper():
                    issue = CodeIssue(
                        file_id=code_file.id,
                        line_number=i,
                        issue_type="maintenance",
                        severity="info",
                        message="发现TODO或FIXME注释",
                        suggestion="及时处理待办事项",
                        rule_id="todo_comment"
                    )
                    issues.append(issue)
            
            # 计算代码异味
            metrics.code_smells = len([issue for issue in issues if issue.severity in ["warning", "error"]])
            
        except Exception as e:
            logger.error(f"Code quality analysis failed: {e}")
    
    async def _analyze_security(self, code_file: CodeFile, issues: List[CodeIssue]):
        """安全检查"""
        try:
            rules = self.analysis_rules.get(code_file.language.value.lower(), {})
            security_rules = rules.get("security_rules", [])
            
            lines = code_file.content.split('\n')
            
            for i, line in enumerate(lines, 1):
                for rule in security_rules:
                    if re.search(rule["pattern"], line, re.IGNORECASE):
                        issue = CodeIssue(
                            file_id=code_file.id,
                            line_number=i,
                            issue_type="security",
                            severity=rule["severity"],
                            message=f"发现潜在安全问题: {rule['rule']}",
                            suggestion="使用更安全的替代方案",
                            rule_id=rule["rule"]
                        )
                        issues.append(issue)
            
        except Exception as e:
            logger.error(f"Security analysis failed: {e}")
    
    async def _analyze_complexity(self, code_file: CodeFile, metrics: CodeMetrics):
        """复杂度分析"""
        try:
            if code_file.language == CodeLanguage.PYTHON:
                # 简化的圈复杂度计算
                complexity_keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'and', 'or']
                complexity_count = 0
                
                for line in code_file.content.split('\n'):
                    for keyword in complexity_keywords:
                        complexity_count += line.count(f' {keyword} ')
                
                metrics.cyclomatic_complexity = complexity_count + 1
                metrics.cognitive_complexity = complexity_count
                
                # 计算可维护性指数
                if metrics.lines_of_code > 0:
                    metrics.maintainability_index = max(0, 100 - (metrics.cyclomatic_complexity * 2) - (metrics.code_smells * 0.5))
            
        except Exception as e:
            logger.error(f"Complexity analysis failed: {e}")

class CodeGenerator:
    """代码生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.templates = self._load_templates()
        self.code_patterns = self._load_code_patterns()
    
    def _load_templates(self) -> Dict[str, Dict[str, str]]:
        """加载代码模板"""
        return {
            "python_function": """
def {function_name}({parameters}):
    \"\"\"
    {description}
    
    Args:
        {args_doc}
    
    Returns:
        {return_doc}
    \"\"\"
    {implementation}
    return {return_value}
""",
            "python_class": """
class {class_name}:
    \"\"\"
    {description}
    \"\"\"
    
    def __init__(self{init_params}):
        {init_implementation}
    
    def {method_name}(self{method_params}):
        \"\"\"
        {method_description}
        \"\"\"
        {method_implementation}
        return {method_return}
""",
            "javascript_function": """
/**
 * {description}
 * @param {{{param_types}}} {param_names}
 * @returns {{{return_type}}}
 */
function {function_name}({parameters}) {{
    {implementation}
    return {return_value};
}}
""",
            "api_endpoint": """
@app.route('/{endpoint}', methods=['{methods}'])
def {function_name}():
    \"\"\"
    {description}
    \"\"\"
    try:
        {implementation}
        return jsonify({{'status': 'success', 'data': {return_data}}}), 200
    except Exception as e:
        return jsonify({{'status': 'error', 'message': str(e)}}), 500
"""
        }
    
    def _load_code_patterns(self) -> Dict[str, List[str]]:
        """加载代码模式"""
        return {
            "data_processing": [
                "数据清洗", "数据转换", "数据验证", "数据聚合"
            ],
            "api_development": [
                "REST API", "GraphQL", "微服务", "API网关"
            ],
            "database_operations": [
                "CRUD操作", "查询优化", "事务处理", "数据迁移"
            ],
            "file_operations": [
                "文件读取", "文件写入", "文件上传", "文件下载"
            ]
        }
    
    async def generate_function(self, description: str, language: CodeLanguage, 
                              context: Dict[str, Any] = None) -> str:
        """生成函数"""
        try:
            # 解析描述
            function_info = self._parse_function_description(description)
            
            # 选择模板
            template_key = f"{language.value.lower()}_function"
            template = self.templates.get(template_key, self.templates["python_function"])
            
            # 生成代码
            generated_code = template.format(
                function_name=function_info["name"],
                parameters=function_info["parameters"],
                description=function_info["description"],
                args_doc=function_info["args_doc"],
                return_doc=function_info["return_doc"],
                implementation=function_info["implementation"],
                return_value=function_info["return_value"]
            )
            
            return generated_code.strip()
            
        except Exception as e:
            logger.error(f"Function generation failed: {e}")
            return f"# 生成失败: {e}"
    
    def _parse_function_description(self, description: str) -> Dict[str, str]:
        """解析函数描述"""
        # 简化的解析逻辑
        function_name = "generated_function"
        if "函数" in description:
            # 尝试提取函数名
            match = re.search(r'(\w+)函数', description)
            if match:
                function_name = match.group(1)
        
        # 根据描述生成参数
        parameters = ""
        if "参数" in description:
            parameters = "param1, param2"
        
        # 生成实现
        implementation = "    # TODO: 实现具体逻辑\n    pass"
        if "计算" in description:
            implementation = "    result = param1 + param2\n    return result"
            return_value = "result"
        else:
            return_value = "None"
        
        return {
            "name": function_name,
            "parameters": parameters,
            "description": description,
            "args_doc": "param1: 参数1\n        param2: 参数2",
            "return_doc": "返回值描述",
            "implementation": implementation,
            "return_value": return_value
        }
    
    async def generate_class(self, description: str, language: CodeLanguage, 
                           context: Dict[str, Any] = None) -> str:
        """生成类"""
        try:
            # 解析描述
            class_info = self._parse_class_description(description)
            
            # 选择模板
            template_key = f"{language.value.lower()}_class"
            template = self.templates.get(template_key, self.templates["python_class"])
            
            # 生成代码
            generated_code = template.format(
                class_name=class_info["name"],
                description=class_info["description"],
                init_params=class_info["init_params"],
                init_implementation=class_info["init_implementation"],
                method_name=class_info["method_name"],
                method_params=class_info["method_params"],
                method_description=class_info["method_description"],
                method_implementation=class_info["method_implementation"],
                method_return=class_info["method_return"]
            )
            
            return generated_code.strip()
            
        except Exception as e:
            logger.error(f"Class generation failed: {e}")
            return f"# 生成失败: {e}"
    
    def _parse_class_description(self, description: str) -> Dict[str, str]:
        """解析类描述"""
        class_name = "GeneratedClass"
        if "类" in description:
            match = re.search(r'(\w+)类', description)
            if match:
                class_name = match.group(1)
        
        return {
            "name": class_name,
            "description": description,
            "init_params": ", name=None",
            "init_implementation": "        self.name = name",
            "method_name": "process",
            "method_params": ", data=None",
            "method_description": "处理方法",
            "method_implementation": "        # TODO: 实现处理逻辑\n        return data",
            "method_return": "data"
        }

class TestGenerator:
    """测试生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.test_templates = self._load_test_templates()
    
    def _load_test_templates(self) -> Dict[str, str]:
        """加载测试模板"""
        return {
            "python_unittest": """
import unittest
from {module_name} import {function_name}

class Test{FunctionName}(unittest.TestCase):
    \"\"\"
    {function_name}的单元测试
    \"\"\"
    
    def test_{function_name}_normal_case(self):
        \"\"\"
        测试正常情况
        \"\"\"
        {test_implementation}
        self.assertEqual(result, expected_result)
    
    def test_{function_name}_edge_case(self):
        \"\"\"
        测试边界情况
        \"\"\"
        {edge_case_implementation}
    
    def test_{function_name}_error_case(self):
        \"\"\"
        测试错误情况
        \"\"\"
        {error_case_implementation}
        self.assertRaises(Exception, {function_name}, invalid_input)

if __name__ == '__main__':
    unittest.main()
""",
            "python_pytest": """
import pytest
from {module_name} import {function_name}

def test_{function_name}_normal_case():
    \"\"\"
    测试正常情况
    \"\"\"
    {test_implementation}
    assert result == expected_result

def test_{function_name}_edge_case():
    \"\"\"
    测试边界情况
    \"\"\"
    {edge_case_implementation}

def test_{function_name}_error_case():
    \"\"\"
    测试错误情况
    \"\"\"
    {error_case_implementation}
    with pytest.raises(Exception):
        {function_name}(invalid_input)
""",
            "javascript_jest": """
const {{ {function_name} }} = require('./{module_name}');

describe('{FunctionName}', () => {{
    test('should handle normal case', () => {{
        {test_implementation}
        expect(result).toBe(expected_result);
    }});
    
    test('should handle edge case', () => {{
        {edge_case_implementation}
    }});
    
    test('should handle error case', () => {{
        {error_case_implementation}
        expect(() => {function_name}(invalid_input)).toThrow();
    }});
}});
"""
        }
    
    async def generate_tests(self, code_file: CodeFile, test_framework: str = "pytest") -> List[TestCase]:
        """生成测试用例"""
        try:
            test_cases = []
            
            # 分析代码文件，提取函数
            functions = self._extract_functions(code_file)
            
            for func_info in functions:
                test_case = await self._generate_function_tests(func_info, test_framework)
                test_cases.append(test_case)
            
            return test_cases
            
        except Exception as e:
            logger.error(f"Test generation failed: {e}")
            return []
    
    def _extract_functions(self, code_file: CodeFile) -> List[Dict[str, Any]]:
        """提取函数信息"""
        functions = []
        
        if code_file.language == CodeLanguage.PYTHON:
            try:
                tree = ast.parse(code_file.content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_info = {
                            "name": node.name,
                            "args": [arg.arg for arg in node.args.args],
                            "docstring": ast.get_docstring(node) or "",
                            "line_number": node.lineno
                        }
                        functions.append(func_info)
            except SyntaxError:
                logger.warning(f"Syntax error in {code_file.name}")
        
        return functions
    
    async def _generate_function_tests(self, func_info: Dict[str, Any], test_framework: str) -> TestCase:
        """为函数生成测试"""
        try:
            template_key = f"python_{test_framework}"
            template = self.test_templates.get(template_key, self.test_templates["python_pytest"])
            
            # 生成测试实现
            test_implementation = self._generate_test_implementation(func_info)
            edge_case_implementation = self._generate_edge_case_implementation(func_info)
            error_case_implementation = self._generate_error_case_implementation(func_info)
            
            # 生成测试代码
            test_code = template.format(
                module_name="module_name",  # 需要从文件路径提取
                function_name=func_info["name"],
                FunctionName=func_info["name"].title(),
                test_implementation=test_implementation,
                edge_case_implementation=edge_case_implementation,
                error_case_implementation=error_case_implementation
            )
            
            test_case = TestCase(
                name=f"test_{func_info['name']}",
                description=f"{func_info['name']}的测试用例",
                test_type=TestType.UNIT,
                code=test_code,
                expected_result="测试通过",
                setup_code="",
                teardown_code=""
            )
            
            return test_case
            
        except Exception as e:
            logger.error(f"Function test generation failed: {e}")
            return TestCase(name="test_failed", description="测试生成失败")
    
    def _generate_test_implementation(self, func_info: Dict[str, Any]) -> str:
        """生成测试实现"""
        if func_info["args"]:
            args = ", ".join([f"test_{arg}" for arg in func_info["args"]])
            return f"result = {func_info['name']}({args})\nexpected_result = 'expected_value'"
        else:
            return f"result = {func_info['name']}()\nexpected_result = 'expected_value'"
    
    def _generate_edge_case_implementation(self, func_info: Dict[str, Any]) -> str:
        """生成边界情况测试"""
        return f"# 测试边界情况\n# result = {func_info['name']}(edge_case_input)"
    
    def _generate_error_case_implementation(self, func_info: Dict[str, Any]) -> str:
        """生成错误情况测试"""
        return f"# 测试错误情况\n# invalid_input = None"

class DocumentationGenerator:
    """文档生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.doc_templates = self._load_doc_templates()
    
    def _load_doc_templates(self) -> Dict[str, str]:
        """加载文档模板"""
        return {
            "api_docs": """
# {title}

## 概述
{description}

## 接口列表

### {endpoint_name}
- **URL**: `{endpoint_url}`
- **方法**: `{method}`
- **描述**: {endpoint_description}

#### 请求参数
{request_params}

#### 响应格式
{response_format}

#### 示例
{example}
""",
            "readme": """
# {project_name}

## 项目描述
{description}

## 安装说明
{installation}

## 使用方法
{usage}

## API文档
{api_docs}

## 贡献指南
{contributing}

## 许可证
{license}
""",
            "function_doc": """
def {function_name}({parameters}):
    \"\"\"
    {description}
    
    Args:
        {args_doc}
    
    Returns:
        {return_doc}
    
    Raises:
        {raises_doc}
    
    Example:
        {example}
    \"\"\"
"""
        }
    
    async def generate_documentation(self, code_file: CodeFile, doc_type: DocumentationType) -> Documentation:
        """生成文档"""
        try:
            if doc_type == DocumentationType.API_DOCS:
                return await self._generate_api_docs(code_file)
            elif doc_type == DocumentationType.README:
                return await self._generate_readme(code_file)
            elif doc_type == DocumentationType.CODE_COMMENTS:
                return await self._generate_code_comments(code_file)
            else:
                return Documentation(title="未知文档类型", content="无法生成文档")
                
        except Exception as e:
            logger.error(f"Documentation generation failed: {e}")
            return Documentation(title="生成失败", content=f"文档生成失败: {e}")
    
    async def _generate_api_docs(self, code_file: CodeFile) -> Documentation:
        """生成API文档"""
        template = self.doc_templates["api_docs"]
        
        # 分析代码文件，提取API信息
        api_info = self._extract_api_info(code_file)
        
        content = template.format(
            title=f"{code_file.name} API文档",
            description="API接口文档",
            endpoint_name="示例接口",
            endpoint_url="/api/example",
            method="GET",
            endpoint_description="示例接口描述",
            request_params="无",
            response_format='{"status": "success", "data": {}}',
            example="curl -X GET /api/example"
        )
        
        return Documentation(
            title=f"{code_file.name} API文档",
            content=content,
            doc_type=DocumentationType.API_DOCS,
            file_id=code_file.id
        )
    
    async def _generate_readme(self, code_file: CodeFile) -> Documentation:
        """生成README文档"""
        template = self.doc_templates["readme"]
        
        content = template.format(
            project_name=code_file.name.replace('.py', ''),
            description="项目描述",
            installation="pip install -r requirements.txt",
            usage="python main.py",
            api_docs="详见API文档",
            contributing="欢迎提交PR",
            license="MIT License"
        )
        
        return Documentation(
            title="README",
            content=content,
            doc_type=DocumentationType.README,
            file_id=code_file.id
        )
    
    async def _generate_code_comments(self, code_file: CodeFile) -> Documentation:
        """生成代码注释"""
        # 分析代码并添加注释
        commented_code = self._add_code_comments(code_file.content, code_file.language)
        
        return Documentation(
            title=f"{code_file.name} 注释版本",
            content=commented_code,
            doc_type=DocumentationType.CODE_COMMENTS,
            file_id=code_file.id
        )
    
    def _extract_api_info(self, code_file: CodeFile) -> Dict[str, Any]:
        """提取API信息"""
        # 简化的API信息提取
        return {
            "endpoints": [],
            "methods": [],
            "parameters": []
        }
    
    def _add_code_comments(self, code: str, language: CodeLanguage) -> str:
        """为代码添加注释"""
        lines = code.split('\n')
        commented_lines = []
        
        for line in lines:
            if line.strip() and not line.strip().startswith('#'):
                # 为函数定义添加注释
                if line.strip().startswith('def '):
                    commented_lines.append(f"    # 函数定义")
                elif line.strip().startswith('class '):
                    commented_lines.append(f"    # 类定义")
                elif 'if ' in line and line.strip().startswith('if'):
                    commented_lines.append(f"    # 条件判断")
                elif 'for ' in line and line.strip().startswith('for'):
                    commented_lines.append(f"    # 循环语句")
            
            commented_lines.append(line)
        
        return '\n'.join(commented_lines)

class CodeAssistant:
    """代码助手主类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.code_analyzer = CodeAnalyzer(config.get("analysis", {}))
        self.code_generator = CodeGenerator(config.get("generation", {}))
        self.test_generator = TestGenerator(config.get("testing", {}))
        self.documentation_generator = DocumentationGenerator(config.get("documentation", {}))
        self.code_files: Dict[str, CodeFile] = {}
        self.analysis_results: Dict[str, Tuple[List[CodeIssue], CodeMetrics]] = {}
    
    async def assist_development(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """协助开发任务"""
        try:
            request_type = request.get("type", "unknown")
            response = {"status": "success", "data": {}}
            
            if request_type == "code_generation":
                # 代码生成
                generated_code = await self.code_generator.generate_function(
                    request["description"], 
                    CodeLanguage(request.get("language", "PYTHON")),
                    request.get("context", {})
                )
                response["data"]["generated_code"] = generated_code
                
                # 生成测试用例
                test_cases = await self._generate_tests_for_code(generated_code)
                response["data"]["test_cases"] = [tc.to_dict() for tc in test_cases]
                
            elif request_type == "code_review":
                # 代码审查
                code_file = CodeFile(
                    name=request.get("filename", "temp.py"),
                    content=request["code"],
                    language=CodeLanguage(request.get("language", "PYTHON"))
                )
                
                issues, metrics = await self.code_analyzer.analyze_code(code_file)
                response["data"]["issues"] = [issue.to_dict() for issue in issues]
                response["data"]["metrics"] = metrics.to_dict()
                
            elif request_type == "documentation":
                # 文档生成
                code_file = CodeFile(
                    name=request.get("filename", "temp.py"),
                    content=request["code"],
                    language=CodeLanguage(request.get("language", "PYTHON"))
                )
                
                docs = await self.documentation_generator.generate_documentation(
                    code_file, 
                    DocumentationType(request.get("doc_type", "CODE_COMMENTS"))
                )
                response["data"]["documentation"] = docs.to_dict()
                
            elif request_type == "refactoring":
                # 代码重构建议
                refactoring_suggestions = await self._suggest_refactoring(request["code"])
                response["data"]["refactoring_suggestions"] = refactoring_suggestions
            
            else:
                response["status"] = "error"
                response["error"] = f"不支持的操作类型: {request_type}"
            
            return response
            
        except Exception as e:
            logger.error(f"Development assistance failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _generate_tests_for_code(self, code: str) -> List[TestCase]:
        """为代码生成测试"""
        try:
            code_file = CodeFile(
                name="generated_code.py",
                content=code,
                language=CodeLanguage.PYTHON
            )
            
            test_cases = await self.test_generator.generate_tests(code_file, "pytest")
            return test_cases
            
        except Exception as e:
            logger.error(f"Test generation for code failed: {e}")
            return []
    
    async def _suggest_refactoring(self, code: str) -> List[str]:
        """建议重构"""
        try:
            suggestions = []
            
            # 检查长函数
            lines = code.split('\n')
            if len(lines) > 50:
                suggestions.append("函数过长，建议拆分为多个小函数")
            
            # 检查重复代码
            if "def " in code and code.count("def ") > 1:
                suggestions.append("发现重复代码，建议提取公共函数")
            
            # 检查复杂条件
            if code.count("if ") > 5:
                suggestions.append("条件判断过于复杂，建议使用策略模式")
            
            # 检查魔法数字
            if re.search(r'\b\d{3,}\b', code):
                suggestions.append("发现魔法数字，建议定义为常量")
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Refactoring suggestions failed: {e}")
            return ["重构建议生成失败"]
    
    def add_code_file(self, code_file: CodeFile) -> bool:
        """添加代码文件"""
        try:
            self.code_files[code_file.id] = code_file
            logger.info(f"Added code file: {code_file.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add code file: {e}")
            return False
    
    def get_code_file(self, file_id: str) -> Optional[CodeFile]:
        """获取代码文件"""
        return self.code_files.get(file_id)
    
    def get_analysis_results(self, file_id: str) -> Optional[Tuple[List[CodeIssue], CodeMetrics]]:
        """获取分析结果"""
        return self.analysis_results.get(file_id)

# 示例用法
async def main_demo():
    """代码助手演示"""
    config = {
        "analysis": {},
        "generation": {},
        "testing": {},
        "documentation": {}
    }
    
    # 创建代码助手
    assistant = CodeAssistant(config)
    
    print("💻 代码助手演示")
    print("=" * 50)
    
    # 1. 代码生成演示
    print("\n1. 代码生成演示...")
    generation_request = {
        "type": "code_generation",
        "description": "创建一个计算两个数之和的函数",
        "language": "PYTHON",
        "context": {"return_type": "int"}
    }
    
    generation_result = await assistant.assist_development(generation_request)
    if generation_result["status"] == "success":
        print("✓ 生成的代码:")
        print(generation_result["data"]["generated_code"])
        print(f"✓ 生成的测试用例数: {len(generation_result['data']['test_cases'])}")
    
    # 2. 代码审查演示
    print("\n2. 代码审查演示...")
    sample_code = """
def calculate_sum(a, b):
    # This is a very long line that exceeds the recommended line length limit and should trigger a warning
    result = a + b
    return result

def process_data(data):
    if data is None:
        return None
    if len(data) == 0:
        return []
    if len(data) > 100:
        return data[:100]
    return data
"""
    
    review_request = {
        "type": "code_review",
        "code": sample_code,
        "language": "PYTHON",
        "filename": "sample.py"
    }
    
    review_result = await assistant.assist_development(review_request)
    if review_result["status"] == "success":
        print("✓ 代码审查结果:")
        print(f"  发现的问题数: {len(review_result['data']['issues'])}")
        for issue in review_result["data"]["issues"]:
            print(f"    - {issue['severity']}: {issue['message']} (行 {issue['line_number']})")
        
        metrics = review_result["data"]["metrics"]
        print(f"✓ 代码指标:")
        print(f"  圈复杂度: {metrics['cyclomatic_complexity']}")
        print(f"  代码异味: {metrics['code_smells']}")
        print(f"  可维护性指数: {metrics['maintainability_index']:.2f}")
    
    # 3. 文档生成演示
    print("\n3. 文档生成演示...")
    doc_request = {
        "type": "documentation",
        "code": sample_code,
        "language": "PYTHON",
        "filename": "sample.py",
        "doc_type": "CODE_COMMENTS"
    }
    
    doc_result = await assistant.assist_development(doc_request)
    if doc_result["status"] == "success":
        print("✓ 生成的文档:")
        print(doc_result["data"]["documentation"]["content"][:200] + "...")
    
    # 4. 重构建议演示
    print("\n4. 重构建议演示...")
    refactor_request = {
        "type": "refactoring",
        "code": sample_code
    }
    
    refactor_result = await assistant.assist_development(refactor_request)
    if refactor_result["status"] == "success":
        print("✓ 重构建议:")
        for suggestion in refactor_result["data"]["refactoring_suggestions"]:
            print(f"  - {suggestion}")
    
    print("\n🎉 代码助手演示完成")

if __name__ == "__main__":
    asyncio.run(main_demo())
