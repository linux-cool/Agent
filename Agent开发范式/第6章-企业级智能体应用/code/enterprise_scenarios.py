# enterprise_scenarios.py
"""
第6章 企业级智能体应用 - 企业应用场景分析
分析企业级智能体应用的核心场景、需求评估和实施策略
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ScenarioType(Enum):
    """场景类型枚举"""
    CUSTOMER_SERVICE = "智能客服"
    CODE_ASSISTANT = "代码助手"
    BUSINESS_AUTOMATION = "业务流程自动化"
    DATA_ANALYSIS = "数据分析"
    KNOWLEDGE_MANAGEMENT = "知识管理"
    HR_ASSISTANT = "HR助手"
    SALES_ASSISTANT = "销售助手"
    FINANCE_ASSISTANT = "财务助手"

class ComplexityLevel(Enum):
    """复杂度等级枚举"""
    LOW = "低"
    MEDIUM = "中"
    HIGH = "高"
    VERY_HIGH = "很高"

class ImplementationEffort(Enum):
    """实施难度枚举"""
    LOW = "低"
    MEDIUM = "中"
    HIGH = "高"
    VERY_HIGH = "很高"

class BusinessValue(Enum):
    """业务价值枚举"""
    LOW = "低"
    MEDIUM = "中"
    HIGH = "高"
    CRITICAL = "关键"

class RiskLevel(Enum):
    """风险等级枚举"""
    LOW = "低"
    MEDIUM = "中"
    HIGH = "高"
    CRITICAL = "严重"

@dataclass
class BusinessRequirement:
    """业务需求"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    priority: int = 1
    category: str = ""
    stakeholders: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "category": self.category,
            "stakeholders": self.stakeholders,
            "acceptance_criteria": self.acceptance_criteria,
            "dependencies": self.dependencies,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

@dataclass
class TechnicalRequirement:
    """技术需求"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    technology_stack: List[str] = field(default_factory=list)
    performance_requirements: Dict[str, Any] = field(default_factory=dict)
    security_requirements: Dict[str, Any] = field(default_factory=dict)
    scalability_requirements: Dict[str, Any] = field(default_factory=dict)
    integration_requirements: List[str] = field(default_factory=list)
    compliance_requirements: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "technology_stack": self.technology_stack,
            "performance_requirements": self.performance_requirements,
            "security_requirements": self.security_requirements,
            "scalability_requirements": self.scalability_requirements,
            "integration_requirements": self.integration_requirements,
            "compliance_requirements": self.compliance_requirements
        }

@dataclass
class BusinessImpact:
    """业务影响"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    scenario_id: str = ""
    impact_type: str = ""  # revenue, cost, efficiency, customer_satisfaction
    current_state: float = 0.0
    target_state: float = 0.0
    improvement_percentage: float = 0.0
    measurement_unit: str = ""
    time_to_realize: timedelta = field(default_factory=lambda: timedelta(days=90))
    confidence_level: float = 0.8
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "scenario_id": self.scenario_id,
            "impact_type": self.impact_type,
            "current_state": self.current_state,
            "target_state": self.target_state,
            "improvement_percentage": self.improvement_percentage,
            "measurement_unit": self.measurement_unit,
            "time_to_realize": self.time_to_realize.total_seconds(),
            "confidence_level": self.confidence_level
        }

@dataclass
class Recommendation:
    """实施建议"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""  # architecture, technology, process, resource
    title: str = ""
    description: str = ""
    priority: str = ""  # low, medium, high, critical
    effort_estimate: str = ""  # low, medium, high
    cost_estimate: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    dependencies: List[str] = field(default_factory=list)
    benefits: List[str] = field(default_factory=list)
    implementation_steps: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "effort_estimate": self.effort_estimate,
            "cost_estimate": self.cost_estimate,
            "risk_level": self.risk_level.value,
            "dependencies": self.dependencies,
            "benefits": self.benefits,
            "implementation_steps": self.implementation_steps
        }

@dataclass
class EnterpriseScenario:
    """企业应用场景"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    scenario_type: ScenarioType = ScenarioType.CUSTOMER_SERVICE
    business_value: BusinessValue = BusinessValue.MEDIUM
    technical_complexity: ComplexityLevel = ComplexityLevel.MEDIUM
    implementation_effort: ImplementationEffort = ImplementationEffort.MEDIUM
    risk_level: RiskLevel = RiskLevel.MEDIUM
    business_requirements: List[BusinessRequirement] = field(default_factory=list)
    technical_requirements: List[TechnicalRequirement] = field(default_factory=list)
    business_impacts: List[BusinessImpact] = field(default_factory=list)
    recommendations: List[Recommendation] = field(default_factory=list)
    stakeholders: List[str] = field(default_factory=list)
    success_metrics: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "scenario_type": self.scenario_type.value,
            "business_value": self.business_value.value,
            "technical_complexity": self.technical_complexity.value,
            "implementation_effort": self.implementation_effort.value,
            "risk_level": self.risk_level.value,
            "business_requirements": [req.to_dict() for req in self.business_requirements],
            "technical_requirements": [req.to_dict() for req in self.technical_requirements],
            "business_impacts": [impact.to_dict() for impact in self.business_impacts],
            "recommendations": [rec.to_dict() for rec in self.recommendations],
            "stakeholders": self.stakeholders,
            "success_metrics": self.success_metrics,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

class RequirementsAnalyzer:
    """需求分析器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.requirement_templates = self._load_requirement_templates()
    
    def _load_requirement_templates(self) -> Dict[str, Dict[str, Any]]:
        """加载需求模板"""
        return {
            "customer_service": {
                "business_requirements": [
                    "24/7客户服务支持",
                    "多语言支持",
                    "快速响应时间",
                    "客户满意度提升",
                    "成本降低"
                ],
                "technical_requirements": [
                    "自然语言处理能力",
                    "知识库集成",
                    "多渠道支持",
                    "实时监控",
                    "数据安全"
                ]
            },
            "code_assistant": {
                "business_requirements": [
                    "开发效率提升",
                    "代码质量改善",
                    "学习成本降低",
                    "错误率减少",
                    "团队协作增强"
                ],
                "technical_requirements": [
                    "代码理解能力",
                    "多语言支持",
                    "IDE集成",
                    "版本控制集成",
                    "安全扫描"
                ]
            },
            "business_automation": {
                "business_requirements": [
                    "流程标准化",
                    "效率提升",
                    "错误减少",
                    "合规性保证",
                    "成本控制"
                ],
                "technical_requirements": [
                    "工作流引擎",
                    "API集成",
                    "数据同步",
                    "异常处理",
                    "审计日志"
                ]
            }
        }
    
    async def analyze(self, requirements_data: Dict[str, Any]) -> Tuple[List[BusinessRequirement], List[TechnicalRequirement]]:
        """分析需求"""
        try:
            scenario_type = requirements_data.get("scenario_type", "customer_service")
            template = self.requirement_templates.get(scenario_type, {})
            
            # 分析业务需求
            business_requirements = []
            for i, req_text in enumerate(template.get("business_requirements", [])):
                req = BusinessRequirement(
                    title=f"业务需求 {i+1}",
                    description=req_text,
                    priority=i+1,
                    category="business"
                )
                business_requirements.append(req)
            
            # 分析技术需求
            technical_requirements = []
            for i, req_text in enumerate(template.get("technical_requirements", [])):
                req = TechnicalRequirement(
                    title=f"技术需求 {i+1}",
                    description=req_text,
                    technology_stack=self._extract_technology_stack(req_text),
                    performance_requirements=self._extract_performance_requirements(req_text),
                    security_requirements=self._extract_security_requirements(req_text)
                )
                technical_requirements.append(req)
            
            return business_requirements, technical_requirements
            
        except Exception as e:
            logger.error(f"Requirements analysis failed: {e}")
            return [], []
    
    def _extract_technology_stack(self, requirement_text: str) -> List[str]:
        """提取技术栈"""
        tech_keywords = {
            "自然语言处理": ["NLP", "BERT", "GPT", "Transformer"],
            "机器学习": ["ML", "TensorFlow", "PyTorch", "Scikit-learn"],
            "数据库": ["PostgreSQL", "MongoDB", "Redis", "Elasticsearch"],
            "API": ["REST", "GraphQL", "FastAPI", "Django"],
            "容器化": ["Docker", "Kubernetes", "OpenShift"],
            "监控": ["Prometheus", "Grafana", "ELK", "Jaeger"]
        }
        
        tech_stack = []
        for category, technologies in tech_keywords.items():
            if any(keyword in requirement_text for keyword in [category] + technologies):
                tech_stack.extend(technologies[:2])  # 取前两个技术
        
        return list(set(tech_stack))
    
    def _extract_performance_requirements(self, requirement_text: str) -> Dict[str, Any]:
        """提取性能需求"""
        performance_reqs = {}
        
        if "响应时间" in requirement_text or "快速" in requirement_text:
            performance_reqs["response_time"] = "< 2s"
        
        if "并发" in requirement_text or "高并发" in requirement_text:
            performance_reqs["concurrency"] = "> 1000"
        
        if "可用性" in requirement_text or "稳定性" in requirement_text:
            performance_reqs["availability"] = "> 99.9%"
        
        return performance_reqs
    
    def _extract_security_requirements(self, requirement_text: str) -> Dict[str, Any]:
        """提取安全需求"""
        security_reqs = {}
        
        if "安全" in requirement_text or "加密" in requirement_text:
            security_reqs["encryption"] = "AES-256"
            security_reqs["authentication"] = "Multi-factor"
        
        if "权限" in requirement_text or "访问控制" in requirement_text:
            security_reqs["authorization"] = "RBAC"
        
        if "审计" in requirement_text or "日志" in requirement_text:
            security_reqs["audit"] = "Complete audit trail"
        
        return security_reqs

class ImpactAssessor:
    """影响评估器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.impact_models = self._load_impact_models()
    
    def _load_impact_models(self) -> Dict[str, Dict[str, Any]]:
        """加载影响模型"""
        return {
            "customer_service": {
                "revenue": {"current": 1000000, "target": 1200000, "unit": "USD/year"},
                "cost": {"current": 500000, "target": 300000, "unit": "USD/year"},
                "efficiency": {"current": 70, "target": 90, "unit": "%"},
                "customer_satisfaction": {"current": 3.5, "target": 4.5, "unit": "rating"}
            },
            "code_assistant": {
                "efficiency": {"current": 60, "target": 85, "unit": "%"},
                "quality": {"current": 75, "target": 95, "unit": "%"},
                "cost": {"current": 200000, "target": 150000, "unit": "USD/year"},
                "time_to_market": {"current": 90, "target": 60, "unit": "days"}
            },
            "business_automation": {
                "efficiency": {"current": 50, "target": 80, "unit": "%"},
                "accuracy": {"current": 85, "target": 98, "unit": "%"},
                "cost": {"current": 300000, "target": 200000, "unit": "USD/year"},
                "compliance": {"current": 90, "target": 99, "unit": "%"}
            }
        }
    
    async def assess(self, impact_data: Dict[str, Any]) -> List[BusinessImpact]:
        """评估业务影响"""
        try:
            scenario_type = impact_data.get("scenario_type", "customer_service")
            impact_model = self.impact_models.get(scenario_type, {})
            
            impacts = []
            for impact_type, metrics in impact_model.items():
                impact = BusinessImpact(
                    scenario_id=impact_data.get("scenario_id", ""),
                    impact_type=impact_type,
                    current_state=metrics["current"],
                    target_state=metrics["target"],
                    improvement_percentage=((metrics["target"] - metrics["current"]) / metrics["current"]) * 100,
                    measurement_unit=metrics["unit"],
                    time_to_realize=timedelta(days=90),
                    confidence_level=0.8
                )
                impacts.append(impact)
            
            return impacts
            
        except Exception as e:
            logger.error(f"Impact assessment failed: {e}")
            return []

class RecommendationGenerator:
    """建议生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.recommendation_templates = self._load_recommendation_templates()
    
    def _load_recommendation_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        """加载建议模板"""
        return {
            "architecture": [
                {
                    "title": "采用微服务架构",
                    "description": "建议采用微服务架构以提高系统的可扩展性和可维护性",
                    "priority": "high",
                    "effort": "high",
                    "benefits": ["可扩展性", "可维护性", "技术多样性"]
                },
                {
                    "title": "实施容器化部署",
                    "description": "使用Docker和Kubernetes实现容器化部署",
                    "priority": "medium",
                    "effort": "medium",
                    "benefits": ["环境一致性", "快速部署", "资源优化"]
                }
            ],
            "technology": [
                {
                    "title": "集成AI/ML能力",
                    "description": "集成先进的AI/ML技术以提升智能化水平",
                    "priority": "high",
                    "effort": "high",
                    "benefits": ["智能化", "自动化", "效率提升"]
                },
                {
                    "title": "实施实时监控",
                    "description": "建立全面的实时监控和告警系统",
                    "priority": "medium",
                    "effort": "medium",
                    "benefits": ["可观测性", "快速响应", "问题预防"]
                }
            ],
            "process": [
                {
                    "title": "建立DevOps流程",
                    "description": "实施CI/CD流程以提高开发和部署效率",
                    "priority": "high",
                    "effort": "medium",
                    "benefits": ["自动化", "质量保证", "快速交付"]
                },
                {
                    "title": "实施敏捷开发",
                    "description": "采用敏捷开发方法以提高开发效率",
                    "priority": "medium",
                    "effort": "low",
                    "benefits": ["快速迭代", "客户反馈", "风险控制"]
                }
            ]
        }
    
    async def generate_recommendations(self, scenario: EnterpriseScenario) -> List[Recommendation]:
        """生成实施建议"""
        try:
            recommendations = []
            
            # 基于技术复杂度生成架构建议
            if scenario.technical_complexity in [ComplexityLevel.HIGH, ComplexityLevel.VERY_HIGH]:
                arch_recs = self.recommendation_templates["architecture"]
                for rec_data in arch_recs:
                    recommendation = Recommendation(
                        type="architecture",
                        title=rec_data["title"],
                        description=rec_data["description"],
                        priority=rec_data["priority"],
                        effort_estimate=rec_data["effort"],
                        cost_estimate=self._estimate_cost(rec_data["effort"]),
                        risk_level=self._assess_risk(rec_data["effort"]),
                        benefits=rec_data["benefits"],
                        implementation_steps=self._generate_implementation_steps(rec_data["title"])
                    )
                    recommendations.append(recommendation)
            
            # 基于业务价值生成技术建议
            if scenario.business_value in [BusinessValue.HIGH, BusinessValue.CRITICAL]:
                tech_recs = self.recommendation_templates["technology"]
                for rec_data in tech_recs:
                    recommendation = Recommendation(
                        type="technology",
                        title=rec_data["title"],
                        description=rec_data["description"],
                        priority=rec_data["priority"],
                        effort_estimate=rec_data["effort"],
                        cost_estimate=self._estimate_cost(rec_data["effort"]),
                        risk_level=self._assess_risk(rec_data["effort"]),
                        benefits=rec_data["benefits"],
                        implementation_steps=self._generate_implementation_steps(rec_data["title"])
                    )
                    recommendations.append(recommendation)
            
            # 基于实施难度生成流程建议
            if scenario.implementation_effort in [ImplementationEffort.HIGH, ImplementationEffort.VERY_HIGH]:
                process_recs = self.recommendation_templates["process"]
                for rec_data in process_recs:
                    recommendation = Recommendation(
                        type="process",
                        title=rec_data["title"],
                        description=rec_data["description"],
                        priority=rec_data["priority"],
                        effort_estimate=rec_data["effort"],
                        cost_estimate=self._estimate_cost(rec_data["effort"]),
                        risk_level=self._assess_risk(rec_data["effort"]),
                        benefits=rec_data["benefits"],
                        implementation_steps=self._generate_implementation_steps(rec_data["title"])
                    )
                    recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return []
    
    def _estimate_cost(self, effort: str) -> float:
        """估算成本"""
        cost_mapping = {
            "low": 10000,
            "medium": 50000,
            "high": 200000
        }
        return cost_mapping.get(effort, 50000)
    
    def _assess_risk(self, effort: str) -> RiskLevel:
        """评估风险"""
        risk_mapping = {
            "low": RiskLevel.LOW,
            "medium": RiskLevel.MEDIUM,
            "high": RiskLevel.HIGH
        }
        return risk_mapping.get(effort, RiskLevel.MEDIUM)
    
    def _generate_implementation_steps(self, title: str) -> List[str]:
        """生成实施步骤"""
        steps_mapping = {
            "采用微服务架构": [
                "1. 分析现有系统架构",
                "2. 识别可拆分的服务边界",
                "3. 设计服务接口",
                "4. 实施服务拆分",
                "5. 建立服务治理机制"
            ],
            "集成AI/ML能力": [
                "1. 评估AI/ML需求",
                "2. 选择合适的技术栈",
                "3. 准备训练数据",
                "4. 开发模型",
                "5. 集成到现有系统"
            ],
            "建立DevOps流程": [
                "1. 建立CI/CD流水线",
                "2. 实施自动化测试",
                "3. 配置部署环境",
                "4. 建立监控告警",
                "5. 培训团队"
            ]
        }
        return steps_mapping.get(title, ["1. 制定实施计划", "2. 执行实施", "3. 验证结果"])

class EnterpriseScenarioAnalyzer:
    """企业应用场景分析器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scenarios: Dict[str, EnterpriseScenario] = {}
        self.requirements_analyzer = RequirementsAnalyzer(config.get("requirements", {}))
        self.impact_assessor = ImpactAssessor(config.get("impact", {}))
        self.recommendation_generator = RecommendationGenerator(config.get("recommendations", {}))
    
    async def analyze_scenario(self, scenario_data: Dict[str, Any]) -> EnterpriseScenario:
        """分析企业应用场景"""
        try:
            # 创建场景对象
            scenario = EnterpriseScenario(
                name=scenario_data["name"],
                description=scenario_data["description"],
                scenario_type=getattr(ScenarioType, scenario_data.get("scenario_type", "CUSTOMER_SERVICE")),
                business_value=getattr(BusinessValue, scenario_data.get("business_value", "MEDIUM")),
                technical_complexity=getattr(ComplexityLevel, scenario_data.get("technical_complexity", "MEDIUM")),
                implementation_effort=getattr(ImplementationEffort, scenario_data.get("implementation_effort", "MEDIUM")),
                risk_level=getattr(RiskLevel, scenario_data.get("risk_level", "MEDIUM")),
                stakeholders=scenario_data.get("stakeholders", []),
                success_metrics=scenario_data.get("success_metrics", [])
            )
            
            # 分析业务需求和技术需求
            business_reqs, technical_reqs = await self.requirements_analyzer.analyze(scenario_data)
            scenario.business_requirements = business_reqs
            scenario.technical_requirements = technical_reqs
            
            # 评估业务影响
            impacts = await self.impact_assessor.assess(scenario_data)
            scenario.business_impacts = impacts
            
            # 生成实施建议
            recommendations = await self.recommendation_generator.generate_recommendations(scenario)
            scenario.recommendations = recommendations
            
            # 保存场景
            self.scenarios[scenario.id] = scenario
            
            logger.info(f"Scenario analyzed: {scenario.name}")
            return scenario
            
        except Exception as e:
            logger.error(f"Scenario analysis failed: {e}")
            raise
    
    async def compare_scenarios(self, scenario_ids: List[str]) -> Dict[str, Any]:
        """比较多个场景"""
        try:
            scenarios = [self.scenarios[sid] for sid in scenario_ids if sid in self.scenarios]
            
            comparison = {
                "scenarios": [s.to_dict() for s in scenarios],
                "comparison_matrix": self._create_comparison_matrix(scenarios),
                "recommendations": self._generate_comparison_recommendations(scenarios)
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Scenario comparison failed: {e}")
            return {}
    
    def _create_comparison_matrix(self, scenarios: List[EnterpriseScenario]) -> Dict[str, Any]:
        """创建比较矩阵"""
        matrix = {
            "business_value": {},
            "technical_complexity": {},
            "implementation_effort": {},
            "risk_level": {},
            "estimated_cost": {},
            "time_to_implement": {}
        }
        
        for scenario in scenarios:
            scenario_id = scenario.id
            matrix["business_value"][scenario_id] = scenario.business_value.value
            matrix["technical_complexity"][scenario_id] = scenario.technical_complexity.value
            matrix["implementation_effort"][scenario_id] = scenario.implementation_effort.value
            matrix["risk_level"][scenario_id] = scenario.risk_level.value
            
            # 估算成本和实施时间
            total_cost = sum(rec.cost_estimate for rec in scenario.recommendations)
            matrix["estimated_cost"][scenario_id] = total_cost
            
            # 基于复杂度估算实施时间
            time_mapping = {
                ComplexityLevel.LOW: 30,
                ComplexityLevel.MEDIUM: 90,
                ComplexityLevel.HIGH: 180,
                ComplexityLevel.VERY_HIGH: 365
            }
            matrix["time_to_implement"][scenario_id] = time_mapping[scenario.technical_complexity]
        
        return matrix
    
    def _generate_comparison_recommendations(self, scenarios: List[EnterpriseScenario]) -> List[str]:
        """生成比较建议"""
        recommendations = []
        
        # 按业务价值排序
        sorted_by_value = sorted(scenarios, key=lambda s: s.business_value.value, reverse=True)
        recommendations.append(f"建议优先实施: {sorted_by_value[0].name} (业务价值最高)")
        
        # 按风险等级排序
        low_risk_scenarios = [s for s in scenarios if s.risk_level == RiskLevel.LOW]
        if low_risk_scenarios:
            recommendations.append(f"低风险快速实施: {low_risk_scenarios[0].name}")
        
        # 按实施难度排序
        easy_scenarios = [s for s in scenarios if s.implementation_effort == ImplementationEffort.LOW]
        if easy_scenarios:
            recommendations.append(f"快速见效: {easy_scenarios[0].name}")
        
        return recommendations
    
    def get_scenario(self, scenario_id: str) -> Optional[EnterpriseScenario]:
        """获取场景"""
        return self.scenarios.get(scenario_id)
    
    def list_scenarios(self) -> List[EnterpriseScenario]:
        """列出所有场景"""
        return list(self.scenarios.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_scenarios = len(self.scenarios)
        scenario_types = {}
        complexity_distribution = {}
        business_value_distribution = {}
        
        for scenario in self.scenarios.values():
            # 场景类型分布
            scenario_type = scenario.scenario_type.value
            scenario_types[scenario_type] = scenario_types.get(scenario_type, 0) + 1
            
            # 复杂度分布
            complexity = scenario.technical_complexity.value
            complexity_distribution[complexity] = complexity_distribution.get(complexity, 0) + 1
            
            # 业务价值分布
            business_value = scenario.business_value.value
            business_value_distribution[business_value] = business_value_distribution.get(business_value, 0) + 1
        
        return {
            "total_scenarios": total_scenarios,
            "scenario_types": scenario_types,
            "complexity_distribution": complexity_distribution,
            "business_value_distribution": business_value_distribution
        }

# 示例用法
async def main_demo():
    """企业应用场景分析演示"""
    config = {
        "requirements": {},
        "impact": {},
        "recommendations": {}
    }
    
    analyzer = EnterpriseScenarioAnalyzer(config)
    
    print("🏢 企业应用场景分析演示")
    print("=" * 50)
    
    # 分析智能客服场景
    print("\n1. 分析智能客服场景...")
    customer_service_data = {
        "name": "智能客服系统",
        "description": "基于AI的24/7客户服务支持系统",
        "scenario_type": "CUSTOMER_SERVICE",
        "business_value": "HIGH",
        "technical_complexity": "MEDIUM",
        "implementation_effort": "MEDIUM",
        "risk_level": "MEDIUM",
        "stakeholders": ["客服部门", "IT部门", "管理层"],
        "success_metrics": ["客户满意度", "响应时间", "成本降低"]
    }
    
    customer_service_scenario = await analyzer.analyze_scenario(customer_service_data)
    print(f"✓ 场景分析完成: {customer_service_scenario.name}")
    print(f"  业务价值: {customer_service_scenario.business_value.value}")
    print(f"  技术复杂度: {customer_service_scenario.technical_complexity.value}")
    print(f"  实施难度: {customer_service_scenario.implementation_effort.value}")
    print(f"  业务需求数: {len(customer_service_scenario.business_requirements)}")
    print(f"  技术需求数: {len(customer_service_scenario.technical_requirements)}")
    print(f"  业务影响数: {len(customer_service_scenario.business_impacts)}")
    print(f"  实施建议数: {len(customer_service_scenario.recommendations)}")
    
    # 分析代码助手场景
    print("\n2. 分析代码助手场景...")
    code_assistant_data = {
        "name": "AI代码助手",
        "description": "基于AI的代码生成和审查助手",
        "scenario_type": "CODE_ASSISTANT",
        "business_value": "HIGH",
        "technical_complexity": "HIGH",
        "implementation_effort": "HIGH",
        "risk_level": "MEDIUM",
        "stakeholders": ["开发团队", "技术负责人", "产品经理"],
        "success_metrics": ["开发效率", "代码质量", "错误率"]
    }
    
    code_assistant_scenario = await analyzer.analyze_scenario(code_assistant_data)
    print(f"✓ 场景分析完成: {code_assistant_scenario.name}")
    print(f"  业务价值: {code_assistant_scenario.business_value.value}")
    print(f"  技术复杂度: {code_assistant_scenario.technical_complexity.value}")
    print(f"  实施难度: {code_assistant_scenario.implementation_effort.value}")
    print(f"  实施建议数: {len(code_assistant_scenario.recommendations)}")
    
    # 分析业务流程自动化场景
    print("\n3. 分析业务流程自动化场景...")
    automation_data = {
        "name": "业务流程自动化",
        "description": "自动化企业业务流程，提高效率",
        "scenario_type": "BUSINESS_AUTOMATION",
        "business_value": "CRITICAL",
        "technical_complexity": "MEDIUM",
        "implementation_effort": "MEDIUM",
        "risk_level": "LOW",
        "stakeholders": ["业务部门", "IT部门", "管理层"],
        "success_metrics": ["流程效率", "错误率", "成本控制"]
    }
    
    automation_scenario = await analyzer.analyze_scenario(automation_data)
    print(f"✓ 场景分析完成: {automation_scenario.name}")
    print(f"  业务价值: {automation_scenario.business_value.value}")
    print(f"  技术复杂度: {automation_scenario.technical_complexity.value}")
    print(f"  实施难度: {automation_scenario.implementation_effort.value}")
    print(f"  实施建议数: {len(automation_scenario.recommendations)}")
    
    # 场景比较
    print("\n4. 场景比较分析...")
    scenario_ids = [customer_service_scenario.id, code_assistant_scenario.id, automation_scenario.id]
    comparison = await analyzer.compare_scenarios(scenario_ids)
    
    print("场景比较矩阵:")
    for metric, values in comparison["comparison_matrix"].items():
        print(f"  {metric}:")
        for scenario_id, value in values.items():
            scenario_name = next(s.name for s in analyzer.scenarios.values() if s.id == scenario_id)
            print(f"    {scenario_name}: {value}")
    
    print("\n比较建议:")
    for rec in comparison["recommendations"]:
        print(f"  - {rec}")
    
    # 统计信息
    print("\n5. 分析统计信息:")
    stats = analyzer.get_stats()
    print(f"  总场景数: {stats['total_scenarios']}")
    print(f"  场景类型分布: {stats['scenario_types']}")
    print(f"  复杂度分布: {stats['complexity_distribution']}")
    print(f"  业务价值分布: {stats['business_value_distribution']}")
    
    print("\n🎉 企业应用场景分析演示完成")

if __name__ == "__main__":
    asyncio.run(main_demo())
