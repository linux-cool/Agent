# threat_analysis.py
"""
第7章 安全隐私防护体系 - 安全威胁分析系统
实现智能体系统的安全威胁检测、分析和风险评估
"""

import logging
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AttackVector(Enum):
    """常见的智能体攻击向量"""
    PROMPT_INJECTION = "Prompt Injection"
    DATA_POISONING = "Data Poisoning"
    MODEL_EVASION = "Model Evasion"
    PRIVILEGE_ESCALATION = "Privilege Escalation"
    DENIAL_OF_SERVICE = "Denial of Service"
    DATA_EXFILTRATION = "Data Exfiltration"
    INSECURE_OUTPUT_HANDLING = "Insecure Output Handling"
    SUPPLY_CHAIN_ATTACKS = "Supply Chain Attacks"
    OTHER = "Other"

class RiskLevel(Enum):
    """风险等级"""
    LOW = "低"
    MEDIUM = "中"
    HIGH = "高"
    CRITICAL = "关键"

class AgentComponent(Enum):
    """智能体组件"""
    LLM = "大型语言模型"
    MEMORY = "记忆系统"
    PLANNING_ENGINE = "规划引擎"
    EXECUTION_ENGINE = "执行引擎"
    TOOL_MANAGER = "工具管理器"
    COMMUNICATION_MODULE = "通信模块"
    DATA_STORAGE = "数据存储"
    USER_INTERFACE = "用户界面"
    OTHER = "其他"

@dataclass
class Threat:
    """安全威胁数据结构"""
    id: str
    name: str
    description: str
    attack_vector: AttackVector
    target_component: AgentComponent
    potential_impact: str
    risk_level: RiskLevel
    mitigation_strategy: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "attack_vector": self.attack_vector.value,
            "target_component": self.target_component.value,
            "potential_impact": self.potential_impact,
            "risk_level": self.risk_level.value,
            "mitigation_strategy": self.mitigation_strategy,
            "metadata": self.metadata
        }

class ThreatAnalyzer:
    """
    智能体安全威胁分析器，用于识别、评估和建议缓解策略。
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config if config is not None else {}
        self.known_threats: Dict[str, Threat] = {}
        self._load_default_threats()
        logger.info("ThreatAnalyzer initialized.")

    def _load_default_threats(self):
        """加载默认的已知威胁"""
        default_threats = [
            Threat(
                id="TI001",
                name="Prompt Injection",
                description="攻击者通过恶意输入操纵LLM行为，使其偏离预期指令。",
                attack_vector=AttackVector.PROMPT_INJECTION,
                target_component=AgentComponent.LLM,
                potential_impact="信息泄露、非授权操作、系统滥用",
                risk_level=RiskLevel.HIGH,
                mitigation_strategy="输入过滤、LLM安全加固、权限隔离"
            ),
            Threat(
                id="DP001",
                name="Data Poisoning",
                description="攻击者向训练数据或RAG检索数据中注入恶意内容，影响模型决策。",
                attack_vector=AttackVector.DATA_POISONING,
                target_component=AgentComponent.MEMORY,
                potential_impact="模型性能下降、错误决策、偏见引入",
                risk_level=RiskLevel.HIGH,
                mitigation_strategy="数据源验证、数据清洗、异常检测"
            ),
            Threat(
                id="ME001",
                name="Model Evasion",
                description="攻击者通过对抗性输入绕过智能体的安全检测或分类器。",
                attack_vector=AttackVector.MODEL_EVASION,
                target_component=AgentComponent.LLM,
                potential_impact="安全机制失效、恶意内容通过",
                risk_level=RiskLevel.MEDIUM,
                mitigation_strategy="对抗性训练、多模态检测、人工审查"
            ),
            Threat(
                id="PE001",
                name="Privilege Escalation",
                description="智能体在执行任务时被诱导获取超出其应有权限的资源或操作。",
                attack_vector=AttackVector.PRIVILEGE_ESCALATION,
                target_component=AgentComponent.EXECUTION_ENGINE,
                potential_impact="系统控制权获取、敏感数据访问",
                risk_level=RiskLevel.CRITICAL,
                mitigation_strategy="最小权限原则、严格的权限管理、沙箱隔离"
            ),
            Threat(
                id="DE001",
                name="Data Exfiltration",
                description="智能体被诱导泄露其访问的敏感数据。",
                attack_vector=AttackVector.DATA_EXFILTRATION,
                target_component=AgentComponent.DATA_STORAGE,
                potential_impact="敏感信息泄露、合规性问题",
                risk_level=RiskLevel.HIGH,
                mitigation_strategy="数据加密、访问控制、输出过滤"
            )
        ]
        for threat in default_threats:
            self.known_threats[threat.id] = threat
        logger.info(f"Loaded {len(self.known_threats)} default threats.")

    async def add_threat(self, threat: Threat) -> bool:
        """添加一个新的安全威胁"""
        if threat.id in self.known_threats:
            logger.warning(f"Threat with ID {threat.id} already exists. Update instead?")
            return False
        self.known_threats[threat.id] = threat
        logger.info(f"Added threat: {threat.name} ({threat.id})")
        return True

    async def get_threat(self, threat_id: str) -> Optional[Threat]:
        """根据ID获取安全威胁"""
        return self.known_threats.get(threat_id)

    async def analyze_agent_system(self, agent_architecture: Dict[str, Any]) -> List[Threat]:
        """
        分析智能体系统架构，识别潜在的安全威胁。
        这是一个简化的示例，实际中会涉及更复杂的匹配逻辑和LLM推理。
        """
        logger.info(f"Analyzing agent system for architecture: {agent_architecture.get('name', 'Unnamed System')}")
        
        identified_threats: List[Threat] = []
        
        # 示例匹配逻辑：根据架构组件和功能识别威胁
        components = agent_architecture.get("components", [])
        
        if "LLM" in components:
            identified_threats.append(self.known_threats["TI001"]) # Prompt Injection
            identified_threats.append(self.known_threats["ME001"]) # Model Evasion
        if "Memory" in components or "Data Storage" in components:
            identified_threats.append(self.known_threats["DP001"]) # Data Poisoning
            identified_threats.append(self.known_threats["DE001"]) # Data Exfiltration
        if "Execution Engine" in components or "Tool Manager" in components:
            identified_threats.append(self.known_threats["PE001"]) # Privilege Escalation
        
        logger.info(f"Identified {len(identified_threats)} potential threats.")
        return identified_threats

    async def assess_risk(self, threat_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估特定威胁在给定上下文中的风险。
        """
        threat = await self.get_threat(threat_id)
        if not threat:
            return {"success": False, "message": f"Threat {threat_id} not found."}
        
        logger.info(f"Assessing risk for threat '{threat.name}' in context: {context}")
        
        # 模拟风险评估逻辑
        # 考虑威胁的固有风险等级、系统敏感度、现有控制措施等
        system_sensitivity = context.get("system_sensitivity", 0.5) # 0-1
        existing_controls_effectiveness = context.get("existing_controls_effectiveness", 0.5) # 0-1

        risk_score_map = {
            RiskLevel.LOW: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4
        }
        
        base_risk_score = risk_score_map.get(threat.risk_level, 1)
        
        # 简化计算：基础风险 * 系统敏感度 / 控制措施有效性
        # 实际中会更复杂，可能使用CVSS等标准
        final_risk_score = (base_risk_score * system_sensitivity) / (existing_controls_effectiveness + 0.1) # 避免除以0
        
        if final_risk_score > 3.0:
            assessment = "极高风险，需立即采取缓解措施。"
        elif final_risk_score > 2.0:
            assessment = "高风险，需优先处理。"
        elif final_risk_score > 1.0:
            assessment = "中等风险，建议采取措施。"
        else:
            assessment = "低风险，可监控。"
        
        return {
            "success": True,
            "threat_id": threat_id,
            "threat_name": threat.name,
            "risk_score": final_risk_score,
            "assessment": assessment,
            "mitigation_strategy": threat.mitigation_strategy
        }

async def main_demo():
    analyzer = ThreatAnalyzer()

    print("\n--- 分析一个示例智能体系统 ---")
    example_agent_architecture = {
        "name": "智能客服Agent",
        "components": ["LLM", "Memory", "Execution Engine", "Tool Manager", "User Interface"],
        "data_sensitivity": "HIGH"
    }
    identified_threats = await analyzer.analyze_agent_system(example_agent_architecture)
    for threat in identified_threats:
        print(f"识别威胁: {threat.name} ({threat.attack_vector.value}) - 影响: {threat.potential_impact}")
        
        context = {
            "system_sensitivity": 0.8 if example_agent_architecture["data_sensitivity"] == "HIGH" else 0.3,
            "existing_controls_effectiveness": 0.6 # 假设有一些现有控制
        }
        risk_assessment = await analyzer.assess_risk(threat.id, context)
        print(f"  风险评估: {risk_assessment['assessment']} (得分: {risk_assessment['risk_score']:.2f})")
        print(f"  缓解策略: {risk_assessment['mitigation_strategy']}")

    print("\n--- 添加并评估一个新威胁 ---")
    new_threat = Threat(
        id="DOS001",
        name="Agent DoS Attack",
        description="攻击者通过大量请求或复杂任务使智能体资源耗尽，导致服务不可用。",
        attack_vector=AttackVector.DENIAL_OF_SERVICE,
        target_component=AgentComponent.EXECUTION_ENGINE,
        potential_impact="服务中断、业务损失",
        risk_level=RiskLevel.HIGH,
        mitigation_strategy="限流、资源配额、弹性伸缩"
    )
    await analyzer.add_threat(new_threat)
    
    context_dos = {
        "system_sensitivity": 0.9, # 关键业务系统
        "existing_controls_effectiveness": 0.4 # 现有控制较弱
    }
    risk_assessment_dos = await analyzer.assess_risk("DOS001", context_dos)
    print(f"新威胁 '{new_threat.name}' 风险评估: {risk_assessment_dos['assessment']} (得分: {risk_assessment_dos['risk_score']:.2f})")
    print(f"  缓解策略: {risk_assessment_dos['mitigation_strategy']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main_demo())
