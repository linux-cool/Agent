# reasoning_engine.py
"""
第4章 记忆与推理系统构建 - 推理引擎
实现智能体的推理引擎，包括规则推理、概率推理、逻辑推理等
"""

import asyncio
import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import re
from collections import defaultdict, deque
import sympy as sp
from sympy.logic import *
import networkx as nx

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReasoningType(Enum):
    """推理类型枚举"""
    DEDUCTIVE = "演绎推理"
    INDUCTIVE = "归纳推理"
    ABDUCTIVE = "溯因推理"
    ANALOGICAL = "类比推理"
    CAUSAL = "因果推理"
    TEMPORAL = "时序推理"
    SPATIAL = "空间推理"
    PROBABILISTIC = "概率推理"

class RuleType(Enum):
    """规则类型枚举"""
    IF_THEN = "如果-那么"
    WHEN_THEN = "当-那么"
    UNLESS_THEN = "除非-那么"
    CAUSAL = "因果"
    TEMPORAL = "时序"
    SPATIAL = "空间"

class InferenceMethod(Enum):
    """推理方法枚举"""
    FORWARD_CHAINING = "前向链接"
    BACKWARD_CHAINING = "后向链接"
    RESOLUTION = "归结"
    MODUS_PONENS = "假言推理"
    MODUS_TOLLENS = "拒取式"
    SYLLOGISM = "三段论"
    BAYESIAN = "贝叶斯"
    FUZZY = "模糊"

@dataclass
class Rule:
    """规则数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    rule_type: RuleType = RuleType.IF_THEN
    antecedent: List[str] = field(default_factory=list)  # 前提条件
    consequent: List[str] = field(default_factory=list)  # 结论
    confidence: float = 1.0
    priority: int = 0
    conditions: Dict[str, Any] = field(default_factory=dict)
    actions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "rule_type": self.rule_type.value,
            "antecedent": self.antecedent,
            "consequent": self.consequent,
            "confidence": self.confidence,
            "priority": self.priority,
            "conditions": self.conditions,
            "actions": self.actions,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Rule':
        """从字典创建规则对象"""
        rule = cls()
        rule.id = data.get("id", str(uuid.uuid4()))
        rule.name = data.get("name", "")
        rule.rule_type = RuleType(data.get("rule_type", "如果-那么"))
        rule.antecedent = data.get("antecedent", [])
        rule.consequent = data.get("consequent", [])
        rule.confidence = data.get("confidence", 1.0)
        rule.priority = data.get("priority", 0)
        rule.conditions = data.get("conditions", {})
        rule.actions = data.get("actions", [])
        rule.created_at = datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        rule.updated_at = datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat()))
        rule.metadata = data.get("metadata", {})
        return rule

@dataclass
class Fact:
    """事实数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    statement: str = ""
    confidence: float = 1.0
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "statement": self.statement,
            "confidence": self.confidence,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Fact':
        """从字典创建事实对象"""
        fact = cls()
        fact.id = data.get("id", str(uuid.uuid4()))
        fact.statement = data.get("statement", "")
        fact.confidence = data.get("confidence", 1.0)
        fact.source = data.get("source", "")
        fact.timestamp = datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat()))
        fact.context = data.get("context", {})
        fact.metadata = data.get("metadata", {})
        return fact

@dataclass
class InferenceResult:
    """推理结果数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    reasoning_type: ReasoningType = ReasoningType.DEDUCTIVE
    method: InferenceMethod = InferenceMethod.FORWARD_CHAINING
    premises: List[str] = field(default_factory=list)
    conclusion: str = ""
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    reasoning_steps: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "reasoning_type": self.reasoning_type.value,
            "method": self.method.value,
            "premises": self.premises,
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "reasoning_steps": self.reasoning_steps,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

class RuleEngine:
    """规则引擎"""
    
    def __init__(self):
        self.rules: Dict[str, Rule] = {}
        self.facts: Dict[str, Fact] = {}
        self.working_memory: Set[str] = set()
        self.rule_graph = nx.DiGraph()
        self.running = False
    
    async def start(self):
        """启动规则引擎"""
        self.running = True
        logger.info("Rule engine started")
    
    async def stop(self):
        """停止规则引擎"""
        self.running = False
        logger.info("Rule engine stopped")
    
    async def add_rule(self, rule: Rule) -> bool:
        """添加规则"""
        try:
            self.rules[rule.id] = rule
            
            # 构建规则图
            for antecedent in rule.antecedent:
                self.rule_graph.add_edge(antecedent, rule.id)
            for consequent in rule.consequent:
                self.rule_graph.add_edge(rule.id, consequent)
            
            logger.info(f"Added rule: {rule.name} ({rule.id})")
            return True
        except Exception as e:
            logger.error(f"Failed to add rule: {e}")
            return False
    
    async def add_fact(self, fact: Fact) -> bool:
        """添加事实"""
        try:
            self.facts[fact.id] = fact
            self.working_memory.add(fact.statement)
            logger.info(f"Added fact: {fact.statement} ({fact.id})")
            return True
        except Exception as e:
            logger.error(f"Failed to add fact: {e}")
            return False
    
    async def remove_rule(self, rule_id: str) -> bool:
        """移除规则"""
        try:
            if rule_id not in self.rules:
                logger.error(f"Rule not found: {rule_id}")
                return False
            
            # 从规则图中移除
            if self.rule_graph.has_node(rule_id):
                self.rule_graph.remove_node(rule_id)
            
            del self.rules[rule_id]
            logger.info(f"Removed rule: {rule_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove rule: {e}")
            return False
    
    async def remove_fact(self, fact_id: str) -> bool:
        """移除事实"""
        try:
            if fact_id not in self.facts:
                logger.error(f"Fact not found: {fact_id}")
                return False
            
            fact = self.facts[fact_id]
            self.working_memory.discard(fact.statement)
            del self.facts[fact_id]
            logger.info(f"Removed fact: {fact_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove fact: {e}")
            return False
    
    async def forward_chaining(self, goal: str = None) -> List[InferenceResult]:
        """前向链接推理"""
        try:
            results = []
            new_facts = True
            
            while new_facts:
                new_facts = False
                applicable_rules = []
                
                # 找到所有可应用的规则
                for rule in self.rules.values():
                    if await self._is_rule_applicable(rule):
                        applicable_rules.append(rule)
                
                # 按优先级排序
                applicable_rules.sort(key=lambda r: r.priority, reverse=True)
                
                # 应用规则
                for rule in applicable_rules:
                    new_conclusions = await self._apply_rule(rule)
                    if new_conclusions:
                        new_facts = True
                        
                        # 创建推理结果
                        result = InferenceResult(
                            reasoning_type=ReasoningType.DEDUCTIVE,
                            method=InferenceMethod.FORWARD_CHAINING,
                            premises=rule.antecedent.copy(),
                            conclusion="; ".join(new_conclusions),
                            confidence=rule.confidence,
                            evidence=[f"Rule: {rule.name}"],
                            reasoning_steps=[{
                                "step": 1,
                                "rule": rule.name,
                                "premises": rule.antecedent,
                                "conclusions": new_conclusions
                            }]
                        )
                        results.append(result)
                        
                        # 如果达到目标，停止推理
                        if goal and goal in new_conclusions:
                            break
                
                # 如果达到目标，停止推理
                if goal and goal in self.working_memory:
                    break
            
            logger.info(f"Forward chaining completed, generated {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Forward chaining failed: {e}")
            return []
    
    async def backward_chaining(self, goal: str) -> List[InferenceResult]:
        """后向链接推理"""
        try:
            results = []
            proof_tree = await self._build_proof_tree(goal)
            
            if proof_tree:
                # 从证明树生成推理结果
                result = InferenceResult(
                    reasoning_type=ReasoningType.DEDUCTIVE,
                    method=InferenceMethod.BACKWARD_CHAINING,
                    premises=list(proof_tree.get("premises", [])),
                    conclusion=goal,
                    confidence=proof_tree.get("confidence", 0.0),
                    evidence=proof_tree.get("evidence", []),
                    reasoning_steps=proof_tree.get("steps", [])
                )
                results.append(result)
            
            logger.info(f"Backward chaining completed, generated {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Backward chaining failed: {e}")
            return []
    
    async def _is_rule_applicable(self, rule: Rule) -> bool:
        """检查规则是否可应用"""
        # 检查前提条件是否都在工作记忆中
        for antecedent in rule.antecedent:
            if antecedent not in self.working_memory:
                return False
        
        # 检查结论是否已经存在
        for consequent in rule.consequent:
            if consequent in self.working_memory:
                return False
        
        return True
    
    async def _apply_rule(self, rule: Rule) -> List[str]:
        """应用规则"""
        new_conclusions = []
        
        for consequent in rule.consequent:
            if consequent not in self.working_memory:
                self.working_memory.add(consequent)
                new_conclusions.append(consequent)
                
                # 创建新事实
                fact = Fact(
                    statement=consequent,
                    confidence=rule.confidence,
                    source=f"Rule: {rule.name}",
                    context={"rule_id": rule.id}
                )
                await self.add_fact(fact)
        
        return new_conclusions
    
    async def _build_proof_tree(self, goal: str) -> Dict[str, Any]:
        """构建证明树"""
        if goal in self.working_memory:
            return {
                "premises": [goal],
                "confidence": 1.0,
                "evidence": [f"Fact: {goal}"],
                "steps": []
            }
        
        # 找到可以推导出目标的规则
        applicable_rules = []
        for rule in self.rules.values():
            if goal in rule.consequent:
                applicable_rules.append(rule)
        
        if not applicable_rules:
            return None
        
        # 选择最佳规则（优先级最高）
        best_rule = max(applicable_rules, key=lambda r: r.priority)
        
        # 递归构建证明树
        premises = []
        evidence = [f"Rule: {best_rule.name}"]
        steps = []
        min_confidence = best_rule.confidence
        
        for antecedent in best_rule.antecedent:
            sub_proof = await self._build_proof_tree(antecedent)
            if sub_proof:
                premises.extend(sub_proof["premises"])
                evidence.extend(sub_proof["evidence"])
                steps.extend(sub_proof["steps"])
                min_confidence = min(min_confidence, sub_proof["confidence"])
            else:
                return None
        
        steps.append({
            "step": len(steps) + 1,
            "rule": best_rule.name,
            "premises": best_rule.antecedent,
            "conclusion": goal
        })
        
        return {
            "premises": premises,
            "confidence": min_confidence,
            "evidence": evidence,
            "steps": steps
        }

class ProbabilisticReasoner:
    """概率推理器"""
    
    def __init__(self):
        self.bayesian_network = nx.DiGraph()
        self.conditional_probabilities: Dict[Tuple[str, str], float] = {}
        self.prior_probabilities: Dict[str, float] = {}
        self.evidence: Dict[str, bool] = {}
    
    async def add_variable(self, variable: str, prior_prob: float = 0.5):
        """添加变量"""
        self.bayesian_network.add_node(variable)
        self.prior_probabilities[variable] = prior_prob
    
    async def add_dependency(self, parent: str, child: str, prob: float):
        """添加依赖关系"""
        self.bayesian_network.add_edge(parent, child)
        self.conditional_probabilities[(parent, child)] = prob
    
    async def set_evidence(self, variable: str, value: bool):
        """设置证据"""
        self.evidence[variable] = value
    
    async def infer_probability(self, variable: str) -> float:
        """推理概率"""
        try:
            # 简化的贝叶斯推理
            if variable in self.evidence:
                return 1.0 if self.evidence[variable] else 0.0
            
            # 计算后验概率
            prior = self.prior_probabilities.get(variable, 0.5)
            
            # 考虑父节点的影响
            parents = list(self.bayesian_network.predecessors(variable))
            if parents:
                parent_influence = 0.0
                for parent in parents:
                    if parent in self.evidence:
                        parent_prob = 1.0 if self.evidence[parent] else 0.0
                        parent_influence += self.conditional_probabilities.get((parent, variable), 0.5) * parent_prob
                    else:
                        parent_influence += self.conditional_probabilities.get((parent, variable), 0.5) * self.prior_probabilities.get(parent, 0.5)
                
                # 加权平均
                posterior = 0.7 * prior + 0.3 * (parent_influence / len(parents))
            else:
                posterior = prior
            
            return max(0.0, min(1.0, posterior))
            
        except Exception as e:
            logger.error(f"Probability inference failed: {e}")
            return 0.5

class LogicalReasoner:
    """逻辑推理器"""
    
    def __init__(self):
        self.knowledge_base = []
        self.variables = set()
    
    async def add_proposition(self, proposition: str):
        """添加命题"""
        self.knowledge_base.append(proposition)
        # 提取变量
        variables = re.findall(r'\b[A-Z][a-z]*\b', proposition)
        self.variables.update(variables)
    
    async def modus_ponens(self, premise1: str, premise2: str) -> Optional[str]:
        """假言推理"""
        try:
            # 检查是否满足假言推理的形式
            # 如果 P -> Q 且 P，则 Q
            if "->" in premise1 and premise1.split("->")[0].strip() == premise2.strip():
                conclusion = premise1.split("->")[1].strip()
                return conclusion
            return None
        except Exception as e:
            logger.error(f"Modus ponens failed: {e}")
            return None
    
    async def modus_tollens(self, premise1: str, premise2: str) -> Optional[str]:
        """拒取式"""
        try:
            # 如果 P -> Q 且 ~Q，则 ~P
            if "->" in premise1:
                parts = premise1.split("->")
                if len(parts) == 2:
                    p = parts[0].strip()
                    q = parts[1].strip()
                    if premise2.strip() == f"~{q}" or premise2.strip() == f"not {q}":
                        return f"~{p}"
            return None
        except Exception as e:
            logger.error(f"Modus tollens failed: {e}")
            return None
    
    async def syllogism(self, major_premise: str, minor_premise: str) -> Optional[str]:
        """三段论"""
        try:
            # 简化的三段论推理
            # 所有A都是B，所有B都是C，所以所有A都是C
            if "所有" in major_premise and "所有" in minor_premise:
                # 提取A、B、C
                major_parts = major_premise.replace("所有", "").replace("都是", " ").split()
                minor_parts = minor_premise.replace("所有", "").replace("都是", " ").split()
                
                if len(major_parts) >= 2 and len(minor_parts) >= 2:
                    a = major_parts[0]
                    b = major_parts[1]
                    c = minor_parts[1]
                    
                    if b == minor_parts[0]:  # B匹配
                        return f"所有{a}都是{c}"
            return None
        except Exception as e:
            logger.error(f"Syllogism failed: {e}")
            return None

class ReasoningEngine:
    """推理引擎主类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rule_engine = RuleEngine()
        self.probabilistic_reasoner = ProbabilisticReasoner()
        self.logical_reasoner = LogicalReasoner()
        self.running = False
    
    async def start(self):
        """启动推理引擎"""
        await self.rule_engine.start()
        self.running = True
        logger.info("Reasoning engine started")
    
    async def stop(self):
        """停止推理引擎"""
        await self.rule_engine.stop()
        self.running = False
        logger.info("Reasoning engine stopped")
    
    async def add_rule(self, rule: Rule) -> bool:
        """添加规则"""
        return await self.rule_engine.add_rule(rule)
    
    async def add_fact(self, fact: Fact) -> bool:
        """添加事实"""
        return await self.rule_engine.add_fact(fact)
    
    async def add_proposition(self, proposition: str) -> bool:
        """添加命题"""
        try:
            await self.logical_reasoner.add_proposition(proposition)
            logger.info(f"Added proposition: {proposition}")
            return True
        except Exception as e:
            logger.error(f"Failed to add proposition: {e}")
            return False
    
    async def add_variable(self, variable: str, prior_prob: float = 0.5) -> bool:
        """添加变量"""
        try:
            await self.probabilistic_reasoner.add_variable(variable, prior_prob)
            logger.info(f"Added variable: {variable}")
            return True
        except Exception as e:
            logger.error(f"Failed to add variable: {e}")
            return False
    
    async def add_dependency(self, parent: str, child: str, prob: float) -> bool:
        """添加依赖关系"""
        try:
            await self.probabilistic_reasoner.add_dependency(parent, child, prob)
            logger.info(f"Added dependency: {parent} -> {child} ({prob})")
            return True
        except Exception as e:
            logger.error(f"Failed to add dependency: {e}")
            return False
    
    async def set_evidence(self, variable: str, value: bool) -> bool:
        """设置证据"""
        try:
            await self.probabilistic_reasoner.set_evidence(variable, value)
            logger.info(f"Set evidence: {variable} = {value}")
            return True
        except Exception as e:
            logger.error(f"Failed to set evidence: {e}")
            return False
    
    async def reason(self, reasoning_type: ReasoningType, method: InferenceMethod, 
                    premises: List[str], goal: str = None) -> List[InferenceResult]:
        """执行推理"""
        try:
            if reasoning_type == ReasoningType.DEDUCTIVE:
                if method == InferenceMethod.FORWARD_CHAINING:
                    return await self.rule_engine.forward_chaining(goal)
                elif method == InferenceMethod.BACKWARD_CHAINING:
                    if goal:
                        return await self.rule_engine.backward_chaining(goal)
                    else:
                        return []
                elif method == InferenceMethod.MODUS_PONENS:
                    if len(premises) >= 2:
                        conclusion = await self.logical_reasoner.modus_ponens(premises[0], premises[1])
                        if conclusion:
                            result = InferenceResult(
                                reasoning_type=reasoning_type,
                                method=method,
                                premises=premises,
                                conclusion=conclusion,
                                confidence=0.9,
                                evidence=["Modus Ponens"]
                            )
                            return [result]
                elif method == InferenceMethod.MODUS_TOLLENS:
                    if len(premises) >= 2:
                        conclusion = await self.logical_reasoner.modus_tollens(premises[0], premises[1])
                        if conclusion:
                            result = InferenceResult(
                                reasoning_type=reasoning_type,
                                method=method,
                                premises=premises,
                                conclusion=conclusion,
                                confidence=0.9,
                                evidence=["Modus Tollens"]
                            )
                            return [result]
                elif method == InferenceMethod.SYLLOGISM:
                    if len(premises) >= 2:
                        conclusion = await self.logical_reasoner.syllogism(premises[0], premises[1])
                        if conclusion:
                            result = InferenceResult(
                                reasoning_type=reasoning_type,
                                method=method,
                                premises=premises,
                                conclusion=conclusion,
                                confidence=0.8,
                                evidence=["Syllogism"]
                            )
                            return [result]
            
            elif reasoning_type == ReasoningType.PROBABILISTIC:
                if method == InferenceMethod.BAYESIAN:
                    # 概率推理
                    results = []
                    for premise in premises:
                        prob = await self.probabilistic_reasoner.infer_probability(premise)
                        result = InferenceResult(
                            reasoning_type=reasoning_type,
                            method=method,
                            premises=[premise],
                            conclusion=f"P({premise}) = {prob:.3f}",
                            confidence=prob,
                            evidence=["Bayesian inference"]
                        )
                        results.append(result)
                    return results
            
            logger.warning(f"Unsupported reasoning type/method: {reasoning_type.value}/{method.value}")
            return []
            
        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            return []
    
    async def query(self, query: str) -> List[InferenceResult]:
        """查询推理"""
        try:
            # 解析查询
            if "概率" in query or "probability" in query.lower():
                # 概率查询
                variable = query.replace("概率", "").replace("probability", "").strip()
                prob = await self.probabilistic_reasoner.infer_probability(variable)
                result = InferenceResult(
                    reasoning_type=ReasoningType.PROBABILISTIC,
                    method=InferenceMethod.BAYESIAN,
                    premises=[variable],
                    conclusion=f"P({variable}) = {prob:.3f}",
                    confidence=prob,
                    evidence=["Probability query"]
                )
                return [result]
            
            elif "推理" in query or "reason" in query.lower():
                # 推理查询
                # 这里可以根据查询内容选择推理方法
                return await self.rule_engine.forward_chaining()
            
            else:
                # 默认前向链接
                return await self.rule_engine.forward_chaining()
                
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_rules": len(self.rule_engine.rules),
            "total_facts": len(self.rule_engine.facts),
            "working_memory_size": len(self.rule_engine.working_memory),
            "rule_graph_nodes": self.rule_engine.rule_graph.number_of_nodes(),
            "rule_graph_edges": self.rule_engine.rule_graph.number_of_edges(),
            "bayesian_network_nodes": self.probabilistic_reasoner.bayesian_network.number_of_nodes(),
            "bayesian_network_edges": self.probabilistic_reasoner.bayesian_network.number_of_edges(),
            "logical_propositions": len(self.logical_reasoner.knowledge_base),
            "logical_variables": len(self.logical_reasoner.variables)
        }

# 示例用法
async def main_demo():
    """推理引擎演示"""
    # 创建推理引擎配置
    config = {
        "max_rules": 1000,
        "max_facts": 5000
    }
    
    # 创建推理引擎
    reasoning_engine = ReasoningEngine(config)
    await reasoning_engine.start()
    
    # 添加规则
    rules = [
        Rule(
            name="鸟类规则",
            rule_type=RuleType.IF_THEN,
            antecedent=["是鸟", "有翅膀"],
            consequent=["会飞"],
            confidence=0.9,
            priority=1
        ),
        Rule(
            name="企鹅规则",
            rule_type=RuleType.IF_THEN,
            antecedent=["是企鹅"],
            consequent=["是鸟", "不会飞"],
            confidence=1.0,
            priority=2
        ),
        Rule(
            name="飞行规则",
            rule_type=RuleType.IF_THEN,
            antecedent=["会飞"],
            consequent=["有翅膀"],
            confidence=0.8,
            priority=1
        )
    ]
    
    print("添加规则...")
    for rule in rules:
        await reasoning_engine.add_rule(rule)
    
    # 添加事实
    facts = [
        Fact(statement="是企鹅", confidence=1.0, source="观察"),
        Fact(statement="有翅膀", confidence=0.9, source="观察")
    ]
    
    print("添加事实...")
    for fact in facts:
        await reasoning_engine.add_fact(fact)
    
    # 前向链接推理
    print("\n前向链接推理:")
    forward_results = await reasoning_engine.reason(
        ReasoningType.DEDUCTIVE,
        InferenceMethod.FORWARD_CHAINING
    )
    
    for result in forward_results:
        print(f"结论: {result.conclusion}")
        print(f"置信度: {result.confidence}")
        print(f"前提: {', '.join(result.premises)}")
        print(f"证据: {', '.join(result.evidence)}")
        print()
    
    # 后向链接推理
    print("后向链接推理:")
    backward_results = await reasoning_engine.reason(
        ReasoningType.DEDUCTIVE,
        InferenceMethod.BACKWARD_CHAINING,
        [],
        "会飞"
    )
    
    for result in backward_results:
        print(f"结论: {result.conclusion}")
        print(f"置信度: {result.confidence}")
        print(f"前提: {', '.join(result.premises)}")
        print(f"证据: {', '.join(result.evidence)}")
        print()
    
    # 假言推理
    print("假言推理:")
    modus_ponens_results = await reasoning_engine.reason(
        ReasoningType.DEDUCTIVE,
        InferenceMethod.MODUS_PONENS,
        ["是鸟 -> 会飞", "是鸟"]
    )
    
    for result in modus_ponens_results:
        print(f"结论: {result.conclusion}")
        print(f"置信度: {result.confidence}")
        print()
    
    # 拒取式
    print("拒取式:")
    modus_tollens_results = await reasoning_engine.reason(
        ReasoningType.DEDUCTIVE,
        InferenceMethod.MODUS_TOLLENS,
        ["是鸟 -> 会飞", "不会飞"]
    )
    
    for result in modus_tollens_results:
        print(f"结论: {result.conclusion}")
        print(f"置信度: {result.confidence}")
        print()
    
    # 三段论
    print("三段论:")
    syllogism_results = await reasoning_engine.reason(
        ReasoningType.DEDUCTIVE,
        InferenceMethod.SYLLOGISM,
        ["所有鸟都是动物", "所有动物都是生物"]
    )
    
    for result in syllogism_results:
        print(f"结论: {result.conclusion}")
        print(f"置信度: {result.confidence}")
        print()
    
    # 概率推理
    print("概率推理:")
    await reasoning_engine.add_variable("下雨", 0.3)
    await reasoning_engine.add_variable("带伞", 0.1)
    await reasoning_engine.add_dependency("下雨", "带伞", 0.8)
    await reasoning_engine.set_evidence("下雨", True)
    
    probabilistic_results = await reasoning_engine.reason(
        ReasoningType.PROBABILISTIC,
        InferenceMethod.BAYESIAN,
        ["带伞"]
    )
    
    for result in probabilistic_results:
        print(f"结论: {result.conclusion}")
        print(f"置信度: {result.confidence}")
        print()
    
    # 查询推理
    print("查询推理:")
    query_results = await reasoning_engine.query("概率 带伞")
    
    for result in query_results:
        print(f"结论: {result.conclusion}")
        print(f"置信度: {result.confidence}")
        print()
    
    # 获取统计信息
    print("推理引擎统计:")
    stats = reasoning_engine.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 停止推理引擎
    await reasoning_engine.stop()
    print("\n推理引擎演示完成")

if __name__ == "__main__":
    asyncio.run(main_demo())