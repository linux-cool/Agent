# test_memory_reasoning_system.py
"""
第4章 记忆与推理系统构建 - 测试用例
测试记忆系统、知识图谱、推理引擎、学习系统和检索系统
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
import json
import tempfile
import os

# 导入第4章的核心模块
from code.memory_system import MemorySystem, MemoryType, MemoryManager, MemoryEntry, MemoryQuery
from code.knowledge_graph import KnowledgeGraph, Entity, Relation, EntityType, RelationType, GraphQuery
from code.reasoning_engine import ReasoningEngine, Rule, Fact, ReasoningType, InferenceMethod, RuleType
from code.learning_system import LearningSystem, LearningTask, TrainingData, LearningType, LearningAlgorithm, LearningMode
from code.retrieval_system import RetrievalSystem, Document, Query, RetrievalType, RetrievalResult

# --- 测试辅助函数 ---

@pytest.fixture
def memory_system():
    """创建记忆系统实例"""
    config = {
        "max_memories": 1000,
        "memory_types": ["episodic", "semantic", "procedural"],
        "retention_policy": "lru"
    }
    return MemorySystem(config)

@pytest.fixture
def knowledge_graph():
    """创建知识图谱实例"""
    config = {
        "max_entities": 1000,
        "max_relations": 5000
    }
    return KnowledgeGraph(config)

@pytest.fixture
def reasoning_engine():
    """创建推理引擎实例"""
    config = {
        "max_rules": 1000,
        "max_facts": 5000
    }
    return ReasoningEngine(config)

@pytest.fixture
def learning_system():
    """创建学习系统实例"""
    config = {
        "max_tasks": 100,
        "max_training_data": 10000
    }
    return LearningSystem(config)

@pytest.fixture
def retrieval_system():
    """创建检索系统实例"""
    config = {
        "max_documents": 10000,
        "max_queries": 1000
    }
    return RetrievalSystem(config)

# --- 记忆系统测试 ---

@pytest.mark.asyncio
async def test_memory_system_start_stop(memory_system):
    """测试记忆系统启动和停止"""
    await memory_system.start()
    assert memory_system.running == True
    
    await memory_system.stop()
    assert memory_system.running == False

@pytest.mark.asyncio
async def test_memory_system_add_memory(memory_system):
    """测试添加记忆"""
    await memory_system.start()
    
    memory_entry = MemoryEntry(
        content="这是一个测试记忆",
        memory_type=MemoryType.EPISODIC,
        importance=0.8,
        context={"source": "test"}
    )
    
    result = await memory_system.add_memory(memory_entry)
    assert result == True
    assert len(memory_system.memories) == 1
    
    await memory_system.stop()

@pytest.mark.asyncio
async def test_memory_system_retrieve_memory(memory_system):
    """测试检索记忆"""
    await memory_system.start()
    
    # 添加测试记忆
    memory_entry = MemoryEntry(
        content="测试记忆内容",
        memory_type=MemoryType.SEMANTIC,
        importance=0.9,
        context={"topic": "测试"}
    )
    await memory_system.add_memory(memory_entry)
    
    # 检索记忆
    query = MemoryQuery(
        query_text="测试",
        memory_types=[MemoryType.SEMANTIC],
        limit=5
    )
    
    results = await memory_system.retrieve_memories(query)
    assert len(results) == 1
    assert results[0].content == "测试记忆内容"
    
    await memory_system.stop()

@pytest.mark.asyncio
async def test_memory_system_update_memory(memory_system):
    """测试更新记忆"""
    await memory_system.start()
    
    # 添加记忆
    memory_entry = MemoryEntry(
        content="原始内容",
        memory_type=MemoryType.EPISODIC,
        importance=0.5
    )
    await memory_system.add_memory(memory_entry)
    
    # 更新记忆
    updated_content = "更新后的内容"
    result = await memory_system.update_memory(memory_entry.id, updated_content)
    assert result == True
    
    # 验证更新
    updated_memory = memory_system.memories[memory_entry.id]
    assert updated_memory.content == updated_content
    
    await memory_system.stop()

@pytest.mark.asyncio
async def test_memory_system_forget_memory(memory_system):
    """测试遗忘记忆"""
    await memory_system.start()
    
    # 添加记忆
    memory_entry = MemoryEntry(
        content="要遗忘的记忆",
        memory_type=MemoryType.PROCEDURAL,
        importance=0.3
    )
    await memory_system.add_memory(memory_entry)
    
    # 遗忘记忆
    result = await memory_system.forget_memory(memory_entry.id)
    assert result == True
    assert memory_entry.id not in memory_system.memories
    
    await memory_system.stop()

# --- 知识图谱测试 ---

@pytest.mark.asyncio
async def test_knowledge_graph_start_stop(knowledge_graph):
    """测试知识图谱启动和停止"""
    await knowledge_graph.start()
    assert knowledge_graph.running == True
    
    await knowledge_graph.stop()
    assert knowledge_graph.running == False

@pytest.mark.asyncio
async def test_knowledge_graph_add_entity(knowledge_graph):
    """测试添加实体"""
    await knowledge_graph.start()
    
    entity = Entity(
        name="Python",
        entity_type=EntityType.CONCEPT,
        attributes={"type": "编程语言"}
    )
    
    result = await knowledge_graph.add_entity(entity)
    assert result == True
    assert len(knowledge_graph.entities) == 1
    assert entity.id in knowledge_graph.entities
    
    await knowledge_graph.stop()

@pytest.mark.asyncio
async def test_knowledge_graph_add_relation(knowledge_graph):
    """测试添加关系"""
    await knowledge_graph.start()
    
    # 添加实体
    entity1 = Entity(name="Python", entity_type=EntityType.CONCEPT)
    entity2 = Entity(name="编程语言", entity_type=EntityType.CONCEPT)
    
    await knowledge_graph.add_entity(entity1)
    await knowledge_graph.add_entity(entity2)
    
    # 添加关系
    relation = Relation(
        source_entity_id=entity1.id,
        target_entity_id=entity2.id,
        relation_type=RelationType.IS_A
    )
    
    result = await knowledge_graph.add_relation(relation)
    assert result == True
    assert len(knowledge_graph.relations) == 1
    
    await knowledge_graph.stop()

@pytest.mark.asyncio
async def test_knowledge_graph_find_entity(knowledge_graph):
    """测试查找实体"""
    await knowledge_graph.start()
    
    # 添加实体
    entity = Entity(name="机器学习", entity_type=EntityType.CONCEPT)
    await knowledge_graph.add_entity(entity)
    
    # 查找实体
    results = await knowledge_graph.find_entity("机器")
    assert len(results) == 1
    assert results[0].name == "机器学习"
    
    await knowledge_graph.stop()

@pytest.mark.asyncio
async def test_knowledge_graph_find_path(knowledge_graph):
    """测试查找路径"""
    await knowledge_graph.start()
    
    # 创建实体和关系
    entities = [
        Entity(name="A", entity_type=EntityType.CONCEPT),
        Entity(name="B", entity_type=EntityType.CONCEPT),
        Entity(name="C", entity_type=EntityType.CONCEPT)
    ]
    
    for entity in entities:
        await knowledge_graph.add_entity(entity)
    
    # 创建关系链 A -> B -> C
    relation1 = Relation(
        source_entity_id=entities[0].id,
        target_entity_id=entities[1].id,
        relation_type=RelationType.RELATED_TO
    )
    relation2 = Relation(
        source_entity_id=entities[1].id,
        target_entity_id=entities[2].id,
        relation_type=RelationType.RELATED_TO
    )
    
    await knowledge_graph.add_relation(relation1)
    await knowledge_graph.add_relation(relation2)
    
    # 查找路径
    paths = await knowledge_graph.find_path(entities[0].id, entities[2].id)
    assert len(paths) > 0
    assert len(paths[0]) == 3  # A -> B -> C
    
    await knowledge_graph.stop()

# --- 推理引擎测试 ---

@pytest.mark.asyncio
async def test_reasoning_engine_start_stop(reasoning_engine):
    """测试推理引擎启动和停止"""
    await reasoning_engine.start()
    assert reasoning_engine.running == True
    
    await reasoning_engine.stop()
    assert reasoning_engine.running == False

@pytest.mark.asyncio
async def test_reasoning_engine_add_rule(reasoning_engine):
    """测试添加规则"""
    await reasoning_engine.start()
    
    rule = Rule(
        name="测试规则",
        rule_type=RuleType.IF_THEN,
        antecedent=["是鸟"],
        consequent=["会飞"],
        confidence=0.9
    )
    
    result = await reasoning_engine.add_rule(rule)
    assert result == True
    assert len(reasoning_engine.rule_engine.rules) == 1
    
    await reasoning_engine.stop()

@pytest.mark.asyncio
async def test_reasoning_engine_add_fact(reasoning_engine):
    """测试添加事实"""
    await reasoning_engine.start()
    
    fact = Fact(
        statement="是鸟",
        confidence=1.0,
        source="观察"
    )
    
    result = await reasoning_engine.add_fact(fact)
    assert result == True
    assert len(reasoning_engine.rule_engine.facts) == 1
    
    await reasoning_engine.stop()

@pytest.mark.asyncio
async def test_reasoning_engine_forward_chaining(reasoning_engine):
    """测试前向链接推理"""
    await reasoning_engine.start()
    
    # 添加规则
    rule = Rule(
        name="鸟类规则",
        rule_type=RuleType.IF_THEN,
        antecedent=["是鸟"],
        consequent=["会飞"],
        confidence=0.9
    )
    await reasoning_engine.add_rule(rule)
    
    # 添加事实
    fact = Fact(statement="是鸟", confidence=1.0)
    await reasoning_engine.add_fact(fact)
    
    # 前向链接推理
    results = await reasoning_engine.reason(
        ReasoningType.DEDUCTIVE,
        InferenceMethod.FORWARD_CHAINING
    )
    
    assert len(results) > 0
    assert "会飞" in results[0].conclusion
    
    await reasoning_engine.stop()

@pytest.mark.asyncio
async def test_reasoning_engine_backward_chaining(reasoning_engine):
    """测试后向链接推理"""
    await reasoning_engine.start()
    
    # 添加规则
    rule = Rule(
        name="鸟类规则",
        rule_type=RuleType.IF_THEN,
        antecedent=["是鸟"],
        consequent=["会飞"],
        confidence=0.9
    )
    await reasoning_engine.add_rule(rule)
    
    # 添加事实
    fact = Fact(statement="是鸟", confidence=1.0)
    await reasoning_engine.add_fact(fact)
    
    # 后向链接推理
    results = await reasoning_engine.reason(
        ReasoningType.DEDUCTIVE,
        InferenceMethod.BACKWARD_CHAINING,
        [],
        "会飞"
    )
    
    assert len(results) > 0
    assert results[0].conclusion == "会飞"
    
    await reasoning_engine.stop()

# --- 学习系统测试 ---

@pytest.mark.asyncio
async def test_learning_system_start_stop(learning_system):
    """测试学习系统启动和停止"""
    await learning_system.start()
    assert learning_system.running == True
    
    await learning_system.stop()
    assert learning_system.running == False

@pytest.mark.asyncio
async def test_learning_system_create_task(learning_system):
    """测试创建学习任务"""
    await learning_system.start()
    
    task = LearningTask(
        name="测试任务",
        learning_type=LearningType.SUPERVISED,
        algorithm=LearningAlgorithm.LINEAR_REGRESSION,
        mode=LearningMode.BATCH
    )
    
    result = await learning_system.create_task(task)
    assert result == True
    assert len(learning_system.tasks) == 1
    
    await learning_system.stop()

@pytest.mark.asyncio
async def test_learning_system_add_training_data(learning_system):
    """测试添加训练数据"""
    await learning_system.start()
    
    # 创建任务
    task = LearningTask(
        name="测试任务",
        learning_type=LearningType.SUPERVISED,
        algorithm=LearningAlgorithm.LINEAR_REGRESSION
    )
    await learning_system.create_task(task)
    
    # 添加训练数据
    training_data = TrainingData(
        features=[1.0, 2.0, 3.0],
        label=10.0
    )
    
    result = await learning_system.add_training_data(task.id, training_data)
    assert result == True
    assert len(learning_system.tasks[task.id].training_data) == 1
    
    await learning_system.stop()

@pytest.mark.asyncio
async def test_learning_system_train_model(learning_system):
    """测试训练模型"""
    await learning_system.start()
    
    # 创建任务
    task = LearningTask(
        name="测试任务",
        learning_type=LearningType.SUPERVISED,
        algorithm=LearningAlgorithm.LINEAR_REGRESSION
    )
    await learning_system.create_task(task)
    
    # 添加训练数据
    training_data = [
        TrainingData(features=[1.0, 2.0], label=5.0),
        TrainingData(features=[2.0, 3.0], label=8.0),
        TrainingData(features=[3.0, 4.0], label=11.0)
    ]
    
    for data in training_data:
        await learning_system.add_training_data(task.id, data)
    
    # 训练模型
    result = await learning_system.train_model(task.id)
    assert result == True
    assert learning_system.tasks[task.id].status == "completed"
    assert learning_system.tasks[task.id].model is not None
    
    await learning_system.stop()

@pytest.mark.asyncio
async def test_learning_system_predict(learning_system):
    """测试预测"""
    await learning_system.start()
    
    # 创建并训练任务
    task = LearningTask(
        name="测试任务",
        learning_type=LearningType.SUPERVISED,
        algorithm=LearningAlgorithm.LINEAR_REGRESSION
    )
    await learning_system.create_task(task)
    
    # 添加训练数据
    training_data = [
        TrainingData(features=[1.0, 2.0], label=5.0),
        TrainingData(features=[2.0, 3.0], label=8.0),
        TrainingData(features=[3.0, 4.0], label=11.0)
    ]
    
    for data in training_data:
        await learning_system.add_training_data(task.id, data)
    
    await learning_system.train_model(task.id)
    
    # 预测
    prediction_result = await learning_system.predict(task.id, [4.0, 5.0])
    assert prediction_result.predictions is not None
    assert len(prediction_result.predictions) > 0
    
    await learning_system.stop()

# --- 检索系统测试 ---

@pytest.mark.asyncio
async def test_retrieval_system_start_stop(retrieval_system):
    """测试检索系统启动和停止"""
    await retrieval_system.start()
    assert retrieval_system.running == True
    
    await retrieval_system.stop()
    assert retrieval_system.running == False

@pytest.mark.asyncio
async def test_retrieval_system_add_document(retrieval_system):
    """测试添加文档"""
    await retrieval_system.start()
    
    document = Document(
        title="测试文档",
        content="这是一个测试文档的内容",
        source="test"
    )
    
    result = await retrieval_system.add_document(document)
    assert result == True
    assert len(retrieval_system.documents) == 1
    assert document.id in retrieval_system.documents
    
    await retrieval_system.stop()

@pytest.mark.asyncio
async def test_retrieval_system_keyword_search(retrieval_system):
    """测试关键词搜索"""
    await retrieval_system.start()
    
    # 添加文档
    document = Document(
        title="人工智能文档",
        content="人工智能是计算机科学的重要分支",
        source="test"
    )
    await retrieval_system.add_document(document)
    
    # 关键词搜索
    query = Query(
        text="人工智能",
        query_type=RetrievalType.KEYWORD,
        limit=5
    )
    
    results = await retrieval_system.search(query)
    assert len(results) > 0
    assert results[0].document.title == "人工智能文档"
    
    await retrieval_system.stop()

@pytest.mark.asyncio
async def test_retrieval_system_vector_search(retrieval_system):
    """测试向量搜索"""
    await retrieval_system.start()
    
    # 添加文档
    document = Document(
        title="机器学习文档",
        content="机器学习是人工智能的核心技术",
        source="test"
    )
    await retrieval_system.add_document(document)
    
    # 向量搜索
    query = Query(
        text="AI技术",
        query_type=RetrievalType.VECTOR,
        limit=5
    )
    
    results = await retrieval_system.search(query)
    assert len(results) > 0
    assert results[0].document.title == "机器学习文档"
    
    await retrieval_system.stop()

@pytest.mark.asyncio
async def test_retrieval_system_hybrid_search(retrieval_system):
    """测试混合搜索"""
    await retrieval_system.start()
    
    # 添加文档
    document = Document(
        title="深度学习文档",
        content="深度学习是机器学习的重要分支",
        source="test"
    )
    await retrieval_system.add_document(document)
    
    # 混合搜索
    query = Query(
        text="深度学习 机器学习",
        query_type=RetrievalType.HYBRID,
        limit=5
    )
    
    results = await retrieval_system.search(query)
    assert len(results) > 0
    assert results[0].document.title == "深度学习文档"
    
    await retrieval_system.stop()

# --- 集成测试 ---

@pytest.mark.asyncio
async def test_memory_reasoning_integration():
    """测试记忆系统与推理引擎的集成"""
    # 创建记忆系统
    memory_config = {"max_memories": 1000}
    memory_system = MemorySystem(memory_config)
    await memory_system.start()
    
    # 创建推理引擎
    reasoning_config = {"max_rules": 1000}
    reasoning_engine = ReasoningEngine(reasoning_config)
    await reasoning_engine.start()
    
    # 添加记忆
    memory_entry = MemoryEntry(
        content="鸟会飞",
        memory_type=MemoryType.SEMANTIC,
        importance=0.9
    )
    await memory_system.add_memory(memory_entry)
    
    # 添加规则
    rule = Rule(
        name="飞行规则",
        rule_type=RuleType.IF_THEN,
        antecedent=["是鸟"],
        consequent=["会飞"],
        confidence=0.9
    )
    await reasoning_engine.add_rule(rule)
    
    # 添加事实
    fact = Fact(statement="是鸟", confidence=1.0)
    await reasoning_engine.add_fact(fact)
    
    # 推理
    results = await reasoning_engine.reason(
        ReasoningType.DEDUCTIVE,
        InferenceMethod.FORWARD_CHAINING
    )
    
    assert len(results) > 0
    assert "会飞" in results[0].conclusion
    
    # 清理
    await memory_system.stop()
    await reasoning_engine.stop()

@pytest.mark.asyncio
async def test_knowledge_graph_retrieval_integration():
    """测试知识图谱与检索系统的集成"""
    # 创建知识图谱
    kg_config = {"max_entities": 1000}
    knowledge_graph = KnowledgeGraph(kg_config)
    await knowledge_graph.start()
    
    # 创建检索系统
    retrieval_config = {"max_documents": 1000}
    retrieval_system = RetrievalSystem(retrieval_config)
    await retrieval_system.start()
    
    # 添加实体到知识图谱
    entity = Entity(
        name="Python",
        entity_type=EntityType.CONCEPT,
        attributes={"description": "Python是一种编程语言"}
    )
    await knowledge_graph.add_entity(entity)
    
    # 添加文档到检索系统
    document = Document(
        title="Python编程指南",
        content="Python是一种高级编程语言，具有简洁的语法",
        source="programming_guide"
    )
    await retrieval_system.add_document(document)
    
    # 搜索
    query = Query(
        text="Python编程",
        query_type=RetrievalType.KEYWORD,
        limit=5
    )
    
    results = await retrieval_system.search(query)
    assert len(results) > 0
    assert "Python" in results[0].document.content
    
    # 清理
    await knowledge_graph.stop()
    await retrieval_system.stop()

# --- 性能测试 ---

@pytest.mark.asyncio
async def test_memory_system_performance():
    """测试记忆系统性能"""
    config = {"max_memories": 10000}
    memory_system = MemorySystem(config)
    await memory_system.start()
    
    # 批量添加记忆
    start_time = datetime.now()
    
    for i in range(1000):
        memory_entry = MemoryEntry(
            content=f"测试记忆 {i}",
            memory_type=MemoryType.EPISODIC,
            importance=0.5
        )
        await memory_system.add_memory(memory_entry)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    assert len(memory_system.memories) == 1000
    assert duration < 10.0  # 应该在10秒内完成
    
    await memory_system.stop()

@pytest.mark.asyncio
async def test_retrieval_system_performance():
    """测试检索系统性能"""
    config = {"max_documents": 10000}
    retrieval_system = RetrievalSystem(config)
    await retrieval_system.start()
    
    # 批量添加文档
    start_time = datetime.now()
    
    for i in range(1000):
        document = Document(
            title=f"文档 {i}",
            content=f"这是第 {i} 个测试文档的内容",
            source="test"
        )
        await retrieval_system.add_document(document)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    assert len(retrieval_system.documents) == 1000
    assert duration < 15.0  # 应该在15秒内完成
    
    # 测试搜索性能
    query = Query(
        text="测试文档",
        query_type=RetrievalType.KEYWORD,
        limit=10
    )
    
    start_time = datetime.now()
    results = await retrieval_system.search(query)
    end_time = datetime.now()
    search_duration = (end_time - start_time).total_seconds()
    
    assert len(results) > 0
    assert search_duration < 1.0  # 搜索应该在1秒内完成
    
    await retrieval_system.stop()

# --- 错误处理测试 ---

@pytest.mark.asyncio
async def test_memory_system_error_handling():
    """测试记忆系统错误处理"""
    memory_system = MemorySystem({})
    
    # 测试未启动时添加记忆
    memory_entry = MemoryEntry(content="测试")
    result = await memory_system.add_memory(memory_entry)
    assert result == False
    
    # 测试无效记忆ID
    await memory_system.start()
    result = await memory_system.update_memory("invalid_id", "新内容")
    assert result == False
    
    await memory_system.stop()

@pytest.mark.asyncio
async def test_reasoning_engine_error_handling():
    """测试推理引擎错误处理"""
    reasoning_engine = ReasoningEngine({})
    
    # 测试未启动时添加规则
    rule = Rule(name="测试规则")
    result = await reasoning_engine.add_rule(rule)
    assert result == False
    
    # 测试无效推理类型
    await reasoning_engine.start()
    results = await reasoning_engine.reason(
        ReasoningType.DEDUCTIVE,
        InferenceMethod.FORWARD_CHAINING,
        [],
        "无效目标"
    )
    assert len(results) == 0
    
    await reasoning_engine.stop()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
