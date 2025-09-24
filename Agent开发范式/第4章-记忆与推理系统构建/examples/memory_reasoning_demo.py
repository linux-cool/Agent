# memory_reasoning_demo.py
"""
第4章 记忆与推理系统构建 - 演示程序
展示记忆系统、知识图谱、推理引擎、学习系统和检索系统的综合应用
"""

import asyncio
import logging
import json
import yaml
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import random

# 导入第4章的核心模块
from code.memory_system import MemorySystem, MemoryEntry, MemoryType, MemoryQuery
from code.knowledge_graph import KnowledgeGraph, Entity, Relation, EntityType, RelationType, GraphQuery
from code.reasoning_engine import ReasoningEngine, Rule, Fact, ReasoningType, InferenceMethod, RuleType
from code.learning_system import LearningSystem, LearningTask, TrainingData, LearningType, LearningAlgorithm, LearningMode
from code.retrieval_system import RetrievalSystem, Document, Query, RetrievalType, RetrievalResult

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryReasoningDemo:
    """记忆与推理系统演示类"""
    
    def __init__(self, config_path: str = "config/memory_reasoning_configs.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # 初始化各个系统
        self.memory_system = MemorySystem(self.config.get("memory_system", {}))
        self.knowledge_graph = KnowledgeGraph(self.config.get("knowledge_graph", {}))
        self.reasoning_engine = ReasoningEngine(self.config.get("reasoning_engine", {}))
        self.learning_system = LearningSystem(self.config.get("learning_system", {}))
        self.retrieval_system = RetrievalSystem(self.config.get("retrieval_system", {}))
        
        self.running = False
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return {}
    
    async def start(self):
        """启动所有系统"""
        logger.info("Starting Memory and Reasoning Systems...")
        
        await self.memory_system.start()
        await self.knowledge_graph.start()
        await self.reasoning_engine.start()
        await self.learning_system.start()
        await self.retrieval_system.start()
        
        self.running = True
        logger.info("All systems started successfully")
    
    async def stop(self):
        """停止所有系统"""
        logger.info("Stopping Memory and Reasoning Systems...")
        
        await self.memory_system.stop()
        await self.knowledge_graph.stop()
        await self.reasoning_engine.stop()
        await self.learning_system.stop()
        await self.retrieval_system.stop()
        
        self.running = False
        logger.info("All systems stopped successfully")
    
    async def demo_memory_system(self):
        """演示记忆系统"""
        print("\n" + "="*60)
        print("🧠 记忆系统演示")
        print("="*60)
        
        # 添加不同类型的记忆
        memories = [
            MemoryEntry(
                content="今天学习了Python编程",
                memory_type=MemoryType.EPISODIC,
                importance=0.8,
                context={"date": "2024-01-15", "topic": "编程"}
            ),
            MemoryEntry(
                content="Python是一种高级编程语言",
                memory_type=MemoryType.SEMANTIC,
                importance=0.9,
                context={"category": "知识", "subject": "编程语言"}
            ),
            MemoryEntry(
                content="如何安装Python包：pip install package_name",
                memory_type=MemoryType.PROCEDURAL,
                importance=0.7,
                context={"skill": "包管理", "tool": "pip"}
            ),
            MemoryEntry(
                content="机器学习是人工智能的重要分支",
                memory_type=MemoryType.SEMANTIC,
                importance=0.8,
                context={"field": "AI", "subfield": "ML"}
            ),
            MemoryEntry(
                content="昨天完成了机器学习项目",
                memory_type=MemoryType.EPISODIC,
                importance=0.6,
                context={"date": "2024-01-14", "project": "ML项目"}
            )
        ]
        
        print("添加记忆...")
        for memory in memories:
            await self.memory_system.add_memory(memory)
            print(f"✓ 添加记忆: {memory.content[:30]}...")
        
        # 检索记忆
        print("\n检索记忆:")
        queries = [
            MemoryQuery(query_text="Python", limit=3),
            MemoryQuery(query_text="机器学习", memory_types=[MemoryType.SEMANTIC], limit=2),
            MemoryQuery(query_text="昨天", memory_types=[MemoryType.EPISODIC], limit=2)
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\n查询 {i}: {query.query_text}")
            results = await self.memory_system.retrieve_memories(query)
            for j, result in enumerate(results, 1):
                print(f"  {j}. {result.content} (重要性: {result.importance:.2f})")
        
        # 更新记忆
        print("\n更新记忆:")
        first_memory = memories[0]
        await self.memory_system.update_memory(first_memory.id, "今天深入学习了Python高级编程技巧")
        print(f"✓ 更新记忆: {first_memory.content} -> 今天深入学习了Python高级编程技巧")
        
        # 遗忘记忆
        print("\n遗忘记忆:")
        last_memory = memories[-1]
        await self.memory_system.forget_memory(last_memory.id)
        print(f"✓ 遗忘记忆: {last_memory.content}")
        
        # 获取统计信息
        stats = self.memory_system.get_stats()
        print(f"\n记忆系统统计:")
        print(f"  总记忆数: {stats['total_memories']}")
        print(f"  情景记忆: {stats['memory_type_counts']['episodic']}")
        print(f"  语义记忆: {stats['memory_type_counts']['semantic']}")
        print(f"  程序记忆: {stats['memory_type_counts']['procedural']}")
    
    async def demo_knowledge_graph(self):
        """演示知识图谱"""
        print("\n" + "="*60)
        print("🕸️ 知识图谱演示")
        print("="*60)
        
        # 创建实体
        entities = [
            Entity(name="Python", entity_type=EntityType.CONCEPT, attributes={"type": "编程语言", "creator": "Guido van Rossum"}),
            Entity(name="机器学习", entity_type=EntityType.CONCEPT, attributes={"type": "技术", "field": "人工智能"}),
            Entity(name="深度学习", entity_type=EntityType.CONCEPT, attributes={"type": "技术", "field": "机器学习"}),
            Entity(name="神经网络", entity_type=EntityType.CONCEPT, attributes={"type": "算法", "field": "深度学习"}),
            Entity(name="TensorFlow", entity_type=EntityType.OBJECT, attributes={"type": "框架", "language": "Python"}),
            Entity(name="PyTorch", entity_type=EntityType.OBJECT, attributes={"type": "框架", "language": "Python"}),
            Entity(name="Guido van Rossum", entity_type=EntityType.PERSON, attributes={"profession": "程序员", "nationality": "荷兰"}),
            Entity(name="Google", entity_type=EntityType.ORGANIZATION, attributes={"type": "科技公司", "founded": "1998"})
        ]
        
        print("添加实体...")
        for entity in entities:
            await self.knowledge_graph.add_entity(entity)
            print(f"✓ 添加实体: {entity.name} ({entity.entity_type.value})")
        
        # 创建关系
        relations = [
            Relation(
                source_entity_id=entities[1].id,  # 机器学习
                target_entity_id=entities[2].id,  # 深度学习
                relation_type=RelationType.PART_OF,
                confidence=0.9
            ),
            Relation(
                source_entity_id=entities[2].id,  # 深度学习
                target_entity_id=entities[3].id,  # 神经网络
                relation_type=RelationType.PART_OF,
                confidence=0.8
            ),
            Relation(
                source_entity_id=entities[0].id,  # Python
                target_entity_id=entities[4].id,  # TensorFlow
                relation_type=RelationType.RELATED_TO,
                confidence=0.9
            ),
            Relation(
                source_entity_id=entities[0].id,  # Python
                target_entity_id=entities[5].id,  # PyTorch
                relation_type=RelationType.RELATED_TO,
                confidence=0.9
            ),
            Relation(
                source_entity_id=entities[6].id,  # Guido van Rossum
                target_entity_id=entities[0].id,  # Python
                relation_type=RelationType.FOUNDED,
                confidence=1.0
            ),
            Relation(
                source_entity_id=entities[7].id,  # Google
                target_entity_id=entities[4].id,  # TensorFlow
                relation_type=RelationType.FOUNDED,
                confidence=0.9
            )
        ]
        
        print("\n添加关系...")
        for relation in relations:
            source_name = entities[0].name  # 临时获取名称
            target_name = entities[0].name   # 临时获取名称
            for entity in entities:
                if entity.id == relation.source_entity_id:
                    source_name = entity.name
                if entity.id == relation.target_entity_id:
                    target_name = entity.name
            
            await self.knowledge_graph.add_relation(relation)
            print(f"✓ 添加关系: {source_name} {relation.relation_type.value} {target_name}")
        
        # 实体搜索
        print("\n实体搜索:")
        search_queries = ["Python", "学习", "网络"]
        for query_text in search_queries:
            results = await self.knowledge_graph.find_entity(query_text)
            print(f"搜索 '{query_text}': 找到 {len(results)} 个实体")
            for result in results:
                print(f"  - {result.name} ({result.entity_type.value})")
        
        # 关系搜索
        print("\n关系搜索:")
        python_entity = entities[0]
        relations = await self.knowledge_graph.find_relations(python_entity.id)
        print(f"与 '{python_entity.name}' 相关的关系:")
        for relation in relations:
            source_entity = self.knowledge_graph.entities[relation.source_entity_id]
            target_entity = self.knowledge_graph.entities[relation.target_entity_id]
            print(f"  - {source_entity.name} {relation.relation_type.value} {target_entity.name}")
        
        # 路径查找
        print("\n路径查找:")
        paths = await self.knowledge_graph.find_path(entities[0].id, entities[3].id)  # Python -> 神经网络
        print(f"从 '{entities[0].name}' 到 '{entities[3].name}' 的路径:")
        for path in paths:
            print(f"  - {' -> '.join(path)}")
        
        # 中心性计算
        print("\n中心性计算:")
        centrality = await self.knowledge_graph.calculate_centrality(entities[0].id)  # Python
        print(f"'{entities[0].name}' 的中心性:")
        for metric, value in centrality.items():
            print(f"  - {metric}: {value:.3f}")
        
        # 社区发现
        print("\n社区发现:")
        communities = await self.knowledge_graph.find_communities()
        print(f"发现 {len(communities)} 个社区:")
        for community_id, entity_ids in communities.items():
            entity_names = [self.knowledge_graph.entities[eid].name for eid in entity_ids if eid in self.knowledge_graph.entities]
            print(f"  - {community_id}: {', '.join(entity_names)}")
        
        # 从文本提取
        print("\n从文本提取:")
        text = "Python是机器学习中常用的编程语言。深度学习使用神经网络。TensorFlow是Google开发的框架。"
        extracted_entities, extracted_relations = await self.knowledge_graph.extract_from_text(text)
        print(f"从文本中提取了 {len(extracted_entities)} 个实体和 {len(extracted_relations)} 个关系")
        
        # 获取统计信息
        stats = self.knowledge_graph.get_stats()
        print(f"\n知识图谱统计:")
        print(f"  总实体数: {stats['total_entities']}")
        print(f"  总关系数: {stats['total_relations']}")
        print(f"  图密度: {stats['density']:.3f}")
        print(f"  是否连通: {stats['is_connected']}")
    
    async def demo_reasoning_engine(self):
        """演示推理引擎"""
        print("\n" + "="*60)
        print("🧩 推理引擎演示")
        print("="*60)
        
        # 添加规则
        rules = [
            Rule(
                name="鸟类规则",
                rule_type=RuleType.IF_THEN,
                antecedent=["是鸟"],
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
            ),
            Rule(
                name="编程语言规则",
                rule_type=RuleType.IF_THEN,
                antecedent=["是编程语言"],
                consequent=["可以写程序"],
                confidence=0.9,
                priority=1
            )
        ]
        
        print("添加规则...")
        for rule in rules:
            await self.reasoning_engine.add_rule(rule)
            print(f"✓ 添加规则: {rule.name}")
        
        # 添加事实
        facts = [
            Fact(statement="是企鹅", confidence=1.0, source="观察"),
            Fact(statement="是编程语言", confidence=1.0, source="知识"),
            Fact(statement="Python是编程语言", confidence=1.0, source="知识")
        ]
        
        print("\n添加事实...")
        for fact in facts:
            await self.reasoning_engine.add_fact(fact)
            print(f"✓ 添加事实: {fact.statement}")
        
        # 前向链接推理
        print("\n前向链接推理:")
        forward_results = await self.reasoning_engine.reason(
            ReasoningType.DEDUCTIVE,
            InferenceMethod.FORWARD_CHAINING
        )
        
        for result in forward_results:
            print(f"结论: {result.conclusion}")
            print(f"置信度: {result.confidence:.2f}")
            print(f"前提: {', '.join(result.premises)}")
            print(f"证据: {', '.join(result.evidence)}")
            print()
        
        # 后向链接推理
        print("后向链接推理:")
        backward_results = await self.reasoning_engine.reason(
            ReasoningType.DEDUCTIVE,
            InferenceMethod.BACKWARD_CHAINING,
            [],
            "会飞"
        )
        
        for result in backward_results:
            print(f"结论: {result.conclusion}")
            print(f"置信度: {result.confidence:.2f}")
            print(f"前提: {', '.join(result.premises)}")
            print(f"证据: {', '.join(result.evidence)}")
            print()
        
        # 假言推理
        print("假言推理:")
        modus_ponens_results = await self.reasoning_engine.reason(
            ReasoningType.DEDUCTIVE,
            InferenceMethod.MODUS_PONENS,
            ["是编程语言 -> 可以写程序", "是编程语言"]
        )
        
        for result in modus_ponens_results:
            print(f"结论: {result.conclusion}")
            print(f"置信度: {result.confidence:.2f}")
            print()
        
        # 拒取式
        print("拒取式:")
        modus_tollens_results = await self.reasoning_engine.reason(
            ReasoningType.DEDUCTIVE,
            InferenceMethod.MODUS_TOLLENS,
            ["是鸟 -> 会飞", "不会飞"]
        )
        
        for result in modus_tollens_results:
            print(f"结论: {result.conclusion}")
            print(f"置信度: {result.confidence:.2f}")
            print()
        
        # 三段论
        print("三段论:")
        syllogism_results = await self.reasoning_engine.reason(
            ReasoningType.DEDUCTIVE,
            InferenceMethod.SYLLOGISM,
            ["所有鸟都是动物", "所有动物都是生物"]
        )
        
        for result in syllogism_results:
            print(f"结论: {result.conclusion}")
            print(f"置信度: {result.confidence:.2f}")
            print()
        
        # 概率推理
        print("概率推理:")
        await self.reasoning_engine.add_variable("下雨", 0.3)
        await self.reasoning_engine.add_variable("带伞", 0.1)
        await self.reasoning_engine.add_dependency("下雨", "带伞", 0.8)
        await self.reasoning_engine.set_evidence("下雨", True)
        
        probabilistic_results = await self.reasoning_engine.reason(
            ReasoningType.PROBABILISTIC,
            InferenceMethod.BAYESIAN,
            ["带伞"]
        )
        
        for result in probabilistic_results:
            print(f"结论: {result.conclusion}")
            print(f"置信度: {result.confidence:.2f}")
            print()
        
        # 获取统计信息
        stats = self.reasoning_engine.get_stats()
        print(f"\n推理引擎统计:")
        print(f"  总规则数: {stats['total_rules']}")
        print(f"  总事实数: {stats['total_facts']}")
        print(f"  工作记忆大小: {stats['working_memory_size']}")
    
    async def demo_learning_system(self):
        """演示学习系统"""
        print("\n" + "="*60)
        print("🎓 学习系统演示")
        print("="*60)
        
        # 创建监督学习任务
        print("创建监督学习任务...")
        supervised_task = LearningTask(
            name="房价预测",
            learning_type=LearningType.SUPERVISED,
            algorithm=LearningAlgorithm.LINEAR_REGRESSION,
            mode=LearningMode.BATCH
        )
        
        await self.learning_system.create_task(supervised_task)
        print(f"✓ 创建任务: {supervised_task.name}")
        
        # 添加训练数据
        print("\n添加训练数据...")
        training_data = [
            TrainingData(features=[100, 3, 2], label=500000),  # 面积, 房间数, 浴室数, 价格
            TrainingData(features=[150, 4, 3], label=750000),
            TrainingData(features=[200, 5, 4], label=1000000),
            TrainingData(features=[120, 3, 2], label=600000),
            TrainingData(features=[180, 4, 3], label=900000),
            TrainingData(features=[250, 6, 5], label=1250000),
            TrainingData(features=[90, 2, 1], label=450000),
            TrainingData(features=[160, 4, 3], label=800000),
            TrainingData(features=[220, 5, 4], label=1100000),
            TrainingData(features=[140, 3, 2], label=700000)
        ]
        
        for data in training_data:
            await self.learning_system.add_training_data(supervised_task.id, data)
        
        print(f"✓ 添加了 {len(training_data)} 条训练数据")
        
        # 训练模型
        print("\n训练模型...")
        await self.learning_system.train_model(supervised_task.id)
        print("✓ 模型训练完成")
        
        # 预测
        print("\n进行预测...")
        test_cases = [
            [130, 3, 2],  # 130平米, 3房间, 2浴室
            [200, 4, 3],  # 200平米, 4房间, 3浴室
            [80, 2, 1]    # 80平米, 2房间, 1浴室
        ]
        
        for i, features in enumerate(test_cases, 1):
            prediction_result = await self.learning_system.predict(supervised_task.id, features)
            print(f"测试案例 {i}: {features} -> 预测价格: {prediction_result.predictions[0]:.0f}元")
            print(f"  准确率: {prediction_result.accuracy:.3f}")
            print(f"  置信度: {prediction_result.confidence:.3f}")
        
        # 创建无监督学习任务
        print("\n创建无监督学习任务...")
        unsupervised_task = LearningTask(
            name="客户聚类",
            learning_type=LearningType.UNSUPERVISED,
            algorithm=LearningAlgorithm.K_MEANS,
            mode=LearningMode.BATCH
        )
        
        await self.learning_system.create_task(unsupervised_task)
        print(f"✓ 创建任务: {unsupervised_task.name}")
        
        # 添加聚类数据
        print("\n添加聚类数据...")
        cluster_data = [
            TrainingData(features=[25, 50000, 2]),  # 年龄, 收入, 消费次数
            TrainingData(features=[35, 80000, 5]),
            TrainingData(features=[45, 120000, 8]),
            TrainingData(features=[28, 60000, 3]),
            TrainingData(features=[38, 90000, 6]),
            TrainingData(features=[48, 150000, 10]),
            TrainingData(features=[22, 40000, 1]),
            TrainingData(features=[32, 70000, 4]),
            TrainingData(features=[42, 110000, 7]),
            TrainingData(features=[52, 180000, 12])
        ]
        
        for data in cluster_data:
            await self.learning_system.add_training_data(unsupervised_task.id, data)
        
        print(f"✓ 添加了 {len(cluster_data)} 条聚类数据")
        
        # 训练聚类模型
        print("\n训练聚类模型...")
        await self.learning_system.train_model(unsupervised_task.id)
        print("✓ 聚类模型训练完成")
        
        # 聚类预测
        print("\n进行聚类预测...")
        test_customers = [
            [30, 70000, 4],  # 30岁, 7万收入, 4次消费
            [40, 100000, 6], # 40岁, 10万收入, 6次消费
            [50, 160000, 9]  # 50岁, 16万收入, 9次消费
        ]
        
        for i, features in enumerate(test_customers, 1):
            cluster_result = await self.learning_system.predict(unsupervised_task.id, features)
            print(f"客户 {i}: {features} -> 聚类: {cluster_result.predictions[0]}")
            print(f"  轮廓系数: {cluster_result.accuracy:.3f}")
        
        # 获取统计信息
        stats = self.learning_system.get_stats()
        print(f"\n学习系统统计:")
        print(f"  总任务数: {stats['total_tasks']}")
        print(f"  完成任务数: {stats['completed_tasks']}")
        print(f"  总数据点数: {stats['total_data_points']}")
        print(f"  总结果数: {stats['total_results']}")
    
    async def demo_retrieval_system(self):
        """演示检索系统"""
        print("\n" + "="*60)
        print("🔍 检索系统演示")
        print("="*60)
        
        # 添加文档
        print("添加文档...")
        documents = [
            Document(
                title="人工智能概述",
                content="人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
                source="AI_101"
            ),
            Document(
                title="机器学习基础",
                content="机器学习是人工智能的一个子领域，它使计算机能够在没有明确编程的情况下学习和改进。",
                source="ML_Guide"
            ),
            Document(
                title="深度学习原理",
                content="深度学习是机器学习的一个子集，它使用多层神经网络来模拟人脑的工作方式。",
                source="DL_Book"
            ),
            Document(
                title="自然语言处理",
                content="自然语言处理是人工智能的一个重要分支，它使计算机能够理解、解释和生成人类语言。",
                source="NLP_Tutorial"
            ),
            Document(
                title="计算机视觉",
                content="计算机视觉是人工智能的一个领域，它使计算机能够从图像和视频中提取有意义的信息。",
                source="CV_Handbook"
            ),
            Document(
                title="Python编程指南",
                content="Python是一种高级编程语言，具有简洁的语法和强大的功能，广泛应用于数据科学和人工智能领域。",
                source="Python_Guide"
            ),
            Document(
                title="TensorFlow框架",
                content="TensorFlow是Google开发的开源机器学习框架，支持深度学习和神经网络模型的构建和训练。",
                source="TF_Docs"
            ),
            Document(
                title="PyTorch教程",
                content="PyTorch是Facebook开发的深度学习框架，以其动态计算图和易用性而闻名。",
                source="PT_Tutorial"
            )
        ]
        
        for doc in documents:
            await self.retrieval_system.add_document(doc)
            print(f"✓ 添加文档: {doc.title}")
        
        # 关键词搜索
        print("\n关键词搜索:")
        keyword_queries = [
            "人工智能 机器学习",
            "深度学习 神经网络",
            "Python 编程"
        ]
        
        for query_text in keyword_queries:
            query = Query(
                text=query_text,
                query_type=RetrievalType.KEYWORD,
                limit=3
            )
            
            results = await self.retrieval_system.search(query)
            print(f"\n搜索 '{query_text}':")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result.document.title} (分数: {result.relevance_score:.3f})")
                print(f"     解释: {result.explanation}")
        
        # 向量搜索
        print("\n向量搜索:")
        vector_queries = [
            "如何让计算机学习",
            "图像识别技术",
            "自然语言理解"
        ]
        
        for query_text in vector_queries:
            query = Query(
                text=query_text,
                query_type=RetrievalType.VECTOR,
                limit=3
            )
            
            results = await self.retrieval_system.search(query)
            print(f"\n搜索 '{query_text}':")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result.document.title} (相似度: {result.similarity_score:.3f})")
                print(f"     解释: {result.explanation}")
        
        # 混合搜索
        print("\n混合搜索:")
        hybrid_queries = [
            "神经网络 深度学习",
            "Python 机器学习 框架",
            "人工智能 应用 技术"
        ]
        
        for query_text in hybrid_queries:
            query = Query(
                text=query_text,
                query_type=RetrievalType.HYBRID,
                limit=3
            )
            
            results = await self.retrieval_system.search(query)
            print(f"\n搜索 '{query_text}':")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result.document.title} (相关性: {result.relevance_score:.3f})")
                print(f"     解释: {result.explanation}")
        
        # 语义搜索
        print("\n语义搜索:")
        semantic_queries = [
            "让机器像人一样思考",
            "从图像中提取信息",
            "理解和生成语言"
        ]
        
        for query_text in semantic_queries:
            query = Query(
                text=query_text,
                query_type=RetrievalType.SEMANTIC,
                limit=3
            )
            
            results = await self.retrieval_system.search(query)
            print(f"\n搜索 '{query_text}':")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result.document.title} (相似度: {result.similarity_score:.3f})")
                print(f"     解释: {result.explanation}")
        
        # 获取统计信息
        stats = self.retrieval_system.get_stats()
        print(f"\n检索系统统计:")
        print(f"  总文档数: {stats['total_documents']}")
        print(f"  总查询数: {stats['total_queries']}")
        print(f"  总结果数: {stats['total_results']}")
        print(f"  倒排索引大小: {stats['inverted_index_size']}")
        print(f"  向量索引大小: {stats['vector_index_size']}")
        print(f"  平均文档长度: {stats['average_document_length']:.1f}")
        print(f"  平均关键词数: {stats['average_keywords_per_document']:.1f}")
    
    async def demo_integration(self):
        """演示系统集成"""
        print("\n" + "="*60)
        print("🔗 系统集成演示")
        print("="*60)
        
        # 场景：智能助手学习用户偏好并推荐内容
        
        print("场景：智能助手学习用户偏好并推荐内容")
        print("-" * 40)
        
        # 1. 记忆系统：记录用户行为
        print("\n1. 记录用户行为到记忆系统...")
        user_behaviors = [
            MemoryEntry(
                content="用户喜欢阅读机器学习相关文章",
                memory_type=MemoryType.EPISODIC,
                importance=0.8,
                context={"user_id": "user_001", "behavior": "reading", "topic": "ML"}
            ),
            MemoryEntry(
                content="用户经常搜索Python编程问题",
                memory_type=MemoryType.EPISODIC,
                importance=0.7,
                context={"user_id": "user_001", "behavior": "searching", "topic": "Python"}
            ),
            MemoryEntry(
                content="用户对深度学习感兴趣",
                memory_type=MemoryType.SEMANTIC,
                importance=0.9,
                context={"user_id": "user_001", "interest": "deep_learning"}
            )
        ]
        
        for behavior in user_behaviors:
            await self.memory_system.add_memory(behavior)
            print(f"✓ 记录行为: {behavior.content}")
        
        # 2. 知识图谱：构建用户兴趣图谱
        print("\n2. 构建用户兴趣图谱...")
        user_entities = [
            Entity(name="用户001", entity_type=EntityType.PERSON, attributes={"user_id": "user_001"}),
            Entity(name="机器学习", entity_type=EntityType.CONCEPT, attributes={"category": "AI"}),
            Entity(name="Python", entity_type=EntityType.CONCEPT, attributes={"type": "编程语言"}),
            Entity(name="深度学习", entity_type=EntityType.CONCEPT, attributes={"category": "ML"})
        ]
        
        for entity in user_entities:
            await self.knowledge_graph.add_entity(entity)
        
        # 创建用户兴趣关系
        user_relations = [
            Relation(
                source_entity_id=user_entities[0].id,  # 用户001
                target_entity_id=user_entities[1].id,  # 机器学习
                relation_type=RelationType.RELATED_TO,
                confidence=0.8
            ),
            Relation(
                source_entity_id=user_entities[0].id,  # 用户001
                target_entity_id=user_entities[2].id,  # Python
                relation_type=RelationType.RELATED_TO,
                confidence=0.7
            ),
            Relation(
                source_entity_id=user_entities[1].id,  # 机器学习
                target_entity_id=user_entities[3].id,  # 深度学习
                relation_type=RelationType.PART_OF,
                confidence=0.9
            )
        ]
        
        for relation in user_relations:
            await self.knowledge_graph.add_relation(relation)
            print(f"✓ 创建关系: {relation.relation_type.value}")
        
        # 3. 推理引擎：推理用户偏好
        print("\n3. 推理用户偏好...")
        preference_rules = [
            Rule(
                name="兴趣传播规则",
                rule_type=RuleType.IF_THEN,
                antecedent=["对机器学习感兴趣"],
                consequent=["对深度学习感兴趣"],
                confidence=0.8
            ),
            Rule(
                name="编程语言偏好规则",
                rule_type=RuleType.IF_THEN,
                antecedent=["使用Python编程"],
                consequent=["对Python框架感兴趣"],
                confidence=0.7
            )
        ]
        
        for rule in preference_rules:
            await self.reasoning_engine.add_rule(rule)
        
        # 添加用户事实
        user_facts = [
            Fact(statement="对机器学习感兴趣", confidence=0.8),
            Fact(statement="使用Python编程", confidence=0.7)
        ]
        
        for fact in user_facts:
            await self.reasoning_engine.add_fact(fact)
        
        # 推理用户偏好
        preference_results = await self.reasoning_engine.reason(
            ReasoningType.DEDUCTIVE,
            InferenceMethod.FORWARD_CHAINING
        )
        
        print("推理结果:")
        for result in preference_results:
            print(f"  - {result.conclusion} (置信度: {result.confidence:.2f})")
        
        # 4. 学习系统：学习用户行为模式
        print("\n4. 学习用户行为模式...")
        behavior_task = LearningTask(
            name="用户行为预测",
            learning_type=LearningType.SUPERVISED,
            algorithm=LearningAlgorithm.LINEAR_REGRESSION,
            mode=LearningMode.BATCH
        )
        
        await self.learning_system.create_task(behavior_task)
        
        # 添加用户行为数据
        behavior_data = [
            TrainingData(features=[1, 0, 1], label=0.8),  # [ML, Python, DL] -> 兴趣度
            TrainingData(features=[1, 1, 0], label=0.7),
            TrainingData(features=[0, 1, 1], label=0.6),
            TrainingData(features=[1, 1, 1], label=0.9)
        ]
        
        for data in behavior_data:
            await self.learning_system.add_training_data(behavior_task.id, data)
        
        await self.learning_system.train_model(behavior_task.id)
        print("✓ 用户行为模式学习完成")
        
        # 5. 检索系统：推荐相关内容
        print("\n5. 推荐相关内容...")
        recommendation_query = Query(
            text="机器学习 Python 深度学习",
            query_type=RetrievalType.HYBRID,
            limit=3
        )
        
        recommendations = await self.retrieval_system.search(recommendation_query)
        print("推荐内容:")
        for i, result in enumerate(recommendations, 1):
            print(f"  {i}. {result.document.title}")
            print(f"     相关性: {result.relevance_score:.3f}")
            print(f"     内容: {result.document.content[:50]}...")
        
        # 6. 综合推荐
        print("\n6. 综合推荐结果...")
        print("基于用户行为、兴趣图谱、推理结果和学习模型的综合推荐:")
        
        # 结合记忆系统检索
        memory_query = MemoryQuery(
            query_text="机器学习",
            memory_types=[MemoryType.EPISODIC, MemoryType.SEMANTIC],
            limit=3
        )
        memory_results = await self.memory_system.retrieve_memories(memory_query)
        
        print(f"\n记忆系统推荐 ({len(memory_results)} 条):")
        for result in memory_results:
            print(f"  - {result.content}")
        
        # 结合知识图谱查询
        kg_query = GraphQuery(
            query_type="entity_search",
            parameters={"name": "机器学习", "entity_type": "概念"},
            limit=3
        )
        kg_results = await self.knowledge_graph.query(kg_query)
        
        print(f"\n知识图谱推荐 ({len(kg_results.get('entities', []))} 个实体):")
        for entity in kg_results.get('entities', []):
            print(f"  - {entity.name} ({entity.entity_type.value})")
        
        print("\n✓ 系统集成演示完成")
    
    async def run_full_demo(self):
        """运行完整演示"""
        print("🚀 第4章 记忆与推理系统构建 - 完整演示")
        print("=" * 80)
        
        try:
            await self.start()
            
            # 运行各个系统演示
            await self.demo_memory_system()
            await self.demo_knowledge_graph()
            await self.demo_reasoning_engine()
            await self.demo_learning_system()
            await self.demo_retrieval_system()
            await self.demo_integration()
            
            print("\n" + "=" * 80)
            print("🎉 所有演示完成！")
            print("=" * 80)
            
        except Exception as e:
            logger.error(f"演示过程中发生错误: {e}")
            print(f"\n❌ 演示过程中发生错误: {e}")
        
        finally:
            await self.stop()

# 主函数
async def main():
    """主函数"""
    demo = MemoryReasoningDemo()
    await demo.run_full_demo()

if __name__ == "__main__":
    asyncio.run(main())
