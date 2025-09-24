# knowledge_graph.py
"""
第4章 记忆与推理系统构建 - 知识图谱
实现智能体的知识图谱构建和查询功能
"""

import asyncio
import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import hashlib
from collections import defaultdict, deque
import networkx as nx

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EntityType(Enum):
    """实体类型枚举"""
    PERSON = "人物"
    ORGANIZATION = "组织"
    LOCATION = "地点"
    EVENT = "事件"
    CONCEPT = "概念"
    OBJECT = "对象"
    TIME = "时间"
    QUANTITY = "数量"

class RelationType(Enum):
    """关系类型枚举"""
    IS_A = "是"
    PART_OF = "部分"
    LOCATED_IN = "位于"
    WORKS_FOR = "工作于"
    FOUNDED = "创立"
    OCCURRED_IN = "发生于"
    RELATED_TO = "相关于"
    CAUSES = "导致"
    PRECEDES = "先于"
    FOLLOWS = "后于"

class GraphNodeType(Enum):
    """图节点类型枚举"""
    ENTITY = "实体"
    RELATION = "关系"
    ATTRIBUTE = "属性"
    VALUE = "值"

@dataclass
class Entity:
    """实体数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    entity_type: EntityType = EntityType.CONCEPT
    attributes: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[List[float]] = None
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type.value,
            "attributes": self.attributes,
            "embeddings": self.embeddings,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        """从字典创建实体对象"""
        entity = cls()
        entity.id = data.get("id", str(uuid.uuid4()))
        entity.name = data.get("name", "")
        entity.entity_type = EntityType(data.get("entity_type", "概念"))
        entity.attributes = data.get("attributes", {})
        entity.embeddings = data.get("embeddings")
        entity.confidence = data.get("confidence", 1.0)
        entity.created_at = datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        entity.updated_at = datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat()))
        entity.metadata = data.get("metadata", {})
        return entity

@dataclass
class Relation:
    """关系数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_entity_id: str = ""
    target_entity_id: str = ""
    relation_type: RelationType = RelationType.RELATED_TO
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    weight: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "source_entity_id": self.source_entity_id,
            "target_entity_id": self.target_entity_id,
            "relation_type": self.relation_type.value,
            "attributes": self.attributes,
            "confidence": self.confidence,
            "weight": self.weight,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Relation':
        """从字典创建关系对象"""
        relation = cls()
        relation.id = data.get("id", str(uuid.uuid4()))
        relation.source_entity_id = data.get("source_entity_id", "")
        relation.target_entity_id = data.get("target_entity_id", "")
        relation.relation_type = RelationType(data.get("relation_type", "相关于"))
        relation.attributes = data.get("attributes", {})
        relation.confidence = data.get("confidence", 1.0)
        relation.weight = data.get("weight", 1.0)
        relation.created_at = datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        relation.updated_at = datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat()))
        relation.metadata = data.get("metadata", {})
        return relation

@dataclass
class GraphQuery:
    """图查询数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query_type: str = "entity_search"  # entity_search, relation_search, path_find, subgraph_extract
    parameters: Dict[str, Any] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)
    limit: int = 10
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "query_type": self.query_type,
            "parameters": self.parameters,
            "filters": self.filters,
            "limit": self.limit,
            "created_at": self.created_at.isoformat()
        }

class EntityExtractor:
    """实体提取器"""
    
    def __init__(self):
        self.entity_patterns = {
            EntityType.PERSON: [r"([A-Z][a-z]+ [A-Z][a-z]+)", r"([A-Z][a-z]+先生)", r"([A-Z][a-z]+女士)"],
            EntityType.ORGANIZATION: [r"([A-Z][a-z]+公司)", r"([A-Z][a-z]+大学)", r"([A-Z][a-z]+机构)"],
            EntityType.LOCATION: [r"([A-Z][a-z]+市)", r"([A-Z][a-z]+省)", r"([A-Z][a-z]+国)"],
            EntityType.TIME: [r"(\d{4}年)", r"(\d{1,2}月)", r"(\d{1,2}日)"]
        }
    
    async def extract_entities(self, text: str) -> List[Tuple[str, EntityType]]:
        """从文本中提取实体"""
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    entities.append((match, entity_type))
        
        return entities

class RelationExtractor:
    """关系提取器"""
    
    def __init__(self):
        self.relation_patterns = {
            RelationType.IS_A: [r"(.+)是(.+)", r"(.+)属于(.+)"],
            RelationType.PART_OF: [r"(.+)是(.+)的一部分", r"(.+)包含(.+)"],
            RelationType.LOCATED_IN: [r"(.+)位于(.+)", r"(.+)在(.+)"],
            RelationType.WORKS_FOR: [r"(.+)工作于(.+)", r"(.+)就职于(.+)"],
            RelationType.FOUNDED: [r"(.+)创立了(.+)", r"(.+)创建了(.+)"],
            RelationType.OCCURRED_IN: [r"(.+)发生于(.+)", r"(.+)在(.+)发生"],
            RelationType.CAUSES: [r"(.+)导致(.+)", r"(.+)引起(.+)"],
            RelationType.PRECEDES: [r"(.+)先于(.+)", r"(.+)在(.+)之前"],
            RelationType.FOLLOWS: [r"(.+)后于(.+)", r"(.+)在(.+)之后"]
        }
    
    async def extract_relations(self, text: str) -> List[Tuple[str, str, RelationType]]:
        """从文本中提取关系"""
        relations = []
        
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    if len(match) == 2:
                        relations.append((match[0], match[1], relation_type))
        
        return relations

class KnowledgeGraph:
    """知识图谱主类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.graph = nx.DiGraph()  # 有向图
        self.entities: Dict[str, Entity] = {}
        self.relations: Dict[str, Relation] = {}
        self.entity_extractor = EntityExtractor()
        self.relation_extractor = RelationExtractor()
        self.running = False
    
    async def start(self):
        """启动知识图谱"""
        self.running = True
        logger.info("Knowledge graph started")
    
    async def stop(self):
        """停止知识图谱"""
        self.running = False
        logger.info("Knowledge graph stopped")
    
    async def add_entity(self, entity: Entity) -> bool:
        """添加实体"""
        try:
            self.entities[entity.id] = entity
            
            # 添加到图中
            self.graph.add_node(entity.id, 
                              name=entity.name,
                              entity_type=entity.entity_type.value,
                              attributes=entity.attributes,
                              confidence=entity.confidence)
            
            logger.info(f"Added entity: {entity.name} ({entity.id})")
            return True
        except Exception as e:
            logger.error(f"Failed to add entity: {e}")
            return False
    
    async def add_relation(self, relation: Relation) -> bool:
        """添加关系"""
        try:
            # 检查实体是否存在
            if relation.source_entity_id not in self.entities:
                logger.error(f"Source entity not found: {relation.source_entity_id}")
                return False
            
            if relation.target_entity_id not in self.entities:
                logger.error(f"Target entity not found: {relation.target_entity_id}")
                return False
            
            self.relations[relation.id] = relation
            
            # 添加到图中
            self.graph.add_edge(relation.source_entity_id, 
                               relation.target_entity_id,
                               relation_id=relation.id,
                               relation_type=relation.relation_type.value,
                               attributes=relation.attributes,
                               confidence=relation.confidence,
                               weight=relation.weight)
            
            logger.info(f"Added relation: {relation.relation_type.value} ({relation.id})")
            return True
        except Exception as e:
            logger.error(f"Failed to add relation: {e}")
            return False
    
    async def remove_entity(self, entity_id: str) -> bool:
        """移除实体"""
        try:
            if entity_id not in self.entities:
                logger.error(f"Entity not found: {entity_id}")
                return False
            
            # 移除相关关系
            relations_to_remove = []
            for relation_id, relation in self.relations.items():
                if relation.source_entity_id == entity_id or relation.target_entity_id == entity_id:
                    relations_to_remove.append(relation_id)
            
            for relation_id in relations_to_remove:
                await self.remove_relation(relation_id)
            
            # 从图中移除节点
            self.graph.remove_node(entity_id)
            
            # 从实体字典中移除
            del self.entities[entity_id]
            
            logger.info(f"Removed entity: {entity_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove entity: {e}")
            return False
    
    async def remove_relation(self, relation_id: str) -> bool:
        """移除关系"""
        try:
            if relation_id not in self.relations:
                logger.error(f"Relation not found: {relation_id}")
                return False
            
            relation = self.relations[relation_id]
            
            # 从图中移除边
            if self.graph.has_edge(relation.source_entity_id, relation.target_entity_id):
                self.graph.remove_edge(relation.source_entity_id, relation.target_entity_id)
            
            # 从关系字典中移除
            del self.relations[relation_id]
            
            logger.info(f"Removed relation: {relation_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove relation: {e}")
            return False
    
    async def find_entity(self, name: str, entity_type: EntityType = None) -> List[Entity]:
        """查找实体"""
        entities = []
        
        for entity in self.entities.values():
            if name.lower() in entity.name.lower():
                if entity_type is None or entity.entity_type == entity_type:
                    entities.append(entity)
        
        return entities
    
    async def find_relations(self, entity_id: str, relation_type: RelationType = None) -> List[Relation]:
        """查找关系"""
        relations = []
        
        for relation in self.relations.values():
            if relation.source_entity_id == entity_id or relation.target_entity_id == entity_id:
                if relation_type is None or relation.relation_type == relation_type:
                    relations.append(relation)
        
        return relations
    
    async def find_path(self, source_entity_id: str, target_entity_id: str, max_length: int = 3) -> List[List[str]]:
        """查找路径"""
        try:
            if source_entity_id not in self.entities or target_entity_id not in self.entities:
                return []
            
            # 使用NetworkX查找所有简单路径
            paths = list(nx.all_simple_paths(self.graph, source_entity_id, target_entity_id, cutoff=max_length))
            
            # 转换为实体名称路径
            entity_paths = []
            for path in paths:
                entity_path = []
                for entity_id in path:
                    if entity_id in self.entities:
                        entity_path.append(self.entities[entity_id].name)
                entity_paths.append(entity_path)
            
            return entity_paths
        except Exception as e:
            logger.error(f"Failed to find path: {e}")
            return []
    
    async def get_subgraph(self, entity_ids: List[str]) -> Dict[str, Any]:
        """获取子图"""
        try:
            # 创建子图
            subgraph = self.graph.subgraph(entity_ids)
            
            # 提取实体和关系
            subgraph_entities = {}
            subgraph_relations = {}
            
            for entity_id in entity_ids:
                if entity_id in self.entities:
                    subgraph_entities[entity_id] = self.entities[entity_id]
            
            for relation in self.relations.values():
                if (relation.source_entity_id in entity_ids and 
                    relation.target_entity_id in entity_ids):
                    subgraph_relations[relation.id] = relation
            
            return {
                "entities": subgraph_entities,
                "relations": subgraph_relations,
                "graph": subgraph
            }
        except Exception as e:
            logger.error(f"Failed to get subgraph: {e}")
            return {}
    
    async def calculate_centrality(self, entity_id: str) -> Dict[str, float]:
        """计算中心性"""
        try:
            if entity_id not in self.entities:
                return {}
            
            # 计算各种中心性指标
            centrality_metrics = {}
            
            # 度中心性
            degree_centrality = nx.degree_centrality(self.graph)
            centrality_metrics["degree"] = degree_centrality.get(entity_id, 0.0)
            
            # 接近中心性
            closeness_centrality = nx.closeness_centrality(self.graph)
            centrality_metrics["closeness"] = closeness_centrality.get(entity_id, 0.0)
            
            # 介数中心性
            betweenness_centrality = nx.betweenness_centrality(self.graph)
            centrality_metrics["betweenness"] = betweenness_centrality.get(entity_id, 0.0)
            
            # 特征向量中心性
            try:
                eigenvector_centrality = nx.eigenvector_centrality(self.graph)
                centrality_metrics["eigenvector"] = eigenvector_centrality.get(entity_id, 0.0)
            except:
                centrality_metrics["eigenvector"] = 0.0
            
            return centrality_metrics
        except Exception as e:
            logger.error(f"Failed to calculate centrality: {e}")
            return {}
    
    async def find_communities(self) -> Dict[str, List[str]]:
        """发现社区"""
        try:
            # 使用Louvain算法发现社区
            communities = nx.community.louvain_communities(self.graph)
            
            # 转换为字典格式
            community_dict = {}
            for i, community in enumerate(communities):
                community_dict[f"community_{i}"] = list(community)
            
            return community_dict
        except Exception as e:
            logger.error(f"Failed to find communities: {e}")
            return {}
    
    async def extract_from_text(self, text: str) -> Tuple[List[Entity], List[Relation]]:
        """从文本中提取实体和关系"""
        try:
            # 提取实体
            extracted_entities = await self.entity_extractor.extract_entities(text)
            entities = []
            
            for name, entity_type in extracted_entities:
                # 检查实体是否已存在
                existing_entities = await self.find_entity(name, entity_type)
                if not existing_entities:
                    entity = Entity(
                        name=name,
                        entity_type=entity_type,
                        confidence=0.8
                    )
                    await self.add_entity(entity)
                    entities.append(entity)
                else:
                    entities.extend(existing_entities)
            
            # 提取关系
            extracted_relations = await self.relation_extractor.extract_relations(text)
            relations = []
            
            for source_name, target_name, relation_type in extracted_relations:
                # 查找源实体和目标实体
                source_entities = await self.find_entity(source_name)
                target_entities = await self.find_entity(target_name)
                
                if source_entities and target_entities:
                    # 创建关系
                    relation = Relation(
                        source_entity_id=source_entities[0].id,
                        target_entity_id=target_entities[0].id,
                        relation_type=relation_type,
                        confidence=0.8
                    )
                    await self.add_relation(relation)
                    relations.append(relation)
            
            logger.info(f"Extracted {len(entities)} entities and {len(relations)} relations from text")
            return entities, relations
            
        except Exception as e:
            logger.error(f"Failed to extract from text: {e}")
            return [], []
    
    async def query(self, query: GraphQuery) -> Dict[str, Any]:
        """执行图查询"""
        try:
            if query.query_type == "entity_search":
                return await self._entity_search_query(query)
            elif query.query_type == "relation_search":
                return await self._relation_search_query(query)
            elif query.query_type == "path_find":
                return await self._path_find_query(query)
            elif query.query_type == "subgraph_extract":
                return await self._subgraph_extract_query(query)
            else:
                logger.error(f"Unknown query type: {query.query_type}")
                return {}
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {}
    
    async def _entity_search_query(self, query: GraphQuery) -> Dict[str, Any]:
        """实体搜索查询"""
        name = query.parameters.get("name", "")
        entity_type = query.parameters.get("entity_type")
        
        if entity_type:
            entity_type = EntityType(entity_type)
        
        entities = await self.find_entity(name, entity_type)
        return {"entities": entities[:query.limit]}
    
    async def _relation_search_query(self, query: GraphQuery) -> Dict[str, Any]:
        """关系搜索查询"""
        entity_id = query.parameters.get("entity_id", "")
        relation_type = query.parameters.get("relation_type")
        
        if relation_type:
            relation_type = RelationType(relation_type)
        
        relations = await self.find_relations(entity_id, relation_type)
        return {"relations": relations[:query.limit]}
    
    async def _path_find_query(self, query: GraphQuery) -> Dict[str, Any]:
        """路径查找查询"""
        source_id = query.parameters.get("source_entity_id", "")
        target_id = query.parameters.get("target_entity_id", "")
        max_length = query.parameters.get("max_length", 3)
        
        paths = await self.find_path(source_id, target_id, max_length)
        return {"paths": paths}
    
    async def _subgraph_extract_query(self, query: GraphQuery) -> Dict[str, Any]:
        """子图提取查询"""
        entity_ids = query.parameters.get("entity_ids", [])
        subgraph = await self.get_subgraph(entity_ids)
        return subgraph
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "graph_nodes": self.graph.number_of_nodes(),
            "graph_edges": self.graph.number_of_edges(),
            "is_connected": nx.is_weakly_connected(self.graph),
            "density": nx.density(self.graph),
            "average_clustering": nx.average_clustering(self.graph.to_undirected()) if self.graph.number_of_nodes() > 0 else 0.0
        }

# 示例用法
async def main_demo():
    """知识图谱演示"""
    # 创建知识图谱配置
    config = {
        "max_entities": 1000,
        "max_relations": 5000
    }
    
    # 创建知识图谱
    knowledge_graph = KnowledgeGraph(config)
    await knowledge_graph.start()
    
    # 创建示例实体
    entities = [
        Entity(name="Python", entity_type=EntityType.CONCEPT, attributes={"type": "编程语言"}),
        Entity(name="机器学习", entity_type=EntityType.CONCEPT, attributes={"type": "技术"}),
        Entity(name="深度学习", entity_type=EntityType.CONCEPT, attributes={"type": "技术"}),
        Entity(name="神经网络", entity_type=EntityType.CONCEPT, attributes={"type": "算法"}),
        Entity(name="数据科学", entity_type=EntityType.CONCEPT, attributes={"type": "领域"}),
        Entity(name="人工智能", entity_type=EntityType.CONCEPT, attributes={"type": "领域"})
    ]
    
    # 添加实体
    print("添加实体...")
    for entity in entities:
        await knowledge_graph.add_entity(entity)
    
    # 创建示例关系
    relations = [
        Relation(
            source_entity_id=entities[0].id,  # Python
            target_entity_id=entities[4].id,  # 数据科学
            relation_type=RelationType.RELATED_TO,
            weight=0.8
        ),
        Relation(
            source_entity_id=entities[1].id,  # 机器学习
            target_entity_id=entities[5].id,  # 人工智能
            relation_type=RelationType.PART_OF,
            weight=0.9
        ),
        Relation(
            source_entity_id=entities[2].id,  # 深度学习
            target_entity_id=entities[1].id,  # 机器学习
            relation_type=RelationType.PART_OF,
            weight=0.9
        ),
        Relation(
            source_entity_id=entities[3].id,  # 神经网络
            target_entity_id=entities[2].id,  # 深度学习
            relation_type=RelationType.PART_OF,
            weight=0.8
        ),
        Relation(
            source_entity_id=entities[1].id,  # 机器学习
            target_entity_id=entities[4].id,  # 数据科学
            relation_type=RelationType.RELATED_TO,
            weight=0.7
        )
    ]
    
    # 添加关系
    print("添加关系...")
    for relation in relations:
        await knowledge_graph.add_relation(relation)
    
    # 实体搜索
    print("\n实体搜索:")
    search_results = await knowledge_graph.find_entity("机器")
    print(f"找到 {len(search_results)} 个实体:")
    for entity in search_results:
        print(f"- {entity.name} ({entity.entity_type.value})")
    
    # 关系搜索
    print("\n关系搜索:")
    relation_results = await knowledge_graph.find_relations(entities[1].id)  # 机器学习
    print(f"找到 {len(relation_results)} 个关系:")
    for relation in relation_results:
        source_entity = knowledge_graph.entities[relation.source_entity_id]
        target_entity = knowledge_graph.entities[relation.target_entity_id]
        print(f"- {source_entity.name} {relation.relation_type.value} {target_entity.name}")
    
    # 路径查找
    print("\n路径查找:")
    paths = await knowledge_graph.find_path(entities[0].id, entities[5].id)  # Python -> 人工智能
    print(f"找到 {len(paths)} 条路径:")
    for path in paths:
        print(f"- {' -> '.join(path)}")
    
    # 中心性计算
    print("\n中心性计算:")
    centrality = await knowledge_graph.calculate_centrality(entities[1].id)  # 机器学习
    print(f"机器学习中心性:")
    for metric, value in centrality.items():
        print(f"- {metric}: {value:.3f}")
    
    # 社区发现
    print("\n社区发现:")
    communities = await knowledge_graph.find_communities()
    print(f"找到 {len(communities)} 个社区:")
    for community_id, entity_ids in communities.items():
        entity_names = [knowledge_graph.entities[eid].name for eid in entity_ids if eid in knowledge_graph.entities]
        print(f"- {community_id}: {', '.join(entity_names)}")
    
    # 从文本提取
    print("\n从文本提取:")
    text = "Python是数据科学中常用的编程语言。机器学习是人工智能的重要分支。深度学习使用神经网络。"
    extracted_entities, extracted_relations = await knowledge_graph.extract_from_text(text)
    print(f"提取了 {len(extracted_entities)} 个实体和 {len(extracted_relations)} 个关系")
    
    # 图查询
    print("\n图查询:")
    query = GraphQuery(
        query_type="entity_search",
        parameters={"name": "学习", "entity_type": "概念"},
        limit=5
    )
    query_result = await knowledge_graph.query(query)
    print(f"查询结果: {len(query_result.get('entities', []))} 个实体")
    
    # 获取统计信息
    print("\n知识图谱统计:")
    stats = knowledge_graph.get_stats()
    print(f"总实体数: {stats['total_entities']}")
    print(f"总关系数: {stats['total_relations']}")
    print(f"图节点数: {stats['graph_nodes']}")
    print(f"图边数: {stats['graph_edges']}")
    print(f"是否连通: {stats['is_connected']}")
    print(f"图密度: {stats['density']:.3f}")
    print(f"平均聚类系数: {stats['average_clustering']:.3f}")
    
    # 停止知识图谱
    await knowledge_graph.stop()
    print("\n知识图谱演示完成")

if __name__ == "__main__":
    asyncio.run(main_demo())
