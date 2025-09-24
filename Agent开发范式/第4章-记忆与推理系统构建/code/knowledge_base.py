# knowledge_base.py
"""
第4章 记忆与推理系统构建 - 知识库管理系统
实现智能体的知识存储、管理和检索功能
"""

import asyncio
import logging
import json
import hashlib
import sqlite3
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
import numpy as np
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KnowledgeType(Enum):
    """知识类型枚举"""
    FACT = "事实"
    RULE = "规则"
    CONCEPT = "概念"
    PROCEDURE = "程序"
    EXPERIENCE = "经验"
    RELATIONSHIP = "关系"

class KnowledgeSource(Enum):
    """知识来源枚举"""
    USER_INPUT = "用户输入"
    SYSTEM_LEARNING = "系统学习"
    EXTERNAL_API = "外部API"
    DOCUMENT = "文档"
    CONVERSATION = "对话"
    OBSERVATION = "观察"

class KnowledgeStatus(Enum):
    """知识状态枚举"""
    ACTIVE = "活跃"
    INACTIVE = "非活跃"
    VERIFIED = "已验证"
    PENDING = "待验证"
    CONFLICTED = "冲突"
    DEPRECATED = "已废弃"

@dataclass
class Knowledge:
    """知识数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    knowledge_type: KnowledgeType = KnowledgeType.FACT
    source: KnowledgeSource = KnowledgeSource.USER_INPUT
    status: KnowledgeStatus = KnowledgeStatus.ACTIVE
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    relationships: List[str] = field(default_factory=list)
    embeddings: Optional[List[float]] = None
    version: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "content": self.content,
            "knowledge_type": self.knowledge_type.value,
            "source": self.source.value,
            "status": self.status.value,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
            "tags": self.tags,
            "metadata": self.metadata,
            "context": self.context,
            "relationships": self.relationships,
            "embeddings": self.embeddings,
            "version": self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Knowledge':
        """从字典创建知识对象"""
        knowledge = cls()
        knowledge.id = data.get("id", str(uuid.uuid4()))
        knowledge.content = data.get("content", "")
        knowledge.knowledge_type = KnowledgeType(data.get("knowledge_type", "事实"))
        knowledge.source = KnowledgeSource(data.get("source", "用户输入"))
        knowledge.status = KnowledgeStatus(data.get("status", "活跃"))
        knowledge.confidence = data.get("confidence", 1.0)
        knowledge.created_at = datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        knowledge.updated_at = datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat()))
        knowledge.accessed_at = datetime.fromisoformat(data.get("accessed_at", datetime.now().isoformat()))
        knowledge.access_count = data.get("access_count", 0)
        knowledge.tags = data.get("tags", [])
        knowledge.metadata = data.get("metadata", {})
        knowledge.context = data.get("context", {})
        knowledge.relationships = data.get("relationships", [])
        knowledge.embeddings = data.get("embeddings")
        knowledge.version = data.get("version", 1)
        return knowledge

class KnowledgeValidator:
    """知识验证器"""
    
    def __init__(self):
        self.validation_rules = {
            KnowledgeType.FACT: self._validate_fact,
            KnowledgeType.RULE: self._validate_rule,
            KnowledgeType.CONCEPT: self._validate_concept,
            KnowledgeType.PROCEDURE: self._validate_procedure,
            KnowledgeType.EXPERIENCE: self._validate_experience,
            KnowledgeType.RELATIONSHIP: self._validate_relationship
        }
    
    async def validate(self, knowledge: Knowledge) -> Tuple[bool, List[str]]:
        """验证知识"""
        errors = []
        
        # 基本验证
        if not knowledge.content.strip():
            errors.append("知识内容不能为空")
        
        if knowledge.confidence < 0 or knowledge.confidence > 1:
            errors.append("置信度必须在0-1之间")
        
        # 类型特定验证
        if knowledge.knowledge_type in self.validation_rules:
            is_valid, type_errors = await self.validation_rules[knowledge.knowledge_type](knowledge)
            if not is_valid:
                errors.extend(type_errors)
        
        return len(errors) == 0, errors
    
    async def _validate_fact(self, knowledge: Knowledge) -> Tuple[bool, List[str]]:
        """验证事实知识"""
        errors = []
        
        # 事实知识应该包含具体的信息
        if len(knowledge.content.split()) < 3:
            errors.append("事实知识内容过于简单")
        
        return len(errors) == 0, errors
    
    async def _validate_rule(self, knowledge: Knowledge) -> Tuple[bool, List[str]]:
        """验证规则知识"""
        errors = []
        
        # 规则知识应该包含条件或逻辑
        if "如果" not in knowledge.content and "当" not in knowledge.content:
            errors.append("规则知识应该包含条件语句")
        
        return len(errors) == 0, errors
    
    async def _validate_concept(self, knowledge: Knowledge) -> Tuple[bool, List[str]]:
        """验证概念知识"""
        errors = []
        
        # 概念知识应该有定义
        if "是" not in knowledge.content and "定义为" not in knowledge.content:
            errors.append("概念知识应该包含定义")
        
        return len(errors) == 0, errors
    
    async def _validate_procedure(self, knowledge: Knowledge) -> Tuple[bool, List[str]]:
        """验证程序知识"""
        errors = []
        
        # 程序知识应该包含步骤
        if "步骤" not in knowledge.content and "首先" not in knowledge.content:
            errors.append("程序知识应该包含步骤说明")
        
        return len(errors) == 0, errors
    
    async def _validate_experience(self, knowledge: Knowledge) -> Tuple[bool, List[str]]:
        """验证经验知识"""
        errors = []
        
        # 经验知识应该有时间或情境信息
        if not knowledge.context.get("time") and not knowledge.context.get("situation"):
            errors.append("经验知识应该包含时间或情境信息")
        
        return len(errors) == 0, errors
    
    async def _validate_relationship(self, knowledge: Knowledge) -> Tuple[bool, List[str]]:
        """验证关系知识"""
        errors = []
        
        # 关系知识应该包含两个或多个实体
        if len(knowledge.content.split()) < 4:
            errors.append("关系知识应该包含两个或多个实体")
        
        return len(errors) == 0, errors

class KnowledgeIndexer:
    """知识索引器"""
    
    def __init__(self):
        self.content_index: Dict[str, List[str]] = {}  # 内容 -> 知识ID列表
        self.tag_index: Dict[str, List[str]] = {}  # 标签 -> 知识ID列表
        self.type_index: Dict[str, List[str]] = {}  # 类型 -> 知识ID列表
        self.source_index: Dict[str, List[str]] = {}  # 来源 -> 知识ID列表
        self.relationship_index: Dict[str, List[str]] = {}  # 关系 -> 知识ID列表
        self.embedding_index: Dict[str, List[float]] = {}  # 知识ID -> 嵌入向量
    
    async def index(self, knowledge: Knowledge):
        """建立索引"""
        # 内容索引
        words = knowledge.content.lower().split()
        for word in words:
            if word not in self.content_index:
                self.content_index[word] = []
            if knowledge.id not in self.content_index[word]:
                self.content_index[word].append(knowledge.id)
        
        # 标签索引
        for tag in knowledge.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = []
            if knowledge.id not in self.tag_index[tag]:
                self.tag_index[tag].append(knowledge.id)
        
        # 类型索引
        knowledge_type = knowledge.knowledge_type.value
        if knowledge_type not in self.type_index:
            self.type_index[knowledge_type] = []
        if knowledge.id not in self.type_index[knowledge_type]:
            self.type_index[knowledge_type].append(knowledge.id)
        
        # 来源索引
        source = knowledge.source.value
        if source not in self.source_index:
            self.source_index[source] = []
        if knowledge.id not in self.source_index[source]:
            self.source_index[source].append(knowledge.id)
        
        # 关系索引
        for relationship in knowledge.relationships:
            if relationship not in self.relationship_index:
                self.relationship_index[relationship] = []
            if knowledge.id not in self.relationship_index[relationship]:
                self.relationship_index[relationship].append(knowledge.id)
        
        # 嵌入索引
        if knowledge.embeddings:
            self.embedding_index[knowledge.id] = knowledge.embeddings
    
    async def search(self, query: str, filters: Dict[str, Any] = None, limit: int = 10) -> List[str]:
        """搜索知识ID"""
        if filters is None:
            filters = {}
        
        query_words = query.lower().split()
        knowledge_scores: Dict[str, int] = {}
        
        # 计算每个知识的匹配分数
        for word in query_words:
            if word in self.content_index:
                for knowledge_id in self.content_index[word]:
                    knowledge_scores[knowledge_id] = knowledge_scores.get(knowledge_id, 0) + 1
        
        # 应用过滤器
        filtered_knowledge_ids = set(knowledge_scores.keys())
        
        if "knowledge_type" in filters:
            type_filter = filters["knowledge_type"]
            if type_filter in self.type_index:
                filtered_knowledge_ids &= set(self.type_index[type_filter])
        
        if "source" in filters:
            source_filter = filters["source"]
            if source_filter in self.source_index:
                filtered_knowledge_ids &= set(self.source_index[source_filter])
        
        if "tags" in filters:
            tag_filter = filters["tags"]
            for tag in tag_filter:
                if tag in self.tag_index:
                    filtered_knowledge_ids &= set(self.tag_index[tag])
        
        # 按分数排序并返回前N个
        filtered_scores = {k: v for k, v in knowledge_scores.items() if k in filtered_knowledge_ids}
        sorted_knowledge = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)
        return [knowledge_id for knowledge_id, score in sorted_knowledge[:limit]]
    
    async def semantic_search(self, query_embedding: List[float], limit: int = 10) -> List[Tuple[str, float]]:
        """语义搜索"""
        similarities = []
        
        for knowledge_id, embedding in self.embedding_index.items():
            similarity = self._cosine_similarity(query_embedding, embedding)
            similarities.append((knowledge_id, similarity))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def reindex(self, knowledge_id: str):
        """重新建立索引"""
        # 从所有索引中移除该知识
        for word_knowledge in self.content_index.values():
            if knowledge_id in word_knowledge:
                word_knowledge.remove(knowledge_id)
        
        for tag_knowledge in self.tag_index.values():
            if knowledge_id in tag_knowledge:
                tag_knowledge.remove(knowledge_id)
        
        for type_knowledge in self.type_index.values():
            if knowledge_id in type_knowledge:
                type_knowledge.remove(knowledge_id)
        
        for source_knowledge in self.source_index.values():
            if knowledge_id in source_knowledge:
                source_knowledge.remove(knowledge_id)
        
        for relationship_knowledge in self.relationship_index.values():
            if knowledge_id in relationship_knowledge:
                relationship_knowledge.remove(knowledge_id)
        
        if knowledge_id in self.embedding_index:
            del self.embedding_index[knowledge_id]
    
    def get_index_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        return {
            "content_index_size": len(self.content_index),
            "tag_index_size": len(self.tag_index),
            "type_index_size": len(self.type_index),
            "source_index_size": len(self.source_index),
            "relationship_index_size": len(self.relationship_index),
            "embedding_index_size": len(self.embedding_index)
        }

class SQLiteStorage:
    """SQLite存储后端"""
    
    def __init__(self, db_path: str = "knowledge_base.db"):
        self.db_path = db_path
        self.connection = None
    
    async def connect(self):
        """连接数据库"""
        self.connection = sqlite3.connect(self.db_path)
        await self._create_tables()
    
    async def disconnect(self):
        """断开数据库连接"""
        if self.connection:
            self.connection.close()
    
    async def _create_tables(self):
        """创建表"""
        cursor = self.connection.cursor()
        
        # 知识表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                knowledge_type TEXT NOT NULL,
                source TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                accessed_at TEXT NOT NULL,
                access_count INTEGER NOT NULL,
                tags TEXT NOT NULL,
                metadata TEXT NOT NULL,
                context TEXT NOT NULL,
                relationships TEXT NOT NULL,
                embeddings TEXT,
                version INTEGER NOT NULL
            )
        """)
        
        # 关系表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                strength REAL NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (source_id) REFERENCES knowledge (id),
                FOREIGN KEY (target_id) REFERENCES knowledge (id)
            )
        """)
        
        self.connection.commit()
    
    async def store(self, knowledge: Knowledge):
        """存储知识"""
        cursor = self.connection.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO knowledge 
            (id, content, knowledge_type, source, status, confidence, created_at, updated_at, 
             accessed_at, access_count, tags, metadata, context, relationships, embeddings, version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            knowledge.id,
            knowledge.content,
            knowledge.knowledge_type.value,
            knowledge.source.value,
            knowledge.status.value,
            knowledge.confidence,
            knowledge.created_at.isoformat(),
            knowledge.updated_at.isoformat(),
            knowledge.accessed_at.isoformat(),
            knowledge.access_count,
            json.dumps(knowledge.tags),
            json.dumps(knowledge.metadata),
            json.dumps(knowledge.context),
            json.dumps(knowledge.relationships),
            json.dumps(knowledge.embeddings) if knowledge.embeddings else None,
            knowledge.version
        ))
        
        self.connection.commit()
    
    async def retrieve(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """检索知识"""
        cursor = self.connection.cursor()
        
        cursor.execute("SELECT * FROM knowledge WHERE id = ?", (knowledge_id,))
        row = cursor.fetchone()
        
        if row:
            return {
                "id": row[0],
                "content": row[1],
                "knowledge_type": row[2],
                "source": row[3],
                "status": row[4],
                "confidence": row[5],
                "created_at": row[6],
                "updated_at": row[7],
                "accessed_at": row[8],
                "access_count": row[9],
                "tags": json.loads(row[10]),
                "metadata": json.loads(row[11]),
                "context": json.loads(row[12]),
                "relationships": json.loads(row[13]),
                "embeddings": json.loads(row[14]) if row[14] else None,
                "version": row[15]
            }
        
        return None
    
    async def update(self, knowledge_id: str, updates: Dict[str, Any]):
        """更新知识"""
        cursor = self.connection.cursor()
        
        # 构建更新语句
        set_clauses = []
        values = []
        
        for key, value in updates.items():
            if key in ["tags", "metadata", "context", "relationships", "embeddings"]:
                set_clauses.append(f"{key} = ?")
                values.append(json.dumps(value))
            else:
                set_clauses.append(f"{key} = ?")
                values.append(value)
        
        if set_clauses:
            set_clauses.append("updated_at = ?")
            values.append(datetime.now().isoformat())
            values.append(knowledge_id)
            
            query = f"UPDATE knowledge SET {', '.join(set_clauses)} WHERE id = ?"
            cursor.execute(query, values)
            self.connection.commit()
    
    async def delete(self, knowledge_id: str):
        """删除知识"""
        cursor = self.connection.cursor()
        
        # 删除知识
        cursor.execute("DELETE FROM knowledge WHERE id = ?", (knowledge_id,))
        
        # 删除相关关系
        cursor.execute("DELETE FROM relationships WHERE source_id = ? OR target_id = ?", 
                      (knowledge_id, knowledge_id))
        
        self.connection.commit()
    
    async def query(self, query: str, params: List[Any] = None) -> List[Dict[str, Any]]:
        """执行查询"""
        cursor = self.connection.cursor()
        
        if params is None:
            params = []
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # 获取列名
        column_names = [description[0] for description in cursor.description]
        
        # 转换为字典列表
        results = []
        for row in rows:
            result = {}
            for i, value in enumerate(row):
                column_name = column_names[i]
                if column_name in ["tags", "metadata", "context", "relationships", "embeddings"]:
                    result[column_name] = json.loads(value) if value else None
                else:
                    result[column_name] = value
            results.append(result)
        
        return results
    
    def count(self) -> int:
        """获取知识数量"""
        cursor = self.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM knowledge")
        return cursor.fetchone()[0]

class InMemoryStorage:
    """内存存储后端"""
    
    def __init__(self):
        self.storage: Dict[str, Dict[str, Any]] = {}
        self.relationships: Dict[str, Dict[str, Any]] = {}
    
    async def store(self, knowledge: Knowledge):
        """存储知识"""
        self.storage[knowledge.id] = knowledge.to_dict()
    
    async def retrieve(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """检索知识"""
        return self.storage.get(knowledge_id)
    
    async def update(self, knowledge_id: str, updates: Dict[str, Any]):
        """更新知识"""
        if knowledge_id in self.storage:
            self.storage[knowledge_id].update(updates)
    
    async def delete(self, knowledge_id: str):
        """删除知识"""
        if knowledge_id in self.storage:
            del self.storage[knowledge_id]
    
    async def query(self, query: str, params: List[Any] = None) -> List[Dict[str, Any]]:
        """执行查询"""
        # 简单的内存查询实现
        results = []
        for knowledge_data in self.storage.values():
            if query.lower() in knowledge_data.get("content", "").lower():
                results.append(knowledge_data)
        return results
    
    def count(self) -> int:
        """获取知识数量"""
        return len(self.storage)

class KnowledgeBase:
    """知识库管理系统"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.storage_backend = self._create_storage_backend()
        self.indexer = KnowledgeIndexer()
        self.validator = KnowledgeValidator()
        self.embedding_model = None  # 嵌入模型（实际应用中会加载预训练模型）
        self.running = False
    
    def _create_storage_backend(self):
        """创建存储后端"""
        storage_type = self.config.get("storage_type", "memory")
        
        if storage_type == "sqlite":
            return SQLiteStorage(self.config.get("db_path", "knowledge_base.db"))
        else:
            return InMemoryStorage()
    
    async def start(self):
        """启动知识库"""
        self.running = True
        if hasattr(self.storage_backend, 'connect'):
            await self.storage_backend.connect()
        logger.info("Knowledge base started")
    
    async def stop(self):
        """停止知识库"""
        self.running = False
        if hasattr(self.storage_backend, 'disconnect'):
            await self.storage_backend.disconnect()
        logger.info("Knowledge base stopped")
    
    async def add_knowledge(self, knowledge: Knowledge) -> Tuple[bool, List[str]]:
        """添加知识"""
        try:
            # 验证知识
            is_valid, errors = await self.validator.validate(knowledge)
            if not is_valid:
                return False, errors
            
            # 生成嵌入向量
            if not knowledge.embeddings:
                knowledge.embeddings = await self._generate_embedding(knowledge.content)
            
            # 存储知识
            await self.storage_backend.store(knowledge)
            
            # 建立索引
            await self.indexer.index(knowledge)
            
            logger.info(f"Added knowledge: {knowledge.id}")
            return True, []
        except Exception as e:
            logger.error(f"Failed to add knowledge: {e}")
            return False, [str(e)]
    
    async def get_knowledge(self, knowledge_id: str) -> Optional[Knowledge]:
        """获取知识"""
        try:
            knowledge_data = await self.storage_backend.retrieve(knowledge_id)
            if knowledge_data:
                knowledge = Knowledge.from_dict(knowledge_data)
                
                # 更新访问记录
                knowledge.accessed_at = datetime.now()
                knowledge.access_count += 1
                await self.storage_backend.update(knowledge_id, {
                    "accessed_at": knowledge.accessed_at.isoformat(),
                    "access_count": knowledge.access_count
                })
                
                return knowledge
            return None
        except Exception as e:
            logger.error(f"Failed to get knowledge: {e}")
            return None
    
    async def update_knowledge(self, knowledge_id: str, updates: Dict[str, Any]) -> bool:
        """更新知识"""
        try:
            # 获取现有知识
            knowledge = await self.get_knowledge(knowledge_id)
            if not knowledge:
                return False
            
            # 应用更新
            for key, value in updates.items():
                if hasattr(knowledge, key):
                    setattr(knowledge, key, value)
            
            # 更新版本
            knowledge.version += 1
            knowledge.updated_at = datetime.now()
            
            # 重新生成嵌入向量（如果内容发生变化）
            if "content" in updates:
                knowledge.embeddings = await self._generate_embedding(knowledge.content)
            
            # 存储更新后的知识
            await self.storage_backend.store(knowledge)
            
            # 重新建立索引
            await self.indexer.reindex(knowledge_id)
            await self.indexer.index(knowledge)
            
            logger.info(f"Updated knowledge: {knowledge_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update knowledge: {e}")
            return False
    
    async def delete_knowledge(self, knowledge_id: str) -> bool:
        """删除知识"""
        try:
            await self.storage_backend.delete(knowledge_id)
            await self.indexer.reindex(knowledge_id)
            logger.info(f"Deleted knowledge: {knowledge_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete knowledge: {e}")
            return False
    
    async def search_knowledge(self, query: str, filters: Dict[str, Any] = None, 
                             limit: int = 10) -> List[Knowledge]:
        """搜索知识"""
        try:
            # 使用索引器搜索
            knowledge_ids = await self.indexer.search(query, filters, limit)
            knowledge_list = []
            
            for knowledge_id in knowledge_ids:
                knowledge = await self.get_knowledge(knowledge_id)
                if knowledge:
                    knowledge_list.append(knowledge)
            
            return knowledge_list
        except Exception as e:
            logger.error(f"Failed to search knowledge: {e}")
            return []
    
    async def semantic_search(self, query: str, limit: int = 10) -> List[Tuple[Knowledge, float]]:
        """语义搜索"""
        try:
            # 生成查询嵌入向量
            query_embedding = await self._generate_embedding(query)
            
            # 语义搜索
            similarities = await self.indexer.semantic_search(query_embedding, limit)
            
            # 获取知识对象
            results = []
            for knowledge_id, similarity in similarities:
                knowledge = await self.get_knowledge(knowledge_id)
                if knowledge:
                    results.append((knowledge, similarity))
            
            return results
        except Exception as e:
            logger.error(f"Failed to perform semantic search: {e}")
            return []
    
    async def get_knowledge_by_type(self, knowledge_type: KnowledgeType, limit: int = 10) -> List[Knowledge]:
        """按类型获取知识"""
        try:
            knowledge_ids = self.indexer.type_index.get(knowledge_type.value, [])
            knowledge_list = []
            
            for knowledge_id in knowledge_ids[:limit]:
                knowledge = await self.get_knowledge(knowledge_id)
                if knowledge:
                    knowledge_list.append(knowledge)
            
            return knowledge_list
        except Exception as e:
            logger.error(f"Failed to get knowledge by type: {e}")
            return []
    
    async def get_knowledge_by_source(self, source: KnowledgeSource, limit: int = 10) -> List[Knowledge]:
        """按来源获取知识"""
        try:
            knowledge_ids = self.indexer.source_index.get(source.value, [])
            knowledge_list = []
            
            for knowledge_id in knowledge_ids[:limit]:
                knowledge = await self.get_knowledge(knowledge_id)
                if knowledge:
                    knowledge_list.append(knowledge)
            
            return knowledge_list
        except Exception as e:
            logger.error(f"Failed to get knowledge by source: {e}")
            return []
    
    async def get_related_knowledge(self, knowledge_id: str, limit: int = 10) -> List[Knowledge]:
        """获取相关知识"""
        try:
            knowledge = await self.get_knowledge(knowledge_id)
            if not knowledge:
                return []
            
            related_knowledge = []
            for related_id in knowledge.relationships[:limit]:
                related = await self.get_knowledge(related_id)
                if related:
                    related_knowledge.append(related)
            
            return related_knowledge
        except Exception as e:
            logger.error(f"Failed to get related knowledge: {e}")
            return []
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """生成嵌入向量"""
        # 简单的嵌入向量生成（实际应用中会使用预训练模型）
        # 这里使用基于字符的简单哈希方法
        hash_value = hashlib.md5(text.encode()).hexdigest()
        embedding = []
        
        for i in range(0, len(hash_value), 2):
            hex_pair = hash_value[i:i+2]
            embedding.append(int(hex_pair, 16) / 255.0)
        
        # 填充到固定长度
        while len(embedding) < 128:
            embedding.append(0.0)
        
        return embedding[:128]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_knowledge": self.storage_backend.count(),
            "index_stats": self.indexer.get_index_stats(),
            "storage_type": self.config.get("storage_type", "memory")
        }

# 示例用法
async def main_demo():
    """知识库演示"""
    # 创建知识库配置
    config = {
        "storage_type": "memory",
        "db_path": "knowledge_base.db"
    }
    
    # 创建知识库
    knowledge_base = KnowledgeBase(config)
    await knowledge_base.start()
    
    # 创建示例知识
    knowledge_items = [
        Knowledge(
            content="Python是一种高级编程语言",
            knowledge_type=KnowledgeType.FACT,
            source=KnowledgeSource.USER_INPUT,
            confidence=0.9,
            tags=["编程", "Python", "语言"],
            metadata={"domain": "计算机科学", "difficulty": "初级"}
        ),
        Knowledge(
            content="如果用户输入无效数据，则显示错误消息",
            knowledge_type=KnowledgeType.RULE,
            source=KnowledgeSource.SYSTEM_LEARNING,
            confidence=0.8,
            tags=["规则", "验证", "错误处理"],
            metadata={"domain": "软件工程", "priority": "高"}
        ),
        Knowledge(
            content="机器学习是人工智能的一个分支",
            knowledge_type=KnowledgeType.CONCEPT,
            source=KnowledgeSource.DOCUMENT,
            confidence=0.95,
            tags=["机器学习", "AI", "概念"],
            metadata={"domain": "人工智能", "complexity": "中等"}
        )
    ]
    
    # 添加知识
    print("添加知识...")
    for knowledge in knowledge_items:
        success, errors = await knowledge_base.add_knowledge(knowledge)
        print(f"添加知识 {knowledge.id}: {'成功' if success else '失败'}")
        if errors:
            print(f"错误: {errors}")
    
    # 搜索知识
    print("\n搜索知识...")
    search_results = await knowledge_base.search_knowledge("Python", limit=5)
    print(f"找到 {len(search_results)} 个相关知识:")
    for knowledge in search_results:
        print(f"- {knowledge.content} (置信度: {knowledge.confidence})")
    
    # 按类型获取知识
    print("\n按类型获取知识...")
    fact_knowledge = await knowledge_base.get_knowledge_by_type(KnowledgeType.FACT, limit=5)
    print(f"找到 {len(fact_knowledge)} 个事实知识:")
    for knowledge in fact_knowledge:
        print(f"- {knowledge.content}")
    
    # 语义搜索
    print("\n语义搜索...")
    semantic_results = await knowledge_base.semantic_search("编程语言", limit=3)
    print(f"找到 {len(semantic_results)} 个语义相关知识:")
    for knowledge, similarity in semantic_results:
        print(f"- {knowledge.content} (相似度: {similarity:.3f})")
    
    # 获取统计信息
    print("\n知识库统计:")
    stats = knowledge_base.get_stats()
    print(f"总知识数: {stats['total_knowledge']}")
    print(f"内容索引大小: {stats['index_stats']['content_index_size']}")
    print(f"标签索引大小: {stats['index_stats']['tag_index_size']}")
    
    # 停止知识库
    await knowledge_base.stop()
    print("\n知识库演示完成")

if __name__ == "__main__":
    asyncio.run(main_demo())
