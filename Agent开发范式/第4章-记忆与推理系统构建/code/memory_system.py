# memory_system.py
"""
第4章 记忆与推理系统构建 - 记忆系统架构
实现智能体的记忆管理、存储和检索功能
"""

import asyncio
import logging
import time
import json
import hashlib
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryType(Enum):
    """记忆类型枚举"""
    SHORT_TERM = "短期记忆"
    LONG_TERM = "长期记忆"
    WORKING = "工作记忆"
    EPISODIC = "情景记忆"
    SEMANTIC = "语义记忆"
    PROCEDURAL = "程序记忆"

class MemoryPriority(Enum):
    """记忆优先级枚举"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class MemoryStatus(Enum):
    """记忆状态枚举"""
    ACTIVE = "活跃"
    INACTIVE = "非活跃"
    ARCHIVED = "已归档"
    DELETED = "已删除"

@dataclass
class Memory:
    """记忆数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    memory_type: MemoryType = MemoryType.SHORT_TERM
    priority: MemoryPriority = MemoryPriority.MEDIUM
    status: MemoryStatus = MemoryStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    importance_score: float = 0.5
    decay_rate: float = 0.1
    ttl: Optional[timedelta] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
            "tags": self.tags,
            "metadata": self.metadata,
            "context": self.context,
            "importance_score": self.importance_score,
            "decay_rate": self.decay_rate,
            "ttl": self.ttl.total_seconds() if self.ttl else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """从字典创建记忆对象"""
        memory = cls()
        memory.id = data.get("id", str(uuid.uuid4()))
        memory.content = data.get("content", "")
        memory.memory_type = MemoryType(data.get("memory_type", "短期记忆"))
        memory.priority = MemoryPriority(data.get("priority", 2))
        memory.status = MemoryStatus(data.get("status", "活跃"))
        memory.created_at = datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        memory.updated_at = datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat()))
        memory.accessed_at = datetime.fromisoformat(data.get("accessed_at", datetime.now().isoformat()))
        memory.access_count = data.get("access_count", 0)
        memory.tags = data.get("tags", [])
        memory.metadata = data.get("metadata", {})
        memory.context = data.get("context", {})
        memory.importance_score = data.get("importance_score", 0.5)
        memory.decay_rate = data.get("decay_rate", 0.1)
        ttl_seconds = data.get("ttl")
        memory.ttl = timedelta(seconds=ttl_seconds) if ttl_seconds else None
        return memory

class ShortTermMemory:
    """短期记忆管理器"""
    
    def __init__(self, capacity: int = 1000, max_age: int = 3600):
        self.capacity = capacity
        self.max_age = max_age  # 最大存活时间（秒）
        self.memories: Dict[str, Memory] = {}
        self.access_order: List[str] = []  # LRU访问顺序
        self.running = False
        self.cleanup_task: Optional[asyncio.Task] = None
    
    async def store(self, memory: Memory) -> bool:
        """存储记忆"""
        try:
            # 检查容量限制
            if len(self.memories) >= self.capacity:
                await self._evict_oldest()
            
            # 存储记忆
            self.memories[memory.id] = memory
            self.access_order.append(memory.id)
            
            logger.info(f"Stored short-term memory: {memory.id}")
            return True
        except Exception as e:
            logger.error(f"Failed to store short-term memory: {e}")
            return False
    
    async def retrieve(self, memory_id: str) -> Optional[Memory]:
        """检索记忆"""
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            memory.accessed_at = datetime.now()
            memory.access_count += 1
            
            # 更新访问顺序
            if memory_id in self.access_order:
                self.access_order.remove(memory_id)
            self.access_order.append(memory_id)
            
            logger.debug(f"Retrieved short-term memory: {memory_id}")
            return memory
        return None
    
    async def search(self, query: str, limit: int = 10) -> List[Memory]:
        """搜索记忆"""
        results = []
        for memory in self.memories.values():
            if query.lower() in memory.content.lower():
                results.append(memory)
        
        # 按重要性排序
        results.sort(key=lambda m: m.importance_score, reverse=True)
        return results[:limit]
    
    async def _evict_oldest(self):
        """驱逐最旧的记忆"""
        if self.access_order:
            oldest_id = self.access_order[0]
            if oldest_id in self.memories:
                del self.memories[oldest_id]
                self.access_order.pop(0)
                logger.debug(f"Evicted oldest short-term memory: {oldest_id}")
    
    async def _cleanup_expired(self):
        """清理过期记忆"""
        current_time = datetime.now()
        expired_ids = []
        
        for memory_id, memory in self.memories.items():
            age = (current_time - memory.created_at).total_seconds()
            if age > self.max_age:
                expired_ids.append(memory_id)
        
        for memory_id in expired_ids:
            del self.memories[memory_id]
            if memory_id in self.access_order:
                self.access_order.remove(memory_id)
            logger.debug(f"Cleaned up expired short-term memory: {memory_id}")
    
    async def start_cleanup(self):
        """启动清理任务"""
        self.running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop_cleanup(self):
        """停止清理任务"""
        self.running = False
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                logger.info("Short-term memory cleanup task cancelled")
    
    async def _cleanup_loop(self):
        """清理循环"""
        while self.running:
            await self._cleanup_expired()
            await asyncio.sleep(60)  # 每分钟清理一次
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_memories": len(self.memories),
            "capacity": self.capacity,
            "usage_percentage": len(self.memories) / self.capacity * 100,
            "oldest_memory": min(self.memories.values(), key=lambda m: m.created_at).created_at if self.memories else None,
            "newest_memory": max(self.memories.values(), key=lambda m: m.created_at).created_at if self.memories else None
        }

class LongTermMemory:
    """长期记忆管理器"""
    
    def __init__(self, storage_backend: Optional[Any] = None):
        self.storage_backend = storage_backend or InMemoryStorage()
        self.indexer = MemoryIndexer()
        self.compression_ratio = 0.8  # 压缩比例
    
    async def store(self, memory: Memory) -> bool:
        """存储记忆"""
        try:
            # 压缩记忆内容
            compressed_memory = await self._compress_memory(memory)
            
            # 存储到后端
            await self.storage_backend.store(compressed_memory)
            
            # 建立索引
            await self.indexer.index(compressed_memory)
            
            logger.info(f"Stored long-term memory: {memory.id}")
            return True
        except Exception as e:
            logger.error(f"Failed to store long-term memory: {e}")
            return False
    
    async def retrieve(self, memory_id: str) -> Optional[Memory]:
        """检索记忆"""
        try:
            memory_data = await self.storage_backend.retrieve(memory_id)
            if memory_data:
                memory = Memory.from_dict(memory_data)
                memory.accessed_at = datetime.now()
                memory.access_count += 1
                
                # 更新访问记录
                await self.storage_backend.update(memory_id, {"accessed_at": memory.accessed_at.isoformat(), "access_count": memory.access_count})
                
                logger.debug(f"Retrieved long-term memory: {memory_id}")
                return memory
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve long-term memory: {e}")
            return None
    
    async def search(self, query: str, limit: int = 10) -> List[Memory]:
        """搜索记忆"""
        try:
            # 使用索引器搜索
            memory_ids = await self.indexer.search(query, limit)
            memories = []
            
            for memory_id in memory_ids:
                memory = await self.retrieve(memory_id)
                if memory:
                    memories.append(memory)
            
            return memories
        except Exception as e:
            logger.error(f"Failed to search long-term memory: {e}")
            return []
    
    async def _compress_memory(self, memory: Memory) -> Memory:
        """压缩记忆"""
        # 简单的压缩策略：减少元数据，保留核心内容
        compressed_memory = Memory(
            id=memory.id,
            content=memory.content,
            memory_type=memory.memory_type,
            priority=memory.priority,
            status=memory.status,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
            accessed_at=memory.accessed_at,
            access_count=memory.access_count,
            tags=memory.tags[:5],  # 限制标签数量
            metadata={k: v for k, v in memory.metadata.items() if k in ["source", "confidence"]},  # 只保留关键元数据
            context=memory.context,
            importance_score=memory.importance_score,
            decay_rate=memory.decay_rate,
            ttl=memory.ttl
        )
        return compressed_memory
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_memories": self.storage_backend.count(),
            "index_size": self.indexer.get_index_size(),
            "compression_ratio": self.compression_ratio
        }

class WorkingMemory:
    """工作记忆管理器"""
    
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.memories: Dict[str, Memory] = {}
        self.current_context: Dict[str, Any] = {}
        self.active_tasks: List[str] = []
    
    async def store(self, memory: Memory) -> bool:
        """存储记忆"""
        try:
            # 检查容量限制
            if len(self.memories) >= self.capacity:
                await self._evict_least_important()
            
            # 存储记忆
            self.memories[memory.id] = memory
            
            logger.info(f"Stored working memory: {memory.id}")
            return True
        except Exception as e:
            logger.error(f"Failed to store working memory: {e}")
            return False
    
    async def retrieve(self, memory_id: str) -> Optional[Memory]:
        """检索记忆"""
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            memory.accessed_at = datetime.now()
            memory.access_count += 1
            logger.debug(f"Retrieved working memory: {memory_id}")
            return memory
        return None
    
    async def get_context_memories(self) -> List[Memory]:
        """获取上下文相关记忆"""
        context_memories = []
        for memory in self.memories.values():
            if self._is_context_relevant(memory):
                context_memories.append(memory)
        
        # 按重要性排序
        context_memories.sort(key=lambda m: m.importance_score, reverse=True)
        return context_memories
    
    def _is_context_relevant(self, memory: Memory) -> bool:
        """判断记忆是否与当前上下文相关"""
        # 简单的相关性判断：检查标签和元数据
        for tag in memory.tags:
            if tag in self.current_context.get("tags", []):
                return True
        
        for key, value in memory.context.items():
            if key in self.current_context and self.current_context[key] == value:
                return True
        
        return False
    
    async def _evict_least_important(self):
        """驱逐最不重要的记忆"""
        if self.memories:
            least_important = min(self.memories.values(), key=lambda m: m.importance_score)
            del self.memories[least_important.id]
            logger.debug(f"Evicted least important working memory: {least_important.id}")
    
    def set_context(self, context: Dict[str, Any]):
        """设置当前上下文"""
        self.current_context = context
    
    def add_active_task(self, task_id: str):
        """添加活跃任务"""
        if task_id not in self.active_tasks:
            self.active_tasks.append(task_id)
    
    def remove_active_task(self, task_id: str):
        """移除活跃任务"""
        if task_id in self.active_tasks:
            self.active_tasks.remove(task_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_memories": len(self.memories),
            "capacity": self.capacity,
            "usage_percentage": len(self.memories) / self.capacity * 100,
            "active_tasks": len(self.active_tasks),
            "context_keys": list(self.current_context.keys())
        }

class MetaMemory:
    """元记忆管理器"""
    
    def __init__(self):
        self.memory_metadata: Dict[str, Dict[str, Any]] = {}
        self.access_patterns: Dict[str, List[datetime]] = {}
        self.importance_scores: Dict[str, float] = {}
        self.relationships: Dict[str, List[str]] = {}
    
    async def update(self, memory_id: str, updates: Dict[str, Any]):
        """更新记忆元数据"""
        if memory_id not in self.memory_metadata:
            self.memory_metadata[memory_id] = {}
        
        self.memory_metadata[memory_id].update(updates)
        
        # 更新访问模式
        if "accessed_at" in updates:
            if memory_id not in self.access_patterns:
                self.access_patterns[memory_id] = []
            self.access_patterns[memory_id].append(datetime.fromisoformat(updates["accessed_at"]))
        
        # 更新重要性分数
        if "importance_score" in updates:
            self.importance_scores[memory_id] = updates["importance_score"]
        
        logger.debug(f"Updated meta-memory for: {memory_id}")
    
    async def get_access_frequency(self, memory_id: str) -> float:
        """获取访问频率"""
        if memory_id not in self.access_patterns:
            return 0.0
        
        access_times = self.access_patterns[memory_id]
        if len(access_times) < 2:
            return 0.0
        
        # 计算最近24小时的访问频率
        recent_time = datetime.now() - timedelta(hours=24)
        recent_accesses = [t for t in access_times if t > recent_time]
        return len(recent_accesses) / 24.0  # 每小时访问次数
    
    async def get_importance_score(self, memory_id: str) -> float:
        """获取重要性分数"""
        return self.importance_scores.get(memory_id, 0.5)
    
    async def add_relationship(self, memory_id: str, related_id: str):
        """添加记忆关系"""
        if memory_id not in self.relationships:
            self.relationships[memory_id] = []
        
        if related_id not in self.relationships[memory_id]:
            self.relationships[memory_id].append(related_id)
        
        logger.debug(f"Added relationship: {memory_id} -> {related_id}")
    
    async def get_related_memories(self, memory_id: str) -> List[str]:
        """获取相关记忆"""
        return self.relationships.get(memory_id, [])
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_metadata_entries": len(self.memory_metadata),
            "total_access_patterns": len(self.access_patterns),
            "total_importance_scores": len(self.importance_scores),
            "total_relationships": len(self.relationships)
        }

class RetrievalSystem:
    """检索系统"""
    
    def __init__(self):
        self.short_term_memory: Optional[ShortTermMemory] = None
        self.long_term_memory: Optional[LongTermMemory] = None
        self.working_memory: Optional[WorkingMemory] = None
        self.meta_memory: Optional[MetaMemory] = None
        self.search_algorithms = {
            "exact": self._exact_search,
            "fuzzy": self._fuzzy_search,
            "semantic": self._semantic_search,
            "contextual": self._contextual_search
        }
    
    def set_memory_components(self, short_term: ShortTermMemory, long_term: LongTermMemory, 
                            working: WorkingMemory, meta: MetaMemory):
        """设置记忆组件"""
        self.short_term_memory = short_term
        self.long_term_memory = long_term
        self.working_memory = working
        self.meta_memory = meta
    
    async def retrieve(self, query: str, context: Dict[str, Any], 
                      algorithm: str = "contextual", limit: int = 10) -> List[Memory]:
        """检索记忆"""
        try:
            search_func = self.search_algorithms.get(algorithm, self._contextual_search)
            results = await search_func(query, context, limit)
            
            # 按重要性排序
            results.sort(key=lambda m: m.importance_score, reverse=True)
            
            logger.info(f"Retrieved {len(results)} memories for query: {query}")
            return results
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []
    
    async def _exact_search(self, query: str, context: Dict[str, Any], limit: int) -> List[Memory]:
        """精确搜索"""
        results = []
        
        # 在短期记忆中搜索
        if self.short_term_memory:
            stm_results = await self.short_term_memory.search(query, limit)
            results.extend(stm_results)
        
        # 在长期记忆中搜索
        if self.long_term_memory:
            ltm_results = await self.long_term_memory.search(query, limit)
            results.extend(ltm_results)
        
        return results[:limit]
    
    async def _fuzzy_search(self, query: str, context: Dict[str, Any], limit: int) -> List[Memory]:
        """模糊搜索"""
        # 简单的模糊搜索实现
        results = []
        query_words = query.lower().split()
        
        # 在短期记忆中搜索
        if self.short_term_memory:
            for memory in self.short_term_memory.memories.values():
                content_words = memory.content.lower().split()
                if any(word in content_words for word in query_words):
                    results.append(memory)
        
        # 在长期记忆中搜索
        if self.long_term_memory:
            ltm_results = await self.long_term_memory.search(query, limit)
            results.extend(ltm_results)
        
        return results[:limit]
    
    async def _semantic_search(self, query: str, context: Dict[str, Any], limit: int) -> List[Memory]:
        """语义搜索"""
        # 简单的语义搜索实现（实际应用中会使用更复杂的语义模型）
        results = []
        
        # 扩展查询词
        expanded_query = await self._expand_query(query)
        
        # 在长期记忆中搜索
        if self.long_term_memory:
            ltm_results = await self.long_term_memory.search(expanded_query, limit)
            results.extend(ltm_results)
        
        return results[:limit]
    
    async def _contextual_search(self, query: str, context: Dict[str, Any], limit: int) -> List[Memory]:
        """上下文搜索"""
        results = []
        
        # 获取工作记忆中的上下文相关记忆
        if self.working_memory:
            context_memories = await self.working_memory.get_context_memories()
            results.extend(context_memories)
        
        # 基于上下文在长期记忆中搜索
        if self.long_term_memory:
            contextual_query = f"{query} {context.get('task', '')} {context.get('domain', '')}"
            ltm_results = await self.long_term_memory.search(contextual_query, limit)
            results.extend(ltm_results)
        
        return results[:limit]
    
    async def _expand_query(self, query: str) -> str:
        """扩展查询词"""
        # 简单的查询扩展（实际应用中会使用更复杂的扩展策略）
        synonyms = {
            "学习": ["学习", "教育", "培训", "知识"],
            "工作": ["工作", "任务", "项目", "职责"],
            "问题": ["问题", "困难", "挑战", "障碍"]
        }
        
        expanded_words = []
        for word in query.split():
            if word in synonyms:
                expanded_words.extend(synonyms[word])
            else:
                expanded_words.append(word)
        
        return " ".join(expanded_words)

class MemoryIndexer:
    """记忆索引器"""
    
    def __init__(self):
        self.content_index: Dict[str, List[str]] = {}  # 内容 -> 记忆ID列表
        self.tag_index: Dict[str, List[str]] = {}  # 标签 -> 记忆ID列表
        self.metadata_index: Dict[str, Dict[str, List[str]]] = {}  # 元数据键 -> 值 -> 记忆ID列表
    
    async def index(self, memory: Memory):
        """建立索引"""
        # 内容索引
        words = memory.content.lower().split()
        for word in words:
            if word not in self.content_index:
                self.content_index[word] = []
            if memory.id not in self.content_index[word]:
                self.content_index[word].append(memory.id)
        
        # 标签索引
        for tag in memory.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = []
            if memory.id not in self.tag_index[tag]:
                self.tag_index[tag].append(memory.id)
        
        # 元数据索引
        for key, value in memory.metadata.items():
            if key not in self.metadata_index:
                self.metadata_index[key] = {}
            if str(value) not in self.metadata_index[key]:
                self.metadata_index[key][str(value)] = []
            if memory.id not in self.metadata_index[key][str(value)]:
                self.metadata_index[key][str(value)].append(memory.id)
    
    async def search(self, query: str, limit: int = 10) -> List[str]:
        """搜索记忆ID"""
        query_words = query.lower().split()
        memory_scores: Dict[str, int] = {}
        
        # 计算每个记忆的匹配分数
        for word in query_words:
            if word in self.content_index:
                for memory_id in self.content_index[word]:
                    memory_scores[memory_id] = memory_scores.get(memory_id, 0) + 1
        
        # 按分数排序并返回前N个
        sorted_memories = sorted(memory_scores.items(), key=lambda x: x[1], reverse=True)
        return [memory_id for memory_id, score in sorted_memories[:limit]]
    
    async def reindex(self, memory_id: str):
        """重新建立索引"""
        # 从所有索引中移除该记忆
        for word_memories in self.content_index.values():
            if memory_id in word_memories:
                word_memories.remove(memory_id)
        
        for tag_memories in self.tag_index.values():
            if memory_id in tag_memories:
                tag_memories.remove(memory_id)
        
        for value_memories in self.metadata_index.values():
            for memories in value_memories.values():
                if memory_id in memories:
                    memories.remove(memory_id)
    
    def get_index_size(self) -> Dict[str, int]:
        """获取索引大小"""
        return {
            "content_index_size": len(self.content_index),
            "tag_index_size": len(self.tag_index),
            "metadata_index_size": len(self.metadata_index)
        }

class InMemoryStorage:
    """内存存储后端"""
    
    def __init__(self):
        self.storage: Dict[str, Dict[str, Any]] = {}
    
    async def store(self, memory: Memory):
        """存储记忆"""
        self.storage[memory.id] = memory.to_dict()
    
    async def retrieve(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """检索记忆"""
        return self.storage.get(memory_id)
    
    async def update(self, memory_id: str, updates: Dict[str, Any]):
        """更新记忆"""
        if memory_id in self.storage:
            self.storage[memory_id].update(updates)
    
    async def delete(self, memory_id: str):
        """删除记忆"""
        if memory_id in self.storage:
            del self.storage[memory_id]
    
    def count(self) -> int:
        """获取记忆数量"""
        return len(self.storage)

class MemorySystem:
    """记忆系统主类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.short_term_memory = ShortTermMemory(
            capacity=config.get("short_term_capacity", 1000),
            max_age=config.get("short_term_max_age", 3600)
        )
        self.long_term_memory = LongTermMemory()
        self.working_memory = WorkingMemory(
            capacity=config.get("working_capacity", 100)
        )
        self.meta_memory = MetaMemory()
        self.retrieval_system = RetrievalSystem()
        
        # 设置检索系统的记忆组件
        self.retrieval_system.set_memory_components(
            self.short_term_memory, self.long_term_memory,
            self.working_memory, self.meta_memory
        )
        
        self.running = False
    
    async def start(self):
        """启动记忆系统"""
        self.running = True
        await self.short_term_memory.start_cleanup()
        logger.info("Memory system started")
    
    async def stop(self):
        """停止记忆系统"""
        self.running = False
        await self.short_term_memory.stop_cleanup()
        logger.info("Memory system stopped")
    
    async def store_memory(self, memory: Memory) -> bool:
        """存储记忆"""
        try:
            # 根据记忆类型存储到相应的记忆系统
            if memory.memory_type == MemoryType.SHORT_TERM:
                success = await self.short_term_memory.store(memory)
            elif memory.memory_type == MemoryType.LONG_TERM:
                success = await self.long_term_memory.store(memory)
            elif memory.memory_type == MemoryType.WORKING:
                success = await self.working_memory.store(memory)
            else:
                # 默认存储到长期记忆
                success = await self.long_term_memory.store(memory)
            
            if success:
                # 更新元记忆
                await self.meta_memory.update(memory.id, memory.to_dict())
            
            return success
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return False
    
    async def retrieve_memory(self, memory_id: str) -> Optional[Memory]:
        """检索记忆"""
        # 按优先级搜索：工作记忆 -> 短期记忆 -> 长期记忆
        memory = await self.working_memory.retrieve(memory_id)
        if memory:
            return memory
        
        memory = await self.short_term_memory.retrieve(memory_id)
        if memory:
            return memory
        
        memory = await self.long_term_memory.retrieve(memory_id)
        if memory:
            return memory
        
        return None
    
    async def search_memories(self, query: str, context: Dict[str, Any] = None, 
                            algorithm: str = "contextual", limit: int = 10) -> List[Memory]:
        """搜索记忆"""
        if context is None:
            context = {}
        
        return await self.retrieval_system.retrieve(query, context, algorithm, limit)
    
    async def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """更新记忆"""
        try:
            # 更新元记忆
            await self.meta_memory.update(memory_id, updates)
            
            # 更新相应的记忆系统
            memory = await self.retrieve_memory(memory_id)
            if memory:
                for key, value in updates.items():
                    if hasattr(memory, key):
                        setattr(memory, key, value)
                
                # 重新存储更新后的记忆
                return await self.store_memory(memory)
            
            return False
        except Exception as e:
            logger.error(f"Failed to update memory: {e}")
            return False
    
    async def delete_memory(self, memory_id: str) -> bool:
        """删除记忆"""
        try:
            # 从所有记忆系统中删除
            if memory_id in self.short_term_memory.memories:
                del self.short_term_memory.memories[memory_id]
            
            if memory_id in self.working_memory.memories:
                del self.working_memory.memories[memory_id]
            
            await self.long_term_memory.storage_backend.delete(memory_id)
            await self.long_term_memory.indexer.reindex(memory_id)
            
            logger.info(f"Deleted memory: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory: {e}")
            return False
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return {
            "short_term_stats": self.short_term_memory.get_stats(),
            "long_term_stats": self.long_term_memory.get_stats(),
            "working_stats": self.working_memory.get_stats(),
            "meta_stats": self.meta_memory.get_stats(),
            "indexer_stats": self.long_term_memory.indexer.get_index_size()
        }

# 示例用法
async def main_demo():
    """记忆系统演示"""
    # 创建记忆系统配置
    config = {
        "short_term_capacity": 100,
        "short_term_max_age": 300,  # 5分钟
        "working_capacity": 50
    }
    
    # 创建记忆系统
    memory_system = MemorySystem(config)
    await memory_system.start()
    
    # 创建示例记忆
    memories = [
        Memory(
            content="学习Python编程语言",
            memory_type=MemoryType.SHORT_TERM,
            priority=MemoryPriority.HIGH,
            tags=["编程", "Python", "学习"],
            metadata={"source": "教程", "confidence": 0.9},
            importance_score=0.8
        ),
        Memory(
            content="完成项目报告",
            memory_type=MemoryType.WORKING,
            priority=MemoryPriority.CRITICAL,
            tags=["工作", "报告", "项目"],
            metadata={"deadline": "2025-09-25", "confidence": 1.0},
            importance_score=0.95
        ),
        Memory(
            content="人工智能的发展历史",
            memory_type=MemoryType.LONG_TERM,
            priority=MemoryPriority.MEDIUM,
            tags=["AI", "历史", "知识"],
            metadata={"source": "书籍", "confidence": 0.8},
            importance_score=0.7
        )
    ]
    
    # 存储记忆
    print("存储记忆...")
    for memory in memories:
        success = await memory_system.store_memory(memory)
        print(f"存储记忆 {memory.id}: {'成功' if success else '失败'}")
    
    # 搜索记忆
    print("\n搜索记忆...")
    search_results = await memory_system.search_memories("Python", limit=5)
    print(f"找到 {len(search_results)} 个相关记忆:")
    for memory in search_results:
        print(f"- {memory.content} (重要性: {memory.importance_score})")
    
    # 获取系统统计
    print("\n系统统计:")
    stats = memory_system.get_system_stats()
    print(f"短期记忆: {stats['short_term_stats']['total_memories']} 个")
    print(f"工作记忆: {stats['working_stats']['total_memories']} 个")
    print(f"长期记忆: {stats['long_term_stats']['total_memories']} 个")
    
    # 停止系统
    await memory_system.stop()
    print("\n记忆系统演示完成")

if __name__ == "__main__":
    asyncio.run(main_demo())
