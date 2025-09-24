# retrieval_system.py
"""
第4章 记忆与推理系统构建 - 检索系统
实现智能体的检索系统，包括语义检索、向量检索、混合检索等
"""

import asyncio
import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import hashlib
from collections import defaultdict, deque
import re
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RetrievalType(Enum):
    """检索类型枚举"""
    SEMANTIC = "语义检索"
    VECTOR = "向量检索"
    KEYWORD = "关键词检索"
    HYBRID = "混合检索"
    CONTEXTUAL = "上下文检索"
    TEMPORAL = "时序检索"
    SPATIAL = "空间检索"

class SimilarityMetric(Enum):
    """相似度度量枚举"""
    COSINE = "余弦相似度"
    EUCLIDEAN = "欧几里得距离"
    MANHATTAN = "曼哈顿距离"
    JACCARD = "杰卡德相似度"
    DOT_PRODUCT = "点积"
    PEARSON = "皮尔逊相关系数"

class IndexType(Enum):
    """索引类型枚举"""
    INVERTED = "倒排索引"
    VECTOR = "向量索引"
    TREE = "树形索引"
    HASH = "哈希索引"
    COMPRESSED = "压缩索引"

@dataclass
class Document:
    """文档数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    title: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[List[float]] = None
    keywords: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    relevance_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "content": self.content,
            "title": self.title,
            "metadata": self.metadata,
            "embeddings": self.embeddings,
            "keywords": self.keywords,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "relevance_score": self.relevance_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """从字典创建文档对象"""
        doc = cls()
        doc.id = data.get("id", str(uuid.uuid4()))
        doc.content = data.get("content", "")
        doc.title = data.get("title", "")
        doc.metadata = data.get("metadata", {})
        doc.embeddings = data.get("embeddings")
        doc.keywords = data.get("keywords", [])
        doc.timestamp = datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat()))
        doc.source = data.get("source", "")
        doc.relevance_score = data.get("relevance_score", 0.0)
        return doc

@dataclass
class Query:
    """查询数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    query_type: RetrievalType = RetrievalType.SEMANTIC
    filters: Dict[str, Any] = field(default_factory=dict)
    limit: int = 10
    similarity_threshold: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "text": self.text,
            "query_type": self.query_type.value,
            "filters": self.filters,
            "limit": self.limit,
            "similarity_threshold": self.similarity_threshold,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

@dataclass
class RetrievalResult:
    """检索结果数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query_id: str = ""
    document_id: str = ""
    document: Document = None
    similarity_score: float = 0.0
    relevance_score: float = 0.0
    rank: int = 0
    explanation: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "query_id": self.query_id,
            "document_id": self.document_id,
            "document": self.document.to_dict() if self.document else None,
            "similarity_score": self.similarity_score,
            "relevance_score": self.relevance_score,
            "rank": self.rank,
            "explanation": self.explanation,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

class TextProcessor:
    """文本处理器"""
    
    def __init__(self):
        self.stop_words = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这"}
    
    def tokenize(self, text: str) -> List[str]:
        """分词"""
        # 简单的分词实现
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [token for token in tokens if token not in self.stop_words]
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """提取关键词"""
        tokens = self.tokenize(text)
        
        # 简单的关键词提取（基于词频）
        word_freq = defaultdict(int)
        for token in tokens:
            word_freq[token] += 1
        
        # 按频率排序
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in keywords[:max_keywords]]
    
    def create_embeddings(self, text: str) -> List[float]:
        """创建文本嵌入（简化版本）"""
        # 这里使用简化的词袋模型作为嵌入
        tokens = self.tokenize(text)
        
        # 创建词汇表
        vocab = list(set(tokens))
        vocab_size = len(vocab)
        
        # 创建词袋向量
        embeddings = [0.0] * vocab_size
        for token in tokens:
            if token in vocab:
                idx = vocab.index(token)
                embeddings[idx] += 1.0
        
        # 归一化
        norm = math.sqrt(sum(x * x for x in embeddings))
        if norm > 0:
            embeddings = [x / norm for x in embeddings]
        
        return embeddings

class InvertedIndex:
    """倒排索引"""
    
    def __init__(self):
        self.index: Dict[str, List[str]] = defaultdict(list)
        self.document_freq: Dict[str, int] = defaultdict(int)
        self.total_documents = 0
    
    def add_document(self, doc_id: str, tokens: List[str]):
        """添加文档到索引"""
        self.total_documents += 1
        
        for token in set(tokens):  # 去重
            self.index[token].append(doc_id)
            self.document_freq[token] += 1
    
    def search(self, query_tokens: List[str]) -> List[Tuple[str, float]]:
        """搜索文档"""
        doc_scores = defaultdict(float)
        
        for token in query_tokens:
            if token in self.index:
                # 计算TF-IDF分数
                idf = math.log(self.total_documents / self.document_freq[token])
                
                for doc_id in self.index[token]:
                    # 简化的TF计算
                    tf = 1.0
                    doc_scores[doc_id] += tf * idf
        
        # 按分数排序
        return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    
    def remove_document(self, doc_id: str):
        """从索引中移除文档"""
        # 这里需要重新构建索引，简化实现
        pass

class VectorIndex:
    """向量索引"""
    
    def __init__(self, similarity_metric: SimilarityMetric = SimilarityMetric.COSINE):
        self.similarity_metric = similarity_metric
        self.documents: Dict[str, Document] = {}
        self.embeddings: Dict[str, List[float]] = {}
    
    def add_document(self, document: Document):
        """添加文档到向量索引"""
        self.documents[document.id] = document
        if document.embeddings:
            self.embeddings[document.id] = document.embeddings
    
    def search(self, query_embedding: List[float], limit: int = 10) -> List[Tuple[str, float]]:
        """向量搜索"""
        similarities = []
        
        for doc_id, doc_embedding in self.embeddings.items():
            similarity = self._calculate_similarity(query_embedding, doc_embedding)
            similarities.append((doc_id, similarity))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]
    
    def _calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算相似度"""
        if self.similarity_metric == SimilarityMetric.COSINE:
            return self._cosine_similarity(vec1, vec2)
        elif self.similarity_metric == SimilarityMetric.EUCLIDEAN:
            return self._euclidean_similarity(vec1, vec2)
        elif self.similarity_metric == SimilarityMetric.DOT_PRODUCT:
            return self._dot_product(vec1, vec2)
        else:
            return self._cosine_similarity(vec1, vec2)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """余弦相似度"""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(a * a for a in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _euclidean_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """欧几里得相似度"""
        if len(vec1) != len(vec2):
            return 0.0
        
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))
        return 1.0 / (1.0 + distance)
    
    def _dot_product(self, vec1: List[float], vec2: List[float]) -> float:
        """点积"""
        if len(vec1) != len(vec2):
            return 0.0
        
        return sum(a * b for a, b in zip(vec1, vec2))

class RetrievalSystem:
    """检索系统主类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.text_processor = TextProcessor()
        self.inverted_index = InvertedIndex()
        self.vector_index = VectorIndex()
        self.documents: Dict[str, Document] = {}
        self.queries: Dict[str, Query] = {}
        self.results: Dict[str, RetrievalResult] = {}
        self.running = False
    
    async def start(self):
        """启动检索系统"""
        self.running = True
        logger.info("Retrieval system started")
    
    async def stop(self):
        """停止检索系统"""
        self.running = False
        logger.info("Retrieval system stopped")
    
    async def add_document(self, document: Document) -> bool:
        """添加文档"""
        try:
            # 处理文档
            document.keywords = self.text_processor.extract_keywords(document.content)
            document.embeddings = self.text_processor.create_embeddings(document.content)
            
            # 添加到文档集合
            self.documents[document.id] = document
            
            # 添加到倒排索引
            tokens = self.text_processor.tokenize(document.content)
            self.inverted_index.add_document(document.id, tokens)
            
            # 添加到向量索引
            self.vector_index.add_document(document)
            
            logger.info(f"Added document: {document.title} ({document.id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return False
    
    async def remove_document(self, doc_id: str) -> bool:
        """移除文档"""
        try:
            if doc_id not in self.documents:
                logger.error(f"Document not found: {doc_id}")
                return False
            
            # 从文档集合中移除
            del self.documents[doc_id]
            
            # 从向量索引中移除
            if doc_id in self.vector_index.documents:
                del self.vector_index.documents[doc_id]
            if doc_id in self.vector_index.embeddings:
                del self.vector_index.embeddings[doc_id]
            
            logger.info(f"Removed document: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove document: {e}")
            return False
    
    async def search(self, query: Query) -> List[RetrievalResult]:
        """执行搜索"""
        try:
            self.queries[query.id] = query
            results = []
            
            if query.query_type == RetrievalType.KEYWORD:
                # 关键词搜索
                query_tokens = self.text_processor.tokenize(query.text)
                doc_scores = self.inverted_index.search(query_tokens)
                
                for rank, (doc_id, score) in enumerate(doc_scores[:query.limit]):
                    if doc_id in self.documents:
                        document = self.documents[doc_id]
                        result = RetrievalResult(
                            query_id=query.id,
                            document_id=doc_id,
                            document=document,
                            similarity_score=score,
                            relevance_score=score,
                            rank=rank + 1,
                            explanation=f"关键词匹配: {', '.join(query_tokens)}"
                        )
                        results.append(result)
            
            elif query.query_type == RetrievalType.VECTOR:
                # 向量搜索
                query_embedding = self.text_processor.create_embeddings(query.text)
                doc_scores = self.vector_index.search(query_embedding, query.limit)
                
                for rank, (doc_id, score) in enumerate(doc_scores):
                    if doc_id in self.documents and score >= query.similarity_threshold:
                        document = self.documents[doc_id]
                        result = RetrievalResult(
                            query_id=query.id,
                            document_id=doc_id,
                            document=document,
                            similarity_score=score,
                            relevance_score=score,
                            rank=rank + 1,
                            explanation=f"向量相似度: {score:.3f}"
                        )
                        results.append(result)
            
            elif query.query_type == RetrievalType.HYBRID:
                # 混合搜索
                keyword_results = await self._keyword_search(query)
                vector_results = await self._vector_search(query)
                
                # 合并结果
                combined_results = self._combine_results(keyword_results, vector_results, query.limit)
                results = combined_results
            
            else:
                # 默认语义搜索
                results = await self._semantic_search(query)
            
            # 保存结果
            for result in results:
                self.results[result.id] = result
            
            logger.info(f"Search completed for query: {query.text}, found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def _keyword_search(self, query: Query) -> List[RetrievalResult]:
        """关键词搜索"""
        query_tokens = self.text_processor.tokenize(query.text)
        doc_scores = self.inverted_index.search(query_tokens)
        
        results = []
        for rank, (doc_id, score) in enumerate(doc_scores[:query.limit]):
            if doc_id in self.documents:
                document = self.documents[doc_id]
                result = RetrievalResult(
                    query_id=query.id,
                    document_id=doc_id,
                    document=document,
                    similarity_score=score,
                    relevance_score=score,
                    rank=rank + 1,
                    explanation=f"关键词匹配: {', '.join(query_tokens)}"
                )
                results.append(result)
        
        return results
    
    async def _vector_search(self, query: Query) -> List[RetrievalResult]:
        """向量搜索"""
        query_embedding = self.text_processor.create_embeddings(query.text)
        doc_scores = self.vector_index.search(query_embedding, query.limit)
        
        results = []
        for rank, (doc_id, score) in enumerate(doc_scores):
            if doc_id in self.documents and score >= query.similarity_threshold:
                document = self.documents[doc_id]
                result = RetrievalResult(
                    query_id=query.id,
                    document_id=doc_id,
                    document=document,
                    similarity_score=score,
                    relevance_score=score,
                    rank=rank + 1,
                    explanation=f"向量相似度: {score:.3f}"
                )
                results.append(result)
        
        return results
    
    async def _semantic_search(self, query: Query) -> List[RetrievalResult]:
        """语义搜索（使用向量搜索）"""
        return await self._vector_search(query)
    
    def _combine_results(self, keyword_results: List[RetrievalResult], 
                        vector_results: List[RetrievalResult], 
                        limit: int) -> List[RetrievalResult]:
        """合并搜索结果"""
        # 创建文档ID到结果的映射
        doc_results = {}
        
        # 添加关键词结果
        for result in keyword_results:
            doc_id = result.document_id
            if doc_id not in doc_results:
                doc_results[doc_id] = result
            else:
                # 合并分数
                doc_results[doc_id].relevance_score = (doc_results[doc_id].relevance_score + result.relevance_score) / 2
        
        # 添加向量结果
        for result in vector_results:
            doc_id = result.document_id
            if doc_id not in doc_results:
                doc_results[doc_id] = result
            else:
                # 合并分数
                doc_results[doc_id].relevance_score = (doc_results[doc_id].relevance_score + result.relevance_score) / 2
        
        # 按分数排序
        combined_results = list(doc_results.values())
        combined_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # 重新分配排名
        for rank, result in enumerate(combined_results[:limit]):
            result.rank = rank + 1
        
        return combined_results[:limit]
    
    async def get_document(self, doc_id: str) -> Optional[Document]:
        """获取文档"""
        return self.documents.get(doc_id)
    
    async def get_query(self, query_id: str) -> Optional[Query]:
        """获取查询"""
        return self.queries.get(query_id)
    
    async def get_results(self, query_id: str) -> List[RetrievalResult]:
        """获取查询结果"""
        return [result for result in self.results.values() if result.query_id == query_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_documents": len(self.documents),
            "total_queries": len(self.queries),
            "total_results": len(self.results),
            "inverted_index_size": len(self.inverted_index.index),
            "vector_index_size": len(self.vector_index.embeddings),
            "average_document_length": sum(len(doc.content) for doc in self.documents.values()) / max(len(self.documents), 1),
            "average_keywords_per_document": sum(len(doc.keywords) for doc in self.documents.values()) / max(len(self.documents), 1)
        }

# 示例用法
async def main_demo():
    """检索系统演示"""
    # 创建检索系统配置
    config = {
        "max_documents": 10000,
        "max_queries": 1000
    }
    
    # 创建检索系统
    retrieval_system = RetrievalSystem(config)
    await retrieval_system.start()
    
    # 添加示例文档
    print("添加示例文档...")
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
        )
    ]
    
    for doc in documents:
        await retrieval_system.add_document(doc)
    
    # 关键词搜索
    print("\n关键词搜索:")
    keyword_query = Query(
        text="人工智能 机器学习",
        query_type=RetrievalType.KEYWORD,
        limit=3
    )
    
    keyword_results = await retrieval_system.search(keyword_query)
    print(f"找到 {len(keyword_results)} 个结果:")
    for result in keyword_results:
        print(f"- {result.document.title} (分数: {result.relevance_score:.3f})")
        print(f"  解释: {result.explanation}")
    
    # 向量搜索
    print("\n向量搜索:")
    vector_query = Query(
        text="如何让计算机学习",
        query_type=RetrievalType.VECTOR,
        limit=3
    )
    
    vector_results = await retrieval_system.search(vector_query)
    print(f"找到 {len(vector_results)} 个结果:")
    for result in vector_results:
        print(f"- {result.document.title} (相似度: {result.similarity_score:.3f})")
        print(f"  解释: {result.explanation}")
    
    # 混合搜索
    print("\n混合搜索:")
    hybrid_query = Query(
        text="神经网络 深度学习",
        query_type=RetrievalType.HYBRID,
        limit=3
    )
    
    hybrid_results = await retrieval_system.search(hybrid_query)
    print(f"找到 {len(hybrid_results)} 个结果:")
    for result in hybrid_results:
        print(f"- {result.document.title} (相关性: {result.relevance_score:.3f})")
        print(f"  解释: {result.explanation}")
    
    # 语义搜索
    print("\n语义搜索:")
    semantic_query = Query(
        text="让机器像人一样思考",
        query_type=RetrievalType.SEMANTIC,
        limit=3
    )
    
    semantic_results = await retrieval_system.search(semantic_query)
    print(f"找到 {len(semantic_results)} 个结果:")
    for result in semantic_results:
        print(f"- {result.document.title} (相似度: {result.similarity_score:.3f})")
        print(f"  解释: {result.explanation}")
    
    # 获取统计信息
    print("\n检索系统统计:")
    stats = retrieval_system.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 停止检索系统
    await retrieval_system.stop()
    print("\n检索系统演示完成")

if __name__ == "__main__":
    asyncio.run(main_demo())