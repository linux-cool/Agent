# customer_service_system.py
"""
第6章 企业级智能体应用 - 智能客服系统
实现基于AI的智能客服系统，包括对话管理、知识库集成、工单系统等
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
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MessageType(Enum):
    """消息类型枚举"""
    TEXT = "文本"
    IMAGE = "图片"
    FILE = "文件"
    VOICE = "语音"
    VIDEO = "视频"

class ConversationStatus(Enum):
    """对话状态枚举"""
    ACTIVE = "活跃"
    WAITING = "等待中"
    ESCALATED = "已升级"
    CLOSED = "已关闭"
    TIMEOUT = "超时"

class TicketStatus(Enum):
    """工单状态枚举"""
    OPEN = "打开"
    IN_PROGRESS = "处理中"
    PENDING = "待处理"
    RESOLVED = "已解决"
    CLOSED = "已关闭"

class EscalationLevel(Enum):
    """升级等级枚举"""
    LEVEL_1 = "一级"
    LEVEL_2 = "二级"
    LEVEL_3 = "三级"
    MANAGER = "经理级"

class SentimentType(Enum):
    """情感类型枚举"""
    POSITIVE = "积极"
    NEUTRAL = "中性"
    NEGATIVE = "消极"
    ANGRY = "愤怒"
    FRUSTRATED = "沮丧"

@dataclass
class Customer:
    """客户信息"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    email: str = ""
    phone: str = ""
    company: str = ""
    customer_type: str = "individual"  # individual, enterprise
    vip_level: int = 1
    language: str = "zh-CN"
    timezone: str = "Asia/Shanghai"
    preferences: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_contact: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "company": self.company,
            "customer_type": self.customer_type,
            "vip_level": self.vip_level,
            "language": self.language,
            "timezone": self.timezone,
            "preferences": self.preferences,
            "created_at": self.created_at.isoformat(),
            "last_contact": self.last_contact.isoformat() if self.last_contact else None
        }

@dataclass
class Message:
    """消息"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = ""
    sender_id: str = ""
    sender_type: str = "customer"  # customer, agent, system
    message_type: MessageType = MessageType.TEXT
    content: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "sender_id": self.sender_id,
            "sender_type": self.sender_type,
            "message_type": self.message_type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

@dataclass
class Conversation:
    """对话"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    customer_id: str = ""
    agent_id: Optional[str] = None
    status: ConversationStatus = ConversationStatus.ACTIVE
    channel: str = "web"  # web, mobile, phone, email
    subject: str = ""
    priority: int = 1
    tags: List[str] = field(default_factory=list)
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    closed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "customer_id": self.customer_id,
            "agent_id": self.agent_id,
            "status": self.status.value,
            "channel": self.channel,
            "subject": self.subject,
            "priority": self.priority,
            "tags": self.tags,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "closed_at": self.closed_at.isoformat() if self.closed_at else None
        }

@dataclass
class Ticket:
    """工单"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = ""
    customer_id: str = ""
    agent_id: Optional[str] = None
    status: TicketStatus = TicketStatus.OPEN
    priority: int = 1
    category: str = ""
    subcategory: str = ""
    subject: str = ""
    description: str = ""
    resolution: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    sla_deadline: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "customer_id": self.customer_id,
            "agent_id": self.agent_id,
            "status": self.status.value,
            "priority": self.priority,
            "category": self.category,
            "subcategory": self.subcategory,
            "subject": self.subject,
            "description": self.description,
            "resolution": self.resolution,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "sla_deadline": self.sla_deadline.isoformat() if self.sla_deadline else None
        }

@dataclass
class KnowledgeArticle:
    """知识库文章"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    content: str = ""
    category: str = ""
    tags: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    language: str = "zh-CN"
    author: str = ""
    version: int = 1
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    view_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "category": self.category,
            "tags": self.tags,
            "keywords": self.keywords,
            "language": self.language,
            "author": self.author,
            "version": self.version,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "view_count": self.view_count
        }

class SentimentAnalyzer:
    """情感分析器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sentiment_keywords = self._load_sentiment_keywords()
    
    def _load_sentiment_keywords(self) -> Dict[str, List[str]]:
        """加载情感关键词"""
        return {
            "positive": ["好", "棒", "满意", "感谢", "优秀", "完美", "喜欢", "推荐"],
            "negative": ["差", "坏", "不满意", "问题", "错误", "故障", "慢", "糟糕"],
            "angry": ["愤怒", "生气", "恼火", "愤怒", "气愤", "暴怒"],
            "frustrated": ["沮丧", "失望", "无奈", "绝望", "无助", "困惑"]
        }
    
    async def analyze_sentiment(self, text: str) -> Tuple[SentimentType, float]:
        """分析情感"""
        try:
            text_lower = text.lower()
            sentiment_scores = {}
            
            for sentiment, keywords in self.sentiment_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                sentiment_scores[sentiment] = score
            
            # 确定主要情感
            if sentiment_scores.get("angry", 0) > 0:
                sentiment_type = SentimentType.ANGRY
                confidence = min(sentiment_scores["angry"] / len(text.split()) * 10, 1.0)
            elif sentiment_scores.get("frustrated", 0) > 0:
                sentiment_type = SentimentType.FRUSTRATED
                confidence = min(sentiment_scores["frustrated"] / len(text.split()) * 10, 1.0)
            elif sentiment_scores.get("negative", 0) > sentiment_scores.get("positive", 0):
                sentiment_type = SentimentType.NEGATIVE
                confidence = min(sentiment_scores["negative"] / len(text.split()) * 10, 1.0)
            elif sentiment_scores.get("positive", 0) > 0:
                sentiment_type = SentimentType.POSITIVE
                confidence = min(sentiment_scores["positive"] / len(text.split()) * 10, 1.0)
            else:
                sentiment_type = SentimentType.NEUTRAL
                confidence = 0.5
            
            return sentiment_type, confidence
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return SentimentType.NEUTRAL, 0.5

class IntentClassifier:
    """意图分类器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.intent_patterns = self._load_intent_patterns()
    
    def _load_intent_patterns(self) -> Dict[str, List[str]]:
        """加载意图模式"""
        return {
            "greeting": ["你好", "您好", "hi", "hello", "早上好", "下午好"],
            "complaint": ["投诉", "抱怨", "不满", "问题", "故障", "错误"],
            "inquiry": ["询问", "咨询", "了解", "想知道", "请问"],
            "request": ["请求", "需要", "要求", "申请", "帮助"],
            "thanks": ["谢谢", "感谢", "多谢", "thank you"],
            "goodbye": ["再见", "拜拜", "bye", "goodbye", "结束"]
        }
    
    async def classify_intent(self, text: str) -> Tuple[str, float]:
        """分类意图"""
        try:
            text_lower = text.lower()
            intent_scores = {}
            
            for intent, patterns in self.intent_patterns.items():
                score = sum(1 for pattern in patterns if pattern in text_lower)
                intent_scores[intent] = score
            
            # 找到最高分的意图
            if intent_scores:
                best_intent = max(intent_scores.items(), key=lambda x: x[1])
                confidence = min(best_intent[1] / len(text.split()) * 5, 1.0)
                return best_intent[0], confidence
            else:
                return "unknown", 0.0
                
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return "unknown", 0.0

class KnowledgeBase:
    """知识库"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.articles: Dict[str, KnowledgeArticle] = {}
        self.search_index: Dict[str, List[str]] = defaultdict(list)
        self._initialize_sample_data()
    
    def _initialize_sample_data(self):
        """初始化示例数据"""
        sample_articles = [
            {
                "title": "如何重置密码",
                "content": "如果您忘记了密码，可以通过以下步骤重置：1. 点击登录页面的'忘记密码'链接 2. 输入您的邮箱地址 3. 查收重置邮件 4. 点击邮件中的链接设置新密码",
                "category": "账户管理",
                "tags": ["密码", "重置", "登录"],
                "keywords": ["密码", "重置", "忘记", "登录"]
            },
            {
                "title": "产品退换货政策",
                "content": "我们提供30天无理由退换货服务。商品必须保持原包装完整，未使用过。请在收到商品后30天内联系客服申请退换货。",
                "category": "售后服务",
                "tags": ["退货", "换货", "政策"],
                "keywords": ["退货", "换货", "政策", "30天"]
            },
            {
                "title": "如何联系客服",
                "content": "您可以通过以下方式联系我们：1. 在线客服：工作日9:00-18:00 2. 客服热线：400-123-4567 3. 邮箱：service@company.com 4. 微信客服：company_service",
                "category": "联系方式",
                "tags": ["客服", "联系", "电话"],
                "keywords": ["客服", "联系", "电话", "邮箱", "微信"]
            }
        ]
        
        for article_data in sample_articles:
            article = KnowledgeArticle(**article_data)
            self.articles[article.id] = article
            self._update_search_index(article)
    
    def _update_search_index(self, article: KnowledgeArticle):
        """更新搜索索引"""
        for keyword in article.keywords:
            self.search_index[keyword].append(article.id)
    
    async def search(self, query: str, limit: int = 5) -> List[KnowledgeArticle]:
        """搜索知识库"""
        try:
            query_lower = query.lower()
            query_words = query_lower.split()
            
            # 计算相关性分数
            article_scores = {}
            for article in self.articles.values():
                if not article.is_active:
                    continue
                
                score = 0
                # 标题匹配
                if any(word in article.title.lower() for word in query_words):
                    score += 3
                
                # 内容匹配
                content_matches = sum(1 for word in query_words if word in article.content.lower())
                score += content_matches
                
                # 关键词匹配
                keyword_matches = sum(1 for word in query_words if word in article.keywords)
                score += keyword_matches * 2
                
                if score > 0:
                    article_scores[article.id] = score
            
            # 按分数排序
            sorted_articles = sorted(article_scores.items(), key=lambda x: x[1], reverse=True)
            
            # 返回前N个结果
            results = []
            for article_id, score in sorted_articles[:limit]:
                article = self.articles[article_id]
                article.view_count += 1
                results.append(article)
            
            return results
            
        except Exception as e:
            logger.error(f"Knowledge base search failed: {e}")
            return []
    
    def add_article(self, article: KnowledgeArticle) -> bool:
        """添加文章"""
        try:
            self.articles[article.id] = article
            self._update_search_index(article)
            logger.info(f"Added knowledge article: {article.title}")
            return True
        except Exception as e:
            logger.error(f"Failed to add article: {e}")
            return False
    
    def get_article(self, article_id: str) -> Optional[KnowledgeArticle]:
        """获取文章"""
        return self.articles.get(article_id)

class ChatEngine:
    """对话引擎"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.knowledge_base = KnowledgeBase(config.get("knowledge_base", {}))
        self.sentiment_analyzer = SentimentAnalyzer(config.get("sentiment", {}))
        self.intent_classifier = IntentClassifier(config.get("intent", {}))
        self.response_templates = self._load_response_templates()
    
    def _load_response_templates(self) -> Dict[str, List[str]]:
        """加载回复模板"""
        return {
            "greeting": [
                "您好！我是智能客服助手，很高兴为您服务！",
                "欢迎！有什么可以帮助您的吗？",
                "您好！请问有什么问题需要我帮助解决？"
            ],
            "complaint": [
                "非常抱歉给您带来了不便，我会尽快帮您解决这个问题。",
                "我理解您的困扰，让我们一起来解决这个问题。",
                "感谢您的反馈，我们会认真处理您的问题。"
            ],
            "inquiry": [
                "好的，我来为您查询相关信息。",
                "让我为您查找这个问题的答案。",
                "我来帮您了解这个情况。"
            ],
            "thanks": [
                "不客气！很高兴能帮助到您！",
                "不用谢！如果还有其他问题，随时可以找我。",
                "很高兴为您服务！"
            ],
            "goodbye": [
                "再见！祝您生活愉快！",
                "感谢您的咨询，再见！",
                "期待下次为您服务，再见！"
            ],
            "no_answer": [
                "抱歉，我没有找到相关的答案。让我为您转接人工客服。",
                "这个问题比较复杂，我为您转接专业客服人员。",
                "我需要更多信息来帮助您，让我为您转接人工客服。"
            ]
        }
    
    async def generate_response(self, message: str, context: Dict[str, Any] = None) -> str:
        """生成回复"""
        try:
            # 分析情感
            sentiment, sentiment_confidence = await self.sentiment_analyzer.analyze_sentiment(message)
            
            # 分类意图
            intent, intent_confidence = await self.intent_classifier.classify_intent(message)
            
            # 搜索知识库
            knowledge_results = await self.knowledge_base.search(message, limit=3)
            
            # 生成回复
            if knowledge_results and intent_confidence > 0.3:
                # 基于知识库生成回复
                response = self._generate_knowledge_based_response(message, knowledge_results, intent)
            elif intent in self.response_templates:
                # 使用模板回复
                templates = self.response_templates[intent]
                response = np.random.choice(templates)
            else:
                # 默认回复
                response = self.response_templates["no_answer"][0]
            
            # 根据情感调整回复
            if sentiment in [SentimentType.ANGRY, SentimentType.FRUSTRATED]:
                response = f"非常抱歉给您带来了困扰。{response}"
            
            return response
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "抱歉，我暂时无法处理您的请求，请稍后再试。"
    
    def _generate_knowledge_based_response(self, message: str, knowledge_results: List[KnowledgeArticle], intent: str) -> str:
        """基于知识库生成回复"""
        if not knowledge_results:
            return self.response_templates["no_answer"][0]
        
        # 选择最相关的文章
        best_article = knowledge_results[0]
        
        # 根据意图调整回复
        if intent == "inquiry":
            response = f"根据您的问题，我找到了相关信息：\n\n{best_article.content}"
        elif intent == "complaint":
            response = f"我理解您的问题，这里是解决方案：\n\n{best_article.content}"
        else:
            response = f"关于您的问题：\n\n{best_article.content}"
        
        return response

class TicketManager:
    """工单管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tickets: Dict[str, Ticket] = {}
        self.sla_rules = self._load_sla_rules()
    
    def _load_sla_rules(self) -> Dict[int, timedelta]:
        """加载SLA规则"""
        return {
            1: timedelta(hours=1),    # 高优先级：1小时
            2: timedelta(hours=4),    # 中优先级：4小时
            3: timedelta(hours=24),   # 低优先级：24小时
            4: timedelta(hours=72),   # 普通优先级：72小时
            5: timedelta(days=7)      # 最低优先级：7天
        }
    
    async def create_ticket(self, conversation: Conversation, customer_query: str) -> Ticket:
        """创建工单"""
        try:
            # 分析查询内容确定分类
            category, subcategory = self._classify_query(customer_query)
            
            # 确定优先级
            priority = self._determine_priority(customer_query, conversation)
            
            # 计算SLA截止时间
            sla_deadline = datetime.now() + self.sla_rules.get(priority, timedelta(hours=24))
            
            ticket = Ticket(
                conversation_id=conversation.id,
                customer_id=conversation.customer_id,
                status=TicketStatus.OPEN,
                priority=priority,
                category=category,
                subcategory=subcategory,
                subject=conversation.subject or "客户咨询",
                description=customer_query,
                sla_deadline=sla_deadline
            )
            
            self.tickets[ticket.id] = ticket
            logger.info(f"Created ticket: {ticket.id}")
            return ticket
            
        except Exception as e:
            logger.error(f"Ticket creation failed: {e}")
            raise
    
    def _classify_query(self, query: str) -> Tuple[str, str]:
        """分类查询"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["密码", "登录", "账户"]):
            return "账户管理", "登录问题"
        elif any(word in query_lower for word in ["退货", "换货", "退款"]):
            return "售后服务", "退换货"
        elif any(word in query_lower for word in ["支付", "订单", "购买"]):
            return "订单管理", "支付问题"
        elif any(word in query_lower for word in ["产品", "功能", "使用"]):
            return "产品咨询", "功能问题"
        else:
            return "一般咨询", "其他"
    
    def _determine_priority(self, query: str, conversation: Conversation) -> int:
        """确定优先级"""
        query_lower = query.lower()
        
        # 紧急关键词
        if any(word in query_lower for word in ["紧急", "急", "立即", "马上"]):
            return 1
        
        # 投诉相关
        if any(word in query_lower for word in ["投诉", "投诉", "不满", "愤怒"]):
            return 2
        
        # VIP客户
        if hasattr(conversation, 'customer') and conversation.customer.vip_level >= 3:
            return 2
        
        # 一般问题
        return 3
    
    def get_ticket(self, ticket_id: str) -> Optional[Ticket]:
        """获取工单"""
        return self.tickets.get(ticket_id)
    
    def update_ticket_status(self, ticket_id: str, status: TicketStatus, resolution: str = None) -> bool:
        """更新工单状态"""
        try:
            if ticket_id in self.tickets:
                ticket = self.tickets[ticket_id]
                ticket.status = status
                ticket.updated_at = datetime.now()
                
                if status == TicketStatus.RESOLVED:
                    ticket.resolved_at = datetime.now()
                    ticket.resolution = resolution
                
                logger.info(f"Updated ticket {ticket_id} status to {status.value}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Ticket status update failed: {e}")
            return False

class EscalationManager:
    """升级管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.escalation_rules = self._load_escalation_rules()
        self.agents: Dict[str, Dict[str, Any]] = {}
    
    def _load_escalation_rules(self) -> Dict[str, Any]:
        """加载升级规则"""
        return {
            "confidence_threshold": 0.3,
            "sentiment_threshold": "negative",
            "timeout_threshold": timedelta(minutes=10),
            "retry_limit": 3
        }
    
    async def escalate(self, conversation: Conversation, confidence: float) -> bool:
        """升级对话"""
        try:
            # 检查是否需要升级
            if confidence < self.escalation_rules["confidence_threshold"]:
                # 分配人工客服
                agent_id = await self._assign_agent(conversation)
                if agent_id:
                    conversation.agent_id = agent_id
                    conversation.status = ConversationStatus.ESCALATED
                    logger.info(f"Escalated conversation {conversation.id} to agent {agent_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Escalation failed: {e}")
            return False
    
    async def _assign_agent(self, conversation: Conversation) -> Optional[str]:
        """分配客服"""
        try:
            # 简化的客服分配逻辑
            available_agents = [aid for aid, agent in self.agents.items() 
                              if agent.get("status") == "available"]
            
            if available_agents:
                # 选择负载最小的客服
                agent_loads = {aid: agent.get("active_conversations", 0) 
                             for aid, agent in self.agents.items() 
                             if agent.get("status") == "available"}
                
                selected_agent = min(agent_loads.keys(), key=lambda k: agent_loads[k])
                return selected_agent
            
            return None
            
        except Exception as e:
            logger.error(f"Agent assignment failed: {e}")
            return None
    
    def register_agent(self, agent_id: str, agent_info: Dict[str, Any]):
        """注册客服"""
        self.agents[agent_id] = agent_info
        logger.info(f"Registered agent: {agent_id}")

class AnalyticsEngine:
    """分析引擎"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.interactions: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {}
    
    async def record_interaction(self, conversation: Conversation, response: str, confidence: float):
        """记录交互"""
        try:
            interaction = {
                "conversation_id": conversation.id,
                "customer_id": conversation.customer_id,
                "timestamp": datetime.now(),
                "response": response,
                "confidence": confidence,
                "channel": conversation.channel,
                "duration": (datetime.now() - conversation.created_at).total_seconds()
            }
            
            self.interactions.append(interaction)
            
            # 更新指标
            await self._update_metrics(interaction)
            
        except Exception as e:
            logger.error(f"Interaction recording failed: {e}")
    
    async def _update_metrics(self, interaction: Dict[str, Any]):
        """更新指标"""
        try:
            # 更新基本指标
            if "total_interactions" not in self.metrics:
                self.metrics["total_interactions"] = 0
            self.metrics["total_interactions"] += 1
            
            # 更新平均置信度
            if "avg_confidence" not in self.metrics:
                self.metrics["avg_confidence"] = 0
            self.metrics["avg_confidence"] = (
                (self.metrics["avg_confidence"] * (self.metrics["total_interactions"] - 1) + 
                 interaction["confidence"]) / self.metrics["total_interactions"]
            )
            
            # 更新渠道分布
            if "channel_distribution" not in self.metrics:
                self.metrics["channel_distribution"] = {}
            channel = interaction["channel"]
            self.metrics["channel_distribution"][channel] = \
                self.metrics["channel_distribution"].get(channel, 0) + 1
            
        except Exception as e:
            logger.error(f"Metrics update failed: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取指标"""
        return self.metrics.copy()

class CustomerServiceSystem:
    """智能客服系统"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.chat_engine = ChatEngine(config.get("chat", {}))
        self.ticket_manager = TicketManager(config.get("tickets", {}))
        self.escalation_manager = EscalationManager(config.get("escalation", {}))
        self.analytics = AnalyticsEngine(config.get("analytics", {}))
        self.conversations: Dict[str, Conversation] = {}
        self.customers: Dict[str, Customer] = {}
    
    async def handle_customer_query(self, customer_id: str, message: str, 
                                  channel: str = "web") -> Dict[str, Any]:
        """处理客户查询"""
        try:
            # 获取或创建客户
            customer = self.customers.get(customer_id)
            if not customer:
                customer = Customer(id=customer_id, name=f"客户{customer_id}")
                self.customers[customer_id] = customer
            
            # 获取或创建对话
            conversation = await self._get_or_create_conversation(customer_id, channel)
            
            # 添加消息
            msg = Message(
                conversation_id=conversation.id,
                sender_id=customer_id,
                sender_type="customer",
                content=message
            )
            conversation.messages.append(msg)
            conversation.updated_at = datetime.now()
            
            # 生成回复
            response_content = await self.chat_engine.generate_response(message)
            confidence = self._calculate_confidence(response_content)
            
            # 检查是否需要升级
            if confidence < self.config.get("escalation_threshold", 0.3):
                await self.escalation_manager.escalate(conversation, confidence)
                if conversation.status == ConversationStatus.ESCALATED:
                    response_content = "您的问题已转给人工客服处理，请稍等。"
            
            # 添加系统回复
            response_msg = Message(
                conversation_id=conversation.id,
                sender_id="system",
                sender_type="system",
                content=response_content
            )
            conversation.messages.append(response_msg)
            
            # 记录交互
            await self.analytics.record_interaction(conversation, response_content, confidence)
            
            return {
                "conversation_id": conversation.id,
                "response": response_content,
                "confidence": confidence,
                "status": conversation.status.value,
                "agent_id": conversation.agent_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Customer query handling failed: {e}")
            return {
                "error": str(e),
                "response": "抱歉，系统暂时无法处理您的请求，请稍后再试。"
            }
    
    async def _get_or_create_conversation(self, customer_id: str, channel: str) -> Conversation:
        """获取或创建对话"""
        # 查找活跃对话
        for conversation in self.conversations.values():
            if (conversation.customer_id == customer_id and 
                conversation.status == ConversationStatus.ACTIVE):
                return conversation
        
        # 创建新对话
        conversation = Conversation(
            customer_id=customer_id,
            channel=channel,
            status=ConversationStatus.ACTIVE
        )
        self.conversations[conversation.id] = conversation
        return conversation
    
    def _calculate_confidence(self, response: str) -> float:
        """计算置信度"""
        # 简化的置信度计算
        if "抱歉" in response or "转接" in response:
            return 0.2
        elif len(response) > 50:
            return 0.8
        else:
            return 0.6
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """获取对话"""
        return self.conversations.get(conversation_id)
    
    def get_customer(self, customer_id: str) -> Optional[Customer]:
        """获取客户"""
        return self.customers.get(customer_id)
    
    def get_analytics(self) -> Dict[str, Any]:
        """获取分析数据"""
        return self.analytics.get_metrics()

# 示例用法
async def main_demo():
    """智能客服系统演示"""
    config = {
        "chat": {
            "knowledge_base": {},
            "sentiment": {},
            "intent": {}
        },
        "tickets": {},
        "escalation": {
            "confidence_threshold": 0.3
        },
        "analytics": {}
    }
    
    # 创建客服系统
    cs_system = CustomerServiceSystem(config)
    
    print("🤖 智能客服系统演示")
    print("=" * 50)
    
    # 模拟客户对话
    print("\n1. 客户咨询 - 密码重置问题...")
    response1 = await cs_system.handle_customer_query(
        "customer_001", 
        "我忘记了密码，怎么重置？", 
        "web"
    )
    print(f"✓ 系统回复: {response1['response']}")
    print(f"  置信度: {response1['confidence']:.2f}")
    print(f"  状态: {response1['status']}")
    
    print("\n2. 客户咨询 - 退货政策...")
    response2 = await cs_system.handle_customer_query(
        "customer_002", 
        "你们的退货政策是什么？", 
        "mobile"
    )
    print(f"✓ 系统回复: {response2['response']}")
    print(f"  置信度: {response2['confidence']:.2f}")
    print(f"  状态: {response2['status']}")
    
    print("\n3. 客户投诉 - 需要升级...")
    response3 = await cs_system.handle_customer_query(
        "customer_003", 
        "你们的产品质量太差了，我要投诉！", 
        "web"
    )
    print(f"✓ 系统回复: {response3['response']}")
    print(f"  置信度: {response3['confidence']:.2f}")
    print(f"  状态: {response3['status']}")
    
    print("\n4. 客户感谢...")
    response4 = await cs_system.handle_customer_query(
        "customer_001", 
        "谢谢你的帮助！", 
        "web"
    )
    print(f"✓ 系统回复: {response4['response']}")
    print(f"  置信度: {response4['confidence']:.2f}")
    print(f"  状态: {response4['status']}")
    
    # 显示分析数据
    print("\n5. 系统分析数据:")
    analytics = cs_system.get_analytics()
    for key, value in analytics.items():
        print(f"  {key}: {value}")
    
    print("\n🎉 智能客服系统演示完成")

if __name__ == "__main__":
    asyncio.run(main_demo())
