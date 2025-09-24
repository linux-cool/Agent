# communication_protocols.py
"""
多智能体通信协议实现
提供多种通信协议和消息传递机制
"""

import asyncio
import logging
import json
import time
import hashlib
import hmac
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import socket
import ssl

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageType(Enum):
    """消息类型枚举"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    HEARTBEAT = "heartbeat"
    TASK_ASSIGNMENT = "task_assignment"
    TASK_RESULT = "task_result"
    COORDINATION = "coordination"
    BROADCAST = "broadcast"
    ERROR = "error"

class ProtocolType(Enum):
    """协议类型枚举"""
    HTTP = "http"
    WEBSOCKET = "websocket"
    MQTT = "mqtt"
    REDIS = "redis"
    RABBITMQ = "rabbitmq"
    GRPC = "grpc"
    ZEROMQ = "zeromq"

class MessagePriority(Enum):
    """消息优先级枚举"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Message:
    """消息数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    receiver: str = ""
    message_type: MessageType = MessageType.REQUEST
    content: Any = None
    timestamp: datetime = field(default_factory=datetime.now)
    priority: MessagePriority = MessagePriority.NORMAL
    reply_to: Optional[str] = None
    correlation_id: Optional[str] = None
    ttl: Optional[timedelta] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    signature: Optional[str] = None

@dataclass
class ProtocolConfig:
    """协议配置"""
    protocol_type: ProtocolType
    host: str = "localhost"
    port: int = 8080
    timeout: int = 30
    retry_count: int = 3
    encryption_key: Optional[str] = None
    compression: bool = False
    authentication: bool = False
    ssl_enabled: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    max_message_size: int = 1024 * 1024  # 1MB
    heartbeat_interval: int = 30
    connection_pool_size: int = 10

class MessageSerializer:
    """消息序列化器"""
    
    @staticmethod
    def serialize(message: Message) -> str:
        """序列化消息"""
        data = {
            "id": message.id,
            "sender": message.sender,
            "receiver": message.receiver,
            "message_type": message.message_type.value,
            "content": message.content,
            "timestamp": message.timestamp.isoformat(),
            "priority": message.priority.value,
            "reply_to": message.reply_to,
            "correlation_id": message.correlation_id,
            "ttl": message.ttl.total_seconds() if message.ttl else None,
            "metadata": message.metadata,
            "signature": message.signature
        }
        return json.dumps(data, default=str)
    
    @staticmethod
    def deserialize(data: str) -> Message:
        """反序列化消息"""
        obj = json.loads(data)
        return Message(
            id=obj["id"],
            sender=obj["sender"],
            receiver=obj["receiver"],
            message_type=MessageType(obj["message_type"]),
            content=obj["content"],
            timestamp=datetime.fromisoformat(obj["timestamp"]),
            priority=MessagePriority(obj["priority"]),
            reply_to=obj.get("reply_to"),
            correlation_id=obj.get("correlation_id"),
            ttl=timedelta(seconds=obj["ttl"]) if obj.get("ttl") else None,
            metadata=obj.get("metadata", {}),
            signature=obj.get("signature")
        )

class MessageEncryption:
    """消息加密"""
    
    def __init__(self, key: str):
        self.key = key.encode()
    
    def encrypt(self, data: str) -> str:
        """加密数据"""
        # 简单的HMAC加密示例
        signature = hmac.new(self.key, data.encode(), hashlib.sha256).hexdigest()
        return f"{data}:{signature}"
    
    def decrypt(self, encrypted_data: str) -> str:
        """解密数据"""
        if ":" not in encrypted_data:
            raise ValueError("Invalid encrypted data format")
        
        data, signature = encrypted_data.rsplit(":", 1)
        expected_signature = hmac.new(self.key, data.encode(), hashlib.sha256).hexdigest()
        
        if not hmac.compare_digest(signature, expected_signature):
            raise ValueError("Invalid signature")
        
        return data

class CommunicationProtocol(ABC):
    """通信协议抽象基类"""
    
    def __init__(self, config: ProtocolConfig):
        self.config = config
        self.serializer = MessageSerializer()
        self.encryption = MessageEncryption(config.encryption_key) if config.encryption_key else None
        self.message_handlers: Dict[MessageType, List[Callable]] = {}
        self.running = False
        self.connections: Dict[str, Any] = {}
    
    @abstractmethod
    async def start(self):
        """启动协议"""
        pass
    
    @abstractmethod
    async def stop(self):
        """停止协议"""
        pass
    
    @abstractmethod
    async def send_message(self, message: Message) -> bool:
        """发送消息"""
        pass
    
    @abstractmethod
    async def receive_message(self, agent_id: str) -> List[Message]:
        """接收消息"""
        pass
    
    @abstractmethod
    async def broadcast_message(self, sender: str, message_type: MessageType, content: Any) -> bool:
        """广播消息"""
        pass
    
    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """注册消息处理器"""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)
    
    def unregister_message_handler(self, message_type: MessageType, handler: Callable):
        """注销消息处理器"""
        if message_type in self.message_handlers:
            self.message_handlers[message_type].remove(handler)
    
    async def handle_message(self, message: Message):
        """处理消息"""
        if message.message_type in self.message_handlers:
            for handler in self.message_handlers[message.message_type]:
                try:
                    await handler(message)
                except Exception as e:
                    logger.error(f"Error handling message {message.id}: {e}")
    
    def _serialize_message(self, message: Message) -> str:
        """序列化消息"""
        data = self.serializer.serialize(message)
        if self.encryption:
            data = self.encryption.encrypt(data)
        return data
    
    def _deserialize_message(self, data: str) -> Message:
        """反序列化消息"""
        if self.encryption:
            data = self.encryption.decrypt(data)
        return self.serializer.deserialize(data)

class HTTPProtocol(CommunicationProtocol):
    """HTTP协议实现"""
    
    def __init__(self, config: ProtocolConfig):
        super().__init__(config)
        self.server = None
        self.client_session = None
        self.message_queues: Dict[str, List[Message]] = {}
        self.lock = asyncio.Lock()
    
    async def start(self):
        """启动HTTP协议"""
        self.running = True
        logger.info("HTTP protocol started")
    
    async def stop(self):
        """停止HTTP协议"""
        self.running = False
        if self.server:
            self.server.close()
        logger.info("HTTP protocol stopped")
    
    async def send_message(self, message: Message) -> bool:
        """发送HTTP消息"""
        try:
            url = f"http://{self.config.host}:{self.config.port}/agents/{message.receiver}/messages"
            data = self._serialize_message(message)
            
            # 模拟HTTP请求
            await asyncio.sleep(0.01)  # 模拟网络延迟
            
            async with self.lock:
                if message.receiver not in self.message_queues:
                    self.message_queues[message.receiver] = []
                self.message_queues[message.receiver].append(message)
            
            logger.debug(f"HTTP message sent from {message.sender} to {message.receiver}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send HTTP message: {e}")
            return False
    
    async def receive_message(self, agent_id: str) -> List[Message]:
        """接收HTTP消息"""
        async with self.lock:
            messages = self.message_queues.get(agent_id, [])
            self.message_queues[agent_id] = []
            return messages
    
    async def broadcast_message(self, sender: str, message_type: MessageType, content: Any) -> bool:
        """广播HTTP消息"""
        message = Message(
            sender=sender,
            receiver="broadcast",
            message_type=message_type,
            content=content
        )
        
        # 发送给所有注册的智能体
        for agent_id in self.message_queues.keys():
            if agent_id != sender:
                message.receiver = agent_id
                await self.send_message(message)
        
        return True

class WebSocketProtocol(CommunicationProtocol):
    """WebSocket协议实现"""
    
    def __init__(self, config: ProtocolConfig):
        super().__init__(config)
        self.connections: Dict[str, Any] = {}
        self.message_queues: Dict[str, List[Message]] = {}
        self.lock = asyncio.Lock()
    
    async def start(self):
        """启动WebSocket协议"""
        self.running = True
        logger.info("WebSocket protocol started")
    
    async def stop(self):
        """停止WebSocket协议"""
        self.running = False
        for connection in self.connections.values():
            await connection.close()
        logger.info("WebSocket protocol stopped")
    
    async def send_message(self, message: Message) -> bool:
        """发送WebSocket消息"""
        try:
            if message.receiver in self.connections:
                connection = self.connections[message.receiver]
                data = self._serialize_message(message)
                await connection.send(data)
                logger.debug(f"WebSocket message sent from {message.sender} to {message.receiver}")
                return True
            else:
                # 如果连接不存在，存储到队列
                async with self.lock:
                    if message.receiver not in self.message_queues:
                        self.message_queues[message.receiver] = []
                    self.message_queues[message.receiver].append(message)
                return True
                
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
            return False
    
    async def receive_message(self, agent_id: str) -> List[Message]:
        """接收WebSocket消息"""
        async with self.lock:
            messages = self.message_queues.get(agent_id, [])
            self.message_queues[agent_id] = []
            return messages
    
    async def broadcast_message(self, sender: str, message_type: MessageType, content: Any) -> bool:
        """广播WebSocket消息"""
        message = Message(
            sender=sender,
            receiver="broadcast",
            message_type=message_type,
            content=content
        )
        
        # 发送给所有连接的智能体
        for agent_id, connection in self.connections.items():
            if agent_id != sender:
                try:
                    data = self._serialize_message(message)
                    await connection.send(data)
                except Exception as e:
                    logger.error(f"Failed to broadcast to {agent_id}: {e}")
        
        return True
    
    async def register_connection(self, agent_id: str, connection: Any):
        """注册WebSocket连接"""
        self.connections[agent_id] = connection
        
        # 发送队列中的消息
        async with self.lock:
            if agent_id in self.message_queues:
                for message in self.message_queues[agent_id]:
                    await self.send_message(message)
                self.message_queues[agent_id] = []

class MQTTProtocol(CommunicationProtocol):
    """MQTT协议实现"""
    
    def __init__(self, config: ProtocolConfig):
        super().__init__(config)
        self.client = None
        self.subscriptions: Dict[str, Set[str]] = {}
        self.message_queues: Dict[str, List[Message]] = {}
        self.lock = asyncio.Lock()
    
    async def start(self):
        """启动MQTT协议"""
        self.running = True
        logger.info("MQTT protocol started")
    
    async def stop(self):
        """停止MQTT协议"""
        self.running = False
        if self.client:
            await self.client.disconnect()
        logger.info("MQTT protocol stopped")
    
    async def send_message(self, message: Message) -> bool:
        """发送MQTT消息"""
        try:
            topic = f"agents/{message.receiver}/messages"
            data = self._serialize_message(message)
            
            # 模拟MQTT发布
            await asyncio.sleep(0.01)  # 模拟网络延迟
            
            # 存储到队列
            async with self.lock:
                if message.receiver not in self.message_queues:
                    self.message_queues[message.receiver] = []
                self.message_queues[message.receiver].append(message)
            
            logger.debug(f"MQTT message sent from {message.sender} to {message.receiver}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send MQTT message: {e}")
            return False
    
    async def receive_message(self, agent_id: str) -> List[Message]:
        """接收MQTT消息"""
        async with self.lock:
            messages = self.message_queues.get(agent_id, [])
            self.message_queues[agent_id] = []
            return messages
    
    async def broadcast_message(self, sender: str, message_type: MessageType, content: Any) -> bool:
        """广播MQTT消息"""
        topic = "agents/broadcast"
        message = Message(
            sender=sender,
            receiver="broadcast",
            message_type=message_type,
            content=content
        )
        
        data = self._serialize_message(message)
        
        # 模拟MQTT广播
        await asyncio.sleep(0.01)
        
        # 发送给所有订阅的智能体
        for agent_id in self.subscriptions.get("agents/broadcast", set()):
            if agent_id != sender:
                async with self.lock:
                    if agent_id not in self.message_queues:
                        self.message_queues[agent_id] = []
                    self.message_queues[agent_id].append(message)
        
        return True
    
    async def subscribe(self, agent_id: str, topic: str):
        """订阅MQTT主题"""
        if topic not in self.subscriptions:
            self.subscriptions[topic] = set()
        self.subscriptions[topic].add(agent_id)
        logger.info(f"Agent {agent_id} subscribed to topic {topic}")
    
    async def unsubscribe(self, agent_id: str, topic: str):
        """取消订阅MQTT主题"""
        if topic in self.subscriptions:
            self.subscriptions[topic].discard(agent_id)
        logger.info(f"Agent {agent_id} unsubscribed from topic {topic}")

class RedisProtocol(CommunicationProtocol):
    """Redis协议实现"""
    
    def __init__(self, config: ProtocolConfig):
        super().__init__(config)
        self.redis_client = None
        self.message_queues: Dict[str, List[Message]] = {}
        self.lock = asyncio.Lock()
    
    async def start(self):
        """启动Redis协议"""
        self.running = True
        logger.info("Redis protocol started")
    
    async def stop(self):
        """停止Redis协议"""
        self.running = False
        if self.redis_client:
            await self.redis_client.close()
        logger.info("Redis protocol stopped")
    
    async def send_message(self, message: Message) -> bool:
        """发送Redis消息"""
        try:
            key = f"agent:{message.receiver}:messages"
            data = self._serialize_message(message)
            
            # 模拟Redis操作
            await asyncio.sleep(0.01)  # 模拟网络延迟
            
            async with self.lock:
                if message.receiver not in self.message_queues:
                    self.message_queues[message.receiver] = []
                self.message_queues[message.receiver].append(message)
            
            logger.debug(f"Redis message sent from {message.sender} to {message.receiver}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Redis message: {e}")
            return False
    
    async def receive_message(self, agent_id: str) -> List[Message]:
        """接收Redis消息"""
        async with self.lock:
            messages = self.message_queues.get(agent_id, [])
            self.message_queues[agent_id] = []
            return messages
    
    async def broadcast_message(self, sender: str, message_type: MessageType, content: Any) -> bool:
        """广播Redis消息"""
        message = Message(
            sender=sender,
            receiver="broadcast",
            message_type=message_type,
            content=content
        )
        
        # 发布到Redis频道
        channel = "agents:broadcast"
        data = self._serialize_message(message)
        
        # 模拟Redis发布
        await asyncio.sleep(0.01)
        
        # 发送给所有订阅的智能体
        async with self.lock:
            for agent_id in self.message_queues.keys():
                if agent_id != sender:
                    if agent_id not in self.message_queues:
                        self.message_queues[agent_id] = []
                    self.message_queues[agent_id].append(message)
        
        return True

class CommunicationManager:
    """通信管理器"""
    
    def __init__(self):
        self.protocols: Dict[ProtocolType, CommunicationProtocol] = {}
        self.active_protocol: Optional[CommunicationProtocol] = None
        self.message_routing: Dict[str, ProtocolType] = {}
        self.message_history: List[Message] = []
        self.max_history_size = 1000
    
    def register_protocol(self, protocol_type: ProtocolType, protocol: CommunicationProtocol):
        """注册通信协议"""
        self.protocols[protocol_type] = protocol
        logger.info(f"Protocol {protocol_type.value} registered")
    
    def set_active_protocol(self, protocol_type: ProtocolType):
        """设置活动协议"""
        if protocol_type in self.protocols:
            self.active_protocol = self.protocols[protocol_type]
            logger.info(f"Active protocol set to {protocol_type.value}")
        else:
            raise ValueError(f"Protocol {protocol_type.value} not registered")
    
    def set_message_routing(self, agent_id: str, protocol_type: ProtocolType):
        """设置消息路由"""
        self.message_routing[agent_id] = protocol_type
        logger.info(f"Message routing for {agent_id} set to {protocol_type.value}")
    
    async def send_message(self, message: Message) -> bool:
        """发送消息"""
        # 确定使用的协议
        protocol = self._get_protocol_for_message(message)
        
        if not protocol:
            logger.error(f"No protocol available for message {message.id}")
            return False
        
        # 发送消息
        success = await protocol.send_message(message)
        
        if success:
            # 记录消息历史
            self._add_to_history(message)
        
        return success
    
    async def receive_message(self, agent_id: str) -> List[Message]:
        """接收消息"""
        protocol = self._get_protocol_for_agent(agent_id)
        
        if not protocol:
            logger.error(f"No protocol available for agent {agent_id}")
            return []
        
        return await protocol.receive_message(agent_id)
    
    async def broadcast_message(self, sender: str, message_type: MessageType, content: Any) -> bool:
        """广播消息"""
        if not self.active_protocol:
            logger.error("No active protocol for broadcast")
            return False
        
        return await self.active_protocol.broadcast_message(sender, message_type, content)
    
    def _get_protocol_for_message(self, message: Message) -> Optional[CommunicationProtocol]:
        """获取消息的协议"""
        # 优先使用路由配置
        if message.receiver in self.message_routing:
            protocol_type = self.message_routing[message.receiver]
            return self.protocols.get(protocol_type)
        
        # 使用活动协议
        return self.active_protocol
    
    def _get_protocol_for_agent(self, agent_id: str) -> Optional[CommunicationProtocol]:
        """获取智能体的协议"""
        if agent_id in self.message_routing:
            protocol_type = self.message_routing[agent_id]
            return self.protocols.get(protocol_type)
        
        return self.active_protocol
    
    def _add_to_history(self, message: Message):
        """添加到消息历史"""
        self.message_history.append(message)
        
        # 保持历史记录大小
        if len(self.message_history) > self.max_history_size:
            self.message_history = self.message_history[-self.max_history_size:]
    
    def get_message_history(self, limit: int = 100) -> List[Message]:
        """获取消息历史"""
        return self.message_history[-limit:]
    
    async def start_all_protocols(self):
        """启动所有协议"""
        for protocol in self.protocols.values():
            await protocol.start()
    
    async def stop_all_protocols(self):
        """停止所有协议"""
        for protocol in self.protocols.values():
            await protocol.stop()

# 使用示例
async def main():
    """主函数示例"""
    # 创建协议配置
    http_config = ProtocolConfig(
        protocol_type=ProtocolType.HTTP,
        host="localhost",
        port=8080,
        encryption_key="test_key"
    )
    
    websocket_config = ProtocolConfig(
        protocol_type=ProtocolType.WEBSOCKET,
        host="localhost",
        port=8081
    )
    
    # 创建通信协议
    http_protocol = HTTPProtocol(http_config)
    websocket_protocol = WebSocketProtocol(websocket_config)
    
    # 创建通信管理器
    comm_manager = CommunicationManager()
    comm_manager.register_protocol(ProtocolType.HTTP, http_protocol)
    comm_manager.register_protocol(ProtocolType.WEBSOCKET, websocket_protocol)
    comm_manager.set_active_protocol(ProtocolType.HTTP)
    
    # 启动协议
    await comm_manager.start_all_protocols()
    
    # 创建消息
    message = Message(
        sender="agent_1",
        receiver="agent_2",
        message_type=MessageType.REQUEST,
        content="Hello from agent 1",
        priority=MessagePriority.HIGH
    )
    
    # 发送消息
    success = await comm_manager.send_message(message)
    print(f"Message sent: {success}")
    
    # 接收消息
    messages = await comm_manager.receive_message("agent_2")
    print(f"Received {len(messages)} messages")
    
    # 广播消息
    await comm_manager.broadcast_message("agent_1", MessageType.NOTIFICATION, "System update")
    
    # 停止协议
    await comm_manager.stop_all_protocols()

if __name__ == "__main__":
    asyncio.run(main())
