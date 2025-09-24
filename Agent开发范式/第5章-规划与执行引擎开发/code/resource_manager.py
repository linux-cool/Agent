# resource_manager.py
"""
第5章 规划与执行引擎开发 - 资源管理器
实现智能体的资源管理、分配、监控和优化功能
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
import psutil
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResourceType(Enum):
    """资源类型枚举"""
    CPU = "CPU"
    MEMORY = "内存"
    STORAGE = "存储"
    NETWORK = "网络"
    GPU = "GPU"
    DISK_IO = "磁盘IO"
    CUSTOM = "自定义"

class ResourceState(Enum):
    """资源状态枚举"""
    AVAILABLE = "可用"
    ALLOCATED = "已分配"
    RESERVED = "已预留"
    UNAVAILABLE = "不可用"
    MAINTENANCE = "维护中"

class AllocationStrategy(Enum):
    """分配策略枚举"""
    FIRST_FIT = "首次适应"
    BEST_FIT = "最佳适应"
    WORST_FIT = "最坏适应"
    NEXT_FIT = "循环适应"
    PRIORITY_BASED = "基于优先级"
    COST_OPTIMIZED = "成本优化"

@dataclass
class ResourceSpec:
    """资源规格"""
    resource_type: ResourceType = ResourceType.CPU
    total_amount: float = 1.0
    unit: str = "cores"
    cost_per_unit: float = 1.0
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "resource_type": self.resource_type.value,
            "total_amount": self.total_amount,
            "unit": self.unit,
            "cost_per_unit": self.cost_per_unit,
            "constraints": self.constraints,
            "metadata": self.metadata
        }

@dataclass
class ResourceAllocation:
    """资源分配"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    resource_type: ResourceType = ResourceType.CPU
    amount: float = 1.0
    unit: str = "cores"
    allocated_to: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration: timedelta = field(default_factory=lambda: timedelta(minutes=1))
    cost: float = 0.0
    priority: int = 1
    status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "resource_type": self.resource_type.value,
            "amount": self.amount,
            "unit": self.unit,
            "allocated_to": self.allocated_to,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration.total_seconds(),
            "cost": self.cost,
            "priority": self.priority,
            "status": self.status,
            "metadata": self.metadata
        }

@dataclass
class ResourcePool:
    """资源池"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    resource_spec: ResourceSpec = field(default_factory=lambda: ResourceSpec())
    available_amount: float = 1.0
    allocated_amount: float = 0.0
    reserved_amount: float = 0.0
    state: ResourceState = ResourceState.AVAILABLE
    allocation_strategy: AllocationStrategy = AllocationStrategy.FIRST_FIT
    max_allocation_per_request: float = 1.0
    min_allocation_per_request: float = 0.1
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "resource_spec": self.resource_spec.to_dict(),
            "available_amount": self.available_amount,
            "allocated_amount": self.allocated_amount,
            "reserved_amount": self.reserved_amount,
            "state": self.state.value,
            "allocation_strategy": self.allocation_strategy.value,
            "max_allocation_per_request": self.max_allocation_per_request,
            "min_allocation_per_request": self.min_allocation_per_request,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }

@dataclass
class ResourceRequest:
    """资源请求"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    requester: str = ""
    resource_type: ResourceType = ResourceType.CPU
    amount: float = 1.0
    unit: str = "cores"
    duration: timedelta = field(default_factory=lambda: timedelta(minutes=1))
    priority: int = 1
    deadline: Optional[datetime] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "requester": self.requester,
            "resource_type": self.resource_type.value,
            "amount": self.amount,
            "unit": self.unit,
            "duration": self.duration.total_seconds(),
            "priority": self.priority,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "constraints": self.constraints,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "metadata": self.metadata
        }

@dataclass
class ResourceMetrics:
    """资源指标"""
    resource_type: ResourceType = ResourceType.CPU
    utilization: float = 0.0
    throughput: float = 0.0
    latency: float = 0.0
    error_rate: float = 0.0
    cost_per_unit: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "resource_type": self.resource_type.value,
            "utilization": self.utilization,
            "throughput": self.throughput,
            "latency": self.latency,
            "error_rate": self.error_rate,
            "cost_per_unit": self.cost_per_unit,
            "timestamp": self.timestamp.isoformat()
        }

class SystemResourceMonitor:
    """系统资源监控器"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics_history: Dict[ResourceType, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.monitoring_interval = 1.0  # 秒
    
    async def start_monitoring(self):
        """开始监控"""
        self.monitoring = True
        asyncio.create_task(self._monitoring_loop())
        logger.info("System resource monitoring started")
    
    async def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        logger.info("System resource monitoring stopped")
    
    async def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                await self._collect_metrics()
                await asyncio.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
    
    async def _collect_metrics(self):
        """收集指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_metrics = ResourceMetrics(
                ResourceType.CPU,
                utilization=cpu_percent / 100.0,
                throughput=cpu_percent,
                latency=0.0,
                error_rate=0.0,
                cost_per_unit=1.0
            )
            self.metrics_history[ResourceType.CPU].append(cpu_metrics)
            
            # 内存使用率
            memory = psutil.virtual_memory()
            memory_metrics = ResourceMetrics(
                ResourceType.MEMORY,
                utilization=memory.percent / 100.0,
                throughput=memory.used / (1024**3),  # GB
                latency=0.0,
                error_rate=0.0,
                cost_per_unit=0.5
            )
            self.metrics_history[ResourceType.MEMORY].append(memory_metrics)
            
            # 磁盘使用率
            disk = psutil.disk_usage('/')
            disk_metrics = ResourceMetrics(
                ResourceType.STORAGE,
                utilization=disk.percent / 100.0,
                throughput=disk.used / (1024**3),  # GB
                latency=0.0,
                error_rate=0.0,
                cost_per_unit=0.1
            )
            self.metrics_history[ResourceType.STORAGE].append(disk_metrics)
            
            # 网络使用率
            network = psutil.net_io_counters()
            network_metrics = ResourceMetrics(
                ResourceType.NETWORK,
                utilization=0.0,  # 需要计算
                throughput=(network.bytes_sent + network.bytes_recv) / (1024**2),  # MB
                latency=0.0,
                error_rate=0.0,
                cost_per_unit=2.0
            )
            self.metrics_history[ResourceType.NETWORK].append(network_metrics)
            
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
    
    def get_current_metrics(self) -> Dict[ResourceType, ResourceMetrics]:
        """获取当前指标"""
        current_metrics = {}
        for resource_type, metrics_deque in self.metrics_history.items():
            if metrics_deque:
                current_metrics[resource_type] = metrics_deque[-1]
        return current_metrics
    
    def get_metrics_history(self, resource_type: ResourceType, limit: int = 100) -> List[ResourceMetrics]:
        """获取指标历史"""
        if resource_type in self.metrics_history:
            return list(self.metrics_history[resource_type])[-limit:]
        return []

class ResourceManager:
    """资源管理器主类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.resource_pools: Dict[str, ResourcePool] = {}
        self.allocations: Dict[str, ResourceAllocation] = {}
        self.requests: Dict[str, ResourceRequest] = {}
        self.monitor = SystemResourceMonitor()
        self.allocation_strategies = {
            AllocationStrategy.FIRST_FIT: self._first_fit_allocation,
            AllocationStrategy.BEST_FIT: self._best_fit_allocation,
            AllocationStrategy.WORST_FIT: self._worst_fit_allocation,
            AllocationStrategy.NEXT_FIT: self._next_fit_allocation,
            AllocationStrategy.PRIORITY_BASED: self._priority_based_allocation,
            AllocationStrategy.COST_OPTIMIZED: self._cost_optimized_allocation
        }
        self.running = False
    
    async def start(self):
        """启动资源管理器"""
        self.running = True
        
        # 初始化默认资源池
        await self._initialize_default_pools()
        
        # 启动监控
        await self.monitor.start_monitoring()
        
        # 启动资源管理循环
        asyncio.create_task(self._management_loop())
        
        logger.info("Resource manager started")
    
    async def stop(self):
        """停止资源管理器"""
        self.running = False
        await self.monitor.stop_monitoring()
        logger.info("Resource manager stopped")
    
    async def _initialize_default_pools(self):
        """初始化默认资源池"""
        default_pools = [
            ResourcePool(
                name="CPU池",
                resource_spec=ResourceSpec(ResourceType.CPU, 8.0, "cores", 1.0),
                available_amount=8.0,
                allocation_strategy=AllocationStrategy.FIRST_FIT
            ),
            ResourcePool(
                name="内存池",
                resource_spec=ResourceSpec(ResourceType.MEMORY, 16.0, "GB", 0.5),
                available_amount=16.0,
                allocation_strategy=AllocationStrategy.BEST_FIT
            ),
            ResourcePool(
                name="存储池",
                resource_spec=ResourceSpec(ResourceType.STORAGE, 100.0, "GB", 0.1),
                available_amount=100.0,
                allocation_strategy=AllocationStrategy.FIRST_FIT
            ),
            ResourcePool(
                name="网络池",
                resource_spec=ResourceSpec(ResourceType.NETWORK, 1.0, "Gbps", 2.0),
                available_amount=1.0,
                allocation_strategy=AllocationStrategy.PRIORITY_BASED
            )
        ]
        
        for pool in default_pools:
            self.resource_pools[pool.id] = pool
            logger.info(f"Initialized resource pool: {pool.name}")
    
    async def _management_loop(self):
        """资源管理循环"""
        while self.running:
            try:
                await self._process_pending_requests()
                await self._cleanup_expired_allocations()
                await self._optimize_resource_usage()
                await asyncio.sleep(1.0)
            except Exception as e:
                logger.error(f"Resource management loop error: {e}")
    
    async def _process_pending_requests(self):
        """处理待处理请求"""
        try:
            pending_requests = [req for req in self.requests.values() if req.status == "pending"]
            
            # 按优先级排序
            pending_requests.sort(key=lambda x: x.priority, reverse=True)
            
            for request in pending_requests:
                if await self._can_allocate(request):
                    await self._allocate_resources(request)
                else:
                    # 检查是否可以等待
                    if request.deadline and datetime.now() > request.deadline:
                        request.status = "expired"
                        logger.warning(f"Resource request {request.id} expired")
                    
        except Exception as e:
            logger.error(f"Request processing failed: {e}")
    
    async def _can_allocate(self, request: ResourceRequest) -> bool:
        """检查是否可以分配资源"""
        try:
            # 查找合适的资源池
            suitable_pools = []
            for pool in self.resource_pools.values():
                if (pool.resource_spec.resource_type == request.resource_type and
                    pool.state == ResourceState.AVAILABLE and
                    pool.available_amount >= request.amount):
                    suitable_pools.append(pool)
            
            return len(suitable_pools) > 0
            
        except Exception as e:
            logger.error(f"Allocation check failed: {e}")
            return False
    
    async def _allocate_resources(self, request: ResourceRequest) -> bool:
        """分配资源"""
        try:
            # 查找合适的资源池
            suitable_pools = []
            for pool in self.resource_pools.values():
                if (pool.resource_spec.resource_type == request.resource_type and
                    pool.state == ResourceState.AVAILABLE and
                    pool.available_amount >= request.amount):
                    suitable_pools.append(pool)
            
            if not suitable_pools:
                return False
            
            # 选择资源池（使用第一个合适的池）
            selected_pool = suitable_pools[0]
            
            # 创建分配
            allocation = ResourceAllocation(
                resource_type=request.resource_type,
                amount=request.amount,
                unit=request.unit,
                allocated_to=request.requester,
                duration=request.duration,
                cost=request.amount * selected_pool.resource_spec.cost_per_unit * request.duration.total_seconds() / 3600,
                priority=request.priority,
                metadata=request.metadata
            )
            
            # 更新资源池
            selected_pool.available_amount -= request.amount
            selected_pool.allocated_amount += request.amount
            selected_pool.updated_at = datetime.now()
            
            # 保存分配和请求
            self.allocations[allocation.id] = allocation
            request.status = "allocated"
            
            logger.info(f"Allocated {request.amount} {request.unit} of {request.resource_type.value} to {request.requester}")
            return True
            
        except Exception as e:
            logger.error(f"Resource allocation failed: {e}")
            return False
    
    async def _cleanup_expired_allocations(self):
        """清理过期分配"""
        try:
            current_time = datetime.now()
            expired_allocations = []
            
            for allocation in self.allocations.values():
                if allocation.status == "active":
                    expected_end_time = allocation.start_time + allocation.duration
                    if current_time > expected_end_time:
                        expired_allocations.append(allocation.id)
            
            for allocation_id in expired_allocations:
                await self.deallocate_resources(allocation_id)
                
        except Exception as e:
            logger.error(f"Allocation cleanup failed: {e}")
    
    async def _optimize_resource_usage(self):
        """优化资源使用"""
        try:
            # 检查资源利用率
            for pool in self.resource_pools.values():
                utilization = pool.allocated_amount / pool.resource_spec.total_amount
                
                # 如果利用率过低，考虑合并分配
                if utilization < 0.3:
                    logger.debug(f"Low utilization in pool {pool.name}: {utilization:.2%}")
                
                # 如果利用率过高，考虑扩展
                elif utilization > 0.9:
                    logger.warning(f"High utilization in pool {pool.name}: {utilization:.2%}")
                    
        except Exception as e:
            logger.error(f"Resource optimization failed: {e}")
    
    # 分配策略实现
    async def _first_fit_allocation(self, request: ResourceRequest) -> Optional[str]:
        """首次适应分配"""
        for pool in self.resource_pools.values():
            if (pool.resource_spec.resource_type == request.resource_type and
                pool.available_amount >= request.amount):
                return pool.id
        return None
    
    async def _best_fit_allocation(self, request: ResourceRequest) -> Optional[str]:
        """最佳适应分配"""
        best_pool = None
        min_waste = float('inf')
        
        for pool in self.resource_pools.values():
            if (pool.resource_spec.resource_type == request.resource_type and
                pool.available_amount >= request.amount):
                waste = pool.available_amount - request.amount
                if waste < min_waste:
                    min_waste = waste
                    best_pool = pool.id
        
        return best_pool
    
    async def _worst_fit_allocation(self, request: ResourceRequest) -> Optional[str]:
        """最坏适应分配"""
        worst_pool = None
        max_available = 0
        
        for pool in self.resource_pools.values():
            if (pool.resource_spec.resource_type == request.resource_type and
                pool.available_amount >= request.amount):
                if pool.available_amount > max_available:
                    max_available = pool.available_amount
                    worst_pool = pool.id
        
        return worst_pool
    
    async def _next_fit_allocation(self, request: ResourceRequest) -> Optional[str]:
        """循环适应分配"""
        # 简化实现：返回第一个合适的池
        return await self._first_fit_allocation(request)
    
    async def _priority_based_allocation(self, request: ResourceRequest) -> Optional[str]:
        """基于优先级分配"""
        # 按优先级排序资源池
        sorted_pools = sorted(
            [pool for pool in self.resource_pools.values() 
             if pool.resource_spec.resource_type == request.resource_type and
             pool.available_amount >= request.amount],
            key=lambda p: p.resource_spec.cost_per_unit
        )
        
        if sorted_pools:
            return sorted_pools[0].id
        return None
    
    async def _cost_optimized_allocation(self, request: ResourceRequest) -> Optional[str]:
        """成本优化分配"""
        # 选择成本最低的资源池
        cheapest_pool = None
        min_cost = float('inf')
        
        for pool in self.resource_pools.values():
            if (pool.resource_spec.resource_type == request.resource_type and
                pool.available_amount >= request.amount):
                cost = pool.resource_spec.cost_per_unit * request.amount
                if cost < min_cost:
                    min_cost = cost
                    cheapest_pool = pool.id
        
        return cheapest_pool
    
    # 公共接口
    async def request_resources(self, request: ResourceRequest) -> str:
        """请求资源"""
        self.requests[request.id] = request
        logger.info(f"Resource request submitted: {request.id}")
        return request.id
    
    async def deallocate_resources(self, allocation_id: str) -> bool:
        """释放资源"""
        try:
            if allocation_id not in self.allocations:
                return False
            
            allocation = self.allocations[allocation_id]
            
            # 查找对应的资源池
            for pool in self.resource_pools.values():
                if pool.resource_spec.resource_type == allocation.resource_type:
                    pool.available_amount += allocation.amount
                    pool.allocated_amount -= allocation.amount
                    pool.updated_at = datetime.now()
                    break
            
            # 更新分配状态
            allocation.status = "released"
            allocation.end_time = datetime.now()
            
            logger.info(f"Deallocated resources: {allocation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Resource deallocation failed: {e}")
            return False
    
    def create_resource_pool(self, pool: ResourcePool) -> str:
        """创建资源池"""
        self.resource_pools[pool.id] = pool
        logger.info(f"Created resource pool: {pool.name}")
        return pool.id
    
    def remove_resource_pool(self, pool_id: str) -> bool:
        """移除资源池"""
        if pool_id in self.resource_pools:
            pool = self.resource_pools[pool_id]
            if pool.allocated_amount > 0:
                logger.warning(f"Cannot remove pool {pool.name} with active allocations")
                return False
            
            del self.resource_pools[pool_id]
            logger.info(f"Removed resource pool: {pool.name}")
            return True
        
        return False
    
    def get_resource_pool(self, pool_id: str) -> Optional[ResourcePool]:
        """获取资源池"""
        return self.resource_pools.get(pool_id)
    
    def get_allocation(self, allocation_id: str) -> Optional[ResourceAllocation]:
        """获取分配"""
        return self.allocations.get(allocation_id)
    
    def get_request(self, request_id: str) -> Optional[ResourceRequest]:
        """获取请求"""
        return self.requests.get(request_id)
    
    def get_resource_utilization(self) -> Dict[str, float]:
        """获取资源利用率"""
        utilization = {}
        
        for pool in self.resource_pools.values():
            resource_name = pool.resource_spec.resource_type.value
            if pool.resource_spec.total_amount > 0:
                utilization[resource_name] = pool.allocated_amount / pool.resource_spec.total_amount
            else:
                utilization[resource_name] = 0.0
        
        return utilization
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_pools = len(self.resource_pools)
        active_allocations = len([a for a in self.allocations.values() if a.status == "active"])
        pending_requests = len([r for r in self.requests.values() if r.status == "pending"])
        
        return {
            "total_resource_pools": total_pools,
            "active_allocations": active_allocations,
            "pending_requests": pending_requests,
            "resource_utilization": self.get_resource_utilization(),
            "current_metrics": {rt.value: metrics.to_dict() for rt, metrics in self.monitor.get_current_metrics().items()}
        }

# 示例用法
async def main_demo():
    """资源管理器演示"""
    # 创建配置
    config = {
        "monitoring_interval": 1.0,
        "optimization_enabled": True
    }
    
    # 创建资源管理器
    resource_manager = ResourceManager(config)
    await resource_manager.start()
    
    print("资源管理器演示")
    print("=" * 50)
    
    # 创建资源请求
    print("\n1. 创建资源请求...")
    requests = [
        ResourceRequest(
            requester="agent_1",
            resource_type=ResourceType.CPU,
            amount=2.0,
            unit="cores",
            duration=timedelta(minutes=5),
            priority=3
        ),
        ResourceRequest(
            requester="agent_2",
            resource_type=ResourceType.MEMORY,
            amount=4.0,
            unit="GB",
            duration=timedelta(minutes=10),
            priority=2
        ),
        ResourceRequest(
            requester="agent_3",
            resource_type=ResourceType.STORAGE,
            amount=10.0,
            unit="GB",
            duration=timedelta(minutes=15),
            priority=1
        )
    ]
    
    # 提交请求
    request_ids = []
    for request in requests:
        request_id = await resource_manager.request_resources(request)
        request_ids.append(request_id)
        print(f"✓ 提交请求: {request.requester} -> {request.resource_type.value} ({request.amount} {request.unit})")
    
    # 等待处理
    print("\n2. 等待资源分配...")
    await asyncio.sleep(3)
    
    # 检查请求状态
    print("\n3. 检查请求状态:")
    for request_id in request_ids:
        request = resource_manager.get_request(request_id)
        if request:
            print(f"  {request.requester}: {request.status}")
    
    # 检查分配
    print("\n4. 检查资源分配:")
    for allocation in resource_manager.allocations.values():
        if allocation.status == "active":
            print(f"  {allocation.allocated_to}: {allocation.amount} {allocation.unit} of {allocation.resource_type.value}")
    
    # 获取统计信息
    print("\n5. 资源管理器统计:")
    stats = resource_manager.get_stats()
    for key, value in stats.items():
        if key != "current_metrics":
            print(f"  {key}: {value}")
    
    # 显示当前指标
    print("\n6. 当前系统指标:")
    current_metrics = stats.get("current_metrics", {})
    for resource_type, metrics in current_metrics.items():
        print(f"  {resource_type}:")
        print(f"    利用率: {metrics['utilization']:.2%}")
        print(f"    吞吐量: {metrics['throughput']:.2f}")
        print(f"    成本: {metrics['cost_per_unit']:.2f}")
    
    # 模拟资源释放
    print("\n7. 模拟资源释放...")
    for allocation in resource_manager.allocations.values():
        if allocation.status == "active":
            await resource_manager.deallocate_resources(allocation.id)
            print(f"✓ 释放资源: {allocation.allocated_to}")
    
    # 最终统计
    print("\n8. 最终统计:")
    final_stats = resource_manager.get_stats()
    print(f"  活跃分配: {final_stats['active_allocations']}")
    print(f"  待处理请求: {final_stats['pending_requests']}")
    
    # 停止资源管理器
    await resource_manager.stop()
    print("\n资源管理器演示完成")

if __name__ == "__main__":
    asyncio.run(main_demo())
