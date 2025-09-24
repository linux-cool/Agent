# fault_tolerance.py
"""
容错与故障恢复实现
提供多种容错机制和故障恢复策略
"""

import asyncio
import logging
import random
import time
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import uuid
import heapq
from collections import defaultdict, deque
import statistics

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaultType(Enum):
    """故障类型枚举"""
    AGENT_FAILURE = "agent_failure"
    COMMUNICATION_FAILURE = "communication_failure"
    TASK_FAILURE = "task_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_PARTITION = "network_partition"
    BYZANTINE_FAULT = "byzantine_fault"
    TIMEOUT = "timeout"
    MEMORY_LEAK = "memory_leak"

class RecoveryStrategy(Enum):
    """恢复策略枚举"""
    RESTART = "restart"
    REPLICA = "replica"
    MIGRATION = "migration"
    ROLLBACK = "rollback"
    COMPENSATION = "compensation"
    ADAPTIVE = "adaptive"

class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    FAILED = "failed"
    RECOVERING = "recovering"

@dataclass
class FaultEvent:
    """故障事件数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    fault_type: FaultType = FaultType.AGENT_FAILURE
    agent_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    severity: int = 1  # 1-5, 5为最严重
    description: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    recovery_time: Optional[datetime] = None

@dataclass
class HealthCheck:
    """健康检查数据结构"""
    agent_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    status: HealthStatus = HealthStatus.HEALTHY
    metrics: Dict[str, float] = field(default_factory=dict)
    response_time: float = 0.0
    error_count: int = 0
    last_successful_check: Optional[datetime] = None

@dataclass
class RecoveryAction:
    """恢复行动数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    fault_event_id: str = ""
    strategy: RecoveryStrategy = RecoveryStrategy.RESTART
    target_agent: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, in_progress, completed, failed
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None

class CircuitBreaker:
    """熔断器"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60, recovery_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.recovery_timeout = recovery_timeout
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        self.success_count = 0
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """调用函数，带熔断保护"""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
                self.success_count = 0
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """是否应该尝试重置"""
        if self.last_failure_time is None:
            return True
        
        return (datetime.now() - self.last_failure_time).total_seconds() > self.recovery_timeout
    
    def _on_success(self):
        """成功回调"""
        self.failure_count = 0
        
        if self.state == "half_open":
            self.success_count += 1
            if self.success_count >= 3:  # 连续成功3次
                self.state = "closed"
    
    def _on_failure(self):
        """失败回调"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
    
    def get_state(self) -> str:
        """获取状态"""
        return self.state

class HealthMonitor:
    """健康监控器"""
    
    def __init__(self, check_interval: int = 30, timeout: int = 10):
        self.check_interval = check_interval
        self.timeout = timeout
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.monitoring = False
        self.check_tasks: Dict[str, asyncio.Task] = {}
    
    async def start_monitoring(self, agent_id: str, check_func: Callable):
        """开始监控智能体"""
        if agent_id in self.check_tasks:
            logger.warning(f"Already monitoring agent {agent_id}")
            return
        
        self.check_tasks[agent_id] = asyncio.create_task(
            self._monitor_agent(agent_id, check_func)
        )
        logger.info(f"Started monitoring agent {agent_id}")
    
    async def stop_monitoring(self, agent_id: str):
        """停止监控智能体"""
        if agent_id in self.check_tasks:
            self.check_tasks[agent_id].cancel()
            del self.check_tasks[agent_id]
            logger.info(f"Stopped monitoring agent {agent_id}")
    
    async def _monitor_agent(self, agent_id: str, check_func: Callable):
        """监控智能体"""
        while True:
            try:
                start_time = time.time()
                
                # 执行健康检查
                health_status = await asyncio.wait_for(
                    check_func(agent_id), 
                    timeout=self.timeout
                )
                
                response_time = time.time() - start_time
                
                # 更新健康检查记录
                health_check = HealthCheck(
                    agent_id=agent_id,
                    status=health_status,
                    response_time=response_time,
                    last_successful_check=datetime.now()
                )
                
                self.health_checks[agent_id] = health_check
                self.health_history[agent_id].append(health_check)
                
                logger.debug(f"Health check for {agent_id}: {health_status.value}")
                
            except asyncio.TimeoutError:
                logger.warning(f"Health check timeout for agent {agent_id}")
                self._record_failure(agent_id, "timeout")
            except Exception as e:
                logger.error(f"Health check failed for agent {agent_id}: {e}")
                self._record_failure(agent_id, str(e))
            
            await asyncio.sleep(self.check_interval)
    
    def _record_failure(self, agent_id: str, error: str):
        """记录失败"""
        health_check = HealthCheck(
            agent_id=agent_id,
            status=HealthStatus.FAILED,
            error_count=1
        )
        
        self.health_checks[agent_id] = health_check
        self.health_history[agent_id].append(health_check)
    
    def get_agent_health(self, agent_id: str) -> Optional[HealthCheck]:
        """获取智能体健康状态"""
        return self.health_checks.get(agent_id)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """获取健康摘要"""
        total_agents = len(self.health_checks)
        healthy_agents = len([h for h in self.health_checks.values() if h.status == HealthStatus.HEALTHY])
        degraded_agents = len([h for h in self.health_checks.values() if h.status == HealthStatus.DEGRADED])
        failed_agents = len([h for h in self.health_checks.values() if h.status == HealthStatus.FAILED])
        
        return {
            "total_agents": total_agents,
            "healthy_agents": healthy_agents,
            "degraded_agents": degraded_agents,
            "failed_agents": failed_agents,
            "health_rate": healthy_agents / total_agents if total_agents > 0 else 0
        }

class FaultDetector:
    """故障检测器"""
    
    def __init__(self):
        self.fault_patterns: Dict[FaultType, List[Callable]] = defaultdict(list)
        self.fault_history: List[FaultEvent] = []
        self.detection_thresholds: Dict[str, float] = {
            "response_time": 5.0,  # 5秒
            "error_rate": 0.1,     # 10%
            "memory_usage": 0.9,   # 90%
            "cpu_usage": 0.9       # 90%
        }
    
    def register_fault_pattern(self, fault_type: FaultType, detector: Callable):
        """注册故障模式"""
        self.fault_patterns[fault_type].append(detector)
    
    async def detect_faults(self, agent_id: str, metrics: Dict[str, Any]) -> List[FaultEvent]:
        """检测故障"""
        detected_faults = []
        
        for fault_type, detectors in self.fault_patterns.items():
            for detector in detectors:
                try:
                    fault = await detector(agent_id, metrics)
                    if fault:
                        detected_faults.append(fault)
                except Exception as e:
                    logger.error(f"Fault detector error: {e}")
        
        # 记录检测到的故障
        for fault in detected_faults:
            self.fault_history.append(fault)
        
        return detected_faults
    
    def _detect_response_time_fault(self, agent_id: str, metrics: Dict[str, Any]) -> Optional[FaultEvent]:
        """检测响应时间故障"""
        response_time = metrics.get("response_time", 0)
        threshold = self.detection_thresholds["response_time"]
        
        if response_time > threshold:
            return FaultEvent(
                fault_type=FaultType.TIMEOUT,
                agent_id=agent_id,
                severity=3,
                description=f"Response time {response_time}s exceeds threshold {threshold}s",
                context={"response_time": response_time, "threshold": threshold}
            )
        
        return None
    
    def _detect_error_rate_fault(self, agent_id: str, metrics: Dict[str, Any]) -> Optional[FaultEvent]:
        """检测错误率故障"""
        error_rate = metrics.get("error_rate", 0)
        threshold = self.detection_thresholds["error_rate"]
        
        if error_rate > threshold:
            return FaultEvent(
                fault_type=FaultType.TASK_FAILURE,
                agent_id=agent_id,
                severity=4,
                description=f"Error rate {error_rate} exceeds threshold {threshold}",
                context={"error_rate": error_rate, "threshold": threshold}
            )
        
        return None
    
    def _detect_resource_exhaustion_fault(self, agent_id: str, metrics: Dict[str, Any]) -> Optional[FaultEvent]:
        """检测资源耗尽故障"""
        memory_usage = metrics.get("memory_usage", 0)
        cpu_usage = metrics.get("cpu_usage", 0)
        
        if memory_usage > self.detection_thresholds["memory_usage"]:
            return FaultEvent(
                fault_type=FaultType.RESOURCE_EXHAUSTION,
                agent_id=agent_id,
                severity=5,
                description=f"Memory usage {memory_usage} exceeds threshold",
                context={"memory_usage": memory_usage, "cpu_usage": cpu_usage}
            )
        
        if cpu_usage > self.detection_thresholds["cpu_usage"]:
            return FaultEvent(
                fault_type=FaultType.RESOURCE_EXHAUSTION,
                agent_id=agent_id,
                severity=4,
                description=f"CPU usage {cpu_usage} exceeds threshold",
                context={"memory_usage": memory_usage, "cpu_usage": cpu_usage}
            )
        
        return None

class RecoveryManager:
    """恢复管理器"""
    
    def __init__(self):
        self.recovery_strategies: Dict[RecoveryStrategy, Callable] = {
            RecoveryStrategy.RESTART: self._restart_recovery,
            RecoveryStrategy.REPLICA: self._replica_recovery,
            RecoveryStrategy.MIGRATION: self._migration_recovery,
            RecoveryStrategy.ROLLBACK: self._rollback_recovery,
            RecoveryStrategy.COMPENSATION: self._compensation_recovery,
            RecoveryStrategy.ADAPTIVE: self._adaptive_recovery
        }
        self.recovery_actions: Dict[str, RecoveryAction] = {}
        self.recovery_history: List[RecoveryAction] = []
        self.agent_replicas: Dict[str, List[str]] = defaultdict(list)
    
    async def recover_from_fault(self, fault_event: FaultEvent) -> Optional[RecoveryAction]:
        """从故障中恢复"""
        # 选择恢复策略
        strategy = self._select_recovery_strategy(fault_event)
        
        # 创建恢复行动
        recovery_action = RecoveryAction(
            fault_event_id=fault_event.id,
            strategy=strategy,
            target_agent=fault_event.agent_id,
            status="pending"
        )
        
        self.recovery_actions[recovery_action.id] = recovery_action
        
        try:
            # 执行恢复
            recovery_action.status = "in_progress"
            result = await self.recovery_strategies[strategy](recovery_action)
            
            recovery_action.status = "completed"
            recovery_action.result = result
            recovery_action.timestamp = datetime.now()
            
            logger.info(f"Recovery completed for fault {fault_event.id}")
            
        except Exception as e:
            recovery_action.status = "failed"
            recovery_action.error = str(e)
            logger.error(f"Recovery failed for fault {fault_event.id}: {e}")
        
        # 记录恢复历史
        self.recovery_history.append(recovery_action)
        
        return recovery_action
    
    def _select_recovery_strategy(self, fault_event: FaultEvent) -> RecoveryStrategy:
        """选择恢复策略"""
        if fault_event.fault_type == FaultType.AGENT_FAILURE:
            return RecoveryStrategy.REPLICA
        elif fault_event.fault_type == FaultType.RESOURCE_EXHAUSTION:
            return RecoveryStrategy.MIGRATION
        elif fault_event.fault_type == FaultType.TASK_FAILURE:
            return RecoveryStrategy.ROLLBACK
        elif fault_event.fault_type == FaultType.BYZANTINE_FAULT:
            return RecoveryStrategy.COMPENSATION
        else:
            return RecoveryStrategy.RESTART
    
    async def _restart_recovery(self, recovery_action: RecoveryAction) -> Any:
        """重启恢复"""
        agent_id = recovery_action.target_agent
        logger.info(f"Restarting agent {agent_id}")
        
        # 模拟重启过程
        await asyncio.sleep(1)
        
        return f"Agent {agent_id} restarted successfully"
    
    async def _replica_recovery(self, recovery_action: RecoveryAction) -> Any:
        """副本恢复"""
        agent_id = recovery_action.target_agent
        
        # 检查是否有副本
        if agent_id in self.agent_replicas and self.agent_replicas[agent_id]:
            replica_id = self.agent_replicas[agent_id][0]
            logger.info(f"Activating replica {replica_id} for agent {agent_id}")
            
            # 模拟激活副本
            await asyncio.sleep(0.5)
            
            return f"Replica {replica_id} activated for agent {agent_id}"
        else:
            # 创建新副本
            new_replica_id = f"{agent_id}_replica_{int(time.time())}"
            self.agent_replicas[agent_id].append(new_replica_id)
            
            logger.info(f"Created new replica {new_replica_id} for agent {agent_id}")
            
            # 模拟创建副本
            await asyncio.sleep(1)
            
            return f"New replica {new_replica_id} created for agent {agent_id}"
    
    async def _migration_recovery(self, recovery_action: RecoveryAction) -> Any:
        """迁移恢复"""
        agent_id = recovery_action.target_agent
        logger.info(f"Migrating agent {agent_id}")
        
        # 模拟迁移过程
        await asyncio.sleep(2)
        
        return f"Agent {agent_id} migrated successfully"
    
    async def _rollback_recovery(self, recovery_action: RecoveryAction) -> Any:
        """回滚恢复"""
        agent_id = recovery_action.target_agent
        logger.info(f"Rolling back agent {agent_id}")
        
        # 模拟回滚过程
        await asyncio.sleep(1)
        
        return f"Agent {agent_id} rolled back successfully"
    
    async def _compensation_recovery(self, recovery_action: RecoveryAction) -> Any:
        """补偿恢复"""
        agent_id = recovery_action.target_agent
        logger.info(f"Compensating for agent {agent_id}")
        
        # 模拟补偿过程
        await asyncio.sleep(1.5)
        
        return f"Compensation completed for agent {agent_id}"
    
    async def _adaptive_recovery(self, recovery_action: RecoveryAction) -> Any:
        """自适应恢复"""
        agent_id = recovery_action.target_agent
        
        # 分析历史故障模式
        recent_faults = [f for f in self.recovery_history if f.target_agent == agent_id][-5:]
        
        if recent_faults:
            # 选择最成功的恢复策略
            successful_recoveries = [r for r in recent_faults if r.status == "completed"]
            if successful_recoveries:
                best_strategy = successful_recoveries[0].strategy
                logger.info(f"Using adaptive strategy {best_strategy} for agent {agent_id}")
                return await self.recovery_strategies[best_strategy](recovery_action)
        
        # 默认使用重启策略
        return await self._restart_recovery(recovery_action)
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """获取恢复统计"""
        total_recoveries = len(self.recovery_history)
        successful_recoveries = len([r for r in self.recovery_history if r.status == "completed"])
        failed_recoveries = len([r for r in self.recovery_history if r.status == "failed"])
        
        strategy_counts = defaultdict(int)
        for recovery in self.recovery_history:
            strategy_counts[recovery.strategy.value] += 1
        
        return {
            "total_recoveries": total_recoveries,
            "successful_recoveries": successful_recoveries,
            "failed_recoveries": failed_recoveries,
            "success_rate": successful_recoveries / total_recoveries if total_recoveries > 0 else 0,
            "strategy_distribution": dict(strategy_counts)
        }

class FaultToleranceManager:
    """容错管理器"""
    
    def __init__(self):
        self.health_monitor = HealthMonitor()
        self.fault_detector = FaultDetector()
        self.recovery_manager = RecoveryManager()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # 注册默认故障检测器
        self.fault_detector.register_fault_pattern(FaultType.TIMEOUT, self.fault_detector._detect_response_time_fault)
        self.fault_detector.register_fault_pattern(FaultType.TASK_FAILURE, self.fault_detector._detect_error_rate_fault)
        self.fault_detector.register_fault_pattern(FaultType.RESOURCE_EXHAUSTION, self.fault_detector._detect_resource_exhaustion_fault)
    
    async def start_fault_tolerance(self, agent_id: str, health_check_func: Callable):
        """启动容错保护"""
        # 启动健康监控
        await self.health_monitor.start_monitoring(agent_id, health_check_func)
        
        # 创建熔断器
        self.circuit_breakers[agent_id] = CircuitBreaker()
        
        logger.info(f"Fault tolerance started for agent {agent_id}")
    
    async def stop_fault_tolerance(self, agent_id: str):
        """停止容错保护"""
        # 停止健康监控
        await self.health_monitor.stop_monitoring(agent_id)
        
        # 移除熔断器
        if agent_id in self.circuit_breakers:
            del self.circuit_breakers[agent_id]
        
        logger.info(f"Fault tolerance stopped for agent {agent_id}")
    
    async def check_agent_health(self, agent_id: str, metrics: Dict[str, Any]) -> List[FaultEvent]:
        """检查智能体健康状态"""
        # 检测故障
        faults = await self.fault_detector.detect_faults(agent_id, metrics)
        
        # 处理检测到的故障
        for fault in faults:
            await self._handle_fault(fault)
        
        return faults
    
    async def _handle_fault(self, fault_event: FaultEvent):
        """处理故障"""
        logger.warning(f"Fault detected: {fault_event.description}")
        
        # 执行恢复
        recovery_action = await self.recovery_manager.recover_from_fault(fault_event)
        
        if recovery_action and recovery_action.status == "completed":
            fault_event.resolved = True
            fault_event.recovery_time = datetime.now()
            logger.info(f"Fault {fault_event.id} resolved successfully")
        else:
            logger.error(f"Failed to resolve fault {fault_event.id}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        health_summary = self.health_monitor.get_health_summary()
        recovery_stats = self.recovery_manager.get_recovery_statistics()
        
        return {
            "health_summary": health_summary,
            "recovery_statistics": recovery_stats,
            "active_circuit_breakers": len(self.circuit_breakers),
            "total_faults_detected": len(self.fault_detector.fault_history)
        }

# 使用示例
async def main():
    """主函数示例"""
    # 创建容错管理器
    fault_tolerance_manager = FaultToleranceManager()
    
    # 模拟健康检查函数
    async def health_check_func(agent_id: str) -> HealthStatus:
        # 模拟健康检查
        await asyncio.sleep(0.1)
        
        # 随机返回健康状态
        statuses = [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
        return random.choice(statuses)
    
    # 启动容错保护
    await fault_tolerance_manager.start_fault_tolerance("agent_1", health_check_func)
    await fault_tolerance_manager.start_fault_tolerance("agent_2", health_check_func)
    await fault_tolerance_manager.start_fault_tolerance("agent_3", health_check_func)
    
    # 模拟故障检测
    print("Testing Fault Detection:")
    
    # 模拟响应时间故障
    metrics = {"response_time": 6.0, "error_rate": 0.05, "memory_usage": 0.7, "cpu_usage": 0.6}
    faults = await fault_tolerance_manager.check_agent_health("agent_1", metrics)
    
    for fault in faults:
        print(f"Fault detected: {fault.fault_type.value} - {fault.description}")
    
    # 模拟资源耗尽故障
    metrics = {"response_time": 2.0, "error_rate": 0.05, "memory_usage": 0.95, "cpu_usage": 0.8}
    faults = await fault_tolerance_manager.check_agent_health("agent_2", metrics)
    
    for fault in faults:
        print(f"Fault detected: {fault.fault_type.value} - {fault.description}")
    
    # 等待一段时间让健康监控运行
    await asyncio.sleep(5)
    
    # 获取系统健康状态
    system_health = fault_tolerance_manager.get_system_health()
    print(f"\nSystem Health: {system_health}")
    
    # 停止容错保护
    await fault_tolerance_manager.stop_fault_tolerance("agent_1")
    await fault_tolerance_manager.stop_fault_tolerance("agent_2")
    await fault_tolerance_manager.stop_fault_tolerance("agent_3")

if __name__ == "__main__":
    asyncio.run(main())
