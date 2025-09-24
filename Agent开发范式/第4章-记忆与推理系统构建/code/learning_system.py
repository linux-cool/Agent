# learning_system.py
"""
第4章 记忆与推理系统构建 - 学习系统
实现智能体的学习系统，包括监督学习、无监督学习、强化学习等
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
import random
from collections import defaultdict, deque
import pickle
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LearningType(Enum):
    """学习类型枚举"""
    SUPERVISED = "监督学习"
    UNSUPERVISED = "无监督学习"
    REINFORCEMENT = "强化学习"
    TRANSFER = "迁移学习"
    META = "元学习"
    CONTINUAL = "持续学习"
    FEW_SHOT = "少样本学习"

class LearningAlgorithm(Enum):
    """学习算法枚举"""
    LINEAR_REGRESSION = "线性回归"
    LOGISTIC_REGRESSION = "逻辑回归"
    DECISION_TREE = "决策树"
    RANDOM_FOREST = "随机森林"
    SVM = "支持向量机"
    NEURAL_NETWORK = "神经网络"
    K_MEANS = "K均值"
    DBSCAN = "DBSCAN"
    PCA = "主成分分析"
    Q_LEARNING = "Q学习"
    POLICY_GRADIENT = "策略梯度"
    ACTOR_CRITIC = "演员-评论家"

class LearningMode(Enum):
    """学习模式枚举"""
    ONLINE = "在线学习"
    BATCH = "批量学习"
    INCREMENTAL = "增量学习"
    ACTIVE = "主动学习"

@dataclass
class TrainingData:
    """训练数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    features: List[float] = field(default_factory=list)
    label: Optional[Union[str, int, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "features": self.features,
            "label": self.label,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingData':
        """从字典创建训练数据对象"""
        training_data = cls()
        training_data.id = data.get("id", str(uuid.uuid4()))
        training_data.features = data.get("features", [])
        training_data.label = data.get("label")
        training_data.metadata = data.get("metadata", {})
        training_data.timestamp = datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat()))
        return training_data

@dataclass
class LearningTask:
    """学习任务数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    learning_type: LearningType = LearningType.SUPERVISED
    algorithm: LearningAlgorithm = LearningAlgorithm.LINEAR_REGRESSION
    mode: LearningMode = LearningMode.BATCH
    training_data: List[TrainingData] = field(default_factory=list)
    model: Optional[Any] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    status: str = "created"  # created, training, completed, failed
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "learning_type": self.learning_type.value,
            "algorithm": self.algorithm.value,
            "mode": self.mode.value,
            "training_data": [data.to_dict() for data in self.training_data],
            "model": self.model,
            "performance_metrics": self.performance_metrics,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningTask':
        """从字典创建学习任务对象"""
        task = cls()
        task.id = data.get("id", str(uuid.uuid4()))
        task.name = data.get("name", "")
        task.learning_type = LearningType(data.get("learning_type", "监督学习"))
        task.algorithm = LearningAlgorithm(data.get("algorithm", "线性回归"))
        task.mode = LearningMode(data.get("mode", "批量学习"))
        task.training_data = [TrainingData.from_dict(td) for td in data.get("training_data", [])]
        task.model = data.get("model")
        task.performance_metrics = data.get("performance_metrics", {})
        task.status = data.get("status", "created")
        task.created_at = datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        task.updated_at = datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat()))
        task.metadata = data.get("metadata", {})
        return task

@dataclass
class LearningResult:
    """学习结果数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    predictions: List[Any] = field(default_factory=list)
    accuracy: float = 0.0
    loss: float = 0.0
    confidence: float = 0.0
    evaluation_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "predictions": self.predictions,
            "accuracy": self.accuracy,
            "loss": self.loss,
            "confidence": self.confidence,
            "evaluation_metrics": self.evaluation_metrics,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

class SimpleLinearRegression:
    """简单线性回归实现"""
    
    def __init__(self):
        self.weights = None
        self.bias = None
        self.trained = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练模型"""
        try:
            # 添加偏置项
            X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
            
            # 计算权重 (最小二乘法)
            self.weights = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
            self.bias = self.weights[0]
            self.weights = self.weights[1:]
            self.trained = True
            
            logger.info("Linear regression model trained successfully")
        except Exception as e:
            logger.error(f"Linear regression training failed: {e}")
            self.trained = False
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if not self.trained:
            raise ValueError("Model not trained")
        
        return X @ self.weights + self.bias
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算R²分数"""
        if not self.trained:
            return 0.0
        
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return 1 - (ss_res / ss_tot)

class LearningSystem:
    """学习系统主类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tasks: Dict[str, LearningTask] = {}
        self.results: Dict[str, LearningResult] = {}
        self.models: Dict[str, Any] = {}
        self.running = False
    
    async def start(self):
        """启动学习系统"""
        self.running = True
        logger.info("Learning system started")
    
    async def stop(self):
        """停止学习系统"""
        self.running = False
        logger.info("Learning system stopped")
    
    async def create_task(self, task: LearningTask) -> bool:
        """创建学习任务"""
        try:
            self.tasks[task.id] = task
            logger.info(f"Created learning task: {task.name} ({task.id})")
            return True
        except Exception as e:
            logger.error(f"Failed to create learning task: {e}")
            return False
    
    async def add_training_data(self, task_id: str, data: TrainingData) -> bool:
        """添加训练数据"""
        try:
            if task_id not in self.tasks:
                logger.error(f"Task not found: {task_id}")
                return False
            
            self.tasks[task_id].training_data.append(data)
            logger.info(f"Added training data to task {task_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add training data: {e}")
            return False
    
    async def train_model(self, task_id: str) -> bool:
        """训练模型"""
        try:
            if task_id not in self.tasks:
                logger.error(f"Task not found: {task_id}")
                return False
            
            task = self.tasks[task_id]
            task.status = "training"
            
            # 准备训练数据
            if not task.training_data:
                logger.error("No training data available")
                task.status = "failed"
                return False
            
            # 转换为numpy数组
            X = np.array([data.features for data in task.training_data])
            
            if task.learning_type == LearningType.SUPERVISED:
                # 监督学习
                if task.algorithm == LearningAlgorithm.LINEAR_REGRESSION:
                    model = SimpleLinearRegression()
                    y = np.array([data.label for data in task.training_data if data.label is not None])
                    model.fit(X, y)
                    task.model = model
                    
                    # 计算性能指标
                    task.performance_metrics["r2_score"] = model.score(X, y)
                    task.performance_metrics["accuracy"] = task.performance_metrics["r2_score"]
                
                else:
                    logger.error(f"Unsupported supervised learning algorithm: {task.algorithm.value}")
                    task.status = "failed"
                    return False
            
            else:
                logger.error(f"Unsupported learning type: {task.learning_type.value}")
                task.status = "failed"
                return False
            
            task.status = "completed"
            task.updated_at = datetime.now()
            self.models[task_id] = task.model
            
            logger.info(f"Model trained successfully for task {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            if task_id in self.tasks:
                self.tasks[task_id].status = "failed"
            return False
    
    async def predict(self, task_id: str, features: List[float]) -> LearningResult:
        """预测"""
        try:
            if task_id not in self.tasks:
                raise ValueError(f"Task not found: {task_id}")
            
            task = self.tasks[task_id]
            if task.status != "completed":
                raise ValueError("Task not completed")
            
            model = task.model
            X = np.array([features])
            
            if task.learning_type == LearningType.SUPERVISED:
                if task.algorithm == LearningAlgorithm.LINEAR_REGRESSION:
                    prediction = model.predict(X)[0]
                else:
                    prediction = 0.0
                
                result = LearningResult(
                    task_id=task_id,
                    predictions=[prediction],
                    accuracy=task.performance_metrics.get("accuracy", 0.0),
                    confidence=0.8
                )
            else:
                result = LearningResult(
                    task_id=task_id,
                    predictions=[0],
                    accuracy=0.0,
                    confidence=0.0
                )
            
            self.results[result.id] = result
            logger.info(f"Prediction completed for task {task_id}")
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return LearningResult(task_id=task_id, predictions=[], accuracy=0.0, confidence=0.0)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_tasks = len(self.tasks)
        completed_tasks = len([task for task in self.tasks.values() if task.status == "completed"])
        failed_tasks = len([task for task in self.tasks.values() if task.status == "failed"])
        training_tasks = len([task for task in self.tasks.values() if task.status == "training"])
        
        total_data_points = sum(len(task.training_data) for task in self.tasks.values())
        
        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "training_tasks": training_tasks,
            "total_data_points": total_data_points,
            "total_results": len(self.results),
            "total_models": len(self.models)
        }

# 示例用法
async def main_demo():
    """学习系统演示"""
    # 创建学习系统配置
    config = {
        "max_tasks": 100,
        "max_training_data": 10000
    }
    
    # 创建学习系统
    learning_system = LearningSystem(config)
    await learning_system.start()
    
    # 创建监督学习任务
    print("创建监督学习任务...")
    supervised_task = LearningTask(
        name="房价预测",
        learning_type=LearningType.SUPERVISED,
        algorithm=LearningAlgorithm.LINEAR_REGRESSION,
        mode=LearningMode.BATCH
    )
    
    await learning_system.create_task(supervised_task)
    
    # 添加训练数据
    print("添加训练数据...")
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
        await learning_system.add_training_data(supervised_task.id, data)
    
    # 训练模型
    print("训练监督学习模型...")
    await learning_system.train_model(supervised_task.id)
    
    # 预测
    print("进行预测...")
    prediction_result = await learning_system.predict(supervised_task.id, [130, 3, 2])
    print(f"预测结果: {prediction_result.predictions[0]:.0f}")
    print(f"准确率: {prediction_result.accuracy:.3f}")
    print(f"置信度: {prediction_result.confidence:.3f}")
    
    # 获取统计信息
    print("\n学习系统统计:")
    stats = learning_system.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 停止学习系统
    await learning_system.stop()
    print("\n学习系统演示完成")

if __name__ == "__main__":
    asyncio.run(main_demo())