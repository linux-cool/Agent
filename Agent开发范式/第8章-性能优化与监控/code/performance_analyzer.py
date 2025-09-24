#!/usr/bin/env python3
"""
性能分析器 - 智能体性能监控和分析的核心组件

本模块实现了智能体系统的性能分析功能，包括：
- 性能指标收集
- 性能数据分析
- 性能报告生成
- 基准对比分析
"""

import asyncio
import time
import psutil
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import statistics

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricType(Enum):
    """指标类型枚举"""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CONCURRENT_REQUESTS = "concurrent_requests"

class PerformanceLevel(Enum):
    """性能等级枚举"""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    CRITICAL = "critical"

@dataclass
class TimeRange:
    """时间范围"""
    start_time: datetime
    end_time: datetime
    
    @property
    def duration(self) -> timedelta:
        return self.end_time - self.start_time

@dataclass
class MetricValue:
    """指标值"""
    value: float
    timestamp: datetime
    metric_type: MetricType

@dataclass
class MetricSummary:
    """指标摘要"""
    metric_type: MetricType
    min_value: float
    max_value: float
    avg_value: float
    median_value: float
    p95_value: float
    p99_value: float
    count: int

@dataclass
class PerformanceMetrics:
    """性能指标集合"""
    cpu_usage: MetricSummary
    memory_usage: MetricSummary
    response_time: MetricSummary
    throughput: MetricSummary
    error_rate: MetricSummary
    concurrent_requests: MetricSummary

@dataclass
class BenchmarkComparison:
    """基准对比结果"""
    cpu_performance: str
    memory_performance: str
    response_performance: str
    overall_performance: str
    improvement_percentage: float

@dataclass
class PerformanceReport:
    """性能报告"""
    agent_id: str
    time_range: TimeRange
    metrics: PerformanceMetrics
    benchmark_comparison: Optional[BenchmarkComparison]
    recommendations: List[str]
    generated_at: datetime

class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_buffer: List[MetricValue] = []
        self.buffer_size = config.get("buffer_size", 1000)
    
    async def collect_metrics(self, agent_id: str, time_range: TimeRange) -> List[MetricValue]:
        """收集性能指标"""
        try:
            metrics = []
            current_time = time_range.start_time
            
            while current_time < time_range.end_time:
                # 收集CPU使用率
                cpu_usage = psutil.cpu_percent(interval=1)
                metrics.append(MetricValue(
                    value=cpu_usage,
                    timestamp=current_time,
                    metric_type=MetricType.CPU_USAGE
                ))
                
                # 收集内存使用率
                memory = psutil.virtual_memory()
                memory_usage = memory.percent
                metrics.append(MetricValue(
                    value=memory_usage,
                    timestamp=current_time,
                    metric_type=MetricType.MEMORY_USAGE
                ))
                
                # 模拟其他指标收集
                response_time = self._simulate_response_time()
                metrics.append(MetricValue(
                    value=response_time,
                    timestamp=current_time,
                    metric_type=MetricType.RESPONSE_TIME
                ))
                
                throughput = self._simulate_throughput()
                metrics.append(MetricValue(
                    value=throughput,
                    timestamp=current_time,
                    metric_type=MetricType.THROUGHPUT
                ))
                
                error_rate = self._simulate_error_rate()
                metrics.append(MetricValue(
                    value=error_rate,
                    timestamp=current_time,
                    metric_type=MetricType.ERROR_RATE
                ))
                
                concurrent_requests = self._simulate_concurrent_requests()
                metrics.append(MetricValue(
                    value=concurrent_requests,
                    timestamp=current_time,
                    metric_type=MetricType.CONCURRENT_REQUESTS
                ))
                
                current_time += timedelta(seconds=1)
                await asyncio.sleep(0.1)  # 避免过于频繁的收集
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            raise
    
    def _simulate_response_time(self) -> float:
        """模拟响应时间"""
        import random
        return random.uniform(50, 500)  # 50-500ms
    
    def _simulate_throughput(self) -> float:
        """模拟吞吐量"""
        import random
        return random.uniform(10, 100)  # 10-100 requests/second
    
    def _simulate_error_rate(self) -> float:
        """模拟错误率"""
        import random
        return random.uniform(0, 5)  # 0-5%
    
    def _simulate_concurrent_requests(self) -> float:
        """模拟并发请求数"""
        import random
        return random.uniform(1, 50)  # 1-50 concurrent requests

class AnalyzerEngine:
    """分析引擎"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def analyze(self, metrics: List[MetricValue]) -> PerformanceMetrics:
        """分析性能指标"""
        try:
            # 按指标类型分组
            grouped_metrics = self._group_metrics_by_type(metrics)
            
            # 计算各指标摘要
            cpu_usage = self._calculate_metric_summary(grouped_metrics[MetricType.CPU_USAGE])
            memory_usage = self._calculate_metric_summary(grouped_metrics[MetricType.MEMORY_USAGE])
            response_time = self._calculate_metric_summary(grouped_metrics[MetricType.RESPONSE_TIME])
            throughput = self._calculate_metric_summary(grouped_metrics[MetricType.THROUGHPUT])
            error_rate = self._calculate_metric_summary(grouped_metrics[MetricType.ERROR_RATE])
            concurrent_requests = self._calculate_metric_summary(grouped_metrics[MetricType.CONCURRENT_REQUESTS])
            
            return PerformanceMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                response_time=response_time,
                throughput=throughput,
                error_rate=error_rate,
                concurrent_requests=concurrent_requests
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze metrics: {e}")
            raise
    
    def _group_metrics_by_type(self, metrics: List[MetricValue]) -> Dict[MetricType, List[MetricValue]]:
        """按指标类型分组"""
        grouped = {}
        for metric in metrics:
            if metric.metric_type not in grouped:
                grouped[metric.metric_type] = []
            grouped[metric.metric_type].append(metric)
        return grouped
    
    def _calculate_metric_summary(self, metrics: List[MetricValue]) -> MetricSummary:
        """计算指标摘要"""
        if not metrics:
            return MetricSummary(
                metric_type=MetricType.CPU_USAGE,
                min_value=0, max_value=0, avg_value=0, median_value=0,
                p95_value=0, p99_value=0, count=0
            )
        
        values = [m.value for m in metrics]
        metric_type = metrics[0].metric_type
        
        return MetricSummary(
            metric_type=metric_type,
            min_value=min(values),
            max_value=max(values),
            avg_value=statistics.mean(values),
            median_value=statistics.median(values),
            p95_value=self._calculate_percentile(values, 95),
            p99_value=self._calculate_percentile(values, 99),
            count=len(values)
        )
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """计算百分位数"""
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def generate_report(self, analysis_result: PerformanceMetrics) -> PerformanceReport:
        """生成性能报告"""
        try:
            # 生成建议
            recommendations = self._generate_recommendations(analysis_result)
            
            # 创建报告
            report = PerformanceReport(
                agent_id="agent_001",
                time_range=TimeRange(
                    start_time=datetime.now() - timedelta(hours=1),
                    end_time=datetime.now()
                ),
                metrics=analysis_result,
                benchmark_comparison=None,
                recommendations=recommendations,
                generated_at=datetime.now()
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            raise
    
    def _generate_recommendations(self, metrics: PerformanceMetrics) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # CPU使用率建议
        if metrics.cpu_usage.avg_value > 80:
            recommendations.append("CPU使用率过高，建议优化算法或增加CPU资源")
        elif metrics.cpu_usage.avg_value < 20:
            recommendations.append("CPU使用率较低，可以考虑增加并发处理能力")
        
        # 内存使用率建议
        if metrics.memory_usage.avg_value > 85:
            recommendations.append("内存使用率过高，建议优化内存使用或增加内存资源")
        
        # 响应时间建议
        if metrics.response_time.avg_value > 1000:
            recommendations.append("响应时间过长，建议优化算法或增加计算资源")
        
        # 吞吐量建议
        if metrics.throughput.avg_value < 10:
            recommendations.append("吞吐量较低，建议优化处理逻辑或增加并发能力")
        
        # 错误率建议
        if metrics.error_rate.avg_value > 2:
            recommendations.append("错误率较高，建议检查系统稳定性和错误处理逻辑")
        
        return recommendations

class BenchmarkManager:
    """基准管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.benchmarks: Dict[str, PerformanceMetrics] = {}
    
    async def get_benchmark(self, agent_id: str) -> Optional[PerformanceMetrics]:
        """获取基准性能数据"""
        return self.benchmarks.get(agent_id)
    
    async def set_benchmark(self, agent_id: str, metrics: PerformanceMetrics):
        """设置基准性能数据"""
        self.benchmarks[agent_id] = metrics

class PerformanceAnalyzer:
    """性能分析器主类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_collector = MetricsCollector(config)
        self.analyzer_engine = AnalyzerEngine(config)
        self.report_generator = ReportGenerator(config)
        self.benchmark_manager = BenchmarkManager(config)
    
    async def analyze_performance(self, agent_id: str, time_range: TimeRange) -> PerformanceReport:
        """分析智能体性能"""
        try:
            # 1. 收集性能指标
            metrics = await self.metrics_collector.collect_metrics(agent_id, time_range)
            
            # 2. 分析性能数据
            analysis_result = await self.analyzer_engine.analyze(metrics)
            
            # 3. 生成性能报告
            report = await self.report_generator.generate_report(analysis_result)
            
            # 4. 与基准对比
            benchmark = await self.benchmark_manager.get_benchmark(agent_id)
            if benchmark:
                report.benchmark_comparison = await self._compare_with_benchmark(analysis_result, benchmark)
            
            return report
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            raise
    
    async def _compare_with_benchmark(self, analysis: PerformanceMetrics, benchmark: PerformanceMetrics) -> BenchmarkComparison:
        """与基准性能对比"""
        comparison = BenchmarkComparison(
            cpu_performance="stable",
            memory_performance="stable",
            response_performance="stable",
            overall_performance="stable",
            improvement_percentage=0.0
        )
        
        # CPU性能对比
        if analysis.cpu_usage.avg_value > benchmark.cpu_usage.avg_value * 1.1:
            comparison.cpu_performance = "degraded"
        elif analysis.cpu_usage.avg_value < benchmark.cpu_usage.avg_value * 0.9:
            comparison.cpu_performance = "improved"
        
        # 内存性能对比
        if analysis.memory_usage.avg_value > benchmark.memory_usage.avg_value * 1.1:
            comparison.memory_performance = "degraded"
        elif analysis.memory_usage.avg_value < benchmark.memory_usage.avg_value * 0.9:
            comparison.memory_performance = "improved"
        
        # 响应时间对比
        if analysis.response_time.avg_value > benchmark.response_time.avg_value * 1.2:
            comparison.response_performance = "degraded"
        elif analysis.response_time.avg_value < benchmark.response_time.avg_value * 0.8:
            comparison.response_performance = "improved"
        
        # 计算整体改进百分比
        cpu_improvement = (benchmark.cpu_usage.avg_value - analysis.cpu_usage.avg_value) / benchmark.cpu_usage.avg_value * 100
        memory_improvement = (benchmark.memory_usage.avg_value - analysis.memory_usage.avg_value) / benchmark.memory_usage.avg_value * 100
        response_improvement = (benchmark.response_time.avg_value - analysis.response_time.avg_value) / benchmark.response_time.avg_value * 100
        
        comparison.improvement_percentage = (cpu_improvement + memory_improvement + response_improvement) / 3
        
        return comparison

# 异常类
class PerformanceAnalysisError(Exception):
    """性能分析错误"""
    pass

# 示例使用
async def main():
    """主函数示例"""
    config = {
        "buffer_size": 1000,
        "collection_interval": 1
    }
    
    analyzer = PerformanceAnalyzer(config)
    
    # 分析最近1小时的性能
    time_range = TimeRange(
        start_time=datetime.now() - timedelta(hours=1),
        end_time=datetime.now()
    )
    
    try:
        report = await analyzer.analyze_performance("agent_001", time_range)
        
        print(f"性能分析报告:")
        print(f"Agent ID: {report.agent_id}")
        print(f"时间范围: {report.time_range.start_time} - {report.time_range.end_time}")
        print(f"CPU使用率: {report.metrics.cpu_usage.avg_value:.2f}%")
        print(f"内存使用率: {report.metrics.memory_usage.avg_value:.2f}%")
        print(f"平均响应时间: {report.metrics.response_time.avg_value:.2f}ms")
        print(f"平均吞吐量: {report.metrics.throughput.avg_value:.2f} req/s")
        print(f"错误率: {report.metrics.error_rate.avg_value:.2f}%")
        
        print(f"\n优化建议:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"{i}. {rec}")
        
    except Exception as e:
        logger.error(f"Performance analysis failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
