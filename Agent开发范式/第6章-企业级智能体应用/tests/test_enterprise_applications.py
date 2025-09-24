# test_enterprise_applications.py
"""
第6章 企业级智能体应用 - 测试用例
测试企业级智能体应用的各个组件和功能
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# 添加代码路径到sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))

from enterprise_scenarios import (
    EnterpriseScenarioAnalyzer, EnterpriseScenario, ScenarioType, 
    BusinessValue, ComplexityLevel, ImplementationEffort, RiskLevel
)
from customer_service_system import (
    CustomerServiceSystem, Customer, Conversation, Message, 
    ConversationStatus, MessageType
)
from code_assistant import (
    CodeAssistant, CodeFile, CodeLanguage, CodeIssue, CodeMetrics,
    TestCase, TestType, Documentation, DocumentationType
)
from business_automation import (
    BusinessAutomationSystem, ProcessDefinition, ProcessInstance,
    ProcessStatus, DeploymentStrategy, EnvironmentType
)
from deployment_ops import (
    DeploymentOpsSystem, Application, Deployment, Service,
    DeploymentStatus, ServiceStatus, EnvironmentType as DeployEnvType
)

class TestEnterpriseScenarioAnalyzer:
    """测试企业应用场景分析器"""
    
    @pytest.fixture
    def analyzer(self):
        """创建分析器实例"""
        config = {
            "requirements": {},
            "impact": {},
            "recommendations": {}
        }
        return EnterpriseScenarioAnalyzer(config)
    
    @pytest.fixture
    def sample_scenario_data(self):
        """示例场景数据"""
        return {
            "name": "智能客服系统",
            "description": "基于AI的24/7客户服务支持系统",
            "scenario_type": "CUSTOMER_SERVICE",
            "business_value": "HIGH",
            "technical_complexity": "MEDIUM",
            "implementation_effort": "MEDIUM",
            "risk_level": "MEDIUM",
            "stakeholders": ["客服部门", "IT部门", "管理层"],
            "success_metrics": ["客户满意度", "响应时间", "成本降低"]
        }
    
    @pytest.mark.asyncio
    async def test_analyze_scenario(self, analyzer, sample_scenario_data):
        """测试场景分析"""
        scenario = await analyzer.analyze_scenario(sample_scenario_data)
        
        assert scenario is not None
        assert scenario.name == "智能客服系统"
        assert scenario.scenario_type == ScenarioType.CUSTOMER_SERVICE
        assert scenario.business_value == BusinessValue.HIGH
        assert scenario.technical_complexity == ComplexityLevel.MEDIUM
        assert len(scenario.business_requirements) > 0
        assert len(scenario.technical_requirements) > 0
        assert len(scenario.business_impacts) > 0
        assert len(scenario.recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_compare_scenarios(self, analyzer, sample_scenario_data):
        """测试场景比较"""
        # 创建两个场景
        scenario1 = await analyzer.analyze_scenario(sample_scenario_data)
        
        scenario2_data = sample_scenario_data.copy()
        scenario2_data["name"] = "代码助手"
        scenario2_data["scenario_type"] = "CODE_ASSISTANT"
        scenario2 = await analyzer.analyze_scenario(scenario2_data)
        
        # 比较场景
        comparison = await analyzer.compare_scenarios([scenario1.id, scenario2.id])
        
        assert "scenarios" in comparison
        assert "comparison_matrix" in comparison
        assert "recommendations" in comparison
        assert len(comparison["scenarios"]) == 2
    
    def test_get_stats(self, analyzer):
        """测试统计信息"""
        stats = analyzer.get_stats()
        
        assert "total_scenarios" in stats
        assert "scenario_types" in stats
        assert "complexity_distribution" in stats
        assert "business_value_distribution" in stats

class TestCustomerServiceSystem:
    """测试智能客服系统"""
    
    @pytest.fixture
    def cs_system(self):
        """创建客服系统实例"""
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
        return CustomerServiceSystem(config)
    
    @pytest.mark.asyncio
    async def test_handle_customer_query(self, cs_system):
        """测试处理客户查询"""
        response = await cs_system.handle_customer_query(
            "customer_001",
            "我忘记了密码，怎么重置？",
            "web"
        )
        
        assert "conversation_id" in response
        assert "response" in response
        assert "confidence" in response
        assert "status" in response
        assert response["confidence"] >= 0.0
        assert response["confidence"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_handle_complaint_query(self, cs_system):
        """测试处理投诉查询"""
        response = await cs_system.handle_customer_query(
            "customer_002",
            "你们的产品质量太差了，我要投诉！",
            "web"
        )
        
        assert "conversation_id" in response
        assert "response" in response
        # 投诉应该触发升级
        assert response["status"] in ["活跃", "已升级"]
    
    @pytest.mark.asyncio
    async def test_conversation_creation(self, cs_system):
        """测试对话创建"""
        # 第一次查询
        response1 = await cs_system.handle_customer_query(
            "customer_003",
            "你好",
            "web"
        )
        
        # 第二次查询（应该使用同一个对话）
        response2 = await cs_system.handle_customer_query(
            "customer_003",
            "有什么可以帮助我的吗？",
            "web"
        )
        
        assert response1["conversation_id"] == response2["conversation_id"]
    
    def test_get_conversation(self, cs_system):
        """测试获取对话"""
        conversation = cs_system.get_conversation("non_existent_id")
        assert conversation is None
    
    def test_get_analytics(self, cs_system):
        """测试获取分析数据"""
        analytics = cs_system.get_analytics()
        assert isinstance(analytics, dict)

class TestCodeAssistant:
    """测试代码助手"""
    
    @pytest.fixture
    def code_assistant(self):
        """创建代码助手实例"""
        config = {
            "analysis": {},
            "generation": {},
            "testing": {},
            "documentation": {}
        }
        return CodeAssistant(config)
    
    @pytest.fixture
    def sample_code_file(self):
        """示例代码文件"""
        return CodeFile(
            name="test.py",
            content="""
def calculate_sum(a, b):
    # This is a very long line that exceeds the recommended line length limit and should trigger a warning
    result = a + b
    return result

def process_data(data):
    if data is None:
        return None
    if len(data) == 0:
        return []
    if len(data) > 100:
        return data[:100]
    return data
""",
            language=CodeLanguage.PYTHON
        )
    
    @pytest.mark.asyncio
    async def test_code_generation(self, code_assistant):
        """测试代码生成"""
        request = {
            "type": "code_generation",
            "description": "创建一个计算两个数之和的函数",
            "language": "PYTHON",
            "context": {"return_type": "int"}
        }
        
        result = await code_assistant.assist_development(request)
        
        assert result["status"] == "success"
        assert "generated_code" in result["data"]
        assert "test_cases" in result["data"]
        assert len(result["data"]["generated_code"]) > 0
    
    @pytest.mark.asyncio
    async def test_code_review(self, code_assistant, sample_code_file):
        """测试代码审查"""
        request = {
            "type": "code_review",
            "code": sample_code_file.content,
            "language": "PYTHON",
            "filename": "test.py"
        }
        
        result = await code_assistant.assist_development(request)
        
        assert result["status"] == "success"
        assert "issues" in result["data"]
        assert "metrics" in result["data"]
        assert len(result["data"]["issues"]) > 0  # 应该发现长行问题
        assert result["data"]["metrics"]["cyclomatic_complexity"] > 0
    
    @pytest.mark.asyncio
    async def test_documentation_generation(self, code_assistant, sample_code_file):
        """测试文档生成"""
        request = {
            "type": "documentation",
            "code": sample_code_file.content,
            "language": "PYTHON",
            "filename": "test.py",
            "doc_type": "CODE_COMMENTS"
        }
        
        result = await code_assistant.assist_development(request)
        
        assert result["status"] == "success"
        assert "documentation" in result["data"]
        assert len(result["data"]["documentation"]["content"]) > 0
    
    @pytest.mark.asyncio
    async def test_refactoring_suggestions(self, code_assistant, sample_code_file):
        """测试重构建议"""
        request = {
            "type": "refactoring",
            "code": sample_code_file.content
        }
        
        result = await code_assistant.assist_development(request)
        
        assert result["status"] == "success"
        assert "refactoring_suggestions" in result["data"]
        assert len(result["data"]["refactoring_suggestions"]) >= 0
    
    @pytest.mark.asyncio
    async def test_invalid_request_type(self, code_assistant):
        """测试无效请求类型"""
        request = {
            "type": "invalid_type",
            "description": "测试无效请求"
        }
        
        result = await code_assistant.assist_development(request)
        
        assert result["status"] == "error"
        assert "error" in result

class TestBusinessAutomationSystem:
    """测试业务流程自动化系统"""
    
    @pytest.fixture
    def automation_system(self):
        """创建自动化系统实例"""
        config = {
            "workflow": {},
            "rules": {}
        }
        return BusinessAutomationSystem(config)
    
    @pytest.fixture
    async def started_system(self, automation_system):
        """启动的系统"""
        await automation_system.start()
        yield automation_system
        await automation_system.stop()
    
    @pytest.mark.asyncio
    async def test_system_start_stop(self, automation_system):
        """测试系统启动和停止"""
        await automation_system.start()
        assert automation_system.workflow_engine.running
        
        await automation_system.stop()
        assert not automation_system.workflow_engine.running
    
    @pytest.mark.asyncio
    async def test_get_process_templates(self, started_system):
        """测试获取流程模板"""
        templates = started_system.get_process_templates()
        
        assert len(templates) > 0
        template_names = [t.name for t in templates]
        assert "员工入职流程" in template_names
        assert "采购审批流程" in template_names
    
    @pytest.mark.asyncio
    async def test_start_employee_onboarding_process(self, started_system):
        """测试启动员工入职流程"""
        variables = {
            "employee_name": "张三",
            "department": "技术部",
            "position": "软件工程师",
            "start_date": "2025-01-01"
        }
        
        instance = await started_system.start_process(
            "员工入职流程",
            variables,
            "HR系统"
        )
        
        assert instance is not None
        assert instance.process_definition_id is not None
        assert instance.variables["employee_name"] == "张三"
        assert instance.status == ProcessStatus.RUNNING
    
    @pytest.mark.asyncio
    async def test_start_purchase_approval_process(self, started_system):
        """测试启动采购审批流程"""
        variables = {
            "item": "办公设备",
            "amount": 15000,
            "department": "行政部",
            "urgency": "normal",
            "department_budget": 50000
        }
        
        instance = await started_system.start_process(
            "采购审批流程",
            variables,
            "采购系统"
        )
        
        assert instance is not None
        assert instance.variables["amount"] == 15000
        assert instance.status == ProcessStatus.RUNNING
    
    @pytest.mark.asyncio
    async def test_get_process_instance(self, started_system):
        """测试获取流程实例"""
        # 启动一个流程
        instance = await started_system.start_process(
            "员工入职流程",
            {"employee_name": "李四"},
            "HR系统"
        )
        
        # 获取流程实例
        retrieved_instance = started_system.get_process_instance(instance.id)
        
        assert retrieved_instance is not None
        assert retrieved_instance.id == instance.id
        assert retrieved_instance.name == instance.name
    
    def test_get_stats(self, started_system):
        """测试获取统计信息"""
        stats = started_system.get_stats()
        
        assert "total_process_templates" in stats
        assert "total_process_instances" in stats
        assert "total_services" in stats
        assert "total_business_rules" in stats

class TestDeploymentOpsSystem:
    """测试部署运维系统"""
    
    @pytest.fixture
    def ops_system(self):
        """创建部署运维系统实例"""
        config = {
            "container": {},
            "monitoring": {"monitoring_interval": 5},
            "logging": {"log_storage": "file"}
        }
        return DeploymentOpsSystem(config)
    
    @pytest.fixture
    async def started_ops_system(self, ops_system):
        """启动的部署运维系统"""
        await ops_system.start()
        yield ops_system
        await ops_system.stop()
    
    @pytest.fixture
    def sample_application(self):
        """示例应用程序"""
        return Application(
            name="智能客服系统",
            version="1.0.0",
            description="基于AI的智能客服系统",
            image="customer-service:1.0.0",
            ports=[8080, 8081],
            environment_variables={
                "ENV": "production",
                "LOG_LEVEL": "INFO"
            },
            resources={
                "cpu": "500m",
                "memory": "1Gi"
            },
            health_check={
                "path": "/health",
                "port": 8080,
                "interval": 30
            }
        )
    
    @pytest.mark.asyncio
    async def test_system_start_stop(self, ops_system):
        """测试系统启动和停止"""
        await ops_system.start()
        assert ops_system.monitoring_system.running
        
        await ops_system.stop()
        assert not ops_system.monitoring_system.running
    
    @pytest.mark.asyncio
    async def test_register_application(self, started_ops_system, sample_application):
        """测试注册应用程序"""
        result = await started_ops_system.register_application(sample_application)
        
        assert result is True
        assert sample_application.id in started_ops_system.applications
    
    @pytest.mark.asyncio
    async def test_deploy_application(self, started_ops_system, sample_application):
        """测试部署应用程序"""
        # 先注册应用程序
        await started_ops_system.register_application(sample_application)
        
        # 部署应用程序
        deployment = await started_ops_system.deploy_application(
            sample_application.id,
            DeployEnvType.PRODUCTION,
            DeploymentStrategy.BLUE_GREEN,
            "admin"
        )
        
        assert deployment is not None
        assert deployment.application_id == sample_application.id
        assert deployment.environment == DeployEnvType.PRODUCTION
        assert deployment.strategy == DeploymentStrategy.BLUE_GREEN
        assert deployment.status == DeploymentStatus.DEPLOYED
    
    @pytest.mark.asyncio
    async def test_scale_service(self, started_ops_system, sample_application):
        """测试服务扩缩容"""
        # 注册并部署应用程序
        await started_ops_system.register_application(sample_application)
        deployment = await started_ops_system.deploy_application(
            sample_application.id,
            DeployEnvType.PRODUCTION
        )
        
        # 获取服务
        services = list(started_ops_system.services.values())
        assert len(services) > 0
        
        service = services[0]
        original_replicas = service.replicas
        
        # 扩容到3个副本
        result = await started_ops_system.scale_service(service.id, 3)
        
        assert result is True
        updated_service = started_ops_system.get_service_status(service.id)
        assert updated_service.replicas == 3
        assert updated_service.running_replicas == 3
    
    def test_get_deployment_status(self, started_ops_system):
        """测试获取部署状态"""
        deployment = started_ops_system.get_deployment_status("non_existent_id")
        assert deployment is None
    
    def test_get_service_status(self, started_ops_system):
        """测试获取服务状态"""
        service = started_ops_system.get_service_status("non_existent_id")
        assert service is None
    
    def test_get_system_metrics(self, started_ops_system):
        """测试获取系统指标"""
        metrics = started_ops_system.get_system_metrics()
        
        assert "total_applications" in metrics
        assert "total_deployments" in metrics
        assert "total_services" in metrics
        assert "active_deployments" in metrics
        assert "healthy_services" in metrics
    
    def test_get_alerts_summary(self, started_ops_system):
        """测试获取告警摘要"""
        summary = started_ops_system.get_alerts_summary()
        
        assert "total_alerts" in summary
        assert "active_alerts" in summary
        assert "critical_alerts" in summary
        assert "error_alerts" in summary
        assert "warning_alerts" in summary

class TestIntegrationScenarios:
    """集成测试场景"""
    
    @pytest.fixture
    async def full_system(self):
        """完整的系统集成"""
        # 创建所有系统
        analyzer = EnterpriseScenarioAnalyzer({})
        cs_system = CustomerServiceSystem({
            "chat": {}, "tickets": {}, "escalation": {}, "analytics": {}
        })
        code_assistant = CodeAssistant({
            "analysis": {}, "generation": {}, "testing": {}, "documentation": {}
        })
        automation_system = BusinessAutomationSystem({"workflow": {}, "rules": {}})
        ops_system = DeploymentOpsSystem({
            "container": {}, "monitoring": {"monitoring_interval": 5}, "logging": {}
        })
        
        # 启动系统
        await automation_system.start()
        await ops_system.start()
        
        yield {
            "analyzer": analyzer,
            "cs_system": cs_system,
            "code_assistant": code_assistant,
            "automation_system": automation_system,
            "ops_system": ops_system
        }
        
        # 清理
        await automation_system.stop()
        await ops_system.stop()
    
    @pytest.mark.asyncio
    async def test_end_to_end_customer_service_workflow(self, full_system):
        """端到端客服工作流测试"""
        # 1. 分析客服场景
        scenario_data = {
            "name": "智能客服系统",
            "description": "基于AI的24/7客户服务支持系统",
            "scenario_type": "CUSTOMER_SERVICE",
            "business_value": "HIGH",
            "technical_complexity": "MEDIUM",
            "implementation_effort": "MEDIUM",
            "risk_level": "MEDIUM",
            "stakeholders": ["客服部门", "IT部门", "管理层"],
            "success_metrics": ["客户满意度", "响应时间", "成本降低"]
        }
        
        scenario = await full_system["analyzer"].analyze_scenario(scenario_data)
        assert scenario is not None
        
        # 2. 处理客户查询
        response = await full_system["cs_system"].handle_customer_query(
            "customer_001",
            "我忘记了密码，怎么重置？",
            "web"
        )
        assert "response" in response
        
        # 3. 生成客服系统代码
        code_request = {
            "type": "code_generation",
            "description": "创建一个智能客服系统的核心类",
            "language": "PYTHON"
        }
        code_result = await full_system["code_assistant"].assist_development(code_request)
        assert code_result["status"] == "success"
        
        # 4. 启动客服流程
        process_instance = await full_system["automation_system"].start_process(
            "员工入职流程",
            {"employee_name": "客服专员", "department": "客服部"},
            "HR系统"
        )
        assert process_instance is not None
        
        # 5. 部署客服系统
        app = Application(
            name="智能客服系统",
            version="1.0.0",
            description="基于AI的智能客服系统",
            image="customer-service:1.0.0",
            ports=[8080]
        )
        
        await full_system["ops_system"].register_application(app)
        deployment = await full_system["ops_system"].deploy_application(
            app.id,
            DeployEnvType.PRODUCTION
        )
        assert deployment is not None
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, full_system):
        """负载下的性能测试"""
        # 并发处理多个客户查询
        tasks = []
        for i in range(10):
            task = full_system["cs_system"].handle_customer_query(
                f"customer_{i:03d}",
                f"这是第{i+1}个客户查询",
                "web"
            )
            tasks.append(task)
        
        # 等待所有任务完成
        responses = await asyncio.gather(*tasks)
        
        # 验证所有响应都成功
        for response in responses:
            assert "response" in response
            assert "conversation_id" in response
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, full_system):
        """错误处理和恢复测试"""
        # 测试无效的客户查询
        response = await full_system["cs_system"].handle_customer_query(
            "customer_error",
            "",  # 空查询
            "web"
        )
        assert "response" in response
        
        # 测试无效的代码生成请求
        invalid_request = {
            "type": "invalid_type",
            "description": "无效请求"
        }
        result = await full_system["code_assistant"].assist_development(invalid_request)
        assert result["status"] == "error"
        
        # 测试不存在的流程启动
        instance = await full_system["automation_system"].start_process(
            "不存在的流程",
            {},
            "测试系统"
        )
        assert instance is None

# 性能测试
class TestPerformance:
    """性能测试"""
    
    @pytest.mark.asyncio
    async def test_customer_service_response_time(self):
        """测试客服系统响应时间"""
        cs_system = CustomerServiceSystem({
            "chat": {}, "tickets": {}, "escalation": {}, "analytics": {}
        })
        
        start_time = datetime.now()
        response = await cs_system.handle_customer_query(
            "customer_perf",
            "性能测试查询",
            "web"
        )
        end_time = datetime.now()
        
        response_time = (end_time - start_time).total_seconds()
        assert response_time < 1.0  # 响应时间应小于1秒
        assert "response" in response
    
    @pytest.mark.asyncio
    async def test_code_assistant_generation_time(self):
        """测试代码助手生成时间"""
        code_assistant = CodeAssistant({
            "analysis": {}, "generation": {}, "testing": {}, "documentation": {}
        })
        
        start_time = datetime.now()
        result = await code_assistant.assist_development({
            "type": "code_generation",
            "description": "创建一个简单的函数",
            "language": "PYTHON"
        })
        end_time = datetime.now()
        
        generation_time = (end_time - start_time).total_seconds()
        assert generation_time < 2.0  # 生成时间应小于2秒
        assert result["status"] == "success"

# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
