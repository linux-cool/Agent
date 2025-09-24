# enterprise_demo.py
"""
第6章 企业级智能体应用 - 综合演示程序
展示企业级智能体应用的完整功能和工作流程
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
import sys
import os

# 添加代码路径到sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))

from enterprise_scenarios import EnterpriseScenarioAnalyzer
from customer_service_system import CustomerServiceSystem
from code_assistant import CodeAssistant
from business_automation import BusinessAutomationSystem
from deployment_ops import DeploymentOpsSystem, Application, EnvironmentType as DeployEnvType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnterpriseAgentDemo:
    """企业级智能体应用演示"""
    
    def __init__(self):
        self.scenario_analyzer = None
        self.customer_service = None
        self.code_assistant = None
        self.business_automation = None
        self.deployment_ops = None
        self.demo_data = self._prepare_demo_data()
    
    def _prepare_demo_data(self):
        """准备演示数据"""
        return {
            "scenarios": [
                {
                    "name": "智能客服系统",
                    "description": "基于AI的24/7客户服务支持系统",
                    "scenario_type": "CUSTOMER_SERVICE",
                    "business_value": "HIGH",
                    "technical_complexity": "MEDIUM",
                    "implementation_effort": "MEDIUM",
                    "risk_level": "MEDIUM",
                    "stakeholders": ["客服部门", "IT部门", "管理层"],
                    "success_metrics": ["客户满意度", "响应时间", "成本降低"]
                },
                {
                    "name": "AI代码助手",
                    "description": "基于AI的代码生成和审查助手",
                    "scenario_type": "CODE_ASSISTANT",
                    "business_value": "HIGH",
                    "technical_complexity": "HIGH",
                    "implementation_effort": "HIGH",
                    "risk_level": "MEDIUM",
                    "stakeholders": ["开发团队", "技术负责人", "产品经理"],
                    "success_metrics": ["开发效率", "代码质量", "错误率"]
                },
                {
                    "name": "业务流程自动化",
                    "description": "自动化企业业务流程，提高效率",
                    "scenario_type": "BUSINESS_AUTOMATION",
                    "business_value": "CRITICAL",
                    "technical_complexity": "MEDIUM",
                    "implementation_effort": "MEDIUM",
                    "risk_level": "LOW",
                    "stakeholders": ["业务部门", "IT部门", "管理层"],
                    "success_metrics": ["流程效率", "错误率", "成本控制"]
                }
            ],
            "customer_queries": [
                "我忘记了密码，怎么重置？",
                "你们的退货政策是什么？",
                "你们的产品质量太差了，我要投诉！",
                "谢谢你的帮助！",
                "如何联系人工客服？",
                "我的订单什么时候能发货？"
            ],
            "code_requests": [
                {
                    "type": "code_generation",
                    "description": "创建一个计算两个数之和的函数",
                    "language": "PYTHON"
                },
                {
                    "type": "code_review",
                    "code": """
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
                    "language": "PYTHON",
                    "filename": "sample.py"
                },
                {
                    "type": "documentation",
                    "code": """
def calculate_sum(a, b):
    result = a + b
    return result
""",
                    "language": "PYTHON",
                    "filename": "math.py",
                    "doc_type": "CODE_COMMENTS"
                },
                {
                    "type": "refactoring",
                    "code": """
def long_function():
    # 这是一个很长的函数，包含很多逻辑
    data = get_data()
    processed_data = process_data(data)
    result = calculate_result(processed_data)
    return result
"""
                }
            ],
            "business_processes": [
                {
                    "name": "员工入职流程",
                    "variables": {
                        "employee_name": "张三",
                        "department": "技术部",
                        "position": "软件工程师",
                        "start_date": "2025-01-01"
                    }
                },
                {
                    "name": "采购审批流程",
                    "variables": {
                        "item": "办公设备",
                        "amount": 15000,
                        "department": "行政部",
                        "urgency": "normal",
                        "department_budget": 50000
                    }
                }
            ],
            "applications": [
                {
                    "name": "智能客服系统",
                    "version": "1.0.0",
                    "description": "基于AI的智能客服系统",
                    "image": "customer-service:1.0.0",
                    "ports": [8080, 8081],
                    "environment_variables": {
                        "ENV": "production",
                        "LOG_LEVEL": "INFO"
                    },
                    "resources": {
                        "cpu": "500m",
                        "memory": "1Gi"
                    }
                },
                {
                    "name": "代码助手服务",
                    "version": "1.0.0",
                    "description": "AI代码助手服务",
                    "image": "code-assistant:1.0.0",
                    "ports": [8082],
                    "environment_variables": {
                        "ENV": "production",
                        "LOG_LEVEL": "INFO"
                    },
                    "resources": {
                        "cpu": "1000m",
                        "memory": "2Gi"
                    }
                }
            ]
        }
    
    async def initialize_systems(self):
        """初始化所有系统"""
        print("🚀 初始化企业级智能体应用系统...")
        
        # 初始化场景分析器
        self.scenario_analyzer = EnterpriseScenarioAnalyzer({
            "requirements": {},
            "impact": {},
            "recommendations": {}
        })
        
        # 初始化智能客服系统
        self.customer_service = CustomerServiceSystem({
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
        })
        
        # 初始化代码助手
        self.code_assistant = CodeAssistant({
            "analysis": {},
            "generation": {},
            "testing": {},
            "documentation": {}
        })
        
        # 初始化业务流程自动化系统
        self.business_automation = BusinessAutomationSystem({
            "workflow": {},
            "rules": {}
        })
        await self.business_automation.start()
        
        # 初始化部署运维系统
        self.deployment_ops = DeploymentOpsSystem({
            "container": {},
            "monitoring": {"monitoring_interval": 5},
            "logging": {"log_storage": "file"}
        })
        await self.deployment_ops.start()
        
        print("✅ 所有系统初始化完成")
    
    async def cleanup_systems(self):
        """清理系统"""
        print("\n🧹 清理系统资源...")
        
        if self.business_automation:
            await self.business_automation.stop()
        
        if self.deployment_ops:
            await self.deployment_ops.stop()
        
        print("✅ 系统清理完成")
    
    async def demo_scenario_analysis(self):
        """演示场景分析"""
        print("\n" + "="*60)
        print("📊 企业应用场景分析演示")
        print("="*60)
        
        scenarios = []
        
        for scenario_data in self.demo_data["scenarios"]:
            print(f"\n🔍 分析场景: {scenario_data['name']}")
            scenario = await self.scenario_analyzer.analyze_scenario(scenario_data)
            scenarios.append(scenario)
            
            print(f"  ✓ 业务价值: {scenario.business_value.value}")
            print(f"  ✓ 技术复杂度: {scenario.technical_complexity.value}")
            print(f"  ✓ 实施难度: {scenario.implementation_effort.value}")
            print(f"  ✓ 业务需求数: {len(scenario.business_requirements)}")
            print(f"  ✓ 技术需求数: {len(scenario.technical_requirements)}")
            print(f"  ✓ 业务影响数: {len(scenario.business_impacts)}")
            print(f"  ✓ 实施建议数: {len(scenario.recommendations)}")
        
        # 场景比较
        print(f"\n📈 场景比较分析:")
        scenario_ids = [s.id for s in scenarios]
        comparison = await self.scenario_analyzer.compare_scenarios(scenario_ids)
        
        print("  比较矩阵:")
        for metric, values in comparison["comparison_matrix"].items():
            print(f"    {metric}:")
            for scenario_id, value in values.items():
                scenario_name = next(s.name for s in scenarios if s.id == scenario_id)
                print(f"      {scenario_name}: {value}")
        
        print("\n  比较建议:")
        for rec in comparison["recommendations"]:
            print(f"    - {rec}")
        
        # 统计信息
        print(f"\n📊 分析统计:")
        stats = self.scenario_analyzer.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    async def demo_customer_service(self):
        """演示智能客服系统"""
        print("\n" + "="*60)
        print("🤖 智能客服系统演示")
        print("="*60)
        
        print("\n💬 客户对话演示:")
        
        for i, query in enumerate(self.demo_data["customer_queries"], 1):
            print(f"\n{i}. 客户查询: {query}")
            
            response = await self.customer_service.handle_customer_query(
                f"customer_{i:03d}",
                query,
                "web"
            )
            
            print(f"   🤖 系统回复: {response['response']}")
            print(f"   📊 置信度: {response['confidence']:.2f}")
            print(f"   📋 状态: {response['status']}")
            
            if response.get('agent_id'):
                print(f"   👤 分配客服: {response['agent_id']}")
            
            # 模拟对话间隔
            await asyncio.sleep(0.5)
        
        # 显示分析数据
        print(f"\n📈 客服系统分析数据:")
        analytics = self.customer_service.get_analytics()
        for key, value in analytics.items():
            print(f"  {key}: {value}")
    
    async def demo_code_assistant(self):
        """演示代码助手"""
        print("\n" + "="*60)
        print("💻 代码助手演示")
        print("="*60)
        
        for i, request in enumerate(self.demo_data["code_requests"], 1):
            print(f"\n{i}. {request['type']} 演示:")
            
            if request["type"] == "code_generation":
                print(f"   📝 描述: {request['description']}")
                print(f"   🔧 语言: {request['language']}")
                
                result = await self.code_assistant.assist_development(request)
                
                if result["status"] == "success":
                    print("   ✅ 生成的代码:")
                    print("   " + "\n   ".join(result["data"]["generated_code"].split("\n")))
                    print(f"   🧪 生成的测试用例数: {len(result['data']['test_cases'])}")
                else:
                    print(f"   ❌ 生成失败: {result.get('error', '未知错误')}")
            
            elif request["type"] == "code_review":
                print(f"   📁 文件: {request['filename']}")
                print(f"   🔧 语言: {request['language']}")
                
                result = await self.code_assistant.assist_development(request)
                
                if result["status"] == "success":
                    print(f"   🔍 发现的问题数: {len(result['data']['issues'])}")
                    for issue in result["data"]["issues"]:
                        print(f"     - {issue['severity']}: {issue['message']} (行 {issue['line_number']})")
                    
                    metrics = result["data"]["metrics"]
                    print(f"   📊 代码指标:")
                    print(f"     圈复杂度: {metrics['cyclomatic_complexity']}")
                    print(f"     代码异味: {metrics['code_smells']}")
                    print(f"     可维护性指数: {metrics['maintainability_index']:.2f}")
                else:
                    print(f"   ❌ 审查失败: {result.get('error', '未知错误')}")
            
            elif request["type"] == "documentation":
                print(f"   📁 文件: {request['filename']}")
                print(f"   📚 文档类型: {request['doc_type']}")
                
                result = await self.code_assistant.assist_development(request)
                
                if result["status"] == "success":
                    print("   ✅ 生成的文档:")
                    doc_content = result["data"]["documentation"]["content"]
                    print("   " + "\n   ".join(doc_content[:200].split("\n")))
                    if len(doc_content) > 200:
                        print("   ... (文档内容已截断)")
                else:
                    print(f"   ❌ 文档生成失败: {result.get('error', '未知错误')}")
            
            elif request["type"] == "refactoring":
                print("   🔧 重构建议:")
                
                result = await self.code_assistant.assist_development(request)
                
                if result["status"] == "success":
                    suggestions = result["data"]["refactoring_suggestions"]
                    if suggestions:
                        for suggestion in suggestions:
                            print(f"     - {suggestion}")
                    else:
                        print("     - 代码质量良好，无需重构")
                else:
                    print(f"   ❌ 重构分析失败: {result.get('error', '未知错误')}")
            
            # 模拟处理间隔
            await asyncio.sleep(0.5)
    
    async def demo_business_automation(self):
        """演示业务流程自动化"""
        print("\n" + "="*60)
        print("🔄 业务流程自动化演示")
        print("="*60)
        
        # 显示可用的流程模板
        print("\n📋 可用的流程模板:")
        templates = self.business_automation.get_process_templates()
        for template in templates:
            print(f"  - {template.name}: {template.description}")
            print(f"    节点数: {len(template.nodes)}, 边数: {len(template.edges)}")
        
        # 启动业务流程
        print(f"\n🚀 启动业务流程:")
        
        for i, process_data in enumerate(self.demo_data["business_processes"], 1):
            print(f"\n{i}. 启动流程: {process_data['name']}")
            print(f"   📊 流程变量: {process_data['variables']}")
            
            instance = await self.business_automation.start_process(
                process_data["name"],
                process_data["variables"],
                "演示系统"
            )
            
            if instance:
                print(f"   ✅ 流程已启动: {instance.id}")
                print(f"   📋 状态: {instance.status.value}")
                print(f"   🎯 当前节点: {instance.current_node_id}")
                
                # 等待流程执行
                await asyncio.sleep(1.0)
                
                # 检查最终状态
                updated_instance = self.business_automation.get_process_instance(instance.id)
                if updated_instance:
                    print(f"   📊 最终状态: {updated_instance.status.value}")
                    if updated_instance.completed_at:
                        print(f"   ⏰ 完成时间: {updated_instance.completed_at}")
            else:
                print(f"   ❌ 流程启动失败")
        
        # 显示系统统计
        print(f"\n📈 业务流程统计:")
        stats = self.business_automation.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    async def demo_deployment_ops(self):
        """演示部署运维"""
        print("\n" + "="*60)
        print("🚀 部署运维演示")
        print("="*60)
        
        # 注册和部署应用程序
        print(f"\n📦 应用程序部署:")
        
        for i, app_data in enumerate(self.demo_data["applications"], 1):
            print(f"\n{i}. 部署应用: {app_data['name']}")
            
            # 创建应用程序对象
            app = Application(
                name=app_data["name"],
                version=app_data["version"],
                description=app_data["description"],
                image=app_data["image"],
                ports=app_data["ports"],
                environment_variables=app_data["environment_variables"],
                resources=app_data["resources"]
            )
            
            # 注册应用程序
            await self.deployment_ops.register_application(app)
            print(f"   ✅ 应用程序已注册: {app.name} v{app.version}")
            
            # 部署应用程序
            deployment = await self.deployment_ops.deploy_application(
                app.id,
                DeployEnvType.PRODUCTION,
                deployed_by="演示系统"
            )
            
            if deployment:
                print(f"   🚀 部署已启动: {deployment.id}")
                print(f"   📋 状态: {deployment.status.value}")
                print(f"   🌍 环境: {deployment.environment.value}")
                
                # 等待部署完成
                await asyncio.sleep(1.0)
                
                # 检查部署状态
                updated_deployment = self.deployment_ops.get_deployment_status(deployment.id)
                if updated_deployment:
                    print(f"   📊 最终状态: {updated_deployment.status.value}")
                    if updated_deployment.completed_at:
                        print(f"   ⏰ 完成时间: {updated_deployment.completed_at}")
                
                # 服务扩缩容演示
                services = list(self.deployment_ops.services.values())
                if services:
                    service = services[-1]  # 使用最后一个服务
                    print(f"   📈 服务扩缩容演示:")
                    print(f"     当前副本数: {service.replicas}")
                    
                    # 扩容到3个副本
                    await self.deployment_ops.scale_service(service.id, 3)
                    updated_service = self.deployment_ops.get_service_status(service.id)
                    if updated_service:
                        print(f"     扩容后副本数: {updated_service.replicas}")
            else:
                print(f"   ❌ 部署失败")
        
        # 等待监控数据收集
        print(f"\n📊 等待监控数据收集...")
        await asyncio.sleep(2.0)
        
        # 显示系统指标
        print(f"\n📈 系统指标:")
        metrics = self.deployment_ops.monitoring_system.get_metrics("system")
        if metrics:
            for metric in metrics[-3:]:  # 显示最近3个指标
                print(f"  {metric.name}: {metric.value} {metric.unit}")
        
        # 显示告警信息
        print(f"\n🚨 告警信息:")
        alerts = self.deployment_ops.monitoring_system.get_alerts(status="active")
        if alerts:
            for alert in alerts:
                print(f"  {alert.level.value}: {alert.message}")
        else:
            print("  ✅ 无活跃告警")
        
        # 显示系统统计
        print(f"\n📊 部署运维统计:")
        system_stats = self.deployment_ops.get_system_metrics()
        for key, value in system_stats.items():
            print(f"  {key}: {value}")
        
        # 显示告警摘要
        print(f"\n🚨 告警摘要:")
        alert_summary = self.deployment_ops.get_alerts_summary()
        for key, value in alert_summary.items():
            print(f"  {key}: {value}")
    
    async def demo_integration_scenario(self):
        """演示集成场景"""
        print("\n" + "="*60)
        print("🔗 集成场景演示")
        print("="*60)
        
        print("\n🎯 场景: 企业智能客服系统完整部署")
        
        # 1. 场景分析
        print("\n1️⃣ 场景分析阶段")
        scenario_data = self.demo_data["scenarios"][0]  # 智能客服场景
        scenario = await self.scenario_analyzer.analyze_scenario(scenario_data)
        print(f"   ✅ 场景分析完成: {scenario.name}")
        print(f"   📊 业务价值: {scenario.business_value.value}")
        print(f"   🔧 技术复杂度: {scenario.technical_complexity.value}")
        
        # 2. 客服系统测试
        print("\n2️⃣ 客服系统测试阶段")
        test_response = await self.customer_service.handle_customer_query(
            "integration_test",
            "这是集成测试查询",
            "web"
        )
        print(f"   ✅ 客服系统测试完成")
        print(f"   📊 响应置信度: {test_response['confidence']:.2f}")
        
        # 3. 代码生成
        print("\n3️⃣ 代码生成阶段")
        code_result = await self.code_assistant.assist_development({
            "type": "code_generation",
            "description": "创建一个智能客服系统的核心类",
            "language": "PYTHON"
        })
        print(f"   ✅ 代码生成完成")
        print(f"   📝 生成代码长度: {len(code_result['data']['generated_code'])} 字符")
        
        # 4. 业务流程启动
        print("\n4️⃣ 业务流程启动阶段")
        process_instance = await self.business_automation.start_process(
            "员工入职流程",
            {"employee_name": "客服专员", "department": "客服部"},
            "集成测试系统"
        )
        print(f"   ✅ 业务流程启动完成")
        if process_instance:
            print(f"   📋 流程状态: {process_instance.status.value}")
        
        # 5. 应用部署
        print("\n5️⃣ 应用部署阶段")
        app = Application(
            name="集成测试应用",
            version="1.0.0",
            description="集成测试应用",
            image="integration-test:1.0.0",
            ports=[8080]
        )
        
        await self.deployment_ops.register_application(app)
        deployment = await self.deployment_ops.deploy_application(
            app.id,
            DeployEnvType.PRODUCTION
        )
        print(f"   ✅ 应用部署完成")
        if deployment:
            print(f"   📋 部署状态: {deployment.status.value}")
        
        print(f"\n🎉 集成场景演示完成！")
        print(f"   📊 所有系统协同工作，展示了完整的企业级智能体应用部署流程")
    
    async def run_demo(self):
        """运行完整演示"""
        try:
            print("🌟 企业级智能体应用综合演示")
            print("="*80)
            print("本演示将展示企业级智能体应用的完整功能和工作流程")
            print("包括：场景分析、智能客服、代码助手、业务流程自动化、部署运维")
            print("="*80)
            
            # 初始化系统
            await self.initialize_systems()
            
            # 运行各个模块演示
            await self.demo_scenario_analysis()
            await self.demo_customer_service()
            await self.demo_code_assistant()
            await self.demo_business_automation()
            await self.demo_deployment_ops()
            await self.demo_integration_scenario()
            
            # 最终总结
            print("\n" + "="*80)
            print("🎊 演示总结")
            print("="*80)
            print("✅ 企业应用场景分析 - 完成")
            print("✅ 智能客服系统 - 完成")
            print("✅ 代码助手 - 完成")
            print("✅ 业务流程自动化 - 完成")
            print("✅ 部署运维系统 - 完成")
            print("✅ 集成场景演示 - 完成")
            print("\n🚀 企业级智能体应用演示全部完成！")
            print("💡 这些系统可以独立使用，也可以集成使用，为企业提供完整的智能体解决方案")
            
        except Exception as e:
            print(f"\n❌ 演示过程中发生错误: {e}")
            logger.error(f"Demo error: {e}")
        finally:
            # 清理系统
            await self.cleanup_systems()

async def main():
    """主函数"""
    demo = EnterpriseAgentDemo()
    await demo.run_demo()

if __name__ == "__main__":
    asyncio.run(main())
