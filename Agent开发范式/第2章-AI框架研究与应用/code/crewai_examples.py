# crewai_examples.py
"""
CrewAI框架示例代码
展示CrewAI多智能体协作的核心功能
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# CrewAI核心组件
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from crewai.memory import ShortTermMemory, LongTermMemory
from crewai.llm import LLM
from crewai.process import Process

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrewAIExamples:
    """CrewAI示例集合"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OpenAI API key not found. Some examples may not work.")
        
        self.llm = LLM(
            model="gpt-3.5-turbo",
            api_key=self.api_key,
            temperature=0.7
        )
    
    def basic_crew_example(self) -> str:
        """基础团队示例"""
        logger.info("Running basic crew example...")
        
        # 创建智能体
        researcher = Agent(
            role='Research Analyst',
            goal='Gather and analyze information about AI trends',
            backstory='You are an expert research analyst with 10 years of experience in technology research.',
            verbose=True,
            allow_delegation=False
        )
        
        writer = Agent(
            role='Content Writer',
            goal='Create engaging and informative content',
            backstory='You are a skilled content writer who specializes in technology articles.',
            verbose=True,
            allow_delegation=False
        )
        
        # 创建任务
        research_task = Task(
            description='Research the latest trends in artificial intelligence for 2024',
            agent=researcher,
            expected_output='A comprehensive report on AI trends including key technologies, market growth, and future predictions'
        )
        
        writing_task = Task(
            description='Write an engaging article based on the research findings',
            agent=writer,
            expected_output='A well-structured article (500-800 words) about AI trends that is informative and engaging for general readers'
        )
        
        # 创建团队
        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, writing_task],
            verbose=True,
            process=Process.sequential
        )
        
        # 执行任务
        result = crew.kickoff()
        
        return str(result)
    
    def hierarchical_crew_example(self) -> str:
        """层次化团队示例"""
        logger.info("Running hierarchical crew example...")
        
        # 创建管理者智能体
        manager = Agent(
            role='Project Manager',
            goal='Coordinate and oversee the project execution',
            backstory='You are an experienced project manager who excels at coordinating teams and ensuring project success.',
            verbose=True,
            allow_delegation=True
        )
        
        # 创建专业智能体
        data_scientist = Agent(
            role='Data Scientist',
            goal='Analyze data and provide insights',
            backstory='You are a data scientist with expertise in machine learning and statistical analysis.',
            verbose=True,
            allow_delegation=False
        )
        
        business_analyst = Agent(
            role='Business Analyst',
            goal='Provide business insights and recommendations',
            backstory='You are a business analyst who understands market dynamics and business strategy.',
            verbose=True,
            allow_delegation=False
        )
        
        # 创建任务
        data_analysis_task = Task(
            description='Analyze customer data to identify patterns and trends',
            agent=data_scientist,
            expected_output='A detailed analysis report with key findings and data visualizations'
        )
        
        business_analysis_task = Task(
            description='Analyze business performance and provide strategic recommendations',
            agent=business_analyst,
            expected_output='A business analysis report with strategic recommendations and action items'
        )
        
        management_task = Task(
            description='Review all analysis reports and create a comprehensive project summary',
            agent=manager,
            expected_output='A comprehensive project summary that integrates all findings and provides executive recommendations'
        )
        
        # 创建层次化团队
        crew = Crew(
            agents=[manager, data_scientist, business_analyst],
            tasks=[data_analysis_task, business_analysis_task, management_task],
            verbose=True,
            process=Process.hierarchical,
            manager_agent=manager
        )
        
        # 执行任务
        result = crew.kickoff()
        
        return str(result)
    
    def parallel_crew_example(self) -> str:
        """并行团队示例"""
        logger.info("Running parallel crew example...")
        
        # 创建并行智能体
        market_researcher = Agent(
            role='Market Researcher',
            goal='Research market trends and competitive landscape',
            backstory='You are a market research expert who specializes in competitive analysis.',
            verbose=True,
            allow_delegation=False
        )
        
        technology_analyst = Agent(
            role='Technology Analyst',
            goal='Analyze technology trends and innovations',
            backstory='You are a technology analyst who stays updated with the latest tech developments.',
            verbose=True,
            allow_delegation=False
        )
        
        financial_analyst = Agent(
            role='Financial Analyst',
            goal='Analyze financial data and market performance',
            backstory='You are a financial analyst who specializes in market performance analysis.',
            verbose=True,
            allow_delegation=False
        )
        
        # 创建并行任务
        market_task = Task(
            description='Research the current market trends in the AI industry',
            agent=market_researcher,
            expected_output='A market research report covering current trends, growth rates, and market size'
        )
        
        technology_task = Task(
            description='Analyze the latest technology innovations in AI',
            agent=technology_analyst,
            expected_output='A technology analysis report covering recent innovations, breakthroughs, and emerging technologies'
        )
        
        financial_task = Task(
            description='Analyze the financial performance of AI companies',
            agent=financial_analyst,
            expected_output='A financial analysis report covering company performance, stock trends, and investment opportunities'
        )
        
        # 创建并行团队
        crew = Crew(
            agents=[market_researcher, technology_analyst, financial_analyst],
            tasks=[market_task, technology_task, financial_task],
            verbose=True,
            process=Process.parallel
        )
        
        # 执行任务
        result = crew.kickoff()
        
        return str(result)
    
    def custom_tool_example(self) -> str:
        """自定义工具示例"""
        logger.info("Running custom tool example...")
        
        class WeatherTool(BaseTool):
            """天气查询工具"""
            name: str = "weather_tool"
            description: str = "Get weather information for a specific city"
            
            def _run(self, city: str) -> str:
                # 模拟天气API调用
                weather_data = {
                    "Beijing": "Sunny, 25°C, Humidity: 60%",
                    "Shanghai": "Cloudy, 22°C, Humidity: 70%",
                    "Guangzhou": "Rainy, 28°C, Humidity: 80%",
                    "Shenzhen": "Sunny, 30°C, Humidity: 65%"
                }
                return weather_data.get(city, f"Weather data not available for {city}")
        
        class CalculatorTool(BaseTool):
            """计算器工具"""
            name: str = "calculator_tool"
            description: str = "Perform mathematical calculations"
            
            def _run(self, expression: str) -> str:
                try:
                    result = eval(expression)
                    return str(result)
                except Exception as e:
                    return f"Error: {e}"
        
        # 创建带工具的智能体
        weather_agent = Agent(
            role='Weather Reporter',
            goal='Provide accurate weather information',
            backstory='You are a professional weather reporter with access to real-time weather data.',
            tools=[WeatherTool()],
            verbose=True,
            allow_delegation=False
        )
        
        calculator_agent = Agent(
            role='Mathematical Assistant',
            goal='Help with mathematical calculations',
            backstory='You are a mathematical assistant who can perform various calculations.',
            tools=[CalculatorTool()],
            verbose=True,
            allow_delegation=False
        )
        
        # 创建任务
        weather_task = Task(
            description='Get the current weather for Beijing and provide a brief weather report',
            agent=weather_agent,
            expected_output='A weather report for Beijing including temperature, conditions, and humidity'
        )
        
        calculation_task = Task(
            description='Calculate the average temperature for the next 7 days if today is 25°C and it increases by 2°C each day',
            agent=calculator_agent,
            expected_output='The calculated average temperature for the next 7 days'
        )
        
        # 创建团队
        crew = Crew(
            agents=[weather_agent, calculator_agent],
            tasks=[weather_task, calculation_task],
            verbose=True,
            process=Process.sequential
        )
        
        # 执行任务
        result = crew.kickoff()
        
        return str(result)
    
    def memory_example(self) -> str:
        """记忆系统示例"""
        logger.info("Running memory example...")
        
        # 创建带记忆的智能体
        researcher = Agent(
            role='Research Assistant',
            goal='Conduct research and remember important information',
            backstory='You are a research assistant who learns from previous research sessions.',
            verbose=True,
            allow_delegation=False,
            memory=True
        )
        
        # 创建任务
        research_task = Task(
            description='Research the history of artificial intelligence',
            agent=researcher,
            expected_output='A comprehensive overview of AI history including key milestones and developments'
        )
        
        follow_up_task = Task(
            description='Based on your previous research, what are the most significant AI breakthroughs?',
            agent=researcher,
            expected_output='A summary of the most significant AI breakthroughs based on previous research'
        )
        
        # 创建团队
        crew = Crew(
            agents=[researcher],
            tasks=[research_task, follow_up_task],
            verbose=True,
            process=Process.sequential
        )
        
        # 执行任务
        result = crew.kickoff()
        
        return str(result)
    
    def delegation_example(self) -> str:
        """任务委派示例"""
        logger.info("Running delegation example...")
        
        # 创建管理者智能体
        manager = Agent(
            role='Project Manager',
            goal='Manage the project and delegate tasks effectively',
            backstory='You are an experienced project manager who knows how to delegate tasks to the right team members.',
            verbose=True,
            allow_delegation=True
        )
        
        # 创建专业智能体
        developer = Agent(
            role='Software Developer',
            goal='Develop software solutions',
            backstory='You are a skilled software developer who specializes in AI applications.',
            verbose=True,
            allow_delegation=False
        )
        
        tester = Agent(
            role='Quality Assurance Tester',
            goal='Ensure software quality through testing',
            backstory='You are a QA tester who ensures software meets quality standards.',
            verbose=True,
            allow_delegation=False
        )
        
        # 创建任务
        development_task = Task(
            description='Develop a simple AI chatbot application',
            agent=developer,
            expected_output='A working AI chatbot application with basic functionality'
        )
        
        testing_task = Task(
            description='Test the AI chatbot application for bugs and performance issues',
            agent=tester,
            expected_output='A comprehensive test report with identified issues and recommendations'
        )
        
        management_task = Task(
            description='Oversee the development and testing process, ensuring quality and timely delivery',
            agent=manager,
            expected_output='A project status report with progress updates and next steps'
        )
        
        # 创建团队
        crew = Crew(
            agents=[manager, developer, tester],
            tasks=[development_task, testing_task, management_task],
            verbose=True,
            process=Process.hierarchical,
            manager_agent=manager
        )
        
        # 执行任务
        result = crew.kickoff()
        
        return str(result)
    
    def error_handling_example(self) -> str:
        """错误处理示例"""
        logger.info("Running error handling example...")
        
        try:
            # 创建智能体
            agent = Agent(
                role='Test Agent',
                goal='Handle errors gracefully',
                backstory='You are a test agent designed to handle various error conditions.',
                verbose=True,
                allow_delegation=False
            )
            
            # 创建任务
            task = Task(
                description='This is a test task that might encounter errors',
                agent=agent,
                expected_output='A response indicating successful completion or error handling'
            )
            
            # 创建团队
            crew = Crew(
                agents=[agent],
                tasks=[task],
                verbose=True,
                process=Process.sequential
            )
            
            # 执行任务
            result = crew.kickoff()
            
            return f"Success: {str(result)}"
            
        except Exception as e:
            logger.error(f"Error occurred: {e}")
            return f"Error handled: {str(e)}"
    
    def performance_optimization_example(self) -> str:
        """性能优化示例"""
        logger.info("Running performance optimization example...")
        
        import time
        
        # 测试不同配置的性能
        configs = [
            {"temperature": 0.1, "max_tokens": 100},
            {"temperature": 0.7, "max_tokens": 200},
            {"temperature": 1.0, "max_tokens": 300}
        ]
        
        results = []
        for config in configs:
            start_time = time.time()
            
            # 创建优化配置的LLM
            llm = LLM(
                model="gpt-3.5-turbo",
                api_key=self.api_key,
                **config
            )
            
            # 创建智能体
            agent = Agent(
                role='Performance Test Agent',
                goal='Test performance with different configurations',
                backstory='You are a test agent for performance evaluation.',
                llm=llm,
                verbose=False,
                allow_delegation=False
            )
            
            # 创建任务
            task = Task(
                description='Provide a brief summary of artificial intelligence',
                agent=agent,
                expected_output='A concise summary of AI'
            )
            
            # 创建团队
            crew = Crew(
                agents=[agent],
                tasks=[task],
                verbose=False,
                process=Process.sequential
            )
            
            # 执行任务
            result = crew.kickoff()
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            results.append({
                "config": config,
                "execution_time": execution_time,
                "result_length": len(str(result))
            })
        
        return f"Performance test results: {results}"
    
    def async_example(self) -> str:
        """异步处理示例"""
        logger.info("Running async example...")
        
        async def async_crew_execution():
            # 创建智能体
            agent = Agent(
                role='Async Test Agent',
                goal='Test asynchronous execution',
                backstory='You are a test agent for async functionality.',
                verbose=True,
                allow_delegation=False
            )
            
            # 创建任务
            task = Task(
                description='Provide information about asynchronous programming',
                agent=agent,
                expected_output='A brief explanation of asynchronous programming concepts'
            )
            
            # 创建团队
            crew = Crew(
                agents=[agent],
                tasks=[task],
                verbose=True,
                process=Process.sequential
            )
            
            # 异步执行任务
            result = await crew.kickoff_async()
            
            return str(result)
        
        # 运行异步任务
        result = asyncio.run(async_crew_execution())
        
        return result

def run_all_examples():
    """运行所有示例"""
    print("🚀 CrewAI框架示例演示")
    print("=" * 50)
    
    # 创建示例实例
    examples = CrewAIExamples()
    
    # 定义示例列表
    example_methods = [
        ("基础团队", examples.basic_crew_example),
        ("层次化团队", examples.hierarchical_crew_example),
        ("并行团队", examples.parallel_crew_example),
        ("自定义工具", examples.custom_tool_example),
        ("记忆系统", examples.memory_example),
        ("任务委派", examples.delegation_example),
        ("错误处理", examples.error_handling_example),
        ("性能优化", examples.performance_optimization_example),
        ("异步处理", examples.async_example)
    ]
    
    results = {}
    
    for name, method in example_methods:
        try:
            print(f"\n📋 运行示例: {name}")
            print("-" * 30)
            
            result = method()
            results[name] = {
                "status": "success",
                "result": result[:200] + "..." if len(result) > 200 else result
            }
            
            print(f"✅ {name} 完成")
            
        except Exception as e:
            print(f"❌ {name} 失败: {e}")
            results[name] = {
                "status": "error",
                "error": str(e)
            }
    
    # 生成报告
    print("\n📊 示例运行报告")
    print("=" * 50)
    
    success_count = sum(1 for r in results.values() if r["status"] == "success")
    total_count = len(results)
    
    print(f"总示例数: {total_count}")
    print(f"成功数: {success_count}")
    print(f"失败数: {total_count - success_count}")
    print(f"成功率: {success_count/total_count*100:.1f}%")
    
    return results

if __name__ == "__main__":
    run_all_examples()
