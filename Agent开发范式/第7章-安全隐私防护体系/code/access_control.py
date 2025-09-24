# access_control.py
"""
第7章 安全隐私防护体系 - 权限控制与访问管理系统
实现RBAC和ABAC权限控制功能
"""

import logging
from typing import Dict, Any, List, Set, Optional
from enum import Enum
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResourceType(Enum):
    """资源类型"""
    LLM_MODEL = "LLM模型"
    MEMORY_SYSTEM = "记忆系统"
    TOOL = "工具"
    AGENT = "智能体"
    DATA_SOURCE = "数据源"
    API_ENDPOINT = "API端点"
    FILE_SYSTEM = "文件系统"
    CONFIGURATION = "配置"
    OTHER = "其他"

class PermissionAction(Enum):
    """权限操作"""
    READ = "读取"
    WRITE = "写入"
    EXECUTE = "执行"
    CREATE = "创建"
    DELETE = "删除"
    UPDATE = "更新"
    INVOKE = "调用"
    MANAGE = "管理"
    ACCESS = "访问"
    NONE = "无"

@dataclass
class Permission:
    """权限定义"""
    resource_type: ResourceType
    resource_id: Optional[str] # Specific ID, e.g., "tool_search_web"
    action: PermissionAction

    def __hash__(self):
        return hash((self.resource_type, self.resource_id, self.action))

    def __eq__(self, other):
        return isinstance(other, Permission) and \
               self.resource_type == other.resource_type and \
               self.resource_id == other.resource_id and \
               self.action == other.action

    def to_dict(self) -> Dict[str, str]:
        return {
            "resource_type": self.resource_type.value,
            "resource_id": self.resource_id if self.resource_id else "any",
            "action": self.action.value
        }

@dataclass
class Role:
    """角色定义"""
    role_id: str
    name: str
    description: str
    permissions: Set[Permission] = field(default_factory=set)

    def add_permission(self, permission: Permission):
        self.permissions.add(permission)
        logger.debug(f"Permission {permission.to_dict()} added to role {self.name}")

    def remove_permission(self, permission: Permission):
        self.permissions.discard(permission)
        logger.debug(f"Permission {permission.to_dict()} removed from role {self.name}")

    def has_permission(self, permission: Permission) -> bool:
        # Check for exact match or broader permission (e.g., if resource_id is None, it means any resource of that type)
        for p in self.permissions:
            if p.resource_type == permission.resource_type and \
               p.action == permission.action and \
               (p.resource_id is None or p.resource_id == permission.resource_id):
                return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role_id": self.role_id,
            "name": self.name,
            "description": self.description,
            "permissions": [p.to_dict() for p in self.permissions]
        }

@dataclass
class User:
    """用户（或智能体）定义"""
    user_id: str
    name: str
    roles: Set[str] = field(default_factory=set) # Set of role_ids
    attributes: Dict[str, Any] = field(default_factory=dict) # For ABAC

    def add_role(self, role_id: str):
        self.roles.add(role_id)
        logger.debug(f"Role {role_id} added to user {self.name}")

    def remove_role(self, role_id: str):
        self.roles.discard(role_id)
        logger.debug(f"Role {role_id} removed from user {self.name}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "roles": list(self.roles),
            "attributes": self.attributes
        }

class AccessControlSystem:
    """
    权限控制与访问管理系统，支持RBAC和ABAC。
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config if config is not None else {}
        self.roles: Dict[str, Role] = {}
        self.users: Dict[str, User] = {}
        self._load_default_configs()
        logger.info("AccessControlSystem initialized.")

    def _load_default_configs(self):
        """加载默认角色和用户"""
        # Default Permissions
        read_llm_perm = Permission(ResourceType.LLM_MODEL, None, PermissionAction.READ)
        invoke_tool_search = Permission(ResourceType.TOOL, "search_web", PermissionAction.INVOKE)
        manage_memory_perm = Permission(ResourceType.MEMORY_SYSTEM, None, PermissionAction.MANAGE)
        execute_agent_task = Permission(ResourceType.AGENT, None, PermissionAction.EXECUTE)
        access_db_read = Permission(ResourceType.DATA_SOURCE, "customer_db", PermissionAction.READ)
        access_db_write = Permission(ResourceType.DATA_SOURCE, "customer_db", PermissionAction.WRITE)

        # Default Roles
        admin_role = Role("admin", "管理员", "拥有系统所有管理权限")
        admin_role.add_permission(Permission(ResourceType.OTHER, None, PermissionAction.MANAGE)) # All management
        self.add_role(admin_role)

        agent_executor_role = Role("agent_executor", "智能体执行者", "可以执行智能体任务和调用工具")
        agent_executor_role.add_permission(execute_agent_task)
        agent_executor_role.add_permission(invoke_tool_search)
        self.add_role(agent_executor_role)

        data_analyst_role = Role("data_analyst", "数据分析师", "可以读取数据源")
        data_analyst_role.add_permission(access_db_read)
        self.add_role(data_analyst_role)

        # Default Users (Agents or Human Users)
        self.add_user(User("user_alice", "Alice", {"admin"}))
        self.add_user(User("agent_cs", "客服智能体", {"agent_executor"}, {"department": "CustomerService", "level": "L1"}))
        self.add_user(User("agent_data", "数据分析智能体", {"data_analyst"}, {"department": "DataAnalytics", "level": "L2"}))
        logger.info("Loaded default roles and users.")

    def add_role(self, role: Role):
        """添加或更新一个角色"""
        self.roles[role.role_id] = role
        logger.info(f"Added role: {role.name} ({role.role_id})")

    def get_role(self, role_id: str) -> Optional[Role]:
        """根据ID获取角色"""
        return self.roles.get(role_id)

    def add_user(self, user: User):
        """添加或更新一个用户"""
        self.users[user.user_id] = user
        logger.info(f"Added user: {user.name} ({user.user_id})")

    def get_user(self, user_id: str) -> Optional[User]:
        """根据ID获取用户"""
        return self.users.get(user_id)

    async def check_permission_rbac(self, user_id: str, permission: Permission) -> bool:
        """
        基于角色的访问控制 (RBAC) 检查。
        检查用户是否通过其角色拥有指定权限。
        """
        user = self.get_user(user_id)
        if not user:
            logger.warning(f"User {user_id} not found.")
            return False

        for role_id in user.roles:
            role = self.get_role(role_id)
            if role and role.has_permission(permission):
                logger.debug(f"User {user_id} has permission {permission.to_dict()} via role {role.name}")
                return True
        logger.warning(f"User {user_id} does not have permission {permission.to_dict()} via RBAC.")
        return False

    async def check_permission_abac(self, user_id: str, permission: Permission, context: Dict[str, Any]) -> bool:
        """
        基于属性的访问控制 (ABAC) 检查。
        检查用户、资源、环境属性是否满足权限策略。
        这是一个简化的示例，实际ABAC策略会更复杂。
        """
        user = self.get_user(user_id)
        if not user:
            logger.warning(f"User {user_id} not found for ABAC check.")
            return False

        # 示例ABAC策略：
        # 1. 只有客服部门的L1智能体可以访问客户数据库进行读取
        if permission.resource_type == ResourceType.DATA_SOURCE and \
           permission.resource_id == "customer_db" and \
           permission.action == PermissionAction.READ:
            if user.attributes.get("department") == "CustomerService" and \
               user.attributes.get("level") == "L1":
                logger.debug(f"ABAC: User {user_id} (CustomerService L1) granted read access to customer_db.")
                return True
        
        # 2. 只有管理员可以管理配置
        if permission.resource_type == ResourceType.CONFIGURATION and \
           permission.action == PermissionAction.MANAGE:
            if "admin" in user.roles: # ABAC可以结合RBAC
                logger.debug(f"ABAC: User {user_id} (Admin) granted manage access to configuration.")
                return True

        # 3. 只有在工作时间 (假设context中传入) 才能执行某些敏感工具
        if permission.resource_type == ResourceType.TOOL and \
           permission.resource_id == "sensitive_tool" and \
           permission.action == PermissionAction.INVOKE:
            current_hour = context.get("current_hour")
            if current_hour and 9 <= current_hour <= 17: # 9 AM to 5 PM
                logger.debug(f"ABAC: User {user_id} granted invoke access to sensitive_tool during working hours.")
                return True

        logger.warning(f"User {user_id} does not have permission {permission.to_dict()} via ABAC with context {context}.")
        return False

    async def authorize(self, user_id: str, permission: Permission, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        综合授权检查，结合RBAC和ABAC。
        """
        context = context if context is not None else {}
        
        # 首先进行RBAC检查
        if await self.check_permission_rbac(user_id, permission):
            return True
        
        # 如果RBAC不通过，再进行ABAC检查
        if await self.check_permission_abac(user_id, permission, context):
            return True
        
        logger.warning(f"Authorization failed for user {user_id} on permission {permission.to_dict()}.")
        return False

async def main_demo():
    acs = AccessControlSystem()

    print("\n--- RBAC 演示 ---")
    # 管理员权限检查
    admin_manage_config = Permission(ResourceType.CONFIGURATION, None, PermissionAction.MANAGE)
    print(f"Alice (admin) can manage config: {await acs.authorize('user_alice', admin_manage_config)}")

    # 客服智能体执行任务
    agent_execute_task = Permission(ResourceType.AGENT, "customer_service_agent", PermissionAction.EXECUTE)
    print(f"客服智能体可以执行任务: {await acs.authorize('agent_cs', agent_execute_task)}")

    # 数据分析师读取客户数据库
    data_analyst_read_db = Permission(ResourceType.DATA_SOURCE, "customer_db", PermissionAction.READ)
    print(f"数据分析智能体可以读取客户数据库: {await acs.authorize('agent_data', data_analyst_read_db)}")

    # 客服智能体尝试写入客户数据库 (应失败)
    agent_cs_write_db = Permission(ResourceType.DATA_SOURCE, "customer_db", PermissionAction.WRITE)
    print(f"客服智能体可以写入客户数据库 (预期失败): {await acs.authorize('agent_cs', agent_cs_write_db)}")

    print("\n--- ABAC 演示 ---")
    # 客服智能体在工作时间调用敏感工具 (假设敏感工具需要ABAC)
    sensitive_tool_invoke = Permission(ResourceType.TOOL, "sensitive_tool", PermissionAction.INVOKE)
    
    # 模拟工作时间
    context_working_hours = {"current_hour": 10}
    print(f"客服智能体在工作时间调用敏感工具 (预期成功): {await acs.authorize('agent_cs', sensitive_tool_invoke, context_working_hours)}")

    # 模拟非工作时间
    context_off_hours = {"current_hour": 20}
    print(f"客服智能体在非工作时间调用敏感工具 (预期失败): {await acs.authorize('agent_cs', sensitive_tool_invoke, context_off_hours)}")

    # 创建一个没有权限的新用户
    acs.add_user(User("new_user", "Bob"))
    print(f"Bob (new_user) 可以执行任务 (预期失败): {await acs.authorize('new_user', agent_execute_task)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main_demo())
