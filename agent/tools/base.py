"""
Base Tool System for AI Agent
Defines the interface for all tools that the agent can use
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result of a tool execution"""
    success: bool
    output: Any
    error: Optional[str] = None
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'output': self.output,
            'error': self.error,
            'metadata': self.metadata or {}
        }

    def __str__(self) -> str:
        if self.success:
            return f"✓ {self.output}"
        else:
            return f"✗ {self.error}"


class Tool(ABC):
    """Base class for all agent tools"""

    def __init__(self):
        self.name = self.__class__.__name__
        self.execution_count = 0
        self.last_result = None

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters"""
        pass

    @abstractmethod
    def get_schema(self) -> Dict:
        """Return JSON schema for tool parameters"""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Return human-readable description of the tool"""
        pass

    def validate_params(self, params: Dict) -> bool:
        """Validate parameters against schema"""
        # Basic validation - can be enhanced with jsonschema
        schema = self.get_schema()
        required = schema.get('required', [])

        for param in required:
            if param not in params:
                logger.error(f"Missing required parameter: {param}")
                return False

        return True

    def run(self, **kwargs) -> ToolResult:
        """Wrapper that handles execution, logging, and error handling"""
        logger.info(f"🔧 Executing tool: {self.name} with params: {kwargs}")

        if not self.validate_params(kwargs):
            return ToolResult(
                success=False,
                output=None,
                error="Invalid parameters"
            )

        try:
            result = self.execute(**kwargs)
            self.execution_count += 1
            self.last_result = result

            logger.info(f"✓ {self.name} completed: {result}")
            return result

        except Exception as e:
            logger.error(f"✗ {self.name} failed: {e}", exc_info=True)
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )

    def get_llm_function_schema(self) -> Dict:
        """
        Return OpenAI function calling compatible schema
        This allows LLMs to call tools directly
        """
        schema = self.get_schema()
        return {
            "name": self.name.lower(),
            "description": self.get_description(),
            "parameters": {
                "type": "object",
                "properties": schema.get('properties', {}),
                "required": schema.get('required', [])
            }
        }


class ToolRegistry:
    """Registry for managing all available tools"""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.execution_history: List[Dict] = []

    def register(self, tool: Tool) -> None:
        """Register a new tool"""
        self.tools[tool.name.lower()] = tool
        logger.info(f"📝 Registered tool: {tool.name}")

    def register_multiple(self, tools: List[Tool]) -> None:
        """Register multiple tools at once"""
        for tool in tools:
            self.register(tool)

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self.tools.get(name.lower())

    def list_tools(self) -> List[str]:
        """List all available tool names"""
        return list(self.tools.keys())

    def get_all_schemas(self) -> List[Dict]:
        """Get LLM function schemas for all tools"""
        return [tool.get_llm_function_schema() for tool in self.tools.values()]

    def execute_tool(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool by name"""
        tool = self.get_tool(name)

        if not tool:
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool '{name}' not found"
            )

        result = tool.run(**kwargs)

        # Log to history
        self.execution_history.append({
            'tool': name,
            'params': kwargs,
            'result': result.to_dict()
        })

        return result

    def get_tool_descriptions(self) -> str:
        """Get formatted descriptions of all tools for LLM context"""
        descriptions = ["Available tools:"]
        for name, tool in self.tools.items():
            descriptions.append(f"\n{name}: {tool.get_description()}")
        return "\n".join(descriptions)
