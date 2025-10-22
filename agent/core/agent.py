"""
Main AI Agent - Autonomous task executor with planning and tools
"""

import json
import logging
from typing import List, Dict, Optional, Any
import ollama

from ..tools.base import ToolRegistry, Tool, ToolResult
from ..tools.web_tools import WebSearchTool, WebBrowserTool, WebFetchTool, YouTubeSearchTool
from ..tools.file_tools import FileReadTool, FileWriteTool, FileListTool, FileOpenTool
from ..tools.system_tools import (
    ShellCommandTool, PythonCodeTool, SystemInfoTool, ApplicationLauncherTool
)
from ..memory.agent_memory import AgentMemory
from .planner import TaskPlanner, Plan, Task, TaskStatus

logger = logging.getLogger(__name__)


class AIAgent:
    """
    Autonomous AI Agent with planning, tools, and memory
    """

    def __init__(
        self,
        model: str = "llama3.2",
        enable_planning: bool = True,
        memory_file: Optional[str] = None
    ):
        """
        Initialize AI Agent

        Args:
            model: LLM model to use
            enable_planning: Whether to use task planning
            memory_file: Path to memory persistence file
        """
        self.model = model
        self.enable_planning = enable_planning

        # Initialize components
        self.tool_registry = ToolRegistry()
        self.memory = AgentMemory(memory_file=memory_file)
        self.planner = TaskPlanner(ollama) if enable_planning else None

        # Register all tools
        self._register_tools()

        logger.info("🤖 AI Agent initialized")
        logger.info(f"   Model: {model}")
        logger.info(f"   Planning: {'enabled' if enable_planning else 'disabled'}")
        logger.info(f"   Tools: {len(self.tool_registry.list_tools())} available")

    def _register_tools(self):
        """Register all available tools"""
        tools = [
            # Web tools
            WebSearchTool(),
            WebBrowserTool(),
            WebFetchTool(),
            YouTubeSearchTool(),

            # File tools
            FileReadTool(),
            FileWriteTool(),
            FileListTool(),
            FileOpenTool(),

            # System tools
            ShellCommandTool(),
            PythonCodeTool(),
            SystemInfoTool(),
            ApplicationLauncherTool(),
        ]

        self.tool_registry.register_multiple(tools)

    def execute_task(self, task_description: str, use_planning: bool = True) -> Dict[str, Any]:
        """
        Execute a task (with or without planning)

        Args:
            task_description: What the user wants to do
            use_planning: Whether to create a plan first

        Returns:
            Result dictionary with success, output, and metadata
        """
        logger.info(f"🎯 Executing task: {task_description}")

        # Add task to memory
        self.memory.add_observation(f"New task: {task_description}", importance=8)

        try:
            if use_planning and self.enable_planning:
                return self._execute_with_planning(task_description)
            else:
                return self._execute_direct(task_description)

        except Exception as e:
            logger.error(f"Task execution failed: {e}", exc_info=True)
            return {
                'success': False,
                'output': None,
                'error': str(e)
            }

    def _execute_with_planning(self, task_description: str) -> Dict[str, Any]:
        """Execute task using planning system"""
        logger.info("📋 Creating execution plan...")

        # Create plan
        available_tools = self.tool_registry.list_tools()
        plan = self.planner.create_plan(task_description, available_tools)

        # Show plan
        plan_summary = self.planner.get_plan_summary(plan)
        logger.info(f"\n{plan_summary}")

        # Execute plan step by step
        results = []

        while not plan.is_complete():
            current_task = plan.get_current_task()

            if not current_task:
                break

            logger.info(f"\n⏳ Executing: {current_task.description}")
            current_task.status = TaskStatus.IN_PROGRESS

            # Execute task
            if current_task.tool:
                # Use tool
                result = self.tool_registry.execute_tool(
                    current_task.tool,
                    **current_task.params
                )

                if result.success:
                    current_task.status = TaskStatus.COMPLETED
                    current_task.result = result.output
                    logger.info(f"✓ Completed: {result.output}")

                    # Add to memory
                    self.memory.add_action(
                        action=f"{current_task.tool}({current_task.params})",
                        result=str(result.output),
                        success=True
                    )
                else:
                    current_task.status = TaskStatus.FAILED
                    current_task.error = result.error
                    logger.error(f"✗ Failed: {result.error}")

                    # Add to memory
                    self.memory.add_action(
                        action=f"{current_task.tool}({current_task.params})",
                        result=result.error,
                        success=False
                    )

                results.append(result.to_dict())
            else:
                # No tool specified - use LLM directly
                response = self._llm_response(current_task.description)
                current_task.status = TaskStatus.COMPLETED
                current_task.result = response
                logger.info(f"✓ Completed: {response[:100]}...")

                results.append({
                    'success': True,
                    'output': response
                })

            plan.advance()

        # Generate final summary
        completed, total = plan.get_progress()
        success = completed == total

        summary = f"Completed {completed}/{total} tasks"
        if success:
            summary = f"✓ {summary} - Task accomplished!"
        else:
            summary = f"⚠ {summary} - Some tasks failed"

        logger.info(f"\n{summary}")

        return {
            'success': success,
            'output': summary,
            'plan': plan.to_dict(),
            'results': results
        }

    def _execute_direct(self, task_description: str) -> Dict[str, Any]:
        """Execute task directly using LLM to decide actions"""
        logger.info("🤔 Agent thinking...")

        # Build context with memory and available tools
        memory_context = self.memory.get_context_summary(max_tokens=500)
        tools_description = self.tool_registry.get_tool_descriptions()

        prompt = f"""You are an autonomous AI agent. Analyze this task and decide what action to take.

TASK: {task_description}

{memory_context}

{tools_description}

Decide which tool to use and with what parameters. Respond in JSON format:
{{
  "reasoning": "why you chose this approach",
  "tool": "tool_name",
  "params": {{"param1": "value1"}}
}}

If no tool is needed, set tool to null and provide a direct answer in reasoning.
Response must be valid JSON only.
"""

        try:
            # Get LLM decision
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.3}
            )

            response_text = response['message']['content'].strip()

            # Extract JSON
            response_text = self._extract_json(response_text)
            decision = json.loads(response_text)

            reasoning = decision.get('reasoning', '')
            tool_name = decision.get('tool')
            params = decision.get('params', {})

            logger.info(f"💭 Reasoning: {reasoning}")

            # Execute tool if specified
            if tool_name:
                logger.info(f"🔧 Using tool: {tool_name}")
                result = self.tool_registry.execute_tool(tool_name, **params)

                # Add to memory
                self.memory.add_action(
                    action=f"{tool_name}({params})",
                    result=str(result.output),
                    success=result.success
                )

                return {
                    'success': result.success,
                    'output': result.output,
                    'reasoning': reasoning,
                    'tool_used': tool_name
                }
            else:
                # No tool needed - return reasoning as answer
                return {
                    'success': True,
                    'output': reasoning,
                    'tool_used': None
                }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            # Fallback: just use LLM response
            return self._execute_fallback(task_description)

    def _execute_fallback(self, task_description: str) -> Dict[str, Any]:
        """Fallback execution when planning/parsing fails"""
        response = self._llm_response(task_description)

        return {
            'success': True,
            'output': response,
            'tool_used': None
        }

    def _llm_response(self, prompt: str) -> str:
        """Get direct LLM response"""
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.7}
            )

            return response['message']['content']

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"Error: {str(e)}"

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text"""
        # Find JSON object
        start = text.find('{')
        end = text.rfind('}')

        if start != -1 and end != -1:
            return text[start:end+1]

        # Find JSON array
        start = text.find('[')
        end = text.rfind(']')

        if start != -1 and end != -1:
            return text[start:end+1]

        return text

    def chat(self, message: str) -> str:
        """
        Conversational interface - decides whether to execute task or just chat

        Args:
            message: User message

        Returns:
            Agent's response
        """
        # Add to memory
        self.memory.add_conversation("user", message)

        # Check if this requires action
        action_keywords = [
            'open', 'search', 'find', 'create', 'write', 'read',
            'list', 'show', 'execute', 'run', 'launch', 'start',
            'download', 'upload', 'delete', 'modify'
        ]

        message_lower = message.lower()
        requires_action = any(keyword in message_lower for keyword in action_keywords)

        if requires_action:
            # Execute as task
            result = self.execute_task(message, use_planning=False)

            response = result.get('output', 'Task completed')

            # Add to memory
            self.memory.add_conversation("assistant", response)

            return response
        else:
            # Normal conversation
            memory_context = self.memory.get_context_summary(max_tokens=300)

            prompt = f"""{memory_context}

User: {message}

Respond naturally and helpfully."""

            response = self._llm_response(prompt)

            # Add to memory
            self.memory.add_conversation("assistant", response)

            return response

    def reflect(self) -> str:
        """
        Agent reflects on recent actions and memories
        """
        recent = self.memory.get_recent_memories(n=10)

        if not recent:
            return "No recent memories to reflect on."

        memories_text = "\n".join([f"- {m.content}" for m in recent])

        prompt = f"""Reflect on these recent memories and actions:

{memories_text}

Provide a brief reflection on what was accomplished and any patterns or insights."""

        reflection = self._llm_response(prompt)

        # Add reflection to memory
        self.memory.add_reflection(reflection, importance=7)

        return reflection

    def get_status(self) -> Dict:
        """Get agent status"""
        return {
            'model': self.model,
            'tools_available': len(self.tool_registry.list_tools()),
            'tools': self.tool_registry.list_tools(),
            'memory_stats': self.memory.get_stats(),
            'planning_enabled': self.enable_planning
        }
