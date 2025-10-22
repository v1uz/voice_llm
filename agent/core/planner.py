"""
Task Planning System for AI Agent
Breaks down complex tasks into executable steps
"""

import json
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a task"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    """Represents a single task in a plan"""
    description: str
    tool: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    dependencies: List[int] = field(default_factory=list)  # Indices of dependent tasks

    def to_dict(self) -> Dict:
        return {
            'description': self.description,
            'tool': self.tool,
            'params': self.params,
            'status': self.status.value,
            'result': str(self.result) if self.result else None,
            'error': self.error,
            'dependencies': self.dependencies
        }


@dataclass
class Plan:
    """Represents a complete execution plan"""
    goal: str
    tasks: List[Task] = field(default_factory=list)
    current_task_index: int = 0
    metadata: Dict = field(default_factory=dict)

    def add_task(self, task: Task) -> int:
        """Add a task and return its index"""
        self.tasks.append(task)
        return len(self.tasks) - 1

    def get_current_task(self) -> Optional[Task]:
        """Get the current task to execute"""
        if self.current_task_index < len(self.tasks):
            return self.tasks[self.current_task_index]
        return None

    def advance(self) -> bool:
        """Move to next task. Returns False if no more tasks"""
        self.current_task_index += 1
        return self.current_task_index < len(self.tasks)

    def is_complete(self) -> bool:
        """Check if all tasks are done"""
        return self.current_task_index >= len(self.tasks)

    def get_progress(self) -> tuple[int, int]:
        """Return (completed, total) tasks"""
        completed = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        return completed, len(self.tasks)

    def to_dict(self) -> Dict:
        return {
            'goal': self.goal,
            'tasks': [t.to_dict() for t in self.tasks],
            'current_task_index': self.current_task_index,
            'progress': f"{self.get_progress()[0]}/{self.get_progress()[1]}",
            'metadata': self.metadata
        }


class TaskPlanner:
    """
    Plans and decomposes complex tasks into executable steps
    """

    def __init__(self, llm_client):
        """
        Initialize planner with LLM client for decomposition

        Args:
            llm_client: Client for making LLM calls (e.g., ollama)
        """
        self.llm_client = llm_client
        self.current_plan: Optional[Plan] = None

    def create_plan(self, goal: str, available_tools: List[str]) -> Plan:
        """
        Create an execution plan for a goal

        Args:
            goal: The high-level goal to achieve
            available_tools: List of available tool names

        Returns:
            A Plan object with decomposed tasks
        """
        logger.info(f"📋 Creating plan for goal: {goal}")

        # Build prompt for LLM
        tools_list = "\n".join([f"- {tool}" for tool in available_tools])

        prompt = f"""You are an AI task planner. Break down this goal into specific, executable steps.

GOAL: {goal}

AVAILABLE TOOLS:
{tools_list}

Create a step-by-step plan. For each step, specify:
1. Clear description of what to do
2. Which tool to use (if applicable)
3. Parameters for the tool (if applicable)

Format your response as a JSON array of steps:
[
  {{
    "description": "Clear description of step",
    "tool": "tool_name or null",
    "params": {{"param1": "value1"}}
  }},
  ...
]

Keep the plan simple and executable. Each step should be atomic and achievable.
Response must be valid JSON only, no additional text.
"""

        try:
            # Call LLM to generate plan
            response = self.llm_client.chat(
                model="llama3.2",
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.3}
            )

            response_text = response['message']['content'].strip()

            # Extract JSON from response
            response_text = self._extract_json(response_text)

            # Parse JSON
            steps = json.loads(response_text)

            # Create plan
            plan = Plan(goal=goal)

            for step_data in steps:
                task = Task(
                    description=step_data.get('description', ''),
                    tool=step_data.get('tool'),
                    params=step_data.get('params', {})
                )
                plan.add_task(task)

            logger.info(f"✓ Plan created with {len(plan.tasks)} tasks")
            self.current_plan = plan
            return plan

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Response was: {response_text}")

            # Fallback: create simple plan
            plan = Plan(goal=goal)
            plan.add_task(Task(
                description=f"Execute: {goal}",
                tool=None,
                params={}
            ))
            return plan

        except Exception as e:
            logger.error(f"Failed to create plan: {e}", exc_info=True)

            # Fallback plan
            plan = Plan(goal=goal)
            plan.add_task(Task(
                description=f"Execute: {goal}",
                tool=None,
                params={}
            ))
            return plan

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that might contain markdown or other content"""
        # Try to find JSON array
        start = text.find('[')
        end = text.rfind(']')

        if start != -1 and end != -1:
            return text[start:end+1]

        # Try to find JSON object
        start = text.find('{')
        end = text.rfind('}')

        if start != -1 and end != -1:
            return text[start:end+1]

        return text

    def simplify_goal(self, goal: str) -> Plan:
        """
        Create a simple single-step plan when decomposition isn't needed
        """
        plan = Plan(goal=goal)
        plan.add_task(Task(
            description=goal,
            tool=None,
            params={}
        ))
        return plan

    def adapt_plan(self, plan: Plan, feedback: str) -> Plan:
        """
        Adapt existing plan based on feedback or errors

        Args:
            plan: Current plan
            feedback: Feedback about what went wrong or needs adjustment

        Returns:
            Modified plan
        """
        logger.info(f"🔄 Adapting plan based on feedback: {feedback}")

        # Get current task
        current_task = plan.get_current_task()

        if current_task and current_task.status == TaskStatus.FAILED:
            # Create retry task with modified approach
            retry_task = Task(
                description=f"Retry: {current_task.description} (with adjustment: {feedback})",
                tool=current_task.tool,
                params=current_task.params
            )
            plan.tasks.insert(plan.current_task_index + 1, retry_task)

        return plan

    def get_plan_summary(self, plan: Plan) -> str:
        """Get human-readable summary of plan"""
        completed, total = plan.get_progress()

        summary = [
            f"📋 Plan: {plan.goal}",
            f"Progress: {completed}/{total} tasks completed",
            "\nTasks:"
        ]

        for i, task in enumerate(plan.tasks, 1):
            status_icon = {
                TaskStatus.COMPLETED: "✓",
                TaskStatus.IN_PROGRESS: "⏳",
                TaskStatus.FAILED: "✗",
                TaskStatus.PENDING: "○",
                TaskStatus.SKIPPED: "⊘"
            }.get(task.status, "○")

            tool_info = f" [{task.tool}]" if task.tool else ""
            summary.append(f"  {i}. {status_icon} {task.description}{tool_info}")

        return "\n".join(summary)
