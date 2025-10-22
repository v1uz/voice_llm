"""
System and code execution tools for the AI agent
"""

import subprocess
import sys
import platform
from typing import Dict, List
import logging
from .base import Tool, ToolResult

logger = logging.getLogger(__name__)


class ShellCommandTool(Tool):
    """Execute safe shell commands"""

    # Whitelist of safe commands
    SAFE_COMMANDS = {
        'ls', 'dir', 'pwd', 'cd', 'echo', 'cat', 'grep', 'find',
        'which', 'whereis', 'date', 'whoami', 'uname', 'ps',
        'git', 'pip', 'python', 'node', 'npm', 'cargo'
    }

    # Blacklist of dangerous commands
    DANGEROUS_COMMANDS = {
        'rm', 'del', 'format', 'shutdown', 'reboot', 'kill',
        'pkill', 'killall', 'halt', 'dd', 'mkfs', '>>', '>'
    }

    def get_description(self) -> str:
        return "Execute safe shell commands. Can run read-only commands and safe development tools."

    def get_schema(self) -> Dict:
        return {
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 30)",
                    "default": 30
                }
            },
            "required": ["command"]
        }

    def is_safe_command(self, command: str) -> tuple[bool, str]:
        """Check if command is safe to execute"""
        command_lower = command.lower()

        # Check for dangerous commands
        for dangerous in self.DANGEROUS_COMMANDS:
            if dangerous in command_lower:
                return False, f"Dangerous command detected: {dangerous}"

        # Get first word (the actual command)
        first_word = command.split()[0] if command.split() else ""

        # Check if it's in safe list or common safe patterns
        if first_word in self.SAFE_COMMANDS or first_word.endswith(('.py', '.sh')):
            return True, "Command is safe"

        return False, f"Command '{first_word}' not in safe list"

    def execute(self, command: str, timeout: int = 30) -> ToolResult:
        """Execute shell command with safety checks"""
        try:
            # Safety check
            is_safe, reason = self.is_safe_command(command)
            if not is_safe:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Command blocked: {reason}"
                )

            # Execute command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            # Prepare output
            output = result.stdout if result.returncode == 0 else result.stderr
            output = output[:2000]  # Limit output size

            return ToolResult(
                success=result.returncode == 0,
                output=output,
                error=None if result.returncode == 0 else result.stderr[:500],
                metadata={
                    "command": command,
                    "returncode": result.returncode,
                    "timeout": timeout
                }
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output=None,
                error=f"Command timed out after {timeout} seconds"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Command execution failed: {str(e)}"
            )


class PythonCodeTool(Tool):
    """Execute Python code safely"""

    def get_description(self) -> str:
        return "Execute Python code snippets. Code runs in isolated environment with limited imports."

    def get_schema(self) -> Dict:
        return {
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                }
            },
            "required": ["code"]
        }

    def execute(self, code: str) -> ToolResult:
        """Execute Python code"""
        try:
            # Check for dangerous imports/operations
            dangerous_patterns = [
                'import os', 'import sys', 'import subprocess',
                'eval(', 'exec(', '__import__', 'open(',
                'file(', 'input(', 'raw_input('
            ]

            code_lower = code.lower()
            for pattern in dangerous_patterns:
                if pattern in code_lower:
                    return ToolResult(
                        success=False,
                        output=None,
                        error=f"Dangerous operation detected: {pattern}"
                    )

            # Create restricted namespace
            namespace = {
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'range': range,
                    'str': str,
                    'int': int,
                    'float': float,
                    'list': list,
                    'dict': dict,
                    'tuple': tuple,
                    'set': set,
                    'bool': bool,
                    'sum': sum,
                    'max': max,
                    'min': min,
                    'abs': abs,
                    'round': round,
                }
            }

            # Capture output
            import io
            from contextlib import redirect_stdout

            output_buffer = io.StringIO()

            with redirect_stdout(output_buffer):
                exec(code, namespace)

            output = output_buffer.getvalue()

            return ToolResult(
                success=True,
                output=output or "Code executed successfully (no output)",
                metadata={"code_length": len(code)}
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Python execution error: {str(e)}"
            )


class SystemInfoTool(Tool):
    """Get system information"""

    def get_description(self) -> str:
        return "Get information about the system (OS, Python version, platform, etc)."

    def get_schema(self) -> Dict:
        return {
            "properties": {},
            "required": []
        }

    def execute(self) -> ToolResult:
        """Get system info"""
        try:
            info = {
                "OS": platform.system(),
                "OS Version": platform.release(),
                "Platform": platform.platform(),
                "Architecture": platform.machine(),
                "Python Version": sys.version.split()[0],
                "Python Path": sys.executable,
            }

            output = "\n".join([f"{k}: {v}" for k, v in info.items()])

            return ToolResult(
                success=True,
                output=output,
                metadata=info
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Failed to get system info: {str(e)}"
            )


class ApplicationLauncherTool(Tool):
    """Launch applications"""

    def get_description(self) -> str:
        return "Launch applications (e.g., 'notepad', 'calculator', 'chrome')."

    def get_schema(self) -> Dict:
        return {
            "properties": {
                "app_name": {
                    "type": "string",
                    "description": "Name of the application to launch"
                }
            },
            "required": ["app_name"]
        }

    def execute(self, app_name: str) -> ToolResult:
        """Launch application"""
        try:
            if sys.platform == 'win32':
                # Windows
                common_apps = {
                    'notepad': 'notepad.exe',
                    'calculator': 'calc.exe',
                    'paint': 'mspaint.exe',
                    'chrome': 'chrome.exe',
                    'firefox': 'firefox.exe',
                    'edge': 'msedge.exe'
                }
                cmd = common_apps.get(app_name.lower(), app_name)
                subprocess.Popen([cmd], shell=True)

            elif sys.platform == 'darwin':
                # macOS
                subprocess.Popen(['open', '-a', app_name])

            else:
                # Linux
                subprocess.Popen([app_name])

            return ToolResult(
                success=True,
                output=f"Launched {app_name}",
                metadata={"app": app_name}
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Failed to launch {app_name}: {str(e)}"
            )
