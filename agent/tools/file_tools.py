"""
File system tools for the AI agent
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import logging
from .base import Tool, ToolResult

logger = logging.getLogger(__name__)


class FileReadTool(Tool):
    """Read contents of a file"""

    def get_description(self) -> str:
        return "Read the contents of a text file. Returns the file content."

    def get_schema(self) -> Dict:
        return {
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Path to the file to read"
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum number of lines to read (default: 100)",
                    "default": 100
                }
            },
            "required": ["filepath"]
        }

    def execute(self, filepath: str, max_lines: int = 100) -> ToolResult:
        """Read file contents"""
        try:
            path = Path(filepath).expanduser().resolve()

            if not path.exists():
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"File not found: {filepath}"
                )

            if not path.is_file():
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Not a file: {filepath}"
                )

            # Read file
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        lines.append(f"\n... (truncated, {max_lines}+ lines)")
                        break
                    lines.append(line.rstrip())

            content = '\n'.join(lines)

            return ToolResult(
                success=True,
                output=content,
                metadata={
                    "filepath": str(path),
                    "size": path.stat().st_size,
                    "lines_read": min(len(lines), max_lines)
                }
            )

        except PermissionError:
            return ToolResult(
                success=False,
                output=None,
                error=f"Permission denied: {filepath}"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Failed to read file: {str(e)}"
            )


class FileWriteTool(Tool):
    """Write content to a file"""

    def get_description(self) -> str:
        return "Write text content to a file. Creates or overwrites the file."

    def get_schema(self) -> Dict:
        return {
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Path where to write the file"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                },
                "append": {
                    "type": "boolean",
                    "description": "Append to file instead of overwriting (default: False)",
                    "default": False
                }
            },
            "required": ["filepath", "content"]
        }

    def execute(self, filepath: str, content: str, append: bool = False) -> ToolResult:
        """Write to file"""
        try:
            path = Path(filepath).expanduser().resolve()

            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)

            mode = 'a' if append else 'w'
            with open(path, mode, encoding='utf-8') as f:
                f.write(content)

            action = "Appended to" if append else "Wrote"

            return ToolResult(
                success=True,
                output=f"{action} {len(content)} characters to {path.name}",
                metadata={
                    "filepath": str(path),
                    "size": len(content),
                    "mode": mode
                }
            )

        except PermissionError:
            return ToolResult(
                success=False,
                output=None,
                error=f"Permission denied: {filepath}"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Failed to write file: {str(e)}"
            )


class FileListTool(Tool):
    """List files in a directory"""

    def get_description(self) -> str:
        return "List files and directories in a given path."

    def get_schema(self) -> Dict:
        return {
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory path to list (default: current directory)",
                    "default": "."
                },
                "pattern": {
                    "type": "string",
                    "description": "Optional glob pattern to filter files (e.g., '*.py')",
                    "default": "*"
                }
            },
            "required": []
        }

    def execute(self, directory: str = ".", pattern: str = "*") -> ToolResult:
        """List directory contents"""
        try:
            path = Path(directory).expanduser().resolve()

            if not path.exists():
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Directory not found: {directory}"
                )

            if not path.is_dir():
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Not a directory: {directory}"
                )

            # List files matching pattern
            files = sorted(path.glob(pattern))

            file_list = []
            for f in files[:50]:  # Limit to 50 items
                file_type = "📁" if f.is_dir() else "📄"
                size = f.stat().st_size if f.is_file() else 0
                file_list.append(f"{file_type} {f.name} ({size} bytes)")

            if len(files) > 50:
                file_list.append(f"... and {len(files) - 50} more items")

            output = "\n".join(file_list) if file_list else "Directory is empty"

            return ToolResult(
                success=True,
                output=output,
                metadata={
                    "directory": str(path),
                    "total_items": len(files),
                    "pattern": pattern
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Failed to list directory: {str(e)}"
            )


class FileOpenTool(Tool):
    """Open a file with default application"""

    def get_description(self) -> str:
        return "Open a file with its default application (editor, viewer, etc)."

    def get_schema(self) -> Dict:
        return {
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Path to the file to open"
                }
            },
            "required": ["filepath"]
        }

    def execute(self, filepath: str) -> ToolResult:
        """Open file with default app"""
        try:
            path = Path(filepath).expanduser().resolve()

            if not path.exists():
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"File not found: {filepath}"
                )

            # Open with default application
            if sys.platform == 'win32':
                os.startfile(path)
            elif sys.platform == 'darwin':
                subprocess.run(['open', str(path)], check=True)
            else:  # Linux
                subprocess.run(['xdg-open', str(path)], check=True)

            return ToolResult(
                success=True,
                output=f"Opened {path.name}",
                metadata={"filepath": str(path)}
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Failed to open file: {str(e)}"
            )
