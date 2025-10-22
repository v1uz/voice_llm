"""
Web-related tools for the AI agent
"""

import webbrowser
import requests
from urllib.parse import quote_plus, urlparse
from typing import Dict, Optional
import logging
from .base import Tool, ToolResult

logger = logging.getLogger(__name__)


class WebSearchTool(Tool):
    """Search the web using Google"""

    def get_description(self) -> str:
        return "Search Google for information. Use this when user asks to search or find information online."

    def get_schema(self) -> Dict:
        return {
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }

    def execute(self, query: str) -> ToolResult:
        """Execute Google search"""
        try:
            search_url = f"https://www.google.com/search?q={quote_plus(query)}"
            webbrowser.open(search_url)

            return ToolResult(
                success=True,
                output=f"Opened Google search for: {query}",
                metadata={"query": query, "url": search_url}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Failed to search: {str(e)}"
            )


class WebBrowserTool(Tool):
    """Open websites in browser"""

    def get_description(self) -> str:
        return "Open a website in the default browser. Use when user wants to visit a specific URL or domain."

    def get_schema(self) -> Dict:
        return {
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to open (e.g., 'google.com' or 'https://example.com')"
                }
            },
            "required": ["url"]
        }

    def execute(self, url: str) -> ToolResult:
        """Open URL in browser"""
        try:
            # Add https:// if no protocol specified
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url

            # Validate URL
            parsed = urlparse(url)
            if not parsed.netloc:
                return ToolResult(
                    success=False,
                    output=None,
                    error="Invalid URL format"
                )

            webbrowser.open(url)

            return ToolResult(
                success=True,
                output=f"Opened {url}",
                metadata={"url": url}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Failed to open URL: {str(e)}"
            )


class WebFetchTool(Tool):
    """Fetch content from a URL"""

    def get_description(self) -> str:
        return "Fetch and read content from a URL. Returns the text content of the page."

    def get_schema(self) -> Dict:
        return {
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 10)",
                    "default": 10
                }
            },
            "required": ["url"]
        }

    def execute(self, url: str, timeout: int = 10) -> ToolResult:
        """Fetch URL content"""
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url

            response = requests.get(url, timeout=timeout, headers={
                'User-Agent': 'Mozilla/5.0 (AI Agent)'
            })
            response.raise_for_status()

            # Get text content (limit to first 5000 chars)
            content = response.text[:5000]

            return ToolResult(
                success=True,
                output=content,
                metadata={
                    "url": url,
                    "status_code": response.status_code,
                    "content_length": len(response.text)
                }
            )
        except requests.RequestException as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Failed to fetch URL: {str(e)}"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Unexpected error: {str(e)}"
            )


class YouTubeSearchTool(Tool):
    """Search YouTube"""

    def get_description(self) -> str:
        return "Search for videos on YouTube. Opens search results in browser."

    def get_schema(self) -> Dict:
        return {
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for on YouTube"
                }
            },
            "required": ["query"]
        }

    def execute(self, query: str) -> ToolResult:
        """Search YouTube"""
        try:
            search_url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
            webbrowser.open(search_url)

            return ToolResult(
                success=True,
                output=f"Opened YouTube search for: {query}",
                metadata={"query": query, "url": search_url}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Failed to search YouTube: {str(e)}"
            )
