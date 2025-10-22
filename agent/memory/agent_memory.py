"""
Memory System for AI Agent
Manages short-term and long-term memory
"""

import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """Single memory entry"""
    timestamp: float
    type: str  # 'conversation', 'action', 'observation', 'reflection'
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: int = 5  # 1-10, higher = more important

    def to_dict(self) -> Dict:
        return asdict(self)

    def age_hours(self) -> float:
        """Get age of memory in hours"""
        return (time.time() - self.timestamp) / 3600


class AgentMemory:
    """
    Memory system with short-term and long-term storage
    """

    def __init__(self, max_short_term: int = 50, memory_file: Optional[str] = None):
        """
        Initialize memory system

        Args:
            max_short_term: Maximum entries in short-term memory
            memory_file: Path to file for persisting long-term memory
        """
        self.short_term_memory: List[MemoryEntry] = []
        self.long_term_memory: List[MemoryEntry] = []
        self.max_short_term = max_short_term
        self.memory_file = Path(memory_file) if memory_file else None

        # Load long-term memory if file exists
        if self.memory_file and self.memory_file.exists():
            self.load_long_term_memory()

    def add_memory(
        self,
        content: str,
        memory_type: str = "observation",
        importance: int = 5,
        metadata: Optional[Dict] = None
    ) -> MemoryEntry:
        """
        Add a new memory entry

        Args:
            content: The memory content
            memory_type: Type of memory (conversation, action, observation, reflection)
            importance: Importance score (1-10)
            metadata: Additional metadata

        Returns:
            The created memory entry
        """
        entry = MemoryEntry(
            timestamp=time.time(),
            type=memory_type,
            content=content,
            metadata=metadata or {},
            importance=importance
        )

        self.short_term_memory.append(entry)

        # Move old/unimportant memories to long-term
        if len(self.short_term_memory) > self.max_short_term:
            self._consolidate_memory()

        logger.debug(f"💭 Added memory: {memory_type} - {content[:50]}...")

        return entry

    def add_conversation(self, role: str, content: str, importance: int = 5):
        """Add conversation memory"""
        self.add_memory(
            content=content,
            memory_type="conversation",
            importance=importance,
            metadata={"role": role}
        )

    def add_action(self, action: str, result: str, success: bool):
        """Add action memory"""
        self.add_memory(
            content=f"Action: {action} -> {result}",
            memory_type="action",
            importance=8 if success else 9,  # Failures are more important
            metadata={
                "action": action,
                "result": result,
                "success": success
            }
        )

    def add_observation(self, observation: str, importance: int = 5):
        """Add observation memory"""
        self.add_memory(
            content=observation,
            memory_type="observation",
            importance=importance
        )

    def add_reflection(self, reflection: str, importance: int = 7):
        """Add reflection memory (agent thinking about its actions)"""
        self.add_memory(
            content=reflection,
            memory_type="reflection",
            importance=importance
        )

    def get_recent_memories(
        self,
        n: int = 10,
        memory_type: Optional[str] = None
    ) -> List[MemoryEntry]:
        """
        Get recent memories

        Args:
            n: Number of memories to retrieve
            memory_type: Filter by memory type (optional)

        Returns:
            List of recent memory entries
        """
        memories = self.short_term_memory.copy()

        if memory_type:
            memories = [m for m in memories if m.type == memory_type]

        return memories[-n:]

    def get_important_memories(
        self,
        threshold: int = 7,
        max_age_hours: Optional[float] = None
    ) -> List[MemoryEntry]:
        """
        Get important memories

        Args:
            threshold: Minimum importance score
            max_age_hours: Maximum age in hours (optional)

        Returns:
            List of important memories
        """
        all_memories = self.short_term_memory + self.long_term_memory

        # Filter by importance
        important = [m for m in all_memories if m.importance >= threshold]

        # Filter by age if specified
        if max_age_hours:
            important = [m for m in important if m.age_hours() <= max_age_hours]

        # Sort by importance and recency
        important.sort(key=lambda m: (m.importance, m.timestamp), reverse=True)

        return important[:20]  # Return top 20

    def search_memories(self, query: str, n: int = 5) -> List[MemoryEntry]:
        """
        Search memories by content

        Args:
            query: Search query
            n: Maximum number of results

        Returns:
            List of matching memories
        """
        query_lower = query.lower()
        all_memories = self.short_term_memory + self.long_term_memory

        # Simple keyword search
        matches = [
            m for m in all_memories
            if query_lower in m.content.lower()
        ]

        # Sort by recency
        matches.sort(key=lambda m: m.timestamp, reverse=True)

        return matches[:n]

    def get_context_summary(self, max_tokens: int = 1000) -> str:
        """
        Get a summary of relevant context for LLM

        Args:
            max_tokens: Approximate maximum tokens (rough estimate: 1 token ≈ 4 chars)

        Returns:
            Formatted context string
        """
        max_chars = max_tokens * 4

        # Get recent and important memories
        recent = self.get_recent_memories(n=10)
        important = self.get_important_memories(threshold=7)

        # Combine and deduplicate
        all_context = []
        seen = set()

        for memory in recent + important:
            if memory.timestamp not in seen:
                seen.add(memory.timestamp)
                all_context.append(memory)

        # Sort by timestamp
        all_context.sort(key=lambda m: m.timestamp)

        # Build context string
        lines = ["[AGENT MEMORY CONTEXT]"]

        for memory in all_context:
            line = f"{memory.type.upper()}: {memory.content}"
            if len("\n".join(lines + [line])) > max_chars:
                break
            lines.append(line)

        return "\n".join(lines)

    def _consolidate_memory(self):
        """Move old/less important memories from short-term to long-term"""
        # Sort by importance and age
        self.short_term_memory.sort(key=lambda m: (m.importance, m.timestamp))

        # Move least important half to long-term
        cutoff = len(self.short_term_memory) // 2
        to_archive = self.short_term_memory[:cutoff]

        self.long_term_memory.extend(to_archive)
        self.short_term_memory = self.short_term_memory[cutoff:]

        logger.debug(f"📦 Consolidated {len(to_archive)} memories to long-term storage")

        # Save to file if configured
        if self.memory_file:
            self.save_long_term_memory()

    def save_long_term_memory(self):
        """Persist long-term memory to file"""
        if not self.memory_file:
            return

        try:
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)

            data = [m.to_dict() for m in self.long_term_memory]

            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"💾 Saved {len(data)} memories to {self.memory_file}")

        except Exception as e:
            logger.error(f"Failed to save long-term memory: {e}")

    def load_long_term_memory(self):
        """Load long-term memory from file"""
        if not self.memory_file or not self.memory_file.exists():
            return

        try:
            with open(self.memory_file, 'r') as f:
                data = json.load(f)

            self.long_term_memory = [
                MemoryEntry(**entry) for entry in data
            ]

            logger.info(f"📂 Loaded {len(self.long_term_memory)} memories from {self.memory_file}")

        except Exception as e:
            logger.error(f"Failed to load long-term memory: {e}")

    def clear_short_term(self):
        """Clear short-term memory"""
        self.short_term_memory.clear()
        logger.info("🧹 Cleared short-term memory")

    def clear_all(self):
        """Clear all memory"""
        self.short_term_memory.clear()
        self.long_term_memory.clear()
        logger.info("🧹 Cleared all memory")

    def get_stats(self) -> Dict:
        """Get memory statistics"""
        return {
            "short_term_count": len(self.short_term_memory),
            "long_term_count": len(self.long_term_memory),
            "total_count": len(self.short_term_memory) + len(self.long_term_memory),
            "memory_types": self._count_by_type()
        }

    def _count_by_type(self) -> Dict[str, int]:
        """Count memories by type"""
        all_memories = self.short_term_memory + self.long_term_memory
        counts = {}

        for memory in all_memories:
            counts[memory.type] = counts.get(memory.type, 0) + 1

        return counts
