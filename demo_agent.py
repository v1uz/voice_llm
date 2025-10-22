#!/usr/bin/env python3
"""
Demo script - Test AI Agent without voice interface
Quick way to test agent capabilities
"""

import logging
from agent import AIAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

def main():
    print("\n" + "="*70)
    print("🤖 AI AGENT DEMO - Testing Agent Capabilities")
    print("="*70 + "\n")

    # Initialize agent
    print("Initializing agent...")
    agent = AIAgent(
        model="llama3.2",
        enable_planning=True,
        memory_file="demo_memory.json"
    )

    # Show status
    status = agent.get_status()
    print(f"\n✓ Agent initialized")
    print(f"   Model: {status['model']}")
    print(f"   Tools: {status['tools_available']}")
    print(f"   Available tools: {', '.join(status['tools'][:5])}...")

    print("\n" + "="*70)
    print("DEMO TASKS")
    print("="*70)

    # Demo 1: Simple web search
    print("\n📌 Demo 1: Web Search")
    print("-" * 70)
    result = agent.execute_task(
        "Search Google for artificial intelligence",
        use_planning=False
    )
    print(f"Result: {result['output']}")

    # Demo 2: File operations
    print("\n📌 Demo 2: File Operations")
    print("-" * 70)
    result = agent.execute_task(
        "Create a file called demo.txt with the text 'Hello from AI Agent!'",
        use_planning=True
    )
    print(f"Result: {result['output']}")

    # Demo 3: System info
    print("\n📌 Demo 3: System Information")
    print("-" * 70)
    result = agent.execute_task(
        "Get system information",
        use_planning=False
    )
    print(f"Result:\n{result['output']}")

    # Demo 4: Chat
    print("\n📌 Demo 4: Conversational")
    print("-" * 70)
    response = agent.chat("What are you capable of?")
    print(f"Agent: {response}")

    # Demo 5: Complex task with planning
    print("\n📌 Demo 5: Complex Task (Planning)")
    print("-" * 70)
    result = agent.execute_task(
        "List all files in current directory and create a report",
        use_planning=True
    )
    print(f"\nResult: {result['output']}")

    if result.get('plan'):
        print("\nExecution plan:")
        for i, task in enumerate(result['plan']['tasks'], 1):
            status = task['status']
            icon = "✓" if status == 'completed' else "✗" if status == 'failed' else "○"
            print(f"  {i}. {icon} {task['description']}")

    # Demo 6: Reflection
    print("\n📌 Demo 6: Agent Reflection")
    print("-" * 70)
    reflection = agent.reflect()
    print(f"Reflection:\n{reflection}")

    # Memory stats
    print("\n" + "="*70)
    print("MEMORY STATISTICS")
    print("="*70)
    mem_stats = agent.memory.get_stats()
    print(f"Short-term memories: {mem_stats['short_term_count']}")
    print(f"Long-term memories: {mem_stats['long_term_count']}")
    print(f"Total memories: {mem_stats['total_count']}")
    print(f"By type: {mem_stats['memory_types']}")

    print("\n" + "="*70)
    print("✓ DEMO COMPLETE")
    print("="*70 + "\n")

    print("💡 To use with voice interface, run: python voice_agent.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
