#!/usr/bin/env python3
"""
Quick test for fixed agent - demonstrates simple task execution
"""

import logging
from agent import AIAgent

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def test_simple_tasks():
    """Test simple tasks that should use direct execution"""

    print("\n" + "="*70)
    print("🧪 TESTING FIXED AGENT - Simple Tasks")
    print("="*70 + "\n")

    agent = AIAgent(model="llama3.2", enable_planning=True)

    # Test 1: Create file (was failing before)
    print("\n📝 Test 1: Create file with content")
    print("-" * 70)
    result = agent.execute_task(
        "Create a file called shopping.txt with milk, eggs, bread"
    )
    print(f"✓ Result: {result['output']}")
    print(f"  Success: {result['success']}")
    if result.get('tool_used'):
        print(f"  Tool: {result['tool_used']}")

    # Test 2: Read the file
    print("\n📖 Test 2: Read the file")
    print("-" * 70)
    result = agent.execute_task("Read the file shopping.txt")
    print(f"✓ Result: {result['output'][:200]}")  # First 200 chars

    # Test 3: Web search
    print("\n🔍 Test 3: Web search")
    print("-" * 70)
    result = agent.execute_task("Search for AI agents")
    print(f"✓ Result: {result['output']}")

    # Test 4: Open website
    print("\n🌐 Test 4: Open website")
    print("-" * 70)
    result = agent.execute_task("Open google.com")
    print(f"✓ Result: {result['output']}")

    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETED")
    print("="*70 + "\n")

    print("💡 If shopping.txt was created successfully, the fix works!")
    print("   Check the file: cat shopping.txt")


if __name__ == "__main__":
    try:
        test_simple_tasks()
    except KeyboardInterrupt:
        print("\n\n🛑 Tests interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
