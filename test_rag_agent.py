#!/usr/bin/env python3
"""
Test script for RAGAgent
This script tests the basic functionality of the RAGAgent without running actual games.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag_agent import RAGAgent
import argparse

class MockArgs:
    """Mock arguments class for testing"""
    def __init__(self):
        self.llm_model = "openai/gpt-4o-mini"
        self.embedding_model = "openai/gpt-4o-mini"
        self.retrieval_top_k = 3
        self.retrieval_threshold = 0.1
        self.rag_temperature = 0.4
        self.rag_max_tokens = 400
        self.llm_temperature = 0.4
        self.max_memory = 5

class MockStateNode:
    """Mock state node for testing"""
    def __init__(self, state):
        self.state = state

def test_rag_agent():
    """Test the RAGAgent functionality"""
    print("=== Testing RAGAgent ===\n")
    
    # Create mock arguments
    args = MockArgs()
    
    # Create RAGAgent instance
    agent = RAGAgent(args, guiding_prompt="Explore systematically and examine objects to make progress.")
    
    print("1. Testing agent initialization...")
    print(f"   Guiding prompt: {agent.guiding_prompt}")
    print(f"   Retrieval top-k: {agent.retrieval_top_k}")
    print(f"   Retrieval threshold: {agent.retrieval_threshold}")
    print(f"   Embedding model: {agent.embedding_model}")
    print("   ✓ Initialization successful\n")
    
    # Test episode start
    print("2. Testing episode start...")
    agent.start_episode()
    print("   ✓ Episode started successfully\n")
    
    # Test adding game history
    print("3. Testing game history addition...")
    test_history = [
        {"state": "You are in a dimly lit room. There's a door to the north and a table in the center.", "action": "look", "full_response": "REASONING: I should examine my surroundings. ACTION: look"},
        {"state": "The table is made of wood. There's a key on it.", "action": "examine table", "full_response": "REASONING: I should examine the table to see what's on it. ACTION: examine table"},
        {"state": "You pick up the key. It's made of brass and feels warm to the touch.", "action": "get key", "full_response": "REASONING: I should pick up the key as it might be useful. ACTION: get key"},
        {"state": "You are in a dimly lit room. There's a door to the north and a table in the center.", "action": "open door", "full_response": "REASONING: Now I should try to open the door with the key. ACTION: open door"},
        {"state": "The door opens with a creak, revealing a dark corridor beyond.", "action": "north", "full_response": "REASONING: I should explore the corridor. ACTION: north"},
        {"state": "You are in a dark corridor. The walls are damp and there's a faint light ahead.", "action": "look", "full_response": "REASONING: I should examine this new area. ACTION: look"},
        {"state": "There's a chest at the end of the corridor. It looks ancient and ornate.", "action": "examine chest", "full_response": "REASONING: I should examine the chest for clues. ACTION: examine chest"}
    ]
    
    for entry in test_history:
        agent._add_to_game_history(entry["state"], entry["action"], entry["full_response"])
    
    print(f"   Added {len(test_history)} history entries")
    print("   ✓ Game history addition successful\n")
    
    # Test memory addition
    print("4. Testing memory addition...")
    agent.add_to_memory("You are in a dimly lit room.", "REASONING: I should examine my surroundings. ACTION: look")
    print("   ✓ Memory addition successful\n")
    
    # Test entity extraction
    print("5. Testing entity extraction...")
    test_text = "You are in a dimly lit room with a wooden table and a brass key."
    entities = agent._extract_key_entities(test_text)
    print(f"   Extracted entities: {entities}")
    print("   ✓ Entity extraction successful\n")
    
    # Test retrieval index update
    print("6. Testing retrieval index update...")
    agent._update_retrieval_index()
    print(f"   History vectors created: {agent.history_vectors is not None}")
    print(f"   History texts count: {len(agent.history_texts)}")
    print("   ✓ Retrieval index update successful\n")
    
    # Test relevant history retrieval
    print("7. Testing relevant history retrieval...")
    current_state = "You are in a dimly lit room. There's a door to the north and a table in the center."
    retrieved_entries = agent._retrieve_relevant_history(current_state)
    print(f"   Retrieved {len(retrieved_entries)} relevant entries")
    if retrieved_entries:
        print(f"   Top similarity score: {retrieved_entries[0].get('similarity', 'N/A')}")
    print("   ✓ Relevant history retrieval successful\n")
    
    # Test context formatting
    print("8. Testing context formatting...")
    if retrieved_entries:
        context_text = agent._format_retrieved_context(retrieved_entries)
        print("   Context text preview:")
        print("   " + context_text[:200] + "...")
        print("   ✓ Context formatting successful\n")
    
    # Test prompt generation
    print("9. Testing prompt generation...")
    state_node = MockStateNode("You are in a dimly lit room. There's a door to the north and a table in the center.")
    sys_prompt, user_prompt = agent.get_prompts(state_node)
    
    print("   System prompt preview:")
    print("   " + sys_prompt[:100] + "...")
    print("   User prompt preview:")
    print("   " + user_prompt[:150] + "...")
    print("   ✓ Prompt generation successful\n")
    
    print("=== All tests completed successfully! ===")
    print("\nNote: This test script doesn't make actual LLM API calls.")
    print("To test the full functionality, run the agent with actual games.")

if __name__ == "__main__":
    test_rag_agent() 