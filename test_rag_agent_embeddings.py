#!/usr/bin/env python3
"""
Test script for RAGAgent with LM embeddings.
This script tests the basic functionality without making actual API calls.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from unittest.mock import Mock, patch
import numpy as np

def test_rag_agent_initialization():
    """Test RAGAgent initialization."""
    print("Testing RAGAgent initialization...")
    
    # Mock args
    args = Mock()
    args.llm_model = "gpt-4"
    args.max_memory = 10
    args.llm_temperature = 0.7
    
    # Mock embedding model
    args.embedding_model = "gpt-4"
    args.embedding_api_key = "sk-test-embedding-key"
    args.retrieval_top_k = 3
    args.retrieval_threshold = 0.1
    args.rag_temperature = 0.4
    args.rag_max_tokens = 400
    
    try:
        from rag_agent import RAGAgent
        agent = RAGAgent(args)
        
        assert agent.guiding_prompt == "Explore systematically and examine objects to make progress."
        assert agent.retrieval_top_k == 3
        assert agent.retrieval_threshold == 0.1
        assert agent.embedding_model == "gpt-4"
        assert agent.embedding_api_key == "sk-test-embedding-key"
        assert agent.history_embeddings is None
        assert len(agent.game_history) == 0
        
        print("✓ RAGAgent initialization successful")
        return True
    except Exception as e:
        print(f"✗ RAGAgent initialization failed: {e}")
        return False

def test_history_management():
    """Test game history management."""
    print("Testing history management...")
    
    args = Mock()
    args.llm_model = "gpt-4"
    args.max_memory = 10
    args.llm_temperature = 0.7
    args.embedding_model = "gpt-4"
    args.embedding_api_key = "sk-test-embedding-key"
    args.retrieval_top_k = 3
    args.retrieval_threshold = 0.1
    args.rag_temperature = 0.4
    args.rag_max_tokens = 400
    
    try:
        from rag_agent import RAGAgent
        agent = RAGAgent(args)
        
        # Test adding to game history
        agent._add_to_game_history("You are in a dark room", "look", "You see a door")
        assert len(agent.game_history) == 1
        assert agent.game_history[0]['state'] == "You are in a dark room"
        assert agent.game_history[0]['action'] == "look"
        
        # Test updating reward
        agent.update_game_history_reward(10, 100)
        assert agent.game_history[0]['reward'] == 10
        assert agent.game_history[0]['score'] == 100
        
        print("✓ History management successful")
        return True
    except Exception as e:
        print(f"✗ History management failed: {e}")
        return False

def test_entity_extraction():
    """Test entity extraction functionality."""
    print("Testing entity extraction...")
    
    args = Mock()
    args.llm_model = "gpt-4"
    args.max_memory = 10
    args.llm_temperature = 0.7
    args.embedding_model = "gpt-4"
    args.embedding_api_key = "sk-test-embedding-key"
    args.retrieval_top_k = 3
    args.retrieval_threshold = 0.1
    args.rag_temperature = 0.4
    args.rag_max_tokens = 400
    
    try:
        from rag_agent import RAGAgent
        agent = RAGAgent(args)
        
        # Test entity extraction
        text = "You are in a dark room with a wooden door and a rusty key on the table"
        entities = agent._extract_key_entities(text)
        
        # Should extract some entities
        assert len(entities) > 0
        assert any('door' in entity for entity in entities)
        assert any('key' in entity for entity in entities)
        assert any('table' in entity for entity in entities)
        
        print("✓ Entity extraction successful")
        return True
    except Exception as e:
        print(f"✗ Entity extraction failed: {e}")
        return False

def test_retrieval_index_update():
    """Test retrieval index update functionality."""
    print("Testing retrieval index update...")
    
    args = Mock()
    args.llm_model = "gpt-4"
    args.max_memory = 10
    args.llm_temperature = 0.7
    args.embedding_model = "gpt-4"
    args.embedding_api_key = "sk-test-embedding-key"
    args.retrieval_top_k = 3
    args.retrieval_threshold = 0.1
    args.rag_temperature = 0.4
    args.rag_max_tokens = 400
    
    try:
        from rag_agent import RAGAgent
        agent = RAGAgent(args)
        
        # Add some game history
        agent._add_to_game_history("You are in a dark room", "look", "You see a door")
        agent._add_to_game_history("You see a wooden door", "examine door", "The door is locked")
        
        # Mock the embedding function to return fake embeddings
        with patch.object(agent, '_get_embedding', return_value=np.random.rand(1536)):
            agent._update_retrieval_index()
            
            assert agent.history_embeddings is not None
            assert agent.history_embeddings.shape[0] == 2  # 2 history entries
            assert agent.history_embeddings.shape[1] == 1536  # embedding dimension
            
        print("✓ Retrieval index update successful")
        return True
    except Exception as e:
        print(f"✗ Retrieval index update failed: {e}")
        return False

def test_prompt_generation():
    """Test prompt generation functionality."""
    print("Testing prompt generation...")
    
    args = Mock()
    args.llm_model = "gpt-4"
    args.max_memory = 10
    args.llm_temperature = 0.7
    args.embedding_model = "gpt-4"
    args.embedding_api_key = "sk-test-embedding-key"
    args.retrieval_top_k = 3
    args.retrieval_threshold = 0.1
    args.rag_temperature = 0.4
    args.rag_max_tokens = 400
    
    try:
        from rag_agent import RAGAgent
        agent = RAGAgent(args)
        
        # Test basic prompt generation
        sys_prompt, user_prompt = agent._generate_basic_prompt("You are in a dark room")
        
        assert "You are an expert player" in sys_prompt
        assert "Your current state is: You are in a dark room" in user_prompt
        assert "REASONING:" in user_prompt
        assert "ACTION:" in user_prompt
        
        print("✓ Prompt generation successful")
        return True
    except Exception as e:
        print(f"✗ Prompt generation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing RAGAgent with LM embeddings...\n")
    
    tests = [
        test_rag_agent_initialization,
        test_history_management,
        test_entity_extraction,
        test_retrieval_index_update,
        test_prompt_generation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! RAGAgent with LM embeddings is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    main() 