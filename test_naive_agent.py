#!/usr/bin/env python3
"""
Test script for NaiveAgent.
This script tests the basic functionality without making actual LLM API calls.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from unittest.mock import Mock

def test_naive_agent_initialization():
    """Test NaiveAgent initialization."""
    print("Testing NaiveAgent initialization...")
    
    # Mock args
    args = Mock()
    args.llm_model = "gpt-4"
    args.llm_temperature = 0.7
    
    try:
        from naive_agent import NaiveAgent
        agent = NaiveAgent(args)
        
        assert agent.guiding_prompt == "Explore systematically and examine objects to make progress."
        assert agent.args == args
        
        print("✓ NaiveAgent initialization successful")
        return True
    except Exception as e:
        print(f"✗ NaiveAgent initialization failed: {e}")
        return False

def test_memory_methods():
    """Test that memory methods do nothing (as expected for naive agent)."""
    print("Testing memory methods...")
    
    args = Mock()
    args.llm_model = "gpt-4"
    args.llm_temperature = 0.7
    
    try:
        from naive_agent import NaiveAgent
        agent = NaiveAgent(args)
        
        # Test that memory methods exist but do nothing
        agent.add_to_memory("test state", "test response")  # Should not crash
        memory_text = agent._format_memory_for_prompt()  # Should return empty string
        assert memory_text == ""
        
        print("✓ Memory methods work as expected (do nothing)")
        return True
    except Exception as e:
        print(f"✗ Memory methods test failed: {e}")
        return False

def test_episode_management():
    """Test episode start/end methods."""
    print("Testing episode management...")
    
    args = Mock()
    args.llm_model = "gpt-4"
    args.llm_temperature = 0.7
    
    try:
        from naive_agent import NaiveAgent
        agent = NaiveAgent(args)
        
        # Test episode methods
        agent.start_episode()  # Should not crash
        agent.end_episode("final state", 100)  # Should not crash
        
        print("✓ Episode management methods work correctly")
        return True
    except Exception as e:
        print(f"✗ Episode management test failed: {e}")
        return False

def test_prompt_generation():
    """Test prompt generation functionality."""
    print("Testing prompt generation...")
    
    args = Mock()
    args.llm_model = "gpt-4"
    args.llm_temperature = 0.7
    
    try:
        from naive_agent import NaiveAgent
        agent = NaiveAgent(args)
        
        # Test basic prompt generation
        sys_prompt, user_prompt = agent._generate_basic_prompt("You are in a dark room")
        
        assert "You are an expert player" in sys_prompt
        assert "Your current state is: You are in a dark room" in user_prompt
        assert "REASONING:" in user_prompt
        assert "ACTION:" in user_prompt
        assert "current game state only" in sys_prompt  # Key difference from other agents
        
        print("✓ Prompt generation successful")
        return True
    except Exception as e:
        print(f"✗ Prompt generation failed: {e}")
        return False

def test_response_parsing():
    """Test LLM response parsing."""
    print("Testing response parsing...")
    
    args = Mock()
    args.llm_model = "gpt-4"
    args.llm_temperature = 0.7
    
    try:
        from naive_agent import NaiveAgent
        agent = NaiveAgent(args)
        
        # Test response parsing
        test_response = "REASONING: I should examine the door.\nACTION: examine door"
        action = agent._parse_llm_response(test_response)
        assert action == "examine door"
        
        # Test with malformed response
        malformed_response = "Just some random text"
        action = agent._parse_llm_response(malformed_response)
        assert action == "look"  # Default action
        
        print("✓ Response parsing successful")
        return True
    except Exception as e:
        print(f"✗ Response parsing failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing NaiveAgent...\n")
    
    tests = [
        test_naive_agent_initialization,
        test_memory_methods,
        test_episode_management,
        test_prompt_generation,
        test_response_parsing
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! NaiveAgent is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    main() 