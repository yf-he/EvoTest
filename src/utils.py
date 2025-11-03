import numpy as np
import json

def softmax(a, T=1):
    a = np.array(a) / T
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def game_file(game_name):
    rom_dict = {'zork1': 'zork1.z5', 
                'zork3': 'zork3.z5', 
                'spellbrkr' : 'spellbrkr.z3',
                'advent': 'advent.z5',                 
                'detective': 'detective.z5', 
                'pentari': 'pentari.z5',
                'enchanter': 'enchanter.z3',
                'library' : 'library.z5',
                'balances' : 'balances.z5',
                'ztuu' : 'ztuu.z5',
                'ludicorp' : 'ludicorp.z5',
                'deephome' : 'deephome.z5',
                'temple' : 'temple.z5',
                'anchor' : 'anchor.z8',
                'awaken' : 'awaken.z5',
                'zenon' : 'zenon.z5'
                }
                
    return rom_dict[game_name]

# Placeholder for initial prompt loading utilities
# These should be implemented based on the actual structure of initial_prompts.json
# and how generic prompts are defined.

def load_initial_prompts(file_path):
    """
    Loads initial prompts from a JSON file.
    Expects a JSON file with a "prompts" key, which is a list of strings.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        if "prompts" not in data or not isinstance(data["prompts"], list):
            print(f"Warning: Prompt file {file_path} is missing 'prompts' list or it's not a list. Returning empty list.")
            return []
        return data["prompts"]
    except FileNotFoundError:
        print(f"Warning: Initial prompts file {file_path} not found. Returning empty list.")
        return []
    except json.JSONDecodeError:
        print(f"Warning: Error decoding JSON from {file_path}. Returning empty list.")
        return []

def get_generic_initial_prompts(num_prompts=1):
    """
    Returns a list of generic initial prompts.
    """
    # This is a basic implementation. You might want to make these more sophisticated.
    generic_prompts = [
        "Analyze the game state and choose the best action to maximize the score.",
        "Think step-by-step to understand the current situation and decide the next move.",
        "Your goal is to achieve the highest score. Observe, think, and act.",
        "Explore the environment, interact with objects, and solve puzzles to progress.",
        "Be methodical. Consider all available actions and their potential outcomes."
    ]
    return generic_prompts[:num_prompts]