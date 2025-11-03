import os
import argparse
from src.evaluation import GameEvaluator


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Game
    parser.add_argument('--rom_path', default='jericho-games/', type=str, help="Path to the directory containing game ROMs.")
    # parser.add_argument('--game_name', default='library', type=str, help="Name of the game to play (e.g., 'zork1', 'library').")
    parser.add_argument('--game_name', default='detective', type=str, help="Name of the game to play (e.g., 'zork1', 'library').")

    parser.add_argument('--output_path', default='output', type=str, help="Base directory for all output, logs, and outputs.") 
    parser.add_argument('--env_step_limit', default=110, type=int, help="Maximum number of steps per game episode.")
    parser.add_argument('--seed', default=0, type=int, help="Random seed for reproducibility. If None, a random seed is used.")

    # LLM
    # parser.add_argument('--llm_model', default='anthropic/claude-4-sonnet-20250522', type=str, help="LLM model for the game-playing agent.")
    parser.add_argument('--llm_model', default='google/gemini-2.5-flash', type=str, help="LLM model for the game-playing agent.")
    # parser.add_argument('--llm_model', default='openai/gpt-4o-mini', type=str, help="LLM model for the game-playing agent.")
    # parser.add_argument('--llm_model', default='openai/gpt-5', type=str, help="LLM model for the game-playing agent.")
    # parser.add_argument('--llm_model', default='openai/gpt-5-mini', type=str, help="LLM model for the game-playing agent.")
    # parser.add_argument('--llm_model', default='openai/gpt-oss-120b', type=str, help="LLM model for the game-playing agent.")


    parser.add_argument('--llm_temperature', default=0.4, type=float, help="Temperature for the agent's LLM.")
    parser.add_argument('--max_memory', default=30, type=int, help="Maximum number of past states to keep in memory for the agent.")

    # Debug options
    parser.add_argument('--debug_info', default=False, action=argparse.BooleanOptionalAction, help='Print detailed info updates during game episodes.')
    parser.add_argument('--track_valid_changes', default=False, action=argparse.BooleanOptionalAction, help='Track valid action changes (if applicable).')

    # Evaluation parameters
    parser.add_argument('--agent_type', type=str, default='our', choices=['memory', 'our', 'summary', 'rag', 'naive'], help='Method to evaluate.')
    # parser.add_argument('--agent_type', type=str, default='memory', choices=['memory', 'our'], help='Method to evaluate.')

    parser.add_argument('--eval_runs', type=int, default=50, help='Number of episodes to run for statistical evaluation.')
    parser.add_argument('--evol_temperature', default=0.7, type=float, help="Temperature for the evolutionary's LLM.")

    # Summary agent parameters
    # parser.add_argument('--summary_llm_model', type=str, help='LLM model for summarization (defaults to game LLM if not specified).')
    parser.add_argument('--summary_temperature', type=float, default=0.3, help='Temperature for the summarization LLM.')
    parser.add_argument('--summary_max_tokens', type=int, default=300, help='Maximum tokens for summarization response.')

    # RAG agent parameters
    parser.add_argument('--retrieval_top_k', type=int, default=3, help='Number of top-k most relevant history entries to retrieve.')
    parser.add_argument('--retrieval_threshold', type=float, default=0.1, help='Similarity threshold for retrieving relevant history entries.')
    # parser.add_argument('--embedding_model', type=str, help='LLM model for RAG enhancement (defaults to game LLM if not specified).')
    parser.add_argument('--rag_temperature', type=float, default=0.4, help='Temperature for the RAG enhancement LLM.')
    parser.add_argument('--rag_max_tokens', type=int, default=400, help='Maximum tokens for RAG enhancement response.')

    # Evolutionary parameters (used by EvolutionaryPrompter and for 'evolved' evaluation)
    parser.add_argument('--evolution_llm_model', default='openai/o3-2025-04-16', type=str, help='LLM model for the evolutionary operator.')
    parser.add_argument('--initial_prompts_file', default='initial_prompts.json', type=str, help='JSON file with initial prompts to seed the pool (relative to project root or absolute).')
    parser.add_argument('--exploration_constant', default=1.0, type=float, help='Exploration constant for UCB calculation in tree-based agent.')
    parser.add_argument('--depth_constant', default=0.8, type=float, help='Decay factor of exploration term in tree-based agent.')

    parser.add_argument('--freeze_on_win', default=True, action=argparse.BooleanOptionalAction,
                    help='Once any node reaches win_freeze_threshold, stop exploration and reuse the best prompt thereafter.')
    parser.add_argument('--win_freeze_threshold', type=int, default=0,
                    help='Score threshold to freeze on win (e.g., 310 for detective). 0 disables freezing.')
    parser.add_argument('--force_best_after_drop', default=True, action=argparse.BooleanOptionalAction,
                    help='If the last episode score drops far below best, force exploiting the best prompt next episode.')
    parser.add_argument('--drop_threshold', type=int, default=50,
                    help='Score drop margin vs best to trigger forced exploit.')

    # Cross-episode memory toggle (few-shot positives + negative contrast during evolution)
    parser.add_argument('--enable_cross_mem', default=True, action=argparse.BooleanOptionalAction,
                    help='Enable cross-episode memory: store successful/failed snippets across episodes, few-shot retrieval, and negative-contrast evolution.')
 
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    
    evaluator = GameEvaluator(args)
    results = evaluator.run_evaluation()
    
    # Exit with appropriate code
    if results.get("success", False):
        print("Evaluation completed successfully!")
        exit(0)
    else:
        print("Evaluation failed!")
        exit(1)