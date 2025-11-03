## EVOTEST: Evolutionary Test-Time Learning for Self-Improving Agentic Systems

EVOTEST is an evolutionary test-time learning framework that improves an agent across episodes without gradients or fine-tuning. It evolves the entire agentic system between attempts by rewriting prompts, updating cross-episode memory, tuning hyperparameters, and refining tool-use routines.

This repository also provides J-TTL (Jericho Test-Time Learning), a benchmark setting where an agent plays the same text adventure game for multiple consecutive episodes and must improve using only within-session experience.

### Paper
- Title: EVOTEST: Evolutionary Test-Time Learning for Self-Improving Agentic Systems
- Authors: Yufei He, Juncheng Liu, Yue Liu, Yibo Li, Tri Cao, Zhiyuan Hu, Xinxing Xu, Bryan Hooi
- Affiliations: National University of Singapore, Microsoft Research

---

## Highlights
- **Benchmark (J-TTL)**: Measures on-the-fly learning across repeated episodes of the same Jericho game.
- **Method (EVOTEST)**: Evolves prompts, code-based state extractors, cross-episode memory, and hyperparameters after each episode—no training required.
- **Results**: Consistent improvements across games, outperforming reflection-, memory-, and gradient-based online methods; uniquely achieves wins on Detective and Library in our evaluations.

---

## Repository Structure
- `main.py`: Entry point for running agents and evaluations.
- `src/`:
  - `evaluation.py`: Evaluation loop over episodes; logging and metrics.
  - `env.py`: Jericho environment wrapper using `FrotzEnv`.
  - `our_agent.py`: EVOTEST Actor/Evolver agent with cross-episode memory and evolutionary prompt/code updates.
  - `summary_agent.py`: Agent variant using LLM-generated summaries.
  - `memory_agent.py`, `rag_agent.py`, `naive_agent.py`: Baseline agents (memory-only, RAG-enhanced, and naive).
  - `openai_helpers.py`: OpenRouter/OpenAI client with retry and token utilities.
  - `utils.py`: Small helpers (e.g., ROM file resolution).
- `jericho-games/`: Game ROMs directory (e.g., `zork1.z5`, `detective.z5`, etc.).
- `*.qzl`: Pre-bundled game files for convenience in some setups.
- `test_*.py`: Pytest-based checks for naive, RAG, and RAG embeddings agents.
- `requirements_rag.txt`: Minimal dependencies for RAG agent; see install notes below for full setup.

---

## Installation

### 1) Python and virtual environment
- Python 3.10+ recommended.
- Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # on macOS/Linux
# .venv\Scripts\activate  # on Windows PowerShell
```

### 2) Dependencies
Install minimal requirements, then Jericho and supporting libs used by the agents:

```bash
pip install -r requirements_rag.txt
pip install jericho tiktoken python-dotenv
```

If you plan to use RAG with larger embedding backends or vector indices:

```bash
# optional extras
pip install sentence-transformers faiss-cpu
```

Note: On Apple Silicon, `faiss-cpu` wheels may vary; if issues arise, skip FAISS or install via conda-forge.

### 3) LLM API via OpenRouter
Create a `.env` file in the project root with your OpenRouter key (get it from `https://openrouter.ai/settings/keys`):

```bash
echo 'OPENAI_API_KEY="sk-..."' > .env
```

All LLM calls route through OpenRouter by default using `openai_helpers.py` (OpenAI SDK with custom base URL). Some OpenAI models on OpenRouter may require additional consent on the OpenRouter dashboard.

---

## Getting Game ROMs
Jericho requires valid Z-machine game files (e.g., `.z5`). This repo includes a `jericho-games/` folder. Ensure the game ROM you want to evaluate exists there. You can also point `--rom_path` to a custom directory.

---

## Quick Start
Run EVOTEST on the Detective game for 10 episodes using a fast model:

```bash
python main.py \
  --game_name detective \
  --rom_path jericho-games/ \
  --agent_type our \
  --llm_model google/gemini-2.5-flash \
  --eval_runs 10 \
  --env_step_limit 110 \
  --llm_temperature 0.4 \
  --evol_temperature 0.7
```

Change `--llm_model` to any OpenRouter-supported model (e.g., `openai/gpt-4o-mini`, `anthropic/claude-4-sonnet-20250522`).

Episodes, summaries, and metrics are written to:

```
output/<game>/<agent_type>/<model>/<timestamp>/
```

Each episode log includes step-by-step observations, chosen actions, rewards, and cumulative scores.

---

## Agents
- **our**: EVOTEST with evolutionary updates (prompt and code state extractor), optional cross-episode memory, UCB-based node selection, and auto-freeze on wins.
- **memory**: Memory-only baseline with recent context window.
- **summary**: Uses an LLM to summarize progress and feed it into action selection.
- **rag**: Retrieves similar prior states/actions (cross-episode positives) to guide decisions.
- **naive**: Minimal baseline issuing generic exploratory actions.

Select with `--agent_type {our,memory,summary,rag,naive}`.

---

## Key EVOTEST Features
- **Whole-system evolution between episodes**: After each run, an Evolver LLM proposes a revised guiding prompt and a Python state extractor (`extract_state(game_history)`) specialized to the current game.
- **Cross-episode memory (optional)**: Stores successful (state → action, +Δscore) and negative loop patterns to encourage progress and avoid plateaus.
- **UCB-driven exploration vs. exploitation**: Chooses which evolved node to try; can auto-freeze on detected wins and resume evolution if a frozen prompt later fails.
- **Tooling**: Explicit control over temperatures for acting, summarization, RAG, and evolution.

Relevant flags in `main.py`:
- `--evolution_llm_model`: Model for the Evolver (default `openai/o3-2025-04-16`).
- `--freeze_on_win`, `--win_freeze_threshold`, `--force_best_after_drop`, `--drop_threshold`.
- `--enable_cross_mem`: Toggle cross-episode memory and negative-contrast evolution.
- `--exploration_constant`, `--depth_constant`: UCB and depth decay controls.

---

## Reproducing J-TTL Evaluations
Example (50 episodes for statistics, EVOTEST on Detective):

```bash
python main.py \
  --game_name detective \
  --rom_path jericho-games/ \
  --agent_type our \
  --eval_runs 50
```

Switch games (e.g., `library`, `zork1`) and agents via `--game_name` and `--agent_type`. Use a consistent `--seed` for reproducibility.

---

## Testing
Run the included tests (naive, RAG, embeddings checks):

```bash
pytest -q
```

---

## Troubleshooting
- "ModuleNotFoundError: jericho": `pip install jericho` (or install via conda if needed).
- OpenRouter errors/rate limits: ensure `.env` has `OPENAI_API_KEY`, and the selected `--llm_model` is enabled on your OpenRouter account.
- FAISS install on macOS: prefer `faiss-cpu` via conda-forge or skip FAISS if not using large RAG indices.
- No score improvements: try increasing `--eval_runs`, enable `--enable_cross_mem`, or use a stronger `--evolution_llm_model`.

---

## Citation
If you find this repository useful, please cite the paper:

```bibtex
@article{he2025evotest,
  title={EvoTest: Evolutionary Test-Time Learning for Self-Improving Agentic Systems},
  author={He, Yufei and Liu, Juncheng and Liu, Yue and Li, Yibo and Cao, Tri and Hu, Zhiyuan and Xu, Xinxing and Hooi, Bryan},
  journal={arXiv preprint arXiv:2510.13220},
  year={2025}
}
```

---

## License
This project is licensed under the terms of the LICENSE file in the repository.

---

## Acknowledgements
- Jericho: the interactive fiction environment used in J-TTL.
- OpenRouter: unified interface for accessing multiple foundation models.
