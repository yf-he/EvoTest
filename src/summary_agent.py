from .openai_helpers import chat_completion_with_retries

class SummaryAgent:
    def __init__(self, args, guiding_prompt: str = None): 
        self.guiding_prompt = guiding_prompt or "Explore systematically and examine objects to make progress."
        self.memory = [] # Used by agent for recent context
        self.game_history = [] # Used for LLM summarization
        self.args = args
        
        # LLM model for summarization (can be different from the game LLM)
        self.summary_llm_model = getattr(args, 'summary_llm_model', args.llm_model)
        self.summary_temperature = getattr(args, 'summary_temperature', 0.3)
        self.summary_max_tokens = getattr(args, 'summary_max_tokens', 300)
        
    def add_to_memory(self, state, response):
        memory_entry = {"state": state, "response": response}
        self.memory.append(memory_entry)
        if len(self.memory) > self.args.max_memory:
            self.memory.pop(0)  # Remove oldest entry if exceeding max_memory
    
    def _format_memory_for_prompt(self):
        if not self.memory:
            return ""
            
        memory_text = "MEMORY (Recent few states and agent's responses):\n"
        for i, entry in enumerate(self.memory):
            memory_text += f"Memory {i+1}:\n"
            memory_text += f"STATE: {entry['state']}\n"
            if entry['response']:
                memory_text += f"AGENT'S RESPONSE: {entry['response']}\n"
        
        return memory_text
    
    def _format_game_history_for_summary(self, history):
        """
        Format game history for LLM summarization.
        Returns a clean, structured format for the LLM to analyze.
        """
        if not history:
            return "No game history available."
            
        history_str = "GAME HISTORY:\n"
        for i, entry in enumerate(history):
            history_str += f"Step {i+1}:\n"
            history_str += f"STATE: {entry['state']}\n"
            if 'action' in entry and entry['action']:
                history_str += f"ACTION: {entry['action']}\n"
            if 'reward' in entry and entry['reward'] is not None:
                history_str += f"REWARD: {entry['reward']}\n"
            if 'score' in entry and entry['score'] is not None:
                history_str += f"SCORE: {entry['score']}\n"
            history_str += "------------\n"
        
        return history_str
    
    def _summarize_game_history(self, game_history):
        """
        Use LLM to summarize the game history and extract key progress information.
        Returns a concise summary of current game state and progress.
        """
        if not game_history:
            return "Game just started."
        
        try:
            history_str = self._format_game_history_for_summary(game_history)
            
            sys_prompt = """You are an expert at analyzing text adventure game progress. Your task is to provide a concise, informative summary of the current game state and progress made so far.

Focus on:
1. Current location and surroundings
2. Important items collected or obtained
3. Key obstacles overcome or puzzles solved
4. Significant progress made toward game objectives
5. Any hints, clues, or important information discovered

Keep the summary concise (2-3 sentences) but informative. Focus on what the agent has accomplished and what the current situation is."""

            user_prompt = f"""Analyze this text adventure game history and provide a concise summary of current progress:

{history_str}

Provide a 2-3 sentence summary of the current game state and progress:"""

            response = chat_completion_with_retries(
                model=self.summary_llm_model,
                sys_prompt=sys_prompt,
                prompt=user_prompt,
                max_tokens=self.summary_max_tokens,
                temperature=self.summary_temperature,
            )

            if response and hasattr(response, 'choices') and response.choices and response.choices[0].message:
                summary = response.choices[0].message.content.strip()
                return summary
            else:
                print("Warning: LLM summarization failed, using fallback summary")
                return self._fallback_summary(game_history)
                
        except Exception as e:
            print(f"Error during game history summarization: {e}")
            return self._fallback_summary(game_history)
    
    def _fallback_summary(self, game_history):
        """
        Fallback summary method when LLM summarization fails.
        Provides a basic summary based on simple heuristics.
        """
        if not game_history:
            return "Game just started."
        
        # Simple heuristics for fallback summary
        total_steps = len(game_history)
        last_state = game_history[-1]['state'] if game_history else ""
        
        # Check for common progress indicators
        progress_indicators = []
        for entry in game_history:
            state_lower = entry['state'].lower()
            if any(keyword in state_lower for keyword in ['key', 'door', 'open', 'unlock']):
                progress_indicators.append("found key")
            if any(keyword in state_lower for keyword in ['treasure', 'gold', 'coin']):
                progress_indicators.append("found treasure")
            if any(keyword in state_lower for keyword in ['defeat', 'kill', 'win']):
                progress_indicators.append("defeated enemy")
        
        if progress_indicators:
            unique_progress = list(set(progress_indicators))
            return f"Completed {total_steps} steps. Progress: {', '.join(unique_progress)}. Current: {last_state[:100]}..."
        else:
            return f"Completed {total_steps} steps. Currently exploring. Last state: {last_state[:100]}..."
    
    def start_episode(self):
        """
        Start a new episode: clear memory and game history.
        """
        self.memory = []
        self.game_history = []
        print(f"Using initial prompt: '{self.guiding_prompt}'")

    def end_episode(self, state, score):
        """
        End an episode: update the last game history entry with final score.
        """
        if self.game_history:
            self.game_history[-1]['score'] = score
        print(f"Ending episode with score: {score}.")

    def get_prompts(self, state_node):
        memory_text = self._format_memory_for_prompt()
        
        # Generate LLM summary of game history
        game_summary = self._summarize_game_history(self.game_history)
        
        sys_prompt = """You are an expert player aiming to complete a text-based adventure game. Points are given for making progress in the game. Select promising actions based on the game state, memory of past interactions, and summary of overall progress."""
        if self.guiding_prompt:
            sys_prompt += f"\n\nFollow this guide: {self.guiding_prompt}"

        user_prompt = f"""
GAME PROGRESS SUMMARY: {game_summary}

{memory_text}

Your current state is: {state_node.state}

Type your next action as if you were playing the game directly. It should be a short command that can be understood by the game parser. Common actions include: look, inventory, directions (north, northeast, up, etc.), examine X, say X, drop X, get X, open X, enter X, ask X about Y, look in X, give X to Y, and other context-specific commands. To avoid parsing errors, any such X or Y MUST be a *SINGLE WORD* that identifies an object in a way the game can understand. When stuck, explore all rooms and objects mentioned in room descriptions systematically and comprehensively. *DO NOT REPEAT* the same failed action multiple times, as it will not lead to new results. Do not use the "help" command.

Your response MUST strictly follow this format and include nothing else:
REASONING: [A short, concise explanation of your choice, 1-2 sentences]
ACTION: [short word or phrase for text command to execute]

For example:
REASONING: I should examine the book to learn more about it.
ACTION: examine book
"""
        return sys_prompt, user_prompt

    def generate_action(self, state_node):
        """
        Generates the next action from the LLM based on its memory, game history summary, and the current state node.
        """
        sys_prompt, user_prompt = self.get_prompts(state_node)
        
        res_obj = chat_completion_with_retries(
            model=self.args.llm_model,
            sys_prompt=sys_prompt,
            prompt=user_prompt,
            max_tokens=400,
            temperature=self.args.llm_temperature,
        )

        if res_obj and hasattr(res_obj, 'choices') and res_obj.choices and res_obj.choices[0].message:
            full_response = res_obj.choices[0].message.content
            action_text = self._parse_llm_response(full_response)
        else:
            print(f"Warning: LLM API call might have failed or returned empty. Defaulting action.")
            full_response = ""
            action_text = "look" # Default action
            
        self.add_to_memory(state_node.state, full_response)
        self._add_to_game_history(state_node.state, action_text, full_response)
        
        return action_text.strip(), full_response

    def _parse_llm_response(self, full_response: str):
        """
        Parses the LLM's full string response to extract action.
        """
        action_text = "look" # Default action

        if not full_response or not isinstance(full_response, str):
            return action_text

        lines = full_response.strip().split('\n')
        try:
            for line in lines:
                if line.upper().startswith("ACTION:"):
                    action_text = line.split(":", 1)[1].strip()
        except Exception as e:
            print(f"Error parsing LLM response: {e}. Response was: '{full_response}'")

        return action_text
    
    def _add_to_game_history(self, state, action, full_response, reward=None, score=None):
        """
        Add a new entry to the game history.
        """
        self.game_history.append({
            "state": state,
            "action": action,
            "full_response": full_response,
            "reward": reward,
            "score": score
        })
    
    def update_game_history_reward(self, reward, score):
        """Update the last entry in game history with reward and score"""
        if self.game_history and len(self.game_history) > 0:
            self.game_history[-1]["reward"] = reward
            self.game_history[-1]["score"] = score 