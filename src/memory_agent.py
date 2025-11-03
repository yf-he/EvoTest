from .openai_helpers import chat_completion_with_retries

class MemoryAgent:
    def __init__(self, args, guiding_prompt: str = None): 
        self.guiding_prompt = guiding_prompt or "Explore systematically and examine objects to make progress."
        self.memory = [] # Used by agent
        self.args = args
        
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
    
    def start_episode(self):
        """
        """
        self.memory = []
        print(f"Using initial prompt: '{self.guiding_prompt}'")

    def end_episode(self, state, score):
        """
        End an episode: update the current node's score and game history.
        """
        print(f"Ending episode with score: {score}.")

    def get_prompts(self, state_node):
        memory_text = self._format_memory_for_prompt()
        
        sys_prompt = """You are an expert player aiming to complete a text-based adventure game. Points are given for making progress in the game. Select promising actions based on the game state and memory of past interactions."""
        if self.guiding_prompt:
            sys_prompt += f"\n\nFollow this guide: {self.guiding_prompt}"

        if memory_text:
            sys_prompt += f"\n\nYour memory of past states and actions:\n{memory_text}"

        user_prompt = f"""
\n\nYour current state is: {state_node.state} \n\n

Type your next action as if you were playing the game directly. It should be a short command that can be understood by the game parser. Common actions include: look, inventory, directions (north, northeast, up, etc.), examine X, say X, drop X, get X, open X, enter X, ask X about Y, look in X, give X to Y, and other context-specific commands. To avoid parsing errors, any such X or Y MUST be a *SINGLE WORD* that identifies an object in a way the game can understand. When stuck, explore all rooms and objects mentioned in room descriptions systematically and comprehensively. *DO NOT REPEAT* the same failed action multiple times, as it will not lead to new results. Do not use the "help" command.

Your response MUST strictly follow this format and include nothing else:
REASONING: [A short, concise explanation of your choice, 1-2 sentences]
ACTION: [short word or phrase for text command to execute]

For example:
REASONING: I should examine the book to learn more about it.
ACTION: examine book
"""
        return sys_prompt, user_prompt

    # Generates the next action from the LLM based on its memory and the current state node.
    def generate_action(self, state_node):
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