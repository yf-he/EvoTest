from .openai_helpers import chat_completion_with_retries
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

class RAGAgent:
    def __init__(self, args, guiding_prompt: str = None): 
        self.guiding_prompt = guiding_prompt or "Explore systematically and examine objects to make progress."
        self.memory = [] # Used by agent for recent context
        self.game_history = [] # Used for RAG retrieval
        self.args = args
        
        # RAG parameters
        self.retrieval_top_k = getattr(args, 'retrieval_top_k', 3)
        self.retrieval_threshold = getattr(args, 'retrieval_threshold', 0.1)
        self.embedding_model = getattr(args, 'embedding_model', args.llm_model)
        self.rag_temperature = getattr(args, 'rag_temperature', 0.4)
        self.rag_max_tokens = getattr(args, 'rag_max_tokens', 400)
        
        # Separate API key for embeddings
        self.embedding_api_key = getattr(args, 'embedding_api_key', None)
        
        # Initialize embedding storage
        self.history_embeddings = None
        self.history_texts = []
        
    def _get_embedding(self, text):
        """
        Get embedding for text using OpenAI's embedding API.
        """
        try:
            # Use OpenAI's text-embedding-3-small model for embeddings
            import openai
            
            # Use separate API key for embeddings if provided
            if self.embedding_api_key:
                client = openai.OpenAI(api_key=self.embedding_api_key)
            else:
                client = openai.OpenAI()
                
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            # Fallback: return random embedding (not ideal but prevents crashes)
            return np.random.rand(1536)  # text-embedding-3-small dimension
    
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
    
    def _extract_key_entities(self, text):
        """
        Extract key entities and concepts from text for better retrieval.
        """
        # Simple entity extraction - can be enhanced with NER models
        entities = []
        
        # Extract location-like entities (words that might be locations)
        location_patterns = [
            r'\b(room|hall|corridor|chamber|tunnel|passage|area|place)\b',
            r'\b(north|south|east|west|up|down|inside|outside)\b'
        ]
        
        # Extract object-like entities
        object_patterns = [
            r'\b(key|door|book|chest|sword|potion|map|treasure|coin|gem)\b',
            r'\b(table|chair|bed|fireplace|window|stairs|ladder)\b'
        ]
        
        # Extract action-like entities
        action_patterns = [
            r'\b(open|close|pick|drop|examine|read|use|give|take|move)\b'
        ]
        
        for pattern in location_patterns + object_patterns + action_patterns:
            matches = re.findall(pattern, text.lower())
            entities.extend(matches)
        
        return list(set(entities))
    
    def _update_retrieval_index(self):
        """
        Update the retrieval index with current game history.
        """
        if not self.game_history:
            self.history_embeddings = None
            self.history_texts = []
            return
        
        # Prepare texts for embedding
        self.history_texts = []
        for entry in self.game_history:
            # Combine state, action, and response for better retrieval
            text_parts = [entry['state']]
            if 'action' in entry and entry['action']:
                text_parts.append(f"Action: {entry['action']}")
            if 'full_response' in entry and entry['full_response']:
                text_parts.append(f"Response: {entry['full_response']}")
            
            # Extract key entities and add them
            entities = self._extract_key_entities(entry['state'])
            if entities:
                text_parts.append(f"Entities: {', '.join(entities)}")
            
            combined_text = " | ".join(text_parts)
            self.history_texts.append(combined_text)
        
        # Update embeddings
        if len(self.history_texts) > 0:
            try:
                # Get embeddings for each text individually
                embeddings = []
                for text in self.history_texts:
                    embedding = self._get_embedding(text)
                    embeddings.append(embedding)
                
                # Convert to numpy array
                self.history_embeddings = np.array(embeddings)
            except Exception as e:
                print(f"Error getting embeddings: {e}")
                # Fallback if embedding fails
                self.history_embeddings = None
        else:
            self.history_embeddings = None
    
    def _retrieve_relevant_history(self, current_state, top_k=None):
        """
        Retrieve the most relevant history entries based on current state.
        """
        if not self.game_history or self.history_embeddings is None or (hasattr(self.history_embeddings, 'shape') and self.history_embeddings.shape[0] == 0):
            return []
        
        top_k = top_k or self.retrieval_top_k
        
        try:
            # Get embedding for current state
            current_embedding = self._get_embedding(current_state)
            current_vector = np.array(current_embedding).reshape(1, -1)
            
            # Calculate similarities
            similarities = cosine_similarity(current_vector, self.history_embeddings).flatten()
            
            # Get top-k most similar entries
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            relevant_entries = []
            for idx in top_indices:
                if similarities[idx] >= self.retrieval_threshold:
                    entry = self.game_history[idx].copy()
                    entry['similarity'] = similarities[idx]
                    relevant_entries.append(entry)
            
            return relevant_entries
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            # Fallback: return recent entries
            return self.game_history[-top_k:] if len(self.game_history) >= top_k else self.game_history
    
    def _format_retrieved_context(self, retrieved_entries):
        """
        Format retrieved entries for inclusion in the prompt.
        """
        if not retrieved_entries:
            return ""
        
        context_text = "RELEVANT GAME HISTORY (Retrieved based on current state):\n"
        
        for i, entry in enumerate(retrieved_entries):
            context_text += f"Relevant Entry {i+1} (Similarity: {entry.get('similarity', 'N/A'):.3f}):\n"
            context_text += f"STATE: {entry['state']}\n"
            if 'action' in entry and entry['action']:
                context_text += f"ACTION: {entry['action']}\n"
            if 'full_response' in entry and entry['full_response']:
                context_text += f"RESPONSE: {entry['full_response']}\n"
            if 'reward' in entry and entry['reward'] is not None:
                context_text += f"REWARD: {entry['reward']}\n"
            context_text += "------------\n"
        
        return context_text
    
    def _generate_rag_enhanced_prompt(self, current_state, retrieved_entries):
        """
        Generate a RAG-enhanced prompt using retrieved context.
        """
        if not retrieved_entries:
            return self._generate_basic_prompt(current_state)
        
        # Use LLM to synthesize retrieved information
        try:
            context_text = self._format_retrieved_context(retrieved_entries)
            
            sys_prompt = """You are an expert at analyzing text adventure game contexts. Your task is to synthesize retrieved game history information into actionable insights for the current game state.

Focus on:
1. What worked or didn't work in similar situations
2. Important patterns or strategies discovered
3. Key items or locations that might be relevant
4. Potential next steps based on past experiences

Provide a concise, actionable summary that the agent can use to make decisions."""

            user_prompt = f"""Based on this retrieved game history context, provide actionable insights for the current state:

CURRENT STATE: {current_state}

RETRIEVED CONTEXT:
{context_text}

Provide 2-3 sentences of actionable insights:"""

            response = chat_completion_with_retries(
                model=self.embedding_model,
                sys_prompt=sys_prompt,
                prompt=user_prompt,
                max_tokens=200,
                temperature=self.rag_temperature,
            )

            if response and hasattr(response, 'choices') and response.choices and response.choices[0].message:
                rag_insights = response.choices[0].message.content.strip()
                return self._generate_enhanced_prompt(current_state, context_text, rag_insights)
            else:
                return self._generate_enhanced_prompt(current_state, context_text, "Using retrieved context directly.")
                
        except Exception as e:
            print(f"Error during RAG enhancement: {e}")
            return self._generate_enhanced_prompt(current_state, self._format_retrieved_context(retrieved_entries), "Using retrieved context directly.")
    
    def _generate_basic_prompt(self, current_state):
        """Generate basic prompt without RAG enhancement."""
        memory_text = self._format_memory_for_prompt()
        
        sys_prompt = """You are an expert player aiming to complete a text-based adventure game. Points are given for making progress in the game. Select promising actions based on the game state and memory of past interactions."""
        if self.guiding_prompt:
            sys_prompt += f"\n\nFollow this guide: {self.guiding_prompt}"

        user_prompt = f"""
{memory_text}

Your current state is: {current_state}

Type your next action as if you were playing the game directly. It should be a short command that can be understood by the game parser. Common actions include: look, inventory, directions (north, northeast, up, etc.), examine X, say X, drop X, get X, open X, enter X, ask X about Y, look in X, give X to Y, and other context-specific commands. To avoid parsing errors, any such X or Y MUST be a *SINGLE WORD* that identifies an object in a way the game can understand. When stuck, explore all rooms and objects mentioned in room descriptions systematically and comprehensively. *DO NOT REPEAT* the same failed action multiple times, as it will not lead to new results. Do not use the "help" command.

Your response MUST strictly follow this format and include nothing else:
REASONING: [A short, concise explanation of your choice, 1-2 sentences]
ACTION: [short word or phrase for text command to execute]

For example:
REASONING: I should examine the book to learn more about it.
ACTION: examine book
"""
        return sys_prompt, user_prompt
    
    def _generate_enhanced_prompt(self, current_state, context_text, rag_insights):
        """Generate enhanced prompt with RAG context."""
        memory_text = self._format_memory_for_prompt()
        
        sys_prompt = """You are an expert player aiming to complete a text-based adventure game. Points are given for making progress in the game. Select promising actions based on the game state, memory of past interactions, and insights from relevant game history."""
        if self.guiding_prompt:
            sys_prompt += f"\n\nFollow this guide: {self.guiding_prompt}"

        user_prompt = f"""
RAG-ENHANCED INSIGHTS: {rag_insights}

{context_text}

{memory_text}

Your current state is: {current_state}

Type your next action as if you were playing the game directly. It should be a short command that can be understood by the game parser. Common actions include: look, inventory, directions (north, northeast, up, etc.), examine X, say X, drop X, get X, open X, enter X, ask X about Y, look in X, give X to Y, and other context-specific commands. To avoid parsing errors, any such X or Y MUST be a *SINGLE WORD* that identifies an object in a way the game can understand. When stuck, explore all rooms and objects mentioned in room descriptions systematically and comprehensively. *DO NOT REPEAT* the same failed action multiple times, as it will not lead to new results. Do not use the "help" command.

Your response MUST strictly follow this format and include nothing else:
REASONING: [A short, concise explanation of your choice, 1-2 sentences]
ACTION: [short word or phrase for text command to execute]

For example:
REASONING: I should examine the book to learn more about it.
ACTION: examine book
"""
        return sys_prompt, user_prompt
    
    def start_episode(self):
        """
        Start a new episode: clear memory and game history.
        """
        self.memory = []
        self.game_history = []
        self._update_retrieval_index()
        print(f"Using initial prompt: '{self.guiding_prompt}'")

    def end_episode(self, state, score):
        """
        End an episode: update the last game history entry with final score.
        """
        if self.game_history:
            self.game_history[-1]['score'] = score
        print(f"Ending episode with score: {score}.")

    def get_prompts(self, state_node):
        """
        Generate prompts using RAG-enhanced context retrieval.
        """
        # Update retrieval index
        self._update_retrieval_index()
        
        # Retrieve relevant history
        retrieved_entries = self._retrieve_relevant_history(state_node.state)
        
        # Generate RAG-enhanced prompt
        if retrieved_entries:
            return self._generate_rag_enhanced_prompt(state_node.state, retrieved_entries)
        else:
            return self._generate_basic_prompt(state_node.state)

    def generate_action(self, state_node):
        """
        Generates the next action using RAG-enhanced context.
        """
        sys_prompt, user_prompt = self.get_prompts(state_node)
        
        res_obj = chat_completion_with_retries(
            model=self.args.llm_model,
            sys_prompt=sys_prompt,
            prompt=user_prompt,
            max_tokens=self.rag_max_tokens,
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