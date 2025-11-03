import os
import json
from typing import List, Dict, Any
from collections import Counter


class CrossEpisodeMemory:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.pos_path = os.path.join(self.base_dir, 'mem.jsonl')
        self.neg_path = os.path.join(self.base_dir, 'neg_mem.jsonl')

    # -------------------- Persistence helpers --------------------
    def _append_jsonl(self, path: str, obj: Dict[str, Any]):
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _load_jsonl(self, path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            return []
        items = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception:
                    continue
        return items

    # -------------------- Positive examples --------------------
    def add_positive(self, state_text: str, action_text: str, delta_score: float, extra: Dict[str, Any] = None):
        item = {
            'state': state_text,
            'action': action_text,
            'delta_score': delta_score
        }
        if extra:
            item.update(extra)
        self._append_jsonl(self.pos_path, item)

    def load_positive(self) -> List[Dict[str, Any]]:
        return self._load_jsonl(self.pos_path)

    # -------------------- Negative examples --------------------
    def add_negative(self, states: List[str], actions: List[str], reason: str = 'loop_zero_gain', extra: Dict[str, Any] = None):
        item = {
            'reason': reason,
            'length': len(states),
            'states': states[-50:],  # cap length for storage
            'actions': actions[-50:]
        }
        if extra:
            item.update(extra)
        self._append_jsonl(self.neg_path, item)

    def load_negative(self) -> List[Dict[str, Any]]:
        return self._load_jsonl(self.neg_path)

    # -------------------- Similarity retrieval --------------------
    def _tokenize(self, text: str) -> List[str]:
        return [t for t in (text or '').lower().replace('\n', ' ').split() if t.isalpha() or t.isalnum()]

    def _jaccard(self, a_tokens: List[str], b_tokens: List[str]) -> float:
        set_a, set_b = set(a_tokens), set(b_tokens)
        if not set_a or not set_b:
            return 0.0
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        return inter / union if union > 0 else 0.0

    def retrieve_similar(self, query_state: str, k: int = 3) -> List[Dict[str, Any]]:
        examples = self.load_positive()
        q_tokens = self._tokenize(query_state)
        scored = []
        for ex in examples:
            s_tokens = self._tokenize(ex.get('state', ''))
            sim = self._jaccard(q_tokens, s_tokens)
            scored.append((sim, ex))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ex for sim, ex in scored[:k] if sim > 0] 