"""
ExIt Self-Improvement Environment with GRPO Integration

This module implements an Exploratory Iteration (ExIt) self-improvement environment
with divergence and multiplicative diversity bonus that integrates cleanly with
an existing GRPO trainer.

Key components:
- PartialHistory: Dataclass-like structure for partial task histories
- TaskBuffer: Priority buffer for task selection based on learnability S = var(r̂)
- SelfImproveEnv: Single-step ExIt tasks with selection/expansion protocol
- Diversity bonus: Multiplicative coefficients based on embedding distances

Custom trainer hooks (not part of base Verifiers API):
- begin_episode(): Returns selected PartialHistory if buffer selection was made
- complete_group(): Updates buffer with group results and performs expansion
- get_diversity_coefficients(): Computes diversity multipliers for GRPO advantages
"""

import verifiers as vf
from verifiers.types import Messages, State
from typing import Tuple, List, Dict, Any, Optional, Callable
import random
import hashlib
import time
import numpy as np
from verifiers.utils.data_utils import (
    BOXED_SYSTEM_PROMPT,
    extract_boxed_answer,
    load_example_dataset,
)


# ====================
# 1) Data structures
# ====================

class PartialHistory:
    """Partial history representing a starting point for self-improvement."""
    __slots__ = ['id', 'task_id', 'prompt_chat', 'prev_solution_text', 'turn_index',
                 'iterate_index', 'start_depth', 'recency_id', 'returns', 'quality',
                 'priority', 'meta']
    
    def __init__(self, id: str, task_id: str, prompt_chat: Messages, 
                 prev_solution_text: str, turn_index: int, iterate_index: int,
                 start_depth: int, recency_id: int, returns: List[float] = None,
                 quality: float = 0.0, priority: float = 0.0, meta: Dict[str, Any] = None):
        self.id = id
        self.task_id = task_id
        self.prompt_chat = prompt_chat
        self.prev_solution_text = prev_solution_text
        self.turn_index = turn_index
        self.iterate_index = iterate_index
        self.start_depth = start_depth
        self.recency_id = recency_id
        self.returns = returns or []
        self.quality = quality
        self.priority = priority
        self.meta = meta or {}


class TaskBuffer:
    """Priority buffer for task selection based on learnability."""
    
    def __init__(self, capacity: int, min_select: int):
        self.capacity = capacity
        self.min_select = min_select
        self._buffer: Dict[str, PartialHistory] = {}
        
    def add(self, ph: PartialHistory) -> None:
        """Upsert partial history and recompute priority."""
        if ph.id in self._buffer:
            # Update existing entry
            existing = self._buffer[ph.id]
            existing.returns.extend(ph.returns)
            existing.quality = ph.quality
            existing.priority = self.compute_priority(existing.returns)
        else:
            # Add new entry
            ph.priority = self.compute_priority(ph.returns)
            self._buffer[ph.id] = ph
        
    def sample(self, k: int = 1, strategy: str = "priority") -> List[PartialHistory]:
        """Sample k partial histories using rank-weighted sampling."""
        if not self._buffer or k <= 0:
            return []
        
        items = list(self._buffer.values())
        if strategy == "priority":
            # Sort by priority descending
            items.sort(key=lambda x: x.priority, reverse=True)
            # Compute rank weights (1/rank)
            weights = [1.0 / (i + 1) for i in range(len(items))]
            # Normalize weights
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
            else:
                weights = [1.0 / len(items)] * len(items)
            
            # Sample without replacement
            sampled_indices = np.random.choice(
                len(items), 
                size=min(k, len(items)), 
                replace=False, 
                p=weights
            )
            return [items[i] for i in sampled_indices]
        else:
            # Random sampling
            return random.sample(items, min(k, len(items)))
    
    def prune(self) -> None:
        """Keep top-N by priority."""
        if len(self._buffer) <= self.capacity:
            return
        
        items = list(self._buffer.values())
        items.sort(key=lambda x: x.priority, reverse=True)
        # Keep top capacity items
        self._buffer = {ph.id: ph for ph in items[:self.capacity]}
    
    def size(self) -> int:
        """Return current buffer size."""
        return len(self._buffer)
    
    def stats(self) -> Dict[str, Any]:
        """Return buffer statistics."""
        if not self._buffer:
            return {"size": 0, "mean_priority": 0.0, "mean_depth": 0.0}
        
        priorities = [ph.priority for ph in self._buffer.values()]
        depths = [ph.start_depth for ph in self._buffer.values()]
        recencies = [ph.recency_id for ph in self._buffer.values()]
        
        return {
            "size": len(self._buffer),
            "mean_priority": np.mean(priorities) if priorities else 0.0,
            "std_priority": np.std(priorities) if priorities else 0.0,
            "mean_depth": np.mean(depths) if depths else 0.0,
            "mean_recency": np.mean(recencies) if recencies else 0.0,
        }
    
    @staticmethod
    def normalize_returns(rs: List[float]) -> List[float]:
        """Min-max normalization to [0,1], guard zero-range."""
        if not rs:
            return []
        min_r = min(rs)
        max_r = max(rs)
        if max_r - min_r < 1e-8:
            # All returns are the same
            return [0.5] * len(rs)
        return [(r - min_r) / (max_r - min_r) for r in rs]
    
    @staticmethod
    def compute_priority(returns: List[float]) -> float:
        """Variance of normalized returns as learnability score."""
        if not returns or len(returns) < 2:
            return 0.0
        normalized = TaskBuffer.normalize_returns(returns)
        return float(np.var(normalized))


# ====================
# 2) Prompts
# ====================

def render_improve_prompt(last_iterate_text: str) -> Dict[str, str]:
    """Generate improvement prompt with previous solution context."""
    content = (
        "IMPROVE: Revise your previous answer to improve correctness, clarity, and structure. "
        "Return only the revised solution.\n\n"
        f"Previous solution:\n{last_iterate_text}"
    )
    return {"role": "user", "content": content}


def render_diverge_prompt(last_iterate_text: str) -> Dict[str, str]:
    """Generate divergence prompt with previous solution context."""
    content = (
        "DIVERGE: Propose a substantively different approach than the last attempt. "
        "Keep it correct; avoid repeating the prior method. Return only the new solution.\n\n"
        f"Previous solution:\n{last_iterate_text}"
    )
    return {"role": "user", "content": content}


# ====================
# 3) Helpers
# ====================

def _tokenize(text: str) -> List[str]:
    """Simple tokenization helper used by embeddings."""
    return text.replace(",", " ").replace(".", " ").replace("!", " ").replace("?", " ").split()


async def correct_answer_reward(parser: vf.Parser, completion: Messages, answer: str, state: Dict[str, Any]) -> float:
    """Returns 1.0 if parsed answer matches ground truth, else 0.0."""
    response = parser.parse_answer(completion) or ""
    return 1.0 if response == (answer or "") else 0.0


# ====================
# 4) Diversity bonus
# ====================

def _simple_hash_embedding(text: str, dim: int = 64) -> np.ndarray:
    """Simple bag-of-words hashed embedding as fallback."""
    tokens = _tokenize(text.lower())
    embedding = np.zeros(dim)
    
    for token in tokens:
        # Hash token to index
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        idx = h % dim
        # Simple count
        embedding[idx] += 1.0
    
    # L2 normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding


def compute_diversity_coefficients(texts: List[str], embedding_fn: Optional[Callable] = None) -> List[float]:
    """
    Compute diversity coefficients based on embedding distances.
    Returns coefficients in [0,1] aligned with input order.
    """
    if not texts:
        return []
    
    if len(texts) == 1:
        return [0.5]  # Single text gets neutral coefficient
    
    # Get embeddings
    if embedding_fn is not None:
        embeddings = [embedding_fn(text) for text in texts]
    else:
        # Use simple fallback
        embeddings = [_simple_hash_embedding(text) for text in texts]
    
    # Compute centroid
    embeddings = [np.array(e) for e in embeddings]
    centroid = np.mean(embeddings, axis=0)
    
    # Compute distances
    distances = [np.linalg.norm(e - centroid) for e in embeddings]
    
    # Range-normalize to [0,1]
    min_d = min(distances)
    max_d = max(distances)
    
    if max_d - min_d < 1e-8:
        # All distances are the same
        return [0.5] * len(texts)
    
    coefficients = [(d - min_d) / (max_d - min_d + 1e-8) for d in distances]
    return coefficients


# ====================
# 5) Environment
# ====================

class SelfImproveEnv(vf.Environment):
    """Single-step ExIt self-improvement environment."""
    
    def __init__(
        self,
        dataset,
        eval_dataset,
        rubric: vf.Rubric,
        parser: vf.Parser,
        system_prompt: str = None,
        buffer_capacity: int = 1024,
        min_buffer_to_select: int = 64,
        p_select: float = 0.5,
        p_div: float = 0.1,
        diversity_enabled: bool = True,
        embedding_backend: str = "math-bert",
        embedding_fn: Optional[Callable] = None,
        use_think: bool = True,
        **kwargs
    ):
        if system_prompt is None:
            system_prompt = BOXED_SYSTEM_PROMPT
            
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            rubric=rubric,
            system_prompt=system_prompt,
            parser=parser,
            **kwargs
        )
        
        # ExIt configuration
        self.cfg = {
            "buffer_capacity": buffer_capacity,
            "min_buffer_to_select": min_buffer_to_select,
            "p_select": p_select,
            "p_div": p_div,
            "diversity_enabled": diversity_enabled,
            "embedding_backend": embedding_backend,
            "use_think": use_think,
        }
        
        self.embedding_fn = embedding_fn
        
        # Initialize task buffer and counters
        self.buffer = TaskBuffer(capacity=buffer_capacity, min_select=min_buffer_to_select)
        self.recency_counter = 0
        
        # Track current episode's selected partial history
        self._current_selected_ph: Optional[PartialHistory] = None
        self._current_episode_metadata: Dict[str, Any] = {}
    
    async def env_response(self, messages: Messages, state: State, **_) -> Tuple[Messages, State]:
        """Generate environment response for single-step ExIt task."""
        
        # Initialize state on first turn
        if "turn" not in state:
            state["turn"] = 0
            state["k"] = 0
            state["mode_history"] = []
            state["quality_history"] = []
            state["partial_history"] = {
                "last_iterate_text": "",
                "turn_index": 0,
                "iterate_index": 0,
                "selected_ph_id": None,
            }
            state["selection_metadata"] = {
                "from_buffer": False,
                "start_depth": 0,
                "recency_id": self.recency_counter,
            }
            
            # Perform selection (Algorithm 1)
            if self.buffer.size() >= self.cfg["min_buffer_to_select"] and random.random() < self.cfg["p_select"]:
                # Select from buffer
                selected = self.buffer.sample(k=1, strategy="priority")
                if selected:
                    ph = selected[0]
                    self._current_selected_ph = ph
                    state["partial_history"]["selected_ph_id"] = ph.id
                    state["partial_history"]["last_iterate_text"] = ph.prev_solution_text
                    state["partial_history"]["turn_index"] = ph.turn_index
                    state["partial_history"]["iterate_index"] = ph.iterate_index
                    state["selection_metadata"]["from_buffer"] = True
                    state["selection_metadata"]["start_depth"] = ph.start_depth
                    state["selection_metadata"]["recency_id"] = self.recency_counter
                    self.recency_counter += 1
                    
                    # Store metadata for trainer
                    self._current_episode_metadata = {
                        "from_buffer": True,
                        "start_depth": ph.start_depth,
                        "recency_id": state["selection_metadata"]["recency_id"],
                        "selected_ph": ph,
                    }
            else:
                # Start from base dataset
                state["selection_metadata"]["from_buffer"] = False
                state["selection_metadata"]["start_depth"] = 0
                state["selection_metadata"]["recency_id"] = self.recency_counter
                self.recency_counter += 1
                self._current_selected_ph = None
                self._current_episode_metadata = {
                    "from_buffer": False,
                    "start_depth": 0,
                    "recency_id": state["selection_metadata"]["recency_id"],
                    "selected_ph": None,
                }
        
        # Extract last assistant text if any
        last_assistant = ""
        for m in reversed(messages):
            if m.get("role") == "assistant":
                last_assistant = m.get("content", "")
                break
        
        if last_assistant:
            state["partial_history"]["last_iterate_text"] = last_assistant
        
        # Choose mode for this step
        mode = "diverge" if random.random() < self.cfg["p_div"] else "improve"
        state["mode_history"].append(mode)
        self._current_episode_metadata["mode"] = mode
        
        # Build prompt
        if mode == "diverge":
            response_msg = render_diverge_prompt(state["partial_history"]["last_iterate_text"])
        else:
            response_msg = render_improve_prompt(state["partial_history"]["last_iterate_text"])
        
        # Update counters
        state["turn"] = state.get("turn", 0) + 1
        state["k"] = state.get("k", 0) + 1
        state["partial_history"]["iterate_index"] = state["k"]
        
        # Log info
        state["info"] = state.get("info", {})
        state["info"]["exit"] = {
            "mode": mode,
            "from_buffer": state["selection_metadata"]["from_buffer"],
            "start_depth": state["selection_metadata"]["start_depth"],
            "recency_id": state["selection_metadata"]["recency_id"],
            "buffer_size": self.buffer.size(),
        }
        
        return [response_msg], state
    
    async def is_completed(self, messages: Messages, state: State, **_) -> bool:
        """Single-step task completes after one assistant response."""
        # Count assistant messages after the initial prompt
        assistant_count = sum(1 for m in messages if m.get("role") == "assistant")
        # Complete after one assistant response to the improvement/divergence prompt
        return assistant_count >= 1
    
    # ====================
    # 6) Buffer lifecycle APIs
    # ====================
    
    def begin_episode(self, context: Dict[str, Any] = None) -> Optional[PartialHistory]:
        """
        Called by trainer at episode start to get selected PartialHistory.
        Returns the selected PartialHistory if buffer selection was made, else None.
        """
        if self._current_episode_metadata.get("from_buffer") and self._current_selected_ph:
            return self._current_selected_ph
        return None
    
    def complete_group(
        self, 
        group_rollouts: List[Messages], 
        group_rewards: List[float], 
        group_texts: List[str]
    ) -> List[str]:
        """
        Called by trainer after group completes to update buffer.
        
        Args:
            group_rollouts: Raw chat traces for each rollout
            group_rewards: Final scalar returns from rubric
            group_texts: Final assistant texts
            
        Returns:
            List of new PartialHistory ids for logging
        """
        new_ids = []
        
        # Normalize returns
        normalized_returns = TaskBuffer.normalize_returns(group_rewards)
        
        # Compute learnability S = var(r̂)
        priority = TaskBuffer.compute_priority(group_rewards)
        
        # Update selected PartialHistory if exists
        if self._current_selected_ph:
            self._current_selected_ph.returns.extend(group_rewards)
            self._current_selected_ph.priority = TaskBuffer.compute_priority(
                self._current_selected_ph.returns
            )
            self.buffer.add(self._current_selected_ph)
        
        # Expansion: create new PartialHistories for each rollout
        for i, (rollout, reward, text) in enumerate(zip(group_rollouts, group_rewards, group_texts)):
            # Generate unique ID
            timestamp = int(time.time() * 1000000)
            task_id = self._current_episode_metadata.get("task_id", "unknown")
            ph_id = f"{task_id}_t{timestamp}_g{i}"
            
            # Determine new start depth
            prev_depth = self._current_episode_metadata.get("start_depth", 0)
            new_depth = prev_depth + 1
            
            # Create new PartialHistory
            new_ph = PartialHistory(
                id=ph_id,
                task_id=task_id,
                prompt_chat=rollout,
                prev_solution_text=text,
                turn_index=1,  # Single-turn domain
                iterate_index=1,
                start_depth=new_depth,
                recency_id=self.recency_counter,
                returns=[reward],
                quality=normalized_returns[i] if i < len(normalized_returns) else 0.5,
                priority=0.0,  # Will be recomputed on add
                meta={"mode": self._current_episode_metadata.get("mode", "improve")}
            )
            self.recency_counter += 1
            
            # Add to buffer
            self.buffer.add(new_ph)
            new_ids.append(ph_id)
        
        # Prune buffer to capacity
        self.buffer.prune()
        
        return new_ids
    
    # ====================
    # 7) Diversity helper
    # ====================
    
    def get_diversity_coefficients(self, group_texts: List[str]) -> List[float]:
        """
        Compute diversity coefficients for GRPO advantage scaling.
        Returns coefficients in [0,1] aligned with group_texts order.
        """
        if not self.cfg["diversity_enabled"]:
            return [1.0] * len(group_texts)
        
        return compute_diversity_coefficients(group_texts, self.embedding_fn)


# ====================
# 8) Load environment
# ====================

def load_environment(**kwargs) -> vf.Environment:
    """
    Load ExIt self-improvement environment with GSM8K dataset.
    
    Environment arguments:
    - num_train_examples: int (default -1 for all)
    - num_eval_examples: int (default -1 for all)
    - use_think: bool (default True)
    - buffer_capacity: int (default 1024)
    - min_buffer_to_select: int (default 64)
    - p_select: float (default 0.5)
    - p_div: float (default 0.1)
    - diversity_enabled: bool (default True)
    - embedding_backend: str (default "math-bert")
    - embedding_fn: Optional[Callable] (default None)
    - system_prompt: str (default BOXED_SYSTEM_PROMPT)
    - judge_prompt: str (override default strict numeric judge prompt)
    """
    # Parse arguments
    num_train_examples = int(kwargs.get("num_train_examples", -1))
    num_eval_examples = int(kwargs.get("num_eval_examples", -1))
    use_think = bool(kwargs.get("use_think", True))
    
    buffer_capacity = int(kwargs.get("buffer_capacity", 1024))
    min_buffer_to_select = int(kwargs.get("min_buffer_to_select", 64))
    p_select = float(kwargs.get("p_select", 0.5))
    p_div = float(kwargs.get("p_div", 0.1))
    
    diversity_enabled = bool(kwargs.get("diversity_enabled", True))
    embedding_backend = kwargs.get("embedding_backend", "math-bert")
    embedding_fn = kwargs.get("embedding_fn", None)
    
    system_prompt = kwargs.get("system_prompt", BOXED_SYSTEM_PROMPT)
    
    # Load GSM8K datasets
    dataset = load_example_dataset("gsm8k", split="train")
    if num_train_examples != -1:
        dataset = dataset.select(range(min(num_train_examples, len(dataset))))
    
    eval_dataset = load_example_dataset("gsm8k", split="test")
    if num_eval_examples != -1:
        eval_dataset = eval_dataset.select(range(min(num_eval_examples, len(eval_dataset))))
    
    # Create parser
    if use_think:
        parser = vf.ThinkParser(extract_fn=extract_boxed_answer)
    else:
        parser = vf.Parser(extract_fn=extract_boxed_answer)
    
    # Create rubric with judge as primary reward and heuristics as metrics
    async def _correct_answer_reward(prompt, completion, answer, state, **kwargs):
        return await correct_answer_reward(parser, completion, answer, state)
    
    async def _format_reward(prompt, completion, answer, state, **kwargs):
        format_fn = parser.get_format_reward_func()
        return await format_fn(prompt, completion, answer, state, **kwargs)
    
    # Judge prompt focuses on improvement quality; requires numeric 0.0-1.0 output
    judge_prompt = kwargs.get(
        "judge_prompt",
        (
            "You are a strict evaluator. Given the full conversation, rate the LATEST assistant "
            "message for correctness, clarity, and substantive improvement over any previous solution "
            "shown in the context. Penalize superficial edits, verbosity without substance, and "
            "keyword-stuffing. Return ONLY a single number between 0.0 and 1.0."
        ),
    )
    judge_rubric = vf.JudgeRubric(judge_prompt=judge_prompt)
    
    # Metrics-only rubric (all weights 0.0)
    metrics_rubric = vf.Rubric(
        funcs=[
            exit_improvement_reward,  # Logged only
            _correct_answer_reward,   # Logged only
            _format_reward,           # Logged only
            normalized_quality,       # Logged only
        ],
        weights=[0.0, 0.0, 0.0, 0.0],
    )
    
    # Final rubric: judge drives reward; metrics are logged
    rubric = vf.RubricGroup([judge_rubric, metrics_rubric])
    
    # Create environment
    return SelfImproveEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        rubric=rubric,
        parser=parser,
        system_prompt=system_prompt,
        buffer_capacity=buffer_capacity,
        min_buffer_to_select=min_buffer_to_select,
        p_select=p_select,
        p_div=p_div,
        diversity_enabled=diversity_enabled,
        embedding_backend=embedding_backend,
        embedding_fn=embedding_fn,
        use_think=use_think,
    )


# ====================
# 9) Self-checks
# ====================

if __name__ == "__main__":
    # Unit tests
    print("Running self-checks...")
    
    # Test normalize_returns
    assert TaskBuffer.normalize_returns([1, 1, 1]) == [0.5, 0.5, 0.5]
    assert TaskBuffer.normalize_returns([0, 1, 2]) == [0.0, 0.5, 1.0]
    print("✓ normalize_returns")
    
    # Test compute_priority
    assert TaskBuffer.compute_priority([]) == 0.0
    assert TaskBuffer.compute_priority([0.5]) == 0.0
    assert TaskBuffer.compute_priority([0, 1]) > 0
    print("✓ compute_priority")
    
    # Test diversity coefficients
    texts = ["hello world", "hello world", "hello world"]
    coeffs = compute_diversity_coefficients(texts)
    assert all(c == 0.5 for c in coeffs)  # All same -> neutral
    print("✓ diversity coefficients (same texts)")
    
    texts = ["apple", "banana", "cherry"]
    coeffs = compute_diversity_coefficients(texts)
    assert len(coeffs) == 3
    assert all(0 <= c <= 1 for c in coeffs)
    print("✓ diversity coefficients (different texts)")
    
    # Test environment loading
    env = load_environment(num_train_examples=8, num_eval_examples=8)
    assert isinstance(env, SelfImproveEnv)
    assert env.cfg["p_select"] == 0.5
    assert env.cfg["p_div"] == 0.1
    print("✓ environment loading")
    
    # Test diversity disabled
    env.cfg["diversity_enabled"] = False
    coeffs = env.get_diversity_coefficients(["a", "b", "c"])
    assert coeffs == [1.0, 1.0, 1.0]
    print("✓ diversity disabled returns all ones")
    
    print("\nAll self-checks passed! ✨")