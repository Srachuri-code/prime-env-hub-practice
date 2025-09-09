import verifiers as vf
from verifiers.types import Messages, State
from typing import Tuple, List, Dict, Any
import random
from verifiers.utils.data_utils import (
    BOXED_SYSTEM_PROMPT,
    extract_boxed_answer,
    load_example_dataset,
)


class SelfImproveEnv(vf.MultiTurnEnv):
    def __init__(self, dataset, rubric: vf.Rubric, max_turns: int = 3, system_prompt: str | None = None, eval_dataset=None, parser: vf.Parser | None = None, p_div: float | None = None, **kwargs):
        if system_prompt is None:
            system_prompt = BOXED_SYSTEM_PROMPT
        # Divergence probability for ExIt-mode scheduling
        self.p_div = float(p_div if p_div is not None else kwargs.get("divergence_prob", 0.0))
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            rubric=rubric,
            max_turns=max_turns,
            system_prompt=system_prompt,
            parser=parser,
            **kwargs,
        )

    async def env_response(self, messages: Messages, state: State, **_) -> Tuple[Messages, State]:
        # Initialize ExIt-style partial history and mode schedule on first turn
        if "turn" not in state:
            state["turn"] = 0  # t
            state["k"] = 0     # iterate index
            state["mode_history"] = []  # list[str]: "improve" or "diverge"
            state["quality_history"] = []  # list[float]
            state["partial_history"] = {
                "turn_index": 1,              # single-turn domain (t = 1)
                "iterate_index": 0,           # k
                "last_iterate_text": "",     # y_t^k
                "prev_turn_last_iterates": [],
                "feedback_current": None,
                "feedback_prev_turns": [],
            }

        # Choose mode for this expansion step (ExIt diverge vs improve)
        mode = "diverge" if (random.random() < self.p_div) else "improve"
        state["mode_history"].append(mode)

        # Construct prompt augmentation akin to render_prompt(m, τ^{k'}_{t-}, mode)
        if mode == "diverge":
            directive = (
                "DIVERGE: Propose a substantively different approach than the last attempt. "
                "Focus on exploring an alternative angle; still answer the same task."
            )
        else:
            directive = (
                "IMPROVE: Revise your previous answer to improve correctness, clarity, and structure. "
                "Make concrete improvements; do not add unrelated content. Return only the revised solution."
            )

        # Optionally surface last iterate text for visibility (τ^k_t)
        last_assistant = _last_assistant_text(messages)
        if last_assistant:
            state["partial_history"]["last_iterate_text"] = last_assistant

        response: Messages = [
            {"role": "user", "content": directive}
        ]

        # Update iterate/turn counters
        state["k"] = state.get("k", 0) + 1
        state["turn"] = state.get("turn", 0) + 1
        state["partial_history"]["iterate_index"] = state["k"]
        return response, state

    async def is_completed(self, messages: Messages, state: State, **_) -> bool:
        # Stop after K improvement/divergence steps (self-iteration budget)
        return state.get("turn", 0) >= self.max_turns


# --- Minimal rubric for self-improvement ---
def _improved_over_previous(prompt: str | Messages, completion: Messages, answer: str | None, state: Dict[str, Any]) -> float:
    # Improvement reward r_i = max(0, Q(new) - Q(prev)), normalized to [0,1]
    prompt_text = _prompt_text(prompt)
    assistant = [m.get("content", "") for m in completion if m.get("role") == "assistant"]
    if not assistant:
        return 0.0
    new_q = _normalize_quality(_quality_heuristic(prompt_text, assistant[-1]))
    if len(assistant) == 1:
        # First attempt gets its normalized quality as reward baseline
        return new_q
    prev_q = _normalize_quality(_quality_heuristic(prompt_text, assistant[-2]))
    return max(0.0, new_q - prev_q)


def _normalized_quality(prompt: str | Messages, completion: Messages, answer: str | None, state: Dict[str, Any]) -> float:
    # Normalized quality of final answer in [0,1]
    prompt_text = _prompt_text(prompt)
    last = _last_assistant_text(completion)
    if not last:
        return 0.0
    return _normalize_quality(_quality_heuristic(prompt_text, last))


# --- Heuristics to approximate env.normalize_quality ---
def _prompt_text(prompt: str | Messages) -> str:
    if isinstance(prompt, str):
        return prompt
    # Assume first user message in dataset prompt list
    for m in prompt:
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def _last_assistant_text(messages: Messages) -> str:
    for m in reversed(messages):
        if m.get("role") == "assistant":
            return m.get("content", "")
    return ""


def _truncate_last_assistant(messages: Messages) -> Messages:
    # Return a shallow-copied list without the last assistant message
    tr: Messages = []
    removed = False
    for m in reversed(messages):
        if (not removed) and m.get("role") == "assistant":
            removed = True
            continue
        tr.append(m)
    tr.reverse()
    return tr


def _make_correct_answer_reward(parser: vf.Parser):
    async def _fn(prompt, completion, answer, state, **kwargs) -> float:
        response = parser.parse_answer(completion) or ""
        return 1.0 if response == (answer or "") else 0.0
    return _fn


def _make_exit_reward(parser: vf.Parser):
    async def _fn(prompt, completion, answer, state, **kwargs) -> float:
        # Base step (first assistant msg): use normalized quality (here: correctness as [0,1])
        # Self-iteration steps: improvement reward = max(0, Q_new - Q_prev)
        if not completion:
            return 0.0
        curr = parser.parse_answer(completion) or ""
        curr_q = 1.0 if curr == (answer or "") else 0.0
        # Determine if there was a previous assistant attempt
        had_prev = any(m.get("role") == "assistant" for m in completion[:-1])
        if not had_prev:
            return curr_q
        prev = parser.parse_answer(_truncate_last_assistant(completion)) or ""
        prev_q = 1.0 if prev == (answer or "") else 0.0
        return max(0.0, curr_q - prev_q)
    return _fn


def _quality_heuristic(prompt_text: str, answer_text: str) -> float:
    # Simple proxy for quality: coverage of prompt terms, structure, and concision
    prompt_terms = set(t.lower() for t in _tokenize(prompt_text) if t.isalpha())
    ans_terms = set(t.lower() for t in _tokenize(answer_text) if t.isalpha())
    coverage = 0.0
    if prompt_terms:
        coverage = len(prompt_terms & ans_terms) / max(1, len(prompt_terms))

    structure = 0.0
    if any(s in answer_text for s in ["1.", "2.", "- ", "* "]):
        structure += 0.5
    if any(k in answer_text.lower() for k in ["final", "summary", "conclusion"]):
        structure += 0.5

    length = len(answer_text.strip())
    concision = 1.0 if 200 <= length <= 1200 else 0.5 if 80 <= length < 200 or 1200 < length <= 2000 else 0.0

    # Weighted sum (unnormalized), later normalized
    return 0.6 * coverage + 0.25 * structure + 0.15 * concision


def _normalize_quality(x: float) -> float:
    # Already in [0, ~1.0]; clamp
    return max(0.0, min(1.0, x))


def _build_default_dataset(max_examples: int = -1) -> List[Dict[str, Any]]:
    prompts = [
        "Summarize the core training loop of ExIt+GRPO in 5-7 sentences.",
        "Explain how a diversity bonus can be applied to group advantages in GRPO.",
        "Describe a simple improvement reward suitable for self-iteration steps.",
    ]
    data = [{"prompt": p} for p in prompts]
    return data if max_examples == -1 else data[: max(0, max_examples)]


def load_environment(**kwargs) -> vf.Environment:
    '''
    ExIt+GRPO-style self-improvement environment on GSM8K using Verifiers.

    Env args (via --env-args JSON):
    - max_turns: int, number of self-iteration steps (default 3)
    - num_train_examples: int, truncate train set (default -1 = all)
    - num_eval_examples: int, truncate eval set (default -1 = all)
    - use_think: bool, use ThinkParser (default True)
    - divergence_prob: float, probability of 'diverge' mode per step (default 0.0)
    - system_prompt: str, defaults to BOXED_SYSTEM_PROMPT
    '''
    max_turns: int = int(kwargs.get("max_turns", 3))
    num_train_examples: int = int(kwargs.get("num_train_examples", -1))
    num_eval_examples: int = int(kwargs.get("num_eval_examples", -1))
    use_think: bool = bool(kwargs.get("use_think", True))
    system_prompt: str = kwargs.get("system_prompt", BOXED_SYSTEM_PROMPT)
    divergence_prob: float = float(kwargs.get("divergence_prob", 0.0))

    # Load GSM8K datasets
    dataset = load_example_dataset("gsm8k", split="train")
    if num_train_examples != -1:
        dataset = dataset.select(range(num_train_examples))
    eval_dataset = load_example_dataset("gsm8k", split="test")
    if num_eval_examples != -1:
        eval_dataset = eval_dataset.select(range(num_eval_examples))

    # Parser
    parser: vf.Parser
    if use_think:
        parser = vf.ThinkParser(extract_fn=extract_boxed_answer)
    else:
        parser = vf.Parser(extract_fn=extract_boxed_answer)

    # Rubric
    rubric = vf.Rubric(
        funcs=[
            _make_exit_reward(parser),                # main reward (quality on first step, improvement thereafter)
            _make_correct_answer_reward(parser),      # logged; not used for reward
            parser.get_format_reward_func(),          # logged; not used for reward
            _normalized_quality,                      # logged heuristic quality
            _improved_over_previous,                  # logged heuristic improvement
        ],
        weights=[1.0, 0.0, 0.0, 0.0, 0.0],
    )

    return SelfImproveEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        max_turns=max_turns,
        p_div=divergence_prob,
    )
