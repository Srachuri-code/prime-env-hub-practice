import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import verifiers as vf
from verifiers.utils.data_utils import (
    BOXED_SYSTEM_PROMPT,
    extract_boxed_answer,
    load_example_dataset,
)


@dataclass
class BufferItem:
    instance_id: str
    prompt: str
    answer: str
    k: int
    prev_response: str
    prev_reward: float
    score: float
    mode: str
    meta: Dict[str, Any]


class ExItBuffer:
    def __init__(self, capacity: int, min_size: int, kappa: float):
        self.capacity = capacity
        self.min_size = min_size
        self.kappa = kappa
        self._items: Dict[str, BufferItem] = {}

    def __len__(self) -> int:
        return len(self._items)

    def upsert(self, item: BufferItem) -> None:
        # If the item already exists, update it in place
        if item.instance_id in self._items:
            self._items[item.instance_id] = item
            self.prune_to_capacity()
            return

        # If we have room, insert directly
        if len(self._items) < self.capacity:
            self._items[item.instance_id] = item
            return

        # At capacity: only evict the current worst if the new score is strictly higher
        worst_id, worst_score = None, float("inf")
        for key, existing in self._items.items():
            if existing.score < worst_score:
                worst_id, worst_score = key, existing.score
        if item.score > worst_score and worst_id is not None:
            self._items.pop(worst_id, None)
            self._items[item.instance_id] = item
        # Otherwise, drop the new item

    def remove(self, instance_id: str) -> None:
        self._items.pop(instance_id, None)

    def prune_to_capacity(self) -> None:
        if len(self._items) <= self.capacity:
            return
        sorted_items = sorted(self._items.items(), key=lambda kv: kv[1].score)
        to_remove = len(self._items) - self.capacity
        for i in range(to_remove):
            key, _ = sorted_items[i]
            self._items.pop(key, None)

    def sample(self) -> Optional[BufferItem]:
        if not self._items:
            return None
        items = list(self._items.values())
        scores = [x.score for x in items]
        max_s = max(scores)
        logits = [self.kappa * (s - max_s) for s in scores]
        weights = [math.exp(x) for x in logits]
        total = sum(weights)
        if total <= 0.0 or not math.isfinite(total):
            return random.choice(items)
        r = random.random() * total
        acc = 0.0
        for item, w in zip(items, weights):
            acc += w
            if r <= acc:
                return item
        return random.choice(items)

    def update_score(self, instance_id: str, score: float) -> None:
        if instance_id in self._items:
            self._items[instance_id].score = score

    def topk_ids(self, k: Optional[int] = None) -> List[str]:
        ordered = sorted(self._items.values(), key=lambda x: x.score, reverse=True)
        if k is None:
            return [x.instance_id for x in ordered]
        return [x.instance_id for x in ordered[:k]]




class GroupTracker:
    def __init__(self, group_size_hint: int):
        self.group_size_hint = max(1, int(group_size_hint))
        # Per-instance ring buffers of most recent returns for computing contemporaneous variance
        self._per_instance: Dict[str, List[float]] = {}
        # Ready-to-consume variances computed when an instance accumulates group_size_hint returns
        self._ready: Dict[str, float] = {}

    def begin_group(self) -> None:
        # Do not clear accumulated per-instance returns; only reset the ready cache
        self._ready = {}

    def record(self, instance_id: str, r: float) -> None:
        if not instance_id:
            return
        lst = self._per_instance.get(instance_id)
        if lst is None:
            lst = []
            self._per_instance[instance_id] = lst
        lst.append(float(r))
        if len(lst) >= self.group_size_hint:
            if len(lst) >= 2:
                mean_r = sum(lst) / len(lst)
                var_r = sum((x - mean_r) ** 2 for x in lst) / (len(lst) - 1)
            else:
                var_r = 0.0
            self._ready[instance_id] = float(var_r)
            # Clear the buffer for the next contemporaneous group for this instance
            self._per_instance[instance_id] = []

    def end_group(self) -> Dict[str, float]:
        # Return any instance variances that reached the group size since last call
        out = self._ready
        self._ready = {}
        return out


class ExItController:
    def __init__(self, buffer: ExItBuffer, select_prob: float, divergence_prob: float):
        self.buffer = buffer
        self.select_prob = select_prob
        self.divergence_prob = divergence_prob

    def build_base_episode(self, base_id: str, sample: Dict[str, Any]) -> Dict[str, Any]:
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        metadata = {
            "mode": "base",
            "prev_reward": 0.0,
            "prev_response": "",
            "instance_id": f"{base_id}:k0",
            "base_id": base_id,
            "k": 0,
        }
        return {
            "prompt_text": question,
            "answer": answer,
            "metadata": metadata,
        }

    def _format_improve(self, prompt: str, prev_response: str) -> str:
        return (
            "Improve your current response to this request:\n\n"
            "## Request:\n"
            f"{prompt}\n\n"
            "## Current response:\n"
            f"{prev_response}"
        )

    def _format_diverge(self, prompt: str, prev_response: str) -> str:
        return (
            "Consider your current response to this request, and provide another response that takes\n"
            "an approach that SIGNIFICANTLY DIFFERS from the approach of the current response.\n\n"
            "## Request:\n"
            f"{prompt}\n\n"
            "## Current response:\n"
            f"{prev_response}\n\n"
            "## Your new response format:\n"
            "- First, provide a brief summary of the approach in the current response above.\n"
            "- Next, state a meaningfully different approach that can be taken instead.\n"
            "- Finally, provide your new response using this different approach."
        )

    def build_iter_episode(self, item: BufferItem) -> Dict[str, Any]:
        use_diverge = random.random() < self.divergence_prob
        mode = "diverge" if use_diverge else "improve"
        if mode == "diverge":
            user_text = self._format_diverge(item.prompt, item.prev_response)
        else:
            user_text = self._format_improve(item.prompt, item.prev_response)
        metadata = {
            "mode": mode,
            "prev_reward": float(item.prev_reward),
            "prev_response": item.prev_response,
            "instance_id": item.instance_id,
            "base_id": item.meta.get("base_id", item.instance_id),
            "k": item.k,
        }
        return {
            "prompt_text": user_text,
            "answer": item.answer,
            "metadata": metadata,
        }

    def sample_episode(self, base_id: str, base_sample: Dict[str, Any]) -> Dict[str, Any]:
        use_buffer = (len(self.buffer) >= self.buffer.min_size) and (random.random() < self.select_prob)
        if use_buffer:
            chosen = self.buffer.sample()
            if chosen is not None:
                return self.build_iter_episode(chosen)
        return self.build_base_episode(base_id, base_sample)

    def on_episode_end(self, metadata: Dict[str, Any], completion: str, base_reward: float) -> Optional[BufferItem]:
        base_id = str(metadata.get("base_id", ""))
        prompt = str(metadata.get("prompt", metadata.get("original_prompt", "")))
        if not prompt:
            return None
        answer = str(metadata.get("answer", ""))
        prev_k = int(metadata.get("k", 0))
        next_k = prev_k + 1
        new_instance_id = f"{base_id}:k{next_k}"
        prev_instance_id = str(metadata.get("instance_id", f"{base_id}:k{prev_k}"))
        # Initialize expanded task score per ExIt: inherit parent's learnability score
        parent_score = 0.0
        if prev_instance_id:
            existing_parent = getattr(self.buffer, "_items", {}).get(prev_instance_id)
            if isinstance(existing_parent, BufferItem):
                parent_score = float(existing_parent.score)
        # Keep solved expansions but with zero learnability to naturally deprioritize
        initial_score = 0.0 if float(base_reward) >= 1.0 else parent_score
        new_item = BufferItem(
            instance_id=new_instance_id,
            prompt=prompt,
            answer=answer,
            k=next_k,
            prev_response=str(completion),
            prev_reward=float(base_reward),
            score=float(initial_score),
            mode=str(metadata.get("mode", "improve")),
            meta={"base_id": base_id},
        )
        self.buffer.upsert(new_item)
        return new_item


class ExItTrainDataset:
    def __init__(
        self,
        base_dataset: Any,
        controller: ExItController,
    ):
        self.base_dataset = base_dataset
        self.controller = controller

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.base_dataset[idx]
        base_id = f"gsm8k_train_{idx}"
        original_prompt = sample.get("question", "")
        episode = self.controller.sample_episode(base_id, sample)
        prompt_text = episode["prompt_text"]
        answer = episode["answer"]
        metadata = episode["metadata"]
        metadata.update({
            "original_prompt": original_prompt,
            "prompt": original_prompt,
            "answer": answer,
            "is_eval": False,
        })
        return {
            "question": prompt_text,
            "answer": answer,
            **metadata,
        }


class ExItEvalDataset:
    def __init__(self, base_dataset: Any):
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.base_dataset[idx]
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        return {
            "question": question,
            "answer": answer,
            "mode": "base",
            "prev_reward": 0.0,
            "prev_response": "",
            "instance_id": f"gsm8k_eval_{idx}:k0",
            "base_id": f"gsm8k_eval_{idx}",
            "k": 0,
            "is_eval": True,
        }


class ExItSingleTurnEnv:
    def __init__(
        self,
        inner_env: vf.Environment,
        controller: ExItController,
        buffer: ExItBuffer,
        group_tracker: GroupTracker,
        group_size_hint: int,
    ):
        self.inner = inner_env
        self.controller = controller
        self.buffer = buffer
        self.group_tracker = group_tracker
        self.group_size_hint = max(1, int(group_size_hint))

    def begin_group(self) -> None:
        self.group_tracker.begin_group()

    def end_group(self) -> Dict[str, float]:
        var_by_instance = self.group_tracker.end_group()
        for iid, var_r in var_by_instance.items():
            self.buffer.update_score(iid, float(var_r))
        self.buffer.prune_to_capacity()
        return var_by_instance

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def load_environment(
    use_think: bool = True,
    system_prompt: str = BOXED_SYSTEM_PROMPT,
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    exit_enabled: bool = True,
    select_prob: float = 0.5,
    divergence_prob: float = 0.2,
    buffer_size: int = 512,
    min_buffer_size: int = 128,
    kappa: float = 1.0,
    group_size_hint: int = 8,
) -> vf.Environment:
    dataset = load_example_dataset("gsm8k", split="train")
    if num_train_examples != -1:
        dataset = dataset.select(range(num_train_examples))
    eval_dataset = load_example_dataset("gsm8k", split="test")
    if num_eval_examples != -1:
        eval_dataset = eval_dataset.select(range(num_eval_examples))

    parser = vf.ThinkParser(extract_fn=extract_boxed_answer) if use_think else vf.Parser(extract_fn=extract_boxed_answer)

    if not exit_enabled:
        def correct_answer_reward_func(parser_obj, completion, answer, **kwargs):
            response = parser_obj.parse_answer(completion) or ""
            return 1.0 if response == answer else 0.0

        rubric = vf.Rubric(
            funcs=[correct_answer_reward_func, parser.get_format_reward_func()],
            weights=[1.0, 0.1],
        )

        inner_env = vf.SingleTurnEnv(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=system_prompt,
            parser=parser,
            rubric=rubric,
        )
        return inner_env

    buffer = ExItBuffer(capacity=buffer_size, min_size=min_buffer_size, kappa=kappa)
    controller = ExItController(buffer=buffer, select_prob=select_prob, divergence_prob=divergence_prob)
    group_tracker = GroupTracker(group_size_hint=group_size_hint)

    train_dataset = ExItTrainDataset(base_dataset=dataset, controller=controller)
    eval_dataset_wrapped = ExItEvalDataset(base_dataset=eval_dataset)

    def exit_reward_func(parser_obj, completion, answer, **kwargs):
        response = parser_obj.parse_answer(completion) or ""
        r_new = 1.0 if response == answer else 0.0

        is_eval = bool(kwargs.get("is_eval", False))
        instance_id = str(kwargs.get("instance_id", ""))
        mode = str(kwargs.get("mode", "base"))
        prev_reward = float(kwargs.get("prev_reward", 0.0))

        if is_eval:
            return r_new

        if mode in ("improve", "diverge"):
            denom = max(1e-6, 1.0 - prev_reward)
            out_reward = max(0.0, (r_new - prev_reward) / denom)
        else:
            out_reward = r_new

        if instance_id:
            group_tracker.record(instance_id, float(r_new))

        metadata_for_expansion = {
            k: kwargs.get(k) for k in [
                "mode",
                "prev_reward",
                "prev_response",
                "instance_id",
                "base_id",
                "k",
                "prompt",
                "original_prompt",
                "answer",
            ]
        }
        if not metadata_for_expansion.get("prompt"):
            metadata_for_expansion["prompt"] = str(kwargs.get("original_prompt", kwargs.get("question", "")))
        metadata_for_expansion["answer"] = metadata_for_expansion.get("answer", answer)

        controller.on_episode_end(metadata_for_expansion, completion, float(r_new))

        return out_reward

    rubric = vf.Rubric(
        funcs=[exit_reward_func, parser.get_format_reward_func()],
        weights=[1.0, 0.1],
    )

    inner_env = vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset_wrapped,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )

    return ExItSingleTurnEnv(
        inner_env=inner_env,
        controller=controller,
        buffer=buffer,
        group_tracker=group_tracker,
        group_size_hint=group_size_hint,
    )


