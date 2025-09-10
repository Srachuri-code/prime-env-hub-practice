# self-improve

### Overview
- **Environment ID**: `self-improve`
- **Short description**: ExIt-style self-iteration over GSM8K (SingleTurn). Improve/diverge modes with per-step improvement reward and parser-verified correctness. Buffer prioritization uses learnability measured as per-instance group return variance.
- **Tags**: self-improvement, single-turn, training, eval, gsm8k

References: [Overview](https://verifiers.readthedocs.io/en/latest/overview.html), [Components](https://verifiers.readthedocs.io/en/latest/components.html)

### Datasets
- **Primary**: GSM8K train/eval loaded via Verifiers utils.
- Use `--env-args` to subset train/eval counts.

### Task
- **Type**: single-turn with K-step self-iteration (ExIt decomposition into single-step tasks)
- **Parser**: `ThinkParser` (default) or `Parser`, extracting boxed final answers.
- **Rubric overview**:
  - Main ExIt reward: correctness on base step; improvement reward on self-iteration/divergence steps: `max(0, (Q_new - Q_prev) / max(1e-6, 1 - Q_prev))`, with `Q` = exact correctness.
  - Optional format reward: `parser.get_format_reward_func()` with small weight (default 0.1) to stabilize formatting.

### Quickstart
Run an evaluation with defaults (built-in ExIt prompts, self-iteration enabled):

```bash
uv run vf-eval self-improve -n 20 -r 1 -m gpt-4o-mini
```

Train/eval sizing and ExIt knobs:

```bash
uv run vf-eval self-improve \
  -m gpt-4o-mini -n 50 -r 1 \
  -a '{"num_train_examples": 50, "num_eval_examples": 50, "use_think": true, "divergence_prob": 0.3, "buffer_size": 512, "min_buffer_size": 128, "select_prob": 0.5, "group_size_hint": 8}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_train_examples` | int | `-1` | Limit GSM8K train size |
| `num_eval_examples` | int | `-1` | Limit GSM8K eval size |
| `use_think` | bool | `true` | Use `ThinkParser` to extract boxed answers |
| `system_prompt` | str | boxed | System prompt for boxed reasoning/answers |
| `exit_enabled` | bool | `true` | Enable ExIt self-iteration protocol |
| `select_prob` | float | `0.5` | Probability of sampling from buffer vs. base |
| `divergence_prob` | float | `0.2` | Probability of using diverge prompt |
| `buffer_size` | int | `512` | ExIt buffer capacity |
| `min_buffer_size` | int | `128` | Minimum size before selection allowed |
| `kappa` | float | `1.0` | Softmax inverse temperature for sampling by score |
| `group_size_hint` | int | `8` | Per-instance contemporaneous group size for variance |
| `embedding_dim` | int | `128` | Embedding dim for (optional) centroid tracking |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main ExIt reward (correctness on base step; improvement thereafter) |
| `correct_answer` | Exact-match correctness |
| `format_reward` | Parser format adherence (weight ≤ 0.2) |

### Mapping to ExIt (Algorithm 1)
- Selection over buffer with probability `select_prob` once `min_buffer_size` is reached; otherwise sample base tasks.
- Expansion creates a new self-iteration instance m+ with `score(m+) = score(m′)` (parent score); solved expansions are kept with score 0.
- Learnability: buffer score is per-instance group return variance, computed only when a contemporaneous group of size `group_size_hint` for that instance is observed.
- Prompts: Appendix B verbatim for improve/diverge.
- No additive diversity reward. If desired, diversity bonuses should multiplicatively scale advantages in the trainer (not implemented here).



Problems to Fix: 
I'll scan the implementation files to assess how closely they follow the ExIt method and the verifiers environment abstraction, focusing on `self_improve.py`, its README, and any related code (e.g., `pseudo.py`, `pyproject.toml`). Then I'll give detailed, blunt feedback on alignment with the paper.

[6 tools called]

I read your `self_improve` environment and the README. Here’s a blunt, high-signal review against the ExIt paper and the verifiers abstraction.

### Top-line verdict
- **Conceptually solid but operationally underpowered.** You mirror ExIt’s prompts, per-step improvement reward, a priority buffer, and a selection/expansion loop. But you don’t actually realize ExIt’s core training dynamics (GRPO groups, contemporaneous variance, multi-turn partial-history insertion, or diversity bonus). As-is, the “ExIt” curriculum will mostly act like uniform replay, especially with your defaults, and won’t reproduce the paper’s effects.

### What you got right (good alignment)
- **Self-iteration prompts (Improve/Diverge)**: Matches Appendix B.
```163:185:/Users/skrachur/Desktop/prime-env-hub-practice/environments/self_improve/self_improve.py
    def _format_improve(self, prompt: str, prev_response: str) -> str:
        return (
            "Improve your current response to this request:\n\n"
            "## Request:\n"
            f"{prompt}\n\n"
            "## Current response:\n"
            f"{prev_response}"
        )
```
- **Per-step improvement reward**: Uses r_imp = max(0, (r_new − r_prev)/(1−r_prev)); correct per paper.
```382:399:/Users/skrachur/Desktop/prime-env-hub-practice/environments/self_improve/self_improve.py
        if mode in ("improve", "diverge"):
            denom = max(1e-6, 1.0 - prev_reward)
            out_reward = max(0.0, (r_new - prev_reward) / denom)
        else:
            out_reward = r_new
```
- **Priority buffer and softmax sampling**: Exists and roughly follows exp(κ S_i).
```71:88:/Users/skrachur/Desktop/prime-env-hub-practice/environments/self_improve/self_improve.py
    def sample(self) -> Optional[BufferItem]:
        ...
        logits = [self.kappa * (s - max_s) for s in scores]
        weights = [math.exp(x) for x in logits]
```
- **Solved instance de-prioritization**: New child gets score 0 when solved (nice touch).
```231:241:/Users/skrachur/Desktop/prime-env-hub-practice/environments/self_improve/self_improve.py
        initial_score = 0.0 if float(base_reward) >= 1.0 else parent_score
        new_item = BufferItem(
            instance_id=new_instance_id,
            ...
            score=float(initial_score),
```

### Major misalignments (these materially break ExIt’s benefits)
- **No GRPO grouping or advantage computation.** ExIt’s learnability S = var(r) is per-group variance (G rollouts of the same instance, same iteration). You never create true groups; you just trickle single samples. Your “group” size is a time-accumulated bucket, not a contemporaneous batch under a fixed policy.
  - Consequence: S will be sparse/unstable or never computed; buffer never learns meaningful priorities.
  - Evidence: You rely on external code to call `begin_group/end_group`; verifiers won’t do that for you.
```320:329:/Users/skrachur/Desktop/prime-env-hub-practice/environments/self_improve/self_improve.py
    def begin_group(self) -> None:
        self.group_tracker.begin_group()
    def end_group(self) -> Dict[str, float]:
        var_by_instance = self.group_tracker.end_group()
        for iid, var_r in var_by_instance.items():
            self.buffer.update_score(iid, float(var_r))
```
- **Variance tracked on r_new (correctness), not on the group return used for training.** The paper defines S over normalized returns for the group. For self-iteration, that should be the step’s return (or final quality), not just binary correctness. You record r_new only:
```400:421:/Users/skrachur/Desktop/prime-env-hub-practice/environments/self_improve/self_improve.py
        if instance_id:
            group_tracker.record(instance_id, float(r_new))
```
- **No insertion of all partial histories at expansion time.** ExIt precomputes and inserts all per-turn partial histories on expansion. You only insert the single child for the next k, not the full per-turn set (important even in multi-turn, essential per paper). Your README claims you match Algorithm 1 here, but you don’t.
```355:368:/Users/skrachur/Desktop/prime-env-hub-practice/context/2509.04575v1 (2).md
In practice, when updating the task buffer with a new instance ... we precompute and insert
all instances corresponding to the per-turn partial histories ...
```
```223:245:/Users/skrachur/Desktop/prime-env-hub-practice/environments/self_improve/self_improve.py
        new_instance_id = f"{base_id}:k{next_k}"
        ...
        self.buffer.upsert(new_item)
```
- **Selection won’t kick in with your defaults.** `min_buffer_size=128` + examples `-n 20/50` means the buffer is below threshold; you keep sampling base tasks. That defeats ExIt’s selection/expansion loop in small runs (exactly your quickstart).
```28:33:/Users/skrachur/Desktop/prime-env-hub-practice/environments/self_improve/README.md
- **Primary**: GSM8K ... 
...
33:  -a '{"num_train_examples": 50, ... "min_buffer_size": 128, ... }'
```
```207:214:/Users/skrachur/Desktop/prime-env-hub-practice/environments/self_improve/self_improve.py
    def sample_episode(...):
        use_buffer = (len(self.buffer) >= self.buffer.min_size) and (random.random() < self.select_prob)
        if use_buffer:
            chosen = self.buffer.sample()
            if chosen is not None:
                return self.build_iter_episode(chosen)
        return self.build_base_episode(...)
```
- **No diversity bonus.** The multiplicative diversity scaling on advantages (distance to centroid) is a core ExIt variant; you acknowledge it but don’t provide a hook in the trainer.
- **No K-step inference-time self-improvement evaluation loop.** Paper evaluates K-step self-improvement at test time. Your env produces only single episodes; there’s no built-in scaffold to iterate k=1..K at inference.

### Medium issues (will degrade effectiveness)
- **Parent score inheritance rarely matters because parent scores are rarely updated.** You only update scores when a “group” hits threshold via `end_group`. That likely never fires in verifiers’ standard loops.
- **Learnability signal is confounded over time.** Even if `group_size_hint` fills, you’re aggregating returns across different policy states and possibly different modes (improve/diverge) for the same `instance_id`. That’s not the same as per-iteration, contemporaneous GRPO grouping.
- **Single-turn only and no per-turn prefix selection.** Fine for GSM8K, but the code doesn’t generalize to multi-turn ExIt where selection picks τ^k′_{t−} prefixes. Your buffer items carry a flat `prev_response` snapshot, not a structured partial history.

### Minor nits
- **README drift**: Mentions `embedding_dim` and “Mapping to Algorithm 1” claims you insert all per-turn partials—code doesn’t.
```54:55:/Users/skrachur/Desktop/prime-env-hub-practice/environments/self_improve/README.md
| `embedding_dim` | int | `128` | Embedding dim for (optional) centroid tracking |
```
- **Scoring scale**: S computed from 0/1 correctness yields many zero-variance episodes early; encourage recording the normalized return actually used for learning.

### How to fix (in order of impact)
- **Wire real GRPO grouping.**
  - Add an adapter so the trainer asks the env for a batch of G rollouts per same `instance_id` under a fixed policy snapshot; call `begin_group()` before generating those G, `end_group()` right after. Record the exact normalized returns used for A_i (not r_new).
  - If you can’t touch the trainer, simulate G per item internally (loop G times in the rubric path or via a custom sampler).
- **Insert all per-turn partial histories on expansion.**
  - On `on_episode_end`, reconstruct τ^{k+1}_*, enumerate per-turn partial histories, upsert each with inherited score (and 0 if solved), per paper.
- **Lower `min_buffer_size` in docs and defaults.**
  - Set `min_buffer_size` ≈ 8–32 for toy runs, or auto-scale based on dataset size.
- **Track and prioritize by the right signal.**
  - Record the normalized per-rollout return used for advantage/selection (for self-iteration steps, the improvement reward or final normalized quality, per domain). Not just r_new.
- **Offer a diversity bonus hook.**
  - Expose a toggle to return per-rollout embeddings (even a simple textual embedding), and provide a multiplicative advantage scaler in the trainer path.
- **Provide a K-step inference scaffold for eval.**
  - Add an evaluation utility that runs K-step self-improvement at test time (paper’s metric), so users can see the actual self-improvement curve.

### Overall
- **Grade for paper-faithfulness (GSM8K setup): 5/10.** You captured the surface mechanics (prompts, improvement reward, a buffer), but the missing grouping, weak S estimation, no full partial-history insertion, and defaults that prevent selection mean you won’t see the ExIt behaviors the paper reports. Tighten those and you’ll be much closer.

I’ve reviewed the code and README and laid out precise breakpoints and fixes. If you want, I can draft the exact edits to: (1) enumerate/insert per-turn partials on expansion, (2) record and prioritize by the correct normalized returns, and (3) add a tiny “grouped rollouts” shim so S is contemporaneous.