# self-improve

### Overview
- **Environment ID**: `self-improve`
- **Short description**: ExIt+GRPO-style self-improvement over GSM8K. Multi-turn protocol with improve/diverge modes, per-step improvement reward, and parser-based correctness.
- **Tags**: self-improvement, multi-turn, training, eval, gsm8k

References: [Overview](https://verifiers.readthedocs.io/en/latest/overview.html), [Components](https://verifiers.readthedocs.io/en/latest/components.html)

### Datasets
- **Primary**: GSM8K train/eval loaded via Verifiers utils.
- Use `--env-args` to subset train/eval counts.

### Task
- **Type**: multi-turn
- **Parser**: `ThinkParser` (default) or `Parser`, extracting boxed final answers.
- **Rubric overview**:
  - `exit_reward(prompt, completion, answer, state)`: quality on first step, improvement reward thereafter (max(0, Q_new - Q_prev)) with Q = exact correctness.
  - Logged metrics: exact `correct_answer`, `format_reward`, heuristic `normalized_quality`, heuristic `improved_over_previous`.

### Quickstart
Run an evaluation with defaults (3 improvement turns, built-in prompts):

```bash
uv run vf-eval self-improve -n 20 -r 1 -m gpt-4o-mini
```

Custom prompts and turns:

```bash
uv run vf-eval self-improve \
  -m gpt-4o-mini -n 50 -r 1 \
  -a '{"max_turns": 3, "num_train_examples": 50, "num_eval_examples": 50, "use_think": true, "divergence_prob": 0.3}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `max_turns` | int | `3` | Number of self-iteration steps |
| `num_train_examples` | int | `-1` | Limit GSM8K train size |
| `num_eval_examples` | int | `-1` | Limit GSM8K eval size |
| `use_think` | bool | `true` | Use `ThinkParser` to extract boxed answers |
| `divergence_prob` | float | `0.0` | Probability of `diverge` mode each step |
| `system_prompt` | str | boxed | System prompt for boxed reasoning/answers |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main ExIt reward (correctness on first step; improvement thereafter) |
| `correct_answer` | Exact-match correctness |
| `format_reward` | Parser format adherence |
| `normalized_quality` | Heuristic quality in [0,1] |
| `improved_over_previous` | Heuristic improvement measure |

### Mapping to pseudocode (ExIt + GRPO)
- `env_response`: chooses mode per step (improve/diverge), updates partial history τ^k_t, and issues mode-specific instruction (render_prompt analogue).
- `is_completed`: enforces K-step budget.
- `Rubric.exit_reward`: uses normalized quality on first step; improvement reward on subsequent steps, matching Section “Self-iteration reward”.
- Diversity/GRPO aspects are left to the trainer; this env focuses on rollout protocol and rewards.

