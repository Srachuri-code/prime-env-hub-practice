# Core abstractions

class Policy:
    # Autoregressive LM with logprobs API
    def logprobs(self, input_tokens): ...
    def generate(self, prompt, stop_tokens): ...  # returns tokens, text, per-token logprobs

class Environment:
    # Wraps a task domain. Knows how to:
    # - render prompts from a task instance and partial history
    # - execute/evaluate outputs to produce turn-level rewards and a quality measure G
    # - normalize scalar scores into [0, 1] for learnability variance
    # - compute improvement reward for self-iteration steps
    def render_prompt(self, task_instance, mode): ...
    def evaluate_completion(self, task_instance, completion): ...
    def normalize_quality(self, quality): ...
    def improvement_reward(self, prev_norm_quality, new_norm_quality): ...
    def extract_solution_embedding(self, completion): ...

@dataclass
class PartialHistory:
    # τ^k_t = (y_t^k, y_<t^K, e_t^k, e_<t^K)
    # Store enough to reconstruct the prompt—text of latest iterate per previous turn, optional feedback
    turn_index: int                  # t
    iterate_index: int               # k
    last_iterate_text: str           # y_t^k
    prev_turn_last_iterates: list[str]  # y_<t^K
    feedback_current: Any | None     # e_t^k
    feedback_prev_turns: list[Any]   # e_<t^K

@dataclass
class TaskInstance:
    # Self-improvement task instance (m, k, τ^k_t) plus budget placeholder (K not used at train-time)
    base_task_id: str                # m
    partial_history: PartialHistory  # τ^k_t
    next_iterate_index: int          # k+1 target to generate
    score: float = 0.0               # learnability score S for prioritized sampling (variance over normalized returns)

class PrioritizedTaskBuffer:
    def __init__(self, capacity_N: int, min_ready_Bmin: int, kappa_inverse_temperature: float):
        self.capacity = capacity_N
        self.min_ready = min_ready_Bmin
        self.kappa = kappa_inverse_temperature
        self.items: list[TaskInstance] = []  # maintained sorted by score descending

    def size(self) -> int: ...
    def _min_score(self) -> float: ...
    def _softmax_probs(self) -> list[float]:
        # P(i) ∝ exp(κ * S_i)
        S = [ti.score for ti in self.items]
        numer = [exp(self.kappa * s) for s in S]
        denom = sum(numer) if numer else 1.0
        return [x / denom for x in numer]

    def sample(self) -> TaskInstance:
        # Sample according to softmax over scores
        probs = self._softmax_probs()
        return random_choice(self.items, probs)

    def insert_or_replace(self, instance: TaskInstance):
        # Keep only top-N by score; replace lowest if incoming >= lowest
        if len(self.items) < self.capacity:
            self.items.append(instance)
        else:
            j = argmin([ti.score for ti in self.items])
            if instance.score >= self.items[j].score:
                self.items[j] = instance
        self.items.sort(key=lambda ti: ti.score, reverse=True)

# GRPO utilities

@dataclass
class Rollout:
    tokens: list[int]
    logprobs_current: list[float]  # log πθ(o_i,t | prefix)
    logprobs_old: list[float]      # log π_old(o_i,t | prefix)
    logprobs_ref: list[float]      # for KL, optional efficient estimator
    text: str
    return_scalar: float           # scalar return for this completion (normalized [0, 1] for S; raw for objective if desired)
    embedding: np.ndarray | None

def compute_group_advantages(returns: list[float]) -> list[float]:
    # GRPO group-relative advantage; same A_i assigned to all tokens in rollout i
    mu = mean(returns)
    sigma = std(returns)
    if sigma <= 1e-8:
        return [0.0 for _ in returns]
    return [(r - mu) / sigma for r in returns]  # Equation (3)

def apply_diversity_bonus(advantages: list[float], embeddings: list[np.ndarray]) -> list[float]:
    # d_i = ||e_i - e_centroid|| / (max_j ||e_j - e_centroid|| - min_j ||e_j - e_centroid||)
    if any(e is None for e in embeddings):
        return advantages
    centroid = mean_vector(embeddings)
    dists = [l2_distance(e, centroid) for e in embeddings]
    denom = (max(dists) - min(dists)) if (max(dists) > min(dists)) else 1.0
    diversity = [(d - min(dists)) / denom for d in dists]
    return [A * diversity[i] for i, A in enumerate(advantages)]  # multiplicative bonus

def kl_per_token(logprobs_current: list[float], logprobs_ref: list[float]) -> list[float]:
    # Approximate D_KL(πθ || πref) via on-policy samples:
    # D_KL ≈ E_t[log πθ(o_t|h_t) - log πref(o_t|h_t)]
    return [lc - lr for lc, lr in zip(logprobs_current, logprobs_ref)]

def grpo_objective_and_grad(rollouts: list[Rollout], advantages: list[float], epsilon_clip: float, beta_kl: float):
    # J(θ) = (1/G) Σ_i (1/|o_i|) Σ_t [ min(ρ_i,t A_i, clip(ρ_i,t, 1-ε, 1+ε) A_i) - β KL(πθ || πref) ]
    # ρ_i,t = exp(logπθ - logπ_old)
    total = 0.0
    token_count = 0
    for i, r in enumerate(rollouts):
        A_i = advantages[i]
        for t in range(len(r.tokens)):
            rho = exp(r.logprobs_current[t] - r.logprobs_old[t])
            unclipped = rho * A_i
            clipped = clip(rho, 1.0 - epsilon_clip, 1.0 + epsilon_clip) * A_i
            surrogate = min(unclipped, clipped)
            kl_t = (r.logprobs_current[t] - r.logprobs_ref[t])  # per-token KL proxy
            total += (surrogate - beta_kl * kl_t)
            token_count += 1
    J = total / max(1, token_count)
    # In practice, auto-diff computes grad; here we return scalar J placeholder
    return J

# ExIt helpers

def enumerate_partial_histories(full_history: PartialHistory | list[PartialHistory]) -> list[PartialHistory]:
    # Given a new expanded history τ^{k+1}_*, generate all per-turn partial histories that can be used by selection
    # Returns a list of τ^{k'}_{t-} for all t- ≤ current and all k' ≤ current iterate at t-
    # Implementation depends on how histories are represented; assume env provides helper:
    return env.enumerate_per_turn_partial_histories(full_history)

def initialize_child_score_from_parent(parent_score: float) -> float:
    # As per paper: initialize new m+ with same score as m′
    return parent_score

def sample_selection_prefix(parent_instance: TaskInstance) -> PartialHistory:
    # Sample random turn prefix (1..t) and iterate index k' (0..k) from τ^k_t
    return env.sample_random_partial_prefix(parent_instance.partial_history)

# Training configuration

@dataclass
class Config:
    buffer_capacity_N: int
    buffer_min_ready_Bmin: int
    selection_prob_p: float
    divergence_prob_pdiv: float
    kappa_inverse_temperature: float
    group_size_G: int
    clip_epsilon: float
    kl_beta: float
    ref_update_interval_M: int
    ref_update_alpha: float  # θ_ref ← α θ_ref + (1-α) θ
    learning_rate: float
    total_train_iterations: int
    prompts_per_batch: int  # number of task instances per train iteration
    rollouts_per_prompt: int  # equals G in vanilla GRPO
    use_diversity_bonus: bool

# Main training loop

def train_exit_grpo(base_tasks: list[str], env: Environment, policy: Policy, cfg: Config):
    buffer = PrioritizedTaskBuffer(cfg.buffer_capacity_N, cfg.buffer_min_ready_Bmin, cfg.kappa_inverse_temperature)
    theta = policy  # current trainable policy
    theta_old = deepcopy(theta)  # snapshot used to generate this iteration's rollouts
    theta_ref = deepcopy(theta)  # reference policy for KL
    optimizer = Adam(theta.parameters(), lr=cfg.learning_rate)

    for it in range(1, cfg.total_train_iterations + 1):

        batch_instances: list[TaskInstance] = []

        # 1) Sample a batch of starting instances via selection vs base tasks
        for b in range(cfg.prompts_per_batch):
            use_selection = (buffer.size() >= buffer.min_ready) and (random_uniform() < cfg.selection_prob_p)
            if use_selection:
                parent = buffer.sample()  # (m, k, τ^k_t) with score S
                prefix = sample_selection_prefix(parent)  # τ^{k'}_{t-}
                instance = TaskInstance(
                    base_task_id=parent.base_task_id,
                    partial_history=prefix,
                    next_iterate_index=prefix.iterate_index + 1,
                    score=parent.score  # temporary; will be updated after rollouts
                )
            else:
                # Sample new base task; no self-iteration yet (t=1, k=0)
                m = random_choice(base_tasks)
                base_ph = env.initial_partial_history(m)  # τ^0_1 (no self-improvement)
                instance = TaskInstance(
                    base_task_id=m,
                    partial_history=base_ph,
                    next_iterate_index=base_ph.iterate_index + 1,
                    score=0.0
                )
            batch_instances.append(instance)

        # 2) For each instance: do one-step self-iteration (expansion) under GRPO with a group of G rollouts
        all_updates: list[tuple[TaskInstance, list[Rollout], float, TaskInstance]] = []
        # tuple = (parent instance m′, rollout group, S_variance, child instance m+)

        for inst in batch_instances:
            # 2.a) Decide Improve vs Diverge for this expansion
            mode = "diverge" if (random_uniform() < cfg.divergence_prob_pdiv) else "improve"

            # 2.b) Build prompts and generate G rollouts with θ_old
            rollouts: list[Rollout] = []
            for g in range(cfg.rollouts_per_prompt):
                prompt = env.render_prompt(inst, mode)  # includes instruction, τ^{k’}_{t-}, optional feedback
                tokens, text, lp_old = theta_old.generate(prompt, stop_tokens=env.stop_tokens())
                # Compute current logprobs (θ) and ref logprobs (θ_ref) for IS ratio and KL
                lp_cur = theta.logprobs(prompt + tokens)
                lp_ref = theta_ref.logprobs(prompt + tokens)

                # 2.c) Evaluate rollout to get scalar return and quality measures
                eval_out = env.evaluate_completion(inst, text)
                # eval_out should include:
                # - per-turn rewards (for base tasks)
                # - final quality measure G(τ_hat) or per-step reward depending on domain
                # - normalized quality in [0, 1] for variance metric
                # - prev_norm_quality if this is a self-iteration step
                # Rollout return policy:
                # - For base tasks: use env's normalized quality (or sum of turn rewards) as r_i
                # - For self-iteration steps: r_i = improvement_reward(prev_norm_quality, new_norm_quality)
                if eval_out.kind == "self_iteration":
                    r_i = env.improvement_reward(eval_out.prev_norm_quality, eval_out.new_norm_quality)
                    r_i_norm_for_S = r_i  # already in [0,1] by construction
                else:
                    # base task or divergence step uses normalized quality in [0,1]
                    r_i = eval_out.scalar_return  # can be raw reward used for objective
                    r_i_norm_for_S = env.normalize_quality(eval_out.quality_scalar)

                emb = env.extract_solution_embedding(text) if cfg.use_diversity_bonus else None

                rollouts.append(Rollout(
                    tokens=tokens,
                    logprobs_current=lp_cur,
                    logprobs_old=lp_old,
                    logprobs_ref=lp_ref,
                    text=text,
                    return_scalar=r_i_norm_for_S,  # used for A_i and S; objective uses lp/ρ and A_i
                    embedding=emb
                ))

            # 2.d) Compute group advantage A_i from normalized returns; apply diversity multiplier if enabled
            returns_for_A = [r.return_scalar for r in rollouts]
            A = compute_group_advantages(returns_for_A)
            if cfg.use_diversity_bonus:
                A = apply_diversity_bonus(A, [r.embedding for r in rollouts])

            # 2.e) Compute GRPO objective (with clipping and KL) and update θ
            J = grpo_objective_and_grad(rollouts, A, cfg.clip_epsilon, cfg.kl_beta)
            optimizer.zero_grad()
            (-J).backward()                      # maximize J
            optimizer.step()

            # 2.f) Compute learnability score S = var(returns) over normalized [0,1] returns
            S = variance(returns_for_A)          # already normalized to [0,1] scale

            # 2.g) Create child instance m+ = (m, k+1, τ^{k+1}_*) from the best rollout or env reconstruction
            # Let env reconstruct τ^{k+1}_* from inst.partial_history and the chosen rollout text
            best_idx = argmax(returns_for_A)
            child_full_history = env.compose_expanded_history(inst.partial_history, rollouts[best_idx].text)
            child_ph_list = enumerate_partial_histories(child_full_history)  # all per-turn partials, including τ^{k+1}_*
            # Insert each partial as a TaskInstance for selection; initialize child scores with parent S
            child_instances = []
            for ph in child_ph_list:
                child_instances.append(TaskInstance(
                    base_task_id=inst.base_task_id,
                    partial_history=ph,
                    next_iterate_index=ph.iterate_index + 1,
                    score=initialize_child_score_from_parent(inst.score)  # init with parent empirical S
                ))

            # 2.h) Stage updates for buffer: parent gets updated score S; children inserted with inherited S
            # Record (parent, group rollouts, S, child set) to apply after the batch (optional)
            all_updates.append((inst, rollouts, S, child_instances))

        # 3) After processing the batch, update buffer with parents and children
        for (parent, rollouts, S, child_instances) in all_updates:
            # Update parent instance score to observed learnability S and insert/replace
            parent.score = S
            buffer.insert_or_replace(parent)
            # Insert all children partial histories (top-N retained by buffer)
            for child in child_instances:
                buffer.insert_or_replace(child)

        # 4) Housekeeping: update old/ref policies on schedule
        theta_old.load_state_dict(theta.state_dict())  # rollout policy = latest θ for next iteration
        if (it % cfg.ref_update_interval_M) == 0:
            # θ_ref ← α θ_ref + (1 - α) θ  (exponential moving ref)
            ema_update(theta_ref, theta, alpha=cfg.ref_update_alpha)

    return theta  # fine-tuned policy

# Inference-time self-improvement (K-step at test time)

def inference_self_improve(env: Environment, policy: Policy, base_task_id: str, K: int, mode_schedule: str = "improve_with_diverge"):
    # Given an instruction/task m, iteratively improve for K steps per turn using prompts
    τ = env.initial_partial_history(base_task_id)  # τ^0_1
    for k in range(1, K + 1):
        if mode_schedule == "improve_with_diverge":
            mode = "diverge" if random_uniform() < 0.5 else "improve"  # or domain-specific schedule
        else:
            mode = "improve"
        prompt = env.render_prompt(TaskInstance(base_task_id, τ, τ.iterate_index + 1), mode)
        tokens, text, _ = policy.generate(prompt, stop_tokens=env.stop_tokens())
        τ = env.compose_expanded_history(τ, text)  # τ^{k}_*
    return env.finalize_solution(τ)