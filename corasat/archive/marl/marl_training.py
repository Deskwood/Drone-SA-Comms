"""
Training utilities for multi-agent reinforcement learning on the CORASAT drone environment.

The module implements a double DQN learner with parameter sharing and configurable
n-step returns between drones.  The focus is on clarity and extensibility rather
than peak performance, making it a good starting point for experimentation inside
the ``marl.ipynb`` notebook.
"""

from __future__ import annotations

import random
from collections import deque
import warnings
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from marl_env import ACTION_LOOKUP, CorasatMultiAgentEnv


Experience = Tuple[np.ndarray, int, float, np.ndarray, bool]


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    discount: float


def _make_n_step_transition(
    buffer: Deque[Experience], gamma: float, n_step: int
) -> Optional[Transition]:
    if not buffer:
        return None
    # Accumulate rewards until n steps or episode termination to build n-step targets.
    reward = 0.0
    gamma_power = 1.0
    steps = 0
    done_flag = False
    next_state = buffer[0][3]

    for idx, (_, _, r, next_state, done) in enumerate(buffer):
        reward += gamma_power * r
        steps = idx + 1
        if done:
            done_flag = True
            break
        if steps == n_step:
            break
        gamma_power *= gamma

    discount = 0.0 if done_flag else gamma ** steps
    state, action = buffer[0][0], buffer[0][1]
    buffer.popleft()
    return Transition(
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        done=done_flag,
        discount=discount,
    )


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_frames: int = 200_000,
        epsilon: float = 1e-5,
    ):
        self.capacity = int(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = max(1, int(beta_frames))
        self.epsilon = epsilon

        self.storage: List[Transition] = []
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.position = 0
        self.frame = 1
        self.max_priority = 1.0

    def __len__(self) -> int:
        return len(self.storage)
    # New transitions inherit max priority so they are sampled promptly.
    def add(self, transition: Transition) -> None:
        idx = self.position
        if len(self.storage) < self.capacity:
            self.storage.append(transition)
        else:
            self.storage[idx] = transition
        self.priorities[idx] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def _current_beta(self) -> float:
        progress = min(1.0, self.frame / float(self.beta_frames))
        return self.beta_start + progress * (self.beta_end - self.beta_start)

    def sample(self, batch_size: int) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        size = len(self.storage)
        if size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        batch_size = min(batch_size, size)
        probs = self.priorities[:size] ** self.alpha
        probs_sum = probs.sum()
        # Fallback to uniform distribution if priorities collapse numerically.
        if probs_sum <= 0:
            probs = np.full(size, 1.0 / size, dtype=np.float32)
        else:
            probs /= probs_sum
        indices = np.random.choice(size, batch_size, p=probs)
        samples = [self.storage[idx] for idx in indices]

        beta = self._current_beta()
        self.frame += batch_size

        weights = (size * probs[indices]) ** (-beta)
        weights /= weights.max() if weights.max() > 0 else 1.0
        return samples, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, errors: np.ndarray) -> None:
        errors = np.abs(errors) + self.epsilon
        for idx, error in zip(indices, errors):
            self.priorities[idx] = error
            self.max_priority = max(self.max_priority, float(error))


# ---------------------------------------------------------------------------
# Neural network policy
# ---------------------------------------------------------------------------


class QNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Tuple[int, ...] = (1024, 1024, 512),
        dueling: bool = True,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            linear = nn.Linear(prev_dim, hidden_dim)
            nn.init.kaiming_uniform_(linear.weight, nonlinearity="relu")
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.feature_extractor = nn.Sequential(*layers)

        self.dueling = dueling
        if dueling:
            self.value_head = nn.Linear(prev_dim, 1)
            self.advantage_head = nn.Linear(prev_dim, output_dim)
            nn.init.kaiming_uniform_(self.value_head.weight, nonlinearity="linear")
            nn.init.zeros_(self.value_head.bias)
            nn.init.kaiming_uniform_(self.advantage_head.weight, nonlinearity="linear")
            nn.init.zeros_(self.advantage_head.bias)
        else:
            self.q_head = nn.Linear(prev_dim, output_dim)
            nn.init.kaiming_uniform_(self.q_head.weight, nonlinearity="linear")
            nn.init.zeros_(self.q_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.feature_extractor(x)
        if not self.dueling:
            return self.q_head(features)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        advantage = advantage - advantage.mean(dim=1, keepdim=True)
        return value + advantage


# ---------------------------------------------------------------------------
# DQN agent with parameter sharing
# ---------------------------------------------------------------------------


class SharedDQNAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        device: Optional[torch.device] = None,
        hidden_dims: Tuple[int, ...] = (1024, 1024, 512),
        gradient_clip: float = 5.0,
        dueling: bool = True,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gradient_clip = gradient_clip
        self.dueling = dueling

        def _init_networks(target_device: torch.device) -> Tuple[QNetwork, QNetwork]:
            policy = QNetwork(obs_dim, action_dim, hidden_dims, dueling=dueling).to(target_device)
            target = QNetwork(obs_dim, action_dim, hidden_dims, dueling=dueling).to(target_device)
            target.load_state_dict(policy.state_dict())
            return policy, target

        self.q_network, self.target_network = _init_networks(self.device)

        if device is None and self.device.type == "cuda":
            try:
                with torch.no_grad():
                    dummy_state = torch.zeros(1, obs_dim, device=self.device)
                    self.q_network(dummy_state)
            except RuntimeError as exc:
                error_message = str(exc).lower()
                if "cuda" not in error_message:
                    raise
                warnings.warn(
                    f"CUDA execution failed during initialisation ('{exc}'); falling back to CPU.",
                    RuntimeWarning,
                )
                self.device = torch.device("cpu")
        self.q_network, self.target_network = _init_networks(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        actions = self.select_action_batch(np.expand_dims(state, axis=0), epsilon)
        return int(actions[0])

    def select_action_batch(self, states: np.ndarray, epsilon: float) -> np.ndarray:
        if states.size == 0:
            return np.array([], dtype=np.int64)
        batch_size = states.shape[0]
        greedy_actions: np.ndarray
        if epsilon < 1.0:
            state_tensor = torch.from_numpy(states).float().to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            greedy_actions = q_values.argmax(dim=1).cpu().numpy()
        else:
            greedy_actions = np.zeros(batch_size, dtype=np.int64)
        if epsilon <= 0.0:
            return greedy_actions
        random_actions = np.random.randint(0, self.action_dim, size=batch_size, dtype=np.int64)
        mask = np.random.rand(batch_size) < epsilon
        greedy_actions[mask] = random_actions[mask]
        return greedy_actions

    def update(
        self, batch: List[Transition], weights: Optional[torch.Tensor] = None
    ) -> Tuple[float, np.ndarray]:
        if not batch:
            return 0.0, np.zeros(0, dtype=np.float32)

        states = torch.from_numpy(np.stack([transition.state for transition in batch])).float().to(
            self.device
        )
        actions = torch.tensor([transition.action for transition in batch], dtype=torch.long).to(
            self.device
        )
        rewards = torch.tensor([transition.reward for transition in batch], dtype=torch.float32).to(
            self.device
        )
        next_states = torch.from_numpy(
            np.stack([transition.next_state for transition in batch])
        ).float().to(self.device)
        dones = torch.tensor([transition.done for transition in batch], dtype=torch.float32).to(
            self.device
        )
        discounts = torch.tensor(
            [transition.discount for transition in batch], dtype=torch.float32
        ).to(self.device)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_online_q = self.q_network(next_states)
            next_actions = next_online_q.argmax(dim=1, keepdim=True)
            next_target_q = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            # Double-DQN target: evaluate argmax under online network, value via target net.
            targets = rewards + (1.0 - dones) * discounts * next_target_q

        td_errors = targets - q_values
        loss_elements = F.smooth_l1_loss(q_values, targets, reduction="none")
        if weights is not None:
            weights = weights.to(self.device)
            loss = (weights * loss_elements).mean()
        else:
            loss = loss_elements.mean()
        self.optimizer.zero_grad()
        loss.backward()
        if self.gradient_clip > 0:
            nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clip)
        self.optimizer.step()
        return float(loss.item()), td_errors.detach().cpu().numpy()

    def sync_target(self, tau: float = 1.0) -> None:
        tau = float(tau)
        if tau >= 1.0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            return

        target_state = self.target_network.state_dict()
        online_state = self.q_network.state_dict()
        for key in target_state:
            target_state[key] = (1 - tau) * target_state[key] + tau * online_state[key]
        self.target_network.load_state_dict(target_state)

    def save(self, filepath: str) -> None:
        torch.save(self.q_network.state_dict(), filepath)

    def load(self, filepath: str) -> None:
        state_dict = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(state_dict)
        self.sync_target()


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------


@dataclass
class TrainingStats:
    episode: int
    epsilon: float
    total_reward: Dict[int, float]
    score: float
    correct_edges: int
    false_edges: int
    reported_edges: int
    loss: float
    eval_score: Optional[float] = None
    eval_correct_edges: Optional[float] = None
    eval_false_edges: Optional[float] = None
    playback: Optional[EpisodePlayback] = None


@dataclass
class EpisodePlayback:
    board_size: Tuple[int, int]
    trajectories: Dict[int, List[Tuple[int, int]]]
    figures: List[Dict[str, object]]
    reported_edges: List[str]
    score_summary: Dict[str, float]
    snapshots: List[Dict[str, object]]


def train_marl(
    env: CorasatMultiAgentEnv,
    num_episodes: int = 500,
    batch_size: int = 128,
    buffer_capacity: int = 100_000,
    learning_rate: float = 5e-4,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    warmup_steps: int = 500,
    target_sync_interval: int = 250,
    n_step: int = 3,
    soft_tau: float = 0.005,
    dueling: bool = True,
    hidden_dims: Tuple[int, ...] = (1024, 1024, 512),
    replay_alpha: float = 0.6,
    replay_beta_start: float = 0.4,
    replay_beta_end: float = 1.0,
    replay_beta_frames: int = 200_000,
    min_priority: float = 1e-5,
    reward_overrides: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
    agent: Optional[SharedDQNAgent] = None,
    eval_interval: Optional[int] = None,
    eval_episodes: int = 5,
    evaluation_env: Optional[CorasatMultiAgentEnv] = None,
    score_threshold: Optional[float] = None,
    stats_callback: Optional[Callable[[TrainingStats, Optional["EvaluationResult"]], None]] = None,
    record_playbacks_every: Optional[int] = None,
    parallel_envs: int = 1,
) -> Tuple[SharedDQNAgent, List[TrainingStats]]:
    """Train the shared DQN agent on the CORASAT environment and return the fitted model plus metrics."""
    if env.observation_size is None:
        # Force initial observation to determine dimensionality.
        env.reset(seed=seed)

    obs_dim = env.observation_size or 1
    action_dim = len(ACTION_LOOKUP)

    if reward_overrides:
        env.reward_config.update(reward_overrides)

    agent = agent or SharedDQNAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr=learning_rate,
        gamma=gamma,
        hidden_dims=hidden_dims,
        dueling=dueling,
    )
    replay_buffer = PrioritizedReplayBuffer(
        capacity=buffer_capacity,
        alpha=replay_alpha,
        beta_start=replay_beta_start,
        beta_end=replay_beta_end,
        beta_frames=replay_beta_frames,
        epsilon=min_priority,
    )
    n_step = max(1, int(n_step))

    eval_env = evaluation_env
    if eval_interval:
        if eval_env is None:
            eval_env = CorasatMultiAgentEnv(reward_config=dict(env.reward_config))

    epsilon = epsilon_start
    stats_history: List[TrainingStats] = []
    global_step = 0
    global_update_count = 0
    global_loss_accum = 0.0

    parallel_envs = max(1, int(parallel_envs))
    env_instances: List[CorasatMultiAgentEnv] = [env]
    for _ in range(1, parallel_envs):
        env_instances.append(CorasatMultiAgentEnv(reward_config=dict(env.reward_config)))

    episodes_started = 0
    episodes_completed = 0

    active_states: List[Dict[str, object]] = []
    batch_states_buffer: Optional[np.ndarray] = None

    def _initialise_env_state(
        env_instance: CorasatMultiAgentEnv,
        episode_number: int,
    ) -> Dict[str, object]:
        """Create bookkeeping structures for a fresh episode within an environment copy."""
        episode_seed = None if seed is None else seed + episode_number
        observations = env_instance.reset(seed=episode_seed)
        episode_reward = {agent_id: 0.0 for agent_id in env_instance.agent_ids}
        episode_buffers = {agent_id: deque() for agent_id in env_instance.agent_ids}
        # Only capture rich playback artefacts for configured episode cadence to save memory.
        capture_episode = (
            record_playbacks_every is not None
            and record_playbacks_every > 0
            and episode_number % record_playbacks_every == 0
        )
        playback_trajectories: Optional[Dict[int, List[Tuple[int, int]]]] = None
        playback_snapshots: Optional[List[Dict[str, object]]] = None
        if capture_episode:
            playback_trajectories = {
                agent_id: [tuple(env_instance.drones[agent_id].position)]
                for agent_id in env_instance.agent_ids
            }
            playback_snapshots = []
        state: Dict[str, object] = {
            "env": env_instance,
            "agent_ids": tuple(env_instance.agent_ids),
            "observations": observations,
            "episode_reward": episode_reward,
            "episode_buffers": episode_buffers,
            "capture_episode": capture_episode,
            "playback_trajectories": playback_trajectories,
            "playback_snapshots": playback_snapshots,
            "episode_number": episode_number,
            "start_update_count": global_update_count,
            "start_loss_accum": global_loss_accum,
        }
        return state

    def _launch_env(env_instance: CorasatMultiAgentEnv) -> Optional[Dict[str, object]]:
        """Start a new episode on the given environment instance if capacity remains."""
        nonlocal episodes_started
        if episodes_started >= num_episodes:
            return None
        episode_number = episodes_started + 1
        state = _initialise_env_state(env_instance, episode_number)
        active_states.append(state)
        episodes_started += 1
        return state

    for env_instance in env_instances:
        if episodes_started >= num_episodes:
            break
        _launch_env(env_instance)

    def _push_experience(
        state: Dict[str, object],
        agent_id: int,
        experience: Experience,
    ) -> None:
        buffer_deque: Deque[Experience] = state["episode_buffers"][agent_id]  # type: ignore[assignment]
        buffer_deque.append(experience)
        if len(buffer_deque) >= n_step:
            # Emit a completed n-step transition once the window is filled.
            transition = _make_n_step_transition(buffer_deque, agent.gamma, n_step)
            if transition is not None:
                replay_buffer.add(transition)
        if experience[-1]:
            while buffer_deque:
                transition = _make_n_step_transition(buffer_deque, agent.gamma, n_step)
                if transition is None:
                    break
                replay_buffer.add(transition)

    while episodes_completed < num_episodes:
        if not active_states:
            break

        total_agents = sum(len(state["agent_ids"]) for state in active_states)  # type: ignore[index]
        if total_agents <= 0:
            break
        if batch_states_buffer is None or batch_states_buffer.shape[0] < total_agents:
            batch_states_buffer = np.empty((total_agents, obs_dim), dtype=np.float32)
        batch_states_array = batch_states_buffer[:total_agents]
        dispatch: List[Tuple[Dict[str, object], int, int]] = []
        cursor = 0
        for state in active_states:
            observations: Dict[int, np.ndarray] = state["observations"]  # type: ignore[assignment]
            for agent_id in state["agent_ids"]:  # type: ignore[index]
                batch_states_array[cursor] = observations[agent_id]
                dispatch.append((state, agent_id, cursor))
                cursor += 1

        # One batched forward pass covers every agent across all parallel environments.
        actions_array = agent.select_action_batch(batch_states_array, epsilon)
        for state, agent_id, idx in dispatch:
            state.setdefault("pending_actions", {})  # type: ignore[call-arg]
            pending_actions: Dict[int, int] = state["pending_actions"]  # type: ignore[assignment]
            pending_actions[agent_id] = int(actions_array[idx])

        for state in list(active_states):
            env_instance: CorasatMultiAgentEnv = state["env"]  # type: ignore[assignment]
            actions: Dict[int, int] = state.pop("pending_actions", {})  # type: ignore[assignment]
            if not actions:
                continue
            observations: Dict[int, np.ndarray] = state["observations"]  # type: ignore[assignment]
            episode_reward: Dict[int, float] = state["episode_reward"]  # type: ignore[assignment]
            next_obs, rewards, terminated, _, _ = env_instance.step(actions)
            global_step += 1

            for agent_id in state["agent_ids"]:  # type: ignore[index]
                episode_reward[agent_id] += rewards[agent_id]
                experience: Experience = (
                    observations[agent_id],
                    actions[agent_id],
                    rewards[agent_id],
                    next_obs[agent_id],
                    terminated[agent_id],
                )
                _push_experience(state, agent_id, experience)

            state["observations"] = next_obs  # type: ignore[assignment]

            if (
                state["capture_episode"]  # type: ignore[index]
                and state["playback_trajectories"] is not None
                and state["playback_snapshots"] is not None
            ):
                playback_trajectories: Dict[int, List[Tuple[int, int]]] = state["playback_trajectories"]  # type: ignore[assignment]
                playback_snapshots: List[Dict[str, object]] = state["playback_snapshots"]  # type: ignore[assignment]
                positions = {
                    agent_id: tuple(env_instance.drones[agent_id].position)
                    for agent_id in state["agent_ids"]  # type: ignore[index]
                }
                for agent_id, path in playback_trajectories.items():
                    path.append(positions[agent_id])
                playback_snapshots.append(
                    {
                        "round": env_instance.round,
                        "positions": positions,
                        "reported_edges": list(env_instance.reported_edges),
                    }
                )

            if len(replay_buffer) >= batch_size and global_step > warmup_steps:
                # Prioritised sampling keeps gradient steps focused on informative transitions.
                sampled, indices, weights_np = replay_buffer.sample(batch_size)
                weights_tensor = torch.from_numpy(weights_np).float().to(agent.device)
                loss, td_errors = agent.update(sampled, weights_tensor)
                replay_buffer.update_priorities(indices, td_errors)
                global_loss_accum += loss
                global_update_count += 1
                if soft_tau > 0:
                    agent.sync_target(tau=soft_tau)

            if target_sync_interval and global_step % target_sync_interval == 0:
                agent.sync_target()

            if all(terminated.values()):
                episode_buffers: Dict[int, Deque[Experience]] = state["episode_buffers"]  # type: ignore[assignment]
                for buffer_deque in episode_buffers.values():
                    while buffer_deque:
                        transition = _make_n_step_transition(buffer_deque, agent.gamma, n_step)
                        if transition is None:
                            break
                        replay_buffer.add(transition)

                epsilon = max(epsilon_end, epsilon * epsilon_decay)
                # The epsilon schedule is applied per finished episode to keep exploration in sync.

                summary = env_instance.get_score_summary()
                playback_record: Optional[EpisodePlayback] = None
                if (
                    state["capture_episode"]  # type: ignore[index]
                    and state["playback_trajectories"] is not None
                    and state["playback_snapshots"] is not None
                ):
                    figures = [
                        {
                            "position": tuple(fig.position),
                            "color": fig.color,
                            "type": fig.figure_type,
                        }
                        for fig in env_instance.figures
                    ]
                    playback_record = EpisodePlayback(
                        board_size=(env_instance.width, env_instance.height),
                        trajectories=state["playback_trajectories"],  # type: ignore[arg-type]
                        figures=figures,
                        reported_edges=list(env_instance.reported_edges),
                        score_summary=summary,
                        snapshots=state["playback_snapshots"],  # type: ignore[arg-type]
                    )

                start_updates = state["start_update_count"]  # type: ignore[assignment]
                start_loss = state["start_loss_accum"]  # type: ignore[assignment]
                episode_update_count = global_update_count - start_updates
                episode_loss_sum = global_loss_accum - start_loss
                mean_loss = episode_loss_sum / max(1, episode_update_count)

                episode_number = state["episode_number"]  # type: ignore[assignment]
                stat_entry = TrainingStats(
                    episode=episode_number,
                    epsilon=epsilon,
                    total_reward=dict(state["episode_reward"]),  # type: ignore[arg-type]
                    score=summary["score"],
                    correct_edges=summary["correct_edges"],
                    false_edges=summary["false_edges"],
                    reported_edges=summary["reported_edges"],
                    loss=mean_loss,
                    playback=playback_record,
                )

                eval_result = None
                if (
                    eval_interval
                    and eval_interval > 0
                    and eval_env is not None
                    and episode_number % eval_interval == 0
                ):
                    eval_seed = None if seed is None else seed + 10_000 + episode_number
                    eval_result = evaluate_policy(
                        eval_env,
                        agent,
                        episodes=eval_episodes,
                        epsilon=0.0,
                        seed=eval_seed,
                    )
                    stat_entry.eval_score = eval_result.average_score
                    stat_entry.eval_correct_edges = eval_result.average_correct_edges
                    stat_entry.eval_false_edges = eval_result.average_false_edges
                    if score_threshold is not None and eval_result.average_score >= score_threshold:
                        stats_history.append(stat_entry)
                        if stats_callback:
                            stats_callback(stat_entry, eval_result)
                        agent.sync_target()
                        return agent, sorted(stats_history, key=lambda entry: entry.episode)

                stats_history.append(stat_entry)
                if stats_callback:
                    stats_callback(stat_entry, eval_result)

                episodes_completed += 1
                active_states.remove(state)

                if episodes_started < num_episodes:
                    new_state = _initialise_env_state(env_instance, episodes_started + 1)
                    active_states.append(new_state)
                    episodes_started += 1
                else:
                    # No more episodes for this environment.
                    pass

    agent.sync_target()
    stats_history.sort(key=lambda entry: entry.episode)
    return agent, stats_history


@dataclass
class EvaluationResult:
    episodes: int
    average_score: float
    average_correct_edges: float
    average_false_edges: float
    average_reward: Dict[int, float]
    raw_scores: List[dict]
    playbacks: List[EpisodePlayback] = field(default_factory=list)


def evaluate_policy(
    env: CorasatMultiAgentEnv,
    agent: SharedDQNAgent,
    episodes: int = 20,
    epsilon: float = 0.0,
    seed: Optional[int] = None,
) -> EvaluationResult:
    if env.observation_size is None:
        env.reset()

    rewards_accumulator = {agent_id: 0.0 for agent_id in env.agent_ids}
    score_history: List[dict] = []
    playback_history: List[EpisodePlayback] = []

    for episode in range(episodes):
        observations = env.reset(seed=None if seed is None else seed + episode)
        episode_rewards = {agent_id: 0.0 for agent_id in env.agent_ids}
        trajectories: Dict[int, List[Tuple[int, int]]] = {
            agent_id: [tuple(env.drones[agent_id].position)] for agent_id in env.agent_ids
        }
        snapshots: List[Dict[str, object]] = []

        while True:
            actions = {
                agent_id: agent.select_action(observations[agent_id], epsilon)
                for agent_id in env.agent_ids
            }
            next_obs, rewards, terminated, _, _ = env.step(actions)
            for agent_id in env.agent_ids:
                episode_rewards[agent_id] += rewards[agent_id]
            observations = next_obs

            positions = {
                agent_id: tuple(env.drones[agent_id].position) for agent_id in env.agent_ids
            }
            for agent_id, path in trajectories.items():
                path.append(positions[agent_id])
            reported_edges = list(env.reported_edges)
            snapshots.append(
                {
                    "round": env.round,
                    "positions": positions,
                    "reported_edges": reported_edges,
                }
            )
            if all(terminated.values()):
                break

        for agent_id in env.agent_ids:
            rewards_accumulator[agent_id] += episode_rewards[agent_id]
        summary = env.get_score_summary()
        score_history.append(summary)

        figures = [
            {
                "position": tuple(fig.position),
                "color": fig.color,
                "type": fig.figure_type,
            }
            for fig in env.figures
        ]

        playback_history.append(
            EpisodePlayback(
                board_size=(env.width, env.height),
                trajectories=trajectories,
                figures=figures,
                reported_edges=list(env.reported_edges),
                score_summary=summary,
                snapshots=snapshots,
            )
        )

    average_score = sum(summary["score"] for summary in score_history) / episodes
    average_correct = sum(summary["correct_edges"] for summary in score_history) / episodes
    average_false = sum(summary["false_edges"] for summary in score_history) / episodes
    average_reward = {
        agent_id: total_reward / episodes for agent_id, total_reward in rewards_accumulator.items()
    }

    return EvaluationResult(
        episodes=episodes,
        average_score=average_score,
        average_correct_edges=average_correct,
        average_false_edges=average_false,
        average_reward=average_reward,
        raw_scores=score_history,
        playbacks=playback_history,
    )


def play_episode(
    env: CorasatMultiAgentEnv,
    agent: SharedDQNAgent,
    epsilon: float = 0.0,
    seed: Optional[int] = None,
) -> EpisodePlayback:
    """Run a single episode using the current agent and return trajectory details."""
    temp_env = CorasatMultiAgentEnv(reward_config=dict(env.reward_config))
    observations = temp_env.reset(seed=seed)
    trajectories: Dict[int, List[Tuple[int, int]]] = {
        agent_id: [tuple(temp_env.drones[agent_id].position)] for agent_id in temp_env.agent_ids
    }
    snapshots: List[Dict[str, object]] = []
    reported_edges: List[str] = []

    while True:
        actions = {
            agent_id: agent.select_action(observations[agent_id], epsilon)
            for agent_id in temp_env.agent_ids
        }
        observations, _, terminated, _, _ = temp_env.step(actions)

        positions = {
            agent_id: tuple(temp_env.drones[agent_id].position) for agent_id in temp_env.agent_ids
        }
        for agent_id, path in trajectories.items():
            path.append(positions[agent_id])
        reported_edges = list(temp_env.reported_edges)
        snapshots.append(
            {
                "round": temp_env.round,
                "positions": positions,
                "reported_edges": reported_edges,
            }
        )

        if all(terminated.values()):
            break

    figures = [
        {
            "position": tuple(fig.position),
            "color": fig.color,
            "type": fig.figure_type,
        }
        for fig in temp_env.figures
    ]

    playback = EpisodePlayback(
        board_size=(temp_env.width, temp_env.height),
        trajectories=trajectories,
        figures=figures,
        reported_edges=reported_edges,
        score_summary=temp_env.get_score_summary(),
        snapshots=snapshots,
    )
    return playback


__all__ = [
    "PrioritizedReplayBuffer",
    "SharedDQNAgent",
    "train_marl",
    "evaluate_policy",
    "play_episode",
    "TrainingStats",
    "EvaluationResult",
    "EpisodePlayback",
]
