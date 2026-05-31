from typing import Tuple

import numpy as np
import os
import torch
import torch.nn as nn


class PPOMemory(object):
    def __init__(self, capacity: int, batch_size: int, state_dim: int, action_dim: int,
                 device: torch.device="cpu"):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.idx = 0

        self.states = torch.zeros(capacity, state_dim, device=device)
        self.actions = torch.zeros(capacity, action_dim, device=device)
        self.rewards = torch.zeros(capacity, device=device)
        self.critic_values = torch.zeros(capacity, device=device)
        self.log_probs = torch.zeros(capacity, device=device)
        self.dones = torch.zeros(capacity, device=device)

    def generate_batches(self, rng: np.random.Generator):
        indices = np.arange(self.idx)
        rng.shuffle(indices)
        batch_start = np.arange(0, self.idx, self.batch_size)
        return [indices[i:i + self.batch_size] for i in batch_start]
    
    def store_memory(self, state, action, reward, critic_value, log_prob, done):
        assert self.idx < self.capacity, "buffer full, call clear_memory"
        self.states[self.idx] = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        self.actions[self.idx] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[self.idx] = float(reward)
        self.critic_values[self.idx] = float(critic_value)
        self.log_probs[self.idx] = float(log_prob)
        self.dones[self.idx] = float(done)
        self.idx += 1

    def clear_memory(self):
        self.idx = 0


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, lr: float, fc1_dim: int, fc2_dim: int, 
                 model_ckpt_file, optim_ckpt_file, ckpt_dir='ckpt'):
        super().__init__()
        self.checkpoint_actor = os.path.join(ckpt_dir, model_ckpt_file)
        self.checkpoint_optim = os.path.join(ckpt_dir, optim_ckpt_file)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, fc1_dim),
            nn.ReLU(),
            nn.Linear(fc1_dim, fc2_dim),
            nn.ReLU(),
            nn.Linear(fc2_dim, action_dim)
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def _distribution(self, state: torch.Tensor) -> torch.distributions.Normal:
        action_mean = self.actor(state)  # (b, action_dim)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        return torch.distributions.Normal(action_mean, action_std)
    
    def act(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self._distribution(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self._distribution(states)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_probs, entropy

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_actor)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optim)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_actor))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optim))


class Critic(nn.Module):
    def __init__(self, state_dim: int, lr: float, fc1_dim: int, fc2_dim: int, 
                 model_ckpt_file, optim_ckpt_file, ckpt_dir='ckpt'):
        super().__init__()
        self.checkpoint_critic = os.path.join(ckpt_dir, model_ckpt_file)
        self.checkpoint_optim = os.path.join(ckpt_dir, optim_ckpt_file)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, fc1_dim),
            nn.ReLU(),
            nn.Linear(fc1_dim, fc2_dim),
            nn.ReLU(),
            nn.Linear(fc2_dim, 1)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_critic)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optim)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_critic))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optim))


class Agent(object):
    def __init__(self, state_dim, action_dim, gamma=0.99, gae_lambda=0.95, 
                 actor_lr=1e-3, critic_lr=1e-3, policy_clip=0.2,
                 fc1_dim=64, fc2_dim=64,
                 batch_size=64, horizon=2048, n_epochs=10,
                 entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5,
                 actor_model='actor_model_', actor_optim='actor_optim_', 
                 critic_model='critic_model_', critic_optim='critic_optim_'):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.batch_size = batch_size
        self.horizon = horizon
        self.n_epochs = n_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim, actor_lr, fc1_dim, fc2_dim, actor_model, actor_optim)
        self.critic = Critic(state_dim, critic_lr, fc1_dim, fc2_dim, critic_model, critic_optim)
        self.memory = PPOMemory(horizon, batch_size, state_dim, action_dim, device=self.device)
        self.np_rng = np.random.default_rng()
    
    def remember(self, state, action, reward, critic_value, log_prob, done):
        self.memory.store_memory(state, action, reward, critic_value, log_prob, done)
    
    def save_checkpoint(self):
        print("saving models...")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def _to_tensor(self, obs) -> torch.Tensor:
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
    
    @staticmethod
    def _clip_action(env, action: np.ndarray) -> np.ndarray:
        space = getattr(env, "action_space", None)
        if space is not None and hasattr(space, "low") and hasattr(space, "high"):
            return np.clip(action, space.low, space.high)
        return action
    
    def _compute_gae(self, last_value: torch.Tensor):
        T = self.memory.idx
        rewards = self.memory.rewards[:T]
        values = self.memory.critic_values[:T]
        dones = self.memory.dones[:T]

        advantages = torch.zeros(T, device=self.device)
        last_gae = 0.0
        for t in reversed(range(T)):
            if t == T-1:
                next_value = last_value
            else:
                next_value = values[t+1]    # a mid-rollout truncation (term=False) 
                # bootstraps to the next episode's first state value; see how it affects performance
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        
        returns = advantages + values
        return advantages, returns

    def learn(self, last_value: torch.Tensor):
        T = self.memory.idx
        states = self.memory.states[:T]
        actions = self.memory.actions[:T]
        old_log_probs = self.memory.log_probs[:T]
        old_values = self.memory.critic_values[:T]
        advantages, returns = self._compute_gae(last_value)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)   # normalization per rollout

        for _ in range(self.n_epochs):
            for batch in self.memory.generate_batches(self.np_rng):
                b_states = states[batch]
                b_actions = actions[batch]
                b_old_log_probs = old_log_probs[batch]
                b_old_values = old_values[batch]
                b_advantages = advantages[batch]
                b_returns = returns[batch]

                new_log_probs, entropy = self.actor.evaluate_actions(b_states, b_actions)
                new_values = self.critic(b_states).squeeze(-1)

                ratio = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip) * b_advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                value_loss = (new_values - b_returns).pow(2).mean()
                entropy_loss = entropy.mean()
                loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor.optimizer.step()
                self.critic.optimizer.step()

    def ppo(self, env, total_steps, seed):
        self.np_rng = np.random.default_rng(seed)

        def next_seed():
            return int(self.np_rng.integers(0, 2**31 - 1))
        
        history = []
        obs, _ = env.reset(seed=next_seed())
        ep_return = 0.0
        step = 0

        while step < total_steps:
            self.memory.clear_memory()      # once per rollout
            for _ in range(self.horizon):
                obs_t = self._to_tensor(obs)
                with torch.no_grad():       # skip calculating gradients
                    action, log_prob = self.actor.act(obs_t)
                    value = self.critic(obs_t)

                action_env = self._clip_action(env, action.squeeze(0).cpu().numpy())
                next_obs, reward, term, trunc, _ = env.step(action_env)

                self.memory.store_memory(obs_t.squeeze(0), action.squeeze(0), reward, value, log_prob, term)
                obs = next_obs
                ep_return += reward
                step += 1

                if term or trunc:
                    history.append((step, ep_return))   
                    ep_return = 0.0
                    obs, _ = env.reset(seed=next_seed())    # reset inside the rollout

            with torch.no_grad():
                last_value = self.critic(self._to_tensor(obs)).squeeze()
            self.learn(last_value)

            if history:
                recent = [r for _, r in history[-10:]]
                print(f"step {step:>8} | episodes {len(history):>5} | "
                      f"mean return (last {len(recent)}): {np.mean(recent):.2f}")
        return history