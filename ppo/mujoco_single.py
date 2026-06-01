import argparse
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from ppo_torch import Agent


def make_env(env_id: str) -> gym.Env:
    env = gym.make(env_id)
    env = gym.wrappers.NormalizeObservation(env)
    return env


def plot_returns(history, out_path: str, window: int = 20):
    steps = np.array([s for s, _ in history])
    returns = np.array([r for _, r in history])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, returns, alpha=0.3, color="tab:blue", label="episode return")
    if len(returns) >= window:
        smoothed = np.convolve(returns, np.ones(window) / window, mode="valid")
        ax.plot(steps[window - 1:], smoothed, color="tab:blue",
                label=f"moving avg ({window} ep)")
    ax.set_xlabel("timesteps")
    ax.set_ylabel("episode return")
    ax.set_title("HalfCheetah-v5")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    print(f"saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="HalfCheetah-v5")
    parser.add_argument("--exp-id", type=int, default=1)
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--horizon", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(args.seed)    # seed before the linear layers are initialized

    env = make_env(args.env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = Agent(
        state_dim,
        action_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        horizon=args.horizon,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        actor_model=f'actor_model_{args.exp_id}',
        actor_optim=f'actor_optim_{args.exp_id}',
        critic_model=f'critic_model_{args.exp_id}',
        critic_optim=f'critic_optim_{args.exp_id}'
    )

    history = agent.ppo(env, total_steps=args.total_steps, seed=args.seed)
    env.close()

    if history:
        agent.save_checkpoint()
        plot_returns(history, f'results/halfcheetah_returns_{args.exp_id}')
    else:
        print("no completed episodes recorded; nothing to plot")


if __name__ == "__main__":
    main()
