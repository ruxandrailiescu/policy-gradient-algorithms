import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch

from nhmrs.simple_assignment.simple_assignment import Scenario
from nhmrs import simple_assignment_v0
from ppo_torch import Agent


def plot_returns(history, out_path: str, window: int = 20):
    steps = np.array([s for s, _ in history])
    returns = np.array([r for _, r in history])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, returns, alpha=0.3, color="tab:blue", label="team return")
    if len(returns) >= window:
        smoothed = np.convolve(returns, np.ones(window) / window, mode="valid")
        ax.plot(steps[window - 1:], smoothed, color="tab:blue",
                label=f"moving avg ({window} ep)")
    ax.set_xlabel("timesteps")
    ax.set_ylabel("team return")
    ax.set_title("simple_assignment_v0")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    print(f"saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", type=int, default=1)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--n-landmarks", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--reward-mode", type=str, default="spread")
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--horizon", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    env = simple_assignment_v0.parallel_env(
        scenario=Scenario(reward_mode=args.reward_mode),
        n_agents=args.n_agents,
        n_landmarks=args.n_landmarks,
        max_steps=args.max_steps,
    )
    env.reset(seed=args.seed)
    agent_0 = env.possible_agents[0]
    state_dim = env.observation_space(agent_0).shape[0]
    action_dim = env.action_space(agent_0).shape[0]

    agent = Agent(
        state_dim,
        action_dim,
        n_agents=len(env.possible_agents),
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

    history = agent.ppo_parallel(env, total_steps=args.total_steps, seed=args.seed)
    env.close()

    if history:
        agent.save_checkpoint()
        plot_returns(history, f'results/assignment_returns_{args.exp_id}')
    else:
        print("no completed episodes recorded; nothing to plot")


if __name__ == "__main__":
    main()
