"""Metrics, training diagnostics, and seed-aggregation plotting for IPPO.

Split into three groups:
  - greedy_eval: deterministic-mean rollout producing task metrics
    (team reward, landmark coverage, collision rate, mean final distance),
    averaged over several episodes for low-variance curves.
  - diagnostics utilities: per-update measures logged to tensorboard
    (gradient norms, policy parameter-/output-space update size, critic
    output-space update size).
  - aggregate_and_plot: load per-seed metric files and plot mean +- std bands.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt


# =============================================================================
# Greedy evaluation (task metrics)
# =============================================================================

@torch.no_grad()
def greedy_eval(agent, env_factory, adapters, args, device, agent_order,
                action_low, action_high, n_episodes=5, base_seed=10_000):
    """Run ``n_episodes`` greedy (deterministic-mean) episodes and average metrics.

    ``env_factory`` builds a fresh env; ``adapters`` is
    ``(world_obs_array, array_to_action_dict, rewards_to_array)`` from the
    trainer. Returns per-episode means of:
      episodic_return (mean per-agent), coverage (fraction of landmarks),
      collision_rate, mean_final_dist, distinct_success.
    """
    world_obs_array, array_to_action_dict, rewards_to_array = adapters
    N = args.n_agents
    collision_threshold = 2.0 * 0.075  # 2 * agent_radius used by SimpleSpreadReward

    rets, covs, colls, dists, distincts = [], [], [], [], []
    for ep in range(n_episodes):
        env = env_factory()
        env.reset(seed=args.seed + base_seed + ep)
        ep_return = 0.0
        collision_steps = 0

        for _ in range(args.max_steps):
            obs = torch.tensor(world_obs_array(env, agent_order), dtype=torch.float32, device=device)
            mean = agent.actor_mean(obs).cpu().numpy()
            mean = np.clip(mean, action_low, action_high)
            _, reward, _, truncated, _ = env.step(array_to_action_dict(mean, agent_order))
            ep_return += float(np.mean(rewards_to_array(reward, agent_order)))

            pos = np.array([ag.state.p_pos for ag in env.world.agents])
            d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
            np.fill_diagonal(d, np.inf)
            if (d < collision_threshold).any():
                collision_steps += 1

            if all(truncated.values()):
                break

        agent_pos = np.array([ag.state.p_pos for ag in env.world.agents])
        lm_pos = np.array([lm.state.p_pos for lm in env.world.landmarks])
        n_landmarks = lm_pos.shape[0]
        d_al = np.linalg.norm(agent_pos[:, None, :] - lm_pos[None, :, :], axis=-1)  # (N, M)
        mean_final_dist = float(d_al.min(axis=1).mean())
        covered = int((d_al.min(axis=0) < args.eval_radius).sum())
        nearest = d_al.argmin(axis=1)
        distinct = len(set(nearest.tolist())) == N and bool((d_al.min(axis=1) < args.eval_radius).all())
        env.close()

        rets.append(ep_return)
        covs.append(covered / n_landmarks)
        colls.append(collision_steps / args.max_steps)
        dists.append(mean_final_dist)
        distincts.append(float(distinct))

    return {
        "episodic_return": float(np.mean(rets)),
        "coverage": float(np.mean(covs)),
        "collision_rate": float(np.mean(colls)),
        "mean_final_dist": float(np.mean(dists)),
        "distinct_success": float(np.mean(distincts)),
    }


# =============================================================================
# Training diagnostics
# =============================================================================

def flat_params(params) -> torch.Tensor:
    """Concatenate a parameter group into one detached vector."""
    return torch.cat([p.detach().reshape(-1) for p in params])


def grad_norm(params) -> float:
    """L2 norm of the gradients across a parameter group (0 if no grads)."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return total ** 0.5


def gaussian_kl(mean0, logstd0, mean1, logstd1) -> torch.Tensor:
    """KL( N(mean0,std0) || N(mean1,std1) ) summed over the action dimension."""
    var0 = torch.exp(2.0 * logstd0)
    var1 = torch.exp(2.0 * logstd1)
    return ((logstd1 - logstd0) + (var0 + (mean0 - mean1) ** 2) / (2.0 * var1) - 0.5).sum(-1)


# =============================================================================
# Seed aggregation + plotting
# =============================================================================

def save_seed_metrics(config_dir, seed, steps, series: dict, extra: dict | None = None):
    """Persist one seed's curves to ``config_dir/seed_<seed>.npz``.

    ``steps``/``series`` are the fixed-grid greedy-eval curves (equal length);
    ``extra`` holds additional arrays (e.g. the per-iteration training-return
    curve on its own grid) saved verbatim.
    """
    os.makedirs(config_dir, exist_ok=True)
    arrays = {"steps": np.asarray(steps)}
    arrays.update({k: np.asarray(v) for k, v in series.items()})
    if extra:
        arrays.update({k: np.asarray(v) for k, v in extra.items()})
    np.savez(os.path.join(config_dir, f"seed_{seed}.npz"), **arrays)


def aggregate_and_plot(config_dir, out_dir=None):
    """Load all ``seed_*.npz`` in ``config_dir`` and plot mean +- std bands.

    Produces one PNG per metric. Convergence value (mean of last 10% of points)
    and max are annotated on the team-reward plot.
    """
    out_dir = out_dir or config_dir
    files = sorted(glob.glob(os.path.join(config_dir, "seed_*.npz")))
    if not files:
        print(f"no seed_*.npz files found in {config_dir}")
        return
    data = [np.load(f) for f in files]
    n_seeds = len(data)
    L = min(len(d["steps"]) for d in data)          # align to shortest grid
    steps = data[0]["steps"][:L]

    specs = [
        ("team_return", "Team reward / episode (mean over agents)", "team_return.png", True),
        ("coverage", "Landmark coverage (fraction)", "coverage.png", False),
        ("collision_rate", "Collision rate", "collision_rate.png", False),
        ("mean_final_dist", "Mean final distance to landmark", "mean_final_dist.png", False),
    ]
    for key, ylabel, fname, annotate in specs:
        arr = np.stack([d[key][:L] for d in data])   # (n_seeds, L)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)

        plt.figure(figsize=(7, 4.5))
        for s in range(n_seeds):
            plt.plot(steps, arr[s], color="tab:blue", alpha=0.12)
        plt.plot(steps, mean, color="tab:blue", lw=2, label=f"mean ({n_seeds} seeds)")
        plt.fill_between(steps, mean - std, mean + std, color="tab:blue", alpha=0.2, label="±1 std")

        if annotate:
            k = max(1, L // 10)
            conv = float(mean[-k:].mean())
            mx = float(mean.max())
            mx_step = int(steps[int(mean.argmax())])
            plt.axhline(conv, color="tab:green", ls="--", lw=1, label=f"convergence ≈ {conv:.2f}")
            plt.scatter([mx_step], [mx], color="tab:red", zorder=5, label=f"max ≈ {mx:.2f}")

        plt.xlabel("Timesteps")
        plt.ylabel(ylabel)
        plt.title(f"IPPO {os.path.basename(os.path.normpath(config_dir))} (n={n_seeds})")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=300)
        plt.close()

    plot_train_returns(config_dir, out_dir, data)
    print(f"saved aggregate plots to {out_dir} ({n_seeds} seeds)")


def plot_train_returns(config_dir, out_dir=None, data=None):
    """Aggregate the per-iteration training-return curves across seeds.

    Episodes are fixed-length (num_steps == max_steps), so every seed records
    returns on the same iteration grid -> stack directly (no interpolation).
    """
    out_dir = out_dir or config_dir
    if data is None:
        files = sorted(glob.glob(os.path.join(config_dir, "seed_*.npz")))
        data = [np.load(f) for f in files]
    if not data or "train_return" not in data[0]:
        return
    n_seeds = len(data)
    L = min(len(d["train_return"]) for d in data)
    steps = data[0]["train_steps"][:L]
    arr = np.stack([d["train_return"][:L] for d in data])   # (n_seeds, L)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)

    plt.figure(figsize=(7, 4.5))
    for s in range(n_seeds):
        plt.plot(steps, arr[s], color="tab:blue", alpha=0.12)
    plt.plot(steps, mean, color="tab:blue", lw=2, label=f"mean ({n_seeds} seeds)")
    plt.fill_between(steps, mean - std, mean + std, color="tab:blue", alpha=0.2, label="±1 std")
    plt.xlabel("Timesteps")
    plt.ylabel("Episodic return (mean per-agent)")
    plt.title(f"IPPO {os.path.basename(os.path.normpath(config_dir))} (n={n_seeds})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "episode_returns.png"), dpi=300)
    plt.close()


def _load_config_metric(config_dir, key, steps_key):
    """Stack one metric across a config's seeds -> (steps, mean, std, n_seeds)."""
    files = sorted(glob.glob(os.path.join(config_dir, "seed_*.npz")))
    if not files:
        return None
    data = [np.load(f) for f in files]
    if key not in data[0] or steps_key not in data[0]:
        return None
    L = min(len(d[key]) for d in data)
    steps = data[0][steps_key][:L]
    arr = np.stack([d[key][:L] for d in data])
    return steps, arr.mean(axis=0), arr.std(axis=0), len(data)


def compare_configs(config_dirs, out_dir, labels=None):
    """Overlay several configs (each a seed-aggregated mean +- std band) per metric.

    Writes one ``compare_<metric>.png`` per metric to ``out_dir``.
    """
    os.makedirs(out_dir, exist_ok=True)
    labels = labels or [os.path.basename(os.path.normpath(c)) for c in config_dirs]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    specs = [
        ("team_return", "steps", "Team reward / episode (mean over agents)", "compare_team_return.png"),
        ("coverage", "steps", "Landmark coverage (fraction)", "compare_coverage.png"),
        ("collision_rate", "steps", "Collision rate", "compare_collision_rate.png"),
        ("mean_final_dist", "steps", "Mean final distance to landmark", "compare_mean_final_dist.png"),
        ("train_return", "train_steps", "Episodic return (mean per-agent)", "compare_episode_returns.png"),
    ]
    for key, steps_key, ylabel, fname in specs:
        plt.figure(figsize=(7, 4.5))
        plotted = False
        for i, cd in enumerate(config_dirs):
            res = _load_config_metric(cd, key, steps_key)
            if res is None:
                continue
            steps, mean, std, n = res
            c = colors[i % len(colors)]
            plt.plot(steps, mean, color=c, lw=2, label=f"{labels[i]} (n={n})")
            plt.fill_between(steps, mean - std, mean + std, color=c, alpha=0.2)
            plotted = True
        if not plotted:
            plt.close()
            continue
        plt.xlabel("Timesteps")
        plt.ylabel(ylabel)
        plt.title("IPPO config comparison")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=300)
        plt.close()
    print(f"saved comparison plots to {out_dir}")
