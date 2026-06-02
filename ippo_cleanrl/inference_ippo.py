"""Minimal inference / visualization for an IPPO policy on simple_assignment_v0.

IPPO trains a single shared actor-critic (parameter sharing across the N
homogeneous agents), saved by ``ippo_simple_assignment.py`` as
``runs/<run_name>/agent_ippo.pt``. At inference only the actor is needed, and we
act greedily by taking the actor mean (no sampling), clipped to the action
bounds -- exactly the greedy rollout used during eval.

The env MUST be built the same way as training (``observe_relative=True`` and the
same agent/landmark counts) so the observation layout matches the trained policy.

Usage:
    python inference_ippo.py runs/<run_name>/agent_ippo.pt
    python inference_ippo.py <ckpt> --render-mode rgb_array --save-video out.mp4
    python inference_ippo.py <ckpt> --render-mode none        # no window (debug)
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from nhmrs import simple_assignment_v0
from nhmrs.simple_assignment.simple_assignment import Scenario

# reuse the exact network + dict<->array adapters from training (no drift)
from ippo_cleanrl.ippo_simple_assignment import Agent, world_obs_array, array_to_action_dict, rewards_to_array


def parse_args():
    p = argparse.ArgumentParser(description="Visualize a trained IPPO policy")
    p.add_argument("checkpoint", help="path to agent_ippo.pt")
    p.add_argument("--reward-mode", default="spread")
    p.add_argument("--n-agents", type=int, default=3)
    p.add_argument("--n-landmarks", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--render-mode", default="human", choices=["human", "rgb_array", "none"])
    p.add_argument("--save-video", default=None, help="path for rgb_array frames (needs imageio)")
    return p.parse_args()


def main():
    args = parse_args()
    render_mode = None if args.render_mode == "none" else args.render_mode

    # build env identically to training
    scenario = Scenario(reward_mode=args.reward_mode, observe_relative=True)
    env = simple_assignment_v0.env(
        scenario=scenario,
        render_mode=render_mode,
        max_steps=args.max_steps,
        n_agents=args.n_agents,
        n_landmarks=args.n_landmarks,
    )
    env.reset(seed=args.seed)

    agent_order = env.possible_agents[:]
    obs_dim = env.observation_space(agent_order[0]).shape[0]
    action_dim = env.action_space(agent_order[0]).shape[0]
    a_low = env.action_space(agent_order[0]).low
    a_high = env.action_space(agent_order[0]).high

    # load the shared actor-critic, eval mode, greedy (mean) actions
    device = torch.device("cpu")
    policy = Agent(obs_dim, action_dim).to(device)
    policy.load_state_dict(torch.load(args.checkpoint, map_location=device))
    policy.eval()
    print(f"loaded {args.checkpoint} | agents={args.n_agents} obs_dim={obs_dim} action_dim={action_dim}")

    frames = []
    for ep in range(args.episodes):
        env.reset()
        ep_return = 0.0
        for _ in range(args.max_steps):
            obs = torch.tensor(world_obs_array(env, agent_order), dtype=torch.float32, device=device)
            with torch.no_grad():
                mean = policy.actor_mean(obs).cpu().numpy()
            action = np.clip(mean, a_low, a_high)
            _, reward, _, truncated, _ = env.step(array_to_action_dict(action, agent_order))
            ep_return += float(np.mean(rewards_to_array(reward, agent_order)))

            if render_mode is not None:
                frame = env.render()
                if render_mode == "rgb_array" and args.save_video:
                    frames.append(frame)
                if render_mode == "human" and args.fps:
                    time.sleep(1.0 / args.fps)

            if all(truncated.values()):
                break
        print(f"episode {ep + 1:02d}/{args.episodes} | return {ep_return:8.2f}")

    env.close()

    if args.save_video and frames:
        import imageio
        imageio.mimsave(args.save_video, frames, fps=args.fps)
        print(f"saved video to {args.save_video} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
