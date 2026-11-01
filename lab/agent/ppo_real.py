"""Model-free PPO in the real game as a sample-efficiency baseline (PokeDreamer).

Rollouts with PyBoy workers in the real game, clipped PPO on CPU.
Mirrors the observation, action space, network, and eval cap of the
imagination-trained baseline in lab/agent/imagine_rl.py and lab/agent/mpc.py.

Reward structure:
Progress on the same map mirrors imagine_rl: prev_dist - dist - STEP_COST.
To guard against post-warp RAM coordinate lag (where PyBoy updates map_id first and
only updates x, y 1-2 steps later, creating an artificial multi-tile teleport jump),
same-map distance progress is only applied when |dx| + |dy| <= 2; larger jumps receive
-STEP_COST only.

For map transitions, real-game rollouts continue through multiple maps (unlike
imagination rollouts which terminate at the first map change). To prevent infinite
reward loops where an agent steps in and out of side buildings (e.g. entering Oak's
lab costs -2, stepping back out is distance <= 1 from exit and earns +5, netting +3
indefinitely), we track an ordered task route of maps. +WARP_BONUS is only awarded
when transitioning to a new map on the route with index greater than the best reached
so far in the episode; entering side buildings, backtracking, or exiting side buildings
earns -HALLUCINATION_PENALTY.
"""
from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from lab.agent.imagine_rl import (
    HALLUCINATION_PENALTY,
    STEP_COST,
    WARP_BONUS,
    Policy,
)
from lab.agent.mpc import DIRECTIONS, A, B, run_episode
from lab.agent.tasks import TASKS, exit_distance, train_geodesic_fields
from lab.dwm.model import CTX, to_float
from lab.env import ROM_PATH, PokemonRed

TASK_CAPS = {
    "route1": 100,
    "leave_house": 60,
    "viridian": 150,
}

TASK_ROUTES = {
    "leave_house": [38, 37, 0],
    "route1": [0, 12],
    "viridian": [0, 12, 1],
}

PAUSE = Path("work/PAUSE")


def compute_reward(
    prev_ram: dict[str, Any],
    cur_ram: dict[str, Any],
    exits: dict[int, list],
    route: Optional[List[int]] = None,
    best_route_idx: int = 0,
    fields: Optional[dict] = None,
) -> Tuple[float, int]:
    """Compute step reward with route-order warp bonus and post-warp lag filter.

    Note: This deliberately differs from the imagination reward in lab/agent/imagine_rl.py.
    In imagination, rollouts are short (H=10) and end immediately upon the first map change,
    so an agent cannot cycle across maps. In real-game rollouts (which continue across maps
    until the goal or truncation cap), a naive warp bonus on exiting any building (e.g.,
    stepping out of Oak's lab door where distance <= 1) creates an infinite reward loop:
    entering costs -2, exiting earns +5, netting +3 every two presses.

    Furthermore, Game Boy RAM updates coordinates 1-2 steps after map_id during warp
    fade-ins, creating false 7-tile jumps if unmitigated. We filter same-map steps to
    |dx| + |dy| <= 2; larger position changes receive -STEP_COST only.

    On a map change:
    - If the new map is on the task route with index > best_route_idx (and prev_d <= 1.0),
      award +WARP_BONUS and advance best_route_idx.
    - Otherwise (entering a detour building, backtracking to an earlier map, or exiting a
      detour back to an already-visited map index), penalize with -HALLUCINATION_PENALTY.
    """
    prev_map = prev_ram["map_id"]
    cur_map = cur_ram["map_id"]
    prev_x, prev_y = prev_ram["x"], prev_ram["y"]
    cur_x, cur_y = cur_ram["x"], cur_ram["y"]

    if cur_map == prev_map:
        dx = abs(cur_x - prev_x)
        dy = abs(cur_y - prev_y)
        if dx + dy <= 2:
            prev_d = float(exit_distance([prev_map], [prev_x], [prev_y], exits, fields)[0])
            cur_d = float(exit_distance([cur_map], [cur_x], [cur_y], exits, fields)[0])
            return float(prev_d - cur_d - STEP_COST), best_route_idx
        else:
            # Post-warp coordinate update glitch: give -STEP_COST only
            return float(-STEP_COST), best_route_idx
    else:
        # Map change: check if forward progress along route
        prev_d = float(exit_distance([prev_map], [prev_x], [prev_y], exits, fields)[0])
        if route is not None and cur_map in route:
            new_idx = route.index(cur_map)
            if new_idx > best_route_idx and prev_d <= 1.0:
                return float(WARP_BONUS), new_idx
        return float(-HALLUCINATION_PENALTY), best_route_idx


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float = 0.95,
    lam: float = 0.9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation (GAE) and lambda-returns.

    Handles termination (next value is 0) and truncation (bootstrapped from next_values).
    Works on 1D (T,) or 2D (T, B) tensors.
    """
    t_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    dones = (terminated | truncated).float()

    if rewards.ndim == 1:
        last_adv = torch.tensor(0.0, dtype=rewards.dtype, device=rewards.device)
    else:
        last_adv = torch.zeros(rewards.shape[1:], dtype=rewards.dtype, device=rewards.device)

    for t in reversed(range(t_steps)):
        delta = rewards[t] + gamma * next_values[t] - values[t]
        advantages[t] = delta + gamma * lam * (1.0 - dones[t]) * last_adv
        last_adv = advantages[t]

    returns = advantages + values
    return advantages, returns


class RealEnvWrapper:
    """Headless Pokemon Red environment wrapper for PPO worker.

    - Loads task state, executes 4 B presses on reset to fill context.
    - Scripted battle reflex: presses A while in_battle (counted in env steps).
    - Checks success_map termination and eval cap truncation.
    - Tracks route progression to award warp bonus only for forward route progress.
    - Auto-resets on termination or truncation, returning terminal observation in info.
    """

    def __init__(self, task: str, rom_path: str | Path = ROM_PATH, cap: Optional[int] = None, geodesic: bool = False):
        self.task = task
        self.spec = TASKS[task]
        self.exits = self.spec["exits"]
        self.success_map = self.spec["success_map"]
        self.route = TASK_ROUTES.get(task, [0, self.success_map])
        self.eval_cap = cap or TASK_CAPS.get(task, 100)
        self.fields = train_geodesic_fields(self.exits) if geodesic else None  # path distance (lab/agent/tasks.py)
        self.env = PokemonRed(rom=rom_path)
        self.state_bytes = Path(self.spec["state"]).read_bytes()
        self.frames: List[np.ndarray] = []
        self.prev_ram: Optional[dict[str, Any]] = None
        self.best_route_idx = 0
        self.episode_presses = 0

    def reset(self) -> np.ndarray:
        self.env.restore(self.state_bytes)
        self.frames = []
        # Context initialization: 4 B presses
        for _ in range(CTX):
            frame, ram = self.env.step(B)
            self.frames.append(frame)
        while ram["in_battle"]:
            frame, ram = self.env.step(A)
            self.frames.append(frame)
            self.frames = self.frames[-CTX:]
        self.prev_ram = ram
        self.episode_presses = 0
        start_map = ram["map_id"]
        self.best_route_idx = self.route.index(start_map) if start_map in self.route else 0
        return np.stack(self.frames[-CTX:], axis=0)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict[str, Any], int]:
        # 1. Step the policy action
        frame, ram = self.env.step(action)
        self.frames.append(frame)
        self.frames = self.frames[-CTX:]
        env_presses = 1

        # 2. Battle reflex: while in_battle, press A itself
        while ram["in_battle"]:
            frame, ram = self.env.step(A)
            self.frames.append(frame)
            self.frames = self.frames[-CTX:]
            env_presses += 1

        # 3. Reward calculation with route order check and position lag filter
        reward, self.best_route_idx = compute_reward(
            self.prev_ram, ram, self.exits, self.route, self.best_route_idx, self.fields
        )
        self.prev_ram = ram
        self.episode_presses += env_presses

        # 4. Check termination and truncation
        terminated = (ram["map_id"] == self.success_map)
        truncated = (not terminated) and (self.episode_presses >= self.eval_cap)

        if terminated or truncated:
            terminal_obs = np.stack(self.frames[-CTX:], axis=0)
            info = {
                "terminal_obs": terminal_obs,
                "success": terminated,
                "presses": self.episode_presses,
                "final_map": ram["map_id"],
                "final_x": ram["x"],
                "final_y": ram["y"],
            }
            # Auto-reset
            next_obs = self.reset()
            return next_obs, reward, terminated, truncated, info, env_presses
        else:
            obs = np.stack(self.frames[-CTX:], axis=0)
            return obs, reward, False, False, {}, env_presses

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None


def worker_fn(pipe: mp.connection.Connection, task_name: str, rom_path: str | Path, seed: int,
              cap: Optional[int] = None, geodesic: bool = False) -> None:
    """Multiprocessing worker function running a single PyBoy instance."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)

    wrapper = RealEnvWrapper(task=task_name, rom_path=rom_path, cap=cap, geodesic=geodesic)
    try:
        while True:
            if not pipe.poll(timeout=180.0):
                break
            cmd, data = pipe.recv()
            if cmd == "step":
                res = wrapper.step(data)
                pipe.send(res)
            elif cmd == "reset":
                obs = wrapper.reset()
                pipe.send(obs)
            elif cmd == "close":
                break
    except Exception as e:
        pipe.send(("error", str(e)))
    finally:
        wrapper.close()


class VectorEnv:
    """Vectorized environment manager using multiprocessing spawn and Pipes."""

    def __init__(self, task: str, num_envs: int, rom_path: str | Path = ROM_PATH, base_seed: int = 0,
                 cap: Optional[int] = None, geodesic: bool = False):
        self.num_envs = num_envs
        self.pipes: List[mp.connection.Connection] = []
        self.procs: List[mp.Process] = []
        ctx = mp.get_context("spawn")
        for i in range(num_envs):
            p_conn, c_conn = ctx.Pipe()
            seed = base_seed + i * 1000
            proc = ctx.Process(target=worker_fn, args=(c_conn, task, rom_path, seed, cap, geodesic), daemon=True)
            proc.start()
            self.pipes.append(p_conn)
            self.procs.append(proc)

    def reset(self, timeout: float = 60.0) -> np.ndarray:
        for p in self.pipes:
            p.send(("reset", None))
        obs_list = []
        for p in self.pipes:
            if not p.poll(timeout):
                raise TimeoutError(f"Worker timed out during reset ({timeout}s)")
            res = p.recv()
            if isinstance(res, tuple) and len(res) == 2 and res[0] == "error":
                raise RuntimeError(f"Worker process error: {res[1]}")
            obs_list.append(res)
        return np.stack(obs_list, axis=0)

    def step(
        self, actions: np.ndarray, timeout: float = 60.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict[str, Any]], np.ndarray]:
        for p, a in zip(self.pipes, actions):
            p.send(("step", int(a)))
        obs_list, rewards, terms, truncs, infos, presses_list = [], [], [], [], [], []
        for p in self.pipes:
            if not p.poll(timeout):
                raise TimeoutError(f"Worker timed out during step ({timeout}s)")
            res = p.recv()
            if isinstance(res, tuple) and len(res) == 2 and res[0] == "error":
                raise RuntimeError(f"Worker process error: {res[1]}")
            obs, r, term, trunc, info, presses = res
            obs_list.append(obs)
            rewards.append(r)
            terms.append(term)
            truncs.append(trunc)
            infos.append(info)
            presses_list.append(presses)
        return (
            np.stack(obs_list, axis=0),
            np.array(rewards, dtype=np.float32),
            np.array(terms, dtype=bool),
            np.array(truncs, dtype=bool),
            infos,
            np.array(presses_list, dtype=np.int32),
        )

    def close(self) -> None:
        for p in self.pipes:
            try:
                p.send(("close", None))
            except Exception:
                pass
        for proc in self.procs:
            proc.join(timeout=3.0)
            if proc.is_alive():
                proc.terminate()


def evaluate_policy(
    policy: nn.Module,
    task: str,
    episodes: int = 30,
    seed: int = 1000,
    device: str = "cpu",
    cap: Optional[int] = None,
) -> dict[str, Any]:
    """Run real evaluation episodes using sampled policy actions."""
    policy.eval()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    max_steps = cap or TASK_CAPS.get(task, 100)
    results = []
    for _ in range(episodes):
        res, _ = run_episode(
            policy="learned",
            model=None,
            probe=None,
            rng=rng,
            max_steps=max_steps,
            n=0,
            h=0,
            device=device,
            record=False,
            net=policy,
            task=task,
        )
        results.append(res)
    policy.train()
    successes = sum(r["success"] for r in results)
    success_rate = successes / len(results)
    steps_success = [r["steps"] for r in results if r["success"]]
    median_presses = float(np.median(steps_success)) if steps_success else None
    return {
        "success_rate": success_rate,
        "median_presses": median_presses,
        "successes": successes,
        "episodes": len(results),
        "results": results,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(args.num_threads)
    dev = torch.device(args.device)

    out_dir = Path("work/ppo_real") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    agent_rl_dir = Path("work/agent_rl") / f"ppo_real_{args.name}"
    agent_rl_dir.mkdir(parents=True, exist_ok=True)

    logf = open(out_dir / "train.log", "a")
    eval_jsonl_path = out_dir / "eval.jsonl"

    def log(msg: str):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    policy = Policy().to(dev)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    ckpt_path = out_dir / "policy.pt"
    update = 0
    total_env_steps = 0
    total_episodes = 0
    elapsed = 0.0

    if ckpt_path.exists():
        st = torch.load(ckpt_path, map_location=dev, weights_only=True)
        policy.load_state_dict(st["policy"])
        optimizer.load_state_dict(st["opt"])
        update = st.get("updates", 0)
        total_env_steps = st.get("env_steps", 0)
        total_episodes = st.get("total_episodes", 0)
        elapsed = st.get("elapsed", 0.0)
        log(f"Resumed from {ckpt_path}: update {update}, {total_env_steps} env steps, {elapsed / 60:.1f} min")

    log(f"Starting training on {args.task} with {args.envs} envs, device {dev}")
    vec_env = VectorEnv(task=args.task, num_envs=args.envs, rom_path=args.rom, base_seed=args.seed,
                        cap=args.cap, geodesic=args.geodesic)
    current_obs = vec_env.reset()

    consecutive_95 = 0
    milestones = {"first_success": None, "first_50": None, "first_90": None, "first_95": None}
    next_eval_steps = ((total_env_steps // args.eval_every) + 1) * args.eval_every

    # Initial eval at step 0 if starting fresh
    if total_env_steps == 0:
        log("Running baseline evaluation at step 0...")
        eval_res0 = evaluate_policy(policy, task=args.task, episodes=args.eval_episodes, seed=args.eval_seed, device=args.device, cap=args.cap)
        eval_entry0 = {
            "env_steps": total_env_steps,
            "updates": update,
            "wall_time": round(elapsed, 2),
            "success_rate": round(eval_res0["success_rate"], 4),
            "median_presses": eval_res0["median_presses"],
            "successes": eval_res0["successes"],
            "episodes": eval_res0["episodes"],
            "total_episodes": total_episodes,
        }
        with open(eval_jsonl_path, "a") as f:
            f.write(json.dumps(eval_entry0) + "\n")
        log(f"Eval @ 0: success {eval_res0['success_rate']:.1%} ({eval_res0['successes']}/{eval_res0['episodes']}), median {eval_res0['median_presses']}")
        rate0 = eval_res0["success_rate"]
        if rate0 > 0.0 and milestones["first_success"] is None:
            milestones["first_success"] = {"steps": 0, "wall_time": 0.0, "episodes": 0, "rate": rate0}
        if rate0 >= 0.50 and milestones["first_50"] is None:
            milestones["first_50"] = {"steps": 0, "wall_time": 0.0, "episodes": 0}
        if rate0 >= 0.90 and milestones["first_90"] is None:
            milestones["first_90"] = {"steps": 0, "wall_time": 0.0, "episodes": 0}
        if rate0 >= 0.95 and milestones["first_95"] is None:
            milestones["first_95"] = {"steps": 0, "wall_time": 0.0, "episodes": 0}
        # Initial checkpoint
        tmp_ckpt = out_dir / "policy.tmp"
        torch.save({
            "policy": policy.state_dict(),
            "opt": optimizer.state_dict(),
            "env_steps": total_env_steps,
            "updates": update,
            "total_episodes": total_episodes,
            "elapsed": elapsed,
        }, tmp_ckpt)
        os.replace(tmp_ckpt, ckpt_path)
        shutil.copyfile(ckpt_path, agent_rl_dir / "policy.pt")

    t_start = time.time() - elapsed
    t_last = time.time()
    steps_at_last = total_env_steps

    stop_reason = ""
    try:
        while True:
            # Check pause
            if PAUSE.exists():
                log("Paused (work/PAUSE exists)")
                while PAUSE.exists():
                    time.sleep(5)
                log("Resumed from pause")

            # Check time / step limits
            now = time.time()
            elapsed = now - t_start
            if total_env_steps >= args.max_env_steps:
                stop_reason = f"Reached max_env_steps ({total_env_steps} >= {args.max_env_steps})"
                break
            if elapsed >= args.minutes * 60:
                stop_reason = f"Reached time limit ({elapsed / 60:.1f} >= {args.minutes} min)"
                break

            # Rollout buffer
            T = args.rollout_steps
            E = args.envs
            obs_buf = np.empty((T, E, CTX, 72, 80), dtype=np.uint8)
            act_buf = np.empty((T, E), dtype=np.int64)
            logp_buf = np.empty((T, E), dtype=np.float32)
            val_buf = np.empty((T, E), dtype=np.float32)
            rew_buf = np.empty((T, E), dtype=np.float32)
            term_buf = np.empty((T, E), dtype=bool)
            trunc_buf = np.empty((T, E), dtype=bool)
            next_val_buf = np.empty((T, E), dtype=np.float32)

            for t in range(T):
                obs_buf[t] = current_obs
                obs_tensor = to_float(torch.from_numpy(current_obs).to(dev))
                with torch.no_grad():
                    logits, v = policy(obs_tensor)
                    dist = torch.distributions.Categorical(logits=logits)
                    actions = dist.sample()
                    logprobs = dist.log_prob(actions)

                act_buf[t] = actions.cpu().numpy()
                logp_buf[t] = logprobs.cpu().numpy()
                val_buf[t] = v.cpu().numpy()

                next_obs, rewards, terms, truncs, infos, presses = vec_env.step(act_buf[t])

                rew_buf[t] = rewards
                term_buf[t] = terms
                trunc_buf[t] = truncs

                for e in range(E):
                    if terms[e] or truncs[e]:
                        total_episodes += 1
                    if truncs[e]:
                        term_obs = infos[e]["terminal_obs"]
                        with torch.no_grad():
                            _, term_v = policy(to_float(torch.from_numpy(term_obs)[None].to(dev)))
                            next_val_buf[t, e] = term_v.item()
                    elif terms[e]:
                        next_val_buf[t, e] = 0.0

                total_env_steps += int(presses.sum())
                current_obs = next_obs

            # Final bootstrap
            obs_tensor = to_float(torch.from_numpy(current_obs).to(dev))
            with torch.no_grad():
                _, last_val = policy(obs_tensor)
                last_val_np = last_val.cpu().numpy()

            for t in range(T - 1):
                not_done = ~(term_buf[t] | trunc_buf[t])
                next_val_buf[t, not_done] = val_buf[t + 1, not_done]

            not_done_last = ~(term_buf[T - 1] | trunc_buf[T - 1])
            next_val_buf[T - 1, not_done_last] = last_val_np[not_done_last]

            # GAE
            advantages, returns = compute_gae(
                torch.from_numpy(rew_buf).to(dev),
                torch.from_numpy(val_buf).to(dev),
                torch.from_numpy(term_buf).to(dev),
                torch.from_numpy(trunc_buf).to(dev),
                torch.from_numpy(next_val_buf).to(dev),
                gamma=args.gamma,
                lam=args.lam,
            )

            # Flatten batch
            batch_size = T * E
            b_obs = torch.from_numpy(obs_buf).flatten(0, 1)
            b_act = torch.from_numpy(act_buf).flatten(0, 1).to(dev)
            b_logp = torch.from_numpy(logp_buf).flatten(0, 1).to(dev)
            b_adv = advantages.flatten(0, 1)
            b_ret = returns.flatten(0, 1)

            # Advantage normalisation
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

            # PPO update
            minibatch_size = min(args.minibatch, batch_size)
            last_pg_loss, last_v_loss, last_ent = 0.0, 0.0, 0.0
            for epoch in range(args.epochs):
                perm = torch.randperm(batch_size)
                for start in range(0, batch_size, minibatch_size):
                    end = min(start + minibatch_size, batch_size)
                    mb_idx = perm[start:end]

                    mb_obs = to_float(b_obs[mb_idx].to(dev))
                    logits, v = policy(mb_obs)
                    dist = torch.distributions.Categorical(logits=logits)
                    new_logp = dist.log_prob(b_act[mb_idx])
                    entropy = dist.entropy().mean()

                    logratio = new_logp - b_logp[mb_idx]
                    ratio = torch.exp(logratio)

                    mb_advantages = b_adv[mb_idx]
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * mb_advantages
                    pg_loss = -torch.min(surr1, surr2).mean()

                    v_loss = 0.5 * F.mse_loss(v, b_ret[mb_idx])

                    loss = pg_loss + 0.5 * v_loss - args.entropy_coef * entropy

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
                    optimizer.step()

                    last_pg_loss = pg_loss.item()
                    last_v_loss = v_loss.item()
                    last_ent = entropy.item()

            update += 1
            now = time.time()
            elapsed = now - t_start
            dt = now - t_last
            fps = (total_env_steps - steps_at_last) / max(dt, 1e-6)
            t_last = now
            steps_at_last = total_env_steps

            if update % 5 == 0 or total_env_steps >= next_eval_steps:
                mean_ret = float(rew_buf.sum(0).mean())
                log(
                    f"update {update:4d} env_steps {total_env_steps:7d} rew {mean_ret:6.2f} "
                    f"pg {last_pg_loss:6.3f} vl {last_v_loss:6.3f} ent {last_ent:5.3f} "
                    f"({fps:5.1f} steps/s, {elapsed / 60:4.1f} min)"
                )

            # Evaluation check
            if total_env_steps >= next_eval_steps:
                next_eval_steps += args.eval_every
                log(f"Evaluating at {total_env_steps} env steps (update {update})...")
                eval_res = evaluate_policy(
                    policy,
                    task=args.task,
                    episodes=args.eval_episodes,
                    seed=args.eval_seed,
                    device=args.device,
                    cap=args.cap,
                )
                rate = eval_res["success_rate"]
                entry = {
                    "env_steps": total_env_steps,
                    "updates": update,
                    "wall_time": round(elapsed, 2),
                    "success_rate": round(rate, 4),
                    "median_presses": eval_res["median_presses"],
                    "successes": eval_res["successes"],
                    "episodes": eval_res["episodes"],
                    "total_episodes": total_episodes,
                }
                with open(eval_jsonl_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")

                log(
                    f"EVAL at {total_env_steps} steps: success {rate:.1%} "
                    f"({eval_res['successes']}/{eval_res['episodes']}), median presses {eval_res['median_presses']}"
                )

                # Track milestones
                if rate > 0.0 and milestones["first_success"] is None:
                    milestones["first_success"] = {"steps": total_env_steps, "wall_time": elapsed, "episodes": total_episodes, "rate": rate}
                    log(f"First success reached: {rate:.1%} at {total_env_steps} steps!")
                if rate >= 0.50 and milestones["first_50"] is None:
                    milestones["first_50"] = {"steps": total_env_steps, "wall_time": elapsed, "episodes": total_episodes}
                if rate >= 0.90 and milestones["first_90"] is None:
                    milestones["first_90"] = {"steps": total_env_steps, "wall_time": elapsed, "episodes": total_episodes}
                if rate >= 0.95 and milestones["first_95"] is None:
                    milestones["first_95"] = {"steps": total_env_steps, "wall_time": elapsed, "episodes": total_episodes}

                # Checkpoint
                tmp_ckpt = out_dir / "policy.tmp"
                torch.save({
                    "policy": policy.state_dict(),
                    "opt": optimizer.state_dict(),
                    "env_steps": total_env_steps,
                    "updates": update,
                    "total_episodes": total_episodes,
                    "elapsed": elapsed,
                }, tmp_ckpt)
                os.replace(tmp_ckpt, ckpt_path)
                shutil.copyfile(ckpt_path, agent_rl_dir / "policy.pt")

                # Stopping check: 2 consecutive evals >= 95%
                if rate >= 0.95:
                    consecutive_95 += 1
                    if consecutive_95 >= 2:
                        stop_reason = f"Reached >= 95% success on 2 consecutive evals ({rate:.1%})"
                        break
                else:
                    consecutive_95 = 0

    finally:
        vec_env.close()

    now = time.time()
    elapsed = now - t_start
    log(f"Training stopped: {stop_reason}. Total env steps: {total_env_steps}, wall time: {elapsed / 60:.1f} min")
    log(f"Milestones: {json.dumps(milestones, indent=2)}")

    # Ensure final checkpoint is in place
    tmp_ckpt = out_dir / "policy.tmp"
    torch.save({
        "policy": policy.state_dict(),
        "opt": optimizer.state_dict(),
        "env_steps": total_env_steps,
        "updates": update,
        "total_episodes": total_episodes,
        "elapsed": elapsed,
    }, tmp_ckpt)
    os.replace(tmp_ckpt, ckpt_path)
    shutil.copyfile(ckpt_path, agent_rl_dir / "policy.pt")

    # Final pre-declared evaluation: 100 fresh episodes, seed 200 via lab.agent.mpc
    log("Running final pre-declared evaluation (100 episodes, seed 200)...")
    cap = args.cap or TASK_CAPS.get(args.task, 100)
    cmd = [
        sys.executable,
        "-m",
        "lab.agent.mpc",
        "--task",
        args.task,
        "--policy",
        "learned",
        "--rl-name",
        f"ppo_real_{args.name}",
        "--episodes",
        "100",
        "--max-steps",
        str(cap),
        "--seed",
        "200",
        "--device",
        "cpu",
    ]
    log(f"Running command: {' '.join(cmd)}")
    eval_proc = subprocess.run(cmd, capture_output=True, text=True)
    log("Final eval output:")
    for l in eval_proc.stdout.splitlines():
        log(f"  {l}")
    if eval_proc.returncode != 0:
        log(f"Final eval stderr: {eval_proc.stderr}")

    final_json_path = Path("work/agent") / f"{args.task}_learned_ppo_real_{args.name}_s200.json"
    final_stats = {}
    if final_json_path.exists():
        with open(final_json_path, "r") as f:
            final_stats = json.load(f)

    return {
        "task": args.task,
        "total_env_steps": total_env_steps,
        "updates": update,
        "elapsed_seconds": elapsed,
        "total_episodes": total_episodes,
        "milestones": milestones,
        "final_stats": final_stats,
        "eval_curve": str(eval_jsonl_path),
        "stop_reason": stop_reason,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS), default="route1")
    ap.add_argument("--envs", type=int, default=12)
    ap.add_argument("--max-env-steps", type=int, default=2000000)
    ap.add_argument("--minutes", type=float, default=120.0)
    ap.add_argument("--eval-every", type=int, default=25000)
    ap.add_argument("--eval-episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-seed", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--lam", type=float, default=0.9)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=0.5)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--rollout-steps", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num-threads", type=int, default=4)
    ap.add_argument("--rom", default=str(ROM_PATH))
    ap.add_argument("--name", default=None, help="run name (default: the task); work/ppo_real/<name>")
    ap.add_argument("--cap", type=int, default=None, help="presses per episode (default: TASK_CAPS)")
    ap.add_argument("--geodesic", action="store_true", help="path distance over the train split's moves")
    args = ap.parse_args()
    args.name = args.name or args.task

    assert args.envs <= 12, "Maximum 12 environment processes allowed"
    train(args)


if __name__ == "__main__":
    main()
