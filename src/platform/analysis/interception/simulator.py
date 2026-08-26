"""单次围捕仿真主循环与结果记录。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import InterceptionConfig
from .guidance import choose_interceptor, evader_force, pursuer_force, EPS


@dataclass
class SimResult:
    captured: bool
    capture_time: Optional[float]
    capture_step: Optional[int]
    best_second_distance: float
    arrival_count: int
    arrival_variance: Optional[float]
    realized_path_ratio: Optional[float]
    min_sep: float
    min_sep_step: Optional[int]
    final_positions: np.ndarray
    evader_traj: np.ndarray
    pursuer_dist_history: np.ndarray


def _initial_positions(cfg: InterceptionConfig, rng: np.random.Generator) -> np.ndarray:
    """按正对/侧向散布生成初始拦截机位置（二维）。"""
    positions = np.zeros((2, cfg.n_pursuers))
    for i in range(cfg.n_pursuers):
        angle = 2.0 * np.pi * i / cfg.n_pursuers + rng.uniform(-0.25, 0.25)
        radius = cfg.init_scale * (0.75 + 0.5 * rng.random())
        positions[0, i] = cfg.evader_init[0] + radius * np.cos(angle)
        positions[1, i] = cfg.evader_init[1] + radius * np.sin(angle)
    return positions


def simulate_trial(cfg: InterceptionConfig, seed: Optional[int] = None) -> SimResult:
    """运行一次协同围捕仿真，返回结果对象。"""
    cfg.validate()
    rng = np.random.default_rng(seed if seed is not None else cfg.seed)
    n = cfg.n_pursuers + 1
    pursuers = list(range(cfg.n_pursuers))
    evader = cfg.n_pursuers

    positions = np.zeros((2, n))
    positions[:, : cfg.n_pursuers] = _initial_positions(cfg, rng)
    positions[:, evader] = cfg.evader_init

    heading = rng.uniform(-np.pi, np.pi)
    evader_vel = cfg.v_evader * np.array([np.cos(heading), np.sin(heading)])
    pursuer_vel_state = np.zeros((2, cfg.n_pursuers))

    captured = False
    capture_step = None
    best_second = np.inf
    has_arrived = np.zeros(cfg.n_pursuers, dtype=bool)
    arrival_time = np.full(cfg.n_pursuers, np.nan)
    evader_path = 0.0
    pursuer_path = np.zeros(cfg.n_pursuers)
    prev_positions = positions.copy()
    prev_evader = positions[:, evader].copy()
    min_sep = np.inf
    min_sep_step = None

    max_steps = int(round(cfg.max_time / cfg.dt))
    dist_history = []
    evader_traj = []

    alpha = min(1.0, cfg.dt / cfg.tau)

    for k in range(1, max_steps + 1):
        evader_pos = positions[:, evader].copy()

        # 路径长度统计
        evader_path += float(np.linalg.norm(evader_pos - prev_evader))
        pursuer_path += np.linalg.norm(positions[:, : cfg.n_pursuers] - prev_positions[:, : cfg.n_pursuers], axis=0)
        prev_evader = evader_pos
        prev_positions = positions.copy()

        # 逃逸策略
        f_evader = evader_force(positions[:, : cfg.n_pursuers], evader_pos, cfg)
        if np.linalg.norm(f_evader) > EPS:
            evader_vel = cfg.v_evader * f_evader / np.linalg.norm(f_evader)
        elif np.linalg.norm(evader_vel) < EPS:
            evader_vel = cfg.v_evader * np.array([np.cos(heading), np.sin(heading)])
        else:
            evader_vel = cfg.v_evader * evader_vel / np.linalg.norm(evader_vel)

        # 到达时序
        dist_to_evader = np.linalg.norm(positions[:, : cfg.n_pursuers] - evader_pos[:, None], axis=0)
        for i in pursuers:
            if not has_arrived[i] and dist_to_evader[i] <= cfg.r_detect:
                has_arrived[i] = True
                arrival_time[i] = (k - 1) * cfg.dt

        # 角色分配
        interceptor = choose_interceptor(positions[:, : cfg.n_pursuers], evader_pos, evader_vel, cfg)

        # 拦截机势场制导 + 一阶速度响应
        dxi = np.zeros((2, cfg.n_pursuers))
        for i in pursuers:
            is_inter = i == interceptor
            f_total = pursuer_force(i, is_inter, positions[:, : cfg.n_pursuers], evader_pos, evader_vel, cfg)
            vel_cmd = (
                cfg.v_pursuer * f_total / np.linalg.norm(f_total)
                if np.linalg.norm(f_total) > EPS
                else np.zeros(2)
            )
            pursuer_vel_state[:, i] += alpha * (vel_cmd - pursuer_vel_state[:, i])
            speed = np.linalg.norm(pursuer_vel_state[:, i])
            if speed > cfg.v_pursuer:
                pursuer_vel_state[:, i] = cfg.v_pursuer * pursuer_vel_state[:, i] / speed
            dxi[:, i] = pursuer_vel_state[:, i]

        # 围捕判定（至少 required_captors 架进入捕获半径）
        sorted_dist = np.sort(dist_to_evader)
        best_second = min(best_second, float(sorted_dist[1]) if sorted_dist.size >= 2 else np.inf)
        if np.sum(dist_to_evader < cfg.capture_radius) >= cfg.required_captors:
            captured = True
            capture_step = k
            dxi[:, :] = 0.0

        # 机间最小间距（安全指标）
        for i in pursuers:
            for j in pursuers:
                if j > i:
                    sep = float(np.linalg.norm(positions[:, i] - positions[:, j]))
                    if sep < min_sep:
                        min_sep = sep
                        min_sep_step = k

        # 积分更新
        positions[:, : cfg.n_pursuers] = positions[:, : cfg.n_pursuers] + dxi * cfg.dt
        positions[:, evader] = positions[:, evader] + evader_vel * cfg.dt

        evader_traj.append(positions[:, evader].copy())
        dist_history.append(dist_to_evader.copy())

        if captured:
            break

    # 指标统计
    valid_arrival = arrival_time[~np.isnan(arrival_time)]
    arrival_var = float(np.var(valid_arrival)) if valid_arrival.size >= 2 else None
    realized_ratio = (
        float(np.mean(pursuer_path)) / max(evader_path, 1e-8) if evader_path > 0 else None
    )
    capture_time = float((capture_step - 1) * cfg.dt) if captured and capture_step else None

    return SimResult(
        captured=captured,
        capture_time=capture_time,
        capture_step=capture_step,
        best_second_distance=float(best_second),
        arrival_count=int(np.sum(has_arrived)),
        arrival_variance=arrival_var,
        realized_path_ratio=realized_ratio,
        min_sep=float(min_sep),
        min_sep_step=min_sep_step,
        final_positions=positions.copy(),
        evader_traj=np.array(evader_traj),
        pursuer_dist_history=np.array(dist_history),
    )
