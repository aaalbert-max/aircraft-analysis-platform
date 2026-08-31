"""单次围捕仿真主循环（向量化）与结果记录。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import InterceptionConfig
from .guidance import choose_interceptor, evader_force

EPS = 1e-8


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
    min_sep_frac: float      # 最小机间距 / d_safe
    final_positions: np.ndarray
    evader_traj: np.ndarray
    pursuer_traj: np.ndarray
    pursuer_dist_history: np.ndarray


def _initial_positions(cfg: InterceptionConfig, rng: np.random.Generator) -> np.ndarray:
    """按随机方向、半径散布生成初始拦截机位置（dim 维）。"""
    dim = cfg.dim
    n = cfg.n_pursuers
    positions = np.zeros((dim, n))
    base = np.asarray(cfg.evader_init, dtype=float)
    if base.shape[0] < dim:
        base = np.concatenate([base, np.zeros(dim - base.shape[0])])
    for i in range(n):
        vec = rng.normal(size=dim)
        vec /= np.linalg.norm(vec) + EPS
        radius = cfg.init_scale * (0.75 + 0.5 * rng.random())
        positions[:, i] = base + radius * vec
    return positions


def _patrol_positions(cfg, rng):
    """拦截机在任务空域内均匀散布（巡逻位），保证初始最小间距。"""
    dim = cfg.dim
    n = cfg.n_pursuers
    positions = np.zeros((dim, n))
    min_gap = max(cfg.hard_safe, cfg.init_scale * 0.2)
    placed = 0
    attempts = 0
    while placed < n and attempts < 4000:
        attempts += 1
        cand = np.array([rng.uniform(cfg.xmin, cfg.xmax), rng.uniform(cfg.ymin, cfg.ymax)])
        if dim > 2:
            extras = [rng.uniform(cfg.zmin, cfg.zmax) for _ in range(dim - 2)]
            cand = np.concatenate([cand, extras])
        if placed and np.linalg.norm(positions[:, :placed] - cand[:, None], axis=0).min() < min_gap:
            continue
        positions[:, placed] = cand
        placed += 1
    return positions


def _spawn_scenario(cfg, rng):
    """生成一次拦截遭遇：返回(拦截机位置, 目标位置, 目标初始航向)。"""
    if cfg.scenario == "patrol":
        positions = _patrol_positions(cfg, rng)
        margin = max(cfg.boundary_margin, cfg.capture_radius)
        evader = np.array([
            rng.uniform(cfg.xmin + margin, cfg.xmax - margin),
            rng.uniform(cfg.ymin + margin, cfg.ymax - margin),
        ])
        if cfg.dim > 2:
            zmargin = max(cfg.boundary_margin, cfg.capture_radius)
            evader = np.concatenate([evader, [rng.uniform(cfg.zmin + zmargin, cfg.zmax - zmargin)]])
    else:
        positions = _initial_positions(cfg, rng)
        evader = np.asarray(cfg.evader_init, dtype=float)
    heading = rng.uniform(-np.pi, np.pi)
    return positions, evader, heading


def _pairwise_separation(positions) -> np.ndarray:
    """拦截机两两间距矩阵（n x n）。"""
    n = positions.shape[1]
    d = np.zeros((n, n))
    for i in range(n):
        d[i] = np.linalg.norm(positions - positions[:, i : i + 1], axis=0)
    return d


def simulate_trial(cfg: InterceptionConfig, seed: Optional[int] = None) -> SimResult:
    """运行一次协同围捕仿真（向量化），返回结果对象。"""
    cfg.validate()
    rng = np.random.default_rng(seed if seed is not None else cfg.seed)
    dim = cfg.dim
    n_p = cfg.n_pursuers

    positions = np.zeros((dim, n_p + 1))
    init_positions, evader_pos0, heading = _spawn_scenario(cfg, rng)
    positions[:, :n_p] = init_positions
    positions[:, n_p] = np.asarray(evader_pos0, dtype=float)
    evader_vel = cfg.v_evader * np.concatenate(
        [np.array([np.cos(heading), np.sin(heading)]), np.zeros(dim - 2)]
    )
    pursuer_vel_state = np.zeros((dim, n_p))

    captured = False
    capture_step = None
    best_second = np.inf
    has_arrived = np.zeros(n_p, dtype=bool)
    arrival_time = np.full(n_p, np.nan)
    evader_path = 0.0
    pursuer_path = np.zeros(n_p)
    prev_positions = positions.copy()
    prev_evader = positions[:, n_p].copy()
    min_sep = np.inf

    max_steps = int(round(cfg.max_time / cfg.dt))
    dist_history: List[np.ndarray] = []
    evader_traj: List[np.ndarray] = []
    pursuer_traj: List[np.ndarray] = []

    alpha = min(1.0, cfg.dt / cfg.tau)

    for k in range(1, max_steps + 1):
        evader_pos = positions[:, n_p].copy()

        # 路径长度
        evader_path += float(np.linalg.norm(evader_pos - prev_evader))
        delta = positions[:, :n_p] - prev_positions[:, :n_p]
        pursuer_path += np.linalg.norm(delta, axis=0)
        prev_evader = evader_pos
        prev_positions = positions.copy()

        # 逃逸策略
        f_evader = evader_force(positions[:, :n_p], evader_pos, cfg)
        fnorm = float(np.linalg.norm(f_evader))
        if fnorm > EPS:
            evader_vel = cfg.v_evader * f_evader / fnorm
        elif np.linalg.norm(evader_vel) < EPS:
            evader_vel = cfg.v_evader * np.concatenate(
                [np.array([np.cos(heading), np.sin(heading)]), np.zeros(dim - 2)]
            )
        else:
            evader_vel = cfg.v_evader * evader_vel / np.linalg.norm(evader_vel)

        # 到达时序
        dist_to_evader = np.linalg.norm(positions[:, :n_p] - evader_pos[:, None], axis=0)
        for i in range(n_p):
            if not has_arrived[i] and dist_to_evader[i] <= cfg.r_detect:
                has_arrived[i] = True
                arrival_time[i] = (k - 1) * cfg.dt

        # 角色分配（拦截者）
        interceptor = choose_interceptor(positions[:, :n_p], evader_pos, evader_vel, cfg)

        # 人工势场：引力 + 机间斥力（向量化）
        lead = np.clip(dist_to_evader / max(cfg.v_pursuer, EPS), cfg.lead_min, cfg.lead_max)
        aim = evader_pos[:, None] + evader_vel[:, None] * lead[None, :]
        aim[:, :] = evader_pos[:, None]
        aim[:, interceptor] = evader_pos + evader_vel * lead[interceptor]

        f_att = cfg.k_att * (aim - positions[:, :n_p])
        w_rep = np.full(n_p, cfg.w_rep_herder)
        w_rep[interceptor] = cfg.w_rep_interceptor

        f_rep = np.zeros((dim, n_p))
        dmat = _pairwise_separation(positions[:, :n_p])
        np.fill_diagonal(dmat, np.inf)
        masked = dmat < cfg.d_safe
        for j in range(n_p):
            row = masked[j]
            if not np.any(row):
                continue
            away = positions[:, :n_p] - positions[:, j : j + 1]
            sep = dmat[j]
            safe_denom = np.where(row, np.maximum(sep, EPS), 1.0)
            coef = np.where(row, cfg.k_rep * (1.0 - sep / cfg.d_safe) / safe_denom, 0.0)
            f_rep += coef[None, :] * away

        f_total = f_att + w_rep[None, :] * f_rep
        f_norm = np.linalg.norm(f_total, axis=0)
        vel_cmd = np.where(
            f_norm[None, :] > EPS,
            cfg.v_pursuer * f_total / np.maximum(f_norm[None, :], EPS),
            0.0,
        )
        pursuer_vel_state += alpha * (vel_cmd - pursuer_vel_state)
        speed = np.linalg.norm(pursuer_vel_state, axis=0)
        over = speed > cfg.v_pursuer
        if np.any(over):
            pursuer_vel_state[:, over] = cfg.v_pursuer * pursuer_vel_state[:, over] / speed[over]
        dxi = pursuer_vel_state

        # 围捕判定
        sorted_dist = np.sort(dist_to_evader)
        best_second = min(best_second, float(sorted_dist[1]) if sorted_dist.size >= 2 else np.inf)
        if np.sum(dist_to_evader < cfg.capture_radius) >= cfg.required_captors:
            captured = True
            capture_step = k
            dxi[:, :] = 0.0

        # 机间最小间距
        off_diag = dmat.copy()
        off_diag[np.isinf(off_diag)] = np.nan
        step_min = np.nanmin(off_diag)
        if np.isfinite(step_min) and step_min < min_sep:
            min_sep = float(step_min)

        # 执行层安全屏障：禁止间距小于 hard_safe 的接近（模拟机载安全约束）
        dmat2 = _pairwise_separation(positions[:, :n_p])
        np.fill_diagonal(dmat2, np.inf)
        for i in range(n_p):
            for j in range(i + 1, n_p):
                d = dmat2[i, j]
                if d < cfg.hard_safe:
                    u = (positions[:, i] - positions[:, j]) / max(d, EPS)
                    vrel = float(np.dot(dxi[:, i] - dxi[:, j], u))
                    if vrel < 0.0:
                        dxi[:, i] -= vrel * u / 2.0
                        dxi[:, j] += vrel * u / 2.0
        speed2 = np.linalg.norm(dxi, axis=0)
        over2 = speed2 > cfg.v_pursuer
        if np.any(over2):
            dxi[:, over2] = cfg.v_pursuer * dxi[:, over2] / speed2[over2]

        # 积分更新
        positions[:, :n_p] += dxi * cfg.dt
        positions[:, n_p:] += evader_vel[:, None] * cfg.dt

        evader_traj.append(positions[:, n_p].copy())
        pursuer_traj.append(positions[:, :n_p].copy())
        dist_history.append(dist_to_evader.copy())

        if captured:
            break

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
        min_sep_frac=float(min_sep / cfg.d_safe) if cfg.d_safe > 0 else 0.0,
        final_positions=positions.copy(),
        evader_traj=np.array(evader_traj),
        pursuer_traj=np.array(pursuer_traj),
        pursuer_dist_history=np.array(dist_history),
    )
