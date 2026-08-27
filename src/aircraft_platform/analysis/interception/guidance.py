"""制导与策略模块：逃逸策略、角色分配、人工势场。"""

from __future__ import annotations

import numpy as np

from .config import InterceptionConfig

EPS = 1e-8


def evader_force(positions, evader_pos, cfg: InterceptionConfig) -> np.ndarray:
    """智能逃逸策略：探测距离内拦截机斥力 + 空域软边界斥力。"""
    d = cfg.n_pursuers
    force = np.zeros(2)
    for j in range(d):
        away = evader_pos - positions[:, j]
        dist = float(np.linalg.norm(away))
        if dist < cfg.r_detect and dist > EPS:
            force = force + cfg.k_e_rep / (dist**2 + 1.0) * away / dist
    # 软边界斥力（按边界裕度触发）
    wall = np.array([
        evader_pos[0] - cfg.xmin,
        cfg.xmax - evader_pos[0],
        evader_pos[1] - cfg.ymin,
        cfg.ymax - evader_pos[1],
    ])
    if wall[0] < cfg.boundary_margin:
        force[0] += cfg.k_bound / max(wall[0] ** 2, 1.0)
    elif wall[1] < cfg.boundary_margin:
        force[0] -= cfg.k_bound / max(wall[1] ** 2, 1.0)
    if wall[2] < cfg.boundary_margin:
        force[1] += cfg.k_bound / max(wall[2] ** 2, 1.0)
    elif wall[3] < cfg.boundary_margin:
        force[1] -= cfg.k_bound / max(wall[3] ** 2, 1.0)
    return force


def choose_interceptor(positions, evader_pos, evader_vel, cfg: InterceptionConfig) -> int:
    """拦截者-驱赶者动态角色分配。

    ``角色角`` = 目标逃逸方向与 (拦截机->目标) 方向的夹角。
    夹角小于阈值的候选取最近者作为拦截者；若无候选则强制取最近者。
    """
    evader_dir = evader_vel / max(float(np.linalg.norm(evader_vel)), EPS)
    role_angle = np.zeros(cfg.n_pursuers)
    for i in range(cfg.n_pursuers):
        p2e = evader_pos - positions[:, i]
        p2e = p2e / max(float(np.linalg.norm(p2e)), EPS)
        role_angle[i] = np.degrees(
            np.clip(float(np.dot(p2e, evader_dir)), -1.0, 1.0)
        )
    candidates = np.where(role_angle < cfg.role_angle_deg)[0]
    if candidates.size == 0:
        interceptor = int(np.argmin(role_angle))
    else:
        interceptor = int(candidates[np.argmin(role_angle[candidates])])
    return interceptor


def pursuer_force(
    i: int,
    is_interceptor: bool,
    positions,
    evader_pos,
    evader_vel,
    cfg: InterceptionConfig,
) -> np.ndarray:
    """人工势场合力：引力（朝目标点）+ 机间斥力（线性衰减避碰）。"""
    v_pursuer = cfg.v_pursuer
    if is_interceptor:
        dist = float(np.linalg.norm(evader_pos - positions[:, i]))
        lead_time = min(cfg.lead_max, max(cfg.lead_min, dist / max(v_pursuer, EPS)))
        aim = evader_pos + evader_vel * lead_time
        w_rep = cfg.w_rep_interceptor
    else:
        aim = evader_pos
        w_rep = cfg.w_rep_herder

    f_att = cfg.k_att * (aim - positions[:, i])
    f_rep = np.zeros(2)
    for j in range(cfg.n_pursuers):
        if j == i:
            continue
        away = positions[:, i] - positions[:, j]
        sep = float(np.linalg.norm(away))
        if sep < cfg.d_safe and sep > EPS:
            f_rep = f_rep + cfg.k_rep * (1.0 - sep / cfg.d_safe) * away / sep
    return cfg.w_att * f_att + w_rep * f_rep
