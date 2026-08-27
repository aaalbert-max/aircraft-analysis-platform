"""三维覆盖立体角与最小编队规模。

依据方案报告 3.5.3 节的阿波罗尼斯球几何：

    不可逃逸锥半张角  sin α(d, ρ, R_cap) = ρ + R_cap (1 - ρ^2) / d
    单机覆盖立体角    Ω = 4π                              , d ≤ R_cap
                       Ω = 2π (1 - sqrt(1 - sin^2 α))     , d > R_cap
    三维全包围条件    sum_i Ω_i ≥ 4π
    最小编队规模      n_ideal = ceil(4π / Ω)
                     n(min)  = ceil(4π / (η Ω))   , η 为球面填充效率 (0,1]

本模块把「给定速度比 / 捕获半径 / 典型距离 → 需要几架拦截机」做成可直接调用的
工程判据，服务于总体设计阶段的编队规模权衡。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CoverageResult:
    """单架拦截机的三维覆盖分析结果。"""

    rho: float
    r_cap: float
    distance: float
    sin_alpha: float
    cos_alpha: float
    omega_sr: float
    ideal_min: int     # ceil(4π / Ω)
    eta_min: int       # 考虑填充效率后的 ceil(4π / (η Ω))
    full_coverage: bool  # 目标已在捕获半径内（全向覆盖）


def sin_alpha(rho: float, r_cap: float, distance: float) -> float:
    """计算不可逃逸锥半张角的正弦。"""
    return rho + r_cap * (1.0 - rho**2) / distance


def coverage_solid_angle(rho: float, r_cap: float, distance: float) -> float:
    """单架拦截机的三维覆盖立体角（球面度）。

    若目标已在捕获半径内（d ≤ R_cap）则视为全向覆盖 4π。
    """
    if distance <= r_cap:
        return 4.0 * math.pi
    sa = sin_alpha(rho, r_cap, distance)
    if sa >= 1.0:
        # 超出适用域：不可逃逸锥张开过头，按几何上不提供封锁处理（返回 0 并提示）
        return 0.0
    ca = math.sqrt(1.0 - sa * sa)
    return 2.0 * math.pi * (1.0 - ca)


def min_formation(rho: float, r_cap: float, distance: float, eta: float = 0.8) -> int:
    """考虑球面填充效率的最小编队规模（向上取整）。"""
    omega = coverage_solid_angle(rho, r_cap, distance)
    if omega <= 0.0:
        # 无法形成有效覆盖，返回一个很大的安全值，提示需要更多拦截机
        return 999
    return int(math.ceil(4.0 * math.pi / (eta * omega)))


def analyze(rho: float, r_cap: float, distance: float, eta: float = 0.8) -> CoverageResult:
    """完整覆盖分析：返回夹角、立体角与最小编队规模。"""
    sa = sin_alpha(rho, r_cap, distance)
    omega = coverage_solid_angle(rho, r_cap, distance)
    ideal = int(math.ceil(4.0 * math.pi / omega)) if omega > 0 else 999
    eta_min = min_formation(rho, r_cap, distance, eta)
    ca = math.sqrt(max(0.0, 1.0 - sa * sa))
    return CoverageResult(
        rho=rho,
        r_cap=r_cap,
        distance=distance,
        sin_alpha=sa,
        cos_alpha=ca,
        omega_sr=omega,
        ideal_min=ideal,
        eta_min=eta_min,
        full_coverage=distance <= r_cap,
    )


def min_formation_table(
    rhos, r_cap: float = 15.0, distance: float = 80.0, eta: float = 0.8
) -> list:
    """不同速度比下的理论最小编队规模表（对应报告表 3.5-3）。"""
    rows = []
    for rho in rhos:
        r = analyze(rho, r_cap, distance, eta)
        rows.append(
            {
                "rho": rho,
                "sin_alpha": r.sin_alpha,
                "omega_sr": r.omega_sr,
                "ideal": 4.0 * math.pi / r.omega_sr if r.omega_sr > 0 else float("inf"),
                "n_ideal": r.ideal_min,
                "n_eta": r.eta_min,
            }
        )
    return rows
