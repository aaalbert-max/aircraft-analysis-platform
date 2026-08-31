"""设计空间采样、蒙特卡洛评估、代理模型与参数寻优。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import InterceptionConfig, NOMINAL
from .metrics import aggregate
from .simulator import simulate_trial


@dataclass
class DesignRecord:
    """单个设计点（rho, n_pursuers）的蒙特卡洛评估结果。"""

    rho: float
    n_pursuers: int
    n_trials: int
    success_rate: float
    mean_capture_time: float
    median_capture_time: float
    mean_min_sep: float
    mean_arrival_variance: float
    timeout_rate: float

    @property
    def features(self) -> np.ndarray:
        return np.array([self.rho, self.n_pursuers], dtype=float)

    def to_dict(self) -> Dict[str, float]:
        return {
            "rho": self.rho,
            "n_pursuers": self.n_pursuers,
            "n_trials": self.n_trials,
            "success_rate": self.success_rate,
            "mean_capture_time": self.mean_capture_time,
            "median_capture_time": self.median_capture_time,
            "mean_min_sep": self.mean_min_sep,
            "mean_arrival_variance": self.mean_arrival_variance,
            "timeout_rate": self.timeout_rate,
        }


def evaluate_design(
    cfg: InterceptionConfig,
    seeds: Sequence[int],
) -> DesignRecord:
    """对一个设计点运行蒙特卡洛并聚合成指标。"""
    results = [simulate_trial(cfg, seed=s) for s in seeds]
    m = aggregate(results)
    return DesignRecord(
        rho=cfg.rho,
        n_pursuers=cfg.n_pursuers,
        n_trials=len(seeds),
        success_rate=m.success_rate,
        mean_capture_time=m.mean_capture_time,
        median_capture_time=m.median_capture_time,
        mean_min_sep=m.mean_min_sep,
        mean_arrival_variance=m.mean_arrival_variance,
        timeout_rate=m.timeout_rate,
    )


def sweep_design_space(
    base: InterceptionConfig,
    rhos: Sequence[float],
    ns: Sequence[int],
    seeds: Sequence[int],
) -> List[DesignRecord]:
    """在 (rho, n) 网格上逐点评估。"""
    records: List[DesignRecord] = []
    for rho in rhos:
        for n in ns:
            cfg = InterceptionConfig(**{**base.__dict__, "rho": float(rho), "n_pursuers": int(n)})
            records.append(evaluate_design(cfg, seeds))
    return records


def split_train_test(
    records: List[DesignRecord], test_frac: float = 0.2, seed: int = 0
) -> Tuple[List[DesignRecord], List[DesignRecord]]:
    """按设计点做训练/测试划分（避免同一设计点不同种子泄漏）。"""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(records))
    rng.shuffle(idx)
    n_test = max(1, int(round(test_frac * len(records))))
    test_idx = set(idx[:n_test].tolist())
    train = [r for i, r in enumerate(records) if i not in test_idx]
    test = [r for i, r in enumerate(records) if i in test_idx]
    return train, test


def fit_surrogate(records: List[DesignRecord]):
    """训练高斯过程代理模型（rho, n -> success_rate），返回 (model, scaler)。"""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.array([r.features for r in records], dtype=float)
    y = np.array([r.success_rate for r in records], dtype=float)
    kernel = ConstantKernel(1.0) * RBF(length_scale=[0.1, 4.0]) + WhiteKernel(noise_level=1e-3)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("gp", GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-6)),
        ]
    )
    model.fit(X, y)
    return model


def surrogate_r2(model, records: List[DesignRecord]) -> float:
    """代理模型在给定设计点上的 R²（用于验证拟合质量）。"""
    X = np.array([r.features for r in records], dtype=float)
    y = np.array([r.success_rate for r in records], dtype=float)
    pred = model.predict(X)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def optimize_on_surrogate(
    model,
    rho_bounds: Tuple[float, float],
    n_bounds: Tuple[float, float],
    grid: int = 40,
    n_max: float = 12.0,
):
    """在代理模型上做网格寻优，返回使预测成功率最高的 (rho, n)。"""
    rhos = np.linspace(rho_bounds[0], rho_bounds[1], grid)
    ns = np.linspace(n_bounds[0], n_bounds[1], grid)
    xx, yy = np.meshgrid(rhos, ns)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    pred = np.clip(model.predict(pts), 0.0, 1.0)
    best = int(np.argmax(pred))
    rho_star = float(pts[best, 0])
    n_star = float(pts[best, 1])
    n_star_int = int(round(min(max(n_star, n_bounds[0]), n_max)))
    return rho_star, n_star_int, float(pred[best])


def fit_surrogates(records):
    """对 success_rate / mean_capture_time / mean_min_sep 各训练一个代理模型。

    使用随机森林以稳健处理含噪声的 0-1 成功率；每个输出单独训练并丢弃该输出的
    NaN 样本，返回 {key: model}。
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.array([r.features for r in records], dtype=float)
    models = {}
    for key in ("success_rate", "mean_capture_time", "mean_min_sep"):
        y = np.array([getattr(r, key) for r in records], dtype=float)
        mask = np.isfinite(y)
        Xm, ym = X[mask], y[mask]
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(n_estimators=300, random_state=0)),
        ])
        if len(ym) >= 3:
            pipe.fit(Xm, ym)
        models[key] = pipe
    return models


def surrogate_predict(model, pts):
    """对代理模型预测，返回一维数组。"""
    return np.asarray(model.predict(pts), dtype=float).ravel()


def constrained_optimize(
    records,
    t_window: float = 60.0,
    hard_safe: float = 10.0,
    grid: int = 60,
    rho_bounds: Tuple[float, float] = (0.4, 0.9),
    n_bounds: Tuple[float, float] = (3, 11),
):
    """约束寻优：在代理模型上最大化成功率，约束 捕获时间<=t_window 且 最小机间距>=hard_safe。

    返回 (rho_star, n_star, pred_success, pred_time, pred_minsep, feasible_frac)。
    """
    models = fit_surrogates(records)
    rhos = np.linspace(rho_bounds[0], rho_bounds[1], grid)
    ns = np.linspace(n_bounds[0], n_bounds[1], grid)
    xx, yy = np.meshgrid(rhos, ns)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    succ = np.clip(surrogate_predict(models["success_rate"], pts), 0, 1)
    time_p = surrogate_predict(models["mean_capture_time"], pts)
    minsep_p = surrogate_predict(models["mean_min_sep"], pts)
    feasible = (time_p <= t_window) & (minsep_p >= hard_safe)
    feasible_frac = float(np.mean(feasible))
    if not np.any(feasible):
        best = int(np.argmax(succ))
    else:
        masked = np.where(feasible, succ, -1.0)
        best = int(np.argmax(masked))
    return (
        float(pts[best, 0]),
        int(round(pts[best, 1])),
        float(succ[best]),
        float(time_p[best]),
        float(minsep_p[best]),
        feasible_frac,
    )


def min_formation_for_success(
    records,
    target_success: float = 0.85,
    t_window: float = 60.0,
    hard_safe: float = 10.0,
    grid: int = 80,
    rho_bounds: Tuple[float, float] = (0.4, 0.9),
    n_bounds: Tuple[float, float] = (3, 11),
    rho_points: Sequence[float] = (0.45, 0.55, 0.65, 0.75, 0.85),
):
    """求满足 成功率>=target_success 且 时间<=t_window 且 机间距>=hard_safe 的最小编队 n(ρ)。

    rho_points 为要输出的候选速度比（默认取设计采样点）；返回 (rows, overall)。
    """
    models = fit_surrogates(records)
    rhos = np.linspace(rho_bounds[0], rho_bounds[1], grid)
    ns = np.linspace(n_bounds[0], n_bounds[1], grid)
    xx, yy = np.meshgrid(rhos, ns)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    succ = np.clip(surrogate_predict(models["success_rate"], pts), 0, 1).reshape(xx.shape)
    timep = surrogate_predict(models["mean_capture_time"], pts).reshape(xx.shape)
    minsepp = surrogate_predict(models["mean_min_sep"], pts).reshape(xx.shape)
    feasible = (timep <= t_window) & (minsepp >= hard_safe)

    def n_at(rho):
        k = int(np.argmin(np.abs(rhos - rho)))
        for j in range(len(ns)):
            if feasible[j, k] and succ[j, k] >= target_success:
                return float(ns[j]), float(succ[j, k])
        return float("nan"), float("nan")

    rows = []
    overall = None
    for rho in list(rho_points):
        n_req, s_req = n_at(rho)
        rows.append({"rho": float(rho), "n_required": n_req, "pred_success": s_req})
        if np.isfinite(n_req) and (overall is None or n_req < overall[1]):
            overall = (float(rho), float(n_req), float(s_req))
    return rows, overall
