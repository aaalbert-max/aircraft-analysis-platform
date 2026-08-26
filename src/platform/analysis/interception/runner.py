"""运行器：单次独立运行与批量/蒙特卡洛扫描。"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .config import InterceptionConfig, NOMINAL
from .metrics import BatchMetrics, aggregate
from .simulator import SimResult, simulate_trial


def run_single(cfg: InterceptionConfig, seed: Optional[int] = 2000) -> SimResult:
    """运行单次围捕仿真。"""
    return simulate_trial(cfg, seed=seed)


def run_batch(
    cfg: InterceptionConfig,
    seeds: Sequence[int],
) -> List[SimResult]:
    """针对一组随机种子运行蒙特卡洛仿真。"""
    return [simulate_trial(cfg, seed=s) for s in seeds]


def run_and_aggregate(cfg: InterceptionConfig, seeds: Sequence[int]) -> BatchMetrics:
    return aggregate(run_batch(cfg, seeds))


def sweep_rho(
    base: InterceptionConfig,
    rhos: Sequence[float],
    seeds: Sequence[int],
) -> Dict[str, BatchMetrics]:
    """速度比扫描，考察捕获成功率等随速度比变化（用于复现倒U形关系）。"""
    out: Dict[str, BatchMetrics] = {}
    for rho in rhos:
        cfg = InterceptionConfig(**{**base.__dict__, "rho": float(rho)})
        out[f"{rho:.3f}"] = run_and_aggregate(cfg, seeds)
    return out


def default_sweep_report(base: InterceptionConfig = NOMINAL, n_trials: int = 20) -> str:
    """生成速度比扫描的文本报告。"""
    rhos = [0.4, 0.5, 0.6, 0.65, 0.7, 0.8]
    seeds = list(range(2000, 2000 + n_trials))
    results = sweep_rho(base, rhos, seeds)
    lines = [
        "速度比扫描（捕获成功率随速度比变化）",
        f"工况: v_evader={base.v_evader} m/s, 每档 trials={n_trials}",
        f"{'rho':>5} | {'成功率':>8} | {'均捕获时间':>10} | {'超时率':>8}",
        "-" * 50,
    ]
    for rho_str, m in results.items():
        lines.append(
            f"{rho_str:>5} | {m.success_rate*100:>7.1f}% | {m.mean_capture_time:>10.2f} s "
            f"| {m.timeout_rate*100:>7.1f}%"
        )
    return "\n".join(lines)
