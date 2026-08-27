"""批量结果聚合指标。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .simulator import SimResult


@dataclass
class BatchMetrics:
    trials: int
    success_rate: float
    mean_capture_time: float
    median_capture_time: float
    std_capture_time: float
    min_capture_time: float
    max_capture_time: float
    mean_best_second: float
    mean_arrival_count: float
    mean_arrival_variance: float
    mean_min_sep: float
    timeout_rate: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "trials": float(self.trials),
            "success_rate": self.success_rate,
            "mean_capture_time": self.mean_capture_time,
            "median_capture_time": self.median_capture_time,
            "std_capture_time": self.std_capture_time,
            "min_capture_time": self.min_capture_time,
            "max_capture_time": self.max_capture_time,
            "mean_best_second": self.mean_best_second,
            "mean_arrival_count": self.mean_arrival_count,
            "mean_arrival_variance": self.mean_arrival_variance,
            "mean_min_sep": self.mean_min_sep,
            "timeout_rate": self.timeout_rate,
        }


def aggregate(results: List[SimResult]) -> BatchMetrics:
    """把多次仿真结果聚合成统计指标。"""
    n = len(results)
    captured = [r for r in results if r.captured]
    success_rate = len(captured) / n if n else 0.0

    times = [r.capture_time for r in captured if r.capture_time is not None]
    mean_t = float(np.mean(times)) if times else float("nan")
    median_t = float(np.median(times)) if times else float("nan")
    std_t = float(np.std(times)) if times else float("nan")
    min_t = float(np.min(times)) if times else float("nan")
    max_t = float(np.max(times)) if times else float("nan")

    best_second = [r.best_second_distance for r in results if np.isfinite(r.best_second_distance)]
    arrival_var = [r.arrival_variance for r in results if r.arrival_variance is not None]
    arr_count = [r.arrival_count for r in results]
    min_sep = [r.min_sep for r in results if np.isfinite(r.min_sep)]

    timeout_rate = 1.0 - success_rate
    return BatchMetrics(
        trials=n,
        success_rate=success_rate,
        mean_capture_time=mean_t,
        median_capture_time=median_t,
        std_capture_time=std_t,
        min_capture_time=min_t,
        max_capture_time=max_t,
        mean_best_second=float(np.mean(best_second)) if best_second else float("nan"),
        mean_arrival_count=float(np.mean(arr_count)) if arr_count else float("nan"),
        mean_arrival_variance=float(np.mean(arrival_var)) if arrival_var else float("nan"),
        mean_min_sep=float(np.mean(min_sep)) if min_sep else float("nan"),
        timeout_rate=timeout_rate,
    )
