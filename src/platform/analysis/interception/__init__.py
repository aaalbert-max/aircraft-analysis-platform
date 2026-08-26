"""多拦截机协同围捕仿真与分析引擎。

面向「基于三维覆盖立体角的多拦截机协同围捕制导」核心技术，提供：
- 追逃运动模型与智能逃逸策略
- 拦截者-驱赶者动态角色分配
- 人工势场协同制导与机间避碰
- 围捕判定与性能指标统计
- 单次 / 批量 / 蒙特卡洛运行与结果导出

算法忠实还原自 Robotarium 仿真集成
``run_04_v9_rho_065_trial_02.m``，并支持本项目标称工况（10 m/s 目标、
6.5 m/s 拦截机）与原参考工况（20 m/s / 13 m/s）。
"""

from .config import CAPTURE, InterceptionConfig, REFERENCE, NOMINAL
from .simulator import SimResult, simulate_trial
from .runner import run_single, run_batch, default_sweep_report

__all__ = [
    "InterceptionConfig",
    "NOMINAL",
    "REFERENCE",
    "CAPTURE",
    "SimResult",
    "simulate_trial",
    "run_single",
    "run_batch",\n    "default_sweep_report",
]
