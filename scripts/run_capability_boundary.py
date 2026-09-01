#!/usr/bin/env python
"""能力边界消融：空域横向尺度 vs 围捕成功率（定位A：有限处置区防护）。

目的：显式量化"本方案的成功在多大程度上依赖有限空域边界"。对每种处置区横向尺度
跑蒙特卡洛，记录：
  - success_rate           捕获成功率（目标在离开处置区前被 >=required_captors 架接近）
  - boundary_assisted_rate 成功样本中，目标在其轨迹上进入过"边界规避层"的比例
                            （即软边界斥力实际参与收口）
  - pure_encirclement_rate 成功样本中，目标全程未进入边界规避层的比例（纯几何包围收口）
  - mean_capture_time      平均捕获时间（仅成功样本）
  - timeout_rate           未在规定时间内捕获的比例（几何上逃逸/耗完窗口）

用法：
    PYTHONPATH=src python scripts/run_capability_boundary.py [--trials 60] [--out data/outputs/capability_boundary]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from aircraft_platform.analysis.interception.config import InterceptionConfig
from aircraft_platform.analysis.interception.simulator import simulate_trial


def _boundary_contact_ratio(cfg: InterceptionConfig, evader_traj: np.ndarray) -> float:
    """返回目标在其完整轨迹上进入边界规避层（距任一墙 < boundary_margin）的采样占比。"""
    if evader_traj.size == 0:
        return 0.0
    e = np.asarray(evader_traj)
    dim = e.shape[1]
    mins = np.full(e.shape[0], np.inf)
    for axis, (lo, hi) in enumerate(
        [(cfg.xmin, cfg.xmax), (cfg.ymin, cfg.ymax), (cfg.zmin, cfg.zmax)]
    ):
        if axis >= dim:
            continue
        mins = np.minimum(mins, np.minimum(e[:, axis] - lo, hi - e[:, axis]))
    return float(np.mean(mins < cfg.boundary_margin))


def evaluate_scale(h: float, n_trials: int, base_seed: int = 1) -> dict:
    """对单个空域半宽 h（方形域 [-h,h]^2）做蒙特卡洛评估。"""
    cfg = InterceptionConfig(
        v_evader=10.0,
        rho=0.65,
        capture_radius=15.0,
        r_detect=80.0,
        d_safe=50.0,
        boundary_margin=30.0,
        max_time=120.0,
        init_scale=min(0.4 * h, 200.0),
        xmin=-float(h),
        xmax=float(h),
        ymin=-float(h),
        ymax=float(h),
    )
    succ: List[float] = []
    succ_has_boundary: List[bool] = []
    times: List[float] = []

    for s in range(base_seed, base_seed + n_trials):
        res = simulate_trial(cfg, seed=s)
        contact = _boundary_contact_ratio(cfg, res.evader_traj)
        if res.captured:
            succ.append(1.0)
            # 目标轨迹中有任一采样进入边界规避层 => 认为是边界辅助
            succ_has_boundary.append(contact > 0.0)
            if res.capture_time is not None:
                times.append(res.capture_time)
        else:
            succ.append(0.0)

    success_rate = float(np.mean(succ)) if succ else 0.0
    n_succ = len(succ_has_boundary)
    boundary_assisted = float(np.mean(succ_has_boundary)) if n_succ else float("nan")
    pure_encirclement = 1.0 - boundary_assisted if n_succ else float("nan")
    return {
        "lateral_half_extent_m": float(h),
        "area_scale_m2": float((2 * h) ** 2),
        "n_trials": int(len(succ)),
        "success_rate": success_rate,
        "boundary_assisted_rate": boundary_assisted,
        "pure_encirclement_rate": pure_encirclement,
        "mean_capture_time": float(np.mean(times)) if times else float("nan"),
        "timeout_rate": 1.0 - success_rate,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60, help="每种空域尺度的蒙特卡洛样本数")
    ap.add_argument("--out", default="data/outputs/capability_boundary")
    args = ap.parse_args()

    half_extents = [250.0, 400.0, 500.0, 600.0, 800.0, 1000.0, 1500.0, 2000.0]
    rows = [evaluate_scale(h, args.trials) for h in half_extents]

    df = pd.DataFrame(rows)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "capability_boundary.csv", index=False, encoding="utf-8-sig")
    (out_dir / "capability_boundary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 图：成功率随横向空域尺度变化 + 成功样本中"边界辅助/纯包围"占比
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    x = df["lateral_half_extent_m"].to_numpy()
    y = df["success_rate"].to_numpy() * 100
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(x, y, "o-", color="tab:blue", lw=2, label="围捕成功率")
    ax.fill_between(x, 0, y, color="tab:blue", alpha=0.12)
    ax.axhline(100, color="tab:gray", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("处置区横向半宽 (m)  [方形域 2h×2h]")
    ax.set_ylabel("成功率 / 占比 (%)")
    ax.set_title("协同围捕的成功对有限处置区边界的依赖（ρ=0.65, n=6）")
    ax.grid(alpha=0.3)

    # 成功样本中边界辅助占比（叠加在成功率线上）
    ba = df["boundary_assisted_rate"].to_numpy() * 100
    # 只在存在成功样本的尺度才有意义；否则置 NaN 让该段不连线
    ba = np.where(np.array(df["success_rate"].to_numpy() > 0) & ~np.isnan(ba), ba, np.nan)
    ax.plot(x, ba, "s--", color="tab:orange", lw=1.8, label="成功中·边界规避层辅助占比")

    # 标注各点成功率与边界辅助
    for xi, yi, bi in zip(x, y, ba):
        ax.annotate(f"{yi:.0f}%", (xi, yi), textcoords="offset points", xytext=(0, 7), fontsize=9)
        if not np.isnan(bi):
            ax.annotate(f"边界{bi:.0f}%", (xi, max(bi, yi * 0.5)), textcoords="offset points",
                        xytext=(0, -14), fontsize=8, color="tab:orange")

    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_capability_boundary.png", dpi=150)
    plt.close(fig)

    print(df[["lateral_half_extent_m", "success_rate", "boundary_assisted_rate",
              "pure_encirclement_rate", "mean_capture_time", "timeout_rate"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n输出 ->", out_dir)


if __name__ == "__main__":
    main()
