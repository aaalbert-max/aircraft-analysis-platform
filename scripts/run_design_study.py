#!/usr/bin/env python
"""完整设计研究：扫描 -> 随机森林代理模型 -> 最小编队寻优 -> 出图 -> 工程结论。

用法（在项目根目录）:
    PYTHONPATH=src python scripts/run_design_study.py
"""
from pathlib import Path

from aircraft_platform.analysis.interception.config import NOMINAL
from aircraft_platform.analysis.interception.report import run_study_and_report
from aircraft_platform.analysis.interception.solidangle import min_formation_table
from aircraft_platform.analysis.interception.study import (
    min_formation_for_success,
)

RHOS = [0.45, 0.55, 0.65, 0.75, 0.85]
NS = [3, 6, 8]
N_TRIALS = 16
TARGET_SUCCESS = 0.9
T_WINDOW = 60.0

if __name__ == "__main__":
    out_dir = Path("data/outputs/design_study")
    res = run_study_and_report(
        base=NOMINAL, rhos=RHOS, ns=NS, n_trials=N_TRIALS, out_dir=out_dir
    )

    # 交叉验证：仿真所需最小编队 vs 三维覆盖立体角最小编队
    rows, overall = min_formation_for_success(
        res["records"],
        target_success=TARGET_SUCCESS,
        t_window=T_WINDOW,
        hard_safe=NOMINAL.hard_safe,
    )
    sa = min_formation_table(RHOS, r_cap=15.0, distance=80.0, eta=0.8)
    lines = ["\n\n## 仿真最小编队 vs 三维覆盖立体角（交叉验证）\n",
             "| 速度比 | 仿真达标n | 立体角理想 | 立体角η=0.8 |",
             "| --- | --- | --- | --- |"]
    for r in rows:
        near = min(sa, key=lambda x: abs(x["rho"] - r["rho"]))
        nr = "NA" if r["n_required"] != r["n_required"] else f"{r['n_required']:.0f}"
        lines.append(f"| {r['rho']:.2f} | {nr} | {near['n_ideal']} | {near['n_eta']} |")
    lines.append(f"\n全局最低达标编队：rho≈{overall[0]:.3f}, n≈{overall[1]:.0f}")
    conclusion = res["conclusion"] + "\n".join(lines) + "\n"
    (out_dir / "engineering_conclusion.md").write_text(conclusion, encoding="utf-8")
    print("r2=%.3f  opt rho=%.3f n=%d  best-min-n rho=%.3f n=%.0f" % (
        res["r2"], res["optimum"]["rho"], res["optimum"]["n"], overall[0], overall[1]))
    print("charts ->", res["figure_dir"])
