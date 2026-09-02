#!/usr/bin/env python
"""构建 GitHub Pages 静态成果站：聚合设计数据 + 预计算若干典型算例回放。

输出到 docs/（GitHub Pages 从 /docs 或根目录发布）：
- docs/data/app.json   （KPI/表/建议/记录 + replays 回放数据）
- docs/figures/*.png   （报告图，从 design_study 拷贝）
"""
import json
import shutil
from pathlib import Path

import numpy as np

from aircraft_platform.analysis.interception.config import InterceptionConfig
from aircraft_platform.analysis.interception.simulator import simulate_trial
from aircraft_platform.analysis.interception.solidangle import min_formation_table

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "outputs" / "design_study"
DOCS = ROOT / "docs"
FIGS = DOCS / "figures"
DATA = DOCS / "data"
FIGS.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

# 1) 聚合设计研究数据
d = json.loads((SRC / "design_study.json").read_text(encoding="utf-8"))
d["solid_angle"] = min_formation_table([0.45, 0.55, 0.65, 0.75, 0.85], r_cap=15.0, distance=80.0, eta=0.8)
d["recommendation"] = d.get("optimum", {})
d["criteria"] = {"target_success": 0.85, "t_window": 60.0, "hard_safe": 10.0}

# 2) 预计算典型算例回放
PRESETS = [
    {"name": "推荐设计", "rho": 0.65, "n": 6, "seed": 14, "scenario": "ring", "init_scale": 55},
    {"name": "低速度比", "rho": 0.45, "n": 6, "seed": 0, "scenario": "patrol"},
    {"name": "高速度比", "rho": 0.85, "n": 6, "seed": 67, "scenario": "patrol"},
    {"name": "少机编组", "rho": 0.65, "n": 3, "seed": 2, "scenario": "patrol"},
]
replays = {}
for p in PRESETS:
    name = p["name"]
    cfg = InterceptionConfig(
        rho=p["rho"], n_pursuers=p["n"], seed=p["seed"], dim=3,
        scenario=p["scenario"], init_scale=p.get("init_scale", 80.0),
    )
    res = simulate_trial(cfg, seed=p["seed"])
    step = max(1, len(res.evader_traj) // 240)
    idx = slice(None, None, step)
    replays[name] = {
        "rho": p["rho"], "n": p["n"], "seed": p["seed"], "scenario": p["scenario"],
        "captured": bool(res.captured),
        "capture_time": res.capture_time,
        "best_second": res.best_second_distance,
        "min_sep": res.min_sep,
        "arena": [cfg.xmin, cfg.xmax, cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax],
        "capture_radius": cfg.capture_radius,
        "frames": {
            "t": [i * cfg.dt * step for i in range(len(np.asarray(res.evader_traj)[idx]))],
            "evader": np.asarray(res.evader_traj)[idx].tolist(),
            "pursuers": np.transpose(np.asarray(res.pursuer_traj)[idx], (0, 2, 1)).tolist(),
        },
    }
d["replays"] = replays

(DATA / "app.json").write_text(json.dumps(d, ensure_ascii=False, allow_nan=False), encoding="utf-8")

# 3) 拷贝图
for f in (SRC / "figures").glob("*.png"):
    shutil.copy2(f, FIGS / f.name)

print("static site data written:", DATA / "app.json", f"({(DATA/'app.json').stat().st_size} bytes)")
print("replays:", list(replays.keys()))
print("figures copied:", len(list(FIGS.glob('*.png'))))
