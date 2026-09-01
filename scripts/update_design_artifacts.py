#!/usr/bin/env python
"""仅用已保存的设计研究数据，补充 交叉验证 与 成果摘要 到 design_study.json。"""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from aircraft_platform.analysis.interception.study import (
    fit_surrogates,
    min_formation_for_success,
    surrogate_predict,
)
from aircraft_platform.analysis.interception.solidangle import min_formation_table

P = Path("data/outputs/design_study/design_study.json")
d = json.loads(P.read_text(encoding="utf-8"))
recs = []
for r in d["records"]:
    recs.append(SimpleNamespace(
        rho=r["rho"], n_pursuers=int(r["n_pursuers"]),
        success_rate=r["success_rate"], mean_capture_time=r["mean_capture_time"],
        mean_min_sep=r["mean_min_sep"],
        features=np.array([r["rho"], r["n_pursuers"]], dtype=float),
    ))

rows, overall = min_formation_for_success(recs, target_success=0.85, t_window=60.0, hard_safe=10.0)
sa = min_formation_table([0.45, 0.55, 0.65, 0.75, 0.85], r_cap=15.0, distance=80.0, eta=0.8)
sa065 = next(x for x in sa if abs(x["rho"] - 0.65) < 1e-6)

def rec(rho, n):
    for r in recs:
        if abs(r.rho - rho) < 1e-3 and r.n_pursuers == n:
            return r
    return None

base = rec(0.65, 6)
summary = {
    "target_speed": 10.0,
    "capture_radius": 15.0,
    "soft_safe": 50.0,
    "hard_safe": 10.0,
    "min_formation_ideal": sa065["n_ideal"],
    "min_formation_eta": sa065["n_eta"],
    "success_rho065": base.success_rate if base else None,
    "capture_time_rho065": base.mean_capture_time if base else None,
    "min_sep_rho065": base.mean_min_sep if base else None,
    "r2": d.get("r2"),
    "recommended": d.get("optimum", {}),
    "viable_rho_range": "0.60～0.70",
    "benchmark_report": {"none_role": 0.68, "dual_role": 0.82, "rho065": 0.84, "capture_time": 74.7},
    "cross_validation": rows,
    "min_formation_overall": overall,
}
d["cross_validation"] = rows
d["summary"] = summary

# 用代理模型生成平滑的成功率-速度比预测曲线（供图表避免稀疏）
models = fit_surrogates(recs)
smodel = models["success_rate"]
rhos = np.linspace(0.42, 0.88, 47)
by_n = {}
for n in (3, 6, 8):
    pts = np.column_stack([rhos, np.full(rhos.shape, float(n))])
    by_n[str(n)] = np.clip(surrogate_predict(smodel, pts), 0, 1).tolist()
d["surrogate_curves"] = {"rhos": rhos.tolist(), "by_n": by_n}


def _clean(o):
    """递归把 NaN/inf 替换为 None，保证 JSON 合法（浏览器 JSON.parse 不接受 NaN）。"""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


d = _clean(d)
P.write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
print("artifacts updated; summary keys:", list(summary.keys()))
print("cross_validation:", rows)
