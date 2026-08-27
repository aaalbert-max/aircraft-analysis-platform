"""设计研究 -> 代理模型 -> 参数寻优 -> 报告图与工程结论。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import InterceptionConfig, NOMINAL
from .solidangle import analyze, min_formation_table
from .study import (
    DesignRecord,
    fit_surrogate,
    optimize_on_surrogate,
    split_train_test,
    surrogate_r2,
    sweep_design_space,
)


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _success_surface(model, rhos, ns, grid=60):
    """用代理模型在 (rho, n) 网格上生成成功率预测曲面。"""
    xx, yy = np.meshgrid(np.linspace(rhos[0], rhos[-1], grid), np.linspace(ns[0], ns[-1], grid))
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    z = np.clip(model.predict(pts), 0.0, 1.0).reshape(xx.shape)
    return xx, yy, z


def plot_success_surface(plt, model, rhos, ns, records, out_path: Path):
    """图1：成功率设计空间面 + 等值线 + 立体角最小编队曲线。"""
    xx, yy, z = _success_surface(model, rhos, ns)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    levels = np.linspace(0, 1.0, 21)
    cs = ax.contourf(xx, yy, z, levels=levels, cmap="viridis", alpha=0.85)
    ax.contour(xx, yy, z, levels=[0.5, 0.7, 0.9], colors="k", linewidths=0.8)
    for r in records:
        ax.scatter(r.rho, r.n_pursuers, c=[r.success_rate], cmap="viridis",
                   vmin=0, vmax=1, s=45, edgecolor="k", zorder=5)
    # 立体角最小编队规模曲线（η=0.8）
    sa_rhos = np.linspace(rhos[0], rhos[-1], 40)
    sa_n = [min_formation_table([rho], r_cap=15.0, distance=80.0, eta=0.8)[0]["n_eta"] for rho in sa_rhos]
    ax.plot(sa_rhos, sa_n, "r--", linewidth=2, label="三维覆盖最小编队 (η=0.8)")
    ax.scatter(sa_rhos, sa_n, color="r", s=12, zorder=6)
    fig.colorbar(cs, ax=ax, label="捕获成功率")
    ax.set_xlabel("速度比 ρ")
    ax.set_ylabel("拦截机数量 n")
    ax.set_title("协同围捕成功率设计空间（灰点=蒙特卡洛，曲面=代理模型）")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_success_vs_rho(plt, records, out_path: Path):
    """图2：固定编队规模下的成功率-速度比（倒U形）与捕获时间。"""
    fig, ax1 = plt.subplots(figsize=(8, 5))
    markers = {3: "o", 4: "s", 6: "^", 8: "D", 9: "v"}
    ns = sorted({r.n_pursuers for r in records})
    for n in ns:
        subset = [r for r in records if r.n_pursuers == n]
        subset.sort(key=lambda r: r.rho)
        ax1.plot([r.rho for r in subset], [r.success_rate for r in subset],
                 marker=markers.get(n, "X"), label=f"n={n}")
    ax1.set_xlabel("速度比 ρ")
    ax1.set_ylabel("捕获成功率", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="center left")

    ax2 = ax1.twinx()
    ax2.set_ylabel("平均捕获时间 (s)", color="tab:red")
    for n in ns:
        subset = [r for r in records if r.n_pursuers == n and np.isfinite(r.mean_capture_time)]
        subset.sort(key=lambda r: r.rho)
        ax2.plot([r.rho for r in subset], [r.mean_capture_time for r in subset],
                 "--", color="tab:red", alpha=0.5)
    ax2.tick_params(axis="y", labelcolor="tab:red")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_min_formation(plt, out_path: Path):
    """图3：不同速度比下三维覆盖最小编队规模。"""
    rhos = np.linspace(0.4, 0.9, 50)
    ideal, with_eta = [], []
    for rho in rhos:
        row = min_formation_table([rho], r_cap=15.0, distance=80.0, eta=0.8)[0]
        ideal.append(row["ideal"])
        with_eta.append(row["n_eta"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(rhos, ideal, "o-", label="理想 4π/Ω")
    ax.plot(rhos, with_eta, "s-", label="考虑填充效率 η=0.8")
    ax.set_xlabel("速度比 ρ")
    ax.set_ylabel("最小编队规模 n_min")
    ax.set_title("三维覆盖立体角 -> 最小编队规模")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_surrogate_fit(plt, records, out_path: Path):
    """图4：代理模型拟合质量（预测 vs 实测）。"""
    model = fit_surrogate(records)
    X = np.array([r.features for r in records], dtype=float)
    y = np.array([r.success_rate for r in records], dtype=float)
    pred = np.clip(model.predict(X), 0, 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y, pred, c="tab:blue", edgecolor="k")
    ax.plot([0, 1], [0, 1], "r--", label="y=x")
    r2 = surrogate_r2(model, records)
    ax.set_xlabel("实测捕获成功率")
    ax.set_ylabel("代理模型预测")
    ax.set_title(f"代理模型拟合（R²={r2:.3f}）")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_study_and_report(
    base: InterceptionConfig = NOMINAL,
    rhos: Sequence[float] = (0.45, 0.55, 0.65, 0.75, 0.85),
    ns: Sequence[int] = (3, 4, 6, 8),
    n_trials: int = 12,
    out_dir: Optional[Path] = None,
) -> Dict:
    """完整运行：采样 -> 代理模型 -> 寻优 -> 出图 -> 返回结论字典。"""
    plt = _setup_matplotlib()
    out_dir = Path(out_dir) if out_dir else Path("data/outputs/design_study")
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(2000, 2000 + n_trials))

    records = sweep_design_space(base, rhos, ns, seeds)
    model = fit_surrogate(records)
    r2 = surrogate_r2(model, records)
    train, _test = split_train_test(records, test_frac=0.25, seed=1)
    r2_train = surrogate_r2(model, train)
    rho_star, n_star, pred_star = optimize_on_surrogate(model, (0.4, 0.9), (3, 12))

    valid_records = [r for r in records if np.isfinite(r.mean_capture_time)]
    best_rec = max(valid_records, key=lambda r: r.success_rate)

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_success_surface(plt, model, rhos, ns, records, fig_dir / "fig1_success_surface.png")
    plot_success_vs_rho(plt, records, fig_dir / "fig2_success_vs_rho.png")
    plot_min_formation(plt, fig_dir / "fig3_min_formation.png")
    plot_surrogate_fit(plt, records, fig_dir / "fig4_surrogate_fit.png")

    table = min_formation_table(list(rhos), r_cap=base.capture_radius, distance=80.0, eta=0.8)
    records_json = [r.to_dict() for r in records]
    (out_dir / "design_study.json").write_text(
        json.dumps({"records": records_json, "r2": r2, "r2_train": r2_train,
                    "optimum": {"rho": rho_star, "n": n_star, "pred_success": pred_star}},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    conclusion = build_conclusion(base, records, table, rho_star, n_star, pred_star,
                                  r2, best_rec)
    (out_dir / "engineering_conclusion.md").write_text(conclusion, encoding="utf-8")
    return {
        "records": records,
        "r2": r2,
        "r2_train": r2_train,
        "optimum": {"rho": rho_star, "n": n_star, "pred_success": pred_star},
        "table": table,
        "best_record": best_rec,
        "conclusion": conclusion,
        "figure_dir": fig_dir,
    }


def build_conclusion(base, records, table, rho_star, n_star, pred_star, r2, best_rec) -> str:
    """根据结果生成工程结论。"""
    lines: List[str] = []
    lines.append("# 协同围捕系统设计研究结论\n")
    lines.append(f"## 目标工况\n- 目标最大速度 {base.v_evader:.1f} m/s，拦截机速度比 ρ 取 {base.rho:.2f}\n"
                 f"- 捕获半径 {base.capture_radius:.1f} m，软安全 {base.d_safe:.1f} m，机间硬安全 10 m\n")
    lines.append("\n## 三维覆盖立体角 -> 最小编队规模\n")
    lines.append("| 速度比 | sinα | 覆盖立体角 (sr) | 4π/Ω | 理想 | 修正 η=0.8 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in table:
        lines.append(f"| {row['rho']:.2f} | {row['sin_alpha']:.3f} | {row['omega_sr']:.3f} "
                     f"| {row['ideal']:.2f} | {row['n_ideal']} | {row['n_eta']} |")
    lines.append("\n## 设计空间结论\n")
    lines.append(f"- 代理模型拟合 R² = {r2:.3f}（{len(records)} 个设计点）")
    lines.append(f"- 推荐设计点：速度比 ρ≈{rho_star:.3f}，拦截机数 n≈{n_star}，"
                 f"预测成功率 ≈ {pred_star*100:.1f}%")
    lines.append(f"- 实测最优设计点：ρ={best_rec.rho:.2f}, n={best_rec.n_pursuers}，"
                 f"成功率 {best_rec.success_rate*100:.1f}%，平均捕获时间 "
                 f"{best_rec.mean_capture_time:.1f}s，平均最小机间距 {best_rec.mean_min_sep:.1f}m")
    lines.append("\n## 重要工程提醒\n")
    minsep_all = [r.mean_min_sep for r in records if np.isfinite(r.mean_min_sep)]
    if minsep_all:
        worst = min(minsep_all)
        if worst < 10.0:
            lines.append(f"- 人工势场引导层本身无法保证机间 ≥10 m 硬安全："
                         f"部分设计点平均最小机间距低至 {worst:.1f} m。"
                         f"实际系统需叠加执行层安全屏障（如避碰证书/速度约束），"
                         f"平台已将其列为安全指标用于筛选设计点。")
        else:
            lines.append(f"- 各设计点平均最小机间距均 ≥10 m，引导层安全性可接受。")
    return "\n".join(lines) + "\n"
