"""协同围捕仿真参数与工况预设。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class InterceptionConfig:
    """协同围捕场景参数（长度单位统一为米，速度 m/s）。"""

    n_pursuers: int = 6
    v_evader: float = 10.0          # 目标最大速度 m/s
    rho: float = 0.65               # 速度比 v_pursuer / v_evader
    capture_radius: float = 15.0    # 捕获半径 m
    required_captors: int = 2       # 判定围捕成功所需同时进入的拦截机数
    r_detect: float = 80.0          # 探测/威胁感知距离 m
    d_safe: float = 50.0            # 机间软斥力启用距离 m
    boundary_margin: float = 30.0   # 空域软边界裕度 m
    k_e_rep: float = 500.0          # 逃逸机对拦截机斥力增益
    k_bound: float = 200.0          # 逃逸机边界斥力增益
    k_att: float = 4.0              # 人工势场引力增益
    k_rep: float = 10.0             # 人工势场机间斥力增益
    dt: float = 0.033               # 仿真步长 s
    tau: float = 0.5                # 一阶速度响应时间常数 s
    max_time: float = 120.0         # 最大围捕时间 s
    role_angle_deg: float = 60.0    # 角色划分角度阈值 deg
    lead_min: float = 0.5           # 引导提前时间下限 s
    lead_max: float = 3.0           # 引导提前时间上限 s
    w_att: float = 1.0              # 引力权重
    w_rep_interceptor: float = 0.3  # 拦截者斥力权重
    w_rep_herder: float = 0.7       # 驱赶者斥力权重
    xmin: float = -250.0
    xmax: float = 250.0
    ymin: float = -150.0
    ymax: float = 150.0
    init_scale: float = 60.0        # 初始拦截机散布半径 m（正对/侧向分布）
    evader_init: Tuple[float, float] = (0.0, 0.0)  # 目标初始位置
    seed: int = 2000                # 初始分布随机种子

    @property
    def v_pursuer(self) -> float:
        return self.rho * self.v_evader

    def validate(self) -> None:
        if self.n_pursuers < 2:
            raise ValueError("至少需要 2 架拦截机才能形成围捕判定")
        if not 0 < self.rho < 1:
            raise ValueError("速度比 rho 需在 (0, 1) 区间（拦截机速度劣势）")
        if self.capture_radius <= 0 or self.d_safe <= 0:
            raise ValueError("捕获半径与安全距离需为正")


# 本项目标称工况：10 m/s 目标、6.5 m/s 拦截机
NOMINAL = InterceptionConfig(
    v_evader=10.0,
    rho=0.65,
    capture_radius=15.0,
    r_detect=80.0,
    d_safe=50.0,
    boundary_margin=30.0,
    max_time=120.0,
    init_scale=80.0,
)

# 原方案参考工况：20 m/s 目标、13 m/s 拦截机（速度比同样 0.65）
REFERENCE = InterceptionConfig(
    v_evader=20.0,
    rho=0.65,
    capture_radius=15.0,
    r_detect=80.0,
    d_safe=50.0,
    boundary_margin=30.0,
    max_time=120.0,
    init_scale=80.0,
)

# Robotarium 演示尺度：把参考工况缩放到场地 3.0m x 1.8m
CAPTURE = InterceptionConfig(
    n_pursuers=6,
    v_evader=0.15,
    rho=0.65,
    capture_radius=0.19,
    r_detect=0.48,
    d_safe=0.30,
    boundary_margin=0.18,
    xmin=-1.5,
    xmax=1.5,
    ymin=-0.9,
    ymax=0.9,
    max_time=90.0,
    init_scale=0.9,
)
