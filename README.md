# 智能分析平台（飞行器方向）

> 项目状态：**脚手架搭建阶段**，正等待需求文档与理论核心材料。

本仓库用于构建面向飞行器课题（"揭榜挂帅"）的智能分析平台。平台目标是打通
**数据接入 → 数据治理 → 理论/算法分析 → 结果可视化与报告** 的完整链路，
并承载异常检测、故障诊断、性能预测、参数优化等典型分析任务。

## 当前进度

- [x] 初始化 Git 仓库（默认分支 `main`）
- [x] 搭建项目骨架与目录结构
- [x] 编写平台架构初稿（`docs/architecture/architecture.md`）
- [ ] 需求文档（待放入 `docs/requirements/`）
- [ ] 理论核心（待放入 `docs/theory/`）
- [ ] 数据接入与治理模块
- [ ] 分析/算法模块
- [ ] Web 可视化平台

## 目录结构

```text
平台搭建/
├── README.md                  # 项目说明（本文件）
├── pyproject.toml             # Python 项目与依赖声明
├── .gitignore
├── docs/
│   ├── requirements/          # 需求文档（待放入）
│   ├── theory/                # 理论核心（待放入）
│   └── architecture/          # 架构设计文档
├── src/
│   └── aircraft_platform/
│       ├── ingest/            # 数据接入
│       ├── governance/        # 数据治理
│       ├── analysis/          # 分析/算法
│       ├── models/            # 模型管理
│       └── serve/             # 服务/API
├── config/
│   └── settings.example.yaml  # 配置模板
├── data/
│   ├── raw/                   # 原始数据（不入库）
│   ├── processed/             # 处理后数据（不入库）
│   └── outputs/               # 分析输出/报告（不入库）
├── scripts/                   # 运维/构建脚本
└── tests/                     # 测试
```

## 建议技术栈

| 层 | 建议选型 |
| --- | --- |
| 后端/分析 | Python 3.12（numpy / pandas / scipy / scikit-learn / PyTorch） |
| 服务接口 | FastAPI + Uvicorn |
| 前端可视化 | React + Vite + ECharts（或先用 Streamlit 快速原型） |
| 数据存储 | Parquet / SQLite（轻量），后续可按需升级 PostgreSQL + MinIO |
| 测试 | pytest |

> 说明：当前 Codex 运行环境自带的 Python 仅覆盖办公文档类依赖，尚未安装上述科研/Web
> 依赖。需求确认后再统一安装，避免提前锁定不必要的包。

## 如何添加需求与理论

把材料直接放进对应目录即可，我会据此落地具体功能：

- 需求文档 → `docs/requirements/`
- 理论核心 → `docs/theory/`

## 待你提供的信息

请重点说明：数据来源与格式、分析对象（气动/结构/控制/轨迹/遥测等）、是否需实时处理、
部署形态（单机/局域网/云端）、主要用户角色与权限。这些决定架构收敛方向。


## 平台使用

依赖：Python 3.12 + numpy / scipy / scikit-learn / matplotlib / pyarrow。

```powershell
pip install scipy scikit-learn matplotlib pyarrow
```

### 1. 设计研究（扫描 -> 代理模型 -> 最小编队寻优 -> 报告图）
```powershell
$env:PYTHONPATH='src'
python scripts/run_design_study.py
# 结果输出到 data/outputs/design_study/（图 + 指标 + 工程结论）
```

### 2. 可交互 Web 演示
```powershell
$env:PYTHONPATH='src'
python web/app.py
# 浏览器打开 http://127.0.0.1:8000
# 可拖速度比/拦截机数/种子，实时看协同围捕动画与成功率曲线
```

### 核心模块
- `src/aircraft_platform/analysis/interception/solidangle.py` 三维覆盖立体角 -> 最小编队
- `.../simulator.py` 协同围捕仿真引擎（巡逻场景 + 执行层安全屏障）
- `.../study.py` 设计空间采样、代理模型、约束寻优
- `.../report.py` 设计研究图与工程结论
