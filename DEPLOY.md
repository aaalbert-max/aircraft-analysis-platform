# 长期部署指南

本平台可在本机/局域网跑（`web/start.ps1`），要做到"长期、公开、不依赖本机常开"，推荐下面两种。

## 方式 A：公开成果站（推荐，免费永久，GitHub Pages）

效果：一个公开网址，任何人打开都能看到全部成果（图/表/KPI/工程设计建议）+ 预置典型算例的三维回放。
局限：GitHub Pages 是纯静态，不能实时拖参数跑仿真（回放为预计算）。

步骤：
1. 在 GitHub 新建一个空仓库（如 `aircraft-analysis-platform`）。
2. 关联并推送：
   ```powershell
   git remote add origin https://github.com/<你>/<仓库名>.git
   git push -u origin main
   ```
3. 仓库 Settings → Pages → Source 选 **GitHub Actions**。
4. 推送后，`.github/workflows/pages.yml` 会自动把 `docs/` 发布为 Pages。
5. 地址形如 `https://<你>.github.io/<仓库名>/`。

> 更新成果：改完 `scripts/run_design_study.py` 或 `scripts/build_static_site.py` 后，重新生成 `docs/`（见下），再提交推送即可。

## 方式 B：完整交互平台（真后端，Render 免费档）

效果：公开网址，能实时拖速度比/数量/种子跑仿真。Render 免费实例会在无访问时休眠（冷启动稍慢），有 750 小时/月限额。

步骤：
1. Root 目录有 `render.yaml` + `requirements.txt` 已就绪。
2. 在 Render → New → Blueprint → 连接该 GitHub 仓库 → 一键部署。
3. 完成后得到 `https://<app>.onrender.com`。

> 如走 Fly.io：`fly launch` 并选 Python；启动命令 `python web/app.py`，监听 `$PORT`。

## 更新成果数据

```powershell
$env:PYTHONPATH='src'
python scripts/run_design_study.py   # 重算设计研究（可选）
python scripts/update_design_artifacts.py   # 补充摘要/交叉验证/平滑曲线
python scripts/build_static_site.py   # 生成 docs/ 静态数据与回放
```
