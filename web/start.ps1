# 一键启动智能协同反制分析平台（成果展示）
# 用法：在项目根目录 PowerShell 中运行  powershell -ExecutionPolicy Bypass -File web/start.ps1
$root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $root "src"

# 优先使用 PATH 上的 python；否则回退到本机 Codex 运行时自带的 python
$py = "python"
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    $cand = "C:\Users\skmmk\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    if (Test-Path $cand) { $py = $cand }
}

Write-Host "启动中... 局域网访问地址会打印在下方。"
& $py (Join-Path $PSScriptRoot "app.py")
