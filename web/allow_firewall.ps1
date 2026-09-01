# 放行 Windows 防火墙：允许其他设备访问本平台(TCP 8000)
# 以管理员运行：右键"以管理员身份运行"，或双击(会自动弹UAC提权)
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Start-Process powershell -Verb RunAs -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    exit
}
New-NetFirewallRule -DisplayName "智能协同反制平台 8000" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8000 -Profile Any | Out-Null
Write-Host "已完成：放行 TCP 8000。现在同一网络的其他人可访问 http://<你的局域网IP>:8000"
