"""智能协同反制分析平台 —— 可交互 Web 演示（标准库 HTTP 服务器，无需额外依赖）。

用法:
    python web/app.py
然后在浏览器打开 http://127.0.0.1:8000
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import urllib.parse

import numpy as np

ROOT = Path(__file__).resolve().parent
INDEX = (ROOT / "index.html").read_text(encoding="utf-8")
DESIGN_PATH = Path(__file__).resolve().parents[1] / "data" / "outputs" / "design_study" / "design_study.json"
DESIGN = json.loads(DESIGN_PATH.read_text(encoding="utf-8")) if DESIGN_PATH.exists() else {"records": []}
FIGS_DIR = Path(__file__).resolve().parents[1] / "data" / "outputs" / "design_study" / "figures"


def run_sim(payload):
    from aircraft_platform.analysis.interception.config import InterceptionConfig
    from aircraft_platform.analysis.interception.simulator import simulate_trial

    cfg = InterceptionConfig(
        rho=float(payload.get("rho", 0.65)),
        n_pursuers=int(payload.get("n", 6)),
        seed=int(payload.get("seed", 0)),
        dim=int(payload.get("dim", 3)),
        scenario="patrol",
    )
    res = simulate_trial(cfg, seed=cfg.seed)
    # 降采样以减小传输
    step = max(1, len(res.evader_traj) // 240)
    idx = slice(None, None, step)
    evader = np.asarray(res.evader_traj)[idx].tolist()
    purs = np.transpose(np.asarray(res.pursuer_traj)[idx], (0, 2, 1)).tolist()
    dist = np.asarray(res.pursuer_dist_history)[idx].tolist()
    return {
        "rho": cfg.rho,
        "n": cfg.n_pursuers,
        "seed": cfg.seed,
        "captured": bool(res.captured),
        "capture_time": res.capture_time,
        "best_second": res.best_second_distance,
        "min_sep": res.min_sep,
        "arrival_variance": res.arrival_variance,
        "arena": [cfg.xmin, cfg.xmax, cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax],
        "capture_radius": cfg.capture_radius,
        "frames": {"t": [i * cfg.dt * step for i in range(len(evader))],
                   "evader": evader, "pursuers": purs, "dist": dist},
    }


def get_layout(payload):
    """仅生成初始布局（拦截机/目标位置），不跑全程，用于秒开预览。"""
    import numpy as np

    from aircraft_platform.analysis.interception.config import InterceptionConfig
    from aircraft_platform.analysis.interception.simulator import _spawn_scenario

    cfg = InterceptionConfig(
        rho=float(payload.get("rho", 0.65)),
        n_pursuers=int(payload.get("n", 6)),
        seed=int(payload.get("seed", 0)),
        dim=int(payload.get("dim", 3)),
        scenario="patrol",
    )
    rng = np.random.default_rng(cfg.seed)
    purs, evader, heading = _spawn_scenario(cfg, rng)
    return {
        "rho": cfg.rho,
        "n": cfg.n_pursuers,
        "seed": cfg.seed,
        "dim": cfg.dim,
        "heading": float(heading),
        "arena": [cfg.xmin, cfg.xmax, cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax],
        "capture_radius": cfg.capture_radius,
        "pursuers_init": np.asarray(purs).T.tolist(),
        "evader_init": np.asarray(evader).tolist(),
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, INDEX.encode("utf-8"), "text/html; charset=utf-8")
        elif path.startswith("/figures/"):
            name = path.rsplit("/", 1)[-1]
            f = FIGS_DIR / name
            if f.exists() and f.suffix.lower() in (".png", ".jpg", ".svg"):
                ctype = "image/svg+xml" if f.suffix.lower() == ".svg" else "image/png"
                self._send(200, f.read_bytes(), ctype)
            else:
                self._send(404, b'{"error":"figure not found"}')
        elif path == "/api/design":
            from aircraft_platform.analysis.interception.solidangle import min_formation_table
            payload = dict(DESIGN)
            payload["solid_angle"] = min_formation_table(
                [0.45, 0.55, 0.65, 0.75, 0.85], r_cap=15.0, distance=80.0, eta=0.8
            )
            payload["recommendation"] = DESIGN.get("optimum", {})
            payload["criteria"] = {
                "target_success": 0.85,
                "t_window": 60.0,
                "hard_safe": 10.0,
            }
            self._send(200, json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"))
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/api/simulate":
            length = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
                data = run_sim(payload)
                self._send(200, json.dumps(data, ensure_ascii=False).encode("utf-8"))
            except Exception as exc:  # noqa: BLE001
                self._send(500, json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8"))
        elif path == "/api/layout":
            length = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
                self._send(200, json.dumps(get_layout(payload), ensure_ascii=False).encode("utf-8"))
            except Exception as exc:  # noqa: BLE001
                self._send(500, json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8"))
        else:
            self._send(404, b'{"error":"not found"}')

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    import os
    import socket
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    host = os.environ.get("PLATFORM_HOST", "0.0.0.0")
    port = int(os.environ.get("PLATFORM_PORT", "8000"))
    print("智能协同反制分析平台（成果展示）")
    print(f"  本机访问:   http://127.0.0.1:{port}")
    try:
        ips = sorted({
            a for a in socket.gethostbyname_ex(socket.gethostname())[2]
            if not a.startswith("127.")
        })
    except Exception:
        ips = []
    for ip in ips:
        print(f"  局域网访问: http://{ip}:{port}   (发给同一网络的其他人)")
    if not ips:
        print("  (未能自动探测局域网 IP，可在 Windows 用 ipconfig 查看本机 IPv4)")
    print("  如需公网访问，请用 cloudflared / ngrok 隧道 或 部署到公网主机。")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
