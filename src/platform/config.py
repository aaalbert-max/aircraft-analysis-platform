"""平台配置加载。

从 config/settings.yaml 读取配置；若不存在则使用默认值。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class PlatformConfig:
    """平台运行时配置。"""

    app_name: str = "智能分析平台"
    host: str = "127.0.0.1"
    port: int = 8000
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    output_dir: str = "data/outputs"


def load_config() -> PlatformConfig:
    """加载配置。当前使用默认值，后续可扩展为读取 YAML。"""
    return PlatformConfig()
