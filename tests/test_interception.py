"""协同围捕引擎的冒烟测试。"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aircraft_platform.analysis.interception import NOMINAL, run_single, run_and_aggregate


def test_single_trial_runs():
    result = run_single(NOMINAL, seed=2000)
    assert result.final_positions.shape[0] == 2
    assert result.best_second_distance >= 0


def test_batch_aggregate():
    seeds = [2000, 2001, 2002]
    metrics = run_and_aggregate(NOMINAL, seeds)
    assert metrics.trials == 3
    assert 0.0 <= metrics.success_rate <= 1.0
    assert metrics.mean_capture_time >= 0
