# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""几何/角度小工具(自包含,无外部依赖)。"""
from __future__ import annotations

import math


def wrap_angle(a: float) -> float:
    """归一化到 (-pi, pi]。"""
    return math.atan2(math.sin(a), math.cos(a))


def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    """四元数 (x,y,z,w) → 绕 Z 的 yaw。"""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def parse_via(text: str) -> list[tuple[float, float]]:
    """'4.4,3.0; 2.1,3.0' -> [(4.4, 3.0), (2.1, 3.0)];空串 -> []。"""
    points: list[tuple[float, float]] = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        xs, ys = chunk.split(",")
        points.append((float(xs), float(ys)))
    return points
