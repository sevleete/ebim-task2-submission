# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""机器人常量与关节读取小工具(自包含,值就地定义,不 import 外部模块)。

—— 关节名、夹爪常量、home/预抓取姿态都从原始文件复制成字面量,拷走即用。
"""
from __future__ import annotations

import math

# 14 个手臂关节 + 脊柱 + 两个夹爪驱动关节(与 bridge / 数据契约一致)
LEFT = [f"left_fr3v2_joint{i}" for i in range(1, 8)]
RIGHT = [f"right_fr3v2_joint{i}" for i in range(1, 8)]
SPINE = "franka_spine_vertical_joint"
LG = "left_right_finger_joint"
RG = "right_right_finger_joint"
GC = 0.8   # 夹爪全闭对应的驱动关节弧度;open_fraction = clip(1 - rad/GC, 0, 1)

# home 姿态 = R 键回的 ready 姿(复制自 bridge_core.py:ARM_READY_POSE)
ARM_READY = [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854]
HOME_LEFT = list(ARM_READY)
HOME_RIGHT = list(ARM_READY)

# 数据采集起始姿(pad 预抓取;复制自 data_collect_prepare.py:LEFT_Q/RIGHT_Q)
PREGRASP_LEFT = [0.04, -0.7433, 0.6971, -2.6956, 0.501, 2.0325, 1.1425]
PREGRASP_RIGHT = [0.7375, -1.7036, -1.9288, -2.373, -1.0073, 3.7526, -0.3325]

# 采集/推理起始脊柱高度(数据首帧脊柱高)
SPINE_H = 0.5


def cand(n: str):
    """关节名回退:场景里可能是 fr3v2_joint 或 fr3v2_1_joint,夹爪也有别名。"""
    yield n
    if "fr3v2_joint" in n:
        yield n.replace("fr3v2_joint", "fr3v2_1_joint")
    if n == LG:
        yield "left_fr3v2_finger_joint1"
    if n == RG:
        yield "right_fr3v2_finger_joint1"


def rj(joint_map: dict, name: str, default: float = math.nan) -> float:
    """从 {joint_name: pos} 里读某关节,带名字回退。"""
    for c in cand(name):
        v = joint_map.get(c)
        if v is not None and math.isfinite(v):
            return float(v)
    return default


def gof(rad: float) -> float:
    """夹爪驱动关节弧度 → 开合度 [0,1](0=闭,1=全开)。"""
    if not math.isfinite(rad):
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - rad / GC)))
