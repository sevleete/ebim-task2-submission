# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""放置完成检测:订阅 /isaac/task2/object_poses(真值位姿),判定热垫是否
已放到目标红板上并停稳,给 eval pipeline "该回 home 了" 的信号。自包含。

object_poses(--record 隐含 --publish-ground-truth 时发布):
  std_msgs/String JSON: {"sim_time": t, "objects": {name: [x,y,z, qw,qx,qy,qz]}}
判据:thermalpad 与 board_target 的水平距 < place_xy_tol 且 |Δz| < place_z_tol,
连续满足 settle_s 秒 → done(去抖,防搬运途中经过目标上方误触发)。
"""
from __future__ import annotations

import json
import math
import time

from std_msgs.msg import String


class PlaceDoneDetector:
    def __init__(self, node, place_xy_tol=0.03, place_z_tol=0.03,
                 settle_s=1.5, pad_key="thermalpad", target_key="board_target",
                 logger=None, gripper_open_getter=None):
        # gripper_open_getter(): 返回右爪开合度[0,1];提供时要求爪已张开(>0.5)
        # 才计入判定 —— 语义 = "夹爪最后打开后"(防夹着 pad 经过目标上方误判)。
        self.gripper_open_getter = gripper_open_getter
        self.place_xy_tol = place_xy_tol
        self.place_z_tol = place_z_tol
        self.settle_s = settle_s
        self.pad_key = pad_key
        self.target_key = target_key
        self.log = logger
        self.done = False
        self._ok_since = None
        node.create_subscription(
            String, "/isaac/task2/object_poses", self._on_poses, 10)

    def _on_poses(self, msg: String):
        if self.done:
            return
        try:
            objs = json.loads(msg.data).get("objects", {})
        except (ValueError, TypeError):
            return
        pad = objs.get(self.pad_key)
        tgt = objs.get(self.target_key)
        if pad is None or tgt is None:
            return
        dxy = math.hypot(pad[0] - tgt[0], pad[1] - tgt[1])
        dz = abs(pad[2] - tgt[2])
        placed = dxy < self.place_xy_tol and dz < self.place_z_tol
        if placed and self.gripper_open_getter is not None:
            placed = self.gripper_open_getter() > 0.5
        now = time.time()
        if placed:
            if self._ok_since is None:
                self._ok_since = now
                if self.log:
                    self.log.info(
                        f"检测到热垫在目标区(dxy={dxy*100:.1f}cm dz={dz*100:.1f}cm)"
                        f",稳定 {self.settle_s:.1f}s 即判完成")
            elif now - self._ok_since >= self.settle_s:
                self.done = True
                if self.log:
                    self.log.info(
                        f"放置完成 ✓ (dxy={dxy*100:.1f}cm dz={dz*100:.1f}cm 稳定)")
        else:
            self._ok_since = None
