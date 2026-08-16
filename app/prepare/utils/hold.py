# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""底盘钉位保持器:温和 P + 脉冲破静摩擦,把底盘钉在目标 fixpos(≤1cm)。

自包含,搬运自 teleop/keyboard/base_holder.py 的两层保持逻辑。停稳后到回 home
全程每拍调 tick() 保持底盘不漂。
"""
from __future__ import annotations

import math


class BaseHold:
    def __init__(self, tx, ty, tyaw, send, logger=None,
                 no_pulse=False, pulse_trigger=0.010):
        self.tx, self.ty, self.tyaw = tx, ty, tyaw
        self.send = send
        self.log = logger
        self.no_pulse = no_pulse
        self.pulse_trigger = pulse_trigger
        self._st = "p"                 # p | align | push | coast
        self._end = 0.0
        self._err_since = None
        self._v = 0.15
        self._cut = 0.0
        self._from = (0.0, 0.0)
        self._from_dist = 0.0

    def tick(self, now, pose, body_vel):
        if pose is None:
            self.send(0.0, 0.0, 0.0)
            return
        x, y, yw = pose
        vx, vy, _ = body_vel
        dx, dy = self.tx - x, self.ty - y
        dist = math.hypot(dx, dy)
        ex = math.cos(yw) * dx + math.sin(yw) * dy
        ey = -math.sin(yw) * dx + math.cos(yw) * dy
        eyaw = math.atan2(math.sin(self.tyaw - yw), math.cos(self.tyaw - yw))
        speed = math.hypot(vx, vy)

        if not self.no_pulse:
            ux = ex / dist if dist > 1e-6 else 0.0
            uy = ey / dist if dist > 1e-6 else 0.0
            if self._st == "align":
                if now < self._end:
                    self.send(0.02 * ux, 0.02 * uy, 0.0)
                    return
                self._end = now + 2.5
                self._st = "push"
            if self._st == "push":
                moved = math.hypot(x - self._from[0], y - self._from[1])
                if moved >= self._cut:
                    self._st = "coast"
                    self._end = now + 0.4
                    self.send(0.0, 0.0, 0.0)
                    return
                if now > self._end:
                    if self._v >= 0.30 - 1e-6:
                        if self.log:
                            self.log.warning(
                                f"hold脉冲放弃: 0.30 推不动 (err {dist*100:.2f}cm),10s 后再试")
                        self._st = "p"
                        self._err_since = now + 10.0
                        self.send(0.0, 0.0, 0.0)
                        return
                    self._v = min(self._v + 0.05, 0.30)
                    self._end = now + 2.5
                self.send(self._v * ux, self._v * uy, 0.0)
                return
            if self._st == "coast":
                self.send(0.0, 0.0, 0.0)
                if now < self._end or (speed >= 0.02 and now < self._end + 1.5):
                    return
                if self.log:
                    self.log.info(
                        f"hold脉冲: {self._from_dist*100:.2f}cm → {dist*100:.2f}cm "
                        f"(v={self._v:.2f})")
                self._v = max(0.15, self._v - 0.05)
                self._st = "p"
                self._err_since = None
            # p:误差超阈值且静止 0.8s → 触发脉冲(先预对准)
            if dist > self.pulse_trigger and speed < 0.02:
                if self._err_since is None:
                    self._err_since = now
                elif now - self._err_since > 0.8:
                    self._from, self._from_dist = (x, y), dist
                    self._cut = min(max(0.5 * dist, 0.003), 0.015)
                    self._end = now + 0.4
                    self._st = "align"
                    if self.log:
                        self.log.info(
                            f"hold脉冲触发: 误差 {dist*100:.2f}cm 被静摩擦锁死 → 纠偏")
                    self.send(0.02 * ux, 0.02 * uy, 0.0)
                    return
            else:
                self._err_since = None

        # 温和 P 兜底(低增益防振荡)
        vx_cmd = vy_cmd = wz_cmd = 0.0
        if dist > 0.03:
            vx_cmd = max(-0.2, min(0.2, 1.5 * ex))
            vy_cmd = max(-0.2, min(0.2, 1.5 * ey))
        if abs(eyaw) > 0.005:
            wz_cmd = max(-0.5, min(0.5, 2.0 * eyaw))
        self.send(vx_cmd, vy_cmd, wz_cmd)
