# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""升降柱控制:导航期并发升柱(每拍发目标,到位判定)。自包含。"""
from __future__ import annotations

from std_msgs.msg import Float32


class SpineRaiser:
    """每拍调 tick() 发一次脊柱目标高度;实测到位或超时后 done=True。

    用法(导航期并发):每个控制 tick 调用一次 tick(now, spine_pos),
    与开车同拍进行——不阻塞驱动。
    """

    def __init__(self, spine_pub, target: float, rate: float,
                 timeout: float, logger=None, tol: float = 0.01):
        self._pub = spine_pub
        self._target = float(target)
        self._tol = tol
        self._rate = rate
        self._log = logger
        self._sent = 0
        self._deadline = None          # 首次 tick 时以 now 设定
        self._timeout = timeout
        self.done = False

    def tick(self, now: float, spine_pos: float | None) -> bool:
        if self.done:
            return True
        if self._deadline is None:
            self._deadline = now + self._timeout
        self._pub.publish(Float32(data=self._target))
        self._sent += 1
        if spine_pos is not None:
            if abs(spine_pos - self._target) < self._tol:
                if self._log:
                    self._log.info(f"spine 到位 {spine_pos:.3f} m")
                self.done = True
                return True
            if self._log and self._sent % (2 * int(self._rate)) == 0:
                self._log.info(
                    f"spine 上升中: {spine_pos:.3f} → {self._target:.3f} m"
                )
        if now > self._deadline:
            if self._log:
                self._log.warning(
                    "spine 到位验证超时(无 joint_states_full 或卡住)—— 视为到位"
                )
            self.done = True
        return self.done
