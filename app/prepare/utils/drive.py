# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""底盘开车到 fixpos:legs 分段连续控制 + 末端脉冲微调。自包含。

忠实搬运自原 eval_prepare.py 的 _drive_base / _pulse_step(逻辑不变),改成
外部注入 pose/body_vel + send 回调 + logger,便于被编排节点复用。

用法:
    d = BaseDriver(cfg, legs, target_yaw, yaw_tol, rate, send=..., logger=...)
    # 每个控制拍:
    d.set_state(pose, body_vel); d.step(now)
    # 直到 d.done(成功 base_done) 或 d.failed(卡死/超残差)
"""
from __future__ import annotations

import math

from .geometry import wrap_angle


class BaseDriver:
    def __init__(self, cfg, legs, target_yaw, yaw_tol, rate, send, logger=None):
        self.c = cfg                     # 有 max_v/kp/pulse_* 等属性的配置对象
        self.legs = legs                 # [(x, y, tol, settle_need), ...]
        self.target_yaw = target_yaw
        self.yaw_tol = yaw_tol
        self.rate = rate
        self.send = send                 # send(vx, vy, wz)
        self.log = logger

        self.pose = None                 # (x, y, yaw)
        self.body_vel = (0.0, 0.0, 0.0)
        self.leg_index = 0
        self.settle_ticks = 0
        self.tick_count = 0
        self.done = False
        self.failed = False

        self._progress_pose = None
        self._progress_time = None
        self._pulse_active = False
        self._pulse_phase = "idle"       # idle | align | push | coast
        self._pulse_v = cfg.pulse_v
        self._push_cut = 0.0
        self._push_deadline = 0.0
        self._align_end = 0.0
        self._coast_end = 0.0
        self._pulse_from = None
        self._pulse_count = 0
        self._pulse_best = float("inf")
        self._pulse_noimp = 0

    def set_state(self, pose, body_vel):
        self.pose = pose
        self.body_vel = body_vel

    def _stop(self):
        self.send(0.0, 0.0, 0.0)

    def _info(self, msg):
        if self.log:
            self.log.info(msg)

    def step(self, now: float):
        if self.done or self.pose is None:
            return
        self.tick_count += 1
        self._drive(now)

    # ---- 末端脉冲微调:align → push → coast ----
    def _pulse_step(self, now, dist, ex, ey, yaw_err, speed):
        c = self.c
        x, y, yaw = self.pose
        ux, uy = ex / max(dist, 1e-9), ey / max(dist, 1e-9)

        if self._pulse_phase == "align":
            if now < self._align_end:
                self.send(ux * c.pulse_align_v, uy * c.pulse_align_v, 0.0)
            else:
                self._push_deadline = now + c.pulse_push_timeout
                self._pulse_phase = "push"
            return

        if self._pulse_phase == "push":
            px, py, _ = self._pulse_from
            moved = math.hypot(x - px, y - py)
            if moved >= self._push_cut:
                self._stop()
                self._pulse_phase = "coast"
                self._coast_end = now + c.pulse_coast
                return
            if now > self._push_deadline:
                if self._pulse_v >= 0.35 - 1e-6:
                    if self.log:
                        self.log.error(
                            f"PULSE failed: 0.35 m/s 持续 {c.pulse_push_timeout:.0f}s "
                            f"推不动 (residual {dist * 1000:.1f}mm) — parking FAILED"
                        )
                    self._stop()
                    self.failed = True
                    self.done = True
                    return
                self._pulse_v = min(self._pulse_v + 0.05, 0.35)
                self._push_deadline = now + c.pulse_push_timeout
                self._info(f"PULSE: 未破静摩擦,升速到 {self._pulse_v:.2f} 续推")
            wz = 0.0
            if abs(yaw_err) > self.yaw_tol * 0.5:
                wz = max(-0.3, min(0.3, c.kp_yaw * yaw_err))
            self.send(ux * self._pulse_v, uy * self._pulse_v, wz)
            return

        if self._pulse_phase == "coast":
            self._stop()
            if now < self._coast_end or (speed >= 0.012 and now < self._coast_end + 8.0):
                return
            if speed >= 0.012 and self.log:
                self.log.warning(
                    f"coast 8s 后仍未停稳 (speed={speed * 100:.1f}cm/s),继续评估")
            px, py, _ = self._pulse_from
            moved = math.hypot(x - px, y - py)
            coast = max(moved - self._push_cut, 0.0)
            self._pulse_count += 1
            if dist < self._pulse_best - 0.001:
                self._pulse_best, self._pulse_noimp = dist, 0
            else:
                self._pulse_noimp += 1
            self._info(
                f"PULSE #{self._pulse_count}: cut {self._push_cut * 1000:.0f}mm "
                f"+ 滑行 {coast * 1000:.1f}mm = moved {moved * 1000:.1f}mm → "
                f"residual {dist * 1000:.1f}mm (v={self._pulse_v:.2f}) "
                f"yaw_err={math.degrees(yaw_err):+.2f}°"
            )
            done = dist <= self.c.pulse_tol
            give_up = self._pulse_noimp >= 3 or self._pulse_count >= 40
            if done or give_up:
                tag = "converged" if done else "no further improvement"
                self._info(
                    f"PULSE done ({tag}): residual {dist * 1000:.1f}mm / "
                    f"{math.degrees(abs(yaw_err)):.2f}° after {self._pulse_count} "
                    f"pulses (best {self._pulse_best * 1000:.1f}mm)"
                )
                if not done and dist > self.c.pulse_accept:
                    if self.log:
                        self.log.error(
                            f"PULSE failed: residual {dist * 1000:.1f}mm > accept "
                            f"{self.c.pulse_accept * 1000:.0f}mm — parking FAILED")
                    self.failed = True
                    self.done = True
                    return
                self._info(
                    f"base parked (pulse): x={x:+.4f} y={y:+.4f} "
                    f"yaw={math.degrees(yaw):+.2f}° (err {dist * 1000:.1f}mm)")
                self.done = True
                return
            self._pulse_phase = "idle"
            return

        # idle:新循环——记起点、算 cut、先预对准转向
        self._pulse_from = self.pose
        self._push_cut = min(max(self.c.pulse_cut_frac * dist, 0.002), 0.02)
        self._align_end = now + self.c.pulse_align
        self._pulse_phase = "align"

    # ---- 分段连续控制 ----
    def _drive(self, now):
        c = self.c
        tx, ty, tol, settle_need = self.legs[self.leg_index]
        final_leg = self.leg_index == len(self.legs) - 1
        x, y, yaw = self.pose
        vx_meas, vy_meas, _ = self.body_vel
        speed = math.hypot(vx_meas, vy_meas)
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)
        ex = math.cos(yaw) * dx + math.sin(yaw) * dy
        ey = -math.sin(yaw) * dx + math.cos(yaw) * dy
        yaw_err = wrap_angle(self.target_yaw - yaw)
        yaw_ok_tol = self.yaw_tol if final_leg else math.radians(5.0)

        if final_leg and c.pulse:
            if not self._pulse_active and dist <= c.pulse_radius:
                self._pulse_active = True
                self._info(
                    f"PULSE mode engaged at {dist * 1000:.0f}mm "
                    f"(v={self._pulse_v:.2f}, target {c.pulse_tol * 1000:.0f}mm)")
            if self._pulse_active:
                self._pulse_step(now, dist, ex, ey, yaw_err, speed)
                return

        if dist <= tol and abs(yaw_err) <= yaw_ok_tol and speed < 0.05:
            self.settle_ticks += 1
            self._stop()
            if self.settle_ticks >= settle_need:
                self._info(
                    f"leg {self.leg_index + 1}/{len(self.legs)} reached: "
                    f"x={x:+.3f} y={y:+.3f} yaw={math.degrees(yaw):+.1f}° "
                    f"(err {dist * 100:.1f} cm)")
                self.settle_ticks = 0
                if final_leg:
                    self._info(
                        f"base parked: x={x:+.3f} y={y:+.3f} "
                        f"yaw={math.degrees(yaw):+.1f}° (err {dist * 100:.1f} cm)")
                    self.done = True
                else:
                    self.leg_index += 1
            return
        self.settle_ticks = 0

        v_des = min(
            c.max_v, c.kp * dist,
            math.sqrt(2.0 * c.a_max * max(dist - tol * 0.5, 0.0)),
        )
        if dist > tol and 0.0 < v_des < c.min_v:
            v_des = c.min_v
        scale = v_des / dist if dist > 1e-6 else 0.0
        vx_cmd = ex * scale if dist > tol else 0.0
        vy_cmd = ey * scale if dist > tol else 0.0
        wz_cmd = max(-c.max_w, min(c.max_w, c.kp_yaw * yaw_err))
        if abs(yaw_err) <= yaw_ok_tol * 0.5:
            wz_cmd = 0.0

        lin_mag = math.hypot(vx_cmd, vy_cmd)
        no_progress = 0.0 if self._progress_time is None else now - self._progress_time
        if lin_mag > 1e-3 and dist > 2.0 * tol and no_progress > c.kick_after:
            boost = c.kick_v / lin_mag
            vx_cmd, vy_cmd = vx_cmd * boost, vy_cmd * boost
            if self.tick_count % max(int(self.rate // 3), 1) == 0:
                self._info(
                    f"KICK: no progress {no_progress:.1f}s → boosting to "
                    f"{c.kick_v:.2f} m/s to break static friction")

        self.send(vx_cmd, vy_cmd, wz_cmd)

        cmd_mag = math.hypot(vx_cmd, vy_cmd) + abs(wz_cmd)
        if self._progress_pose is None or cmd_mag < 1e-3:
            self._progress_pose, self._progress_time = self.pose, now
        else:
            px, py, pyaw = self._progress_pose
            moved = math.hypot(x - px, y - py) > 0.03 or abs(
                wrap_angle(yaw - pyaw)) > math.radians(2.0)
            if moved:
                self._progress_pose, self._progress_time = self.pose, now
            elif now - self._progress_time > c.stuck:
                if final_leg and dist <= 0.05:
                    if self.log:
                        self.log.warning(
                            f"base parked (stuck near target): err {dist * 100:.1f} cm "
                            "— accepted")
                    self._stop()
                    self.done = True
                    return
                if self.log:
                    self.log.error(
                        f"STUCK: cmd=({vx_cmd:+.2f},{vy_cmd:+.2f},{wz_cmd:+.2f}) 但 "
                        f"{c.stuck:.0f}s 没动 at ({x:+.2f},{y:+.2f},"
                        f"{math.degrees(yaw):+.1f}°) — 障碍?Stopping.")
                self._stop()
                self.failed = True
                self.done = True
                return

        if self.tick_count % int(self.rate) == 0:
            self._info(
                f"leg {self.leg_index + 1}/{len(self.legs)} "
                f"cmd=({vx_cmd:+.2f},{vy_cmd:+.2f},{wz_cmd:+.2f}) "
                f"dist={dist * 100:.1f}cm yaw_err={math.degrees(yaw_err):+.1f}° "
                f"pose=({x:+.2f}, {y:+.2f}, {math.degrees(yaw):+.1f}°)")
