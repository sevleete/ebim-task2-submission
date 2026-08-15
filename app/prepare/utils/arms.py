# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""双臂关节控制:限速插值到目标姿态 + 开爪。自包含(端口 data_collect 的 move_arms)。

复用于两处:
  - 摆预抓取姿(interp_to(PREGRASP_LEFT, PREGRASP_RIGHT))
  - 推理结束回 home(interp_to(HOME_LEFT, HOME_RIGHT))
"""
from __future__ import annotations

import time

from sensor_msgs.msg import JointState
from std_msgs.msg import String

from .robot import LEFT, RIGHT, LG, RG, rj


class ArmMover:
    """持有双臂/双爪/模式发布器;把双臂从当前实测姿限速插值到目标。"""

    def __init__(self, node, max_vel: float = 0.4, rate: float = 30.0):
        self._node = node
        self._max_vel = max_vel
        self._rate = rate
        self.l_pub = node.create_publisher(JointState, "/isaac/left_joint_commands", 10)
        self.r_pub = node.create_publisher(JointState, "/isaac/right_joint_commands", 10)
        self.lg_pub = node.create_publisher(
            JointState, "/isaac/left_robotiq_joint_commands", 10)
        self.rg_pub = node.create_publisher(
            JointState, "/isaac/right_robotiq_joint_commands", 10)
        self.mode_pub = node.create_publisher(String, "/isaac/arm_control_mode", 10)

    def set_joint_mode(self, n: int = 6):
        """连发 joint 模式,确保 bridge 走关节命令通路(防 DDS 首包丢失)。"""
        for _ in range(n):
            self.mode_pub.publish(String(data="joint"))
            self._spin(0.05)

    def _spin(self, sec: float):
        import rclpy
        t = time.time()
        while time.time() - t < sec:
            rclpy.spin_once(self._node, timeout_sec=0.01)

    def publish_arms(self, ql, qr):
        now = self._node.get_clock().now().to_msg()
        for pub, names, qs in ((self.l_pub, LEFT, ql), (self.r_pub, RIGHT, qr)):
            m = JointState()
            m.header.stamp = now
            m.name = names
            m.position = [float(v) for v in qs]
            pub.publish(m)

    def open_grippers(self):
        now = self._node.get_clock().now().to_msg()
        for pub, name in ((self.lg_pub, LG), (self.rg_pub, RG)):
            m = JointState()
            m.header.stamp = now
            m.name = [name]
            m.position = [0.0]      # 0 rad = 全开
            pub.publish(m)

    def interp_to(self, joint_map_getter, ql_target, qr_target,
                  open_grip: bool = True, settle_s: float = 6.0,
                  settle_tol: float = 0.02, logger=None) -> float:
        """从当前实测姿限速插值到 (ql_target, qr_target);末端保持并等收敛。

        joint_map_getter(): 返回最新 {joint_name: pos} 的可调用对象。
        返回收敛后的最大关节误差 (rad)。
        """
        import rclpy
        jm = joint_map_getter()
        ql0 = [rj(jm, n) for n in LEFT]
        qr0 = [rj(jm, n) for n in RIGHT]
        dmax = max(
            max(abs(a - b) for a, b in zip(ql0, ql_target)),
            max(abs(a - b) for a, b in zip(qr0, qr_target)),
        )
        T = min(max(dmax / self._max_vel, 2.0), 12.0)
        steps = int(T * self._rate)
        if logger:
            logger.info(f"双臂插值 {T:.1f}s(最大位移 {dmax:.2f} rad, {steps} 步)")
        for i in range(1, steps + 1):
            a = i / steps
            ql = [q0 + (qt - q0) * a for q0, qt in zip(ql0, ql_target)]
            qr = [q0 + (qt - q0) * a for q0, qt in zip(qr0, qr_target)]
            self.publish_arms(ql, qr)
            if open_grip:
                self.open_grippers()
            rclpy.spin_once(self._node, timeout_sec=0.0)
            time.sleep(1.0 / self._rate)
        # 保持发最终目标并等实测收敛
        t0 = time.time()
        err = dmax
        while time.time() - t0 < settle_s:
            self.publish_arms(ql_target, qr_target)
            if open_grip:
                self.open_grippers()
            rclpy.spin_once(self._node, timeout_sec=0.0)
            time.sleep(1.0 / self._rate)
            jm = joint_map_getter()
            el = [abs(rj(jm, n) - t) for n, t in zip(LEFT, ql_target)]
            er = [abs(rj(jm, n) - t) for n, t in zip(RIGHT, qr_target)]
            err = max(max(el), max(er))
            if err < settle_tol:
                break
        if logger:
            logger.info(f"双臂到位,最大误差 {err:.4f} rad")
        return err
