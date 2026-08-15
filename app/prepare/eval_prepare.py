#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Task2 端到端 eval pipeline(单进程编排,sev/env 里运行)。

    cd sev/env
    pixi run python ../task2/prepare/eval_prepare.py \
        --config ../task2/prepare/eval_pipeline.yaml

流程(全部参数见 eval_pipeline.yaml):
  1. 导航:出生点 → fixpos(legs 连续控制 + 末端脉冲微调),**同时并发升柱**;
  2. 停稳:底盘切到 P+脉冲钉位(BaseHold),贯穿到回 home;
  3. 摆臂:到位稳定后双臂插值到预抓取姿 + 开爪(此前全程不动臂);
  4. 推理:起 rollout client 子进程(右臂8;act8/dp8/pi 由 config.model.port 定);
  5. 结束:热垫放到红板(object_poses 检测)或 Ctrl-C → 杀 client → 双臂回 home。

工具函数全部在 ./utils(自包含);外部依赖仅:Isaac 场景(--record)、
model server(外部起好)、client 程序(config.env.rollout_client)。

兼容:不带 --config 时退回"纯泊车"旧行为(run_eval.sh 仍用 --skip-spine 调它)。
"""
from __future__ import annotations

import argparse
import math
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.geometry import parse_via, yaw_from_quat          # noqa: E402
from utils.robot import (                                     # noqa: E402
    SPINE, PREGRASP_LEFT, PREGRASP_RIGHT, HOME_LEFT, HOME_RIGHT, RG, rj, gof,
)
from utils.drive import BaseDriver                            # noqa: E402
from utils.spine import SpineRaiser                           # noqa: E402
from utils.hold import BaseHold                               # noqa: E402
from utils.arms import ArmMover                               # noqa: E402


# ---- 默认配置(yaml 缺项回落到这里)----
DEFAULTS = {
    "model": {"type": "pi", "host": "127.0.0.1", "port": 5557,
              "exec_horizon": 36, "blend": 10},
    "fixpos": {"x": 2.1, "y": 3.05, "yaw_deg": -90.0},
    "spine": {"target": 0.5, "timeout": 20.0},
    "drive": {"via": "4.4,3.0; 2.1,3.0", "via_tol": 0.15, "pos_tol": 0.02,
              "yaw_tol_deg": 1.0, "settle": 15, "rate": 30.0, "timeout": 180.0,
              "stuck": 5.0, "kick_after": 1.0, "kick_v": 0.30, "max_v": 0.4,
              "max_w": 0.8, "min_v": 0.10, "kp": 0.8, "kp_yaw": 1.5, "a_max": 0.3,
              "pulse": True, "pulse_radius": 0.10, "pulse_tol": 0.005,
              "pulse_accept": 0.05, "pulse_v": 0.20, "pulse_push_timeout": 3.0,
              "pulse_cut_frac": 0.4, "pulse_coast": 0.5, "pulse_align": 0.5,
              "pulse_align_v": 0.02},
    "hold": {"no_pulse": False, "pulse_trigger": 0.010},
    "arms": {"max_vel": 0.4, "settle_s": 6.0, "settle_tol": 0.02},
    "done": {"timeout_s": 120.0},
    "end": {"grip_confirm_s": 3.0, "back_m": 0.10},
    "home_on_done": True,
    "env": {"sev_env": "", "rollout_client": ""},
}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and k in base else v
    return out


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    return _merge(DEFAULTS, user)


class EvalPipeline(Node):
    def __init__(self, cfg: dict):
        super().__init__("task2_eval_pipeline")
        self.cfg = cfg
        self.pose = None
        self.body_vel = (0.0, 0.0, 0.0)
        self.jm: dict = {}
        self.spine_pos = None
        self.sim_time = None
        self._interrupted = False

        self.cmd_pub = self.create_publisher(Twist, "/isaac/base_cmd_vel", 10)
        self.spine_pub = self.create_publisher(Float32, "/isaac/spine_command", 10)
        self.create_subscription(Odometry, "/isaac/odom", self._on_odom, 10)
        self.create_subscription(
            JointState, "/isaac/joint_states_full", self._on_js, 10)
        self.create_subscription(
            Clock, "/isaac/clock",
            lambda m: setattr(self, "sim_time",
                              m.clock.sec + m.clock.nanosec * 1e-9), 10)

    def _on_odom(self, m: Odometry):
        p, q, t = m.pose.pose.position, m.pose.pose.orientation, m.twist.twist
        self.pose = (p.x, p.y, yaw_from_quat(q.x, q.y, q.z, q.w))
        self.body_vel = (t.linear.x, t.linear.y, t.angular.z)

    def _on_js(self, m: JointState):
        self.jm = {n: p for n, p in zip(m.name, m.position)}
        if SPINE in m.name:
            self.spine_pos = float(m.position[m.name.index(SPINE)])

    def _send(self, vx, vy, wz):
        t = Twist(); t.linear.x = float(vx); t.linear.y = float(vy)
        t.angular.z = float(wz); self.cmd_pub.publish(t)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _spin(self, sec):
        t = time.time()
        while time.time() - t < sec and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.02)


def _drive_cfg(d: dict) -> SimpleNamespace:
    """把 drive dict 变成 BaseDriver 需要的属性对象。"""
    return SimpleNamespace(**d)


def run_pipeline(cfg: dict) -> int:
    node = EvalPipeline(cfg)
    log = node.get_logger()
    dcfg = _drive_cfg(cfg["drive"])
    rate = dcfg.rate
    period = 1.0 / rate

    # --- Ctrl-C:置标志,主循环里优雅收尾(推理段 → 回 home)---
    signal.signal(signal.SIGINT, lambda *_: setattr(node, "_interrupted", True))

    # --- preflight:等 odom ---
    log.info("等 /isaac/odom(场景在跑?)...")
    t0 = time.time()
    while node.pose is None and rclpy.ok() and time.time() - t0 < 15:
        rclpy.spin_once(node, timeout_sec=0.1)
    if node.pose is None:
        log.error("收不到 /isaac/odom —— 场景没跑或没带 --record")
        node.destroy_node(); return 1

    # --- 阶段1:导航 + 并发升柱 ---
    legs = [(x, y, dcfg.via_tol, 2) for x, y in parse_via(dcfg.via)] + \
           [(cfg["fixpos"]["x"], cfg["fixpos"]["y"], dcfg.pos_tol, dcfg.settle)]
    driver = BaseDriver(dcfg, legs, math.radians(cfg["fixpos"]["yaw_deg"]),
                        math.radians(dcfg.yaw_tol_deg), rate, node._send, log)
    spine = SpineRaiser(node.spine_pub, cfg["spine"]["target"], rate,
                        cfg["spine"]["timeout"], log)
    log.info("阶段1:导航到 fixpos + 并发升柱")
    while rclpy.ok() and not driver.done and not node._interrupted:
        rclpy.spin_once(node, timeout_sec=period)
        now = node.now()
        spine.tick(now, node.spine_pos)          # 并发升柱,不阻塞开车
        driver.set_state(node.pose, node.body_vel)
        driver.step(now)
    if node._interrupted:
        node._send(0, 0, 0); log.info("导航中断,退出"); node.destroy_node(); return 0
    if driver.failed:
        log.error("泊车失败,退出"); node._send(0, 0, 0); node.destroy_node(); return 1

    # --- 阶段2:起底盘钉位(定时器,贯穿到回 home)---
    hold = BaseHold(cfg["fixpos"]["x"], cfg["fixpos"]["y"],
                    math.radians(cfg["fixpos"]["yaw_deg"]), node._send, log,
                    no_pulse=cfg["hold"]["no_pulse"],
                    pulse_trigger=cfg["hold"]["pulse_trigger"])
    node.create_timer(period, lambda: hold.tick(node.now(), node.pose, node.body_vel))
    log.info("阶段2:底盘钉位(P+脉冲)已启动")

    # --- 阶段3:摆预抓取姿(hold 定时器在 spin_once 期间持续钉底盘)---
    arms = ArmMover(node, cfg["arms"]["max_vel"], rate)
    arms.set_joint_mode()
    log.info("阶段3:双臂 → 预抓取姿")
    arms.interp_to(lambda: node.jm, PREGRASP_LEFT, PREGRASP_RIGHT,
                   open_grip=True, settle_s=cfg["arms"]["settle_s"],
                   settle_tol=cfg["arms"]["settle_tol"], logger=log)

    # --- 阶段4:起 client 子进程,推理;监测放置完成 / Ctrl-C ---
    m = cfg["model"]
    client = cfg["env"]["rollout_client"]
    cmd = [sys.executable, client, "--host", str(m["host"]), "--port", str(m["port"]),
           "--rate", str(rate), "--exec-horizon", str(m["exec_horizon"]),
           "--blend", str(m["blend"])]
    log.info(f"阶段4:起推理 client → {os.path.basename(client)} :{m['port']}")
    proc = subprocess.Popen(cmd)

    # --- 结束判定:夹爪"闭合过(抓取)→ 再张开持续 grip_confirm_s 仿真秒" 或 超时 ---
    ec = cfg["end"]
    def _now_sim():
        return node.sim_time if node.sim_time is not None else time.time()
    log.info("推理中… 结束=夹爪闭合过后张开%.0f仿真秒;超时=%.0f仿真秒;或 Ctrl-C"
             % (ec["grip_confirm_s"], cfg["done"]["timeout_s"]))
    t_infer0 = _now_sim()
    grasped = False
    t_close = t_open = None
    t_log = t_infer0
    while rclpy.ok() and not node._interrupted:
        rclpy.spin_once(node, timeout_sec=period)
        now = _now_sim()
        if proc.poll() is not None:
            log.warning("client 子进程已退出"); break
        g = gof(rj(node.jm, RG))
        if not grasped:
            if g < 0.3:
                t_close = t_close or now
                if now - t_close >= 0.5:
                    grasped = True
                    log.info(f"检测到抓取(夹爪闭合)@ sim {now - t_infer0:.1f}s")
            else:
                t_close = None
        else:
            if g > 0.5:
                t_open = t_open or now
                if now - t_open >= ec["grip_confirm_s"]:
                    log.info(f"释放确认(张开{ec['grip_confirm_s']:.0f}仿真秒)@ "
                             f"sim {now - t_infer0:.1f}s → 退场")
                    break
            else:
                t_open = None
        if now - t_infer0 > cfg["done"]["timeout_s"]:
            log.warning("推理达总超时(仿真 %.0fs)→ 强制退场" % cfg["done"]["timeout_s"])
            break
        if now - t_log >= 10.0:
            t_log = now
            log.info(f"推理中 sim+{now - t_infer0:.0f}s 爪={g:.2f} 已抓取={grasped}")

    # --- 阶段5:杀 client,双臂回 home ---
    log.info("阶段5:停 client")
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    if cfg["home_on_done"]:
        back = float(cfg["end"]["back_m"])
        yaw = math.radians(cfg["fixpos"]["yaw_deg"])
        hold.tx -= math.cos(yaw) * back
        hold.ty -= math.sin(yaw) * back
        log.info(f"退场:双臂→预抓取姿 + 底盘后退 {back*100:.0f}cm"
                 f"(hold 新目标 {hold.tx:.3f},{hold.ty:.3f})")
        arms.set_joint_mode()
        arms.interp_to(lambda: node.jm, PREGRASP_LEFT, PREGRASP_RIGHT,
                       open_grip=True, settle_s=cfg["arms"]["settle_s"],
                       settle_tol=cfg["arms"]["settle_tol"], logger=log)
        node._spin(3.0)   # 再给 hold 几秒把底盘钉到后退点
    node._send(0, 0, 0)
    log.info("pipeline 完成")
    node.destroy_node()
    return 0


def run_prepare_only(a) -> int:
    """兼容旧用法:仅泊车(可选升柱),run_eval.sh 用 --skip-spine 调。"""
    node = EvalPipeline(_defaults_for_prepare_only())
    log = node.get_logger()
    rate = a.rate
    dcfg = SimpleNamespace(
        via=a.via, via_tol=a.via_tol, pos_tol=a.pos_tol, yaw_tol_deg=a.yaw_tol_deg,
        settle=a.settle, rate=rate, timeout=a.timeout, stuck=a.stuck,
        kick_after=a.kick_after, kick_v=a.kick_v, max_v=a.max_v, max_w=a.max_w,
        min_v=a.min_v, kp=a.kp, kp_yaw=a.kp_yaw, a_max=a.a_max, pulse=a.pulse,
        pulse_radius=a.pulse_radius, pulse_tol=a.pulse_tol,
        pulse_accept=a.pulse_accept, pulse_v=a.pulse_v,
        pulse_push_timeout=a.pulse_push_timeout, pulse_cut_frac=a.pulse_cut_frac,
        pulse_coast=a.pulse_coast, pulse_align=a.pulse_align,
        pulse_align_v=a.pulse_align_v)
    t0 = time.time()
    while node.pose is None and rclpy.ok() and time.time() - t0 < 15:
        rclpy.spin_once(node, timeout_sec=0.1)
    legs = [(x, y, a.via_tol, 2) for x, y in parse_via(a.via)] + \
           [(a.x, a.y, a.pos_tol, a.settle)]
    driver = BaseDriver(dcfg, legs, math.radians(a.yaw_deg),
                        math.radians(a.yaw_tol_deg), rate, node._send, log)
    spine = None if a.skip_spine else SpineRaiser(
        node.spine_pub, a.spine, rate, a.spine_timeout, log)
    try:
        while rclpy.ok() and not driver.done:
            rclpy.spin_once(node, timeout_sec=1.0 / rate)
            now = node.now()
            if spine is not None and not spine.done:
                spine.tick(now, node.spine_pos)
            if a.skip_base:
                if spine is None or spine.done:
                    break
                continue
            driver.set_state(node.pose, node.body_vel)
            driver.step(now)
    except KeyboardInterrupt:
        node._send(0, 0, 0)
    failed = driver.failed
    node.destroy_node()
    return 1 if failed else 0


def _defaults_for_prepare_only() -> dict:
    return DEFAULTS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="",
                    help="eval_pipeline.yaml 路径;给了走完整 pipeline,不给走纯泊车")
    # --- 纯泊车兼容参数(仅 --config 缺省时用)---
    ap.add_argument("--x", type=float, default=2.1)
    ap.add_argument("--y", type=float, default=3.05)
    ap.add_argument("--yaw-deg", type=float, default=-90.0)
    ap.add_argument("--spine", type=float, default=0.35)
    ap.add_argument("--via", type=str, default="4.4,3.0; 2.1,3.0")
    ap.add_argument("--via-tol", type=float, default=0.15)
    ap.add_argument("--pos-tol", type=float, default=0.02)
    ap.add_argument("--yaw-tol-deg", type=float, default=1.0)
    ap.add_argument("--settle", type=int, default=15)
    ap.add_argument("--rate", type=float, default=30.0)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--stuck", type=float, default=5.0)
    ap.add_argument("--kick-after", type=float, default=1.0)
    ap.add_argument("--kick-v", type=float, default=0.30)
    ap.add_argument("--spine-timeout", type=float, default=20.0)
    ap.add_argument("--max-v", type=float, default=0.4)
    ap.add_argument("--max-w", type=float, default=0.8)
    ap.add_argument("--min-v", type=float, default=0.10)
    ap.add_argument("--kp", type=float, default=0.8)
    ap.add_argument("--kp-yaw", type=float, default=1.5)
    ap.add_argument("--a-max", type=float, default=0.3)
    ap.add_argument("--pulse", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--pulse-radius", type=float, default=0.10)
    ap.add_argument("--pulse-tol", type=float, default=0.005)
    ap.add_argument("--pulse-accept", type=float, default=0.05)
    ap.add_argument("--pulse-v", type=float, default=0.20)
    ap.add_argument("--pulse-push-timeout", type=float, default=3.0)
    ap.add_argument("--pulse-cut-frac", type=float, default=0.4)
    ap.add_argument("--pulse-coast", type=float, default=0.5)
    ap.add_argument("--pulse-align", type=float, default=0.5)
    ap.add_argument("--pulse-align-v", type=float, default=0.02)
    ap.add_argument("--skip-base", action="store_true")
    ap.add_argument("--skip-spine", action="store_true")
    a = ap.parse_args()

    rclpy.init()
    try:
        if a.config:
            return run_pipeline(load_config(a.config))
        return run_prepare_only(a)
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
