#!/usr/bin/env python
"""Eval 版 HIL 客户端:π chunk + 冻结残差头 Δ 修正,常开无录制门控。在 sev/env 运行。

与 ros_rollout_client_act8.py 同底盘(sim-clock 步进 + 块边界 blend),叠加:
  - 每个执行拍向 HIL server 发 {腕图jpg, state8, a_base8} 拿 Δ(独立线程,不阻塞);
  - a_final = clip(a_base + EMA(Δ), ±bounds);Δ 超时 0.2s 归零(安全回退);
  - 无录制/接管逻辑 —— 专供 eval_prepare 的阶段4 子进程。
搭配:HIL server 以冻结模式运行(--resume head --serve-after 0 --min-human 大数)。

  pixi run python eval_hil_client.py --host 127.0.0.1 --port 5557 --exec-horizon 36
"""
import argparse, math, pickle, queue, socket, struct, threading, time
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, JointState
from rosgraph_msgs.msg import Clock

RIGHT = [f"right_fr3v2_joint{i}" for i in range(1, 8)]
RG, GC = "right_right_finger_joint", 0.8
TASK = "Pick up the thermal pad and place it on the target RAM board."
BOUNDS = np.array([0.05] * 7 + [0.15], np.float32)
IMG_SIZE = 126


def cand(n):
    yield n
    if "fr3v2_joint" in n: yield n.replace("fr3v2_joint", "fr3v2_1_joint")
    if n == RG: yield "right_fr3v2_finger_joint1"
def rj(m, n, d=math.nan):
    for c in cand(n):
        if c in m and math.isfinite(m[c]): return float(m[c])
    return d
def gof(r): return 0.0 if not math.isfinite(r) else float(np.clip(1 - r / GC, 0, 1))
def recv_exact(s, n):
    b = b""
    while len(b) < n:
        c = s.recv(n - len(b))
        if not c: return None
        b += c
    return b
def send_msg(s, o):
    p = pickle.dumps(o, protocol=pickle.HIGHEST_PROTOCOL)
    s.sendall(struct.pack(">Q", len(p)) + p)
def recv_msg(s):
    h = recv_exact(s, 8)
    if h is None: return None
    (ln,) = struct.unpack(">Q", h)
    return pickle.loads(recv_exact(s, ln))


class HilLink(threading.Thread):
    def __init__(self, host, port, log):
        super().__init__(daemon=True)
        self.host, self.port, self.log = host, port, log
        self.q = queue.Queue(maxsize=1)
        self.delta = np.zeros(8, np.float32)
        self.delta_time = 0.0
        self.sock = None

    def submit(self, msg):
        try:
            self.q.put_nowait(msg)
        except queue.Full:
            try: self.q.get_nowait()
            except queue.Empty: pass
            self.q.put_nowait(msg)

    def _connect(self):
        while self.sock is None:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(3.0); s.connect((self.host, self.port)); s.settimeout(2.0)
                self.sock = s; self.log(f"HIL server 已连 {self.host}:{self.port}")
            except OSError:
                time.sleep(2.0)

    def run(self):
        self._connect()
        while True:
            msg = self.q.get()
            try:
                send_msg(self.sock, msg)
                r = recv_msg(self.sock)
                if r is None: raise ConnectionResetError
                d = r.get("delta") if isinstance(r, dict) else None
                if d is not None:
                    self.delta = np.asarray(d, np.float32)
                    self.delta_time = time.time()
            except (OSError, ConnectionResetError, struct.error):
                self.log("HIL 断开,重连")
                try: self.sock.close()
                except Exception: pass
                self.sock = None; self._connect()

    def latest(self, max_age=0.2):
        return self.delta if time.time() - self.delta_time <= max_age else np.zeros(8, np.float32)


class EvalHilClient(Node):
    def __init__(self, args):
        super().__init__("eval_hil_client")
        self.args = args
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((args.host, args.port))
        self.get_logger().info(f"已连 π server {args.host}:{args.port}")
        self.hil = HilLink(args.hil_host, args.hil_port, lambda m: self.get_logger().info(m))
        self.hil.start()
        self.imgs = {"head": None, "wrist_right": None}
        self.jm = {}; self.sim_time = None
        self.buf = None; self.idx = 0; self.exec_h = args.exec_horizon
        self.step_dt = 1.0 / args.rate; self._last_step_sim = None
        self.blend_k = args.blend; self._last_pub = None; self._off = np.zeros(7, np.float32)
        self._dema = np.zeros(8, np.float32)
        qi = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        q = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.create_subscription(Image, "/isaac/head_camera/image_raw", lambda m: self._img("head", m), qi)
        self.create_subscription(Image, "/isaac/right_wrist_camera/image_raw", lambda m: self._img("wrist_right", m), qi)
        self.create_subscription(JointState, "/isaac/joint_states_full", self._js, q)
        self.create_subscription(Clock, "/isaac/clock", lambda m: self._clk(m), 10)
        self.pub_r = self.create_publisher(JointState, "/isaac/right_joint_commands", 10)
        self.pub_rg = self.create_publisher(JointState, "/isaac/right_robotiq_joint_commands", 10)
        self.create_timer(1.0 / (args.rate * 6.0), self._tick)
        self.get_logger().info(
            f"Eval-HIL 客户端就绪(π+冻结残差),exec_horizon={self.exec_h} "
            f"ema={args.delta_ema} grip_delta={'off' if args.no_grip_delta else 'on'}")

    def _img(self, k, m):
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)[:, :m.width*3].reshape(m.height, m.width, 3)
        if m.encoding == "bgr8": a = a[:, :, ::-1]
        self.imgs[k] = np.ascontiguousarray(a)
    def _js(self, m): self.jm = {n: p for n, p in zip(m.name, m.position)}
    def _clk(self, m): self.sim_time = m.clock.sec + m.clock.nanosec * 1e-9

    def _ready(self):
        return all(v is not None for v in self.imgs.values()) and self.jm

    def _state8(self):
        s = np.full(8, np.nan, np.float32)
        for i, n in enumerate(RIGHT): s[i] = rj(self.jm, n)
        s[7] = gof(rj(self.jm, RG, 0.0))
        return s

    def _query(self, state, images):
        req = {"state": state.astype(np.float32), "images": images, "task": TASK}
        p = pickle.dumps(req, protocol=pickle.HIGHEST_PROTOCOL)
        self.sock.sendall(struct.pack(">Q", len(p)) + p)
        (ln,) = struct.unpack(">Q", recv_exact(self.sock, 8))
        return pickle.loads(recv_exact(self.sock, ln))

    def _publish(self, a):
        now = self.get_clock().now().to_msg()
        arm = np.asarray(a[0:7], np.float32)
        m = JointState(); m.header.stamp = now; m.name = RIGHT; m.position = [float(v) for v in arm]
        self.pub_r.publish(m)
        g = JointState(); g.header.stamp = now; g.name = [RG]
        g.position = [float((1.0 - np.clip(a[7], 0, 1)) * GC)]
        self.pub_rg.publish(g)
        self._last_pub = arm.copy()

    def _ask_delta(self, a_base):
        img = cv2.resize(self.imgs["wrist_right"], (IMG_SIZE, IMG_SIZE))
        ok, buf = cv2.imencode(".jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            self.hil.submit({"kind": "sample", "mid": "eval", "sim_time": self.sim_time,
                             "mode": "auto", "img": buf.tobytes(), "state": self._state8(),
                             "a_base": a_base.astype(np.float32), "a_applied": None,
                             "want_delta": True})

    def _corrected(self, a, update_ema=True):
        d_raw = np.clip(self.hil.latest(), -BOUNDS, BOUNDS)
        if update_ema:
            k = self.args.delta_ema
            self._dema = k * self._dema + (1.0 - k) * d_raw
        d = np.clip(self._dema, -BOUNDS, BOUNDS)
        if self.args.no_grip_delta:
            d[7] = 0.0
        a[0:8] = a[0:8] + d
        a[7] = float(np.clip(a[7], 0.0, 1.0))
        return a

    def _tick(self):
        if not self._ready() or self.sim_time is None:
            return
        if (self.buf is not None and self._last_step_sim is not None
                and (self.sim_time - self._last_step_sim) < self.step_dt):
            self._publish(self._corrected(self.buf[max(0, self.idx - 1)].copy(), update_ema=False))
            return
        need = self.buf is None or self.idx >= min(self.exec_h, len(self.buf))
        if need:
            try:
                self.buf = self._query(self._state8(), {k: v for k, v in self.imgs.items()})
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                self.get_logger().error(f"π server 断开: {e}"); self.buf = None; return
            self.idx = 0
            if self.blend_k > 0 and self._last_pub is not None:
                self._off = self._last_pub - self.buf[0][0:7]
            else:
                self._off = np.zeros(7, np.float32)
            self.get_logger().info(f"新动作块 {self.buf.shape}")
        a = self.buf[self.idx].copy()
        if self.blend_k > 0:
            a[0:7] = a[0:7] + self._off * max(0.0, 1.0 - self.idx / self.blend_k)
        self._ask_delta(a)
        self._publish(self._corrected(a))
        self.idx += 1; self._last_step_sim = self.sim_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5557)
    ap.add_argument("--hil-host", default="127.0.0.1")
    ap.add_argument("--hil-port", type=int, default=5561)
    ap.add_argument("--rate", type=float, default=30.0)
    ap.add_argument("--exec-horizon", type=int, default=36)
    ap.add_argument("--blend", type=int, default=10)
    ap.add_argument("--delta-ema", type=float, default=0.6)
    ap.add_argument("--no-grip-delta", action="store_true")
    args = ap.parse_args()
    rclpy.init(); n = EvalHilClient(args)
    try: rclpy.spin(n)
    except KeyboardInterrupt: pass
    finally: n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
