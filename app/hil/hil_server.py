#!/usr/bin/env python
"""HIL 中央残差服务器 v3(回合缓冲制):边收(双机 DAgger 样本)边训(DINOv2 残差头)边服务(返回 Δaction)。

v2 修正(针对"残差生效后反而变差"的两个病根):
  1. 饱和降权:人接管的 Δ* 大面积打满 ±bounds(人接管后轨迹与 π 影子大幅偏离),
     旧版全权重训练 → 头学成"见接管场景就打满舵" → 部署过度修正。
     现按打满维数分级权重:0维=1.0 / 1维=0.5 / ≥2维=0.2(留方向、压幅值)。
  2. 隔离带:按 z 之前 1-2s 是 π 正在犯错的帧,旧版当 target=0 的中性样本 →
     在最需要纠正的状态上教"别纠正"。现自主帧先进 2s 隔离队列,
     接管发生即丢弃前 2s;正常老化/回合正常结束才转正。

协议同 v1(8字节长度前缀 + pickle):
  sample: {"kind","mid","sim_time","mode","img"(jpg),"state","a_base","a_applied","want_delta"}
  → {"delta": f32[8]|None, "ver": int}
  event:  {"kind":"event","name","mid","sim_time"} → {"ok":1}

  pixi run python hil_server.py --port 5561 --device cuda [--preload-human] [--resume ckpt]
"""
import argparse
import io
import json
import pickle
import socket
import struct
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

import residual_net as RN

BOUNDS = np.array(RN.DEFAULT_BOUNDS, dtype=np.float32)
QUARANTINE_S = 2.0        # 接管前多少秒的自主帧作废


def recv_exact(c, n):
    b = b""
    while len(b) < n:
        d = c.recv(n - len(b))
        if not d:
            return None
        b += d
    return b


def recv_msg(c):
    h = recv_exact(c, 8)
    if h is None:
        return None
    (ln,) = struct.unpack(">Q", h)
    return pickle.loads(recv_exact(c, ln))


def send_msg(c, o):
    p = pickle.dumps(o, protocol=pickle.HIGHEST_PROTOCOL)
    c.sendall(struct.pack(">Q", len(p)) + p)


def decode_jpg(b):
    from PIL import Image
    img = Image.open(io.BytesIO(b)).convert("RGB")
    if img.size != (RN.IMG_SIZE, RN.IMG_SIZE):
        img = img.resize((RN.IMG_SIZE, RN.IMG_SIZE))
    return np.asarray(img, dtype=np.uint8)


def human_weight(target):
    """按打满维数分级权重(饱和降权)。"""
    sat = int((np.abs(target[:7]) >= BOUNDS[:7] - 1e-4).sum())
    sat += int(abs(target[7]) >= BOUNDS[7] - 1e-4)
    return 1.0 if sat == 0 else (0.5 if sat == 1 else 0.2)


class Store:
    """双池重放缓冲(条目含 per-sample 权重)+ 磁盘分片持久化(v2 含 mid/sim_time/weight)。"""

    def __init__(self, data_dir, cap=200_000, shard=2000):
        self.lock = threading.Lock()
        self.human, self.auto = [], []          # 条目 (img, state, a_base, target, w)
        self.cap, self.shard = cap, shard
        self.data_dir = Path(data_dir); self.data_dir.mkdir(parents=True, exist_ok=True)
        self._pending = []
        self._shard_idx = len(list(self.data_dir.glob("shard_*.npz")))
        self.n_seen = {"human": 0, "auto": 0}

    def add(self, kind, img, state, a_base, target, w=1.0, mid="", sim_time=0.0, persist=True):
        rec = (img, state, a_base, target, np.float32(w))
        with self.lock:
            pool = self.human if kind == "human" else self.auto
            pool.append(rec)
            if len(pool) > self.cap:
                pool.pop(np.random.randint(len(pool) // 2))
            self.n_seen[kind] += 1
            if persist:
                self._pending.append((kind, img, state, a_base, target, np.float32(w), mid, np.float64(sim_time)))
                if len(self._pending) >= self.shard:
                    self._flush_locked()

    def _flush_locked(self):
        if not self._pending:
            return
        ks, im, st, ab, tg, w, mid, ts = zip(*self._pending)
        np.savez_compressed(
            self.data_dir / f"shard_{self._shard_idx:05d}.npz",
            kind=np.array(ks), img=np.stack(im), state=np.stack(st),
            a_base=np.stack(ab), target=np.stack(tg),
            weight=np.array(w), mid=np.array(mid), sim_time=np.array(ts))
        self._shard_idx += 1
        self._pending = []

    def flush(self):
        with self.lock:
            self._flush_locked()

    def sample(self, n_h, n_a):
        with self.lock:
            bh = [self.human[np.random.randint(len(self.human))] for _ in range(n_h)] if self.human else []
            ba = [self.auto[np.random.randint(len(self.auto))] for _ in range(n_a)] if self.auto else []
        return bh, ba

    def counts(self):
        with self.lock:
            return len(self.human), len(self.auto), dict(self.n_seen)

    def preload_human(self):
        """从既有 shard 载入人纠正样本(重算饱和权重;不重复落盘)。"""
        n = 0
        for f in sorted(self.data_dir.glob("shard_*.npz")):
            d = np.load(f, allow_pickle=True)
            for k, im, st, ab, tg in zip(d["kind"], d["img"], d["state"], d["a_base"], d["target"]):
                if k != "human":
                    continue
                self.add("human", im, st, ab, tg, w=human_weight(tg), persist=False)
                n += 1
        print(f"[hil] 预载人纠正样本 {n} 帧(饱和降权后)", flush=True)


def augment(img_u8):
    b, h, w, _ = img_u8.shape
    pad = 8
    x = torch.from_numpy(np.ascontiguousarray(img_u8)).float()
    x = torch.nn.functional.pad(x.permute(0, 3, 1, 2), (pad, pad, pad, pad), mode="reflect")
    ox = np.random.randint(0, 2 * pad + 1, size=b)
    oy = np.random.randint(0, 2 * pad + 1, size=b)
    out = torch.stack([x[i, :, oy[i]:oy[i] + h, ox[i]:ox[i] + w] for i in range(b)])
    scale = torch.empty(b, 1, 1, 1).uniform_(0.85, 1.15)
    return (out * scale).clamp(0, 255).permute(0, 2, 3, 1).to(torch.uint8)


class Trainer(threading.Thread):
    def __init__(self, model, store, args):
        super().__init__(daemon=True)
        self.model, self.store, self.args = model, store, args
        self.opt = torch.optim.Adam(model.trainable_parameters(), lr=args.lr)
        self.updates = 0
        self.serve_lock = threading.Lock()
        self.loss_ema = None
        self.ckpt_dir = Path(args.ckpt_dir); self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        dev = self.args.device
        while True:
            nh, na, _ = self.store.counts()
            if nh < self.args.min_human:
                time.sleep(2.0)
                continue
            bh, ba = self.store.sample(self.args.batch // 2, self.args.batch // 2)
            batch = bh + ba
            if not batch:
                time.sleep(1.0)
                continue
            w_entry = torch.tensor([float(r[4]) for r in batch], device=dev)
            w_kind = torch.tensor([1.0] * len(bh) + [self.args.w_auto] * len(ba), device=dev)
            img = augment(np.stack([r[0] for r in batch])).to(dev)
            st = torch.tensor(np.stack([r[1] for r in batch]), device=dev)
            ab = torch.tensor(np.stack([r[2] for r in batch]), device=dev)
            tg = torch.tensor(np.stack([r[3] for r in batch]), device=dev)
            with self.serve_lock:
                pred = self.model.head(self.model.encode(img), st, ab)
                per = torch.nn.functional.huber_loss(pred, tg, delta=0.02, reduction="none").mean(-1)
                loss = (per * w_entry * w_kind).mean() + 1e-3 * pred.pow(2).mean()
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()
            self.updates += 1
            l = float(loss.detach())
            self.loss_ema = l if self.loss_ema is None else 0.98 * self.loss_ema + 0.02 * l
            if self.updates % 50 == 0:
                print(f"[hil-train] upd={self.updates} loss_ema={self.loss_ema:.5f} "
                      f"buf(h/a)={nh}/{na}", flush=True)
            if self.updates % 200 == 0:
                RN.save_head(self.model, self.ckpt_dir / "head_latest.pt")
            if self.updates % 1000 == 0:
                RN.save_head(self.model, self.ckpt_dir / f"head_u{self.updates}.pt")

    @torch.no_grad()
    def infer(self, img, state, a_base):
        dev = self.args.device
        im = torch.from_numpy(np.ascontiguousarray(img[None])).to(dev)
        st = torch.tensor(state[None], device=dev)
        ab = torch.tensor(a_base[None], device=dev)
        with self.serve_lock:
            d = self.model(im, st, ab)
        return d[0].float().cpu().numpy()


def client_thread(conn, addr, store, trainer, args, ev_log):
    """v3 回合缓冲制:样本先攒在回合缓冲,录制器裁决 save 才入池;discard 整回合作废。
    隔离带规则在提交时应用:每段接管开始前 2s 内的自主帧丢弃。"""
    print(f"[hil] 客户端连接 {addr}", flush=True)
    ep_buf = []            # (mode, ts, img, st, ab, target, w)
    auto_ct = 0
    pending_commit = False  # record_stop 后等待裁决

    def commit(reason):
        nonlocal ep_buf
        if not ep_buf:
            return
        h_starts = [ep_buf[i][1] for i in range(len(ep_buf))
                    if ep_buf[i][0] == "human" and (i == 0 or ep_buf[i - 1][0] != "human")]
        n_h = n_a = n_drop = 0
        for mode, ts, img, st, ab, tgt, w in ep_buf:
            if mode == "human":
                store.add("human", img, st, ab, tgt, w=w, sim_time=ts)
                n_h += 1
            else:
                if any(0.0 <= hs - ts <= QUARANTINE_S for hs in h_starts):
                    n_drop += 1        # 失败前奏:接管前2s的自主帧作废
                    continue
                store.add("auto", img, st, ab, tgt, w=w, sim_time=ts)
                n_a += 1
        store.flush()
        print(f"[hil] 回合入池({reason}): human={n_h} auto={n_a} 隔离丢弃={n_drop}", flush=True)
        ep_buf = []

    try:
        while True:
            m = recv_msg(conn)
            if m is None:
                break
            if m["kind"] == "event":
                name = m["name"]
                if name == "record_start":
                    if pending_commit and ep_buf:
                        commit("兼容:未收到裁决,默认保存")
                    ep_buf = []; pending_commit = False
                elif name == "record_stop":
                    pending_commit = True
                elif name == "episode_save":
                    commit("save"); pending_commit = False
                elif name == "episode_discard":
                    print(f"[hil] 回合丢弃: {len(ep_buf)} 帧作废", flush=True)
                    ep_buf = []; pending_commit = False
                with open(ev_log, "a") as f:
                    f.write(json.dumps({"wall": time.time(), **{k: m[k] for k in ("name", "mid", "sim_time")}}) + "\n")
                send_msg(conn, {"ok": 1})
                continue
            img = decode_jpg(m["img"])
            st = np.asarray(m["state"], np.float32)
            ab = np.asarray(m["a_base"], np.float32)
            ts = float(m.get("sim_time") or 0.0)
            if m["mode"] == "human" and m.get("a_applied") is not None:
                tgt = np.clip(np.asarray(m["a_applied"], np.float32) - ab, -BOUNDS, BOUNDS)
                ep_buf.append(("human", ts, img, st, ab, tgt, human_weight(tgt)))
            elif m["mode"] == "auto":
                auto_ct += 1
                if auto_ct % args.auto_keep_every == 0:
                    ep_buf.append(("auto", ts, img, st, ab, np.zeros(8, np.float32), 1.0))
            delta = None
            if m.get("want_delta"):
                delta = (np.zeros(8, np.float32) if trainer.updates < args.serve_after
                         else trainer.infer(img, st, ab).astype(np.float32))
            send_msg(conn, {"delta": delta, "ver": trainer.updates})
    except (ConnectionResetError, BrokenPipeError):
        pass
    finally:
        conn.close()
        store.flush()
        print(f"[hil] 客户端断开 {addr}(未裁决缓冲 {len(ep_buf)} 帧作废)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5561)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--min-human", type=int, default=128)
    ap.add_argument("--serve-after", type=int, default=100)
    ap.add_argument("--w-auto", type=float, default=0.3)
    ap.add_argument("--auto-keep-every", type=int, default=3)
    ap.add_argument("--data-dir", default="/workspace/yangtong/gq/WorkSpace/EBiM/outputs/hil_data")
    ap.add_argument("--ckpt-dir", default="/workspace/yangtong/gq/WorkSpace/EBiM/outputs/hil_ckpt")
    ap.add_argument("--resume", default="")
    ap.add_argument("--preload-human", action="store_true",
                    help="从既有 shard 载入人纠正样本(重算饱和权重),自主池重建")
    args = ap.parse_args()

    print("[hil] 构建 DINOv2-S 残差头 ...", flush=True)
    model = RN.build(device=args.device)
    if args.resume:
        RN.load_head(model, args.resume, map_location=args.device)
        print(f"[hil] 已加载 {args.resume}", flush=True)
    store = Store(args.data_dir)
    if args.preload_human:
        store.preload_human()
    trainer = Trainer(model, store, args)
    trainer.start()
    ev_log = Path(args.data_dir) / "hil_events.jsonl"

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port)); srv.listen(4)
    print(f"[hil] v2 就绪,监听 {args.host}:{args.port} 隔离带={QUARANTINE_S}s "
          f"bounds={BOUNDS.tolist()}", flush=True)
    while True:
        conn, addr = srv.accept()
        threading.Thread(target=client_thread, daemon=True,
                         args=(conn, addr, store, trainer, args, ev_log)).start()


if __name__ == "__main__":
    main()
