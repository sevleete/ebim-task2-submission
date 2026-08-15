#!/usr/bin/env python
"""π0.5 推理服务端(跑在服务器 172.26.0.50 的 pi_env)。

双进程 rollout 的"大脑":加载训好的 pi05 checkpoint(用非 EMA;EMA 实测不行)+ 预/后处理器,
监听 TCP。每收到一份观测(23 维 state + 3 路原生图像),用 predict_action_chunk
出一整块 50 步动作(chunk_size=50×20,已反归一化),回给本地 ROS 客户端。

关键对齐(来自训练 config.json / normalizer):
  - 图像键需改名:head->base_0_rgb, wrist_left->left_wrist_0_rgb,
    wrist_right->right_wrist_0_rgb(训练时 rename_map 就是这样);
  - 图像 resize 到 224(resize_with_pad,与 pi05 内部一致),值 [0,1];
  - state 23 维(已去 EE),归一化器就是 23 维,pi05 内部再 pad 到 32;
  - action 20 维。

协议:8 字节大端长度前缀 + pickle。
  请求: {"state": float32[23], "images": {short_name: uint8 HWC}, "task": str}
  响应: float32[50, 20]  (已反归一化的动作块)

用法(一般由 run_pi_server.sh 调用):
  pixi run python policy_server_pi.py \
      --ckpt <.../checkpoints/last/pretrained_model> --host 0.0.0.0 --port 5557
"""
import argparse
import pickle
import socket
import struct

import numpy as np
import torch

TASK_DEFAULT = "Pick up the thermal pad and place it on the target RAM board."
# 模型图像键(短名)-> 客户端发来的短名
REV = {"base_0_rgb": "head",
       "left_wrist_0_rgb": "wrist_left",
       "right_wrist_0_rgb": "wrist_right"}


def recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        c = conn.recv(n - len(buf))
        if not c:
            return None
        buf += c
    return buf


def recv_msg(conn):
    hdr = recv_exact(conn, 8)
    if hdr is None:
        return None
    (ln,) = struct.unpack(">Q", hdr)
    return pickle.loads(recv_exact(conn, ln))


def send_msg(conn, obj):
    p = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(struct.pack(">Q", len(p)) + p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="pretrained_model(_ema) 目录")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5557)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--img-size", type=int, default=224)
    args = ap.parse_args()

    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.policies.common.vla_utils import resize_with_pad_torch

    print(f"[pi-server] 加载 pi05: {args.ckpt}", flush=True)
    policy = PI05Policy.from_pretrained(args.ckpt)
    policy.to(args.device)
    policy.eval()

    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.ckpt,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    img_keys = list(policy.config.image_features.keys())   # observation.images.base_0_rgb ...
    S = args.img_size
    print(f"[pi-server] image_features={img_keys}", flush=True)
    print(f"[pi-server] chunk_size={policy.config.chunk_size} "
          f"n_action_steps={policy.config.n_action_steps}", flush=True)

    @torch.no_grad()
    def infer(state_np, images_np, task):
        obs = {"observation.state": torch.from_numpy(state_np).float().unsqueeze(0)}
        for k in img_keys:
            short = k.split(".")[-1]                       # base_0_rgb / left_wrist_0_rgb ...
            client_key = REV.get(short, short)             # -> head / wrist_left / ...
            if client_key not in images_np:
                # 客户端没这路相机(如 8维模型不发 wrist_left):不放进 obs,
                # 让 pi05 内部走与训练一致的"缺失键→零图+mask=False"路径
                continue
            hwc = torch.from_numpy(images_np[client_key]).float().div(255.0)  # HWC [0,1]
            chw = hwc.permute(2, 0, 1)
            r = resize_with_pad_torch(chw, S, S).squeeze(0).clamp(0, 1)        # (3,224,224)
            obs[k] = r.unsqueeze(0)
        obs["task"] = [task]
        obs = pre(obs)
        chunk = policy.predict_action_chunk(obs)           # (1, 50, 20) 归一化
        chunk = post(chunk)                                # 反归一化
        return chunk.squeeze(0).float().cpu().numpy()      # (50, 20)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    print(f"[pi-server] 就绪,监听 {args.host}:{args.port}", flush=True)

    while True:
        conn, addr = srv.accept()
        print(f"[pi-server] 客户端连接 {addr}", flush=True)
        try:
            n = 0
            while True:
                req = recv_msg(conn)
                if req is None:
                    break
                chunk = infer(req["state"], req["images"], req.get("task", TASK_DEFAULT))
                send_msg(conn, chunk)
                n += 1
                print(f"[pi-server] 第 {n} 次推理 -> chunk {chunk.shape}", flush=True)
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            conn.close()
            print("[pi-server] 客户端断开,等待下一个", flush=True)


if __name__ == "__main__":
    main()
