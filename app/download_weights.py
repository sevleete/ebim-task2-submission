#!/usr/bin/env python3
"""首次运行时下载权重到 /weights(挂卷可跨容器缓存)。"""
import os
from huggingface_hub import snapshot_download
PI_REPO = os.environ.get("PI_HF_REPO", "Sevleete/ebim-task2-final")
W = "/weights/pi05"
if not os.path.exists(os.path.join(W, "model.safetensors")):
    print(f"[weights] downloading {PI_REPO} -> {W}", flush=True)
    snapshot_download(PI_REPO, local_dir=W)
print("[weights] pi05 ready", flush=True)
snapshot_download("facebook/dinov2-small")   # 残差头骨干(公开,~90MB)
print("[weights] dinov2 ready", flush=True)
