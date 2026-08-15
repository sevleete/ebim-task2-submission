#!/usr/bin/env python
"""HIL 残差校正头:冻结 DINOv2-S 提特征 + 可训小头,输出有界 Δaction(8维,右臂7+右爪)。

设计要点(与方案一致):
  - 场景理解由 π 完成(a_base 是输入);本头只学"腕相机局部几何 → 微修正"。
  - DINOv2-S(21M)全程冻结,只训 head(~0.4M):patch tokens → 线性降维 →
    注意力池化 → concat[state8, a_base8] → MLP → tanh × bounds。
  - 输出层零初始化 → 初始 Δ≡0(= 原模型行为),只有人示范过纠正的状态才长出非零残差。

输入图像:126x126 uint8 RGB(126 = 14*9,DINOv2 patch 对齐)。
"""
import torch
import torch.nn as nn

IMG_SIZE = 126                       # 14 * 9
# Δ 上限:右臂 7 关节 ±0.05 rad,右爪开合度 ±0.15
DEFAULT_BOUNDS = [0.05] * 7 + [0.15]


class ResidualHead(nn.Module):
    def __init__(self, dino, feat_dim=384, tok_dim=64, hidden=256,
                 bounds=None):
        super().__init__()
        self.dino = dino                       # 冻结,eval
        for p in self.dino.parameters():
            p.requires_grad_(False)
        self.register_buffer(
            "bounds", torch.tensor(bounds or DEFAULT_BOUNDS, dtype=torch.float32))
        # ImageNet 归一化(DINOv2 预训练口径)
        self.register_buffer("im_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("im_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.tok_proj = nn.Linear(feat_dim, tok_dim)
        self.attn_q = nn.Parameter(torch.zeros(1, 4, tok_dim))   # 4 个可学查询做注意力池化
        self.mlp = nn.Sequential(
            nn.Linear(4 * tok_dim + 8 + 8, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 8),
        )
        nn.init.zeros_(self.mlp[-1].weight)    # 零初始化:初始输出恒 0
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.normal_(self.attn_q, std=0.02)

    def trainable_parameters(self):
        return [p for n, p in self.named_parameters()
                if p.requires_grad and not n.startswith("dino.")]

    @torch.no_grad()
    def encode(self, img_u8):
        """img_u8: [B,126,126,3] uint8 → patch tokens [B,N,384](冻结,无梯度)。"""
        x = img_u8.permute(0, 3, 1, 2).float().div_(255.0)
        x = (x - self.im_mean) / self.im_std
        out = self.dino(pixel_values=x)
        return out.last_hidden_state[:, 1:]     # 去 CLS,留 patch tokens

    def head(self, toks, state8, a_base8):
        """toks [B,N,384], state8/a_base8 [B,8] → Δ [B,8](已 tanh×bounds)。"""
        t = self.tok_proj(toks)                                   # [B,N,64]
        q = self.attn_q.expand(t.shape[0], -1, -1)                # [B,4,64]
        att = torch.softmax(q @ t.transpose(1, 2) / t.shape[-1] ** 0.5, dim=-1)
        pooled = (att @ t).flatten(1)                             # [B,4*64]
        h = torch.cat([pooled, state8, a_base8], dim=-1)
        return torch.tanh(self.mlp(h)) * self.bounds

    def forward(self, img_u8, state8, a_base8):
        return self.head(self.encode(img_u8), state8, a_base8)


def build(device="cuda", dtype=torch.float32, bounds=None):
    from transformers import Dinov2Model
    dino = Dinov2Model.from_pretrained("facebook/dinov2-small")
    m = ResidualHead(dino, bounds=bounds).to(device=device, dtype=dtype)
    m.dino.eval()
    return m


def save_head(model, path):
    sd = {k: v for k, v in model.state_dict().items() if not k.startswith("dino.")}
    torch.save({"head": sd}, path)


def load_head(model, path, map_location="cpu"):
    ck = torch.load(path, map_location=map_location)
    missing, unexpected = model.load_state_dict(ck["head"], strict=False)
    bad = [k for k in missing if not k.startswith("dino.")]
    assert not bad and not unexpected, f"load_head 键不匹配: missing={bad} unexpected={unexpected}"
    return model
