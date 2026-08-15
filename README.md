# EBiM Task-2 Submission — π0.5 + Residual Corrector (Team: Hajimi)

End-to-end autonomous solution for **Task 2 — Deformable Material Handling (Thermal Pad
Placement)**: the robot drives itself from spawn to the workbench, pins its base, poses the
arms, then a post-trained **π0.5** VLA (with a small frozen **residual corrector** for
interaction-phase精修) grasps the thermal pad and places it on the randomly-slotted target
RAM board, and finally retreats to clear the eval camera.

```
[organizer's Isaac Sim scene]  ⇄ ROS2 (/isaac/* topics, host network, FastDDS UDPv4)
        │
[this container]
  eval_prepare.py  (orchestrator)
    stage 1  navigate spawn→fixpos (waypoints + pulse fine-park) ∥ raise spine
    stage 2  base hold (P + pulse, pins base ≤1cm through the whole run)
    stage 3  arms → pre-grasp pose, grippers open
    stage 4  policy loop: π0.5 server (:5557) chunks + residual server (:5561) per-tick Δ
             a(t) = clip( a_π(t) + EMA(Δ(t)) ⊙ [±0.05 rad, ±0.15 grip] )
    stage 5  end = gripper closed-once → reopened 3 sim-s (or 120 sim-s timeout)
             → arms back to pre-grasp ∥ base retreats 10 cm (clears eval camera)
```

## Requirements on the evaluation side
- The Task-2 Isaac Sim scene running with the ROS2 bridge (`/isaac/*` topic contract of the
  official benchmark repo), reachable on the **host network** (FastDDS UDPv4).
  Needed topics: `/isaac/odom`, `/isaac/clock`, `/isaac/joint_states_full`,
  `/isaac/head_camera/image_raw`, `/isaac/right_wrist_camera/image_raw`,
  and the command topics (`base_cmd_vel`, `spine_command`, `left/right_joint_commands`,
  `left/right_robotiq_joint_commands`).
- One NVIDIA GPU (≥16 GB) for the π0.5 policy server + residual corrector.

## Build

```bash
docker build -t ebim-task2-sub .
```

## Run

```bash
docker run --rm --gpus all --network host --ipc host \
  -e PI_HF_REPO=Sevleete/ebim-task2-dagger-hil \
  -v $HOME/ebim_weights:/weights \
  ebim-task2-sub
```

- **Weights**: the π0.5 checkpoint (~8.8 GB) is downloaded from Hugging Face on first run
  into the mounted `/weights` volume (cached across runs). The residual corrector head
  (0.16 M params) ships inside the image. Weights repo: **https://huggingface.co/Sevleete/ebim-task2-dagger-hil**.
- The pipeline starts as soon as `/isaac/odom` is available, runs one full episode
  autonomously, retreats, and exits. Re-run the container for another episode.

## What's inside
| Path | Purpose |
|---|---|
| `app/prepare/` | Orchestrator (navigate / park / hold / pose / run policy / retreat) + config |
| `app/rollout/policy_server_pi.py` | π0.5 inference server (chunk 50 × 8-dim right-arm actions) |
| `app/hil/hil_server.py` + `residual_net.py` | Frozen residual corrector server (DINOv2-S frozen + 0.16 M head) |
| `app/hil/eval_hil_client.py` | 30 Hz ROS client: sim-clock-paced chunk execution + per-tick Δ correction |
| `app/vendor/lerobot/` | Vendored inference subset of LeRobot (Apache-2.0) with the π0.5 port |
| `weights/residual_head.pt` | Residual corrector weights (trained via human-in-the-loop interventions) |

## Method summary
1. **Data**: 134 keyboard-teleop demos + 103 DAgger/HIL takeover episodes + 34 grasp-specialist
   episodes, all collected under the evaluation randomization (±2 cm object jitter + random
   target slot + base pose jitter).
2. **Policy**: π0.5 post-trained in two rounds (BC → DAgger-aggregated resume), 8-dim
   right-arm joint actions, head + right-wrist cameras.
3. **Residual corrector**: frozen π + zero-initialized 0.16 M head on frozen DINOv2-S features,
   trained online from human takeover interventions (HG-DAgger-style selective BC),
   bounded ±0.05 rad — polishes grasp/placement millimetre-level errors.
