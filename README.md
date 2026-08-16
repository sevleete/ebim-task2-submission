# EBiM Task-2 Submission — Team Hajimi

Submission container for **Task 2 — Deformable Material Handling (Thermal Pad Placement)**.
The container is a self-contained autonomous controller: it connects to the running Task-2
Isaac Sim scene over ROS 2, drives the robot from spawn to the workbench, performs the
thermal-pad placement, then retreats and exits.

## Build

```bash
docker build -t ebim-task2-hajimi .
```

## Run

Prerequisite: the official Task-2 scene is running with its ROS 2 bridge on the host network,
publishing the standard `/isaac/*` topics (robot cameras enabled and recording/state topics on,
i.e. the `--enable-robot-cameras` + `--publish-recording-topics` topic set; no arm-teleop mode).

```bash
docker run --rm --gpus all --network host --ipc host \
  -v $HOME/ebim_weights:/weights \
  ebim-task2-hajimi
```

- Requires one NVIDIA GPU (≥ 16 GB).
- Model weights are fetched automatically on first run from
  <https://huggingface.co/Sevleete/ebim-task2-final> into the mounted `/weights`
  volume (cached across runs). No manual steps needed.
- The container waits for `/isaac/odom`, runs one full episode autonomously
  (navigate → operate → retreat to clear the eval camera), and exits.
  Re-run the container for another episode.

## Topics used

| Direction | Topics |
|---|---|
| subscribe | `/isaac/odom`, `/isaac/clock`, `/isaac/joint_states_full`, `/isaac/head_camera/image_raw`, `/isaac/right_wrist_camera/image_raw` |
| publish | `/isaac/base_cmd_vel`, `/isaac/spine_command`, `/isaac/left_joint_commands`, `/isaac/right_joint_commands`, `/isaac/left_robotiq_joint_commands`, `/isaac/right_robotiq_joint_commands` |

## Contact

Team **Hajimi** — see the submission issue for the point of contact.
