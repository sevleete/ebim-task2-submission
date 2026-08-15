# EBiM Task-2 submission: pi0.5 (post-trained) + frozen residual corrector,
# full autonomous pipeline: navigate -> park&hold -> pose arms -> policy -> retreat.
# Connects to the organizer's Isaac Sim scene over ROS2 (/isaac/* topics, host network).
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8
# --- ROS 2 Jazzy (ros-base: rclpy + core msgs) ---
RUN apt-get update && apt-get install -y curl gnupg lsb-release && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main" \
      > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && apt-get install -y ros-jazzy-ros-base python3-pip ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# --- Python deps (torch cu128 + inference stack) ---
# PIP_PROXY is optional (speeds up builds behind restricted networks); leave unset normally.
ARG PIP_PROXY=""
COPY requirements.txt /tmp/requirements.txt
RUN http_proxy=$PIP_PROXY https_proxy=$PIP_PROXY \
    pip3 install --break-system-packages --no-cache-dir --retries 5 \
      torch torchvision --index-url https://download.pytorch.org/whl/cu128
# --ignore-installed: the ROS apt layer ships debian-packaged numpy/etc without RECORD
# files, which pip cannot uninstall; install fresh copies on top instead.
RUN http_proxy=$PIP_PROXY https_proxy=$PIP_PROXY \
    pip3 install --break-system-packages --no-cache-dir --retries 5 --ignore-installed \
      -r /tmp/requirements.txt

# --- App code (pipeline + servers + vendored lerobot subset) + residual head ---
COPY app /app
COPY weights/residual_head.pt /app/weights/residual_head.pt
COPY entrypoint.sh /entrypoint.sh
ENV PYTHONPATH=/app/vendor
# pi0.5 weights are pulled on first run from HF into /weights (mount a volume to cache):
#   -e PI_HF_REPO=<team>/ebim-task2-pi05  -v $HOME/ebim_weights:/weights
VOLUME /weights
ENTRYPOINT ["/entrypoint.sh"]
