#!/usr/bin/env bash
set -e
source /opt/ros/jazzy/setup.bash
export PYTHONPATH=/app/vendor:$PYTHONPATH
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp FASTDDS_BUILTIN_TRANSPORTS=UDPv4
python3 /app/download_weights.py

echo "[entrypoint] starting pi policy server (GPU, :5557)"
python3 /app/rollout/policy_server_pi.py --ckpt /weights/pi05 --host 127.0.0.1 --port 5557 &
echo "[entrypoint] starting residual server (frozen, :5561)"
( cd /app/hil && python3 hil_server.py --port 5561 --device cuda \
    --resume /app/weights/residual_head.pt --serve-after 0 --min-human 999999999 \
    --data-dir /tmp/hil_data --ckpt-dir /tmp/hil_ckpt ) &

for p in 5557 5561; do
  echo "[entrypoint] waiting :$p ..."
  until (exec 3<>/dev/tcp/127.0.0.1/$p) 2>/dev/null; do sleep 3; done
  echo "[entrypoint] :$p ready"
done
echo "[entrypoint] running eval pipeline"
exec python3 /app/prepare/eval_prepare.py --config /app/prepare/eval_pipeline_container.yaml
