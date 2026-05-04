#!/bin/bash

export DISPLAY=:0
export XAUTHORITY=/home/amare-ub20046/.Xauthority

# ============================================================
# CONFIGURATION
# ============================================================
ARDUPILOT_DIR="/home/amare-ub20046/ardupilot"
QGC_PATH="/home/amare-ub20046/Downloads/QGroundControl.AppImage"
PROJECT_DIR="/home/amare-ub20046/PycharmProjects/Autonomous-Ai-drone"

# ============================================================
# 1. Start ArduPilot SITL
# ============================================================
echo "[1/4] Starting ArduPilot SITL..."
cd "$ARDUPILOT_DIR"
Tools/autotest/sim_vehicle.py -v ArduCopter -f quad --console --out tcp:127.0.0.1:5762 &

# Wait for SITL to be ready
echo "Waiting for SITL to initialize (port 5762)..."
for i in {1..30}; do
    if bash -c 'echo > /dev/tcp/127.0.0.1/5762' 2>/dev/null; then
        echo "SITL is ready!"
        break
    fi
    sleep 2
done

# ============================================================
# 2. Start QGroundControl
# ============================================================
echo "[2/4] Starting QGroundControl..."
"$QGC_PATH" &

# Give QGC time to connect
sleep 10

# ============================================================
# 3. Start the Drone Project
# ============================================================
echo "[3/4] Starting Autonomous Drone System..."
cd "$PROJECT_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
exec python autonomous_drone_main.py --mode sitl --control PID
