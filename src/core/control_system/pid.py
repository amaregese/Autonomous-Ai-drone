from simple_pid import PID

from src.core.control_system import state
from src.core.control_system.config import D_ROLL, D_YAW, I_ROLL, I_YAW, MAX_SPEED, MAX_YAW, P_ROLL, P_YAW


def configure_PID(control):
    print("Configuring control")
    if control == "PID":
        state.pidYaw = PID(P_YAW, I_YAW, D_YAW, setpoint=0)
        state.pidRoll = PID(P_ROLL, I_ROLL, D_ROLL, setpoint=0)
        print("Configuring PID")
    else:
        state.pidYaw = PID(P_YAW, 0, 0, setpoint=0)
        state.pidRoll = PID(P_ROLL, 0, 0, setpoint=0)
        print("Configuring P")

    state.pidYaw.output_limits = (-MAX_YAW, MAX_YAW)
    state.pidRoll.output_limits = (-MAX_SPEED, MAX_SPEED)

