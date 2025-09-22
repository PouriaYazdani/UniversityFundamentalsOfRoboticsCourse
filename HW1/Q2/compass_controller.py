# epuck_turn.py  – spin the robot to the requested compass heading
# Webots R2021a+ | e‑puck | Python controller
import math, sys, time
from controller import Robot, Compass, Motor
import matplotlib.pyplot as plt   # Webots bundles matplotlib

# ---------------- configuration ----------------
TIME_STEP_MS = 32                      # default e‑puck basicTimeStep
MAX_SPEED   = 6.28                      # rads‑1 wheel limit in Webots
Kp, Ki, Kd  = 7.0, 1.7, 3           # PID gains – tune if needed
STEADY_ERR_THR = 2.0                    # [deg] acceptable steady‑state error
STEADY_TIME    = 2                    # [s] how long error must stay low
# ------------------------------------------------

target = 0
target = target % 360                   # normalise
print("hello")
robot     = Robot()
compass   = robot.getDevice("compass")
compass.enable(TIME_STEP_MS)

lmotor = robot.getDevice("left wheel motor")
rmotor = robot.getDevice("right wheel motor")
for m in (lmotor, rmotor):
    m.setPosition(float('inf'))         # velocity‑controlled
    m.setVelocity(0)

def heading_deg():
    # Compass returns a 3‑D vector pointing to north in robot coords.
    x, y, _ = compass.getValues()
    theta = math.degrees(math.atan2(x, y))   # atan2(x,z) -> CW from +Z
    return (theta + 360) % 360

def clamp(v, lo, hi): return max(lo, min(hi, v))

# --- control loop ----------------------------------------------------------
err_i = err_prev = 0.0
log_t, log_h = [], []
t0 = robot.getTime()
steady_start = None

while robot.step(TIME_STEP_MS) != -1:
    now    = robot.getTime()
    h      = heading_deg()
    log_t.append(now - t0)
    log_h.append(h)
    
    print("heading is"+str(h))
    # shortest signed angular error in [‑180, 180]
    err = ((target - h + 540) % 360) - 180
    dt  = TIME_STEP_MS / 1000.0
    err_i += err * dt
    derr  = (err - err_prev) / dt if dt else 0
    err_prev = err

    turn = Kp*err + Ki*err_i + Kd*derr      # +ve => spin left
    left  = clamp(-turn/180, -MAX_SPEED, MAX_SPEED)
    right = clamp( turn/180, -MAX_SPEED, MAX_SPEED)
    lmotor.setVelocity(left)
    rmotor.setVelocity(right)

    # termination test
    if abs(err) < STEADY_ERR_THR:
        steady_start = steady_start or now
        if now - steady_start > STEADY_TIME:   # stayed within band long enough
            break
    else:
        steady_start = None

# stop motors cleanly
lmotor.setVelocity(0); rmotor.setVelocity(0)

# --------------- post‑run diagnostics & plot -------------------------------
import numpy as np
h_arr = np.unwrap(np.radians(log_h)) * 180/np.pi  # unwrap for overshoot calc
overshoot = max(h_arr) - target if target >= h_arr[0] else target - min(h_arr)
settle_time = log_t[-1]

print(f"\nTarget={target:.1f}°  |  settletime≈{settle_time:.2f}s  "
      f"|  overshoot≈{overshoot:.1f}°")

plt.plot(log_t, log_h)
plt.title(f"e‑puck heading response to {target:.0f}° step")
plt.xlabel("time[s]")
plt.ylabel("heading[deg]")
plt.grid(True)
plt.show()
