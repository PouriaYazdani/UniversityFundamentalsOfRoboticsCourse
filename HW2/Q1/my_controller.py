from controller import Robot
from controller import GPS, Compass, Motor, PositionSensor
import math
import matplotlib.pyplot as plt

# ------------- PARAMETERS -------------
TIME_STEP = 32            # ms
MAX_TIME  = 3.0          # seconds to run both methods
WHEEL_RADIUS = 0.0205     # m (for odometry)
WHEEL_BASE   = 0.053      # m (distance between wheels)
# --------------------------------------

# initialize robot
robot = Robot()
robot.step(TIME_STEP)
# devices: GPS + compass
gps     = robot.getDevice('gps')
compass = robot.getDevice('compass')
gps.enable(TIME_STEP)
compass.enable(TIME_STEP)

# devices: wheel motors
left_motor  = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
for m in (left_motor, right_motor):
    m.setPosition(float('inf'))
    m.setVelocity(0.0)

# devices: wheel encoders for odometry
left_ps  = robot.getPositionSensor('left wheel sensor')
right_ps = robot.getPositionSensor('right wheel sensor')
left_ps.enable(TIME_STEP)
right_ps.enable(TIME_STEP)

# odometry state
prev_left  = left_ps.getValue()
prev_right = right_ps.getValue()
odom_x     = 0.0
odom_y     = 0.0
odom_th    = 0.0

# storage
time_vals      = []
gps_x, gps_y   = [], []
gps_th         = []
odom_xs, odom_ys, odom_ths = [], [], []

# main loop
t = 0.0
while robot.step(TIME_STEP) != -1 and t <= MAX_TIME:
    # 1) send identical wheel commands
    left_vel  = math.pi
    right_vel = -math.pi
    left_motor.setVelocity(left_vel)
    right_motor.setVelocity(right_vel)

    # 2) read GPS + compass
    pos   = gps.getValues()           # [x, y, z]
    north = compass.getValues()       # [x, y, z]
    heading = math.atan2(north[0], north[1])
    heading_deg = math.degrees(heading)


    # 3) compute odometry
    left_val  = left_ps.getValue()
    right_val = right_ps.getValue()
    dL = (left_val  - prev_left)  * WHEEL_RADIUS
    dR = (right_val - prev_right) * WHEEL_RADIUS

    dC     = (dR + dL) / 2.0
    dTheta = (dR - dL) / WHEEL_BASE
    odom_x += dC * math.cos(odom_th + dTheta/2.0)
    odom_y += dC * math.sin(odom_th + dTheta/2.0)
    odom_th += dTheta

    # wrap odom_th to [−π, π]
    if odom_th > math.pi:
        odom_th -= 2*math.pi
    elif odom_th < -math.pi:
        odom_th += 2*math.pi

    prev_left, prev_right = left_val, right_val

    # 4) store
    time_vals.append(t)
    gps_x.append(pos[0]);    gps_y.append(pos[1]);    gps_th.append(heading_deg)
    odom_xs.append(odom_x);  odom_ys.append(odom_y);  odom_ths.append(math.degrees(odom_th))

    # advance time
    t += TIME_STEP / 1000.0

# ---- PLOTTING ----
# 1) X–Y trajectories
plt.figure()
plt.plot(gps_x, gps_y,  label='GPS+Compass')
plt.plot(odom_xs, odom_ys, label='Odometry')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.title('Trajectory Comparison')
plt.axis('equal')
plt.legend()

# 2) Heading vs. Time
plt.figure()
plt.plot(time_vals, gps_th,   label='GPS Heading (°)')
plt.plot(time_vals, odom_ths, label='Odometry θ (°)')
plt.xlabel('Time [s]')
plt.ylabel('Angle [deg]')
plt.title('Heading Comparison over Time')
plt.legend()

plt.show()
