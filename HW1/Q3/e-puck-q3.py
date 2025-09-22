# === e-puck kinematic simulation controller ===
# Simulates motion based on v and omega

from controller import Robot
import math
import matplotlib.pyplot as plt

# === Initialize robot and devices ===
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Motors
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# === Robot parameters ===
wheel_radius = 0.0205  # meters
axle_length = 0.053    # meters

# === Simulation parameters ===
# You can switch between the two cases here
v = 0.04      # linear velocity (m/s)
omega = 0.2   # angular velocity (rad/s)

# Uncomment this for second case:
# v = 0.01
# omega = -0.3

simulation_time = 10  # seconds
dt = timestep / 1000  # convert ms to seconds

# === Calculate wheel speeds ===
omega_r = (2 * v + omega * axle_length) / (2 * wheel_radius)
omega_l = (2 * v - omega * axle_length) / (2 * wheel_radius)

print(f"Right wheel speed (rad/s): {omega_r:.2f}")
print(f"Left wheel speed (rad/s): {omega_l:.2f}")

# === Set wheel speeds ===
left_motor.setVelocity(omega_l)
right_motor.setVelocity(omega_r)

# === Initialize position and orientation ===
x = 0.0  # meters
y = 0.0  # meters
theta = 0.0  # radians (robot heading)

# Lists to store trajectory for plotting
x_list = []
y_list = []
t_list = []

# === Main simulation loop ===
elapsed_time = 0.0

while robot.step(timestep) != -1 and elapsed_time < simulation_time:
    # Update robot position using simple kinematics
    x += v * math.cos(theta) * dt
    y += v * math.sin(theta) * dt
    theta += omega * dt

    # Save data
    x_list.append(x)
    y_list.append(y)
    t_list.append(elapsed_time)

    elapsed_time += dt

# === After simulation, stop the robot ===
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# === Plot trajectory ===
plt.figure()
plt.plot(x_list, y_list, marker='o')
plt.xlabel('X (meters)')
plt.ylabel('Y (meters)')
plt.title('E-puck Robot Trajectory (X-Y plane)')
plt.grid(True)
plt.axis('equal')
plt.show()
