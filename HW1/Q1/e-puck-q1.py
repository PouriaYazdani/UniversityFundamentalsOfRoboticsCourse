"""e-puck-q1 controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot

import matplotlib.pyplot as plt
import math

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# You should insert a getDevice-like function in order to get the
gps = robot.getDevice('gps')
gps.enable(timestep)

# instance of a device of the robot. Something like:
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))


left_motor.setVelocity(1.0)   # φ₁ = 1 rad/s
right_motor.setVelocity(-1.0) # φ₂ = -1 rad/s




# Simulation params
duration = 10  # seconds
steps = int((1000 * duration) / timestep)

# For storing positions
trajectory = []

# --- Choose the case (1 or 2) ---
CASE = 1

time = 0

for step in range(steps):
    if robot.step(timestep) == -1:
        break

    # Update velocities
    if CASE == 1:
        phi_right = 1.0           # φ₁ → right wheel
        phi_left = -1.0           # φ₂ → left wheel
    elif CASE == 2:
        phi_right = math.cos(2 * time)
        phi_left = -math.sin(time)

    right_motor.setVelocity(phi_right)
    left_motor.setVelocity(phi_left)

    # Save position
    pos = gps.getValues()
    trajectory.append((pos[0], pos[1]))  # X-Y plane 


    time += timestep / 1000.0  # time in seconds

# Stop robot
left_motor.setVelocity(0)
right_motor.setVelocity(0)

# Plotting
x_vals = [p[0] for p in trajectory]  # X
y_vals = [p[1] for p in trajectory]  # Y 

plt.figure(figsize=(6,6))
plt.plot(x_vals, y_vals, marker='o', markersize=2)
plt.title(f"Robot Trajectory in X-Y Plane (Case {CASE})")
plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.grid(True)
plt.axis('equal')
plt.show()
