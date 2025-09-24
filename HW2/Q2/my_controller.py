from controller import Robot
import math
import csv
import numpy as np

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# === Motors ===
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
velocity = 3.0

# === Inertial Unit for orientation ===
imu = robot.getDevice('inertial unit')
imu.enable(timestep)

# === GPS ===
gps = robot.getDevice('gps')
gps.enable(timestep)

# === Custom Distance Sensors (ds0 to ds7) ===
sensor_names = [f'ds{i}' for i in range(8)]
sensor_angles_deg = [i * 45 for i in range(8)]  # 0°, 45°, ..., 315°
sensor_angles_rad = [math.radians(a) for a in sensor_angles_deg]

sensors = []
for name in sensor_names:
    sensor = robot.getDevice(name)
    sensor.enable(timestep)
    sensors.append(sensor)

# === Mapping Loop Parameters ===
step_counter = 0
STOP_INTERVAL = 10
target_y = 7.56
obstacle_points = []


# === Stop completely and rotate in place to map surroundings ===
left_motor.setVelocity(0)
right_motor.setVelocity(0)
robot.step(timestep)

# Get starting yaw angle
initial_yaw = imu.getRollPitchYaw()[2]
last_yaw = initial_yaw
angle_turned = 0.0
rotation_speed = 1.0  # rad/s wheel velocity
SAMPLE_INTERVAL_DEG = 5
sampled_angles = set()

# Start rotating (left wheel forward, right wheel backward)
left_motor.setVelocity(-rotation_speed)
right_motor.setVelocity(rotation_speed)

position = gps.getValues()
x_robot = position[0]
y_robot = position[1]  # In Webots, forward is along the Z-axis

while angle_turned < 2 * math.pi:
    if robot.step(timestep) == -1:
        break

    yaw = imu.getRollPitchYaw()[2]

    # Normalize yaw difference
    delta_yaw = yaw - last_yaw
    if delta_yaw > math.pi:
        delta_yaw -= 2 * math.pi
    elif delta_yaw < -math.pi:
        delta_yaw += 2 * math.pi

    angle_turned += abs(delta_yaw)
    last_yaw = yaw

    # Sample every SAMPLE_INTERVAL_DEG degrees
    current_angle_deg = math.degrees(angle_turned)
    rounded_angle = int(current_angle_deg // SAMPLE_INTERVAL_DEG) * SAMPLE_INTERVAL_DEG
    if rounded_angle not in sampled_angles:
        sampled_angles.add(rounded_angle)
        
        # Take sensor readings at this orientation
        for i, sensor in enumerate(sensors):
            raw_val = sensor.getValue()
            if raw_val == 1000:  # No obstacle detected
                continue
            distance = (raw_val / 1000.0) * 6.0  # Convert to meters
            angle = sensor_angles_rad[i]

            # Adjust angle with robot yaw
            yaw_corrected_angle = angle + yaw
            print(f"yaw_corrected_angle: {math.degrees(yaw_corrected_angle)}, Yaw: {math.degrees(yaw)}")
            dx = math.cos(yaw_corrected_angle) * distance
            dy = math.sin(yaw_corrected_angle) * distance

            x_obs = x_robot + dx
            y_obs = y_robot + dy
            obstacle_points.append([x_obs, y_obs, f"{sensor_names[i]}"])

            print(f"Rotating {rounded_angle}° | {sensor_names[i]}: Obstacle at ({x_obs:.2f}, {y_obs:.2f})")


# Stop rotation
left_motor.setVelocity(0)
right_motor.setVelocity(0)
robot.step(timestep)

# === Start moving ===
left_motor.setVelocity(velocity)
right_motor.setVelocity(velocity)

while robot.step(timestep) != -1:
    position = gps.getValues()
    x_robot = position[0]
    y_robot = position[1]  # In Webots, forward is along the Z-axis

    if y_robot >= target_y:
        break

    # Stop every N steps to take sensor readings
    if step_counter % STOP_INTERVAL == 0:
        left_motor.setVelocity(0)
        right_motor.setVelocity(0)

        # Wait 1 time step to stabilize
        robot.step(timestep)

        print(f"Step {step_counter}: Stopped at ({x_robot:.2f}, {y_robot:.2f})")
        # Take readings and project to world coordinates
        for i, sensor in enumerate(sensors):
            raw_val = sensor.getValue()
            if raw_val == 1000:  # No obstacle detected
                continue
            distance = (raw_val / 1000.0) * 6.0  # Convert to meters
            angle = sensor_angles_rad[i]

            dx = math.cos(angle + math.pi/2) * distance
            dy = math.sin(angle + math.pi/2) * distance

            x_obs = x_robot + dx
            y_obs = y_robot + dy
            obstacle_points.append([x_obs, y_obs, sensor_names[i]])
            
            print (f"{sensor_names[i]}: Distance: {distance}, "
                   f"Angle: {(math.degrees(angle) + 90) % 360:.2f}°, "
                   f"Obstacle Position: ({x_obs:.2f}, {y_obs:.2f})")
        
        robot.step(timestep)

        # Resume motion
        left_motor.setVelocity(velocity)
        right_motor.setVelocity(velocity)

    step_counter += 1
    

# === Stop completely and rotate in place to map surroundings ===
left_motor.setVelocity(0)
right_motor.setVelocity(0)
robot.step(timestep)

# Get starting yaw angle
initial_yaw = imu.getRollPitchYaw()[2]
last_yaw = initial_yaw
angle_turned = 0.0
rotation_speed = 1.0  # rad/s wheel velocity
SAMPLE_INTERVAL_DEG = 5
sampled_angles = set()

# Start rotating (left wheel forward, right wheel backward)
left_motor.setVelocity(-rotation_speed)
right_motor.setVelocity(rotation_speed)

position = gps.getValues()
x_robot = position[0]
y_robot = position[1]  # In Webots, forward is along the Z-axis

while angle_turned < 2 * math.pi:
    if robot.step(timestep) == -1:
        break

    yaw = imu.getRollPitchYaw()[2]

    # Normalize yaw difference
    delta_yaw = yaw - last_yaw
    if delta_yaw > math.pi:
        delta_yaw -= 2 * math.pi
    elif delta_yaw < -math.pi:
        delta_yaw += 2 * math.pi

    angle_turned += abs(delta_yaw)
    last_yaw = yaw

    # Sample every SAMPLE_INTERVAL_DEG degrees
    current_angle_deg = math.degrees(angle_turned)
    rounded_angle = int(current_angle_deg // SAMPLE_INTERVAL_DEG) * SAMPLE_INTERVAL_DEG
    if rounded_angle not in sampled_angles:
        sampled_angles.add(rounded_angle)
        
        # Take sensor readings at this orientation
        for i, sensor in enumerate(sensors):
            raw_val = sensor.getValue()
            if raw_val == 1000:  # No obstacle detected
                continue
            distance = (raw_val / 1000.0) * 6.0  # Convert to meters
            angle = sensor_angles_rad[i]

            # Adjust angle with robot yaw
            yaw_corrected_angle = angle + yaw
            print(f"yaw_corrected_angle: {math.degrees(yaw_corrected_angle)}, Yaw: {math.degrees(yaw)}")
            dx = math.cos(yaw_corrected_angle) * distance
            dy = math.sin(yaw_corrected_angle) * distance

            x_obs = x_robot + dx
            y_obs = y_robot + dy
            obstacle_points.append([x_obs, y_obs, f"{sensor_names[i]}"])

            print(f"Rotating {rounded_angle}° | {sensor_names[i]}: Obstacle at ({x_obs:.2f}, {y_obs:.2f})")


# Stop rotation
left_motor.setVelocity(0)
right_motor.setVelocity(0)
robot.step(timestep)

# === Save Obstacle Map ===
with open("environment_map.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x", "y", "sensor"])
    writer.writerows(obstacle_points)

print("Environment map saved to environment_map.csv")
