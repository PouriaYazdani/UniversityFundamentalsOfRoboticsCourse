import numpy as np
import matplotlib.pyplot as plt
from controller import Robot
import math

# === Parameters ===
TIME_STEP = 32
SPEED = 8.0
WHEEL_RADIUS = 0.0205
LINEAR_SPEED = SPEED * WHEEL_RADIUS

SAMPLE_INTERVAL = 0.1
GOAL_THRESHOLD = 0.1

START_POS = (1.5, -8.03)
END_POS = (1.5, 7.56)

SENSOR_COUNT = 8
SENSOR_ANGLES = np.deg2rad([-90, -60, -45, -10, 10, 45, 60, 90])
NUM_SAMPLE_CYCLES = 10

SENSOR_MAX_RANGE = 10
MAX_RAW_VALUE = 1000
SENSOR_ACTIVE = [True, False, True, False,False , True, False, False]  # length must match SENSOR_COUNT



def sample_sensors(distance_sensors):
    points = []
    for idx, ds in enumerate(distance_sensors):
        if not SENSOR_ACTIVE[idx]:
            continue
        raw = ds.getValue()
        if raw <= 1e-6:
            continue

        # Map raw value to a realistic "distance" – closer means higher raw
        # Here we invert it so that high raw = close distance
        # You can fine-tune this mapping for your environment
        dist = SENSOR_MAX_RANGE * (1.0 - raw / MAX_RAW_VALUE)

        if dist <= 0.01:  # too close or invalid
            continue

        θ = SENSOR_ANGLES[idx]
        x_r = dist * math.cos(θ)
        y_r = dist * math.sin(θ)
        points.append([x_r, y_r])
    return np.asarray(points)



def transform_points(points, robot_pose):
    if points.size == 0:
        return points
    x_r, y_r, θ = robot_pose
    c, s = math.cos(θ), math.sin(θ)
    rot = np.array([[c, -s], [s, c]])
    return points @ rot.T + np.array([x_r, y_r])


def get_yaw(compass):
    nx, ny, _ = compass.getValues()
    return math.atan2(nx, ny)


# === IEPF + Merge ===
def fit_line(points):
    x, y = points[:, 0], points[:, 1]
    if np.ptp(x) < 1e-6:
        return np.inf, x[0]
    m, c = np.polyfit(x, y, 1)
    return m, c


def point_line_dist(points, m, c):
    if m is np.inf:
        return np.abs(points[:, 0] - c)
    denom = math.hypot(m, 1.0)
    return np.abs(m * points[:, 0] - points[:, 1] + c) / denom


def iepf(points, threshold=0.01):
    segments = []

    def _split(seg_pts):
        n = len(seg_pts)
        if n <= 2:
            segments.append(np.vstack((seg_pts[0], seg_pts[-1])))
            return
        p1, p2 = seg_pts[0], seg_pts[-1]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        if abs(dx) < 1e-9:
            m, c = np.inf, p1[0]
            dists = np.abs(seg_pts[1:-1, 0] - c)
        else:
            m = dy / dx
            c = p1[1] - m * p1[0]
            denom = math.hypot(m, 1.0)
            dists = np.abs(m * seg_pts[1:-1, 0] - seg_pts[1:-1, 1] + c) / denom
        if len(dists) == 0:
            segments.append(np.vstack((p1, p2)))
            return
        i_rel = np.argmax(dists)
        d_max = dists[i_rel]
        if d_max <= threshold:
            segments.append(np.vstack((p1, p2)))
            return
        i_abs = i_rel + 1
        _split(seg_pts[: i_abs + 1])
        _split(seg_pts[i_abs:])

    _split(points)
    return segments


def merge_segments(segments, threshold=0.01):
    if not segments:
        return segments
    merged = [segments[0]]
    for seg in segments[1:]:
        candidate = np.vstack((merged[-1], seg))
        m, c = fit_line(candidate)
        if point_line_dist(candidate, m, c).max() <= threshold:
            merged[-1] = np.vstack((merged[-1][0], seg[-1]))
        else:
            merged.append(seg)
    return merged


# === Main controller ===
def main():
    robot = Robot()

    left_m = robot.getDevice("left wheel motor")
    right_m = robot.getDevice("right wheel motor")
    left_m.setPosition(float("inf"))
    right_m.setPosition(float("inf"))
    left_m.setVelocity(0.0)
    right_m.setVelocity(0.0)

    distance_sensors = []
    for i in range(SENSOR_COUNT):
        ps = robot.getDevice(f"ps{i}")
        if SENSOR_ACTIVE[i]:
            ps.enable(TIME_STEP)
        distance_sensors.append(ps)  # Still append for indexing to match SENSOR_ANGLES
        
    gps = robot.getDevice("gps")
    gps.enable(TIME_STEP)

    compass = robot.getDevice("compass")
    compass.enable(TIME_STEP)

    last_sample_x, last_sample_y = START_POS
    all_points = []
    moving = True
    sample_count = 0

    robot.step(TIME_STEP)

    while robot.step(TIME_STEP) != -1:
        g = gps.getValues()
        cur_x, cur_y = g[0], g[1]
        orientation = get_yaw(compass)
        pose = [cur_x, cur_y, orientation]

        if moving:
            dist = math.hypot(cur_x - last_sample_x, cur_y - last_sample_y)
            if dist >= SAMPLE_INTERVAL:
                moving = False
                sample_count = 0
                left_m.setVelocity(0.0)
                right_m.setVelocity(0.0)
                robot.step(TIME_STEP)
            else:
                left_m.setVelocity(SPEED)
                right_m.setVelocity(SPEED)
        else:
            pts = sample_sensors(distance_sensors)
            if pts.size:
                gps_pts = transform_points(pts, pose)
                all_points.extend(gps_pts)
            sample_count += 1
            if sample_count >= NUM_SAMPLE_CYCLES:
                last_sample_x, last_sample_y = cur_x, cur_y
                moving = True
                if math.hypot(cur_x - END_POS[0], cur_y - END_POS[1]) < GOAL_THRESHOLD:
                    break

    all_pts = np.array(all_points)
    if not len(all_pts):
        print("No points collected!")
        return

    center = all_pts.mean(axis=0)
    angs = np.arctan2(all_pts[:, 1] - center[1], all_pts[:, 0] - center[0])
    all_pts = all_pts[np.argsort(angs)]

    split_segs = iepf(all_pts, threshold=10)
    merged_segs = merge_segments(split_segs, threshold=0.02)

    # # --- Identify unexplained points (complement of merged lines) ---
    # explained_mask = np.zeros(len(all_pts), dtype=bool)

    # for seg in merged_segs:
    #     m, c = fit_line(seg)
    #     dists = point_line_dist(all_pts, m, c)
    #     explained_mask |= (dists <= 0.02)  # match threshold used in merge

    # complement_pts = all_pts[~explained_mask]  # not explained by any segment

    # # --- Plot ---
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # # IEPF (split-only) view
    # ax1.set_title("IEPF (Split)")
    # ax1.scatter(all_pts[:, 0], all_pts[:, 1], s=0.5, c="gray")
    # for seg in split_segs:
    #     ax1.plot(seg[:, 0], seg[:, 1], linewidth=0.3, color='blue')
    # ax1.set_aspect("equal")

    # # Split & Merge with Complement
    # ax2.set_title("Split & Merge + Complement")
    # ax2.scatter(all_pts[:, 0], all_pts[:, 1], s=0.5, c="gray", label="All Points")
    # ax2.scatter(complement_pts[:, 0], complement_pts[:, 1], s=3, c="red", label="Unexplained Points")
    # for seg in merged_segs:
    #     ax2.plot(seg[:, 0], seg[:, 1], linewidth=0.3, color='blue')
    # ax2.set_aspect("equal")
    # ax2.legend()

    # plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.set_title("IEPF (Split)")
    ax1.scatter(all_pts[:, 0], all_pts[:, 1], s=0.5, c="gray")
    for seg in split_segs:
        ax1.plot(seg[:, 0], seg[:, 1], linewidth=0.1)
    ax1.set_aspect("equal")

    ax2.set_title("Split & Merge")
    ax2.scatter(all_pts[:, 0], all_pts[:, 1], s=0.5, c="gray")
    for seg in merged_segs:
        ax2.plot(seg[:, 0], seg[:, 1], linewidth=0.1)
    ax2.set_aspect("equal")

    plt.show()



if __name__ == "__main__":
    main()
