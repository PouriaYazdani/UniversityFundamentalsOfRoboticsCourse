import numpy as np
import cv2
import heapq
import json
import yaml
from controller import Robot, Motor, DistanceSensor, Camera, RangeFinder, InertialUnit

# =================== grid <-> world ===================
def world_to_grid(xw, yw, origin, resolution, h, w):  # FIX: Add w param
    if np.isnan(xw) or np.isnan(yw) or np.isinf(xw) or np.isinf(yw):
        return -1, -1
    col = int((xw - origin[0]) / resolution)
    row = h - 1 - int((yw - origin[1]) / resolution)
    if col < 0 or col >= w or row < 0 or row >= h:
        return -1, -1
    return row, col

def grid_to_world(row, col, origin, resolution, h):
    xw = origin[0] + (col + 0.5) * resolution
    yw = origin[1] + ((h - 1 - row) + 0.5) * resolution
    return xw, yw

# =================== A* ===================
def astar(grid, start, goal):
    rows, cols = grid.shape
    if not (0 <= start[0] < rows and 0 <= start[1] < cols): return None
    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols): return None
    if grid[int(start[0]), int(start[1])] != 0 or grid[int(goal[0]), int(goal[1])] != 0: return None
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    g_score = {start: 0}
    def h(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
    while open_list:
        _, current = heapq.heappop(open_list)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = current[0]+dx, current[1]+dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[int(nx),int(ny)] == 0:
                ng = g_score[current] + 1
                nxt = (nx, ny)
                if nxt not in g_score or ng < g_score[nxt]:
                    g_score[nxt] = ng
                    heapq.heappush(open_list, (ng + h(nxt, goal), nxt))
                    came_from[nxt] = current
    return None

# =================== Particle Filter ===================
class ParticleFilter:
    def __init__(self, num_particles, occ_grid, resolution, origin, h, w):  # FIX: Pass h,w
        self.num_particles = num_particles
        self.resolution = resolution
        self.origin = origin
        self.h = h
        self.w = w
        self.grid_occ = occ_grid
        self.Q = np.array([0.01, 0.01, 0.005])
        self.sigma_ds = 0.08
        self.particles = np.zeros((num_particles, 3))
        self.weights = np.ones(num_particles) / num_particles

        free_cells = np.argwhere(self.grid_occ == 0)
        if len(free_cells) == 0:
            self.particles[:, :2] = np.random.uniform(-10, 10, (num_particles, 2))
        else:
            idx = np.random.choice(len(free_cells), self.num_particles, replace=True)
            for k, (row, col) in enumerate(free_cells[idx]):
                x, y = grid_to_world(row, col, self.origin, self.resolution, self.h)
                self.particles[k, 0] = x
                self.particles[k, 1] = y
                self.particles[k, 2] = np.random.uniform(-np.pi, np.pi)

    def reset_with_qr(self, rx, ry, theta):
        noise_std = 0.05
        self.particles[:, 0] = rx + np.random.normal(0, noise_std, self.num_particles)
        self.particles[:, 1] = ry + np.random.normal(0, noise_std, self.num_particles)
        self.particles[:, 2] = theta + np.random.normal(0, 0.02, self.num_particles)
        self.weights[:] = 1.0 / self.num_particles
        # print(f"[QR RESET] Pose=({rx:.2f},{ry:.2f},{theta:.2f})")

    def predict(self, u):
        noise = np.random.randn(self.num_particles, 3) * self.Q
        self.particles += u + noise
        self.particles[:, 2] = np.arctan2(np.sin(self.particles[:, 2]), np.cos(self.particles[:, 2]))

    def update_ds_all(self, z_dict):
        likelihood = np.ones(self.num_particles)
        for ang, z in z_dict.values():
            exp = []
            for p in self.particles:
                exp.append(self._raycast(p[0], p[1], p[2] + ang))
            exp = np.array(exp)
            diff = z - exp
            likelihood *= np.exp(-0.5 * (diff / self.sigma_ds)**2)
        self.weights *= likelihood
        if np.sum(self.weights) == 0 or np.isnan(np.sum(self.weights)):
            self.weights[:] = 1.0 / self.num_particles
        else:
            self.weights /= self.weights.sum()
        self.resample()

    def update_imu(self, imu_yaw):
        orientation_diff = np.abs(self.particles[:, 2] - imu_yaw)
        orientation_diff = np.minimum(orientation_diff, 2 * np.pi - orientation_diff)
        imu_sigma = 0.05
        imu_likelihood = np.exp(-0.5 * (orientation_diff / imu_sigma)**2)
        self.weights *= imu_likelihood
        if np.sum(self.weights) == 0 or np.isnan(np.sum(self.weights)):
            self.weights[:] = 1.0 / self.num_particles
        else:
            self.weights /= self.weights.sum()
        eff = 1.0 / np.sum(self.weights**2)
        if eff < self.num_particles / 2:
            self.resample()

    def update_without_qr(self, z_dict):
        zL, zR, zF, zB = [z for _, z in z_dict.values()]
        avg_length = (zL + zR + zF + zB) / 4
        est_area = avg_length ** 2
        likelihood = np.ones(self.num_particles)
        for i, p in enumerate(self.particles):
            local_lengths = [self._raycast(p[0], p[1], p[2] + ang) for ang in [0, np.pi/2, np.pi, -np.pi/2]]
            local_avg = np.mean(local_lengths)
            local_area = local_avg ** 2
            diff = abs(local_area - est_area)
            sigma_area = 0.5
            likelihood[i] = np.exp(-0.5 * (diff / sigma_area)**2)
        self.weights *= likelihood
        if np.sum(self.weights) == 0 or np.isnan(np.sum(self.weights)):
            self.weights[:] = 1.0 / self.num_particles
        else:
            self.weights /= self.weights.sum()
        self.resample()

    def _raycast(self, x, y, theta, max_range=3.0):
        step = self.resolution
        r = 0.0
        while r < max_range:
            xx = x + r * np.cos(theta)
            yy = y + r * np.sin(theta)
            row, col = world_to_grid(xx, yy, self.origin, self.resolution, self.h, self.w)
            if row < 0 or col < 0 or row >= self.h or col >= self.w:
                return max_range
            if self.grid_occ[row, col] == 1: return r
            r += step
        return max_range

    def resample(self):
        eff = 1.0 / np.sum(self.weights**2)
        if eff < self.num_particles / 3:
            idx = np.random.choice(np.arange(self.num_particles), size=self.num_particles, p=self.weights)
            self.particles = self.particles[idx]
            self.weights[:] = 1.0 / self.num_particles

    def get_est(self):
        m = np.average(self.particles, weights=self.weights, axis=0)
        if np.any(np.isnan(m)):
            m = np.mean(self.particles, axis=0)
        m[2] = np.arctan2(np.sin(m[2]), np.cos(m[2]))
        return m

# =================== Init Webots ===================
TIME_STEP = 64
robot = Robot()
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
for m in [left_motor, right_motor]:
    m.setPosition(float("inf"))
    m.setVelocity(0.0)

ds_left = robot.getDevice("ds_left");   ds_left.enable(TIME_STEP)
ds_right = robot.getDevice("ds_right"); ds_right.enable(TIME_STEP)
ds_front = robot.getDevice("ds_front"); ds_front.enable(TIME_STEP)
ds_back = robot.getDevice("ds_back");   ds_back.enable(TIME_STEP)
camera = robot.getDevice("qr_camera");  camera.enable(TIME_STEP)
range_finder = robot.getDevice("qr_rangefinder"); range_finder.enable(TIME_STEP)
imu = robot.getDevice("imu"); imu.enable(TIME_STEP)
l_enc = robot.getDevice('left wheel sensor'); r_enc = robot.getDevice('right wheel sensor')
l_enc.enable(TIME_STEP); r_enc.enable(TIME_STEP)

WHEEL_RADIUS = 0.033
WHEEL_BASE   = 0.16
MAX_SENSOR_RANGE = 5.0
DEBUG_DS = True
WAYPOINT_RADIUS = 0.20
GOAL_RADIUS     = 1.0

# =================== Load Map & Doors ===================
pgm = cv2.imread("point_cloud_run.pgm", cv2.IMREAD_GRAYSCALE)
occ = (pgm < 128).astype(np.uint8)
with open("point_cloud_run.yaml") as f:
    yml = yaml.safe_load(f)
origin = yml["origin"][:2]
resolution = yml["resolution"]
h, w = occ.shape
with open("point_cloud_run_doors.json") as f:
    doors = json.load(f)["doors"]
landmarks = {d["id"]: (d["x"], d["y"]) for d in doors}

print(f"[DEBUG] Loaded landmarks: {landmarks}")
pf = ParticleFilter(1000, occ, resolution, origin, h, w)

# =================== Trajectory logging & rendering ===================
traj_pix = []          # list of (row, col) pixels along the path
_saved_traj = False    # guard to save only once

def _traj_push_from_est():
    """Append current PF estimate as a pixel (row,col) if valid & changed."""
    est = pf.get_est()
    row, col = world_to_grid(est[0], est[1], origin, resolution, h, w)
    if row >= 0:
        if not traj_pix or (row, col) != traj_pix[-1]:
            traj_pix.append((row, col))

def _traj_save_overlay(out_path="trajectory_on_map.png"):
    """Render thin blue trajectory over the original map, mark start/end."""
    if not traj_pix:
        print("[TRAJ] No trajectory points to draw.")
        return
    img = cv2.cvtColor(pgm.copy(), cv2.COLOR_GRAY2BGR)
    pts = np.array([[c, r] for (r, c) in traj_pix], dtype=np.int32)
    cv2.polylines(img, [pts], False, (255, 0, 0), 1, cv2.LINE_AA)  # blue path
    # start (green) and end (red)
    r0, c0 = traj_pix[0]
    r1, c1 = traj_pix[-1]
    cv2.circle(img, (c0, r0), 3, (0, 255, 0), -1)  # start
    cv2.circle(img, (c1, r1), 3, (0,   0, 255), -1)  # end
    cv2.imwrite(out_path, img)
    print(f"[TRAJ] Saved trajectory overlay → {out_path}")

def _traj_save_once():
    global _saved_traj
    if not _saved_traj:
        _traj_save_overlay()
        _saved_traj = True

# =================== Helpers ===================
robot.step(TIME_STEP)
last_l, last_r = l_enc.getValue(), r_enc.getValue()
_traj_push_from_est()  # log initial pose

def get_imu_orientation():
    _, _, yaw = imu.getRollPitchYaw()
    return yaw

def odom_step():
    global last_l, last_r
    l = l_enc.getValue()
    r = r_enc.getValue()
    dl = (l - last_l) * WHEEL_RADIUS
    dr = (r - last_r) * WHEEL_RADIUS
    last_l, last_r = l, r
    d = (dl + dr) / 2
    dth = (dr - dl) / WHEEL_BASE
    dx = d * np.cos(pf.get_est()[2] + dth / 2)
    dy = d * np.sin(pf.get_est()[2] + dth / 2)
    pf.predict(np.array([dx, dy, dth]))
    _traj_push_from_est()  # <-- log trajectory after motion update

# --- Replace the body of ds_update() with this (same function name/signature) ---
def ds_update():
    # Raw sensor units
    rawL = ds_left.getValue()
    rawR = ds_right.getValue()
    rawF = ds_front.getValue()
    rawB = ds_back.getValue()

    zL = (rawL / 1000.0) * MAX_SENSOR_RANGE
    zR = (rawR / 1000.0) * MAX_SENSOR_RANGE
    zF = (rawF / 1000.0) * MAX_SENSOR_RANGE
    zB = (rawB / 1000.0) * MAX_SENSOR_RANGE

    if DEBUG_DS:
        est = pf.get_est()
        print(f"[DS RAW]   L={rawL:.1f}  R={rawR:.1f}  F={rawF:.1f}  B={rawB:.1f}")
        print(f"[DS METERS]L={zL:.2f}m R={zR:.2f}m F={zF:.2f}m B={zB:.2f}m | Currently estiamted Pose≈({est[0]:.2f},{est[1]:.2f},{np.degrees(est[2]):.1f}°)")

    z_dict = {"L": (np.pi / 2, zL), "R": (-np.pi / 2, zR), "F": (0, zF), "B": (np.pi, zB)}
    return z_dict

def ds_debug_compare(z_dict):
    if not DEBUG_DS:
        return
    est = pf.get_est()
    exp = {k: pf._raycast(est[0], est[1], est[2] + ang) for k, (ang, _) in z_dict.items()}
    meas = {k: z for k, (_, z) in z_dict.items()}
    line = " | ".join([f"{k}: {meas[k]:.2f} vs {exp[k]:.2f} (Δ={meas[k]-exp[k]:+.2f})" for k in ["F","L","R","B"]])
    print(f"[DS vs EXP] {line}")

imu_update_counter = 0
def imu_update():
    global imu_update_counter
    imu_update_counter += 1
    if imu_update_counter % 2 == 0:
        imu_yaw = get_imu_orientation()
        pf.update_imu(imu_yaw)

ds_update_counter = 0
def maybe_ds_update():
    global ds_update_counter
    ds_update_counter += 1
    if ds_update_counter % 15 == 0:
        z_dict = ds_update()
        ds_debug_compare(z_dict)
        pf.update_ds_all(z_dict)
        pf.update_without_qr(z_dict)

# =================== QR Detection ===================
def detect_qr(camera):
    width = camera.getWidth()
    height = camera.getHeight()
    raw_image = camera.getImage()
    image_data = np.frombuffer(raw_image, dtype=np.uint8).reshape((height, width, 4))
    image_gray = cv2.cvtColor(image_data, cv2.COLOR_RGBA2GRAY)
    detector = cv2.QRCodeDetector()
    retval, decoded_info, points, _ = detector.detectAndDecodeMulti(image_gray)
    if retval and len(decoded_info) > 0:
        for i, obj in enumerate(decoded_info):
            if obj:
                rel_bearing = 0.0
                if points is not None and len(points) > i:
                    pts = points[i].reshape(-1, 2)
                    center = np.mean(pts, axis=0)
                    center_x = float(center[0])
                    width = camera.getWidth()
                    hfov = camera.getFov()
                    nx = (center_x - width / 2.0) / (width / 2.0)
                    rel_bearing = -0.5 * hfov * np.clip(nx, -1.0, 1.0)
                print(f"[QR DETECT] Found QR={obj}, rel_bearing={np.degrees(rel_bearing):.1f}°")
                return {'number': obj, 'bearing': rel_bearing}
    return None

def normalize_qr_id(qr_id):
    try:
        normalized = str(int(qr_id))
        return normalized
    except (ValueError, TypeError):
        return str(qr_id)

def qr_update():
    det = detect_qr(camera)
    if not det:
        return
    qr_id = det['number']
    normalized_id = normalize_qr_id(qr_id)
    rel_bearing = det.get('bearing', 0.0)
    if normalized_id in landmarks:
        range_data = range_finder.getRangeImage()
        observed_range = 2.0
        if range_data and len(range_data) > 0:
            center_idx = len(range_data) // 2
            observed_range = range_data[center_idx]
            if observed_range == float('inf') or observed_range > 5.0:
                observed_range = 2.0
        landmark_x, landmark_y = landmarks[normalized_id]
        imu_yaw = get_imu_orientation()
        qr_abs_bearing = imu_yaw + rel_bearing
        dx = observed_range * np.cos(qr_abs_bearing)
        dy = observed_range * np.sin(qr_abs_bearing)
        new_x = landmark_x - dx
        new_y = landmark_y - dy
        new_theta = imu_yaw
        print(f"[QR DEBUG] 🚪 Door {normalized_id} at ({landmark_x:.2f},{landmark_y:.2f}), Range={observed_range:.2f}m, Rel Bearing={np.degrees(rel_bearing):.1f}°")
        print(f"[QR RESET] 📍 Calculated position: ({new_x:.2f},{new_y:.2f}), theta={np.degrees(new_theta):.1f}°")
        pf.reset_with_qr(new_x, new_y, new_theta)
        _traj_push_from_est()  # log the jump as a trajectory point
        global replan_pending
        replan_pending = True
        print(f"[REPLAN] 🧠 New path after QR reset.")
    else:
        print(f"[QR ERROR] QR ID {qr_id} (normalized: {normalized_id}) not found in landmarks!")

# =================== Navigation ===================
def plan_to_goal(goal_id):
    est = pf.get_est()
    start = world_to_grid(est[0], est[1], origin, resolution, h, w)
    if start[0] == -1: return None
    gx, gy = landmarks[goal_id]
    goal = world_to_grid(gx, gy, origin, resolution, h, w)
    if goal[0] == -1: return None
    path = astar(occ, start, goal)
    if path:
        print(f"[PLAN] Path found with {len(path)} points")
    else:
        print(f"[PLAN] No path to goal {goal_id}")
    return path

# --- Obstacle avoidance tuning ---
AVOID_FRONT_DIST    = 0.25
CLEAR_FRONT_DIST    = 0.45
BACK_SPEED          = -0.08
FWD_SPEED           =  0.10
TURN_RATE           =  1.6
ARC_RATE            =  0.6
BACK_TIME_MS        = 600
TURN_TIMEOUT_MS     = 1800
SIDESTEP_TIME_MS    = 500

def _ratio_scaled_wheels(v, w):
    vl = (v - w * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    vr = (v + w * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    max_vel = 6.67
    scale = max(1.0, max(abs(vl), abs(vr)) / max_vel)
    return vl / scale, vr / scale

def _step_drive(v, w, dt_ms):
    t_end = robot.getTime() + dt_ms / 1000.0
    while robot.getTime() < t_end:
        vl, vr = _ratio_scaled_wheels(v, w)
        left_motor.setVelocity(vl)
        right_motor.setVelocity(vr)
        if robot.step(TIME_STEP) == -1:
            break
        odom_step()

def _front_dist_m():
    return (ds_front.getValue() / 1000.0) * MAX_SENSOR_RANGE

def _side_dists_m():
    zL = (ds_left.getValue()  / 1000.0) * MAX_SENSOR_RANGE
    zR = (ds_right.getValue() / 1000.0) * MAX_SENSOR_RANGE
    return zL, zR

total_steps = 0
def move_along(path, lookahead=15, v_lin=0.5, k_ang=1.5):
    global replan_counter, total_steps, replan_pending
    if not path or len(path) < 2: return False
    path_idx = 0
    max_steps_per_segment = 300
    print_counter = 0
    while path_idx < len(path) - 1 and total_steps < 2000:
        row, col = path[path_idx]
        local_goal_x, local_goal_y = grid_to_world(row, col, origin, resolution, h)
        est = pf.get_est()
        dist = np.hypot(local_goal_x - est[0], local_goal_y - est[1])
        print_counter += 1
        if print_counter % 1 == 0:
            print(f"[MOVE] Waypoint {path_idx}: ({local_goal_x:.2f},{local_goal_y:.2f}), Dist={dist:.3f}m")
        radius = GOAL_RADIUS if path_idx == len(path) - 1 else WAYPOINT_RADIUS
        if dist < radius:
            path_idx += 1
            continue
        for s in range(max_steps_per_segment):
            gx, gy = landmarks[goal_id]
            est_x, est_y, _ = pf.get_est()
            goal_dist = float(np.hypot(gx - est_x, gy - est_y))
            if goal_dist < GOAL_RADIUS:
                left_motor.setVelocity(0.0)
                right_motor.setVelocity(0.0)
                print(f"[GOAL] Reached final door {goal_id} at ({gx:.2f},{gy:.2f}); dist={goal_dist:.3f} m")
                global DONE
                DONE = True
                return True
            total_steps += 1
            est = pf.get_est()
            dx, dy = local_goal_x - est[0], local_goal_y - est[1]
            th_des = np.arctan2(dy, dx)
            ang_err = np.arctan2(np.sin(th_des - est[2]), np.cos(th_des - est[2]))
            dist = np.hypot(dx, dy)
            if _front_dist_m() < AVOID_FRONT_DIST:
                print(f"[AVOID] ⚠️ Front blocked: { _front_dist_m():.2f} m → backing up")
                left_motor.setVelocity(0.0); right_motor.setVelocity(0.0)
                _step_drive(BACK_SPEED, 0.0, BACK_TIME_MS)
                zL, zR = _side_dists_m()
                turn_dir = +1.0 if zL >= zR else -1.0
                print(f"[AVOID] zL={zL:.2f} zR={zR:.2f} → turn_dir={'LEFT' if turn_dir>0 else 'RIGHT'}")
                t0 = robot.getTime()
                while (robot.getTime() - t0) * 1000.0 < TURN_TIMEOUT_MS:
                    _step_drive(0.0, turn_dir * TURN_RATE, TIME_STEP)
                    if _front_dist_m() > CLEAR_FRONT_DIST:
                        break
                print(f"[AVOID] Turn done. Front={_front_dist_m():.2f} m")
                _step_drive(FWD_SPEED, turn_dir * ARC_RATE, SIDESTEP_TIME_MS)
                left_motor.setVelocity(0.0); right_motor.setVelocity(0.0)
                replan_pending = True
                print("[AVOID-REPLAN] 🧠 Replan requested after escape")
                break
            if abs(ang_err) > 0.5:
                v = 0.0
                w = k_ang * ang_err
            else:
                v = min(0.10, v_lin * dist)
                w = k_ang * ang_err
            vl = (v - w * WHEEL_BASE / 2) / WHEEL_RADIUS
            vr = (v + w * WHEEL_BASE / 2) / WHEEL_RADIUS
            max_vel = 6.67
            scale = max(1.0, max(abs(vl), abs(vr)) / max_vel)
            vl /= scale; vr /= scale
            if total_steps % 5 == 0:
                print(f"[CTRL] 🛂🛂  v={v:.2f} m/s w={w:.2f} rad/s, vl={vl}m/s, vr={vr}m/s, angle_error={ang_err}rad, dist={dist}m")
            left_motor.setVelocity(vl)
            right_motor.setVelocity(vr)
            if robot.step(TIME_STEP) == -1: break
            odom_step()
            imu_update()
            maybe_ds_update()
            qr_update()
            if replan_pending:
                print("🧠🧠🧠🧠🧠🧠🧠")
                new_path = plan_to_goal(goal_id)
                if new_path and len(new_path) > 1:
                    path = new_path
                    path_idx = 0
                    print(f"[REPLAN] 🧠 New path with {len(path)} points after QR reset.")
                else:
                    print(f"[REPLAN] ❌ Failed")
                replan_pending = False
                break
            new_dist = np.hypot(local_goal_x - pf.get_est()[0], local_goal_y - pf.get_est()[1])
            radius = GOAL_RADIUS if path_idx == len(path) - 1 else WAYPOINT_RADIUS
            if new_dist < radius and dist > 1.0:
                pass
            elif new_dist < radius:
                path_idx += 1
                print(f"[MOVE] ✅ Reached waypoint {path_idx-1} in {s+1} steps, dist was={new_dist:.3f}m, Now Goal is waypoint {path_idx}")
                break
        left_motor.setVelocity(0)
        right_motor.setVelocity(0)
    return path_idx >= len(path) - 1

# =================== Main ===================
goal_id = "5"
print(f"[DEBUG] Goal door {goal_id} position: {landmarks.get(goal_id, 'NOT FOUND')}")
current_path = None
replan_counter = 0
replan_pending = False
DONE = False

while robot.step(TIME_STEP) != -1:
    odom_step()
    imu_update()
    maybe_ds_update()
    qr_update()

    gx, gy = landmarks[goal_id]
    est = pf.get_est()
    dist = np.hypot(gx - est[0], gy - est[1])
    imu_yaw = get_imu_orientation()
    if total_steps % 20 == 0:
        print(f"[POSE] ({est[0]:.2f},{est[1]:.2f},{est[2]:.2f}) IMU_yaw={imu_yaw:.2f} → Goal {goal_id} Dist={dist:.2f}")
        var = np.var(pf.particles, axis=0)
        print(f"[PF VAR] Pos/Theta variance: ({var[0]:.3f}, {var[1]:.3f}, {np.degrees(var[2]):.1f}°)")

    if dist < 0.4 or DONE:
        print("Goal Reached")
        _traj_save_once()
        break

    replan_counter += 1
    if current_path is None or replan_counter > 50 or len(current_path) < 2:
        print(f"[REPLAN] from main loop 🧠")
        current_path = plan_to_goal(goal_id)
        replan_counter = 0

    if current_path and len(current_path) > 1:
        reached = move_along(current_path)
        if reached:
            current_path = None
    else:
        dx, dy = gx - est[0], gy - est[1]
        th_des = np.arctan2(dy, dx)
        ang_err = np.arctan2(np.sin(th_des - est[2]), np.cos(th_des - est[2]))
        dist_to_goal = np.hypot(dx, dy)
        print(f"[DIRECT] Distance: {dist_to_goal:.2f}m, Angle: {np.degrees(ang_err):.1f}°")
        for direct_steps in range(100):
            if robot.step(TIME_STEP) == -1: break
            odom_step()
            imu_update()
            maybe_ds_update()
            qr_update()
            if abs(ang_err) > 0.15:
                w = 4.0 * ang_err
                v = 0.6
            else:
                v = 1.5 * np.clip(dist_to_goal / 2.0, 0.6, 1.2)
                w = 2.0 * ang_err
            vl = (v - w * WHEEL_BASE / 2) / WHEEL_RADIUS
            vr = (v + w * WHEEL_BASE / 2) / WHEEL_RADIUS
            max_vel = 6.67
            left_motor.setVelocity(np.clip(vl, -max_vel, max_vel))
            right_motor.setVelocity(np.clip(vr, -max_vel, max_vel))
            est = pf.get_est()
            dx, dy = gx - est[0], gy - est[1]
            th_des = np.arctan2(dy, dx)
            ang_err = np.arctan2(np.sin(th_des - est[2]), np.cos(th_des - est[2]))
            dist_to_goal = np.hypot(dx, dy)
            if dist_to_goal < 0.4: break
        left_motor.setVelocity(0)
        right_motor.setVelocity(0)

# In case the loop exits without hitting the "Goal Reached" branch:
_traj_save_once()
