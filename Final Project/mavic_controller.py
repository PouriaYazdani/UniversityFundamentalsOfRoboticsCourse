# Mavic 2 Pro — Webots Python controller
# Integrated LEFT/RIGHT wall following + QR detection/approach/freeze/return
# + Real-time Occupancy Grid mapping from 8 range sensors (log-odds)

from controller import Robot
import math, sys, time
try:
    import numpy as np
except ImportError:
    sys.exit("Warning: 'numpy' module not found.")
try:
    import cv2
except Exception:
    cv2 = None

# ---------- utils ----------
def clamp(x, lo, hi): return lo if x < lo else hi if x > hi else x
def wrap_pi(a): return (a + math.pi) % (2.0 * math.pi) - math.pi
def _fmt(v, n=3):
    try: return f"{float(v):.{n}f}"
    except Exception: return str(v)
import os, json, sys

def _controller_dir():
    # Robust path resolution: prefer file dir, else CWD
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()

DOORS_JSON = os.path.join(_controller_dir(), "doors.json")

def _load_doors_db(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict) or "doors" not in data or not isinstance(data["doors"], list):
                raise ValueError("Malformed JSON, resetting.")
            return data
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return {"doors": []}

def _atomic_write_json(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def save_qr_door(door_id, x, y, path: str = DOORS_JSON, ndigits: int = 2):
    """Save/Update a door record: {'id': str, 'x': float, 'y': float}."""
    data = _load_doors_db(path)
    door_id = str(door_id)
    x = round(float(x), ndigits)
    y = round(float(y), ndigits)

    for d in data["doors"]:
        if str(d.get("id")) == door_id:
            d["x"], d["y"] = x, y
            break
    else:
        data["doors"].append({"id": door_id, "x": x, "y": y})

    # Keep sorted (numeric IDs first)
    def _key(d):
        s = str(d.get("id", ""))
        return (0, int(s)) if s.isdigit() else (1, s)
    data["doors"].sort(key=_key)

    _atomic_write_json(path, data)
# ============= ### OCCUPANCY GRID ### =============
class OccupancyGrid:
    """
    Simple log-odds occupancy grid centered at takeoff point.
    Coordinates: ground plane (x, y) from GPS; altitude (z) is ignored.
    """
    def __init__(self, size_m=20.0, res=0.05):
        self.res = float(res)
        self.size_m = float(size_m)
        self.W = int(math.ceil(self.size_m / self.res))
        self.H = self.W  # square
        self.cx = self.W // 2
        self.cy = self.H // 2
        self.logodds = np.zeros((self.H, self.W), dtype=np.float32)

        # log-odds increments (tunable)
        self.L_FREE = -0.40
        self.L_OCC  = +0.85
        self.L_CLIP_MIN = -4.0
        self.L_CLIP_MAX = +4.0

        # world origin (set on first update)
        self.x0 = None
        self.y0 = None

    def set_origin_if_needed(self, x, y):
        if self.x0 is None or self.y0 is None:
            self.x0, self.y0 = float(x), float(y)

    def world_to_grid(self, x, y):
        """Map world (x,y) to grid indices (ix, iy). iy increases downward."""
        if self.x0 is None:
            return None
        dx = (float(x) - self.x0) / self.res
        dy = (float(y) - self.y0) / self.res
        ix = int(round(self.cx + dx))
        iy = int(round(self.cy - dy))  # minus so +y is "up" in the saved image
        if 0 <= ix < self.W and 0 <= iy < self.H:
            return ix, iy
        return None

    def _inc_cell(self, ix, iy, delta):
        self.logodds[iy, ix] = np.clip(self.logodds[iy, ix] + delta, self.L_CLIP_MIN, self.L_CLIP_MAX)

    def ray_update(self, x, y, theta, dist, max_range):
        """
        March a ray in small steps; mark free along the way; mark end as occupied
        iff we actually hit something (dist < max_range).
        """
        self.set_origin_if_needed(x, y)
        if self.x0 is None:
            return

        # bound distance
        dist = float(max(0.0, min(dist, max_range)))

        # sampling step ~ half a cell for clean lines
        step = max(0.01, 0.5 * self.res)
        n = max(1, int(dist / step))
        last_ixiy = None

        cos_t, sin_t = math.cos(theta), math.sin(theta)

        # free cells along the beam (up to just before the hit)
        max_free_range = dist - 0.5 * step
        free_steps = max(0, int(max_free_range / step))
        for k in range(free_steps):
            px = x + k * step * cos_t
            py = y + k * step * sin_t
            g = self.world_to_grid(px, py)
            if g is None:
                break
            ix, iy = g
            self._inc_cell(ix, iy, self.L_FREE)
            last_ixiy = (ix, iy)

        # end cell (occupied) if obstacle was detected within range
        if dist < (max_range - 1e-3):
            ex = x + dist * cos_t
            ey = y + dist * sin_t
            ge = self.world_to_grid(ex, ey)
            if ge is not None:
                ixe, iye = ge
                self._inc_cell(ixe, iye, self.L_OCC)

    def to_pgm_bytes(self):
        """
        Convert to 8-bit grayscale:
          occupied -> 0 (black)
          free     -> 255 (white)
          unknown  -> 127 (mid-gray)
        """
        l = self.logodds
        img = np.full((self.H, self.W), 127, dtype=np.uint8)
        free_mask = l <= -0.75  # slightly stricter than L_FREE
        occ_mask  = l >= +0.75  # slightly stricter than L_OCC
        img[free_mask] = 255
        img[occ_mask]  = 0
        # no flip needed; world_to_grid already makes +y go up
        header = f"P5\n{self.W} {self.H}\n255\n".encode("ascii")
        return header + img.tobytes(order="C")

    def save_pgm(self, path="occ_latest.pgm"):
        try:
            with open(path, "wb") as f:
                f.write(self.to_pgm_bytes())
            return True
        except Exception as e:
            print(f"[MAP] Save failed: {e}")
            return False
# ============= ### END OCCUPANCY GRID ### =============


class Mavic(Robot):
    # ===== Logging =====
    LOG_PERIOD = 2.0     # quieter = faster

    # ===== Core stabilization (Cyberbotics example) =====
    K_VERTICAL_THRUST = 68.5
    K_VERTICAL_OFFSET = 0.6
    K_VERTICAL_P = 3.0
    K_ROLL_P = 50.0
    K_PITCH_P = 30.0
    YAW_SIGN = -1.0

    # ===== Wall-follow setup (unchanged) =====
    SIDE = "left"       # "left" or "right"
    D0 = 0.40
    BODY_LEFT_MARGIN  = 0.45
    BODY_RIGHT_MARGIN = 0.45
    D_TURN = 0.65
    D_STOP = 0.42
    D_GAP  = 1.20
    D_LOST = 1.60
    V_FWD  = 0.22
    V_SLOW = 0.10
    MAX_YAW_DISTURBANCE = 0.22
    MAX_PITCH_DISTURBANCE = -1.0
    TURN_RATE     = 0.15
    GAP_HOOK_RATE = 0.12
    K_ALIGN = 0.50
    K_LAT   = 0.55
    FILTER_ALPHA = 0.15
    MAX_RANGE = 2.00
    RIGHT_BASELINE = 0.30
    LEFT_BASELINE  = 0.30
    SLOPE_MAX = 0.50
    E_LAT_MAX = 0.50
    FRONT_CLEAR_FOR_SLOPE = 1.00
    SIDE_SAFE = None
    HARD_WALL_BRAKE = None
    BACK_AWAY = 0.15
    target_altitude = 1.62

    # ===== QR behavior thresholds (looser) =====
    QR_DETECT_FRAMES = 2
    QR_KEEP_ALIVE    = 0.6
    QR_FREEZE_TIME   = 0.25
    QR_DUP_COOLDOWN  = 2.0

    # Framing tolerances
    AREA_MIN   = 0.002
    AREA_TGT   = 0.06
    AREA_BAND  = 0.08
    CENTER_THR = 0.09
    YAW_IMG_THR = math.radians(14)

    # IBVS mapping & limits
    K_U2RIGHT = 1.4
    K_A2FWD   = 4.0
    K_V2ALT   = 0.30
    K_YAW_IMG = 2.0
    K_XY_P = 0.30
    K_XY_D = 0.20
    ROLL_DISTURBANCE_LIM  = 1.0
    PITCH_DISTURBANCE_LIM = 1.2
    YAW_DISTURBANCE_LIM   = 0.50
    D_APPROACH_MIN = 0.26

    # ===== Performance knobs (aggressive) =====
    QR_FRAME_STRIDE_WALL     = 8   # detect every 8th step while wall-following
    QR_FRAME_STRIDE_APPROACH = 3
    QR_FRAME_STRIDE_FREEZE   = 1
    QR_DOWNSCALE_MAX = 240         # detect on <=240px wide ROI
    QR_DECODE_EVERY = 24           # decode rarely during approach
    USE_BLUR_SCORE  = False
    ROI_GROW = 1.6
    ROI_PAD  = 8

    # ===== ### OCCUPANCY GRID PARAMS ### =====
    MAP_ENABLE = True
    MAP_SIZE_M = 30.0     # covers ~20m x 20m area around takeoff
    MAP_RES    = 0.05     # 5 cm cells
    MAP_UPDATE_STRIDE = 2 # update every N control steps to keep it light
    MAP_SAVE_PERIOD = 5.0 # write occ_latest.pgm every 5 sec

    # Sensor angles in body frame (radians); adjust if your Webots model differs.
    SENSOR_ANGLES = {
        "ds_front":   0.0,
        "ds_fright": -math.pi/4,    # -45°
        "ds_fleft":  +math.pi/4,    # +45°
        "ds_right":  -math.pi/2,    # -90°
        "ds_left":   +math.pi/2,    # +90°
        "ds_bright": -3*math.pi/4,  # -135°
        "ds_bleft":  +3*math.pi/4,  # +135°
        "ds_back":    math.pi,      # 180°
    }

    def __init__(self):
        super().__init__()
        self.time_step = int(self.getBasicTimeStep())

        # Devices
        self.camera = self.getDevice("camera"); self.camera.enable(self.time_step)
        self.imu = self.getDevice("inertial unit"); self.imu.enable(self.time_step)
        self.gps = self.getDevice("gps"); self.gps.enable(self.time_step)
        self.gyro = self.getDevice("gyro"); self.gyro.enable(self.time_step)
        self.cam_roll = None; self.cam_pitch = None
        try:
            self.cam_roll = self.getDevice("camera roll")
            self.cam_pitch = self.getDevice("camera pitch")
        except Exception:
            try: self.cam_pitch = self.getDevice("camera pitch")
            except Exception: pass
        if self.cam_pitch:
            try: self.cam_pitch.setPosition(0.6)
            except Exception: pass

        # Motors
        self.front_left_motor  = self.getDevice("front left propeller")
        self.front_right_motor = self.getDevice("front right propeller")
        self.rear_left_motor   = self.getDevice("rear left propeller")
        self.rear_right_motor  = self.getDevice("rear right propeller")
        for m in (self.front_left_motor, self.front_right_motor, self.rear_left_motor, self.rear_right_motor):
            m.setPosition(float('inf')); m.setVelocity(1)

        # Distance sensors
        self.ds_names = ["ds_front","ds_fright","ds_fleft","ds_right","ds_left","ds_bright","ds_bleft","ds_back"]
        self.ds = {}
        for name in self.ds_names:
            try:
                dev = self.getDevice(name); dev.enable(self.time_step); self.ds[name] = dev
            except Exception:
                print(f("[WARN] DistanceSensor '{name}' not found."))
        # ---- ### OCC MAP init ###
        self.map = OccupancyGrid(self.MAP_SIZE_M, self.MAP_RES) if self.MAP_ENABLE else None
        self._map_tick = 0
        self._map_last_save_t = 0.0

        # Wall-follow state
        self.current_pose = [0.0]*6
        self.state = "ASCEND"
        self._guidance_started = False

        # Top-level behavior
        self.behavior_mode = "WALL"
        self.saved_pose = None
        self.pre_qr_state = None

        # QR detector
        if cv2 is not None:
            self.qr_detector = cv2.QRCodeDetector()
            print("[INFO] OpenCV QR detector active.")
        else:
            self.qr_detector = None
            print("[WARN] OpenCV not found; QR logic idle.")
        self.qr_stable = 0
        self.qr_last_seen_t = -1.0
        self.qr_freeze_until = 0.0
        self.qr_last_id_time = {}
        self.qr_frame_idx = 0
        self.qr_decode_tick = 0
        self.qr_roi = None

        # IBVS helpers
        self.prev_forward_err = 0.0
        self.prev_right_err = 0.0
        self.prev_time = 0.0
        self.alt_bias = 0.0
        self.base_altitude = self.target_altitude

        # Filters
        self._ds_filt = {n: None for n in self.ds_names}

        # Geometry for chosen side
        self.body_margin = self.BODY_RIGHT_MARGIN if self.SIDE.lower().startswith("r") else self.BODY_LEFT_MARGIN
        self.desired_measured = self.body_margin + self.D0
        self.HARD_WALL_BRAKE = self.body_margin + 0.08
        self.SIDE_SAFE = self.body_margin + self.D0 + 0.10

        self._last_log_t = 0.0
        print("[INIT] basicTimeStep:", self.time_step, "ms")

    # ---------- small helpers ----------
    def set_position(self, pos): self.current_pose = pos
    def _ema(self, name, val):
        if val is None: return None
        f = self._ds_filt.get(name); f = val if f is None else (1 - self.FILTER_ALPHA) * f + self.FILTER_ALPHA * val
        self._ds_filt[name] = f; return f
    def _read_ds_meters(self, name):
        if name not in self.ds: return None
        raw = self.ds[name].getValue()
        if raw is None: return None
        if not np.isfinite(raw): return self.MAX_RANGE
        val = float(raw)
        if val > 10 * self.MAX_RANGE:  # LUT case
            val = self.MAX_RANGE * (1.0 - np.exp(-val / 4096.0))
        return self._ema(name, clamp(val, 0.0, self.MAX_RANGE))
    @property
    def side_sign(self): return +1 if self.SIDE.lower().startswith("r") else -1
    def _side_pair(self):
        return ("ds_fright","ds_bright",self.RIGHT_BASELINE) if self.side_sign==+1 else ("ds_fleft","ds_bleft",self.LEFT_BASELINE)
    def _side_avg_slope(self):
        f_name,b_name,base = self._side_pair()
        df,db = self._read_ds_meters(f_name), self._read_ds_meters(b_name)
        if df is None or db is None: return None, None
        avg = 0.5*(df+db); slope = np.arctan2(abs(db-df), base) if base>1e-6 else 0.0; return avg, slope
    def _side_scalar(self):
        names = ["ds_right","ds_fright","ds_bright"] if self.side_sign==+1 else ["ds_left","ds_fleft","ds_bleft"]
        vals = [self._read_ds_meters(n) for n in names if n in self.ds]; vals=[v for v in vals if v is not None]
        return min(vals) if vals else None
    def _front_dist(self): return self._read_ds_meters("ds_front")
    def yaw_toward_wall(self, rate): return self.side_sign * rate
    def yaw_away_from_wall(self, rate): return -self.side_sign * rate

    # ---------- wall-follow (unchanged behavior) ----------
    def wall_follow_cmd(self):
        d_front = self._front_dist()
        d_side_avg, side_slope = self._side_avg_slope()
        d_side_min = self._side_scalar()
        yaw_cmd = 0.0; v_cmd = self.V_FWD

        if d_side_min is not None and d_side_min < self.HARD_WALL_BRAKE:
            yaw_cmd = self.yaw_away_from_wall(self.TURN_RATE)
            yaw_cmd = clamp(yaw_cmd, -self.MAX_YAW_DISTURBANCE, self.MAX_YAW_DISTURBANCE)
            pitch_cmd = +self.BACK_AWAY
            return yaw_cmd, pitch_cmd, 0.0

        if self.state == "SEEK_WALL":
            yaw_cmd = self.yaw_toward_wall(self.TURN_RATE); v_cmd = self.V_SLOW
            if d_side_min is not None and d_side_min < self.D_GAP:
                self.state = "FOLLOW"
        elif self.state == "FOLLOW":
            if d_front is not None and d_front < self.D_STOP:
                yaw_cmd = self.yaw_away_from_wall(self.TURN_RATE)
                yaw_cmd = clamp(yaw_cmd, -self.MAX_YAW_DISTURBANCE, self.MAX_YAW_DISTURBANCE)
                pitch_cmd = +self.BACK_AWAY; return yaw_cmd, pitch_cmd, 0.0
            if d_front is not None and d_front < self.D_TURN:
                v_cmd = self.V_SLOW; yaw_cmd = self.yaw_away_from_wall(self.TURN_RATE)
                yaw_cmd = clamp(yaw_cmd, -self.MAX_YAW_DISTURBANCE, self.MAX_YAW_DISTURBANCE)
                pitch_cmd = -clamp(v_cmd, 0.0, -self.MAX_PITCH_DISTURBANCE); return yaw_cmd, pitch_cmd, 0.0
            if d_side_min is not None and d_side_min > self.D_GAP:
                yaw_cmd += self.yaw_toward_wall(self.GAP_HOOK_RATE); v_cmd = max(self.V_SLOW, 0.7*self.V_FWD)
            slope_term = 0.0
            if d_side_avg is not None: e_lat = clamp(d_side_avg - self.desired_measured, -self.E_LAT_MAX, self.E_LAT_MAX)
            else: e_lat = 0.0
            use_slope = (d_front is None) or (d_front > self.FRONT_CLEAR_FOR_SLOPE)
            if d_side_avg is not None and side_slope is not None and use_slope:
                slope_term = clamp(side_slope, -self.SLOPE_MAX, self.SLOPE_MAX)
            yaw_cmd += self.yaw_toward_wall(self.K_LAT * e_lat + self.K_ALIGN * slope_term)
            if d_front is not None:
                scale_f = clamp((d_front - self.D_TURN)/max(1e-3,(self.D_GAP-self.D_TURN)), 0.2, 1.0)
            else: scale_f = 1.0
            if d_side_min is not None and d_side_min < self.SIDE_SAFE:
                rng = max(1e-3, self.SIDE_SAFE - self.HARD_WALL_BRAKE)
                scale_s = clamp((d_side_min - self.HARD_WALL_BRAKE)/rng, 0.0, 1.0)
            else: scale_s = 1.0
            v_cmd = clamp(self.V_FWD * min(scale_f, scale_s), 0.0, self.V_FWD)
            if (d_side_min is not None and d_side_min > self.D_LOST) and (d_front is not None and d_front > self.D_GAP):
                self.state = "SEEK_WALL"
        else:
            self.state = "SEEK_WALL"; yaw_cmd = self.yaw_toward_wall(self.TURN_RATE); v_cmd = self.V_SLOW

        yaw_cmd = clamp(yaw_cmd, -self.MAX_YAW_DISTURBANCE, self.MAX_YAW_DISTURBANCE)
        pitch_cmd = -clamp(v_cmd, 0.0, -self.MAX_PITCH_DISTURBANCE)
        return yaw_cmd, pitch_cmd, 0.0

    # ---------- QR fast path ----------
    def _detect_points_small(self, img_gray_small):
        pts = None
        try:
            if hasattr(self.qr_detector, "detectMulti"):
                ok, pts = self.qr_detector.detectMulti(img_gray_small);  pts = pts if ok else None
            elif hasattr(self.qr_detector, "detect"):
                ok, pts = self.qr_detector.detect(img_gray_small);       pts = pts if ok else None
        except Exception: pts = None
        if pts is None:
            try:
                out = self.qr_detector.detectAndDecodeMulti(img_gray_small)
                if isinstance(out, tuple):
                    if len(out) >= 3: pts = out[1]
            except Exception: pts = None
        return pts

    def _decode_on_roi(self, gray_full, roi_box):
        x0,y0,x1,y1 = roi_box
        H,W = gray_full.shape[:2]
        x0 = int(clamp(x0,0,W-1)); x1 = int(clamp(x1,1,W))
        y0 = int(clamp(y0,0,H-1)); y1 = int(clamp(y1,1,H))
        crop = gray_full[y0:y1, x0:x1]
        txt = ""
        try:
            t, _, _ = self.qr_detector.detectAndDecode(crop)
            if t: txt = t
            else:
                out = self.qr_detector.detectAndDecodeMulti(crop)
                if isinstance(out, tuple):
                    if len(out) >= 2 and out[0]:
                        for s in (out[0] if isinstance(out[0], (list,tuple)) else [out[0]]):
                            if s: txt = s; break
        except Exception:
            pass
        return txt

    def qr_look(self, force_decode=False):
        if self.qr_detector is None: return None
        raw = self.camera.getImage()
        if not raw: return None

        W = self.camera.getWidth(); H = self.camera.getHeight()
        buf = np.frombuffer(raw, dtype=np.uint8).reshape((H, W, 4))
        gray_full = cv2.cvtColor(buf, cv2.COLOR_BGRA2GRAY)

        # ROI crop first
        if self.qr_roi is not None:
            x0,y0,x1,y1 = self.qr_roi
            x0 = int(clamp(x0, 0, W-1)); x1 = int(clamp(x1, 1, W))
            y0 = int(clamp(y0, 0, H-1)); y1 = int(clamp(y1, 1, H))
            roi = gray_full[y0:y1, x0:x1]; base_x, base_y = x0, y0
        else:
            roi = gray_full; base_x = base_y = 0

        h,w = roi.shape[:2]
        scale = 1.0
        if w > self.QR_DOWNSCALE_MAX:
            scale = self.QR_DOWNSCALE_MAX / float(w)
            roi_small = cv2.resize(roi, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        else:
            roi_small = roi

        blur_score = 999.0
        if force_decode and self.USE_BLUR_SCORE:
            blur_score = cv2.Laplacian(roi_small, cv2.CV_64F).var()

        pts_small = self._detect_points_small(roi_small)
        if pts_small is None or len(pts_small)==0:
            self.qr_roi = None; return None

        quads=[]
        pts_small = np.array(pts_small).reshape(-1,4,2)
        for q in pts_small:
            q = (q.astype(np.float32) / scale)
            q[:,0] += base_x; q[:,1] += base_y
            A = float(cv2.contourArea(q)) / float(W*H)
            if A < self.AREA_MIN: continue
            xs,ys = q[:,0],q[:,1]
            if xs.min()<2 or ys.min()<2 or xs.max()>(W-3) or ys.max()>(H-3): continue
            rect = cv2.minAreaRect(q.astype(np.float32))
            ang = rect[2] + (90.0 if rect[1][0] < rect[1][1] else 0.0)
            while ang <= -45.0: ang += 90.0
            while ang > 45.0:   ang -= 90.0
            yaw_img = math.radians(ang)
            cx,cy = q.mean(axis=0)
            quads.append({"pts": q, "A": A, "cx": cx, "cy": cy, "yaw_img": yaw_img})

        if not quads:
            self.qr_roi = None; return None

        best = max(quads, key=lambda d: d["A"])

        # Update ROI for next tick
        xmn = int(best["pts"][:,0].min()); xmx = int(best["pts"][:,0].max())
        ymn = int(best["pts"][:,1].min()); ymx = int(best["pts"][:,1].max())
        cx_i = (xmn + xmx)//2; cy_i = (ymn + ymx)//2
        wbox = int((xmx - xmn)*self.ROI_GROW) + self.ROI_PAD
        hbox = int((ymx - ymn)*self.ROI_GROW) + self.ROI_PAD
        self.qr_roi = (cx_i - wbox, cy_i - hbox, cx_i + wbox, cy_i + hbox)

        # errors
        e_u = (best["cx"] - (W*0.5)) / float(W)
        e_v = (best["cy"] - (H*0.5)) / float(H)
        e_A = (self.AREA_TGT - best["A"])

        text = ""
        if force_decode:
            text = self._decode_on_roi(gray_full, self.qr_roi)

        return (best, e_u, e_v, e_A, best["yaw_img"], blur_score, text)


    # ---------- main ----------
    def run(self):
        print(f"[START] Target altitude: {self.target_altitude} m; Wall side: {self.SIDE}")
        while self.step(self.time_step) != -1:
            if self.getTime()>0.02: break
        _,_,yaw0 = self.imu.getRollPitchYaw()
        self.prev_time = self.getTime()
        self.base_altitude = self.target_altitude

        def log_qr(txt,x,y,z,yaw):
            print(f"[QR-READ] id='{txt}' @ ({x:+.2f},{y:+.2f},{z:+.2f}) yaw={yaw:+.2f}")
            save_qr_door(txt, x, y)

        while self.step(self.time_step) != -1:
            t = self.getTime(); dt = max(1e-3, t - self.prev_time)

            # sensors
            roll,pitch,yaw = self.imu.getRollPitchYaw()
            x,y_pos,z = self.gps.getValues()
            gx,gy,_ = self.gyro.getValues()
            self.set_position([x, y_pos, z, roll, pitch, yaw])

            # optional gimbal
            if self.cam_roll:
                try: self.cam_roll.setPosition(clamp(-0.05*gx,-0.5,0.5))
                except Exception: pass
            if self.cam_pitch:
                try: self.cam_pitch.setPosition(clamp(-0.04*gy + 0.6,-0.5,0.8))
                except Exception: pass

            yaw_disturbance=0.0; pitch_disturbance = (-0.10 if self.state=="ASCEND" else 0.0); roll_disturbance=0.0
            forward_err=0.0; right_err=0.0

            # vision cadence
            self.qr_frame_idx += 1
            stride = (self.QR_FRAME_STRIDE_WALL if self.behavior_mode=="WALL"
                      else self.QR_FRAME_STRIDE_APPROACH if self.behavior_mode=="QR_APPROACH"
                      else self.QR_FRAME_STRIDE_FREEZE)
            should_scan = (self.qr_detector is not None) and ((self.qr_frame_idx % stride)==0)

            # vision-driven transitions
            qr_best=None
            if self.behavior_mode in ("WALL","QR_APPROACH","QR_FREEZE") and should_scan:
                force_decode = (self.behavior_mode=="QR_FREEZE")
                if self.behavior_mode=="QR_APPROACH":
                    self.qr_decode_tick += 1
                    if (self.qr_decode_tick % self.QR_DECODE_EVERY)==0:
                        force_decode = True
                qi = self.qr_look(force_decode=force_decode)
                if qi is not None:
                    best,e_u,e_v,e_A,yaw_img,blur,text = qi
                    qr_best = (best,e_u,e_v,e_A,yaw_img,blur,text)
                    self.qr_last_seen_t = t
                    if text:
                        last = self.qr_last_id_time.get(text, -1e9)
                        if (time.time()-last) > self.QR_DUP_COOLDOWN:
                            self.qr_last_id_time[text] = time.time(); log_qr(text,x,y_pos,z,yaw)
                    if self.behavior_mode=="WALL":
                        self.qr_stable = min(self.QR_DETECT_FRAMES, self.qr_stable+1)
                        if self.qr_stable >= self.QR_DETECT_FRAMES:
                            self.saved_pose = (x,y_pos,z,yaw); self.pre_qr_state = self.state
                            self.behavior_mode="QR_APPROACH"; self.alt_bias=0.0
                            print("[MODE] WALL → QR_APPROACH")
                else:
                    if self.behavior_mode=="QR_APPROACH" and (t - self.qr_last_seen_t) > self.QR_KEEP_ALIVE:
                        print("[QR] Lost target → WALL"); self.behavior_mode="WALL"
                        self.state = "FOLLOW" if self._guidance_started else "SEEK_WALL"
                        self.alt_bias=0.0; self.qr_stable=0; self.qr_roi=None

            # behavior execution
            if self.state=="ASCEND":
                if z > self.target_altitude - 0.2:
                    self.state="SEEK_WALL"; self._guidance_started=True
                    print(f"[ALT] z={_fmt(z,2)} → SEEK_WALL")
                yaw_cmd_wall, pitch_cmd_wall, _ = 0.0, -0.10, 0.0
                yaw_disturbance=yaw_cmd_wall; pitch_disturbance=pitch_cmd_wall

            elif self.behavior_mode=="WALL":
                yaw_cmd_wall, pitch_cmd_wall, _ = self.wall_follow_cmd()
                yaw_disturbance=yaw_cmd_wall; pitch_disturbance=pitch_cmd_wall; roll_disturbance=0.0

            elif self.behavior_mode=="QR_APPROACH":
                if qr_best is not None:
                    best,e_u,e_v,e_A,yaw_img,blur,text = qr_best
                    right_err = self.K_U2RIGHT*e_u; forward_err = self.K_A2FWD*e_A
                    d_forward = (forward_err - self.prev_forward_err)/dt
                    d_right   = (right_err   - self.prev_right_err)/dt
                    roll_disturbance  = clamp(self.K_XY_P*right_err + self.K_XY_D*d_right, -self.ROLL_DISTURBANCE_LIM, self.ROLL_DISTURBANCE_LIM)
                    pitch_disturbance = clamp(-(self.K_XY_P*forward_err + self.K_XY_D*d_forward), -self.PITCH_DISTURBANCE_LIM, self.PITCH_DISTURBANCE_LIM)
                    d_front = self._front_dist()
                    if d_front is not None and d_front < self.D_APPROACH_MIN:
                        pitch_disturbance = max(0.0, pitch_disturbance)
                    self.alt_bias = clamp(self.alt_bias + (-self.K_V2ALT*e_v)*0.5*dt, -0.35, +0.35)
                    self.target_altitude = clamp(self.base_altitude + self.alt_bias, 0.30, 2.20)
                    yaw_disturbance = clamp(0.25*(2.0*wrap_pi(yaw0-yaw)) + self.K_YAW_IMG*(-yaw_img),
                                            -self.YAW_DISTURBANCE_LIM, self.YAW_DISTURBANCE_LIM)
                    centered = (abs(e_u)<=self.CENTER_THR); sized=(abs(e_A)<=self.AREA_BAND)
                    squared  = (abs(yaw_img)<=self.YAW_IMG_THR)
                    if centered and sized and squared:
                        self.behavior_mode="QR_FREEZE"; self.qr_freeze_until=t+self.QR_FREEZE_TIME
                        print("[QR] Freeze & capture…")
                else:
                    roll_disturbance=pitch_disturbance=yaw_disturbance=0.0

            elif self.behavior_mode=="QR_FREEZE":
                roll_disturbance=pitch_disturbance=0.0
                if t >= self.qr_freeze_until:
                    self.behavior_mode="QR_RETURN"; self.target_altitude=self.base_altitude
                    self.alt_bias=0.0; print("[MODE] QR_FREEZE → QR_RETURN")

            elif self.behavior_mode=="QR_RETURN":
                if self.saved_pose is None:
                    self.behavior_mode="WALL"; self.state="FOLLOW"
                else:
                    x0,y0,z0,yaw_saved = self.saved_pose
                    ex,ey = (x0-x),(y0-y_pos)
                    cy,sy = math.cos(-yaw), math.sin(-yaw)
                    forward_err =  cy*ex + sy*ey; right_err = -sy*ex + cy*ey
                    d_forward = (forward_err - self.prev_forward_err)/dt
                    d_right   = (right_err   - self.prev_right_err)/dt
                    roll_disturbance  = clamp(self.K_XY_P*right_err + self.K_XY_D*d_right, -self.ROLL_DISTURBANCE_LIM, self.ROLL_DISTURBANCE_LIM)
                    pitch_disturbance = clamp(-(self.K_XY_P*forward_err + self.K_XY_D*d_forward), -self.PITCH_DISTURBANCE_LIM, self.PITCH_DISTURBANCE_LIM)
                    yaw_err_lock = wrap_pi(yaw_saved - yaw)
                    yaw_disturbance = clamp(2.0*yaw_err_lock, -self.YAW_DISTURBANCE_LIM, self.YAW_DISTURBANCE_LIM)
                    if (abs(z0 - z) < 0.15) and (abs(ex) < 0.40) and (abs(ey) < 0.40):
                        self.behavior_mode="WALL"
                        self.state = self.pre_qr_state if self.pre_qr_state in ("FOLLOW","SEEK_WALL") else "FOLLOW"
                        self.qr_roi=None
                        print("[MODE] QR_RETURN → WALL")

            # ----- mixer (unchanged) -----
            clamped_dz  = clamp(self.target_altitude - z + self.K_VERTICAL_OFFSET, -1, 1)
            vertical_input = self.K_VERTICAL_P * (clamped_dz ** 3)
            gx_now,gy_now,_ = self.gyro.getValues()
            roll_input  = self.K_ROLL_P  * clamp(roll,  -1, 1) + gx_now + roll_disturbance
            pitch_input = self.K_PITCH_P * clamp(pitch, -1, 1) + gy_now + pitch_disturbance
            yaw_input   = self.YAW_SIGN * yaw_disturbance

            fl = self.K_VERTICAL_THRUST + vertical_input - yaw_input + pitch_input - roll_input
            fr = self.K_VERTICAL_THRUST + vertical_input + yaw_input + pitch_input + roll_input
            rl = self.K_VERTICAL_THRUST + vertical_input + yaw_input - pitch_input - roll_input
            rr = self.K_VERTICAL_THRUST + vertical_input - yaw_input - pitch_input + roll_input

            self.front_left_motor.setVelocity(fl);  self.front_right_motor.setVelocity(-fr)
            self.rear_left_motor.setVelocity(-rl);  self.rear_right_motor.setVelocity(rr)

            # ----- ### OCCUPANCY GRID UPDATE ### -----
            if self.map is not None:
                self._map_tick += 1
                if (self._map_tick % self.MAP_UPDATE_STRIDE) == 0:
                    # cast a ray for each sensor that exists
                    for name, angle in self.SENSOR_ANGLES.items():
                        if name not in self.ds:
                            continue
                        d = self._read_ds_meters(name)
                        if d is None:
                            continue
                        theta_world = yaw + angle
                        self.map.ray_update(x, y_pos, theta_world, d, self.MAX_RANGE)

                # periodic save (PGM, safe header)
                if (t - self._map_last_save_t) >= self.MAP_SAVE_PERIOD:
                    ok = self.map.save_pgm("occ_latest.pgm")
                    if ok:
                        print("[MAP] Saved occ_latest.pgm")
                    self._map_last_save_t = t

            # light logging
            if t - self._last_log_t >= self.LOG_PERIOD:
                d_front = self._front_dist(); d_side_avg, side_slope = self._side_avg_slope(); d_side_min = self._side_scalar()
                mot_min = min(fl, fr, rl, rr); mot_max = max(fl, fr, rl, rr)
                print(f"[STAT t={_fmt(t,2)}] qr_mode={self.behavior_mode} state={self.state} "
                      f"pos=({_fmt(x,2)},{_fmt(y_pos,2)},{_fmt(z,2)}) d_front={_fmt(d_front,2)} "
                      f"d_side_avg={_fmt(d_side_avg,2)} d_side_min={_fmt(d_side_min,2)} slope={_fmt(side_slope,3)} "
                      f"yd={_fmt(yaw_disturbance,3)} pd={_fmt(pitch_disturbance,3)} rd={_fmt(roll_disturbance,3)} "
                      f"mot[min,max]=({_fmt(mot_min,1)},{_fmt(mot_max,1)})")
                self._last_log_t = t

            # PD memory
            self.prev_time = t; self.prev_forward_err = forward_err; self.prev_right_err = right_err

# Run
if __name__ == "__main__":
    robot = Mavic()
    robot.run()
