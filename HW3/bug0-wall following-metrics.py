"""
bug0_epuck_path_plot.py
Bug-0 controller for Webots e-puck

•  Logs CSV + PNG of the trajectory
•  Right-hand wall-follow after the first time the goal disc is entered
•  **Performance metrics collected ONLY until that first goal hit**
     – total path length
     – extra distance over the ideal straight line (start → goal)
     – mean perpendicular deviation from that straight line
"""
from controller import Robot, DistanceSensor, Motor, GPS, Compass
import math, csv, os, sys

# ─── (optional) plotting back-end – comment out if matplotlib missing ──
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
# ───────────────────────────────────────────────────────────────────────

# --- Parameters -------------------------------------------------------
TIME_STEP  = 64
WHEEL_R    = 0.0205
BASE       = 0.053
MAX_SPEED  = 6.28

GOAL_X, GOAL_Y = 0.0, 5.0
GOAL_RADIUS    = 0.30
STOP_Y         = 4.0
STOP_TOL       = 0.05

IR_NAMES       = ["ps0","ps1","ps2","ps3","ps4","ps5","ps6","ps7"]
FRONT_IR       = (0,1,7)
LEFT_IR_IDS    = (5,6)
RIGHT_IR_IDS   = (2,3)
IR_THRESHOLD   = 80.0
DESIRED_IR     = 300.0

Kp_ang  = 4.0
Kp_side = 0.02
FWD_SPD = 0.5*MAX_SPEED
WALL_SPD= 0.3*MAX_SPEED
ANGLE_TOL = 0.20

# FSM states
GO_TO_GOAL, FOLLOW_LEFT, FOLLOW_RIGHT = range(3)
STATE_NAMES = {
    GO_TO_GOAL   : "GO_TO_GOAL",
    FOLLOW_LEFT  : "FOLLOW_LEFT",
    FOLLOW_RIGHT : "FOLLOW_RIGHT"
}

def norm(a): return (a+math.pi)%(2*math.pi)-math.pi


class Bug0:
    def __init__(self):
        r = self.robot = Robot()
        self.lm = r.getDevice("left wheel motor")
        self.rm = r.getDevice("right wheel motor")
        for m in (self.lm,self.rm):
            m.setPosition(float('inf')); m.setVelocity(0)

        # sensors
        self.ir = [r.getDevice(n) for n in IR_NAMES]
        for s in self.ir: s.enable(TIME_STEP)
        self.gps = r.getDevice("gps");     self.gps.enable(TIME_STEP)
        self.cmp = r.getDevice("compass"); self.cmp.enable(TIME_STEP)

        # pose & m-line
        self.x = self.y = self.th = 0.0
        self.alpha     = 0.3
        self.init_pose = False
        self.sx = self.sy = None
        self.line_dx = self.line_dy = self.line_len = None

        # fsm
        self.state = GO_TO_GOAL
        self.hit   = float('inf')

        # ── METRICS (until first goal hit) ─────────────────────────
        self.total_dist     = 0.0
        self.prev_p         = None
        self.perp_err_sum   = 0.0
        self.perp_err_count = 0
        self.metrics_frozen = False
        self.frozen_total_dist = 0.0
        self.frozen_mean_perp  = 0.0

        # ── path logging ───────────────────────────────────────────
        self.path = []
        self.csv_file   = open("path_log.csv", "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["x", "y"])

    # ─── pose from gps + compass ───────────────────────────────────
    def pose(self):
        g = self.gps.getValues(); c = self.cmp.getValues()
        if any(math.isnan(v) for v in g+c): return False
        x_raw, y_raw = g[0], g[1]
        th_raw       = norm(math.atan2(c[0], c[1]))

        if not self.init_pose:
            self.x, self.y, self.th = x_raw, y_raw, th_raw
            # start-to-goal line
            self.sx, self.sy = x_raw, y_raw
            self.line_dx     = GOAL_X - self.sx
            self.line_dy     = GOAL_Y - self.sy
            self.line_len    = math.hypot(self.line_dx, self.line_dy)
            self.init_pose   = True
        else:
            α = self.alpha
            self.x  += α*(x_raw - self.x)
            self.y  += α*(y_raw - self.y)
            self.th = norm(self.th + α*norm(th_raw - self.th))
        return True

    # ─── wheels helper ─────────────────────────────────────────────
    def wheels(self, wl, wr):
        wl = max(min(wl,MAX_SPEED),-MAX_SPEED)
        wr = max(min(wr,MAX_SPEED),-MAX_SPEED)
        self.lm.setVelocity(wl); self.rm.setVelocity(wr)

    # ─── behaviours ────────────────────────────────────────────────
    def go_to_goal(self, err):
        if abs(err)>ANGLE_TOL:
            w  = Kp_ang*err
            wl = -w*(BASE/2)/WHEEL_R
            wr =  w*(BASE/2)/WHEEL_R
        else:
            v  = FWD_SPD*WHEEL_R
            w  = Kp_ang*err
            wl = (v - w*BASE/2)/WHEEL_R
            wr = (v + w*BASE/2)/WHEEL_R
        self.wheels(wl,wr)

    def wall_follow(self, ir, follow_left=True):
        if follow_left:
            side_ids = LEFT_IR_IDS; turn=(+0.5*MAX_SPEED,-0.5*MAX_SPEED); sign=+1
        else:
            side_ids = RIGHT_IR_IDS;turn=(-0.5*MAX_SPEED,+0.5*MAX_SPEED); sign=-1
        side = sum(ir[i] for i in side_ids)/len(side_ids)
        if any(ir[i]>IR_THRESHOLD for i in FRONT_IR):
            self.wheels(*turn); return
        e = DESIRED_IR - side
        v = WALL_SPD*WHEEL_R
        w = sign*Kp_side*e
        wl = (v - w*BASE/2)/WHEEL_R
        wr = (v + w*BASE/2)/WHEEL_R
        self.wheels(wl,wr)

    # ─── metrics helper ────────────────────────────────────────────
    def line_distance(self):
        num = abs(self.line_dy*(self.x-self.sx) - self.line_dx*(self.y-self.sy))
        return num / self.line_len

    # ─── plotting helper ───────────────────────────────────────────
    def save_plot(self, skip=12):
        if len(self.path) <= skip:
            print(f"Not enough data to plot (have {len(self.path)} points).")
            return
        xs, ys = zip(*self.path[skip:])
        fig, ax = plt.subplots()
        ax.plot(xs, ys, '-', label="Path")
        ax.scatter([xs[0]], [ys[0]], c='g', marker='o', label="Start")
        goal = plt.Circle((GOAL_X, GOAL_Y), GOAL_RADIUS,
                          edgecolor='r', facecolor='none', linestyle='--',
                          label='Goal')
        ax.add_artist(goal)
        ax.axhline(STOP_Y, color='k', linestyle=':', label=f"Stop y={STOP_Y} m")
        ax.set_aspect('equal'); ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
        ax.legend(); ax.grid(True)
        plt.tight_layout()
        plt.savefig("bug0_path.png", dpi=150)
        print("Path plot saved as bug0_path.png (first", skip, "points omitted)")

    # ─── run loop ──────────────────────────────────────────────────
    def run(self):
        try:
            while self.robot.step(TIME_STEP)!=-1:
                if not self.pose(): continue

                # ── METRICS update (until goal) ───────────────────
                if not self.metrics_frozen:
                    p = (self.x, self.y)
                    if self.prev_p is not None:
                        self.total_dist += math.hypot(p[0]-self.prev_p[0],
                                                       p[1]-self.prev_p[1])
                    self.prev_p = p
                    self.perp_err_sum   += self.line_distance()
                    self.perp_err_count += 1

                # log pose
                self.path.append((self.x, self.y))
                self.csv_writer.writerow([self.x, self.y])

                dx, dy = GOAL_X-self.x, GOAL_Y-self.y
                dist   = math.hypot(dx,dy)
                err    = norm(math.atan2(dy,dx)-self.th)

                ir  = [s.getValue() for s in self.ir]
                obs = any(ir[i]>IR_THRESHOLD for i in FRONT_IR)

                # freeze metrics on first goal hit
                if (not self.metrics_frozen) and (dist < GOAL_RADIUS):
                    self.metrics_frozen    = True
                    self.frozen_total_dist = self.total_dist
                    self.frozen_mean_perp  = (self.perp_err_sum /
                                              max(1, self.perp_err_count))
                    print("📊 Metrics frozen (goal reached)")

                print(f"👉 STATE={STATE_NAMES[self.state]} | dist={dist:.2f}")

                # FSM ------------------------------------------------
                if self.state == GO_TO_GOAL:
                    if dist < GOAL_RADIUS:
                        self.state = FOLLOW_RIGHT
                        print("   reached goal → FOLLOW_RIGHT")
                    elif obs:
                        self.hit   = dist
                        self.state = FOLLOW_LEFT
                        print("   obstacle ahead → FOLLOW_LEFT")
                    else:
                        self.go_to_goal(err)

                elif self.state == FOLLOW_LEFT:
                    self.wall_follow(ir, follow_left=True)
                    if not obs and dist < self.hit-0.20:
                        self.state = GO_TO_GOAL
                        print("   front clear & closer → GO_TO_GOAL")

                elif self.state == FOLLOW_RIGHT:
                    if abs(self.y - STOP_Y) < STOP_TOL:
                        print(f"   y≈{STOP_Y:.2f} m → STOP")
                        break
                    self.wall_follow(ir, follow_left=False)

        finally:
            self.wheels(0,0)
            self.csv_file.close()
            try:    self.save_plot()
            except Exception as e:
                print("Could not save plot:", e)
            self.report_metrics()

    # ─── metrics report ───────────────────────────────────────────
    def report_metrics(self):
        if self.metrics_frozen:
            total_dist = self.frozen_total_dist
            mean_perp  = self.frozen_mean_perp
        else:  # goal never reached
            total_dist = self.total_dist
            mean_perp  = (self.perp_err_sum /
                          max(1, self.perp_err_count))

        straight_dist = self.line_len
        extra_dist    = total_dist - straight_dist

        msg = (
            "\n────── METRICS (until goal) ──────\n"
            f"Total distance       : {total_dist:.3f} m\n"
            f"Straight-line length : {straight_dist:.3f} m\n"
            f"   ➜ Extra distance  : {extra_dist:.3f} m\n"
            f"Mean perp. error     : {mean_perp*100:.1f} cm\n"
            "────────────────────────────────────\n"
        )
        print(msg)
        with open("run_metrics.txt", "a") as f:
            f.write(f"{total_dist:.3f},{extra_dist:.3f},{mean_perp:.4f}\n")

# ─── run ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    Bug0().run()
