"""
bug1_epuck_path_plot.py
Bug-1 controller for Webots e-puck

•  Right-hand wall-follow after first entering the goal disc (y → 4 m)
•  CSV & PNG logging of the path (first 12 samples skipped in PNG)
•  **RUN-METRICS gathered only until the FIRST time the robot’s centre
   enters the goal disc**:

      – total distance travelled
      – extra distance over the ideal straight line (start → goal)
      – mean perpendicular error to that straight line
"""

from controller import Robot, DistanceSensor, Motor, GPS, Compass
import math, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Simulation & robot parameters ─────────────────────────────────
TIME_STEP  = 64
WHEEL_R    = 0.0205
BASE       = 0.053
MAX_SPEED  = 6.28

GOAL_X, GOAL_Y = 0.0, 5.0
GOAL_RADIUS    = 0.30

STOP_Y   = 4.0          # stop line
STOP_TOL = 0.05

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

# ─── FSM states for Bug-1 ───────────────────────────────────────────
GO_TO_GOAL, BOUNDARY_FOLLOW, RETURN_TO_BEST, FOLLOW_TO_STOP = range(4)
STATE_NAMES = {
    GO_TO_GOAL       : "GO_TO_GOAL",
    BOUNDARY_FOLLOW  : "BOUNDARY_FOLLOW",
    RETURN_TO_BEST   : "RETURN_TO_BEST",
    FOLLOW_TO_STOP   : "FOLLOW_TO_STOP",
}

def norm(a): return (a+math.pi)%(2*math.pi) - math.pi


# ────────────────────────────────────────────────────────────────────
class Bug1:
    def __init__(self):
        r = self.robot = Robot()
        self.lm = r.getDevice("left wheel motor")
        self.rm = r.getDevice("right wheel motor")
        for m in (self.lm, self.rm):
            m.setPosition(float("inf")); m.setVelocity(0)

        self.ir  = [r.getDevice(n) for n in IR_NAMES]
        for s in self.ir: s.enable(TIME_STEP)
        self.gps = r.getDevice("gps");     self.gps.enable(TIME_STEP)
        self.cmp = r.getDevice("compass"); self.cmp.enable(TIME_STEP)

        # pose & m-line
        self.x = self.y = self.th = 0.0
        self.alpha = 0.3
        self.init_pose = False
        self.sx = self.sy = None
        self.line_dx = self.line_dy = self.line_len = None

        # FSM vars
        self.state   = GO_TO_GOAL
        self.hit_pt  = self.best_pt = None
        self.best_d  = float("inf")
        self.arc_len = 0.0
        self.prev_x  = self.prev_y = None

        # ── METRICS (until first goal hit) ─────────────────────────
        self.total_dist     = 0.0
        self.prev_p         = None
        self.perp_err_sum   = 0.0
        self.perp_err_count = 0
        self.metrics_frozen = False
        self.frozen_total_dist = 0.0
        self.frozen_mean_perp  = 0.0

        # path logging
        self.path  = []
        self.csv   = open("path_log.csv", "w", newline="")
        self.writer= csv.writer(self.csv); self.writer.writerow(["x","y"])

    # ─── pose estimation ───────────────────────────────────────────
    def update_pose(self):
        g = self.gps.getValues(); c = self.cmp.getValues()
        if any(math.isnan(v) for v in g+c): return False
        x_raw, y_raw = g[0], g[1]
        th_raw       = norm(math.atan2(c[0], c[1]))

        if not self.init_pose:
            self.x, self.y, self.th = x_raw, y_raw, th_raw
            # m-line defined once at start
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

    # ─── wheel helper ──────────────────────────────────────────────
    def wheels(self, wl, wr):
        wl = max(min(wl, MAX_SPEED), -MAX_SPEED)
        wr = max(min(wr, MAX_SPEED), -MAX_SPEED)
        self.lm.setVelocity(wl); self.rm.setVelocity(wr)

    # ─── behaviours ────────────────────────────────────────────────
    def go_to_goal_ctrl(self, ang_err):
        if abs(ang_err) > ANGLE_TOL:
            w  = Kp_ang*ang_err
            wl = -w*(BASE/2)/WHEEL_R
            wr =  w*(BASE/2)/WHEEL_R
        else:
            v  = FWD_SPD*WHEEL_R
            w  = Kp_ang*ang_err
            wl = (v - w*BASE/2)/WHEEL_R
            wr = (v + w*BASE/2)/WHEEL_R
        self.wheels(wl, wr)

    def wall_follow_ctrl(self, ir, follow_left=True):
        if follow_left:
            side_ids, sign = LEFT_IR_IDS, +1
            turn = (+0.5*MAX_SPEED, -0.5*MAX_SPEED)
        else:
            side_ids, sign = RIGHT_IR_IDS, -1
            turn = (-0.5*MAX_SPEED, +0.5*MAX_SPEED)

        side = sum(ir[i] for i in side_ids)/len(side_ids)
        if any(ir[i] > IR_THRESHOLD for i in FRONT_IR):
            self.wheels(*turn); return

        e  = DESIRED_IR - side
        v  = WALL_SPD * WHEEL_R
        w  = sign * Kp_side * e
        wl = (v - w*BASE/2)/WHEEL_R
        wr = (v + w*BASE/2)/WHEEL_R
        self.wheels(wl, wr)

    # ─── helpers ───────────────────────────────────────────────────
    def dist_to(self, pt): return math.hypot(self.x - pt[0], self.y - pt[1])

    def line_distance(self):
        """Perpendicular distance from current pose to start→goal line."""
        num = abs(self.line_dy*(self.x-self.sx) - self.line_dx*(self.y-self.sy))
        return num / self.line_len

    # ─── main loop ─────────────────────────────────────────────────
    def run(self):
        try:
            while self.robot.step(TIME_STEP) != -1:
                if not self.update_pose(): continue

                # ── METRICS (only until first goal hit) ──────────
                if not self.metrics_frozen:
                    p = (self.x, self.y)
                    if self.prev_p is not None:
                        self.total_dist += math.hypot(p[0]-self.prev_p[0],
                                                       p[1]-self.prev_p[1])
                    self.prev_p = p
                    self.perp_err_sum   += self.line_distance()
                    self.perp_err_count += 1

                # log path
                self.path.append((self.x, self.y))
                self.writer.writerow([round(self.x,4), round(self.y,4)])

                ir = [s.getValue() for s in self.ir]
                obstacle = any(ir[i] > IR_THRESHOLD for i in FRONT_IR)

                # goal geometry
                dx, dy = GOAL_X - self.x, GOAL_Y - self.y
                dist   = math.hypot(dx, dy)
                ang_err= norm(math.atan2(dy, dx) - self.th)

                # freeze metrics when first entering goal disc
                if (not self.metrics_frozen) and (dist < GOAL_RADIUS):
                    self.metrics_frozen    = True
                    self.frozen_total_dist = self.total_dist
                    self.frozen_mean_perp  = (self.perp_err_sum /
                                              max(1, self.perp_err_count))
                    print("📊 Metrics frozen (goal reached)")

                print(f"👉 STATE={STATE_NAMES[self.state]} | dist={dist:.2f}")

                # === FSM =================================================
                if self.state == GO_TO_GOAL:
                    if dist < GOAL_RADIUS:
                        self.state = FOLLOW_TO_STOP
                        print("   inside goal disc → FOLLOW_TO_STOP")
                    elif obstacle:
                        self.state  = BOUNDARY_FOLLOW
                        self.hit_pt = self.best_pt = (self.x, self.y)
                        self.best_d = dist
                        self.arc_len= 0.0
                        self.prev_x,self.prev_y = self.x, self.y
                        print("   hit obstacle → BOUNDARY_FOLLOW")
                    else:
                        self.go_to_goal_ctrl(ang_err)

                elif self.state == BOUNDARY_FOLLOW:
                    if dist < self.best_d:
                        self.best_d, self.best_pt = dist, (self.x, self.y)

                    step = math.hypot(self.x - self.prev_x,
                                      self.y - self.prev_y)
                    self.arc_len += step
                    self.prev_x, self.prev_y = self.x, self.y

                    if self.arc_len > 0.20 and self.dist_to(self.hit_pt) < 0.05:
                        self.state = RETURN_TO_BEST
                        print("   full loop → RETURN_TO_BEST")
                    self.wall_follow_ctrl(ir, follow_left=True)

                elif self.state == RETURN_TO_BEST:
                    if self.dist_to(self.best_pt) < 0.05:
                        self.state = GO_TO_GOAL
                        print("   at best point → GO_TO_GOAL")
                    self.wall_follow_ctrl(ir, follow_left=True)

                elif self.state == FOLLOW_TO_STOP:
                    if abs(self.y - STOP_Y) < STOP_TOL:
                        print(f"   y≈{STOP_Y} m → STOP & save")
                        break
                    self.wall_follow_ctrl(ir, follow_left=False)

        finally:
            self.wheels(0,0)
            self.csv.close()
            self.save_plot(skip=12)
            self.report_metrics()

    # ─── plotting helper ──────────────────────────────────────────
    def save_plot(self, skip=12):
        if len(self.path) <= skip: return
        xs, ys = zip(*self.path[skip:])
        fig, ax = plt.subplots()
        ax.plot(xs, ys, '-', label="Path")
        ax.scatter([xs[0]],[ys[0]], c='g', label="Start")
        goal = plt.Circle((GOAL_X, GOAL_Y), GOAL_RADIUS,
                          edgecolor='r', facecolor='none',
                          linestyle='--', label='Goal')
        ax.add_artist(goal)
        ax.axhline(STOP_Y, color='k', linestyle=':', label=f"Stop y={STOP_Y}")
        ax.set_aspect('equal'); ax.grid(True)
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
        ax.legend(); plt.tight_layout()
        plt.savefig("bug1_path.png", dpi=150)
        print("Path plot saved as bug1_path.png")

    # ─── metrics helper ───────────────────────────────────────────
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

# ─── run ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    Bug1().run()
