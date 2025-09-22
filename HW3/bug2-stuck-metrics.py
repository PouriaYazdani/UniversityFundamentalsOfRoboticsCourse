"""
bug2_epuck_path_plot.py
Bug-2 controller for Webots e-puck

•  CSV & PNG logging of the travelled path
•  Reverse-pivot-wiggle escape routine
•  **Metrics collected only until the FIRST time the robot reaches the
   goal disc**:
      – total distance travelled
      – extra distance over the straight line
      – mean perpendicular error to the m-line
"""
from controller import Robot, DistanceSensor, Motor, GPS, Compass
import math, csv, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── basic robot & task parameters ───────────────────────────────────
TIME_STEP  = 64
WHEEL_R    = 0.0205          # wheel radius (m)
BASE       = 0.053           # axle length (m)
MAX_SPEED  = 6.28            # Webots rad s⁻¹

GOAL_X, GOAL_Y = 0.0, 5.0
GOAL_RADIUS    = 0.30

STOP_Y   = 4.0
STOP_TOL = 0.05

IR_NAMES       = ["ps0","ps1","ps2","ps3","ps4","ps5","ps6","ps7"]
FRONT_IR       = (0,1,7)
LEFT_IR_IDS    = (5,6)
RIGHT_IR_IDS   = (2,3)
IR_THRESHOLD   = 80.0
DESIRED_IR     = 300.0

Kp_ang   = 4.0
Kp_side  = 0.02
FWD_SPD  = 0.5*MAX_SPEED
WALL_SPD = 0.3*MAX_SPEED
ANGLE_TOL= 0.20

LINE_TOL  = 0.05
EXIT_BIAS = 0.05

# ─── stuck-recovery parameters ───────────────────────────────────────
STUCK_WIN   = 50
STUCK_MOV   = 0.005
BACK_TIME   = 50
TURN_TIME   = 10
WIGGLE_TIME = 20
BACK_SPEED  = 0.3*MAX_SPEED
TURN_SPEED  = 0.6*MAX_SPEED
WIGGLE_SPEED= 0.4*MAX_SPEED

# ─── finite-state machine ────────────────────────────────────────────
GO_TO_GOAL, BOUNDARY_FOLLOW, FOLLOW_TO_STOP, ESCAPE = range(4)
STATE_NAMES = {GO_TO_GOAL:"GO_TO_GOAL",
               BOUNDARY_FOLLOW:"BOUNDARY_FOLLOW",
               FOLLOW_TO_STOP:"FOLLOW_TO_STOP",
               ESCAPE:"ESCAPE"}

def norm(a):                              # wrap angle to (-π, π]
    return (a+math.pi)%(2*math.pi)-math.pi


# ─────────────────────────────────────────────────────────────────────
class Bug2:
    def __init__(self):
        r = self.robot = Robot()
        # motors
        self.lm = r.getDevice("left wheel motor")
        self.rm = r.getDevice("right wheel motor")
        for m in (self.lm, self.rm):
            m.setPosition(float('inf')); m.setVelocity(0)

        # sensors
        self.ir  = [r.getDevice(n) for n in IR_NAMES]
        for s in self.ir: s.enable(TIME_STEP)
        self.gps = r.getDevice("gps");     self.gps.enable(TIME_STEP)
        self.cmp = r.getDevice("compass"); self.cmp.enable(TIME_STEP)

        # pose
        self.x = self.y = self.th = 0.0
        self.alpha = 0.3
        self.init_pose = False

        # m-line
        self.sx = self.sy = None
        self.line_dx = self.line_dy = self.line_len = None

        # FSM bookkeeping
        self.state    = GO_TO_GOAL
        self.hit_dist = None

        # stuck bookkeeping
        self.pos_buf      = []
        self.escape_phase = 0
        self.timer        = 0
        self.turn_sign    = 1

        # path logging
        self.path = []
        self.csv = open("path_log.csv", "w", newline="")
        self.w   = csv.writer(self.csv); self.w.writerow(["x","y"])

        # ── metrics (up to goal only) ───────────────────────────────
        self.total_dist     = 0.0
        self.prev_p         = None
        self.perp_err_sum   = 0.0
        self.perp_err_count = 0
        self.metrics_frozen = False      # becomes True at first goal hit
        self.frozen_total_dist = 0.0
        self.frozen_mean_perp = 0.0

    # ─── pose update ────────────────────────────────────────────────
    def update_pose(self):
        g, c = self.gps.getValues(), self.cmp.getValues()
        if any(math.isnan(v) for v in g+c): return False
        xr, yr = g[0], g[1]
        thr    = norm(math.atan2(c[0], c[1]))
        if not self.init_pose:
            self.x, self.y, self.th = xr, yr, thr
            self.sx, self.sy       = xr, yr
            self.line_dx           = GOAL_X - self.sx
            self.line_dy           = GOAL_Y - self.sy
            self.line_len          = math.hypot(self.line_dx, self.line_dy)
            self.init_pose         = True
        else:
            a     = self.alpha
            self.x += a*(xr - self.x)
            self.y += a*(yr - self.y)
            self.th = norm(self.th + a*norm(thr - self.th))
        return True

    # ─── wheel helper ───────────────────────────────────────────────
    def wheels(self, wl, wr):
        wl = max(min(wl, MAX_SPEED), -MAX_SPEED)
        wr = max(min(wr, MAX_SPEED), -MAX_SPEED)
        self.lm.setVelocity(wl); self.rm.setVelocity(wr)

    # ─── basic behaviours ───────────────────────────────────────────
    def go_to_goal_ctrl(self, err):
        if abs(err) > ANGLE_TOL:
            w  = Kp_ang*err
            wl = -w*(BASE/2)/WHEEL_R;  wr =  w*(BASE/2)/WHEEL_R
        else:
            v  = FWD_SPD*WHEEL_R
            w  = Kp_ang*err
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

        if any(ir[i] > IR_THRESHOLD for i in FRONT_IR):
            self.wheels(*turn); return

        side = sum(ir[i] for i in side_ids)/len(side_ids)
        e    = DESIRED_IR - side
        v    = WALL_SPD*WHEEL_R
        w    = sign*Kp_side*e
        wl   = (v - w*BASE/2)/WHEEL_R
        wr   = (v + w*BASE/2)/WHEEL_R
        self.wheels(wl, wr)

    # ─── geometry helpers ───────────────────────────────────────────
    def dist_to_goal(self):      # Euclidean distance to goal
        return math.hypot(GOAL_X - self.x, GOAL_Y - self.y)

    def line_distance(self):     # ⟂ distance to m-line
        num = abs(self.line_dy*(self.x-self.sx) - self.line_dx*(self.y-self.sy))
        return num / self.line_len

    def projection_along_line(self):
        return ((self.x-self.sx)*self.line_dx + (self.y-self.sy)*self.line_dy) / self.line_len

    # ─── stuck detection & recovery ─────────────────────────────────
    def detect_stuck(self):
        self.pos_buf.append((self.x, self.y))
        if len(self.pos_buf) > STUCK_WIN:
            self.pos_buf.pop(0)
        if len(self.pos_buf) < STUCK_WIN:
            return False
        dx = self.pos_buf[-1][0] - self.pos_buf[0][0]
        dy = self.pos_buf[-1][1] - self.pos_buf[0][1]
        return math.hypot(dx, dy) < STUCK_MOV

    def choose_turn_dir(self, ir):
        left  = ir[LEFT_IR_IDS[0]]  + ir[LEFT_IR_IDS[1]]
        right = ir[RIGHT_IR_IDS[0]] + ir[RIGHT_IR_IDS[1]]
        return -1 if right < left else +1    # pivot towards freer side

    def start_escape(self, ir):
        self.state        = ESCAPE
        self.escape_phase = 0
        self.timer        = BACK_TIME
        self.turn_sign    = self.choose_turn_dir(ir)
        self.pos_buf.clear()
        print("👉 STATE=ESCAPE (detected stuck)")

    def escape_ctrl(self):
        if self.escape_phase == 0:             # reverse
            self.wheels(-BACK_SPEED, -BACK_SPEED)
        elif self.escape_phase == 1:           # pivot
            self.wheels(-self.turn_sign*TURN_SPEED,
                          self.turn_sign*TURN_SPEED)
        else:                                  # wiggle forward
            self.wheels(WIGGLE_SPEED*0.6, WIGGLE_SPEED)

        self.timer -= 1
        if self.timer == 0:
            self.escape_phase += 1
            if self.escape_phase == 1:
                self.timer = TURN_TIME
            elif self.escape_phase == 2:
                self.timer = WIGGLE_TIME
            else:
                print("   escape complete → GO_TO_GOAL")
                self.state = GO_TO_GOAL

    # ─── main loop ──────────────────────────────────────────────────
    def run(self):
        try:
            while self.robot.step(TIME_STEP) != -1:
                if not self.update_pose(): continue

                # ── METRICS (until goal) ──────────────────────────
                if not self.metrics_frozen:
                    p = (self.x, self.y)
                    if self.prev_p is not None:
                        self.total_dist += math.hypot(p[0]-self.prev_p[0],
                                                       p[1]-self.prev_p[1])
                    self.prev_p = p

                    self.perp_err_sum   += self.line_distance()
                    self.perp_err_count += 1

                # always collect path for plot
                self.path.append((self.x, self.y))
                self.w.writerow([round(self.x,4), round(self.y,4)])

                # sensor readings
                ir       = [s.getValue() for s in self.ir]
                obstacle = any(ir[i] > IR_THRESHOLD for i in FRONT_IR)
                d_goal   = self.dist_to_goal()
                ang_err  = norm(math.atan2(GOAL_Y-self.y, GOAL_X-self.x) - self.th)

                # freeze metrics the first time we reach goal disc
                if (not self.metrics_frozen) and (d_goal < GOAL_RADIUS):
                    self.metrics_frozen   = True
                    self.frozen_total_dist= self.total_dist
                    self.frozen_mean_perp = (self.perp_err_sum /
                                             max(1, self.perp_err_count))
                    print("📊 Metrics frozen (goal reached)")

                # stuck detection
                if self.state != ESCAPE and self.detect_stuck():
                    self.start_escape(ir)

                # ── FSM actions ───────────────────────────────────
                if self.state == GO_TO_GOAL:
                    if d_goal < GOAL_RADIUS:
                        self.state = FOLLOW_TO_STOP
                    elif obstacle:
                        self.state, self.hit_dist = BOUNDARY_FOLLOW, d_goal
                    else:
                        self.go_to_goal_ctrl(ang_err)

                elif self.state == BOUNDARY_FOLLOW:
                    if ( self.line_distance() < LINE_TOL and
                         d_goal            < self.hit_dist - EXIT_BIAS and
                         self.projection_along_line() > 0 ):
                        self.state = GO_TO_GOAL
                    self.wall_follow_ctrl(ir, follow_left=True)

                elif self.state == FOLLOW_TO_STOP:
                    if abs(self.y - STOP_Y) < STOP_TOL:
                        break
                    self.wall_follow_ctrl(ir, follow_left=False)

                elif self.state == ESCAPE:
                    self.escape_ctrl()
                    continue

        finally:
            self.wheels(0,0)
            self.csv.close()
            self.save_plot(skip=12)
            self.report_metrics()

    # ─── plotting helper ────────────────────────────────────────────
    def save_plot(self, skip=12):
        if len(self.path) <= skip: return
        xs, ys = zip(*self.path[skip:])
        fig, ax = plt.subplots()
        ax.plot(xs, ys, '-', label="Path")
        ax.scatter([xs[0]],[ys[0]],c='g',label="Start")
        goal = plt.Circle((GOAL_X, GOAL_Y), GOAL_RADIUS,
                          edgecolor='r', facecolor='none',
                          linestyle='--', label='Goal')
        ax.add_artist(goal)
        ax.axhline(STOP_Y, color='k', linestyle=':', label=f"Stop y={STOP_Y}")
        ax.set_aspect('equal'); ax.grid(True)
        ax.set_xlabel("X [m]");  ax.set_ylabel("Y [m]")
        ax.legend(); plt.tight_layout()
        plt.savefig("bug2_path.png", dpi=150)
        print("Path plot saved as bug2_path.png")

    # ─── metrics helper ─────────────────────────────────────────────
    def report_metrics(self):
        # choose the frozen snapshot if we reached the goal
        if self.metrics_frozen:
            total_dist = self.frozen_total_dist
            mean_perp  = self.frozen_mean_perp
        else:   # goal never reached
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
        # append to cumulative log
        with open("run_metrics.txt", "a") as f:
            f.write(f"{total_dist:.3f},{extra_dist:.3f},{mean_perp:.4f}\n")

# ─── run the controller ─────────────────────────────────────────────
if __name__ == "__main__":
    Bug2().run()
