"""
2d_joint_space.py

Interactive 2D illustration of joint-space DS modulation for obstacle
avoidance (Koptev, Figueroa, Billard, IJRR 2024, Figs. 2-5), built on the
same classes as the FR3 simulation (utils/joint_space_oa):

    nominal DS     f(q) = LinearJointDS            (Eq. 36)
    distance       Gamma(q), grad Gamma(q)         (Eq. 3)   - PlanarArmDistance (exact, instead of the network)
    modulation     q_dot = M(q) f(q) + f_rep(q)    (Eqs. 5-7, 12-13, 29-30, 33-35) - JointSpaceModulator

Left:  task space - planar 2-link arm, circular obstacles with their safety
       (eps) and influence (d2) rings, goal ghost arm, end-effector trail.
Right: joint space - C-obstacle map (red: Gamma <= 0, orange: modulation band
       0 < Gamma < d2), modulated vector field, trajectory, and at the current
       state the nominal f (grey), modulated q_dot (blue) and normal n (red).
Bottom: Gamma, lambda_n, lambda_tau, lambda_v, f_rep and the sigmoid gains.

Controls
    task panel   left-drag obstacle: move | left click / drag elsewhere: goal (IK)
                 right click: add obstacle / remove obstacle | wheel on obstacle: resize
    joint panel  left click / drag: goal in joint space | shift + left click: move robot there
    keys         1 2 3 scenarios | SPACE pause | R reset robot | S swap start/goal
                 M modulation on/off | T tail-effect on/off | F vector field | E elbow up/down
                 + / - sim speed | C clear trails | ESC quit

Run:  python3 animations/2d_joint_space.py
"""

import argparse
import math
import os
import sys

import numpy as np
import pygame

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.joint_space_oa import (PlanarArm, PlanarArmDistance, JointSpaceModulator,
                                  LinearJointDS, sigmoid_gain)

# ---------------------------------------------------------------------- #
# layout & style
# ---------------------------------------------------------------------- #
W, H = 1420, 870
TASK = pygame.Rect(30, 70, 640, 640)
JOINT = pygame.Rect(750, 70, 640, 640)
HUD = pygame.Rect(30, 725, 1360, 130)
TASK_HALF = 1.05                       # task panel shows [-1.05, 1.05] m
MAP_RES = 170                          # joint-space map resolution
FIELD_RES = 21                         # vector-field arrows per axis

BG = (243, 245, 249)
PANEL = (255, 255, 255)
INK = (33, 37, 41)
MUTED = (120, 128, 138)
GRID = (228, 232, 238)
LINK1, LINK2 = (44, 62, 80), (52, 120, 190)
OBST = (220, 68, 55)
BAND = (252, 205, 160)
FREE_NEAR, FREE_FAR = (255, 247, 240), (255, 255, 255)
TRAJ = (35, 110, 220)
GOAL = (240, 160, 30)
GOOD, WARN, BAD = (34, 150, 80), (230, 140, 20), (210, 50, 50)

# ---------------------------------------------------------------------- #
# model (same classes as the FR3 simulation)
# ---------------------------------------------------------------------- #
SIM_DT = 0.004
DS_SPEED = 1.0                          # rad/s
ARM = PlanarArm(l1=0.5, l2=0.4, link_radius=0.03)
DIST = PlanarArmDistance(ARM)


def polar(angle, r):
    return np.array([r * math.cos(angle), r * math.sin(angle)])


# start q, goal q, obstacles [(centre, radius)]
SCENARIOS = {
    1: dict(name="single obstacle (passed around its tip)",
            start=[-2.4, -0.8], goal=[2.4, -0.8], obstacles=[(polar(0.5, 0.75), 0.09)]),
    2: dict(name="two obstacles: avoids the first, LOCAL MINIMUM at the second (cf. IJRR Fig. 3)",
            start=[-2.6, 0.16], goal=[2.6, 0.16],
            obstacles=[(polar(-1.11, 0.72), 0.06), (polar(0.98, 0.78), 0.09)]),
    3: dict(name="three obstacles (cluttered, passed)",
            start=[-2.6, -0.94], goal=[2.6, -0.94],
            obstacles=[(polar(-1.3, 0.73), 0.09), (polar(0.26, 0.74), 0.06), (polar(0.96, 0.66), 0.08)]),
}


# ---------------------------------------------------------------------- #
# coordinate maps
# ---------------------------------------------------------------------- #
def task_to_px(x):
    s = TASK.width / (2 * TASK_HALF)
    return (int(TASK.centerx + x[0] * s), int(TASK.centery - x[1] * s))


def px_to_task(p):
    s = TASK.width / (2 * TASK_HALF)
    return np.array([(p[0] - TASK.centerx) / s, (TASK.centery - p[1]) / s])


def task_len(d):
    return d * TASK.width / (2 * TASK_HALF)


def q_to_px(q):
    return (int(JOINT.left + (q[0] + math.pi) / (2 * math.pi) * JOINT.width),
            int(JOINT.bottom - (q[1] + math.pi) / (2 * math.pi) * JOINT.height))


def px_to_q(p):
    return np.array([(p[0] - JOINT.left) / JOINT.width * 2 * math.pi - math.pi,
                     (JOINT.bottom - p[1]) / JOINT.height * 2 * math.pi - math.pi])


# ---------------------------------------------------------------------- #
# drawing helpers
# ---------------------------------------------------------------------- #
def arrow(surf, color, start, vec_px, width=3, head=9):
    end = (start[0] + vec_px[0], start[1] + vec_px[1])
    L = math.hypot(*vec_px)
    if L < 2:
        return
    pygame.draw.line(surf, color, start, end, width)
    ux, uy = vec_px[0] / L, vec_px[1] / L
    left = (end[0] - head * (ux - 0.5 * uy), end[1] - head * (uy + 0.5 * ux))
    right = (end[0] - head * (ux + 0.5 * uy), end[1] - head * (uy - 0.5 * ux))
    pygame.draw.polygon(surf, color, [end, left, right])


def dashed_circle(surf, color, center, radius, dash=10, width=1):
    n = max(12, int(2 * math.pi * radius / dash))
    for i in range(0, n, 2):
        a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
        pygame.draw.line(surf, color, (center[0] + radius * math.cos(a0), center[1] + radius * math.sin(a0)),
                         (center[0] + radius * math.cos(a1), center[1] + radius * math.sin(a1)), width)


def dashed_line(surf, color, a, b, dash=8, width=1):
    L = math.hypot(b[0] - a[0], b[1] - a[1])
    n = max(1, int(L / dash))
    for i in range(0, n, 2):
        t0, t1 = i / n, min(1.0, (i + 1) / n)
        pygame.draw.line(surf, color, (a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0),
                         (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1), width)


def star(surf, color, c, r=10):
    pts = [(c[0] + (r if i % 2 == 0 else r * 0.45) * math.cos(-math.pi / 2 + i * math.pi / 5),
            c[1] + (r if i % 2 == 0 else r * 0.45) * math.sin(-math.pi / 2 + i * math.pi / 5)) for i in range(10)]
    pygame.draw.polygon(surf, color, pts)
    pygame.draw.polygon(surf, INK, pts, 1)


def draw_arm(surf, q, alpha=255, width_scale=1.0):
    base, elbow, ee = ARM.fk(q)
    w = max(3, int(2 * task_len(ARM.w) * width_scale))
    layer = pygame.Surface((W, H), pygame.SRCALPHA)
    for a, b, col in ((base, elbow, LINK1), (elbow, ee, LINK2)):
        pygame.draw.line(layer, col + (alpha,), task_to_px(a), task_to_px(b), w)
        pygame.draw.circle(layer, col + (alpha,), task_to_px(a), w // 2)
        pygame.draw.circle(layer, col + (alpha,), task_to_px(b), w // 2)
    pygame.draw.circle(layer, (90, 90, 90, alpha), task_to_px(base), w // 2 + 4)
    pygame.draw.circle(layer, (255, 255, 255, alpha), task_to_px(elbow), max(2, w // 4))
    surf.blit(layer, (0, 0))


# ---------------------------------------------------------------------- #
# simulation state
# ---------------------------------------------------------------------- #
class Sim:
    def __init__(self):
        self.mod = JointSpaceModulator(dt=SIM_DT, max_qdot=2.0, q_min=[-math.pi] * 2, q_max=[math.pi] * 2)
        self.modulation = True
        self.paused = False
        self.show_field = True
        self.elbow_up = True
        self.speed = 1.0
        self.load(1)

    def load(self, k):
        sc = SCENARIOS[k]
        self.scenario = k
        self.start = np.array(sc["start"], dtype=float)
        self.obstacles = [[np.array(c, dtype=float), float(r)] for c, r in sc["obstacles"]]
        self.set_goal(np.array(sc["goal"], dtype=float))
        self.reset()

    def reset(self):
        self.q = self.start.copy()
        self.qdot = np.zeros(2)
        self.trail_q, self.trail_ee = [self.q.copy()], [ARM.fk(self.q)[2]]
        self.still_time = 0.0
        self.collided = False
        self.update_readout()

    def set_goal(self, q_goal):
        self.ds = LinearJointDS(np.clip(q_goal, -math.pi + 0.06, math.pi - 0.06), speed=DS_SPEED)
        self.dirty = True

    def arrays(self):
        if not self.obstacles:
            return np.zeros((0, 2)), np.zeros(0)
        return np.array([c for c, _ in self.obstacles]), np.array([r for _, r in self.obstacles])

    def update_readout(self):
        C, R = self.arrays()
        self.f = self.ds(self.q)
        self.gamma, self.grad, self.dinfo = DIST(self.q, C, R)
        if self.modulation:
            _, self.minfo = self.mod.velocity(self.f, self.gamma, self.grad)
        else:
            self.minfo = {}

    def step(self, frame_dt):
        if self.paused:
            return
        C, R = self.arrays()
        n_sub = max(1, int(round(self.speed * frame_dt / SIM_DT)))
        for _ in range(n_sub):
            f = self.ds(self.q)
            g, grad, _ = DIST(self.q, C, R)
            if self.modulation:
                self.q, self.qdot, _ = self.mod.step(self.q, f, g, grad)
            else:
                self.qdot = self.mod._limit(f)
                self.q = np.clip(self.q + self.qdot * SIM_DT, -math.pi + 0.05, math.pi - 0.05)
            self.collided |= g <= 0.0
        if np.linalg.norm(self.q - self.trail_q[-1]) > 0.01:
            self.trail_q.append(self.q.copy())
            self.trail_ee.append(ARM.fk(self.q)[2])
            self.trail_q, self.trail_ee = self.trail_q[-1500:], self.trail_ee[-1500:]
        moving = np.linalg.norm(self.qdot) > 0.03
        self.still_time = 0.0 if moving else self.still_time + frame_dt
        self.update_readout()

    def status(self):
        if self.ds.distance(self.q) < 0.03:
            return "GOAL REACHED", GOOD
        if self.gamma <= 0.0:
            return "IN COLLISION", BAD
        if self.still_time > 0.6:
            return "LOCAL MINIMUM  (f anti-parallel to n - needs MPC navigation kernels)", WARN
        return ("MOVING" + ("" if self.modulation else "  (modulation OFF)")), (INK if self.modulation else BAD)


# ---------------------------------------------------------------------- #
# joint-space map & vector field (recomputed when the scene changes)
# ---------------------------------------------------------------------- #
class JointMap:
    def __init__(self):
        g = np.linspace(-math.pi, math.pi, MAP_RES)
        self.Q = np.stack(np.meshgrid(g, g[::-1], indexing="xy"), -1).reshape(-1, 2)   # row-major, top row = +pi
        self.surface = None
        self.field = []

    def update(self, sim):
        C, R = sim.arrays()
        G = DIST.gamma_grid(self.Q, C, R).reshape(MAP_RES, MAP_RES)                     # [row (q2), col (q1)]
        d2 = sim.mod.d2
        t = np.clip((G - d2) / 0.6, 0, 1)[..., None]
        img = (1 - t) * np.array(FREE_NEAR) + t * np.array(FREE_FAR)
        img = np.where((G < d2)[..., None], np.array(BAND), img)
        coll = G <= 0
        img = np.where(coll[..., None], np.array(OBST), img)
        edge = coll & ~(np.roll(coll, 1, 0) & np.roll(coll, -1, 0) & np.roll(coll, 1, 1) & np.roll(coll, -1, 1))
        img = np.where(edge[..., None], np.array((150, 30, 25)), img)
        surf = pygame.surfarray.make_surface(img.transpose(1, 0, 2).astype(np.uint8))
        self.surface = pygame.transform.smoothscale(surf, JOINT.size)

        self.field = []
        if sim.show_field:
            g = np.linspace(-math.pi, math.pi, FIELD_RES + 2)[1:-1]
            for q1 in g:
                for q2 in g:
                    q = np.array([q1, q2])
                    gam, grad, _ = DIST(q, C, R)
                    if gam <= 0:
                        continue
                    f = sim.ds(q)
                    v = sim.mod.velocity(f, gam, grad)[0] if sim.modulation else f
                    nv = np.linalg.norm(v)
                    if nv > 1e-6:
                        self.field.append((q, v / nv, gam < d2))
        sim.dirty = False

    def draw(self, screen, sim):
        screen.blit(self.surface, JOINT.topleft)
        for q, u, near in self.field:
            p = q_to_px(q)
            L = 13
            arrow(screen, (95, 105, 120) if not near else (180, 110, 40), p, (u[0] * L, -u[1] * L), width=1, head=5)


# ---------------------------------------------------------------------- #
# panels
# ---------------------------------------------------------------------- #
def draw_task_panel(screen, sim, fonts):
    pygame.draw.rect(screen, PANEL, TASK, border_radius=8)
    screen.set_clip(TASK)
    for v in np.arange(-1.0, 1.01, 0.25):
        pygame.draw.line(screen, GRID, task_to_px((v, -TASK_HALF)), task_to_px((v, TASK_HALF)))
        pygame.draw.line(screen, GRID, task_to_px((-TASK_HALF, v)), task_to_px((TASK_HALF, v)))
    dashed_circle(screen, (200, 205, 215), task_to_px((0, 0)), task_len(ARM.l1 + ARM.l2), dash=8)

    band = pygame.Surface((W, H), pygame.SRCALPHA)
    for c, r in sim.obstacles:
        pygame.draw.circle(band, BAND + (110,), task_to_px(c), int(task_len(r + ARM.w + sim.mod.d2)))
    screen.blit(band, (0, 0))
    for c, r in sim.obstacles:
        dashed_circle(screen, (235, 150, 90), task_to_px(c), task_len(r + ARM.w + sim.mod.d2), dash=7)
        dashed_circle(screen, OBST, task_to_px(c), task_len(r + ARM.w + sim.mod.eps), dash=5)
        pygame.draw.circle(screen, OBST, task_to_px(c), int(task_len(r)))
        pygame.draw.circle(screen, (150, 30, 25), task_to_px(c), int(task_len(r)), 2)

    if len(sim.trail_ee) > 1:
        pygame.draw.lines(screen, TRAJ, False, [task_to_px(p) for p in sim.trail_ee], 2)
    draw_arm(screen, sim.ds.q_goal, alpha=70)
    star(screen, GOAL, task_to_px(ARM.fk(sim.ds.q_goal)[2]), 11)
    draw_arm(screen, sim.q)

    if sim.dinfo.get("point") is not None and np.isfinite(sim.gamma) and sim.gamma < 0.25:
        c, r = sim.obstacles[sim.dinfo["sphere"]]
        p = sim.dinfo["point"]
        u = (p - c) / max(np.linalg.norm(p - c), 1e-9)
        a, b = task_to_px(p - u * ARM.w), task_to_px(c + u * r)
        col = BAD if sim.gamma < sim.mod.eps else (WARN if sim.gamma < sim.mod.d2 else MUTED)
        dashed_line(screen, col, a, b, dash=5, width=2)
        lbl = fonts["s"].render(f"Γ {100 * sim.gamma:.1f} cm", True, col)
        box = lbl.get_rect(midleft=((a[0] + b[0]) // 2 + 10, (a[1] + b[1]) // 2))
        pygame.draw.rect(screen, (255, 255, 255), box.inflate(8, 4), border_radius=4)
        pygame.draw.rect(screen, col, box.inflate(8, 4), 1, border_radius=4)
        screen.blit(lbl, box)
    screen.set_clip(None)
    pygame.draw.rect(screen, (210, 215, 222), TASK, 1, border_radius=8)
    screen.blit(fonts["b"].render("TASK SPACE", True, INK), (TASK.left, TASK.top - 26))
    screen.blit(fonts["s"].render("drag obstacles / click goal / right-click add-remove / wheel resize",
                                  True, MUTED), (TASK.left + 120, TASK.top - 24))


def draw_joint_panel(screen, sim, jmap, fonts):
    jmap.draw(screen, sim)
    for v in (-math.pi / 2, 0, math.pi / 2):
        pygame.draw.line(screen, (215, 220, 228), q_to_px((v, -math.pi)), q_to_px((v, math.pi)))
        pygame.draw.line(screen, (215, 220, 228), q_to_px((-math.pi, v)), q_to_px((math.pi, v)))
    labels = {-math.pi: "-π", -math.pi / 2: "-π/2", 0: "0", math.pi / 2: "π/2", math.pi: "π"}
    for v, s in labels.items():
        t = fonts["s"].render(s, True, MUTED)
        x, _ = q_to_px((v, -math.pi))
        screen.blit(t, (x - t.get_width() // 2, JOINT.bottom + 4))
        _, y = q_to_px((-math.pi, v))
        screen.blit(t, (JOINT.left - t.get_width() - 6, y - 8))

    screen.set_clip(JOINT)
    dashed_line(screen, (120, 130, 145), q_to_px(sim.q), q_to_px(sim.ds.q_goal), dash=7)
    if len(sim.trail_q) > 1:
        pts = [q_to_px(q) for q in sim.trail_q]
        pygame.draw.lines(screen, TRAJ, False, pts, 3)
    star(screen, GOAL, q_to_px(sim.ds.q_goal), 12)
    pygame.draw.circle(screen, (40, 170, 90), q_to_px(sim.start), 6)

    p = q_to_px(sim.q)
    s = 55
    fn = sim.f / max(np.linalg.norm(sim.f), 1e-9)
    arrow(screen, (140, 145, 155), p, (fn[0] * s, -fn[1] * s), width=3)
    if np.linalg.norm(sim.qdot) > 1e-4:
        qn = sim.qdot / np.linalg.norm(sim.qdot) * min(1.0, np.linalg.norm(sim.qdot) / DS_SPEED)
        arrow(screen, TRAJ, p, (qn[0] * s, -qn[1] * s), width=4)
    if np.isfinite(sim.gamma) and sim.gamma < sim.mod.d2 and np.linalg.norm(sim.grad) > 1e-9:
        n = sim.grad / np.linalg.norm(sim.grad)
        arrow(screen, OBST, p, (n[0] * 40, -n[1] * 40), width=3)
    pygame.draw.circle(screen, INK, p, 8)
    pygame.draw.circle(screen, (255, 255, 255), p, 4)
    screen.set_clip(None)
    pygame.draw.rect(screen, (210, 215, 222), JOINT, 1)
    for txt, pos in (("q2 [rad] ↑", (JOINT.left + 6, JOINT.top + 5)), ("q1 [rad] →", (JOINT.right - 82, JOINT.bottom - 22))):
        t = fonts["s"].render(txt, True, INK)
        r = t.get_rect(topleft=pos)
        pygame.draw.rect(screen, (255, 255, 255), r.inflate(6, 2))
        screen.blit(t, r)
    head = fonts["b"].render("JOINT SPACE", True, INK)
    screen.blit(head, (JOINT.left, JOINT.top - 26))
    legend = [((140, 145, 155), "f(q)"), (TRAJ, "q̇"), (OBST, "n"),
              (OBST, "Γ ≤ 0 (C-obstacle)"), (BAND, "0 < Γ < d2 (modulated)")]
    x = JOINT.left + head.get_width() + 24
    for col, txt in legend:
        pygame.draw.rect(screen, col, (x, JOINT.top - 19, 12, 12), border_radius=2)
        t = fonts["s"].render(txt, True, MUTED)
        screen.blit(t, (x + 16, JOINT.top - 22))
        x += t.get_width() + 30


def draw_hud(screen, sim, fonts):
    pygame.draw.rect(screen, PANEL, HUD, border_radius=8)
    pygame.draw.rect(screen, (210, 215, 222), HUD, 1, border_radius=8)
    status, col = sim.status()
    screen.blit(fonts["b"].render(status, True, col), (HUD.left + 16, HUD.top + 12))
    screen.blit(fonts["s"].render(f"scenario {sim.scenario}: {SCENARIOS[sim.scenario]['name']}", True, MUTED),
                (HUD.left + 16, HUD.top + 40))
    flags = (f"modulation {'ON' if sim.modulation else 'OFF'}  |  tail effect {'ON' if sim.mod.tail_effect else 'OFF'}"
             f"  |  speed x{sim.speed:.2g}  |  {'PAUSED' if sim.paused else 'running'}"
             f"  |  goal distance {sim.ds.distance(sim.q):.2f} rad")
    screen.blit(fonts["s"].render(flags, True, MUTED), (HUD.left + 16, HUD.top + 62))
    keys = "1 2 3 scenario · SPACE pause · R reset · S swap · M modulation · T tail · F field · E elbow · +/- speed · C clear"
    screen.blit(fonts["s"].render(keys, True, (160, 166, 175)), (HUD.left + 16, HUD.top + 100))

    m = sim.minfo
    vals = [("Γ(q)", f"{100 * sim.gamma:5.1f}cm" if np.isfinite(sim.gamma) else "  ∞"),
            ("λ_n'", f"{m.get('lambda_n', 1.0):.2f}"), ("λ_τ", f"{m.get('lambda_tau', 1.0):.2f}"),
            ("λ_v", f"{m.get('lambda_v', 1.0):.2f}"), ("f_rep", f"{m.get('repulsion', 0.0):.2f}"),
            ("|q̇|", f"{np.linalg.norm(sim.qdot):.2f}"), ("link", f"{sim.dinfo.get('link') or '-'}")]
    x0 = HUD.left + 600
    for i, (k, v) in enumerate(vals):
        x, y = x0 + (i % 3) * 145, HUD.top + 12 + (i // 3) * 24
        screen.blit(fonts["s"].render(k, True, MUTED), (x, y))
        screen.blit(fonts["m"].render(v, True, INK), (x + 46, y - 1))

    # sigmoid gains of Eq. (12)-(13) with the current Gamma
    plot = pygame.Rect(HUD.right - 300, HUD.top + 14, 280, 100)
    pygame.draw.rect(screen, (248, 249, 251), plot)
    gmax = 0.15
    gx = lambda gm: plot.left + gm / gmax * plot.width
    gy = lambda lam: plot.bottom - lam / 2.2 * plot.height
    for d, lbl in ((sim.mod.eps, "ε"), (sim.mod.d2, "d2")):
        dashed_line(screen, (190, 195, 205), (gx(d), plot.top), (gx(d), plot.bottom), dash=4)
        screen.blit(fonts["s"].render(lbl, True, MUTED), (gx(d) + 3, plot.top))
    gs = np.linspace(0, gmax, 120)
    for fn, col in ((sim.mod.lambda_n, OBST), (sim.mod.lambda_tau, TRAJ)):
        pygame.draw.lines(screen, col, False, [(gx(g), gy(fn(g))) for g in gs], 2)
    if np.isfinite(sim.gamma) and sim.gamma < gmax:
        g = max(sim.gamma, 0.0)
        pygame.draw.line(screen, INK, (gx(g), plot.top), (gx(g), plot.bottom), 1)
        for fn, col in ((sim.mod.lambda_n, OBST), (sim.mod.lambda_tau, TRAJ)):
            pygame.draw.circle(screen, col, (int(gx(g)), int(gy(fn(g)))), 4)
    screen.blit(fonts["s"].render("λ_n", True, OBST), (plot.left - 34, gy(1.0) - 8))
    screen.blit(fonts["s"].render("λ_τ", True, TRAJ), (plot.left - 34, gy(2.0) - 8))
    t = fonts["s"].render("Γ: 0 … 15 cm", True, MUTED)
    screen.blit(t, (plot.right - t.get_width() - 4, plot.bottom - 18))


# ---------------------------------------------------------------------- #
# main loop
# ---------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=int, default=1, choices=list(SCENARIOS))
    ap.add_argument("--screenshot", default=None, help="save a screenshot after --frames frames and exit")
    ap.add_argument("--frames", type=int, default=300)
    args = ap.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Joint-space DS modulation (IJRR 2024) - planar 2-link arm")
    clock = pygame.time.Clock()
    fonts = {"s": pygame.font.SysFont("dejavusans,arial", 14),
             "m": pygame.font.SysFont("dejavusansmono,monospace", 15),
             "b": pygame.font.SysFont("dejavusans,arial", 17, bold=True)}
    title = pygame.font.SysFont("dejavusans,arial", 22, bold=True)

    sim = Sim()
    sim.load(args.scenario)
    jmap = JointMap()
    drag_obs, drag_goal_task, drag_goal_joint = None, False, False
    frame = 0

    running = True
    while running:
        frame_dt = min(clock.tick(60) / 1000.0, 0.05)
        mouse = pygame.mouse.get_pos()
        shift = pygame.key.get_mods() & pygame.KMOD_SHIFT

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k == pygame.K_ESCAPE:
                    running = False
                elif k in (pygame.K_1, pygame.K_2, pygame.K_3):
                    sim.load(int(ev.unicode))
                elif k == pygame.K_SPACE:
                    sim.paused = not sim.paused
                elif k == pygame.K_r:
                    sim.reset()
                elif k == pygame.K_s:
                    sim.start, goal = sim.ds.q_goal.copy(), sim.start.copy()
                    sim.set_goal(goal)
                    sim.reset()
                elif k == pygame.K_m:
                    sim.modulation = not sim.modulation
                    sim.dirty = True
                elif k == pygame.K_t:
                    sim.mod.tail_effect = not sim.mod.tail_effect
                    sim.dirty = True
                elif k == pygame.K_f:
                    sim.show_field = not sim.show_field
                    sim.dirty = True
                elif k == pygame.K_e:
                    sim.elbow_up = not sim.elbow_up
                    sim.set_goal(ARM.ik(ARM.fk(sim.ds.q_goal)[2], sim.elbow_up))
                elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    sim.speed = min(4.0, sim.speed * 1.5)
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    sim.speed = max(0.1, sim.speed / 1.5)
                elif k == pygame.K_c:
                    sim.trail_q, sim.trail_ee = [sim.q.copy()], [ARM.fk(sim.q)[2]]
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if TASK.collidepoint(ev.pos):
                    x = px_to_task(ev.pos)
                    hit = next((i for i, (c, r) in enumerate(sim.obstacles)
                                if np.linalg.norm(x - c) < r + 0.03), None)
                    if ev.button == 1:
                        if hit is not None:
                            drag_obs = hit
                        else:
                            drag_goal_task = True
                            sim.set_goal(ARM.ik(x, sim.elbow_up))
                    elif ev.button == 3:
                        if hit is not None:
                            sim.obstacles.pop(hit)
                        else:
                            sim.obstacles.append([x, 0.08])
                        sim.dirty = True
                    elif ev.button in (4, 5) and hit is not None:
                        sim.obstacles[hit][1] = float(np.clip(sim.obstacles[hit][1] * (1.1 if ev.button == 4 else 0.9),
                                                              0.03, 0.3))
                        sim.dirty = True
                elif JOINT.collidepoint(ev.pos) and ev.button == 1:
                    q = px_to_q(ev.pos)
                    if shift:
                        sim.start = q.copy()
                        sim.reset()
                    else:
                        drag_goal_joint = True
                        sim.set_goal(q)
            elif ev.type == pygame.MOUSEBUTTONUP:
                drag_obs, drag_goal_task, drag_goal_joint = None, False, False

        if drag_obs is not None and TASK.collidepoint(mouse):
            sim.obstacles[drag_obs][0] = px_to_task(mouse)
            sim.dirty = True
        if drag_goal_task and TASK.collidepoint(mouse):
            sim.set_goal(ARM.ik(px_to_task(mouse), sim.elbow_up))
        if drag_goal_joint and JOINT.collidepoint(mouse):
            sim.set_goal(px_to_q(mouse))

        sim.step(frame_dt)
        if sim.dirty or jmap.surface is None:
            jmap.update(sim)

        screen.fill(BG)
        t = title.render("Joint-space obstacle avoidance by DS modulation", True, INK)
        screen.blit(t, (30, 16))
        screen.blit(fonts["s"].render("q̇ = M(q) f(q) + f_rep(q),   M = E D Eᵀ,   n = ∇Γ/|∇Γ|   (Koptev et al., IJRR 2024)",
                                      True, MUTED), (30 + t.get_width() + 24, 24))
        draw_task_panel(screen, sim, fonts)
        draw_joint_panel(screen, sim, jmap, fonts)
        draw_hud(screen, sim, fonts)
        pygame.display.flip()

        frame += 1
        if args.screenshot and frame >= args.frames:
            pygame.image.save(screen, args.screenshot)
            running = False

    pygame.quit()


if __name__ == "__main__":
    main()
