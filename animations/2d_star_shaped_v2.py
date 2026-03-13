"""
star_sim.py
===========
Pygame 2D simulation of Huber2019 star-shaped obstacle avoidance.
All math lives in modulation_avoider.py.

Controls:
    Left-click   — move goal
    R            — reset robot
    SPACE        — pause / unpause
    V            — toggle basis vectors (r(xi) and e1)
    F            — toggle flow field
    ESC          — quit
"""
from __future__ import annotations
import math
import sys
import numpy as np
import pygame

from modulation.class_working import ModulationAvoider

_EPS = 1e-9

# ══════════════════════════════════════════════════════════════════════════════
#  OBSTACLE PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

OBS_CENTER = np.array([0.4, 0.0])
R_BASE     = 1.9
AMP        = 0.42
N_TIPS     = 5

def _radius_func(x_tilde: np.ndarray) -> float:
    theta = float(np.arctan2(x_tilde[1], x_tilde[0]))
    return R_BASE * (1.0 + AMP * math.cos(N_TIPS * theta))

gamma_fn      = lambda xt: ModulationAvoider.gamma_star_shaped(
    xt, np.zeros(2), _radius_func)
grad_gamma_fn = lambda xt: ModulationAvoider.grad_gamma_star_shaped(
    xt, np.zeros(2), R_BASE, AMP, N_TIPS)

avoider = ModulationAvoider(
    gamma           = gamma_fn,
    grad_gamma      = grad_gamma_fn,
    dim             = 2,
    rho             = 1.0,
    safety_eta      = 1.0,
    tail_effect     = False,
    lambda_min      = 0.0,
    reference_point = np.zeros(2),
)

# ══════════════════════════════════════════════════════════════════════════════
#  SCREEN / COORDINATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

W, H   = 900, 700
SCALE  = 72
CX, CY = W // 2, H // 2

def w2s(wx, wy):
    return (int(CX + wx * SCALE), int(CY - wy * SCALE))

def s2w(sx, sy):
    return ((sx - CX) / SCALE, (CY - sy) / SCALE)

# ══════════════════════════════════════════════════════════════════════════════
#  COLORS
# ══════════════════════════════════════════════════════════════════════════════

BG         = (255, 255, 255)
GRID_C     = (230, 230, 230)
AXIS_C     = (200, 200, 200)
OBS_FILL_C = (210, 210, 210)
OBS_EDGE_C = (80,  80,  80)
REF_PT_C   = (180,  40,  40)
ROBOT_C    = (30,  100, 200)
ROBOT_EDG  = (10,   50, 140)
GOAL_C     = (20,  150,  70)
TRAIL_C    = (120, 170, 220)
MOD_VEL_C  = (200, 100,   0)
NOM_VEL_C  = (180, 180, 180)
REF_DIR_C  = (200,  50,  20)
TAN_DIR_C  = (20,  110, 200)
TEXT_C     = (60,   60,  60)
HEAD_C     = (20,   20,  20)
FLOW_C     = (160, 200, 240)

# ══════════════════════════════════════════════════════════════════════════════
#  PRECOMPUTED ASSETS
# ══════════════════════════════════════════════════════════════════════════════

def build_obs_polygon(n: int = 360):
    thetas = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return [
        w2s(
            OBS_CENTER[0] + R_BASE*(1.0 + AMP*math.cos(N_TIPS*th))*math.cos(th),
            OBS_CENTER[1] + R_BASE*(1.0 + AMP*math.cos(N_TIPS*th))*math.sin(th),
        )
        for th in thetas
    ]

def compute_flow_field(goal: np.ndarray, cols: int = 22, rows: int = 16):
    xs = np.linspace(-W/(2*SCALE)+0.5, W/(2*SCALE)-0.5, cols)
    ys = np.linspace(-H/(2*SCALE)+0.5, H/(2*SCALE)-0.5, rows)
    arrows = []
    for wy in ys:
        for wx in xs:
            pos = np.array([wx, wy])
            xt  = pos - OBS_CENTER
            if avoider.gamma(xt) < 0.85:
                continue
            f   = 1.5 * (goal - pos)
            v   = avoider.modulate(pos, xt, f)
            mag = np.linalg.norm(v)
            if mag < _EPS:
                continue
            vn     = v / mag
            sx, sy = w2s(wx, wy)
            length = min(int(mag * 18), 26)
            arrows.append(((sx, sy), (int(sx + vn[0]*length), int(sy - vn[1]*length))))
    return arrows

# ══════════════════════════════════════════════════════════════════════════════
#  DRAWING
# ══════════════════════════════════════════════════════════════════════════════

def draw_arrow(screen, color, start, end, width=2, head=7):
    if math.hypot(end[0]-start[0], end[1]-start[1]) < 2:
        return
    pygame.draw.line(screen, color, start, end, width)
    angle = math.atan2(end[1]-start[1], end[0]-start[0])
    for sign in (+1, -1):
        ax = end[0] - head * math.cos(angle - sign * 0.45)
        ay = end[1] - head * math.sin(angle - sign * 0.45)
        pygame.draw.line(screen, color, end, (int(ax), int(ay)), width)

def draw_grid(screen):
    for ix in range(-8, 10):
        pygame.draw.line(screen, GRID_C, w2s(ix, -6), w2s(ix, 6), 1)
    for iy in range(-6, 7):
        pygame.draw.line(screen, GRID_C, w2s(-9, iy), w2s(9, iy), 1)
    pygame.draw.line(screen, AXIS_C, w2s(-9, 0), w2s(9, 0), 1)
    pygame.draw.line(screen, AXIS_C, w2s(0, -6), w2s(0, 6), 1)

def draw_obs(screen, poly_pts):
    pygame.draw.polygon(screen, OBS_FILL_C, poly_pts)
    pygame.draw.polygon(screen, OBS_EDGE_C, poly_pts, 2)
    rp = w2s(*OBS_CENTER)
    pygame.draw.circle(screen, REF_PT_C, rp, 5)
    pygame.draw.circle(screen, REF_PT_C, rp, 9, 1)

def draw_robot(screen, pos):
    rs = w2s(*pos)
    pygame.draw.circle(screen, ROBOT_C,   rs, 9)
    pygame.draw.circle(screen, ROBOT_EDG, rs, 9, 2)

def draw_goal(screen, pos):
    gs = w2s(*pos)
    pygame.draw.circle(screen, GOAL_C, gs, 10, 2)
    pygame.draw.circle(screen, GOAL_C, gs,  5)

def draw_trail(screen, trail):
    if len(trail) < 2:
        return
    for i in range(1, len(trail)):
        pygame.draw.line(screen, TRAIL_C, w2s(*trail[i-1]), w2s(*trail[i]), 2)

def draw_vectors(screen, robot_pos, xt):
    E_mat = avoider.E(xt * avoider.safety_eta)
    rs    = w2s(*robot_pos)
    VL    = 50
    r = E_mat[:, 0]
    draw_arrow(screen, REF_DIR_C, rs,
               (int(rs[0]+r[0]*VL), int(rs[1]-r[1]*VL)), width=2, head=7)
    e = E_mat[:, 1]
    draw_arrow(screen, TAN_DIR_C, rs,
               (int(rs[0]+e[0]*VL), int(rs[1]-e[1]*VL)), width=2, head=7)

def draw_velocity_arrows(screen, robot_pos, xt, goal_pos):
    f  = 1.5 * (goal_pos - robot_pos)
    v  = avoider.modulate(robot_pos, xt, f)
    rs = w2s(*robot_pos)
    fn = np.linalg.norm(f)
    if fn > _EPS:
        fu = f / fn
        ne = (int(rs[0]+fu[0]*26), int(rs[1]-fu[1]*26))
        draw_arrow(screen, NOM_VEL_C, rs, ne, width=1, head=5)
    vm = np.linalg.norm(v)
    if vm > _EPS:
        vu = v / vm
        sc = min(int(vm * 22), 50)
        ve = (int(rs[0]+vu[0]*sc), int(rs[1]-vu[1]*sc))
        draw_arrow(screen, MOD_VEL_C, rs, ve, width=2, head=8)
    return v

def draw_hud(screen, font, font_h, robot_pos, xt, v_mod, paused, show_vecs, show_flow):
    g_val   = avoider.gamma(xt)
    g_color = (200, 30, 30) if g_val < 1.0 else TEXT_C
    lines = [
        ("Star-Shaped Obstacle Avoidance  [Huber 2019]", HEAD_C, font_h),
        (f"Gamma = {g_val:.3f}" + ("  INSIDE!" if g_val < 1.0 else ""), g_color, font),
        (f"|v|   = {np.linalg.norm(v_mod):.3f}", TEXT_C, font),
        (f"pos   = ({robot_pos[0]:+.2f}, {robot_pos[1]:+.2f})", TEXT_C, font),
        ("", TEXT_C, font),
        ("[click] goal   [R] reset   [SPACE] pause", TEXT_C, font),
        (f"[V] vectors {'ON ' if show_vecs else 'OFF'}   [F] flow {'ON' if show_flow else 'OFF'}", TEXT_C, font),
    ]
    y = 10
    for text, col, fnt in lines:
        screen.blit(fnt.render(text, True, col), (12, y))
        y += fnt.size("A")[1] + 4

    legend = [
        ("-- r(xi)  reference dir  [Huber2019 E col-0]", REF_DIR_C),
        ("-- e1    tangent dir     [KB2012  E col-1]",   TAN_DIR_C),
        ("-- modulated velocity  xdot = M*f",            MOD_VEL_C),
        ("-- nominal DS  f = k*(goal - x)",              NOM_VEL_C),
        ("o   reference point xi^r (inside obstacle)",   REF_PT_C),
    ]
    for i, (txt, col) in enumerate(legend):
        s = font.render(txt, True, col)
        screen.blit(s, (W - s.get_width() - 10, H - (len(legend)-i)*18 - 8))

    if paused:
        ps = font_h.render("  PAUSED  ", True, (180, 60, 60))
        screen.blit(ps, (W//2 - ps.get_width()//2, H//2 - 12))

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Star-Shaped Obstacle Avoidance  [Huber 2019]")
    clock  = pygame.time.Clock()

    try:
        font   = pygame.font.SysFont("Courier New", 13)
        font_h = pygame.font.SysFont("Courier New", 15, bold=True)
    except Exception:
        font   = pygame.font.SysFont("monospace", 13)
        font_h = pygame.font.SysFont("monospace", 15, bold=True)

    obs_poly = build_obs_polygon()

    START_POS = np.array([-5.0, 1.8])
    robot_pos = START_POS.copy()
    goal_pos  = np.array([4.5, -1.5])
    trail: list = []
    MAX_TRAIL   = 500
    dt          = 0.018

    show_vecs = True
    show_flow = False
    paused    = False

    flow_cache_goal = None
    flow_data       = []

    while True:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            elif event.type == pygame.KEYDOWN:
                if   event.key == pygame.K_ESCAPE: pygame.quit(); sys.exit()
                elif event.key == pygame.K_SPACE:  paused    = not paused
                elif event.key == pygame.K_r:
                    robot_pos = START_POS.copy(); trail.clear()
                elif event.key == pygame.K_v: show_vecs = not show_vecs
                elif event.key == pygame.K_f:
                    show_flow = not show_flow
                    flow_cache_goal = None
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                goal_pos = np.array(s2w(*event.pos))
                flow_cache_goal = None

        if not paused:
            xt    = robot_pos - OBS_CENTER
            f     = 1.5 * (goal_pos - robot_pos)
            v_mod = avoider.modulate(robot_pos, xt, f)

            trail.append(robot_pos.copy())
            if len(trail) > MAX_TRAIL:
                trail.pop(0)

            robot_pos = robot_pos + dt * v_mod

            if np.linalg.norm(robot_pos - goal_pos) < 0.18:
                robot_pos = START_POS.copy(); trail.clear()

        if show_flow:
            if flow_cache_goal is None or not np.allclose(flow_cache_goal, goal_pos):
                flow_data       = compute_flow_field(goal_pos)
                flow_cache_goal = goal_pos.copy()

        screen.fill(BG)
        draw_grid(screen)

        if show_flow:
            for ss, se in flow_data:
                draw_arrow(screen, FLOW_C, ss, se, width=1, head=4)

        draw_obs(screen, obs_poly)
        draw_trail(screen, trail)

        xt_disp    = robot_pos - OBS_CENTER
        v_mod_disp = draw_velocity_arrows(screen, robot_pos, xt_disp, goal_pos)

        if show_vecs:
            draw_vectors(screen, robot_pos, xt_disp)

        draw_goal(screen, goal_pos)
        draw_robot(screen, robot_pos)
        draw_hud(screen, font, font_h, robot_pos, xt_disp,
                 v_mod_disp, paused, show_vecs, show_flow)

        pygame.display.flip()


if __name__ == "__main__":
    main()