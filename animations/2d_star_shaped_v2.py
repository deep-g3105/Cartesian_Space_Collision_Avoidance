"""
star_sim.py
===========
Pygame 2D simulation of Huber2019 star-shaped obstacle avoidance.
All math lives in modulation_avoider.py — this file only does simulation + rendering.

Run:
    python star_sim.py

Controls:
    Left-click   — move goal
    R            — reset robot
    SPACE        — pause / unpause
    V            — toggle basis vectors (r(ξ) and e₁)
    F            — toggle flow field
    G            — toggle gamma heatmap
    ESC          — quit
"""
from __future__ import annotations

import math
import sys

import numpy as np
import pygame

# ── Import your math class ────────────────────────────────────────────────────
# Adjust the import path if your file lives in a subdirectory, e.g.:
#   from utils.modulation_avoider import ModulationAvoider
from modulation.class_working import ModulationAvoider

_EPS = 1e-9

# ══════════════════════════════════════════════════════════════════════════════
#  OBSTACLE PARAMETERS
#  Star shape: R(θ) = R_BASE · (1 + AMP · cos(N_TIPS · θ))
#  Γ(x) = ||x - ξ^r|| / R(θ)   [Huber2019 Eq.3]
# ══════════════════════════════════════════════════════════════════════════════

OBS_CENTER = np.array([0.4, 0.0])   # world-frame obstacle centre = ξ^r
R_BASE     = 1.9                    # mean radius
AMP        = 0.42                   # tip amplitude  (0 → circle, <1 → star)
N_TIPS     = 5                      # number of tips

# ── Build gamma / grad_gamma callables for x_tilde = x - OBS_CENTER ─────────
# gamma_star_shaped  takes (x, reference_point, radius_func)
# We bind reference_point = [0,0] because x_tilde is already obstacle-relative.

def _radius_func(x_tilde: np.ndarray) -> float:
    theta = float(np.arctan2(x_tilde[1], x_tilde[0]))
    return R_BASE * (1.0 + AMP * math.cos(N_TIPS * theta))

gamma_fn     = lambda xt: ModulationAvoider.gamma_star_shaped(
    xt, np.zeros(2), _radius_func
)
grad_gamma_fn = lambda xt: ModulationAvoider.grad_gamma_star_shaped(
    xt, np.zeros(2), R_BASE, AMP, N_TIPS
)

# ── Build the avoider ─────────────────────────────────────────────────────────
#   reference_point = np.zeros(2)  →  Huber2019 E-matrix mode
#   (E col-0 = r(ξ) = x_tilde / ||x_tilde||, always radially outward)
avoider = ModulationAvoider(
    gamma           = gamma_fn,
    grad_gamma      = grad_gamma_fn,
    dim             = 2,
    rho             = 1.0,
    safety_eta      = 1.0,
    tail_effect     = False,
    lambda_min      = 0.0,          # must be ≥ 0 for impenetrability
    reference_point = np.zeros(2),  # Huber2019 star-shaped mode
)

# ══════════════════════════════════════════════════════════════════════════════
#  SCREEN / COORDINATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

W, H   = 960, 720
SCALE  = 72          # pixels per world unit
CX, CY = W // 2, H // 2


def w2s(wx, wy):
    return (int(CX + wx * SCALE), int(CY - wy * SCALE))


def s2w(sx, sy):
    return ((sx - CX) / SCALE, (CY - sy) / SCALE)


# ══════════════════════════════════════════════════════════════════════════════
#  PRECOMPUTED ASSETS
# ══════════════════════════════════════════════════════════════════════════════

def build_obs_polygon(n: int = 360):
    thetas = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return [
        w2s(
            OBS_CENTER[0] + R_BASE * (1.0 + AMP * math.cos(N_TIPS * th)) * math.cos(th),
            OBS_CENTER[1] + R_BASE * (1.0 + AMP * math.cos(N_TIPS * th)) * math.sin(th),
        )
        for th in thetas
    ]


def build_gamma_surf(res_x: int = 240, res_y: int = 180) -> pygame.Surface:
    """Vectorised Γ-field → colour-mapped Surface (computed once at startup)."""
    xmin, xmax =  -W / (2 * SCALE),  W / (2 * SCALE)
    ymax, ymin =   H / (2 * SCALE), -H / (2 * SCALE)

    world_xs = np.linspace(xmin, xmax, res_x)
    world_ys = np.linspace(ymax, ymin, res_y)
    WX, WY   = np.meshgrid(world_xs, world_ys, indexing='ij')  # (res_x, res_y)

    PX = WX - OBS_CENTER[0]
    PY = WY - OBS_CENTER[1]
    R  = np.sqrt(PX**2 + PY**2) + _EPS
    TH = np.arctan2(PY, PX)
    Rb = R_BASE * (1.0 + AMP * np.cos(N_TIPS * TH))
    G  = R / (Rb + _EPS)

    arr = np.zeros((res_x, res_y, 3), dtype=np.uint8)

    # Inside (Γ < 1): dark red
    m = G < 1.0
    arr[m, 0] = np.clip(55 * G[m], 0, 60).astype(np.uint8)

    # Near boundary (1 ≤ Γ < 2.5): red → dark blue
    m = (G >= 1.0) & (G < 2.5)
    t = np.clip((G[m] - 1.0) / 1.5, 0.0, 1.0)
    arr[m, 0] = np.clip(200 * (1.0 - t) + 10 * t, 0, 255).astype(np.uint8)
    arr[m, 2] = np.clip(50  * t, 0, 55).astype(np.uint8)

    # Far (Γ ≥ 2.5): near-background dark blue
    m = G >= 2.5
    arr[m, 2] = np.clip(40 + 12 * np.clip((G[m] - 2.5) / 5.0, 0.0, 1.0),
                        0, 60).astype(np.uint8)

    surf = pygame.Surface((res_x, res_y))
    pygame.surfarray.blit_array(surf, arr)
    return pygame.transform.smoothscale(surf, (W, H))


def compute_flow_field(goal: np.ndarray, cols: int = 20, rows: int = 14):
    """Returns list of (start_px, end_px) arrow tuples for the modulated flow."""
    xs = np.linspace(-W / (2 * SCALE) + 0.5, W / (2 * SCALE) - 0.5, cols)
    ys = np.linspace(-H / (2 * SCALE) + 0.5, H / (2 * SCALE) - 0.5, rows)
    arrows = []
    for wy in ys:
        for wx in xs:
            pos = np.array([wx, wy])
            xt  = pos - OBS_CENTER
            if avoider.gamma(xt) < 0.85:
                continue                    # skip interior points
            f   = 1.5 * (goal - pos)
            v   = avoider.modulate(pos, xt, f)   # ModulationAvoider.modulate signature
            mag = np.linalg.norm(v)
            if mag < _EPS:
                continue
            scale_px = min(mag * 18, 28)
            vn  = v / mag
            sx, sy = w2s(wx, wy)
            ex  = int(sx + vn[0] * scale_px)
            ey  = int(sy - vn[1] * scale_px)
            arrows.append(((sx, sy), (ex, ey)))
    return arrows


# ══════════════════════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

BG          = (8, 11, 22)
GRID_C      = (22, 28, 48)
OBS_EDGE_C  = (240, 80, 60)
REF_PT_C    = (255, 200, 50)
ROBOT_C     = (60, 210, 255)
ROBOT_EDG_C = (255, 255, 255)
GOAL_C      = (70, 255, 140)
MOD_VEL_C   = (255, 220, 40)
NOM_VEL_C   = (170, 170, 170)
REF_DIR_C   = (255, 120, 40)    # r(ξ)  orange
TAN_DIR_C   = (80, 190, 255)    # e₁    cyan
TRAIL_C     = (40, 90, 160)
TEXT_C      = (180, 195, 220)
HEAD_C      = (130, 145, 190)


def draw_arrow(screen, color, start, end, width=2, head=8):
    dx, dy = end[0] - start[0], end[1] - start[1]
    if math.hypot(dx, dy) < 2:
        return
    pygame.draw.line(screen, color, start, end, width)
    angle = math.atan2(dy, dx)
    for sign in (+1, -1):
        ax = end[0] - head * math.cos(angle - sign * 0.45)
        ay = end[1] - head * math.sin(angle - sign * 0.45)
        pygame.draw.line(screen, color, end, (int(ax), int(ay)), width)


def draw_grid(screen):
    for ix in range(-8, 10):
        pygame.draw.line(screen, GRID_C, w2s(ix, -6), w2s(ix, 6), 1)
    for iy in range(-6, 7):
        pygame.draw.line(screen, GRID_C, w2s(-9, iy), w2s(9, iy), 1)


def draw_robot(screen, pos):
    rs = w2s(*pos)
    # Soft glow rings
    for gr in (18, 14, 11):
        a = max(0, 30 - (18 - gr) * 8)
        gs = pygame.Surface((gr * 2 + 2, gr * 2 + 2), pygame.SRCALPHA)
        pygame.draw.circle(gs, (*ROBOT_C, a), (gr + 1, gr + 1), gr)
        screen.blit(gs, (rs[0] - gr - 1, rs[1] - gr - 1))
    pygame.draw.circle(screen, ROBOT_C,     rs, 9)
    pygame.draw.circle(screen, ROBOT_EDG_C, rs, 9, 2)
    return rs


def draw_goal(screen, pos, t):
    gs = w2s(*pos)
    pulse = int(3 * math.sin(t * 5))
    for r in (10 + pulse, 16 + pulse):
        pygame.draw.circle(screen, GOAL_C, gs, r, 1)
    pygame.draw.circle(screen, GOAL_C, gs, 7)


def draw_obs(screen, poly_pts, fill_surf):
    screen.blit(fill_surf, (0, 0))
    pygame.draw.polygon(screen, OBS_EDGE_C, poly_pts, 2)
    rp = w2s(*OBS_CENTER)
    pygame.draw.circle(screen, REF_PT_C, rp, 4)
    pygame.draw.circle(screen, REF_PT_C, rp, 9, 1)


def draw_vectors(screen, robot_pos, xt):
    """Draw E-matrix basis vectors and velocity arrows at robot position."""
    E_mat = avoider.E(xt * avoider.safety_eta)
    rs    = w2s(*robot_pos)
    VL    = 52   # vector length in pixels

    # r(ξ): col-0 of E  (Huber2019 reference direction)
    r = E_mat[:, 0]
    draw_arrow(screen, REF_DIR_C, rs,
               (int(rs[0] + r[0]*VL), int(rs[1] - r[1]*VL)), width=2, head=7)

    # e₁: col-1 of E  (tangent ⊥ ∇Γ)
    e = E_mat[:, 1]
    draw_arrow(screen, TAN_DIR_C, rs,
               (int(rs[0] + e[0]*VL), int(rs[1] - e[1]*VL)), width=2, head=7)


def draw_velocity_arrows(screen, robot_pos, xt, goal_pos):
    """Draw nominal DS and modulated velocity arrows."""
    f = 1.5 * (goal_pos - robot_pos)
    v = avoider.modulate(robot_pos, xt, f)
    rs = w2s(*robot_pos)

    # Nominal f (white)
    fn = np.linalg.norm(f)
    if fn > _EPS:
        fu = f / fn
        ne = (int(rs[0] + fu[0]*28), int(rs[1] - fu[1]*28))
        draw_arrow(screen, NOM_VEL_C, rs, ne, width=1, head=5)

    # Modulated v (yellow)
    vm = np.linalg.norm(v)
    if vm > _EPS:
        vu = v / vm
        scale = min(int(vm * 22), 55)
        ve = (int(rs[0] + vu[0]*scale), int(rs[1] - vu[1]*scale))
        draw_arrow(screen, MOD_VEL_C, rs, ve, width=2, head=8)

    return v


def draw_trail(screen, trail):
    if len(trail) < 2:
        return
    for i in range(1, len(trail)):
        age = i / len(trail)
        col = (int(TRAIL_C[0]*age), int(TRAIL_C[1]*age),
               int(TRAIL_C[2]*(0.4 + 0.6*age)))
        pygame.draw.line(screen, col,
                         w2s(*trail[i-1]), w2s(*trail[i]), 2)


def draw_hud(screen, font, font_h, robot_pos, xt, v_mod, stuck_frames, paused):
    g_val   = avoider.gamma(xt)
    g_color = (255, 60, 60) if g_val < 1.0 else TEXT_C
    lines = [
        ("Star-Shaped Obstacle Avoidance [Huber 2019]", HEAD_C, font_h),
        (f"Gamma (Γ)  = {g_val:.4f}" +
         ("  ← INSIDE!" if g_val < 1.0 else ""),              g_color, font),
        (f"|v_mod|    = {np.linalg.norm(v_mod):.4f}",          TEXT_C, font),
        (f"robot      ({robot_pos[0]:+.2f}, {robot_pos[1]:+.2f})", TEXT_C, font),
        (f"stuck t    = {stuck_frames} frames",                 TEXT_C, font),
        ("",                                                    TEXT_C, font),
        ("[SPACE] pause   [R] reset   [click] goal",            TEXT_C, font),
        ("[V] vectors   [F] flow   [G] gamma",                  TEXT_C, font),
    ]
    y = 10
    for text, col, fnt in lines:
        screen.blit(fnt.render(text, True, col), (12, y))
        y += fnt.size(text)[1] + 3

    legend = [
        ("━━  r(ξ)  reference dir  [Huber2019 E col-0]", REF_DIR_C),
        ("━━  e₁   tangent dir    [KB2012  E col-1]",    TAN_DIR_C),
        ("━━  modulated velocity  ẋ = M·f",              MOD_VEL_C),
        ("━━  nominal DS          f = k(goal - x)",      NOM_VEL_C),
        ("◆   reference point ξ^r (inside obstacle)",    REF_PT_C),
    ]
    for i, (txt, col) in enumerate(legend):
        s = font.render(txt, True, col)
        screen.blit(s, (W - s.get_width() - 10, H - (len(legend) - i) * 20 - 8))

    if paused:
        ps = font_h.render("  PAUSED  ", True, (255, 200, 50))
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
        font_h = pygame.font.SysFont("Courier New", 16, bold=True)
    except Exception:
        font   = pygame.font.SysFont("monospace", 13)
        font_h = pygame.font.SysFont("monospace", 16, bold=True)

    print("Precomputing gamma field…")
    gamma_surf = build_gamma_surf()
    obs_poly   = build_obs_polygon()
    obs_fill   = pygame.Surface((W, H), pygame.SRCALPHA)
    pygame.draw.polygon(obs_fill, (*OBS_EDGE_C, 45), obs_poly)

    # ── State ─────────────────────────────────────────────────────────────────
    START_POS    = np.array([-5.0, 1.8])
    robot_pos    = START_POS.copy()
    goal_pos     = np.array([4.5, -1.5])
    trail: list  = []
    MAX_TRAIL    = 400
    stuck_frames = 0
    sim_time     = 0.0
    dt           = 0.018

    show_vecs  = True
    show_gamma = True
    show_flow  = False
    paused     = False

    flow_cache_goal = None
    flow_data       = []

    print("Ready.")

    while True:
        clock.tick(60)

        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            elif event.type == pygame.KEYDOWN:
                if   event.key == pygame.K_ESCAPE: pygame.quit(); sys.exit()
                elif event.key == pygame.K_SPACE:  paused    = not paused
                elif event.key == pygame.K_r:
                    robot_pos = START_POS.copy(); trail.clear(); stuck_frames = 0
                elif event.key == pygame.K_v: show_vecs  = not show_vecs
                elif event.key == pygame.K_f: show_flow  = not show_flow
                elif event.key == pygame.K_g: show_gamma = not show_gamma
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                goal_pos = np.array(s2w(*event.pos))
                flow_cache_goal = None

        # ── Physics ───────────────────────────────────────────────────────────
        if not paused:
            xt = robot_pos - OBS_CENTER

            def f_func(p): return 1.5 * (goal_pos - p)

            # 1. Saddle-point escape FIRST (KB2012 Alg.1)
            #    Must run before modulate so the escaped position is used
            #    for integration — not overwritten by stale velocity.
            

            # 2. Nominal DS and modulated velocity
            f     = f_func(robot_pos)
            v_mod = avoider.modulate(robot_pos, xt, f)

            stuck_frames = (stuck_frames + 1) if np.linalg.norm(v_mod) < 5e-3 \
                           else 0

            # 3. Integrate
            trail.append(robot_pos.copy())
            if len(trail) > MAX_TRAIL:
                trail.pop(0)
            robot_pos = robot_pos + dt * v_mod
            sim_time += dt

            if np.linalg.norm(robot_pos - goal_pos) < 0.18:
                robot_pos = START_POS.copy(); trail.clear(); stuck_frames = 0

        # Flow field cache
        if show_flow:
            if flow_cache_goal is None or not np.allclose(flow_cache_goal, goal_pos):
                flow_data       = compute_flow_field(goal_pos)
                flow_cache_goal = goal_pos.copy()

        # ── Draw ──────────────────────────────────────────────────────────────
        screen.fill(BG)

        if show_gamma:
            screen.blit(gamma_surf, (0, 0))

        draw_grid(screen)

        if show_flow:
            for ss, se in flow_data:
                pygame.draw.line(screen, (50, 80, 130), ss, se, 1)
                pygame.draw.circle(screen, (60, 100, 160), se, 3)

        draw_obs(screen, obs_poly, obs_fill)
        draw_trail(screen, trail)

        xt_disp = robot_pos - OBS_CENTER
        v_mod_disp = draw_velocity_arrows(screen, robot_pos, xt_disp, goal_pos)

        if show_vecs:
            draw_vectors(screen, robot_pos, xt_disp)

        draw_goal(screen, goal_pos, sim_time)
        draw_robot(screen, robot_pos)
        draw_hud(screen, font, font_h, robot_pos, xt_disp,
                 v_mod_disp, stuck_frames, paused)

        pygame.display.flip()


if __name__ == "__main__":
    main()