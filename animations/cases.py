"""
interactive_algo1.py

Interactive visualization of Algorithm-1 (paper) — medium feature set.

Controls:
    - Left-click & drag near AGENT to move agent
    - Left-click & drag near TARGET to move target
    - Middle-click (or right-click if no middle) to set OBSTACLE CENTER
    - Mouse wheel up / down: increase / decrease obstacle radius
    - SPACE : toggle Algorithm-1 mode (AUTO vs MANUAL)
    - N : perform one manual Algorithm-1 iteration (when stuck)
    - L : toggle Local-minimum mode (a small sink at a boundary point)
    - S : toggle Saddle-prone mode (align target so f is collinear with normal)
    - UP / DOWN : increase / decrease ALPHA (perturb step)
    - R : reset agent to start
    - ESC : quit
"""

import pygame, sys, math, random
import numpy as np

# ---------- Config ----------
W, H = 1000, 700
FPS = 60

# initial obstacle (circle)
obs_center = np.array([W//2, H//2], dtype=float)
obs_radius = 120.0

# initial agent / target
agent_start = np.array([obs_center[0] - 300.0, obs_center[1] + 30.0], dtype=float)
agent = agent_start.copy()
target = np.array([obs_center[0] + 300.0, obs_center[1] - 60.0], dtype=float)

# algorithm1 params
ALPHA = 0.12          # step amplitude along tangent
DT = 1.0 / FPS        # integration dt
STUCK_VEL_THRESH = 0.12
EPS_BOUND = 0.02      # tolerance for boundary (in Gamma units)
RHO = 1.0

# DS gain
K = 1.5

# local-min sink parameters
local_min_on = False
local_min_angle = math.radians(130)  # angle for boundary sink
local_min_point = np.array([obs_center[0] + obs_radius*math.cos(local_min_angle),
                            obs_center[1] + obs_radius*math.sin(local_min_angle)])
LOCAL_MIN_GAIN = 6.0
LOCAL_MIN_RADIUS = 70.0

# saddle-prone mode
saddle_on = False

# algorithm modes
algo_auto = False    # if True: auto nudging; if False: manual/step
step_request = False

# visual & interaction
dragging_agent = False
dragging_target = False
dragging_offset = np.zeros(2)
dragging_radius = False

show_vectors = True

# colors
WHITE = (245,245,245)
BLACK = (10,10,10)
RED = (220,70,70)
BLUE = (55,120,220)
GREEN = (60,200,120)
CYAN = (80,220,220)
YELLOW = (240,200,50)
ORANGE = (255,140,20)
GREY = (200,200,200)
DARK = (40,40,40)

pygame.init()
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Interactive Algorithm-1 visualization (Option B)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Consolas", 18)

# ---------- Helper functions ----------
def Gamma(pt, center=obs_center, radius=obs_radius):
    dx = pt[0] - center[0]
    dy = pt[1] - center[1]
    return (dx*dx + dy*dy) / (radius*radius)

def gradGamma(pt, center=obs_center, radius=obs_radius):
    dx = pt[0] - center[0]
    dy = pt[1] - center[1]
    return np.array([2*dx / (radius*radius), 2*dy / (radius*radius)])

def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.zeros_like(v), 0.0
    return v / n, n

def ds_velocity(pt):
    """Original DS velocity f(pt). In saddle mode, we align target to create collinearity."""
    base = K * (target - pt)
    if local_min_on:
        # apply localized sink around local_min_point (on boundary)
        d = np.linalg.norm(pt - local_min_point)
        if d < LOCAL_MIN_RADIUS:
            sink = -LOCAL_MIN_GAIN * (pt - local_min_point) * math.exp(- (d*d) / (LOCAL_MIN_RADIUS**2))
        else:
            sink = np.zeros(2)
        return base + sink
    else:
        return base

def modulation_matrix(pt, v, rho=RHO, tail_effect=True, center=obs_center, radius=obs_radius):
    """Return M, grad, extras=(n_unit, e_unit, lam1, lam2), gamma"""
    G = Gamma(pt, center, radius)
    grad = gradGamma(pt, center, radius)
    gnorm = np.linalg.norm(grad)
    if gnorm < 1e-9:
        return np.eye(2), grad, None, G
    n_unit, _ = normalize(grad)
    e_unit = np.array([-n_unit[1], n_unit[0]])
    gamma_abs = abs(G) if abs(G) > 1e-9 else 1e-9
    pow_term = gamma_abs ** (1.0 / max(1e-6, rho))
    lam1 = 1.0 - 1.0 / (pow_term)
    lam2 = 1.0 + 1.0 / (pow_term)
    if (not tail_effect) and np.dot(n_unit, v) >= 0:
        lam1 = 1.0
    D = np.diag([lam1, lam2])
    E = np.column_stack((n_unit, e_unit))
    try:
        E_inv = np.linalg.inv(E)
    except np.linalg.LinAlgError:
        return np.eye(2), grad, (n_unit, e_unit, lam1, lam2), G
    M = E @ D @ E_inv
    return M, grad, (n_unit, e_unit, lam1, lam2), G

def is_on_boundary(pt, tol=EPS_BOUND):
    return abs(Gamma(pt) - 1.0) <= tol

def is_stuck(pt, v_mod):
    return is_on_boundary(pt) and (np.linalg.norm(v_mod) < STUCK_VEL_THRESH)

def draw_arrow(surf, start, vec, color, scale=1.0, width=3):
    if np.linalg.norm(vec) < 1e-9:
        return
    start = (int(start[0]), int(start[1]))
    end = (int(start[0] + vec[0]*scale), int(start[1] + vec[1]*scale))
    pygame.draw.line(surf, color, start, end, width)
    ang = math.atan2(-vec[1], vec[0])
    ah = 6
    p1 = (end[0] - ah * math.cos(ang - 0.3), end[1] + ah * math.sin(ang - 0.3))
    p2 = (end[0] - ah * math.cos(ang + 0.3), end[1] + ah * math.sin(ang + 0.3))
    pygame.draw.polygon(surf, color, [end, p1, p2])

def reset_agent():
    global agent
    agent[:] = agent_start.copy()

# ---------- Interaction helpers ----------
def near(pt, pos, rad=12):
    return np.linalg.norm(pt - pos) <= rad

# ---------- Main loop ----------
running = True
status_msg = "Flowing normally"
auto_nudging = algo_auto

while running:
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = False

        elif ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                running = False
            elif ev.key == pygame.K_SPACE:
                algo_auto = not algo_auto
            elif ev.key == pygame.K_n:
                step_request = True
            elif ev.key == pygame.K_l:
                local_min_on = not local_min_on
                # recompute local_min_point on current obstacle
                local_min_point = np.array([
                    obs_center[0] + obs_radius * math.cos(local_min_angle),
                    obs_center[1] + obs_radius * math.sin(local_min_angle)
                ])
            elif ev.key == pygame.K_s:
                saddle_on = not saddle_on
                # if saddle on, align target on opposite side of obstacle relative to agent
                if saddle_on:
                    # put target at opposite radial direction from agent through obstacle center
                    rel = agent - obs_center
                    if np.linalg.norm(rel) < 1e-6:
                        rel = np.array([1.0, 0.0])
                    unit = rel / np.linalg.norm(rel)
                    target[:] = obs_center - unit * (obs_radius + 200.0)
                else:
                    # restore default target if toggled off (push away a bit)
                    target[:] = np.array([obs_center[0] + 300.0, obs_center[1] - 60.0])
            elif ev.key == pygame.K_UP:
                ALPHA = min(0.6, ALPHA + 0.01)
            elif ev.key == pygame.K_DOWN:
                ALPHA = max(0.01, ALPHA - 0.01)
            elif ev.key == pygame.K_r:
                reset_agent()
            elif ev.key == pygame.K_v:
                show_vectors = not show_vectors

        elif ev.type == pygame.MOUSEBUTTONDOWN:
            mx, my = ev.pos
            mp = np.array([mx, my], dtype=float)
            if ev.button == 1:  # left click — choose drag agent or target
                if near(mp, agent, rad=12):
                    dragging_agent = True
                    dragging_offset = agent - mp
                elif near(mp, target, rad=12):
                    dragging_target = True
                    dragging_offset = target - mp
                else:
                    # left click elsewhere: set target
                    target[:] = mp
            elif ev.button == 2 or ev.button == 3:  # middle or right click — reposition obstacle center
                obs_center[:] = mp
                # update local_min point on boundary if enabled
                local_min_point = np.array([
                    obs_center[0] + obs_radius * math.cos(local_min_angle),
                    obs_center[1] + obs_radius * math.sin(local_min_angle)
                ])
            elif ev.button == 4:  # wheel up -> increase radius
                obs_radius = min(400.0, obs_radius + 6.0)
                local_min_point = np.array([
                    obs_center[0] + obs_radius * math.cos(local_min_angle),
                    obs_center[1] + obs_radius * math.sin(local_min_angle)
                ])
            elif ev.button == 5:  # wheel down -> decrease radius
                obs_radius = max(30.0, obs_radius - 6.0)
                local_min_point = np.array([
                    obs_center[0] + obs_radius * math.cos(local_min_angle),
                    obs_center[1] + obs_radius * math.sin(local_min_angle)
                ])

        elif ev.type == pygame.MOUSEBUTTONUP:
            if ev.button == 1:
                dragging_agent = False
                dragging_target = False

        elif ev.type == pygame.MOUSEMOTION:
            mx, my = ev.pos
            mp = np.array([mx, my], dtype=float)
            if dragging_agent:
                agent[:] = mp + dragging_offset
            if dragging_target:
                target[:] = mp + dragging_offset

    # ---------- compute DS and modulation ----------
    # optionally, if saddle_on, align target to make collinearity likely
    if saddle_on:
        rel = agent - obs_center
        if np.linalg.norm(rel) < 1e-6:
            rel = np.array([1.0, 0.0])
        unit = rel / np.linalg.norm(rel)
        target[:] = obs_center - unit * (obs_radius + 220.0)

    f = ds_velocity(agent)
    M_agent, grad_agent, extras, G = modulation_matrix(agent, f, rho=RHO)
    v_mod = M_agent @ f

    stuck_flag = is_stuck(agent, v_mod)

    # ---------- Algorithm-1 logic ----------
    # If stuck and algo_auto -> auto-nudge until free
    if stuck_flag and algo_auto:
        # auto nudge along tangent
        if extras is not None:
            n_unit, e_unit, lam1, lam2 = extras
            agent[:] = agent + ALPHA * e_unit * DT * 60.0
            # recompute v_mod after move (will be evaluated next loop)
    elif stuck_flag and (not algo_auto):
        # manual mode: wait for N key to step
        if step_request:
            if extras is not None:
                n_unit, e_unit, lam1, lam2 = extras
                # single nudging step
                agent[:] = agent + ALPHA * e_unit * DT * 60.0
            step_request = False

    else:
        # not stuck: integrate normally
        agent[:] = agent + v_mod * DT

    # ---------- keep local_min_point if local_min_on ----------
    if local_min_on:
        local_min_point = np.array([
            obs_center[0] + obs_radius * math.cos(local_min_angle),
            obs_center[1] + obs_radius * math.sin(local_min_angle)
        ])

    # ---------- DRAW ----------
    screen.fill(WHITE)

    # obstacle fill (light) and boundary
    pygame.draw.circle(screen, (235,245,235), (int(obs_center[0]), int(obs_center[1])), int(obs_radius))
    pygame.draw.circle(screen, GREEN, (int(obs_center[0]), int(obs_center[1])), int(obs_radius), 3)

    # local-min marker
    if local_min_on:
        pygame.draw.circle(screen, (180,30,30), (int(local_min_point[0]), int(local_min_point[1])), 8)
        pygame.draw.circle(screen, (180,30,30), (int(local_min_point[0]), int(local_min_point[1])), 12, 2)

    # target
    pygame.draw.circle(screen, DARK, (int(target[0]), int(target[1])), 6)
    pygame.draw.line(screen, DARK, (int(target[0]-8), int(target[1])), (int(target[0]+8), int(target[1])), 2)
    pygame.draw.line(screen, DARK, (int(target[0]), int(target[1]-8)), (int(target[0]), int(target[1]+8)), 2)

    # agent trail small dot
    pygame.draw.circle(screen, ORANGE, (int(agent[0]), int(agent[1])), 8)
    pygame.draw.circle(screen, BLUE, (int(agent[0]), int(agent[1])), 5)

    # draw original & modulated velocity arrows and normal/tangent if available
    if extras is not None:
        n_unit, e_unit, lam1, lam2 = extras
        # normal (cyan) and tangent (yellow)
        if show_vectors:
            draw_arrow(screen, agent, n_unit*60.0, CYAN, scale=1.0, width=4)
            draw_arrow(screen, agent, e_unit*60.0, YELLOW, scale=1.0, width=4)
        # eigenvalue text
        txt = font.render(f"λ_n={lam1:.3f}  λ_t={lam2:.3f}", True, DARK)
        screen.blit(txt, (12, 12))

    # draw original & modulated arrows
    if show_vectors:
        draw_arrow(screen, agent, f, ORANGE, scale=0.04, width=3)
        draw_arrow(screen, agent, v_mod, BLUE, scale=0.12, width=3)

    # status text
    mode_text = "LOCAL_MIN" if local_min_on else "NORMAL"
    mode_text = "SADDLE+" + (" (ON)" if saddle_on else " (OFF)") + "  " + mode_text
    algo_text = "AUTO" if algo_auto else "MANUAL"
    stuck_text = "STUCK" if stuck_flag else "FREE"
    info1 = font.render(f"Mode: {mode_text}   Algorithm-1: {algo_text}   Status: {stuck_text}", True, BLACK)
    screen.blit(info1, (12, H - 110))
    info2 = font.render(f"ALPHA: {ALPHA:.3f}   Obstacle center: ({int(obs_center[0])},{int(obs_center[1])})   R: {int(obs_radius)}", True, BLACK)
    screen.blit(info2, (12, H - 82))
    info3 = font.render("Left-drag agent / target | Middle/right-click reposition obstacle | Wheel radius | L toggle local-min | S toggle saddle", True, BLACK)
    screen.blit(info3, (12, H - 54))
    info4 = font.render("SPACE toggle AUTO | N single-step Algorithm-1 (manual) | UP/DOWN adjust ALPHA | R reset agent", True, BLACK)
    screen.blit(info4, (12, H - 30))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
sys.exit()
