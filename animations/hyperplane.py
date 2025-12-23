"""
ds_obstacle_demo.py

Pygame demo visualizing 2D dynamical-system modulation around a convex obstacle.
- Left-click and drag mouse to set target. Agent will attempt to reach it.
- Red dashed trail: original DS (no avoidance)
- Blue solid trail: modulated DS (with obstacle avoidance)
- Green ellipse: obstacle (Γ = 1)
- Cyan normal and yellow tangent vectors shown at agent projected point
Controls:
  - UP / DOWN : increase / decrease reactivity ρ
  - RIGHT / LEFT : increase / decrease safety factor η
  - T : toggle tail-effect (on/off)
  - R : reset trails / agent positions
  - ESC or window close : quit
"""
import pygame
import sys
import math
import numpy as np

# -------- PARAMETERS --------
WIDTH, HEIGHT = 1000, 700
FPS = 60

# DS gain
K = 2.0

# Obstacle ellipse half-axes (in pixels)
AX = 140.0
AY = 90.0

# Obstacle center
OX, OY = WIDTH // 2 + 120, HEIGHT // 2 - 20

# Initial agent positions (modulated and original)
start_orig = np.array([100.0, HEIGHT - 120.0])
start_mod = np.array([100.0, HEIGHT - 200.0])

# Integration timestep
DT = 1.0 / FPS

# Visual scale for vectors
V_SCALE = 0.12

# Colors
WHITE = (245, 245, 245)
BLACK = (10, 10, 10)
RED = (220, 50, 50)
BLUE = (40, 120, 220)
GREEN = (60, 200, 120)
CYAN = (100, 220, 220)
YELLOW = (240, 220, 80)
GREY = (180, 180, 180)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont("Consolas", 16)

def Gamma(pt, ox=OX, oy=OY, ax=AX, ay=AY, eta=1.0):
    # scaled coordinate in obstacle frame: xi = (x-ox)/ (ax * eta)
    x = (pt[0] - ox) / (ax * eta)
    y = (pt[1] - oy) / (ay * eta)
    return x * x + y * y

def gradGamma(pt, ox=OX, oy=OY, ax=AX, ay=AY, eta=1.0):
    # gradient of Gamma wrt world coordinates
    # Gamma = ((x-ox)/(ax*eta))^2 + ((y-oy)/(ay*eta))^2
    # dGamma/dx = 2*(x-ox) / (ax^2 * eta^2)
    x = pt[0] - ox
    y = pt[1] - oy
    denom_x = (ax * eta) ** 2
    denom_y = (ay * eta) ** 2
    return np.array([2.0 * x / denom_x, 2.0 * y / denom_y])

def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.zeros_like(v), 0.0
    return v / n, n

def build_modulation_matrix(pt, v, eta=1.0, rho=1.0, tail_effect=True):
    """
    Return modulation matrix M (2x2) computed at point pt (world coords).
    v is current velocity (used for tail-effect sign).
    """
    G = Gamma(pt, eta=eta)
    grad = gradGamma(pt, eta=eta)
    # If gradient is zero (at obstacle center), fallback to identity
    gnorm = np.linalg.norm(grad)
    if gnorm < 1e-9:
        return np.eye(2), grad, None, G

    # normal (not necessarily unit)
    n_raw = grad
    n_unit, n_len = normalize(n_raw)

    # tangent vector (perp)
    e = np.array([-n_unit[1], n_unit[0]])  # 90 deg rotation (perp)

    # Build E matrix [n e] -- columns
    E = np.column_stack((n_unit, e))  # 2x2

    # compute eigenvalues (scales) based on paper formula
    # avoid division by zero; use |Gamma|
    # For outside region Gamma >= 1 we operate; but function also defined everywhere
    gamma_abs = abs(G) if abs(G) > 1e-9 else 1e-9

    # eigenvalue exponent scaling: lambda1 = 1 - 1/|Gamma|^{1/rho}; lambda2 = 1 + 1/|Gamma|^{1/rho}
    pow_term = gamma_abs ** (1.0 / max(rho, 1e-9))
    lam1 = 1.0 - 1.0 / pow_term
    lam2 = 1.0 + 1.0 / pow_term

    # tail effect: if turning away (n^T v >= 0) and tail_effect False then set lam1 = 1
    if (not tail_effect) and np.dot(n_unit, v) >= 0:
        lam1 = 1.0

    D = np.diag([lam1, lam2])

    # Build M = E D E^{-1}
    try:
        E_inv = np.linalg.inv(E)
    except np.linalg.LinAlgError:
        # numerical issue: fallback to identity
        return np.eye(2), grad, (n_unit, e, lam1, lam2), G

    M = E @ D @ E_inv
    return M, grad, (n_unit, e, lam1, lam2), G

def draw_text(surf, text, x, y, color=BLACK):
    surf.blit(font.render(text, True, color), (x, y))

def main():
    running = True

    # agent states
    pos_orig = start_orig.copy()
    pos_mod = start_mod.copy()

    traj_orig = []
    traj_mod = []

    # target initial
    target = np.array([WIDTH - 120.0, 140.0])

    # parameters
    eta = 1.0   # safety factor
    rho = 1.0   # reactivity
    tail_effect = True

    dragging = False

    info_y = 10

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_t:
                    tail_effect = not tail_effect
                elif event.key == pygame.K_r:
                    # reset
                    pos_orig[:] = start_orig
                    pos_mod[:] = start_mod
                    traj_orig.clear()
                    traj_mod.clear()
                elif event.key == pygame.K_UP:
                    rho = min(10.0, rho + 0.2)
                elif event.key == pygame.K_DOWN:
                    rho = max(0.2, rho - 0.2)
                elif event.key == pygame.K_RIGHT:
                    eta = min(3.0, eta + 0.05)
                elif event.key == pygame.K_LEFT:
                    eta = max(1.0, eta - 0.05)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    dragging = True
                    mx, my = pygame.mouse.get_pos()
                    target[:] = [mx, my]
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    mx, my = pygame.mouse.get_pos()
                    target[:] = [mx, my]

        # compute original DS velocity (attractor)
        v_orig = K * (target - pos_orig)

        # integrate original
        pos_orig += v_orig * DT
        traj_orig.append(tuple(pos_orig.copy()))
        if len(traj_orig) > 1000:
            traj_orig.pop(0)

        # for modulated, compute the attractor DS at pos_mod as well
        v_target = K * (target - pos_mod)

        # compute modulation matrix at current pos_mod
        M, grad, extras, G = build_modulation_matrix(pos_mod, v_target, eta=eta, rho=rho, tail_effect=tail_effect)

        # apply modulation
        v_mod = M @ v_target

        # integrate modulated
        pos_mod += v_mod * DT
        traj_mod.append(tuple(pos_mod.copy()))
        if len(traj_mod) > 1000:
            traj_mod.pop(0)

        # Drawing
        screen.fill(WHITE)

        # draw obstacle ellipse (inflated by eta)
        rect = pygame.Rect(0, 0, 2 * AX * eta, 2 * AY * eta)
        rect.center = (OX, OY)
        pygame.draw.ellipse(screen, GREY, rect, width=0)  # filled light background
        pygame.draw.ellipse(screen, GREEN, rect, width=3)  # boundary

        # draw original trail (dashed red)
        if len(traj_orig) > 2:
            pts = [(int(x), int(y)) for x, y in traj_orig]
            for i in range(1, len(pts)):
                if i % 4 < 2:
                    pygame.draw.line(screen, RED, pts[i - 1], pts[i], 2)

        # draw modulated trail (solid blue)
        if len(traj_mod) > 2:
            pts = [(int(x), int(y)) for x, y in traj_mod]
            pygame.draw.lines(screen, BLUE, False, pts, 3)

        # draw agent positions
        pygame.draw.circle(screen, RED, (int(pos_orig[0]), int(pos_orig[1])), 8)
        pygame.draw.circle(screen, BLUE, (int(pos_mod[0]), int(pos_mod[1])), 8)

        # draw target
        pygame.draw.circle(screen, BLACK, (int(target[0]), int(target[1])), 6)
        pygame.draw.line(screen, BLACK, (int(target[0]) - 10, int(target[1])), (int(target[0]) + 10, int(target[1])), 2)
        pygame.draw.line(screen, BLACK, (int(target[0]), int(target[1]) - 10), (int(target[0]), int(target[1]) + 10), 2)

        # draw vectors at mod agent: original and modulated
        # original velocity (at agent pos)
        v_target_display = v_target * V_SCALE
        v_mod_display = v_mod * V_SCALE

        o = pos_mod.astype(int)
        tail = (int(o[0] - v_target_display[0]), int(o[1] - v_target_display[1]))
        pygame.draw.line(screen, RED, (int(o[0]), int(o[1])), tail, 2)
        pygame.draw.polygon(screen, RED, [tail,
                                         (tail[0] + int(6 * math.cos(math.atan2(-v_target_display[1], -v_target_display[0]) - 0.3)),
                                          tail[1] + int(6 * math.sin(math.atan2(-v_target_display[1], -v_target_display[0]) - 0.3))),
                                         (tail[0] + int(6 * math.cos(math.atan2(-v_target_display[1], -v_target_display[0]) + 0.3)),
                                          tail[1] + int(6 * math.sin(math.atan2(-v_target_display[1], -v_target_display[0]) + 0.3)))], 0)

        tail2 = (int(o[0] - v_mod_display[0]), int(o[1] - v_mod_display[1]))
        pygame.draw.line(screen, BLUE, (int(o[0]), int(o[1])), tail2, 2)
        pygame.draw.polygon(screen, BLUE, [tail2,
                                         (tail2[0] + int(6 * math.cos(math.atan2(-v_mod_display[1], -v_mod_display[0]) - 0.3)),
                                          tail2[1] + int(6 * math.sin(math.atan2(-v_mod_display[1], -v_mod_display[0]) - 0.3))),
                                         (tail2[0] + int(6 * math.cos(math.atan2(-v_mod_display[1], -v_mod_display[0]) + 0.3)),
                                          tail2[1] + int(6 * math.sin(math.atan2(-v_mod_display[1], -v_mod_display[0]) + 0.3)))], 0)

        # draw normal & tangent if grad exists
        if extras is not None:
            n_unit, e_unit, lam1, lam2 = extras
            if n_unit is not None:
                # show normal (cyan) and tangent (yellow) scaled
                n_show = n_unit * 60.0
                e_show = e_unit * 60.0
                p1 = (int(pos_mod[0]), int(pos_mod[1]))
                pn = (int(pos_mod[0] + n_show[0]), int(pos_mod[1] + n_show[1]))
                pe = (int(pos_mod[0] + e_show[0]), int(pos_mod[1] + e_show[1]))
                pygame.draw.line(screen, CYAN, p1, pn, 3)
                pygame.draw.line(screen, YELLOW, p1, pe, 3)

                # small text about eigenvalues
                txt = f"λ_normal={lam1:.3f}  λ_tangent={lam2:.3f}"
                draw_text(screen, txt, 10, info_y + 40)

        # HUD
        draw_text(screen, f"Controls: LEFT/RIGHT safety η  UP/DOWN reactivity ρ  T toggle tail-effect  R reset", 10, info_y)
        draw_text(screen, f"η (safety) = {eta:.2f}", 10, info_y + 20)
        draw_text(screen, f"ρ (reactivity) = {rho:.2f}", 200, info_y + 20)
        draw_text(screen, f"Tail-effect = {'ON' if tail_effect else 'OFF'}", 420, info_y + 20)
        draw_text(screen, f"Gamma at mod-agent = {G:.4f}", 640, info_y + 20)
        draw_text(screen, f"Original pos = ({pos_orig[0]:.1f},{pos_orig[1]:.1f})", 10, info_y + 60)
        draw_text(screen, f"Mod pos = ({pos_mod[0]:.1f},{pos_mod[1]:.1f})", 300, info_y + 60)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
