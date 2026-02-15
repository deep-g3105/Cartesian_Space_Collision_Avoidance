# ds_2R_visualizer.py
# 2R planar manipulator DS visualizer (Pygame)
# Manual joint-control version (robot does NOT auto-animate)

import pygame
import sys
import math
import numpy as np

# ---------- Config ----------
WIDTH, HEIGHT = 1100, 700
FPS = 60

LINK1 = 180.0
LINK2 = 140.0

BG = (18, 18, 20)
GRID = (30, 30, 36)
ROBOT_COL = (200, 200, 240)
GOAL_COL = (120, 240, 140)
OBST_COL = (220, 90, 90)
ARROW_F = (100, 170, 255)       # nominal f
ARROW_MF = (255, 180, 80)      # M f
ARROW_TAN = (120, 255, 130)    # alpha M g
ARROW_REP = (255, 120, 120)    # repulsion
ARROW_TOTAL = (230, 230, 255)  # total

def clamp(a, lo, hi): return max(lo, min(hi, a))

def normvec(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v, 0.0
    return v / n, n

# ---------- Robot kinematics ----------
def fk(q):
    q1, q2 = q
    p0 = np.array([300.0, 350.0])
    p1 = p0 + np.array([LINK1*math.cos(q1), LINK1*math.sin(q1)])
    p2 = p1 + np.array([LINK2*math.cos(q1+q2), LINK2*math.sin(q1+q2)])
    return [p0, p1, p2], p2

def jacobian(q):
    q1, q2 = q
    J = np.zeros((2,2))
    J[0,0] = -LINK1*math.sin(q1) - LINK2*math.sin(q1+q2)
    J[1,0] =  LINK1*math.cos(q1) + LINK2*math.cos(q1+q2)
    J[0,1] = -LINK2*math.sin(q1+q2)
    J[1,1] =  LINK2*math.cos(q1+q2)
    return J

def damped_pinv(J, lam=1e-2):
    JJ = J @ J.T
    inv = np.linalg.inv(JJ + lam*np.eye(2))
    return J.T @ inv

# ---------- Nominal DS ----------
def nominal_ds(q, x_goal, kappa=1.0):
    _, x = fk(q)
    err = x_goal - x
    v = kappa * err
    J = jacobian(q)
    Jpinv = damped_pinv(J, lam=1e-3)
    qdot_nom = Jpinv @ v
    return qdot_nom, v

# ---------- Gamma and gradient ----------
def gamma_and_grad(q, obst_center, obst_radius):
    _, x = fk(q)
    vec = x - obst_center
    nx, dist = normvec(vec)
    Gamma = dist - obst_radius
    J = jacobian(q)
    grad_q = J.T @ nx
    return float(Gamma), grad_q, np.array(nx)

def sigma(Gamma, d1, d2, lam1, lam2, k):
    mid = 0.5 * (d1 + d2)
    return lam1 + (lam2 - lam1) / (1.0 + math.exp(-k*(Gamma - mid)))

# ---------- Modulation ----------
def modulation_matrix(q, obst_center, obst_radius, params):
    Gamma, grad_q, _ = gamma_and_grad(q, obst_center, obst_radius)

    n_norm = np.linalg.norm(grad_q)
    if n_norm < 1e-9:
        n_unit = np.array([1.0, 0.0])
    else:
        n_unit = grad_q / n_norm

    lam_n = sigma(Gamma, params['d1'], params['d2'],
                  params['lam1_n'], params['lam2_n'], params['k'])

    lam_t = sigma(Gamma, params['d1'], params['d2'],
                  params['lam1_t'], params['lam2_t'], params['k'])

    f_q, _ = nominal_ds(q, params['xg'], params['kappa'])
    f_unit, _ = normvec(f_q)
    dot_nf = np.dot(n_unit, f_unit)
    lam_v = 1.0 / (1.0 + math.exp(-40.0 * dot_nf))

    lam_n_prime = lam_v + (1.0 - lam_v) * lam_n

    n = n_unit
    t = np.array([-n[1], n[0]])
    E = np.column_stack((n, t))
    D = np.diag([lam_n_prime, lam_t])
    M = E @ D @ E.T

    return M, float(Gamma), n_unit

# ---------- Kernels ----------
def navigation_g(q, kernels):
    g = np.zeros(2)
    for k in kernels:
        bq = np.array(k['bq'])
        bg = np.array(k['bg'])
        gamma = k['gamma']
        r2 = np.linalg.norm(q - bq)**2
        g += bg * math.exp(-gamma * r2)
    return g

# ---------- Activation ----------
def activation_alpha(q, Gamma, n_unit, f_q, params):
    lam_n = sigma(Gamma, params['d1'], params['d2'],
                  params['lam1_n'], params['lam2_n'], params['k'])
    alpha_G = max(0.0, 1.0 - lam_n)

    f_unit, _ = normvec(f_q)
    dot_nf = np.dot(n_unit, f_unit)
    lam_v = 1.0 / (1.0 + math.exp(-40.0 * dot_nf))
    alpha_n = max(0.0, 1.0 - lam_v)

    _, x = fk(q)
    dist_goal = np.linalg.norm(x - params['xg'])
    alpha_f = 1.0 if dist_goal > params['alpha_f_radius'] else 0.0

    if dist_goal <= params['alpha_off_radius']:
        return 0.0

    return float(clamp(alpha_G * alpha_n * alpha_f, 0.0, 1.0))

# ---------- Repulsion ----------
def frep(q, Gamma, n_unit, params):
    eps = params['eps']
    if Gamma < eps:
        return ((eps - Gamma) / params['dt']) * n_unit
    return np.zeros(2)

# ---------- Slider ----------
class Slider:
    def __init__(self, x, y, w, label, val, vmin, vmax):
        self.rect = pygame.Rect(x, y, w, 18)
        self.label = label
        self.vmin = vmin
        self.vmax = vmax
        self.val = float(val)
        self.handle_w = 12
        self.drag = False

    def draw(self, surf, font):
        pygame.draw.rect(surf, (60, 60, 66), self.rect)
        t = font.render(f"{self.label}: {self.val:.3f}", True, (220,220,220))
        surf.blit(t, (self.rect.x, self.rect.y - 18))
        frac = clamp((self.val - self.vmin) / (self.vmax - self.vmin), 0.0, 1.0)
        hx = int(self.rect.x + frac * (self.rect.w - self.handle_w))
        handle = pygame.Rect(hx, self.rect.y-1, self.handle_w, self.rect.h+2)
        pygame.draw.rect(surf, (200,200,200), handle)

    def handle_event(self, e):
        if e.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(e.pos):
            self.drag = True
        if e.type == pygame.MOUSEBUTTONUP:
            self.drag = False
        if e.type == pygame.MOUSEMOTION and self.drag:
            mx = e.pos[0]
            frac = clamp((mx - self.rect.x) / (self.rect.w - self.handle_w), 0.0, 1.0)
            self.val = self.vmin + frac*(self.vmax - self.vmin)

# ---------- Draw arrows ----------
def draw_arrow(screen, start, vec, color, head=8, width=3):
    end = start + vec
    pygame.draw.line(screen, color, tuple(start.astype(int)),
                     tuple(end.astype(int)), width)
    ang = math.atan2(vec[1], vec[0])
    p1 = end - np.array([math.cos(ang-0.2), math.sin(ang-0.2)])*head
    p2 = end - np.array([math.cos(ang+0.2), math.sin(ang+0.2)])*head
    pygame.draw.polygon(screen, color, [tuple(end.astype(int)),
                                        tuple(p1.astype(int)),
                                        tuple(p2.astype(int))])

# ---------- Main loop ----------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 14)

    params = {
        'kappa': 1.2,
        'd1': 0.02, 'd2': 0.12,
        'lam1_n': 0.02, 'lam2_n': 1.0,
        'lam1_t': 1.5, 'lam2_t': 1.0,
        'k': 8.0,
        'alpha_f_radius': 30.0,
        'alpha_off_radius': 20.0,
        'eps': 12.0,
        'dt': 0.02,
    }

    # Manual robot pose (controlled only by sliders)
    q = np.array([math.radians(-40), math.radians(70)])

    x_goal = np.array([650.0, 300.0])
    obst_center = np.array([460.0, 340.0])
    obst_radius = 70.0

    kernels = [
        {'bq': np.array([math.radians(-30), math.radians(50)]),
         'bg': np.array([0.4, -0.2]),
         'gamma': 60.0},
    ]

    dragging_goal = False
    dragging_obst = False

    # sliders
    s_kappa = Slider(20,20,260,'kappa',params['kappa'],0,4)
    s_d1 = Slider(20,70,260,'d1',params['d1'],0,0.2)
    s_d2 = Slider(20,110,260,'d2',params['d2'],0.02,0.5)
    s_k = Slider(20,150,260,'sig_k',params['k'],1,30)
    s_lam1t = Slider(20,190,260,'lam1_t',params['lam1_t'],0.3,4)
    s_eps = Slider(20,230,260,'eps',params['eps'],1,40)
    s_dt = Slider(20,270,260,'dt',params['dt'],0.002,0.05)
    s_gamma = Slider(20,310,260,'kernel_gamma',kernels[0]['gamma'],1,200)
    s_bg_angle = Slider(20,350,260,'kernel_angle_deg',270,0,360)
    s_bg_mag = Slider(20,390,260,'kernel_mag',0.4,0,2)
    s_q1 = Slider(20,430,260,'q1_deg',math.degrees(q[0]),-180,180)
    s_q2 = Slider(20,470,260,'q2_deg',math.degrees(q[1]),-180,180)
    s_obst_r = Slider(20,510,260,'obstacle_radius',obst_radius,10,200)

    sliders = [s_kappa,s_d1,s_d2,s_k,s_lam1t,s_eps,s_dt,
               s_gamma,s_bg_angle,s_bg_mag,s_q1,s_q2,s_obst_r]

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            # dragging goal or obstacle
            if e.type == pygame.MOUSEBUTTONDOWN:
                mx,my = e.pos
                if np.linalg.norm(np.array([mx,my]) - x_goal) < 15:
                    dragging_goal = True
                elif np.linalg.norm(np.array([mx,my]) - obst_center) < 15:
                    dragging_obst = True

            if e.type == pygame.MOUSEBUTTONUP:
                dragging_goal = False
                dragging_obst = False
                for s in sliders: s.drag = False

            if e.type == pygame.MOUSEMOTION:
                if dragging_goal:
                    x_goal = np.array(e.pos, dtype=float)
                if dragging_obst:
                    obst_center = np.array(e.pos, dtype=float)

            # sliders
            for s in sliders:
                s.handle_event(e)

        # update parameters
        params['kappa'] = s_kappa.val
        params['d1'] = s_d1.val
        params['d2'] = s_d2.val
        params['k'] = s_k.val
        params['lam1_t'] = s_lam1t.val
        params['eps'] = s_eps.val
        params['dt'] = s_dt.val
        kernels[0]['gamma'] = s_gamma.val
        kernels[0]['bg'] = np.array([
            math.cos(math.radians(s_bg_angle.val))*s_bg_mag.val,
            math.sin(math.radians(s_bg_angle.val))*s_bg_mag.val
        ])

        # obstacle radius slider
        obst_radius = s_obst_r.val

        # manual joint positions
        q = np.array([math.radians(s_q1.val), math.radians(s_q2.val)])

        params['xg'] = x_goal

        # compute DS fields (NO integration)
        f_q, v_task = nominal_ds(q, x_goal, params['kappa'])
        M, Gamma, n_unit = modulation_matrix(q, obst_center, obst_radius, params)
        g_q = navigation_g(q, kernels)
        alpha = activation_alpha(q, Gamma, n_unit, f_q, params)
        tan_term = alpha*(M @ g_q)
        mod_term = M @ f_q
        rep = frep(q, Gamma, n_unit, params)
        total = mod_term + tan_term + rep

        # ---- DRAWING ----
        screen.fill(BG)

        # grid
        for x in range(0,WIDTH,40):
            pygame.draw.line(screen, GRID, (x,0),(x,HEIGHT))
        for y in range(0,HEIGHT,40):
            pygame.draw.line(screen, GRID, (0,y),(WIDTH,y))

        # obstacle
        pygame.draw.circle(screen,(40,40,40),obst_center.astype(int),int(obst_radius+10))
        pygame.draw.circle(screen,OBST_COL,obst_center.astype(int),int(obst_radius))

        # goal
        pygame.draw.circle(screen,GOAL_COL,tuple(x_goal.astype(int)),8)

        # robot links
        joints, p2 = fk(q)
        p0,p1,p2 = joints
        pygame.draw.line(screen,ROBOT_COL,tuple(p0.astype(int)),tuple(p1.astype(int)),8)
        pygame.draw.line(screen,ROBOT_COL,tuple(p1.astype(int)),tuple(p2.astype(int)),8)
        pygame.draw.circle(screen,(120,120,200),tuple(p0.astype(int)),8)
        pygame.draw.circle(screen,(160,160,220),tuple(p1.astype(int)),6)
        pygame.draw.circle(screen,(200,200,255),tuple(p2.astype(int)),6)

        # vector arrows (mapped through Jacobian)
        J = jacobian(q)
        def draw_term(vec_q, col):
            v = J @ vec_q
            draw_arrow(screen, p2, v*0.6, col)

        draw_term(f_q, ARROW_F)
        draw_term(mod_term, ARROW_MF)
        draw_term(tan_term, ARROW_TAN)
        draw_term(rep, ARROW_REP)
        draw_term(total, ARROW_TOTAL)

        # text
        hud = [
            f"Gamma={Gamma:.3f}   alpha={alpha:.3f}",
            f"||f||={np.linalg.norm(f_q):.3f}",
            f"||Mf||={np.linalg.norm(mod_term):.3f}",
            f"||alphaMg||={np.linalg.norm(tan_term):.3f}",
            f"||rep||={np.linalg.norm(rep):.3f}",
            f"||total||={np.linalg.norm(total):.3f}",
            "Robot moves ONLY via q1,q2 sliders.",
            "Drag obstacle (red) or goal (green)."
        ]
        for i,t in enumerate(hud):
            screen.blit(font.render(t,True,(220,220,220)),(20,560+i*20))

        # sliders
        for s in sliders: s.draw(screen,font)

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()
