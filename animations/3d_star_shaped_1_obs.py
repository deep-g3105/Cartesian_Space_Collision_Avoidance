import pygame
import numpy as np
from modulation.class_working import ModulationAvoider

# -------------------------------------------------------
# Setup
# -------------------------------------------------------
WIDTH, HEIGHT = 1200, 800
WHITE = (255, 255, 255)
BLUE = (50, 100, 255)
RED = (255, 80, 80)
GREEN = (50, 200, 50)
BLACK = (0, 0, 0)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Interactive 3D Star-Shaped Obstacle Avoidance")
clock = pygame.time.Clock()

CENTER = np.array([WIDTH // 2, HEIGHT // 2])
CAMERA_DIST = 900.0
yaw, pitch = np.pi/4, np.pi/4

# -------------------------------------------------------
# Camera
# -------------------------------------------------------
def get_camera_matrix():
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)

    Ry = np.array([[cy, 0, sy],
                   [0, 1, 0],
                   [-sy, 0, cy]])

    Rx = np.array([[1, 0, 0],
                   [0, cp, -sp],
                   [0, sp, cp]])

    return Rx @ Ry

def project(point3d):
    R = get_camera_matrix()
    rotated = R @ point3d
    x, y, z = rotated
    factor = CAMERA_DIST / (CAMERA_DIST + z)
    px = x * factor + CENTER[0]
    py = -y * factor + CENTER[1]
    return np.array([px, py])

def screen_to_world(mouse_pos):
    x = mouse_pos[0] - CENTER[0]
    y = -(mouse_pos[1] - CENTER[1])
    return np.array([x, y, 0.0])

# -------------------------------------------------------
# Star Parameters
# -------------------------------------------------------
star_center = np.array([0.0, 0.0, 0.0])
R_base = 180.0
epsilon = 0.3
k = 5
m = 2   # controls vertical smoothness

# -------------------------------------------------------
# Star Gamma + Gradient
# -------------------------------------------------------
def gamma_star(x):
    d = x - star_center
    r = np.linalg.norm(d) + 1e-9

    theta = np.arctan2(d[1], d[0])
    phi = np.arccos(d[2]/r)

    r_theta_phi = R_base * (1 + epsilon * np.cos(k*theta) * np.sin(phi)**m)
    return r / r_theta_phi

def grad_gamma_star(x):
    d = x - star_center
    r = np.linalg.norm(d) + 1e-9
    theta = np.arctan2(d[1], d[0])
    phi = np.arccos(d[2]/r)

    r_tp = R_base * (1 + epsilon*np.cos(k*theta)*np.sin(phi)**m)

    # Approximate gradient numerically (stable + easy)
    eps = 1e-5
    grad = np.zeros(3)
    for i in range(3):
        dx = np.zeros(3)
        dx[i] = eps
        grad[i] = (gamma_star(x+dx) - gamma_star(x-dx))/(2*eps)

    return grad

avoid = ModulationAvoider(
    gamma=gamma_star,
    grad_gamma=grad_gamma_star,
    rho=1.0,
    safety_eta=1.0,
    tail_effect=False,
)

# -------------------------------------------------------
# Robot + Goal
# -------------------------------------------------------
robot_pos = np.array([-400.0, 0.0, 0.0])
goal_pos = np.array([400.0, 0.0, 0.0])
trail = []

def nominal_velocity(x, goal):
    v = goal - x
    n = np.linalg.norm(v)
    if n < 1e-6:
        return np.zeros(3)
    return v / n * 80.0

# -------------------------------------------------------
# Interaction Flags
# -------------------------------------------------------
dragging = False
left_mouse = False

# -------------------------------------------------------
# Main Loop
# -------------------------------------------------------
running = True
while running:
    dt = clock.tick(60)/1000.0
    screen.fill(WHITE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                left_mouse = True
            elif event.button == 3:
                dragging = True
            elif event.button == 4:
                star_center[2] += 20
            elif event.button == 5:
                star_center[2] -= 20

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                goal_pos = screen_to_world(pygame.mouse.get_pos())
            left_mouse = False
            dragging = False

        elif event.type == pygame.MOUSEMOTION:
            if dragging:
                star_center[:2] = screen_to_world(pygame.mouse.get_pos())[:2]

            if left_mouse:
                dx, dy = event.rel
                yaw += dx * 0.01
                pitch += dy * 0.01
                pitch = np.clip(pitch, -np.pi/2, np.pi/2)

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                epsilon += 0.05
            elif event.key == pygame.K_DOWN:
                epsilon = max(0.0, epsilon - 0.05)

    # Dynamics
    v_nom = nominal_velocity(robot_pos, goal_pos)
    v_mod = avoid.modulate(robot_pos, robot_pos, v_nom)
    robot_pos += v_mod * dt

    trail.append(robot_pos.copy())
    if len(trail) > 250:
        trail.pop(0)

    # -------------------------------------------------------
    # Draw Star Surface
    # -------------------------------------------------------
    lat_steps = 20
    lon_steps = 30

    for i in range(1, lat_steps):
        phi = np.pi * i / lat_steps
        circle = []
        for j in range(lon_steps+1):
            theta = 2*np.pi*j/lon_steps
            r = R_base * (1 + epsilon*np.cos(k*theta)*np.sin(phi)**m)
            x = r*np.sin(phi)*np.cos(theta)
            y = r*np.sin(phi)*np.sin(theta)
            z = r*np.cos(phi)
            circle.append(project(np.array([x,y,z])+star_center))
        pygame.draw.lines(screen, RED, False, circle, 1)

    # Trail
    for i in range(1,len(trail)):
        pygame.draw.line(screen,
                         (50,50,200),
                         project(trail[i-1]),
                         project(trail[i]),
                         2)

    pygame.draw.circle(screen, BLUE,
                       project(robot_pos).astype(int), 8)
    pygame.draw.circle(screen, GREEN,
                       project(goal_pos).astype(int), 10)

    pygame.display.flip()

pygame.quit()
