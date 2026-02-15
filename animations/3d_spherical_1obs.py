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
GREY = (180, 180, 180)
BLACK = (0, 0, 0)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Interactive 3D Spherical Obstacle Avoidance")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 18)

# -------------------------------------------------------
# Camera
# -------------------------------------------------------
CAMERA_DIST = 800.0
CENTER = np.array([WIDTH // 2, HEIGHT // 2])
yaw, pitch = np.pi / 4, np.pi / 4

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
    x = (mouse_pos[0] - CENTER[0])
    y = -(mouse_pos[1] - CENTER[1])
    return np.array([x, y, 0.0])

# -------------------------------------------------------
# Simulation Parameters
# -------------------------------------------------------
rho = 1.0
safety_eta = 1.0
show_vectors = True

sphere_center = np.array([0.0, 0.0, 0.0])
sphere_radius = 150.0

avoid = ModulationAvoider(
    gamma=lambda x: ModulationAvoider.gamma_sphere(x, sphere_center, sphere_radius),
    grad_gamma=lambda x: ModulationAvoider.grad_gamma_sphere(x, sphere_center, sphere_radius),
    rho=rho,
    safety_eta=safety_eta,
    tail_effect=False,
)

robot_pos = np.array([-400.0, 0.0, 0.0])
goal_pos = np.array([400.0, 0.0, 0.0])

trail = []
TRAIL_LENGTH = 200

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
def nominal_velocity(x, goal):
    v = goal - x
    n = np.linalg.norm(v)
    if n < 1e-6:
        return np.zeros(3)
    return v / n * 200.0

# -------------------------------------------------------
# Interaction Flags
# -------------------------------------------------------
dragging_sphere = False
rotating_camera = False
left_mouse_down = False
mouse_moved = False

# -------------------------------------------------------
# Main Loop
# -------------------------------------------------------
running = True
while running:
    dt = clock.tick(60) / 1000.0
    screen.fill(WHITE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # ---------------- Mouse Controls ----------------
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                left_mouse_down = True
                mouse_moved = False
                rotating_camera = True
            elif event.button == 3:
                dragging_sphere = True
            elif event.button == 4:
                sphere_center[2] += 20
            elif event.button == 5:
                sphere_center[2] -= 20

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                if not mouse_moved:
                    goal_pos = screen_to_world(pygame.mouse.get_pos())
            left_mouse_down = False
            rotating_camera = False
            dragging_sphere = False

        elif event.type == pygame.MOUSEMOTION:
            if dragging_sphere:
                sphere_center[:2] = screen_to_world(pygame.mouse.get_pos())[:2]

            if left_mouse_down:
                dx, dy = event.rel
                if abs(dx) > 1 or abs(dy) > 1:
                    mouse_moved = True
                yaw += dx * 0.01
                pitch += dy * 0.01
                pitch = np.clip(pitch, -np.pi/2, np.pi/2)

        # ---------------- Keyboard Controls ----------------
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                rho += 0.2
            elif event.key == pygame.K_DOWN:
                rho = max(0.1, rho - 0.2)
            elif event.key == pygame.K_SPACE:
                show_vectors = not show_vectors

    avoid.rho = rho

    # -------------------------------------------------------
    # 3D dynamics
    # -------------------------------------------------------
    v_nom = nominal_velocity(robot_pos, goal_pos)
    v_mod = avoid.modulate(robot_pos, robot_pos, v_nom)
    robot_pos += v_mod * dt

    trail.append(robot_pos.copy())
    if len(trail) > TRAIL_LENGTH:
        trail.pop(0)

    # -------------------------------------------------------
    # Draw meshed sphere
    # -------------------------------------------------------
    lat_steps = 18
    lon_steps = 24

    for i in range(1, lat_steps):
        phi = np.pi * i / lat_steps
        circle_points = []
        for j in range(lon_steps + 1):
            theta = 2 * np.pi * j / lon_steps
            x = sphere_radius * np.sin(phi) * np.cos(theta)
            y = sphere_radius * np.sin(phi) * np.sin(theta)
            z = sphere_radius * np.cos(phi)
            p = np.array([x, y, z]) + sphere_center
            circle_points.append(project(p))
        pygame.draw.lines(screen, RED, False, circle_points, 1)

    for j in range(lon_steps):
        theta = 2 * np.pi * j / lon_steps
        circle_points = []
        for i in range(lat_steps + 1):
            phi = np.pi * i / lat_steps
            x = sphere_radius * np.sin(phi) * np.cos(theta)
            y = sphere_radius * np.sin(phi) * np.sin(theta)
            z = sphere_radius * np.cos(phi)
            p = np.array([x, y, z]) + sphere_center
            circle_points.append(project(p))
        pygame.draw.lines(screen, RED, False, circle_points, 1)

    # -------------------------------------------------------
    # Vector field slice
    # -------------------------------------------------------
    if show_vectors:
        grid = 140
        for x in np.arange(-500, 500, grid):
            for y in np.arange(-300, 300, grid):
                p = np.array([x, y, 0.0])
                if avoid.gamma(p) < 1.0:
                    continue
                v_n = nominal_velocity(p, goal_pos)
                v_m = avoid.modulate(p, p, v_n)
                start = project(p)
                end = project(p + v_m * 0.3)
                pygame.draw.line(screen, GREY, start, end, 1)

    # -------------------------------------------------------
    # Draw fading robot trail
    # -------------------------------------------------------
    for i in range(1, len(trail)):
        p1 = project(trail[i - 1])
        p2 = project(trail[i])
        alpha_factor = i / len(trail)
        color_intensity = int(255 * alpha_factor)
        color = (color_intensity // 3, color_intensity // 3, color_intensity)
        pygame.draw.line(screen, color, p1, p2, 2)

    # Draw robot and goal
    pygame.draw.circle(screen, BLUE, project(robot_pos).astype(int), 8)
    pygame.draw.circle(screen, GREEN, project(goal_pos).astype(int), 10)

    # UI text
    texts = [
        f"rho: {rho:.2f}",
        f"safety_eta: {safety_eta}",
        "Left Click: Move Goal",
        "Left Drag: Rotate Camera",
        "Right Drag: Move Sphere",
        "Mouse Wheel: Move Sphere in Z",
        "UP/DOWN: Change rho | SPACE: Toggle Field",
    ]

    for i, t in enumerate(texts):
        screen.blit(font.render(t, True, BLACK), (10, 10 + 22 * i))

    pygame.display.flip()

pygame.quit()
