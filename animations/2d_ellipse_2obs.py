import pygame
import numpy as np
import math
from modulation.class_working import ModulationAvoider

# ---------------------------------------------------------------------
# Constants & Setup
# ---------------------------------------------------------------------
WIDTH, HEIGHT = 1080, 720
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (50, 100, 255)
RED = (255, 50, 50)
GREEN = (50, 200, 50)
GREY = (200, 200, 200)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("DS Modulation – Ellipsoidal Obstacles")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 18)

# ---------------------------------------------------------------------
# Simulation Parameters
# ---------------------------------------------------------------------
rho = 1.0
safety_eta = 1.0
show_vectors = True

# ---------------------------------------------------------------------
# Ellipsoid Math Helpers
# ---------------------------------------------------------------------
def get_ellipsoid_matrix(axes, angle_rad):
    """Computes matrix A = R * S * R^T for the ellipsoid definition."""
    a, b = axes
    # Scaling matrix S
    S = np.array([[1.0/(a**2), 0.0],
                  [0.0, 1.0/(b**2)]])
    # Rotation matrix R
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    R = np.array([[c, -s],
                  [s,  c]])
    return R @ S @ R.T

def is_inside_ellipsoid(p, obs):
    """Checks if point p is inside the ellipsoid for visualization skipping."""
    # Using the gamma function logic: if Gamma < 1, it's inside
    return ModulationAvoider.gamma_ellipsoid(p, obs["pos"], obs["A"]) < 1.0

# ---------------------------------------------------------------------
# ModulationAvoider Instance
# ---------------------------------------------------------------------
avoid = ModulationAvoider(
    gamma=lambda x: 1.0,
    grad_gamma=lambda x: np.array([1.0, 0.0]),
    rho=rho,
    safety_eta=safety_eta,
    tail_effect=False
)

def get_nominal_velocity(position, goal):
    error = goal - position
    dist = np.linalg.norm(error)
    if dist < 1e-3:
        return np.zeros(2)
    return error

# ---------------------------------------------------------------------
# Multi-Obstacle Modulation (Ellipsoidal)
# ---------------------------------------------------------------------
def compute_modulated_velocity(position, velocity_nominal, obstacles):
    v_mod = velocity_nominal.copy()
    gamma_vals = []

    for obs in obstacles:
        c = obs["pos"]
        A = obs["A"]

        # Override class functions with ellipsoidal math 
        avoid.gamma = lambda x, c=c, A=A: ModulationAvoider.gamma_ellipsoid(x, c, A)
        avoid.grad_gamma = lambda x, c=c, A=A: ModulationAvoider.grad_gamma_ellipsoid(x, c, A)
        avoid.rho = rho

        v_mod = avoid.modulate(position, position, v_mod)
        gamma_vals.append(avoid.gamma(position))

    return v_mod, min(gamma_vals) if gamma_vals else 10.0

# ---------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------
def main():
    global rho, show_vectors

    robot_pos = np.array([100.0, 300.0])
    goal_pos = np.array([900.0, 300.0])

    # Obstacles now have axes (a, b) and an angle [cite: 157]
    obstacles = [
        {"pos": np.array([350.0, 300.0]), "axes": [180.0, 40.0], "angle": 0.0},
        {"pos": np.array([600.0, 400.0]), "axes": [100.0, 50.0], "angle": math.radians(45)},
    ]

    # Pre-calculate matrices
    for obs in obstacles:
        obs["A"] = get_ellipsoid_matrix(obs["axes"], obs["angle"])

    dragging_idx = None
    running = True

    while running:
        dt = clock.tick(60) / 1000.0
        screen.fill(WHITE)
        mouse_pos = np.array(pygame.mouse.get_pos(), dtype=float)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left Click
                    goal_pos = mouse_pos
                elif event.button == 3: # Right Click
                    for i, obs in enumerate(obstacles):
                        if np.linalg.norm(mouse_pos - obs["pos"]) < 100:
                            dragging_idx = i
                            break
                elif event.button == 4: # Scroll Up
                    if dragging_idx is not None:
                        obstacles[dragging_idx]["angle"] += math.radians(5)
                        obstacles[dragging_idx]["A"] = get_ellipsoid_matrix(
                            obstacles[dragging_idx]["axes"], obstacles[dragging_idx]["angle"])
                elif event.button == 5: # Scroll Down
                    if dragging_idx is not None:
                        obstacles[dragging_idx]["angle"] -= math.radians(5)
                        obstacles[dragging_idx]["A"] = get_ellipsoid_matrix(
                            obstacles[dragging_idx]["axes"], obstacles[dragging_idx]["angle"])

            elif event.type == pygame.MOUSEBUTTONUP:
                dragging_idx = None

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    rho += 0.5
                elif event.key == pygame.K_DOWN:
                    rho = max(0.1, rho - 0.5)
                elif event.key == pygame.K_SPACE:
                    show_vectors = not show_vectors

        if dragging_idx is not None:
            obstacles[dragging_idx]["pos"] = mouse_pos

        # ---------------- Simulation ----------------
        v_nom = get_nominal_velocity(robot_pos, goal_pos)
        speed = 100.0
        if np.linalg.norm(v_nom) > 0:
            v_nom = (v_nom / np.linalg.norm(v_nom)) * speed

        v_mod, gamma_val = compute_modulated_velocity(robot_pos, v_nom, obstacles)
        robot_pos += v_mod * dt

        # ---------------- Vector Field ----------------
        if show_vectors:
            step = 45
            for x in range(0, WIDTH, step):
                for y in range(0, HEIGHT, step):
                    p = np.array([x, y], dtype=float)
                    if any(is_inside_ellipsoid(p, obs) for obs in obstacles):
                        continue

                    v_n = get_nominal_velocity(p, goal_pos)
                    if np.linalg.norm(v_n) > 0:
                        v_n = (v_n / np.linalg.norm(v_n)) * 20
                    
                    v_m, _ = compute_modulated_velocity(p, v_n, obstacles)
                    pygame.draw.line(screen, GREY, p, p + v_m, 1)

        # ---------------- Draw ----------------
        pygame.draw.circle(screen, GREEN, goal_pos.astype(int), 10)

        for obs in obstacles:
            # Draw rotated ellipsoid using a surface for transparency/rotation
            a, b = obs["axes"]
            ellipse_surface = pygame.Surface((a*2, b*2), pygame.SRCALPHA)
            pygame.draw.ellipse(ellipse_surface, (*RED, 180), (0, 0, a*2, b*2))
            
            rotated_surface = pygame.transform.rotate(ellipse_surface, -math.degrees(obs["angle"]))
            rect = rotated_surface.get_rect(center=obs["pos"].astype(int))
            screen.blit(rotated_surface, rect)

        pygame.draw.circle(screen, BLUE, robot_pos.astype(int), 8)
        
        # UI
        texts = [
            f"Reactivity (rho): {rho:.1f}",
            f"Gamma (closest): {gamma_val:.2f}",
            "Left Click: Set Goal | Right Click + Drag: Move Ellipsoid",
            "Mouse Wheel (while dragging): Rotate Obstacle",
            "SPACE: Toggle Field",
        ]
        for i, t in enumerate(texts):
            screen.blit(font.render(t, True, BLACK), (10, 10 + 25 * i))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()