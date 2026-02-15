import pygame
import numpy as np
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
pygame.display.set_caption("DS Modulation – Dual Obstacles")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 18)

# ---------------------------------------------------------------------
# Simulation Parameters
# ---------------------------------------------------------------------
rho = 1.0
safety_eta = 1.0
show_vectors = True

# ---------------------------------------------------------------------
# Gamma and Gradient (Circle)
# ---------------------------------------------------------------------
def gamma_circle(x, center, radius):
    return ModulationAvoider.gamma_sphere(x, center=center, radius=radius)

def grad_gamma_circle(x, center, radius):
    return ModulationAvoider.grad_gamma_sphere(x, center=center, radius=radius)

# ---------------------------------------------------------------------
# ModulationAvoider Instance (functions overwritten at runtime)
# ---------------------------------------------------------------------
avoid = ModulationAvoider(
    gamma=lambda x: 1.0,
    grad_gamma=lambda x: np.array([1.0, 0.0]),
    rho=rho,
    safety_eta=safety_eta,
    tail_effect=True
)

# ---------------------------------------------------------------------
# Nominal Dynamical System
# ---------------------------------------------------------------------
def get_nominal_velocity(position, goal):
    error = goal - position
    dist = np.linalg.norm(error)
    if dist < 1e-3:
        return np.zeros(2)
    return error

# ---------------------------------------------------------------------
# Multi-Obstacle Modulation
# ---------------------------------------------------------------------
def compute_modulated_velocity(position, velocity_nominal, obstacles):
    v_mod = velocity_nominal.copy()
    gamma_vals = []

    for obs in obstacles:
        c = obs["pos"]
        r = obs["radius"]

        avoid.gamma = lambda x, c=c, r=r: gamma_circle(x, c, r)
        avoid.grad_gamma = lambda x, c=c, r=r: grad_gamma_circle(x, c, r)
        avoid.rho = rho

        v_mod = avoid.modulate(position, position, v_mod)
        gamma_vals.append(avoid.gamma(position))

    return v_mod, min(gamma_vals)

# ---------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------
def main():
    global rho, show_vectors

    robot_pos = np.array([100.0, 300.0])
    goal_pos = np.array([800.0, 300.0])

    obstacles = [
        {"pos": np.array([350.0, 300.0]), "radius": 40},
        {"pos": np.array([520.0, 340.0]), "radius": 40},
    ]

    dragging_idx = None
    running = True

    while running:
        dt = clock.tick(60) / 1000.0
        screen.fill(WHITE)
        mouse_pos = np.array(pygame.mouse.get_pos(), dtype=float)

        # ---------------- Input ----------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    goal_pos = mouse_pos
                elif event.button == 3:
                    for i, obs in enumerate(obstacles):
                        if np.linalg.norm(mouse_pos - obs["pos"]) < obs["radius"] + 20:
                            dragging_idx = i
                            break

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

        # ---------------- Nominal Velocity ----------------
        v_nom = get_nominal_velocity(robot_pos, goal_pos)
        speed = 200.0
        if np.linalg.norm(v_nom) > 0:
            v_nom = v_nom / np.linalg.norm(v_nom) * speed

        # ---------------- Modulation ----------------
        v_mod, gamma_val = compute_modulated_velocity(
            robot_pos, v_nom, obstacles
        )
        robot_pos += v_mod * dt

        # ---------------- Vector Field ----------------
        if show_vectors:
            step = 40
            for x in range(0, WIDTH, step):
                for y in range(0, HEIGHT, step):
                    p = np.array([x, y], dtype=float)

                    skip = False
                    for obs in obstacles:
                        if np.linalg.norm(p - obs["pos"]) < obs["radius"]:
                            skip = True
                            break
                    if skip:
                        continue

                    v_n = get_nominal_velocity(p, goal_pos)
                    if np.linalg.norm(v_n) > 0:
                        v_n = v_n / np.linalg.norm(v_n) * 20

                    v_m, _ = compute_modulated_velocity(p, v_n, obstacles)
                    end = p + v_m

                    pygame.draw.line(screen, GREY, p, end, 1)
                    pygame.draw.circle(screen, GREY, end.astype(int), 2)

        # ---------------- Draw ----------------
        pygame.draw.circle(screen, GREEN, goal_pos.astype(int), 10)

        for obs in obstacles:
            pygame.draw.circle(
                screen, RED, obs["pos"].astype(int), obs["radius"]
            )

        pygame.draw.circle(screen, BLUE, robot_pos.astype(int), 8)
        pygame.draw.line(
            screen, BLUE, robot_pos, robot_pos + 0.5 * v_mod, 3
        )

        # ---------------- UI ----------------
        texts = [
            f"Reactivity (rho): {rho:.1f}",
            f"Gamma (closest): {gamma_val:.2f}",
            "Left Click = Move Goal",
            "Right Click = Drag Obstacles",
            "SPACE = Toggle Vector Field",
        ]

        for i, t in enumerate(texts):
            screen.blit(font.render(t, True, BLACK), (10, 10 + 25 * i))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
