import pygame
import numpy as np
from modulation.class_working import ModulationAvoider   # <-- IMPORT YOUR CLASS

# --- Constants & Setup ---
WIDTH, HEIGHT = 1080, 720
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (50, 100, 255)
RED = (255, 50, 50)
GREEN = (50, 200, 50)
GREY = (200, 200, 200)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("DS Modulation: Using ModulationAvoider Class")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 18)

# --- Simulation Parameters ---
rho = 1.0
safety_eta = 1.0
show_vectors = True

# -----------------------------------------------------------------------------
#  OBSTACLE GAMMA + GRAD_GAMMA — these call the class's static functions
# -----------------------------------------------------------------------------

def gamma_circle(x, center, radius):
    return ModulationAvoider.gamma_sphere(x, center=center, radius=radius)

def grad_gamma_circle(x, center, radius):
    return ModulationAvoider.grad_gamma_sphere(x, center=center, radius=radius)

# -----------------------------------------------------------------------------
#  Instantiate ModulationAvoider (BUT gamma/grad defined at runtime)
# -----------------------------------------------------------------------------

# We give dummy gamma function; we will replace center/radius per call using closures
avoid = ModulationAvoider(
    gamma=lambda x: 1.0, 
    grad_gamma=lambda x: np.array([1.0, 0.0]), 
    rho=rho, safety_eta=safety_eta, tail_effect=True
)

# --- Dynamical System Functions ---        
def get_nominal_velocity(position, goal):
    error = goal - position
    dist = np.linalg.norm(error)
    if dist < 1e-3:
        return np.zeros(2)
    return error

# -----------------------------------------------------------------------------
#  Helper to compute M(x) using the class
# -----------------------------------------------------------------------------

def compute_modulated_velocity(position, velocity_nominal, obs_pos, obs_radius):
    """
    Uses ModulationAvoider class to return modulated velocity.
    """

    # Update gamma + gradient functions for the current obstacle
    avoid.gamma = lambda x: gamma_circle(x, center=np.zeros(2), radius=obs_radius)
    avoid.grad_gamma = lambda x: grad_gamma_circle(x, center=np.zeros(2), radius=obs_radius)
    avoid.rho = rho

    # The class API: v_mod = modulate(x, x_tilde, f)
    # For circular obstacle, x_tilde = x - center
    x_tilde = position - obs_pos

    v_mod = avoid.modulate(position, x_tilde, velocity_nominal)

    # also return gamma for UI display
    gamma_val = avoid.gamma(x_tilde)

    return v_mod, gamma_val

# -----------------------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------------------

def main():
    global rho, show_vectors
    
    robot_pos = np.array([100.0, 300.0])
    goal_pos = np.array([700.0, 300.0])
    obs_pos = np.array([400.0, 305.0])
    obs_radius = 60
    
    dragging_obs = False
    running = True

    while running:
        dt = clock.tick(60) / 1000.0
        screen.fill(WHITE)
        
        mouse_pos = np.array(pygame.mouse.get_pos(), dtype=float)

        # ------------ INPUT ------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    goal_pos = mouse_pos
                elif event.button == 3:
                    if np.linalg.norm(mouse_pos - obs_pos) < obs_radius + 20:
                        dragging_obs = True

            elif event.type == pygame.MOUSEBUTTONUP:
                dragging_obs = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    rho += 0.5
                elif event.key == pygame.K_DOWN:
                    rho = max(0.1, rho - 0.5)
                elif event.key == pygame.K_SPACE:
                    show_vectors = not show_vectors

        if dragging_obs:
            obs_pos = mouse_pos

        # ------------ NOMINAL VELOCITY ------------
        v_nom = get_nominal_velocity(robot_pos, goal_pos)
        
        speed = 200.0
        if np.linalg.norm(v_nom) > 0:
            v_nom = v_nom / np.linalg.norm(v_nom) * speed

        # ------------ MODULATION USING CLASS ------------
        v_mod, gamma_val = compute_modulated_velocity(robot_pos, v_nom, obs_pos, obs_radius)

        robot_pos += v_mod * dt

        # ------------ VECTOR FIELD (OPTIONAL) ------------
        if show_vectors:
            step = 40
            for x in range(0, WIDTH, step):
                for y in range(0, HEIGHT, step):
                    p = np.array([x, y], dtype=float)
                    if np.linalg.norm(p - obs_pos) < obs_radius:
                        continue

                    v_n = get_nominal_velocity(p, goal_pos)
                    if np.linalg.norm(v_n) > 0:
                        v_n = v_n / np.linalg.norm(v_n) * 20

                    v_m, _ = compute_modulated_velocity(p, v_n, obs_pos, obs_radius)
                    
                    end = p + v_m
                    pygame.draw.line(screen, GREY, p, end, 1)
                    pygame.draw.circle(screen, GREY, end.astype(int), 2)

        # ------------ DRAW ------------
        pygame.draw.circle(screen, GREEN, goal_pos.astype(int), 10)
        pygame.draw.circle(screen, RED, obs_pos.astype(int), obs_radius)
        pygame.draw.circle(screen, BLUE, robot_pos.astype(int), 8)
        pygame.draw.line(screen, BLUE, robot_pos, robot_pos + v_mod * 0.5, 3)

        # ------------ UI TEXT ------------
        texts = [
            f"Reactivity (rho): {rho:.1f}",
            f"Gamma: {gamma_val:.2f}",
            "Left Click = Move Goal | Right Click = Move Obstacle",
            "SPACE = Toggle Vector Field"
        ]
        for i, t in enumerate(texts):
            s = font.render(t, True, BLACK)
            screen.blit(s, (10, 10 + i * 25))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
