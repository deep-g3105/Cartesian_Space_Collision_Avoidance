import pygame
import numpy as np

# --- Constants & Setup ---
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (50, 100, 255)
RED = (255, 50, 50)
GREEN = (50, 200, 50)
GREY = (200, 200, 200)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("DS Modulation: Understanding Eq 5, 6, 7")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 18)

# --- Simulation Parameters ---
rho = 1.0  # Reactivity parameter (Eq 7 exponent)
show_vectors = True

# --- Dynamical System Functions ---

def get_nominal_velocity(position, goal):
    """
    The Nominal DS (f(x)). 
    Linear attractor: moves straight to goal.
    """
    error = goal - position
    dist = np.linalg.norm(error)
    if dist < 1e-3:
        return np.zeros(2)
    return error # proportional control

def get_modulation_matrix(position, obstacle_pos, obstacle_radius, rho):
    """
    Calculates M(x) based on Eq 5, 6, 7.
    M(x) = E(x) * D(x) * E(x)^-1
    """
    # 1. Calculate Distance Function Gamma (simplified for circle)
    # Gamma > 1 means outside, Gamma = 1 is surface
    diff = position - obstacle_pos
    dist_to_center = np.linalg.norm(diff)
    
    # Avoid division by zero inside obstacle
    if dist_to_center < 1e-6: 
        dist_to_center = 1e-6
        
    # Standard Gamma for a sphere: Gamma = dist / radius
    gamma = dist_to_center / obstacle_radius 
    
    # If inside obstacle, force repulsion
    if gamma < 1.0:
        gamma = 1.001 

    # 2. Calculate Basis Matrix E (Eq 7 related)
    # Normal vector (n)
    n = diff / dist_to_center
    # Tangent vector (t) - perpendicular to n in 2D [-y, x]
    t = np.array([-n[1], n[0]])
    
    E = np.column_stack((n, t))
    
    # 3. Calculate Eigenvalue Matrix D (Eq 6 & 7)
    # lambda_n: Repulsion along normal (goes to 0 at surface)
    # lambda_t: Tangential flow (remains 1 or increases)
    
    # Weight w goes from 0 (far away) to 1 (at surface)
    # Reactivity 'rho' controls how sharp this transition is
    w = 1 / (gamma**rho) 
    
    lambda_n = 1 - w  # Stops velocity towards obstacle at surface
    lambda_t = 1 + w  # Accelerates along the tangent
    
    D = np.diag([lambda_n, lambda_t])
    
    # 4. Compute M = E * D * inv(E) (Eq 5)
    # Since E is orthogonal (rotation), inv(E) == E.T
    M = E @ D @ E.T
    
    return M, gamma

# --- Main Logic ---

def main():
    global rho, show_vectors
    
    # Initial State
    robot_pos = np.array([100.0, 300.0])
    goal_pos = np.array([700.0, 300.0])
    obs_pos = np.array([400.0, 305.0]) # Slightly offset to show modulation
    obs_radius = 60
    
    dragging_obs = False
    
    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        screen.fill(WHITE)
        
        # --- Input Handling ---
        mouse_pos = np.array(pygame.mouse.get_pos(), dtype=float)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left click moves goal
                    goal_pos = mouse_pos
                elif event.button == 3: # Right click grabs obstacle
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

        # --- Physics Update (The Robot) ---
        
        # 1. Get Nominal Velocity (f(x))
        v_nom = get_nominal_velocity(robot_pos, goal_pos)
        
        # Normalize nominal speed for consistent movement
        speed = 200.0 
        if np.linalg.norm(v_nom) > 0:
            v_nom = (v_nom / np.linalg.norm(v_nom)) * speed
            
        # 2. Get Modulation Matrix M(x)
        M, gamma_val = get_modulation_matrix(robot_pos, obs_pos, obs_radius, rho)
        
        # 3. Apply Modulation: v_mod = M * v_nom (Eq 5)
        v_mod = M @ v_nom
        
        # Update Position
        robot_pos += v_mod * dt

        # --- Visualization ---

        # 1. Draw Vector Field (Grid)
        if show_vectors:
            step = 40
            for x in range(0, WIDTH, step):
                for y in range(0, HEIGHT, step):
                    p = np.array([x, y], dtype=float)
                    # Skip inside obstacle
                    if np.linalg.norm(p - obs_pos) < obs_radius: continue
                    
                    v_n = get_nominal_velocity(p, goal_pos)
                    if np.linalg.norm(v_n) > 0: v_n = (v_n/np.linalg.norm(v_n))*20
                    
                    # Apply Modulation math here too for visualization
                    M_local, _ = get_modulation_matrix(p, obs_pos, obs_radius, rho)
                    v_m = M_local @ v_n
                    
                    # Draw arrow
                    end_pos = p + v_m
                    pygame.draw.line(screen, GREY, p, end_pos, 1)
                    pygame.draw.circle(screen, GREY, end_pos.astype(int), 2)

        # 2. Draw Entities
        # Goal
        pygame.draw.circle(screen, GREEN, goal_pos.astype(int), 10)
        # Obstacle
        pygame.draw.circle(screen, RED, obs_pos.astype(int), obs_radius)
        # Robot
        pygame.draw.circle(screen, BLUE, robot_pos.astype(int), 8)
        
        # Draw Trajectory/Velocity Arrow for robot
        pygame.draw.line(screen, BLUE, robot_pos, robot_pos + v_mod * 0.5, 3)

        # --- UI Text ---
        texts = [
            f"Reactivity (rho): {rho:.1f} (UP/DOWN to change)",
            f"Gamma (Distance metric): {gamma_val:.2f}",
            "Left Click: Move Goal | Right Click: Move Obstacle",
            "SPACE: Toggle Vector Field"
        ]
        
        for i, t in enumerate(texts):
            s = font.render(t, True, BLACK)
            screen.blit(s, (10, 10 + i * 25))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()