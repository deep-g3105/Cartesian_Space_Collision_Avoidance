import pygame
import numpy as np
import math

# <-- IMPORT YOUR NEW CLASSES HERE
# Adjust the import path as needed to match your project structure
from modulation.class_joints import JointSpaceAvoider, JointSpaceObstacle   

# --- Constants & Setup ---
WIDTH, HEIGHT = 1200, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (50, 100, 255)
RED = (255, 50, 50)
GREEN = (50, 200, 50)
GREY = (220, 220, 220)
DARK_GREY = (100, 100, 100)
ORANGE = (255, 150, 50)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dual View: Cartesian & Joint Space Modulation (Updated Class)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 16)

# --- Robot Parameters ---
BASE_POS = np.array([WIDTH // 4, HEIGHT // 2], dtype=float)
L1 = 120.0  
L2 = 100.0  

# --- Global Simulation State (Allows stateless gamma functions to read current UI state) ---
class SimState:
    obs_pos = BASE_POS + np.array([100, 50])
    obs_radius = 40.0

# -----------------------------------------------------------------------------
#  KINEMATICS & GEOMETRY (Required for physical arm representation)
# -----------------------------------------------------------------------------

def forward_kinematics(q):
    theta1, theta2 = q[0], q[1]
    elbow = BASE_POS + np.array([L1 * np.cos(theta1), L1 * np.sin(theta1)])
    ee = elbow + np.array([L2 * np.cos(theta1 + theta2), L2 * np.sin(theta1 + theta2)])
    return elbow, ee

def inverse_kinematics(target_pos):
    dx, dy = target_pos[0] - BASE_POS[0], target_pos[1] - BASE_POS[1]
    dist_sq = dx**2 + dy**2
    c2 = max(-1.0, min(1.0, (dist_sq - L1**2 - L2**2) / (2 * L1 * L2)))
    theta2 = math.acos(c2) 
    s2 = math.sin(theta2)
    theta1 = math.atan2(dy, dx) - math.atan2(L2 * s2, L1 + L2 * c2)
    return np.array([(theta1 + np.pi) % (2 * np.pi) - np.pi, 
                     (theta2 + np.pi) % (2 * np.pi) - np.pi])

def get_closest_point_on_segment(A, B, P):
    AB, AP = B - A, P - A
    t = max(0.0, min(1.0, np.dot(AP, AB) / (np.dot(AB, AB) + 1e-8)))
    return A + t * AB, t

def calculate_jacobian_at_point(q, link_idx, t):
    theta1, theta2 = q[0], q[1]
    J = np.zeros((2, 2))
    if link_idx == 1:
        l = L1 * t
        J[0, 0], J[1, 0] = -l * np.sin(theta1), l * np.cos(theta1)
    elif link_idx == 2:
        l = L2 * t
        J[0, 0] = -L1 * np.sin(theta1) - l * np.sin(theta1 + theta2)
        J[1, 0] =  L1 * np.cos(theta1) + l * np.cos(theta1 + theta2)
        J[0, 1], J[1, 1] = -l * np.sin(theta1 + theta2), l * np.cos(theta1 + theta2)
    return J

def get_closest_arm_dist_and_jac(q, obs_pos):
    """Helper to find the closest point on either link to the obstacle."""
    elbow, ee = forward_kinematics(q)
    cp1, t1 = get_closest_point_on_segment(BASE_POS, elbow, obs_pos)
    cp2, t2 = get_closest_point_on_segment(elbow, ee, obs_pos)
    
    d1, d2 = np.linalg.norm(cp1 - obs_pos), np.linalg.norm(cp2 - obs_pos)
    
    if d1 < d2: return d1, cp1, 1, t1
    else:       return d2, cp2, 2, t2

# -----------------------------------------------------------------------------
#  JOINT SPACE AVOIDER SETUP
# -----------------------------------------------------------------------------

def gamma_fn(q):
    dist, _, _, _ = get_closest_arm_dist_and_jac(q, SimState.obs_pos)
    return (dist**2) / (SimState.obs_radius**2)

def grad_gamma_fn(q):
    dist, cp, link_idx, t = get_closest_arm_dist_and_jac(q, SimState.obs_pos)
    if dist < 1e-5: return np.array([1.0, 0.0])
    
    grad_x = 2 * (cp - SimState.obs_pos) / (SimState.obs_radius**2)
    J = calculate_jacobian_at_point(q, link_idx, t)
    return J.T @ grad_x

# Instantiate the obstacle and the avoider
dynamic_obstacle = JointSpaceObstacle(gamma_fn=gamma_fn, grad_gamma_fn=grad_gamma_fn, name="UI_Obstacle")
avoider = JointSpaceAvoider(obstacles=[dynamic_obstacle], dim=2)

# -----------------------------------------------------------------------------
# VISUAL MAPPING HELPERS
# -----------------------------------------------------------------------------
def q_to_screen(q):
    x = np.interp(q[0], [-np.pi, np.pi], [WIDTH//2 + 50, WIDTH - 50])
    y = np.interp(q[1], [-np.pi, np.pi], [HEIGHT - 50, 50])
    return np.array([x, y])

def screen_to_q(pos):
    t1 = np.interp(pos[0], [WIDTH//2 + 50, WIDTH - 50], [-np.pi, np.pi])
    t2 = np.interp(pos[1], [HEIGHT - 50, 50], [-np.pi, np.pi])
    return np.array([t1, t2])

# -----------------------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------------------

def main():
    q_robot = np.array([0.0, np.pi/4])
    cartesian_target = BASE_POS + np.array([150, -50])
    q_goal = inverse_kinematics(cartesian_target)
    
    q_history = []
    dragging_obs, dragging_goal = False, False
    running = True

    while running:
        dt = clock.tick(60) / 1000.0
        screen.fill(WHITE)
        mouse_pos = np.array(pygame.mouse.get_pos(), dtype=float)

        # ------------ INPUT ------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: 
                    if mouse_pos[0] < WIDTH // 2: dragging_goal = True
                    else:
                        q_goal = screen_to_q(mouse_pos)
                        _, cartesian_target = forward_kinematics(q_goal)
                elif event.button == 3: 
                    if mouse_pos[0] < WIDTH // 2 and np.linalg.norm(mouse_pos - SimState.obs_pos) < SimState.obs_radius + 20:
                        dragging_obs = True
            elif event.type == pygame.MOUSEBUTTONUP:
                dragging_obs = False
                dragging_goal = False

        if dragging_obs: 
            SimState.obs_pos = mouse_pos
        if dragging_goal:
            cartesian_target = mouse_pos
            q_goal = inverse_kinematics(cartesian_target)

        # ------------ DYNAMICS ------------
        # 1. Nominal Velocity
        v_nom_q = q_goal - q_robot
        v_nom_q = (v_nom_q + np.pi) % (2 * np.pi) - np.pi
        if np.linalg.norm(v_nom_q) > 2.5:
            v_nom_q = (v_nom_q / np.linalg.norm(v_nom_q)) * 2.5
            
        # 2. Modulate using the clean JointSpaceAvoider API
        if np.linalg.norm(v_nom_q) > 1e-3:
            v_mod_q = avoider.modulate(q_robot, v_nom_q)
        else:
            v_mod_q = np.zeros(2)
            
        q_robot += v_mod_q * dt
        q_robot = (q_robot + np.pi) % (2 * np.pi) - np.pi
        
        # History trailing for UI
        if len(q_history) == 0 or np.linalg.norm(q_history[-1] - q_robot) > 0.05:
            q_history.append(q_robot.copy())
            if len(q_history) > 100: q_history.pop(0)

        # ------------ DRAWING ------------
        pygame.draw.line(screen, BLACK, (WIDTH//2, 0), (WIDTH//2, HEIGHT), 2)

        # --- LEFT PANEL: CARTESIAN SPACE ---
        pygame.draw.circle(screen, RED, SimState.obs_pos.astype(int), int(SimState.obs_radius)) 
        pygame.draw.circle(screen, GREEN, cartesian_target.astype(int), 8) 
        pygame.draw.circle(screen, BLACK, cartesian_target.astype(int), 8, 1)

        curr_elbow, curr_ee = forward_kinematics(q_robot)
        pygame.draw.line(screen, BLACK, BASE_POS, curr_elbow, 8)
        pygame.draw.line(screen, BLACK, curr_elbow, curr_ee, 8)
        pygame.draw.circle(screen, DARK_GREY, BASE_POS.astype(int), 10)
        pygame.draw.circle(screen, DARK_GREY, curr_elbow.astype(int), 8)
        pygame.draw.circle(screen, BLUE, curr_ee.astype(int), 6)

        # --- RIGHT PANEL: JOINT SPACE ---
        pygame.draw.line(screen, GREY, (WIDTH//2 + 50, HEIGHT//2), (WIDTH - 50, HEIGHT//2), 1) 
        pygame.draw.line(screen, GREY, (WIDTH//4 * 3, 50), (WIDTH//4 * 3, HEIGHT - 50), 1) 
        
        if len(q_history) > 1:
            points = [q_to_screen(q) for q in q_history]
            for i in range(1, len(points)):
                if np.linalg.norm(q_history[i] - q_history[i-1]) < np.pi:
                    pygame.draw.line(screen, BLUE, points[i-1], points[i], 2)

        pygame.draw.circle(screen, ORANGE, q_to_screen(q_goal).astype(int), 8)
        pygame.draw.circle(screen, BLUE, q_to_screen(q_robot).astype(int), 6)

        # ------------ UI TEXT ------------
        # Retrieve gamma for UI display using the current robot position
        current_gamma = dynamic_obstacle.gamma_fn(q_robot)
        
        texts = [
            "CARTESIAN VIEW (Drag Left Click to set target)",
            f"Gamma: {current_gamma:.2f}",
            "Right click to move obstacle"
        ]
        for i, t in enumerate(texts):
            screen.blit(font.render(t, True, BLACK), (10, 10 + i * 25))
            
        screen.blit(font.render("JOINT SPACE (Theta 1 vs Theta 2)", True, BLACK), (WIDTH//2 + 10, 10))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()