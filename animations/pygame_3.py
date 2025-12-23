import pygame
import numpy as np

# --- Constants ---
WIDTH, HEIGHT = 1000, 500
half_width = WIDTH // 2
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 50, 50)
BLUE = (50, 100, 255)
GREEN = (50, 200, 50)
GREY = (200, 200, 200)
GHOST_BLUE = (50, 100, 255, 100) # Transparent blue

# Robot Parameters
L1, L2 = 150, 150
# Joint limits (radians)
J1_MIN, J1_MAX = np.radians(-180), np.radians(180)
J2_MIN, J2_MAX = np.radians(-150), np.radians(150)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Left Click LEFT side to set Goal (IK) | Right Click Drag Obstacle")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 16)

# --- Kinematics ---

def forward_kinematics(q):
    """ Returns (x1, y1) elbow and (x2, y2) end-effector for angles q=[th1, th2] """
    x1 = L1 * np.cos(q[0])
    y1 = L1 * np.sin(q[0])
    x2 = x1 + L2 * np.cos(q[0] + q[1])
    y2 = y1 + L2 * np.sin(q[0] + q[1])
    return np.array([x1, y1]), np.array([x2, y2])

def inverse_kinematics(target_pos):
    """ 
    Geometric IK for 2-link planar arm.
    Returns best q = [q1, q2] to reach target_pos (x, y).
    """
    x, y = target_pos
    
    # 1. Check if target is reachable
    r = np.sqrt(x**2 + y**2)
    if r > (L1 + L2):
        # Scale back to max reach
        ratio = (L1 + L2) / r
        x *= ratio
        y *= ratio
        r = L1 + L2
    
    # 2. Law of Cosines for q2 (Elbow angle)
    # r^2 = L1^2 + L2^2 - 2*L1*L2*cos(pi - q2) 
    # -> cos(q2) = (r^2 - L1^2 - L2^2) / (2*L1*L2)
    cos_q2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    
    # Clamp for numerical stability
    cos_q2 = max(-1.0, min(1.0, cos_q2))
    q2 = np.arccos(cos_q2) # We pick the positive solution (Elbow down-ish)
    
    # 3. Calculate q1
    # theta = atan2(y, x)
    # alpha = atan2(L2*sin(q2), L1 + L2*cos(q2))
    # q1 = theta - alpha
    theta = np.arctan2(y, x)
    alpha = np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    q1 = theta - alpha
    
    return np.array([q1, q2])

def get_distance_to_obstacle(q, obs_pos, obs_radius):
    elbow, ee = forward_kinematics(q)
    base = np.array([0, 0])
    min_dist = float('inf')
    
    # Sample points along the arm
    steps = 10
    for i in range(steps + 1):
        t = i / steps
        # Link 1 point
        p1 = base + (elbow - base) * t
        d1 = np.linalg.norm(p1 - obs_pos) - obs_radius
        min_dist = min(min_dist, d1)
        # Link 2 point
        p2 = elbow + (ee - elbow) * t
        d2 = np.linalg.norm(p2 - obs_pos) - obs_radius
        min_dist = min(min_dist, d2)
            
    return min_dist

def get_joint_space_obstacle_map(obs_pos, obs_radius):
    """ Renders the 'Forbidden Region' for the right panel """
    s = pygame.Surface((half_width, HEIGHT))
    s.fill(WHITE)
    
    # Low res scan for performance
    step = 8
    for x in range(0, half_width, step): 
        for y in range(0, HEIGHT, step): 
            q1 = np.interp(x, [0, half_width], [J1_MIN, J1_MAX])
            q2 = np.interp(y, [HEIGHT, 0], [J2_MIN, J2_MAX])
            
            # Simple collision check for center of pixel
            dist = get_distance_to_obstacle([q1, q2], obs_pos, obs_radius)
            
            if dist <= 0:
                pygame.draw.rect(s, (255, 180, 180), (x, y, step, step))
            elif dist < 20:
                val = 240
                pygame.draw.rect(s, (val, val, val), (x, y, step, step))
    return s

# --- Modulation ---

def get_modulation(q, obs_pos, obs_radius, goal_q):
    # Nominal DS (Linear in Joint Space)
    v_nom = goal_q - q
    dist_to_goal = np.linalg.norm(v_nom)
    if dist_to_goal > 0:
        speed = 2.0
        v_nom = v_nom / dist_to_goal * speed

    # Gradient of Gamma
    eps = 1e-4
    gamma = get_distance_to_obstacle(q, obs_pos, obs_radius)
    
    g_q1 = (get_distance_to_obstacle(q + [eps, 0], obs_pos, obs_radius) - gamma) / eps
    g_q2 = (get_distance_to_obstacle(q + [0, eps], obs_pos, obs_radius) - gamma) / eps
    
    normal = np.array([g_q1, g_q2])
    norm_len = np.linalg.norm(normal)
    
    if norm_len > 1e-6:
        normal = normal / norm_len
        tangent = np.array([-normal[1], normal[0]])
        E = np.column_stack((normal, tangent))
        
        # Eigenvalues
        safe_margin = 40.0 
        w = max(0, min(1, gamma / safe_margin))
        D = np.diag([w, 1.0 + (1-w)])
        
        M = E @ D @ E.T
        v_mod = M @ v_nom
        return v_mod
    else:
        return v_nom # Too far from obstacle or inside flat region

def main():
    # State
    q = np.array([0.0, 0.0]) 
    q_goal = np.array([np.radians(45), np.radians(45)])
    
    obs_pos_world = np.array([120.0, 50.0]) 
    obs_radius = 35
    
    base_screen = np.array([half_width // 2, HEIGHT // 2])
    
    # Pre-compute map initially
    map_surface = get_joint_space_obstacle_map(obs_pos_world, obs_radius)
    
    dragging_obs = False
    
    running = True
    while running:
        dt = clock.tick(60) / 500.0
        
        mx, my = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left Click
                    if mx < half_width:
                        # --- CLICK IN WORKSPACE (Use IK) ---
                        # Convert screen (y-flipped) to robot coords
                        rx = mx - base_screen[0]
                        ry = -(my - base_screen[1])
                        
                        # Solve IK
                        new_q = inverse_kinematics([rx, ry])
                        
                        # Simple check if IK result is valid/safe
                        if not np.isnan(new_q).any():
                            q_goal = new_q
                    else:
                        # --- CLICK IN JOINT SPACE (Manual) ---
                        t1 = np.interp(mx - half_width, [0, half_width], [J1_MIN, J1_MAX])
                        t2 = np.interp(my, [HEIGHT, 0], [J2_MIN, J2_MAX])
                        q_goal = np.array([t1, t2])
                        
                elif event.button == 3: # Right Click (Drag Obstacle)
                    # Check if clicking near obstacle in workspace
                    obs_scr_x = base_screen[0] + obs_pos_world[0]
                    obs_scr_y = base_screen[1] - obs_pos_world[1]
                    if np.hypot(mx - obs_scr_x, my - obs_scr_y) < obs_radius + 10:
                        dragging_obs = True

            elif event.type == pygame.MOUSEBUTTONUP:
                if dragging_obs:
                    dragging_obs = False
                    # Recompute map when obstacle stops moving
                    map_surface = get_joint_space_obstacle_map(obs_pos_world, obs_radius)
        
        if dragging_obs:
            if mx < half_width:
                obs_pos_world[0] = mx - base_screen[0]
                obs_pos_world[1] = -(my - base_screen[1])

        # --- Update Physics ---
        v = get_modulation(q, obs_pos_world, obs_radius, q_goal)
        q += v * dt * 0.5 # Integrate
        
        # --- Draw ---
        screen.fill(WHITE)
        
        # LEFT PANEL: Workspace
        # 1. Obstacle
        obs_scr = base_screen + np.array([obs_pos_world[0], -obs_pos_world[1]])
        pygame.draw.circle(screen, RED, obs_scr.astype(int), obs_radius)
        
        # 2. Ghost Robot (Goal)
        g_elb, g_ee = forward_kinematics(q_goal)
        g_elb_scr = base_screen + np.array([g_elb[0], -g_elb[1]])
        g_ee_scr = base_screen + np.array([g_ee[0], -g_ee[1]])
        
        # Draw ghost lines (light grey/blue)
        pygame.draw.line(screen, (200, 200, 255), base_screen, g_elb_scr, 3)
        pygame.draw.line(screen, (200, 200, 255), g_elb_scr, g_ee_scr, 3)
        pygame.draw.circle(screen, GREEN, g_ee_scr.astype(int), 6) # Goal End Effector
        
        # 3. Real Robot
        elb, ee = forward_kinematics(q)
        elb_scr = base_screen + np.array([elb[0], -elb[1]])
        ee_scr = base_screen + np.array([ee[0], -ee[1]])
        
        pygame.draw.line(screen, BLACK, base_screen, elb_scr, 5)
        pygame.draw.line(screen, BLACK, elb_scr, ee_scr, 5)
        pygame.draw.circle(screen, BLUE, base_screen, 8)
        pygame.draw.circle(screen, BLUE, elb_scr, 8)
        pygame.draw.circle(screen, BLUE, ee_scr, 8)
        
        # RIGHT PANEL: Joint Space
        screen.blit(map_surface, (half_width, 0))
        pygame.draw.line(screen, BLACK, (half_width, 0), (half_width, HEIGHT), 3)
        
        # Map Q to pixels
        qx = np.interp(q[0], [J1_MIN, J1_MAX], [0, half_width])
        qy = np.interp(q[1], [J2_MIN, J2_MAX], [HEIGHT, 0])
        gx = np.interp(q_goal[0], [J1_MIN, J1_MAX], [0, half_width])
        gy = np.interp(q_goal[1], [J2_MIN, J2_MAX], [HEIGHT, 0])
        
        pygame.draw.circle(screen, GREEN, (int(half_width + gx), int(gy)), 8)
        pygame.draw.circle(screen, BLUE, (int(half_width + qx), int(qy)), 8)
        
        # Instructions
        s1 = font.render("Left Panel: Workspace (Click to set Goal)", True, BLACK)
        s2 = font.render("Right Panel: Joint Space (The Math)", True, BLACK)
        s3 = font.render("Right Click & Drag Obstacle on Left Panel", True, RED)
        screen.blit(s1, (10, 10))
        screen.blit(s2, (half_width + 10, 10))
        screen.blit(s3, (10, HEIGHT - 30))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()