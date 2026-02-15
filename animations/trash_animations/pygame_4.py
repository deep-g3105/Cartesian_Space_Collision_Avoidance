import pygame
import numpy as np

# --- Constants ---
WIDTH, HEIGHT = 1920, 1080
CENTER = np.array([WIDTH // 2, HEIGHT // 2])
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (150, 150, 150)
RED = (200, 50, 50)     # Normal Component (Blocked)
BLUE = (50, 100, 255)   # Tangent Component (Allowed)
GREEN = (50, 200, 50)   # Input Nominal Velocity
PURPLE = (180, 50, 180) # Output Modulated Velocity

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Visualizing Nullspace and Rank Loss")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Consolas", 18)
font_large = pygame.font.SysFont("Consolas", 24, bold=True)

def draw_arrow(surface, start, vector, color, label=None, scale=1.0):
    if np.linalg.norm(vector) < 1: return
    
    end = start + vector * scale
    pygame.draw.line(surface, color, start, end, 3)
    
    # Arrowhead
    angle = np.arctan2(vector[1], vector[0])
    arrow_size = 10
    p1 = end - np.array([np.cos(angle - 0.5), np.sin(angle - 0.5)]) * arrow_size
    p2 = end - np.array([np.cos(angle + 0.5), np.sin(angle + 0.5)]) * arrow_size
    
    pygame.draw.polygon(surface, color, [end, p1, p2])
    
    if label:
        text = font.render(label, True, color)
        surface.blit(text, (end[0] + 5, end[1] + 5))

def main():
    running = True
    
    # Simulation State
    # We place the robot effectively "on" the boundary
    wall_x = 450
    robot_pos = np.array([400.0, 300.0]) 
    
    # The normal vector points AWAY from the obstacle (Left)
    normal_vector = np.array([-1.0, 0.0]) 
    # The tangent vector (Up)
    tangent_vector = np.array([0.0, -1.0]) 
    
    # Lambda values
    # We simulate being ON the boundary, so Lambda_n is effectively 0
    lambda_n = 0.0 
    lambda_t = 1.5 # Tangent allows sliding
    
    while running:
        screen.fill(WHITE)
        
        # Input Handling
        mouse_pos = np.array(pygame.mouse.get_pos())
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # 1. Define Nominal Velocity (Green Arrow)
        # It points from robot towards mouse
        v_nom = mouse_pos - robot_pos
        # Cap length for visualization
        if np.linalg.norm(v_nom) > 150:
            v_nom = v_nom / np.linalg.norm(v_nom) * 150
            
        # 2. Decompose Vector (The Math behind Matrix multiplication)
        # Project v_nom onto Normal
        # (v . n) * n
        mag_normal = np.dot(v_nom, normal_vector)
        vec_normal_component = mag_normal * normal_vector
        
        # Project v_nom onto Tangent
        # (v . t) * t
        mag_tangent = np.dot(v_nom, tangent_vector)
        vec_tangent_component = mag_tangent * tangent_vector
        
        # 3. Apply Eigenvalues (The Modulation)
        # This is where the Nullspace is created!
        final_normal = vec_normal_component * lambda_n # Multiplied by 0!
        final_tangent = vec_tangent_component * lambda_t
        
        v_mod = final_normal + final_tangent

        # --- DRAWING ---
        
        # Draw Wall
        pygame.draw.rect(screen, GREY, (wall_x, 100, 200, 400))
        text_wall = font_large.render("OBSTACLE", True, BLACK)
        screen.blit(text_wall, (wall_x + 40, 280))
        
        # Draw Robot
        pygame.draw.circle(screen, BLACK, robot_pos.astype(int), 8)
        
        # Draw Coordinate Basis (Ghosted)
        draw_arrow(screen, robot_pos, normal_vector * 50, (200, 200, 200), "n")
        draw_arrow(screen, robot_pos, tangent_vector * 50, (200, 200, 200), "t")

        # Draw Vectors
        # 1. Nominal (The Input)
        draw_arrow(screen, robot_pos, v_nom, GREEN, "v_nom (Input)")
        
        # 2. Components (The Decomposition)
        # Shift start point slightly so lines don't overlap
        offset = np.array([0, 150]) 
        
        center_diagram = np.array([150, 450])
        
        # Show Decomposition UI
        pygame.draw.rect(screen, (240, 240, 240), (20, 350, 300, 230))
        pygame.draw.circle(screen, BLACK, center_diagram, 5)
        
        # Draw the Red/Blue components in the UI box
        draw_arrow(screen, center_diagram, vec_normal_component, RED, "Normal Part")
        draw_arrow(screen, center_diagram, vec_tangent_component, BLUE, "Tangent Part")
        
        # Draw the Math
        t1 = font.render(f"Normal Component: {int(mag_normal)}", True, RED)
        t2 = font.render(f" * Lambda_n ({lambda_n}) = 0 (DELETED)", True, RED)
        
        t3 = font.render(f"Tangent Component: {int(mag_tangent)}", True, BLUE)
        t4 = font.render(f" * Lambda_t ({lambda_t}) = {int(mag_tangent*lambda_t)}", True, BLUE)
        
        screen.blit(t1, (30, 370))
        screen.blit(t2, (30, 390))
        screen.blit(t3, (30, 420))
        screen.blit(t4, (30, 440))
        
        # 3. Final Result (Purple) on Main Screen
        draw_arrow(screen, robot_pos, v_mod, PURPLE, "v_mod (Output)", scale=1.0)

        # --- Explain the Nullspace ---
        
        # Check if stuck
        if np.linalg.norm(v_mod) < 5 and np.linalg.norm(v_nom) > 50:
            status_color = RED
            status_text = "STATUS: STUCK IN NULLSPACE!"
            sub_text = "Input is purely Normal. Output is Zero."
        else:
            status_color = BLACK
            status_text = "STATUS: Moving (Tangent component exists)"
            sub_text = "Tangential energy slides robot along wall."

        status_s = font_large.render(status_text, True, status_color)
        sub_s = font.render(sub_text, True, BLACK)
        screen.blit(status_s, (20, 20))
        screen.blit(sub_s, (20, 50))
        
        inst = font.render("Move MOUSE to change intended direction.", True, GREY)
        screen.blit(inst, (20, 80))

        pygame.display.flip()
        
    pygame.quit()

if __name__ == "__main__":
    main()