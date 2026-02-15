import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
from typing import Callable, Optional

# Assuming your class is in modulation/class_working.py
from modulation.class_working import ModulationAvoider

# ---------------------------------------------------------------------
# 3D Math Helpers
# ---------------------------------------------------------------------
def get_ellipsoid_matrix_3d(axes, euler_angles):
    a, b, c_len = axes
    S = np.diag([1.0/(a**2), 1.0/(b**2), 1.0/(c_len**2)])
    rx, ry, rz = euler_angles
    
    # Rotation Matrices
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    
    R = Rz @ Ry @ Rx
    return R @ S @ R.T

# ---------------------------------------------------------------------
# Driver-Safe Rendering Helpers
# ---------------------------------------------------------------------
# Create a global quadric once to avoid SegFaults and memory leaks
_QUADRIC = None

def get_quadric():
    global _QUADRIC
    if _QUADRIC is None:
        _QUADRIC = gluNewQuadric()
        gluQuadricNormals(_QUADRIC, GLU_SMOOTH)
    return _QUADRIC

def draw_sphere(radius=1.0, slices=16, stacks=16):
    quad = get_quadric()
    gluSphere(quad, radius, slices, stacks)

def draw_ellipsoid(pos, axes, euler_angles, color):
    glPushMatrix()
    glTranslatef(pos[0], pos[1], pos[2])
    # OpenGL rotations: Yaw (Z), Pitch (Y), Roll (X)
    glRotatef(math.degrees(euler_angles[2]), 0, 0, 1)
    glRotatef(math.degrees(euler_angles[1]), 0, 1, 0)
    glRotatef(math.degrees(euler_angles[0]), 1, 0, 0)
    glScalef(axes[0], axes[1], axes[2])
    glColor4f(*color)
    draw_sphere(radius=1.0)
    glPopMatrix()

# ---------------------------------------------------------------------
# Main Simulation
# ---------------------------------------------------------------------
def main():
    pygame.init()
    display = (1080, 720)
    
    # Try to set specific OpenGL attributes before creating the window
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    
    try:
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    except pygame.error:
        # Fallback for drivers that don't like version requests
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    pygame.display.set_caption("3D DS Modulation - Ellipsoids")

    # OpenGL Setup
    glViewport(0, 0, display[0], display[1])
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (display[0]/display[1]), 0.1, 500.0)
    glMatrixMode(GL_MODELVIEW)
    
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Simulation State
    robot_pos = np.array([-15.0, 0.0, 0.0])
    goal_pos = np.array([15.0, 2.0, 0.0])
    rho = 1.0
    
    obstacles = [
        {
            "pos": np.array([0.0, 0.0, 0.0]), 
            "axes": [4.0, 2.0, 2.0], 
            "euler": [0, 0, math.radians(45)],
            "color": (1.0, 0.2, 0.2, 0.6)
        },
        {
            "pos": np.array([8.0, 3.0, 1.0]), 
            "axes": [2.0, 5.0, 2.0], 
            "euler": [math.radians(30), 0, 0],
            "color": (0.2, 1.0, 0.2, 0.6)
        }
    ]

    for obs in obstacles:
        obs["A"] = get_ellipsoid_matrix_3d(obs["axes"], obs["euler"])

    avoid = ModulationAvoider(
        gamma=lambda x: 1.0,
        grad_gamma=lambda x: np.array([1.0, 0.0, 0.0]),
        rho=rho,
        tail_effect=True
    )

    clock = pygame.time.Clock()
    cam_rot = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        dt = clock.tick(60) / 1000.0
        cam_rot += 0.2

        # 1. Nominal Velocity
        v_nom = goal_pos - robot_pos
        dist_to_goal = np.linalg.norm(v_nom)
        if dist_to_goal > 0.1:
            v_nom = (v_nom / dist_to_goal) * 5.0
        else:
            v_nom = np.zeros(3)

        # 2. Sequential Modulation
        v_mod = v_nom.copy()
        for obs in obstacles:
            c, A = obs["pos"], obs["A"]
            avoid.gamma = lambda x, c=c, A=A: ModulationAvoider.gamma_ellipsoid(x, c, A)
            avoid.grad_gamma = lambda x, c=c, A=A: ModulationAvoider.grad_gamma_ellipsoid(x, c, A)
            
            # Use relative position x_tilde = x - center
            v_mod = avoid.modulate(robot_pos, robot_pos - c, v_mod)

        robot_pos += v_mod * dt

        # 3. Rendering
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        gluLookAt(25 * math.cos(math.radians(cam_rot)), 15, 25 * math.sin(math.radians(cam_rot)), 
                  0, 0, 0, 
                  0, 1, 0)

        # Draw Goal
        glPushMatrix()
        glTranslatef(*goal_pos)
        glColor3f(0.2, 1.0, 0.2)
        draw_sphere(0.5)
        glPopMatrix()

        # Draw Obstacles
        for obs in obstacles:
            draw_ellipsoid(obs["pos"], obs["axes"], obs["euler"], obs["color"])

        # Draw Robot
        glPushMatrix()
        glTranslatef(*robot_pos)
        glColor3f(0.2, 0.2, 1.0)
        draw_sphere(0.4)
        glPopMatrix()

        # Draw Grid
        glColor3f(0.5, 0.5, 0.5)
        glBegin(GL_LINES)
        for i in range(-20, 21, 2):
            glVertex3f(i, -5, -20); glVertex3f(i, -5, 20)
            glVertex3f(-20, -5, i); glVertex3f(20, -5, i)
        glEnd()

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()