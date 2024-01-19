import pygame
import sys
import math
import numpy as np
from config import config
from plant import RobotArmPlant, CournotPlant, BathtubPlant

def pid(params, err_hist):
    e = err_hist[-1]
    ei = sum(err_hist)
    ed = err_hist[-1] - err_hist[-2]
    x = np.array([e, ei, ed])

    return np.dot(params, x)

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2
POLE_LENGTH = 100

# Colors
WHITE = (255, 255, 255)
RED = (255, 50, 100)
BLACK = (0, 0, 0)
GREEN = (50, 255, 100)


# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Plant Simulation")

# Clock to control the frame rate
clock = pygame.time.Clock()

# Function to draw the pole and circle
def draw_pole(angle, target):
    pole_end_x = CENTER_X + POLE_LENGTH * math.cos(angle)
    pole_end_y = CENTER_Y - POLE_LENGTH * math.sin(angle)

    target_end_x = CENTER_X + POLE_LENGTH*math.cos(target)
    target_end_y = CENTER_Y - POLE_LENGTH*math.sin(target)

    pygame.draw.circle(screen, WHITE, (CENTER_X, CENTER_Y), 10)
    pygame.draw.line(screen, WHITE, (CENTER_X, CENTER_Y), (pole_end_x, pole_end_y), 5)
    pygame.draw.circle(screen, GREEN, (target_end_x, target_end_y), 7)
    pygame.draw.circle(screen, RED, (pole_end_x, pole_end_y), 7) 

# Main simulation loop
def main():
    # config['plant']['robot']['delta_time'] = .01
    # plant = RobotArmPlant(**config['plant']['robot'])
    # plant = CournotPlant(**config['plant']['cournot'])
    plant = BathtubPlant(**config['plant']['bathtub'])

    params = np.array([1, 0.2, 0.1])
    
    state, error = plant.reset()
    err_hist = np.array([error, error])
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            pygame.quit()

        if keys[pygame.K_LEFT]:
            U = -4.5
        elif keys[pygame.K_RIGHT]:
            U = 4.5
        else:
            U = 0
        # U = pid(params, err_hist)

        # Update angle based on torque or any other mechanism
        state, error = plant.step(state, U, np.random.uniform(-0.01, 0.01))

        # Clear the screen
        screen.fill(BLACK)

        # Draw the pole and circle
        draw_pole(state[0]*np.pi, plant.TARGET*np.pi)

        # Update the display
        pygame.display.flip()

        # Control the frame rate
        clock.tick(60)

        err_hist = np.append(err_hist, error)

if __name__ == "__main__":
    main()
