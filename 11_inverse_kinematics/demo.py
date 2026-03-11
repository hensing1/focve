# /// script
# dependencies = [
#   "numpy",
#   "torch",
#   "pygame",
# ]
# ///

import sys
import torch
import pygame

import src.inverse_kinematics as ik



def draw_screen(screen, background_color, shape_color, points, parents):
    screen.fill(background_color)

    # Draw the points
    for i in range(points.shape[0]):
        pygame.draw.circle(screen, shape_color, points[i].cpu().numpy().astype(int), 10.0)

    # Draw lines connecting the points
    for i in range(1, parents.shape[0]):
        pygame.draw.line(screen, shape_color, points[i].cpu().numpy().astype(int), points[parents[i]].cpu().numpy().astype(int), 2)



def main() -> None:
    model = sys.argv[1]

    # Initialize pygame
    pygame.init()

    # Set up the screen dimensions
    WIDTH, HEIGHT = 1400, 800
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Inverse Kinematics")

    # Define colors
    BACKGROUND_COLOR = (25, 25, 25)
    SHAPE_COLOR = (170, 170, 170)

    if model == "arm" or model == "0":
        # first entries not used, just for nicer indexing
        parents = torch.tensor([-1, 0, 1, 2, 3], dtype=torch.int32)
        angles = torch.tensor([torch.inf, -90, -30, 30, 0]) * (torch.pi / 180)
        lengths = torch.tensor([0, 100, 80, 50, 100])
    elif model == "skeleton" or model == "1":
        # first entries not used, just for nicer indexing
        parents = torch.tensor([-1, 0, 1, 1, 3, 4, 1, 6, 7, 0, 9, 10, 0, 12, 13], dtype=torch.int32)
        angles = torch.tensor([torch.inf, -90, -90, 0, 60, 90, 180, 120, 90, 45, 90, 90, 135, 90, 90]) * (torch.pi / 180)
        lengths = torch.tensor([0, 150, 80, 50, 100, 100, 50, 100, 100, 50, 100, 100, 50, 100, 100])


    # create all joint positions
    points = ik.compute_joint_positions(torch.tensor([WIDTH // 2, HEIGHT // 2]), angles, lengths, parents)

    # Setup for moving the points
    dragged_point_index = None

    # Game loop
    running = True
    while running:
        target_positions = points.clone()

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                # Check if the mouse clicked inside any point's area
                mouse_x, mouse_y = pygame.mouse.get_pos()
                mouse_position = torch.tensor([mouse_x, mouse_y])
                for i in range(points.shape[0]):
                    if torch.linalg.norm(points[i] - mouse_position) <= 10:  # radius of 10px for "grabbable" area
                        dragged_point_index = i
                        break
            
            if event.type == pygame.MOUSEBUTTONUP:
                # Stop dragging when mouse is released
                dragged_point_index = None
            
        if dragged_point_index is not None:
            # Update the position of the dragged point
            mouse_x, mouse_y = pygame.mouse.get_pos()
            target_positions[dragged_point_index, 0] = mouse_x
            target_positions[dragged_point_index, 1] = mouse_y
        
            jacobian = ik.ik_jacobian(parents, angles, lengths)
            shift    = ik.ik_shift(points, target_positions)
            changes  = ik.ik_solve(jacobian, shift)
            points, angles = ik.apply_changes(points, angles, changes, parents, lengths)


        # Update the screen
        draw_screen(screen, BACKGROUND_COLOR, SHAPE_COLOR, points, parents)
        pygame.display.flip()

    # Quit pygame
    pygame.quit()



if __name__ == "__main__":
    main()
