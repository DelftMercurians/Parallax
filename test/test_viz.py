import jax.numpy as jnp
import pygame
import pygame.locals

from cotix._bodies import Ball
from cotix._visualizer_tools import BallDrawer, Visualizer


def test_viz():
    framerate = 60

    viz = Visualizer(100, 6, 6, name="Test Viz")

    b = Ball()

    b.set_position(b.position + jnp.array([3.0, 3.0]))
    # b.update_transform()

    print(b.position)

    b_drawer = BallDrawer(b, pygame.Color(255, 0, 0))

    viz.add_element(b_drawer)

    run = True

    while run:
        # Iterating over all the events received from
        # pygame.event.get()
        for event in viz.get_events():
            # If the type of the event is quit
            # then setting the run variable to false
            if event.type == pygame.locals.QUIT:
                run = False
        # Draws the surface object to the screen.

        # static ball for now, the properties of the Ball should change over time
        viz.draw()
        # b_drawer.object.set_position(b.position + jnp.array([0.1, 0.1]))
        # print(b_drawer.object.position)
        # print("-------------------")
        pygame.time.Clock().tick(framerate)
