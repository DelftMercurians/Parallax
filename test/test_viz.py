import jax.numpy as jnp
import pygame
import pygame.locals

from cotix._bodies import Ball
from cotix._visualizer_tools import BallDrawer, Visualizer


def test_viz():
    framerate = 60

    viz = Visualizer(100, 6, 6, name="Test Viz")

    b = Ball()

    b = b.set_position(b.position + jnp.array([3.0, 3.0]))

    print(b.position)

    b_drawer = BallDrawer(b, pygame.Color(255, 0, 0))

    run = True
    framecount = 0
    inc = jnp.array([0.1, 0.1])

    while run:
        # Iterating over all the events received from
        # pygame.event.get()
        for event in viz.get_events():
            # If the type of the event is quit
            # then setting the run variable to false
            if event.type == pygame.locals.QUIT:
                run = False
        # Draws the surface object to the screen.

        viz.draw([b_drawer])

        pygame.time.Clock().tick(framerate)

        framecount += 1

        # update the ball
        b = b.set_position(b.position + inc)
        # update the drawer with the ball
        b_drawer.object = b

        if framecount == framerate:
            run = False
