import jax
from jax import numpy as jnp, random as jr

import pygame
import pygame.locals

from cotix._bodies import Ball

from cotix._visualizer_tools import *


def test_viz():

    framerate = 60

    viz = Visualizer(100, 6, 6, name="Test Viz")

    b = Ball()
    b_drawer = BallDrawer(b, pygame.Color(255, 0, 0))

    run = True

    # to visualize we will use a discretize t
    t = 0
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
        viz.draw_at_t(t)

        pygame.time.Clock().tick(framerate)