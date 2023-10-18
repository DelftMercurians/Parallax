import abc

import jax.numpy as jnp
import pygame
import pygame.locals

from ._bodies import Ball


class AbstractDrawer:
    @abc.abstractmethod
    def draw(self, window, pix2meter):
        """
        Draw the object (e.g. a body) in the selected pygame window at a selected
        moment of time.
        This will typically call pygame.draw() methods based on the properties of
        the object
        """

        raise NotImplementedError


class BallDrawer(AbstractDrawer):
    object: Ball
    color: pygame.Color
    # the list of visualization properties will change depending on the object to draw

    def __init__(self, obj: Ball, c: pygame.Color):
        self.object = obj
        self.color = c

    def draw(self, window, pix2meter):
        center_x, center_y = self.object.position[0], self.object.position[1]
        # retrieve the first coord of the global support as it should be equal
        # to the radius
        # for more complex shapes we will need more info (get 360 points around
        # a circle with the global_support?)
        radius = self.object.shape.get_global_support(jnp.array([1.0, 0.0]))[0]

        # convert meters to pixels with scale
        center_x *= pix2meter
        center_y *= pix2meter
        radius *= pix2meter

        pygame.draw.circle(
            window, self.color, (int(center_x), int(center_y)), int(radius)
        )


class Visualizer:
    def __init__(
        self, pix2meter_ratio, width_m, height_m, name="Cotix Visualizer"
    ) -> None:
        pygame.init()
        pygame.display.set_caption(name)

        self.window = pygame.display.set_mode(
            (width_m * pix2meter_ratio, height_m * pix2meter_ratio)
        )

        self.pix2meter = pix2meter_ratio

        # Fill the screen with white color
        self.window.fill((255, 255, 255))

    # return the list of events that happened since the last update as pygame
    def get_events(self):
        return pygame.event.get()

    def clear(self, c=(255, 255, 255)):
        # fills the screen with a plain color c, deleting the view we had previously
        self.window.fill(c)

    def draw(self, elements):
        self.clear()

        for e in elements:
            e.draw(self.window, self.pix2meter)

        pygame.display.flip()
