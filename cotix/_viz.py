import equinox as eqx
import jax
import pygame


class PyPainter:
    def __init__(self):
        pass

    def setup(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        self.screen.fill((0, 0, 0))
        pygame.display.flip()
        self.clock = pygame.time.Clock()

    def draw_circle(self, center, radius, color):
        radius = self._scale_size(radius)
        center = self._scale_vector(center)

        pygame.draw.circle(self.screen, color, center, radius)

    def draw_line(self, start, end, color):
        pygame.draw.line(
            self.screen, color, self._scale_vector(start), self._scale_vector(end)
        )

    def draw_polygon(self, vertices, color):
        pygame.draw.polygon(
            self.screen, color, [self._scale_vector(x) for x in vertices]
        )

    def _scale_vector(self, vector):
        # maps [-8, 8]x[-6, 6] to [0, 800]x[0, 600]
        return (
            int((vector[0] + 8) / 16 * 800),
            int((-vector[1] + 6) / 12 * 600),  # y axis is flipped
        )

    def _scale_size(self, size):
        return int(size * 40)

    def next(self):
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        self.clock.tick(240)
        self.screen.fill((0, 0, 0))


_painter = PyPainter()


class Painter(eqx.Module, strict=True):
    """Wrapper of PyPainter, that allows to draw shapes on the screen while
    being inside a jitted region"""

    def __init__(self):
        jax.debug.callback(_painter.setup)

    def draw_circle(self, center, radius, color):
        jax.debug.callback(_painter.draw_circle, center, radius, color)

    def draw_line(self, start, end, color):
        jax.debug.callback(_painter.draw_line, start, end, color)

    def draw_polygon(self, vertices, color):
        jax.debug.callback(_painter.draw_polygon, vertices, color)

    def next(self):
        jax.debug.callback(_painter.next)

    def draw(self, state):
        for body in state:
            body.draw(self)
        self.next()
