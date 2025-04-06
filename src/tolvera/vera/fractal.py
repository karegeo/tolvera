"""
Fractal (Julia set) visualisation based on Taichi example

TODO: replace time with circular periodic phase. It will improve the long-term behaviour
TODO: colormaps
"""

import taichi as ti
from ..pixels import Pixels

@ti.data_oriented
class Fractal:
    """
    This class generates a dynamic Julia set fractal using a Taichi kernel.
    The fractal evolves over time, creating complex and organic patterns that
    can be used as a visual background or as an input field for agent-based behaviours
    such as flocking or slime trails.

    The Julia set is computed for each pixel of the simulation window, where the iteration
    count determines the pixel's colour. Time is used to modulate the complex parameter `c`,
    allowing the fractal to animate smoothly.

    This vera does not operate on discrete cells like the Game of Life, but instead draws directly
    to a high-resolution floating-point `Pixels` buffer, which can be stamped or blended into
    the global canvas. It can also serve as a dynamic trail or field that other veras can sense or modify.

    Iimplementation is fully based on
    https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/fractal.py
    """

    def __init__(self, tv, **kwargs) -> None:
        """
        Initialise the Fractal vera.

        This sets up an internal time field and a drawing buffer using Tölvera's `Pixels`.
        Speed controls how fast the fractal animates by adjusting the time increment per frame.
        The pixel buffer is rendered using the Julia set formula and can be used in interaction
        with other veras (e.g., slime, flock).

        Args:
            tv (Tolvera): A Tolvera instance.
            speed (float, optional): Speed of fractal time evolution. Defaults to 0.03.
        """

        self.tv = tv
        self.kwargs = kwargs
        self.time = ti.field(dtype=ti.f32, shape=())
        self.time[None] = 0.0
        self.speed = ti.field(dtype=ti.f32, shape=())
        self.speed[None] = kwargs.get("speed", 0.03)
        self.px = Pixels(self.tv)

    @ti.func
    def complex_sqr(self, z):
        return ti.Vector([z[0] ** 2 - z[1] ** 2, 2.0 * z[0] * z[1]])

    @ti.kernel
    def draw(self):
        t = self.time[None]
        for i, j in ti.ndrange(self.tv.x, self.tv.y): # Parallelized over all pixels
            uv = ti.Vector([i / self.tv.x, j / self.tv.y]) * 2 - 1
            z = uv * ti.Vector([1.5, 1.0])
            c = ti.Vector([-0.8, ti.cos(t) * 0.2])
            iterations = 0
            while z.norm() < 20 and iterations < 50:
                z = self.complex_sqr(z) + c
                iterations += 1
            val = 1.0 - iterations * 0.02
            # colormap
            self.px.px.rgba[i, j] = ti.Vector([val, 0.3 * val, 0.8 * val, 1.0])
        
    def set_speed(self, speed: ti.f32):
        self.speed[None] = speed

    def step(self):
        self.time[None] += self.speed[None]
        self.draw()

    def __call__(self):
        self.step()
        return self.px
