#!/usr/bin/env python
# -*- coding: utf-8 -*-

__appname__ = "TheThreeBodyProblem"
__version__ = "0.1"
__author__ = "Nicolas Hennion <nicolas@nicolargo.com>"
__license__ = "MIT"

import numpy as np
from scipy.constants import G
import matplotlib.pyplot as plt
from matplotlib import animation
from copy import deepcopy


class NBodyProblem:

    def __init__(self, bodies, start, end, iterations):
        self.bodies = bodies
        self.start = start
        self.end = end
        self.iterations = iterations
        self._steps = np.linspace(self.start, self.end, self.iterations)
        self._delta = self.end - self.start / self.iterations
        self.bodies_over_time = [deepcopy(bodies)]

    def _acceleration(self, body):
        a = np.zeros(3)
        for other in self.bodies:
            if other != body:
                r = other.position - body.position
                a += (G * (6e24 * (365*24*60*60)**2) / (1.496e11)**3) * other.mass * r / np.linalg.norm(r)**3
        return a

    def acceleration(self):
        """Compute bodies acceleration
        """
        for b in self.bodies:
            b.acceleration = self._acceleration(b)

    def compute(self):
        """Compute the NBodyProblem
        """
        for step in self._steps:
            self.acceleration()
            for b in self.bodies:
                b.position += b.velocity * self._delta
                b.velocity += b.acceleration * self._delta
            self.bodies_over_time.append(deepcopy(self.bodies))

    def display(self):
        for i, bodies in enumerate(self.bodies_over_time):
            for j, b in enumerate(bodies):
                print(f"Step {i}: Body {j} mass={b.mass}, position={b.position}, velocity={b.velocity}, acceleration={b.acceleration}")


class Body:

    def __init__(self, mass, position, velocity):
        """Init a body

        Args:
            mass (int): Body mass in kg
            position (np.array): 3D position of the body
            velocity (np.array): 3D velocity of the body
        """
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.acceleration = np.zeros(3)


def random_body():
    """Generate a random body

    Returns:
        Body: A random body
    """
    mass = np.random.uniform(low=6e24, high=2e30) / 6e24
    position = np.random.uniform(-10, 10, 3)
    velocity = np.random.uniform(-3, 3, 3)
    return Body(mass, position, velocity)


def main():
    # 3 bodies init
    n = 3
    bodies = [random_body() for i in range(n)]
    bpb = NBodyProblem(bodies, 0, 10, 500)

    # Compute
    bpb.compute()

    # Display
    # bpb.display()
    plt.style.use('dark_background')
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    ax.zaxis.set_rotate_label(False)
    ax.grid(False)
    # ax.set_xlim3d(np.min(x), np.max(x))
    # ax.set_ylim3d(np.min(y), np.max(y))
    # ax.set_zlim3d(np.min(z), np.max(z))
    ax.set_xlabel("x (UA)")
    ax.set_ylabel("y (UA)")
    ax.set_zlabel("z (UA)")
    for p in bpb.bodies_over_time:
        for b in p:
            plt.plot(b.position[0], b.position[1], b.position[2], 'ro')
    plt.show()


if __name__ == "__main__":
    main()
