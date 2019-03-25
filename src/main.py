"""main.py

Uses the Jackknife and Bootstrap methods to evaluate lattice data.
"""

import numpy as np
from math import sqrt, floor
import matplotlib.pyplot as plt

import reading_and_statistics

POINTS_IN_SPHERE = None
LAMBDA_CUTOFF = 10


def main():
    global POINTS_IN_SPHERE
    filename_pion = "../data/piondata.h5"
    filename_nn = "../data/nndata.h5"

    # gamma = reading_and_statistics.extract_energy(filename_nn)

    # Testing luscher_s
    # xdata = np.linspace(-3.0, -0.1)
    # ydata = luscher_s(xdata, 10)

    print(luscher_s(0.1, 500))


def luscher_s(eta, lambda_cutoff):
    """
    Implementation of the function S as it appears in the Lüscher formula.
    :param eta: Real valued function argument.
    :param lambda_cutoff: Cutoff of the theory. Send lambda to infinity for the continuum limit.
    :return: Returns the value for S(eta) at the given cutoff
    """
    global POINTS_IN_SPHERE
    # TODO: Check for numerical stability.
    s = -4 * np.pi * lambda_cutoff

    # Start by generating all momenta `j` with ``abs(j) < lambda`` when calling this function for the first time
    if POINTS_IN_SPHERE is None:
        POINTS_IN_SPHERE = generate_points_in_sphere(lambda_cutoff)

    # Do the sum
    for (jx, jy, jz) in POINTS_IN_SPHERE:
        s += 1.0 / (np.sqrt(jx ** 2 + jy ** 2 + jz ** 2) - eta)

    return s


def generate_points_in_sphere(radius):
    points_in_sphere = []
    radius_int = int(floor(radius))
    for jx in range(-radius_int, radius_int + 1):
        for jy in range(-radius_int, radius_int + 1):
            for jz in range(-radius_int, radius_int + 1):
                if sqrt(jx ** 2 + jy ** 2 + jz ** 2) < radius:
                    points_in_sphere.append((jx, jy, jz))

    return points_in_sphere


if __name__ == "__main__":
    main()
