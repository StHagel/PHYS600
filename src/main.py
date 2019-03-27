"""main.py

Uses the Jackknife and Bootstrap methods to evaluate lattice data.
"""
# TODO: DivisionByZeroError handling.

import numpy as np
from math import sqrt, floor, fabs, atan
import matplotlib.pyplot as plt

import reading_and_statistics

POINTS_IN_SPHERE = None
LAMBDA_CUTOFF = 10


def main():
    global POINTS_IN_SPHERE
    # filename_pion = "../data/piondata.h5"
    filename_nn = "../data/nndata.h5"

    gamma = reading_and_statistics.extract_energy(filename_nn)

    mass_proton = 0.9382720813
    momentum = sqrt(gamma * mass_proton)
    x_length = np.linspace(1.0, 15.0, 500)
    y_phase = phaseshift_delta(momentum, x_length, 200)

    plt.grid(True)
    plt.xlabel("Box size [1/GeV]")
    plt.ylabel("Phase shift")
    plt.plot(x_length, y_phase, 'bo')
    plt.show()


def phaseshift_delta(momentum_p, lenght_l, lambda_cutoff=100):
    """
    Evaluate the continuum phase shift delta(p) using the Lüscher formula.
    :param momentum_p: Momentum to evaluate delta at.
    :param lenght_l: Size of the box.
    :param lambda_cutoff: The cutoff to evaluate S(eta) at. Physical results appear at lambda -> inf.
    :return: Continuum phase shift delta(p).
    """

    return np.arctan(momentum_p * lenght_l * np.pi / (luscher_s((lenght_l * momentum_p / (2 * np.pi)) ** 2,
                                                                lambda_cutoff)))


def test_luscher_s():
    """
    This function can be used to test the S(eta) function.
    :return: None
    """
    xdata = np.linspace(-3.0, 2.99, 1000)
    ydata = luscher_s(xdata, 100)

    plt.grid(True)
    plt.xlim((-3, 3))
    plt.ylim((-100, 100))

    plt.plot(xdata, ydata, 'bo')
    plt.show()
    return None


def luscher_s(eta, lambda_cutoff):
    """
    Implementation of the function S as it appears in the Lüscher formula.
    :param eta: Real valued function argument.
    :param lambda_cutoff: Cutoff of the theory. Send lambda to infinity for the continuum limit.
    :return: Returns the value for S(eta) at the given cutoff
    """
    global POINTS_IN_SPHERE
    # TODO: Check for numerical stability.
    s = -4.0 * np.pi * lambda_cutoff

    # Start by generating all momenta `j` with ``abs(j) < lambda`` when calling this function for the first time
    if POINTS_IN_SPHERE is None:
        POINTS_IN_SPHERE = generate_points_in_sphere(lambda_cutoff)

    # Do the sum
    for (jx, jy, jz) in POINTS_IN_SPHERE:
        # Actually we are only summing over the first octand of the sphere and multiply the result by 8.
        # To avoid double counting we have to handle the origin, the edges and the faces separately
        if (jx, jy, jz) == (0, 0, 0):
            s += 1.0 / (jx ** 2 + jy ** 2 + jz ** 2 - eta)
        elif (jx, jy, jz).count(0) == 2:
            s += 2.0 / (jx ** 2 + jy ** 2 + jz ** 2 - eta)
        elif 0 in (jx, jy, jz):
            s += 4.0 / (jx ** 2 + jy ** 2 + jz ** 2 - eta)
        else:
            s += 8.0 / (jx ** 2 + jy ** 2 + jz ** 2 - eta)

    return s


def generate_points_in_sphere(radius):
    points_in_sphere = []
    radius_int = int(floor(radius))
    for jx in range(radius_int + 1):
        for jy in range(radius_int + 1):
            for jz in range(radius_int + 1):
                if sqrt(jx ** 2 + jy ** 2 + jz ** 2) < radius:
                    points_in_sphere.append((jx, jy, jz))

    return points_in_sphere


if __name__ == "__main__":
    main()
