"""reading_and_statistics.py

Contains all functions needed to read lattice data and extract energies from it.
"""

import h5py
import numpy as np
from scipy.optimize import curve_fit


def extract_energy(filename):
    """
    Extracts the energy from a correlation function stored in `filename`. Uses Jackknifing to improve statistics.
    :param filename: Path to the correlation function.
    :return: Best fit parameter `gamma` from fitting ``f(t)=f0*exp(-gamma*t)`` to the correlation function.
    """
    data = read_data(filename)

    data = remove_additional_index(data)

    n_times = len(data[0])

    # Let's do the Jackknifing
    jackknife_means, jackknife_variances = jackknife(data)

    # Try some naive fitting to means and jackknife means
    popt, pcov = curve_fit(exp_decay, range(n_times), jackknife_means)

    # We get the LO energy from fitting an exponential decay to the correlation function
    gamma = popt[1]

    return gamma


def jackknife(data):
    """
    Does a jackknife analysis of a given dataset.
    :param data: The dataset to perform the analysis on. The expected shape is
    (`n_points` = Number of data points per time slice, `n_times` = Number of time slices)
    :return: `jackknife_means` list of length `n_times` containing the means of the dataset,
    `jackknife_variances` list of the corresponding variances.
    """
    n_times = len(data[0])
    n_points = len(data)
    jackknife_means = []
    jackknife_variances = []

    for t in range(n_times):
        # Do the Jackknife for each time step
        tempmeans = []

        for i in range(n_points):
            # calculate mean_i
            sum1 = 0
            for j in range(n_points):
                if i != j:
                    sum1 += data[j][t]

            jackmean = sum1 / (n_points - 1)
            tempmeans.append(jackmean)

        sum2 = 0
        for k in range(n_points):
            # Average over mean_i
            sum2 += tempmeans[k]

        jm = sum2 / n_points
        jackknife_means.append(jm)

        # Get the Jackknife variances
        sum1 = 0
        for l_ in range(n_points):
            sum1 += (tempmeans[l_] - jm) ** 2

        jv = (n_points - 1) * sum1 / n_points
        jackknife_variances.append(jv)

    return jackknife_means, jackknife_variances


def exp_decay(t, x0, gamma):
    return x0 * np.exp(-gamma * t)


def remove_additional_index(data):
    """
    Removes an unused index from the given dataset.
    :param data: Dataset to remove index from.
    :return: list of shape (`n_points`, `n_times`).
    """
    new_data = []
    for i in range(len(data)):
        new_data.append([])
        for j in range(len(data[0])):
            new_data[i].append(data[i][j][1])

    return new_data


def read_data(filepath):
    """
    Reads data from a given file in .h5 format.
    :param filepath: Path to the file to read in .h5 format
    :return: list containing raw data from `filepath`
    """
    f = h5py.File(filepath, 'r')

    # List all groups
    a_group_key = list(f.keys())[0]

    # Get the data
    data = list(f[a_group_key])

    return data
