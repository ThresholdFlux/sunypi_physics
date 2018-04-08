from __future__ import division, unicode_literals
from math import pi, cos, sin, cosh, sinh
from cmath import exp
import logging
import numpy as np
from utilities.find_roots import find_roots
from utilities.rank_nullspace import nullspace
from constants import PLANCK_CONST_Js, J_eV_CONVERSION_JpeV, MASS_ELECTRON_kg


def compute_phase_const(len_a_m, len_b_m, propagation_const, num_periods):
    return 2 * pi * propagation_const / (num_periods * (len_a_m + len_b_m))


def compute_wavenumber(energy, potential=0.0, total_gt_potential=True, mass=MASS_ELECTRON_kg):
    """
    Function to Compute the Wavenumber of a Wavefunction from the Total and Potential Energies

    :param energy: The determinate state of the total energy in joules (eigenvalue of Hamiltonian): E.
    :param potential: The potential energy, which must be constant in this region of the wavefunction: V.
    :param total_gt_potential: Total energy is greater than the potential energy, for this wavefunction. This can be
        True or False. This determines if a factor of -1 is applied under the radical or not, to keep the wavenumber
        real and positive.
    :param mass: The mass of the particle in kg: m.
    :type energy: Number or subclass: float, int, ...
    :type potential: Number or subclass: float, int, ...
    :type total_gt_potential: bool
    :type mass: Number or subclass: float, int, ...
    :return: The wavenumber of the wavefunction in radians per meter.
    :rtype float
    """
    if total_gt_potential:
        sign = 1.0
    else:
        sign = -1.0
    return (sign * 2.0 * mass * (energy - potential))**0.5 / PLANCK_CONST_Js


def compute_eigenvalue_negative(well_depth, len_a_m, len_b_m, start, stop, num, propagation_const, num_periods):
    logging.info('Finding the Negative Eigenvalues (Bound States) of the Kronig Penney Model')
    phase_const = compute_phase_const(len_a_m=len_a_m, len_b_m=len_b_m, propagation_const=propagation_const,
                                      num_periods=num_periods)

    def k(energy):
        return compute_wavenumber(energy=energy, potential=0.0, total_gt_potential=False)

    def q(energy):
        return compute_wavenumber(energy=energy, potential=-well_depth, total_gt_potential=True)

    def eigenvalue_eq_n(energy):
        return (k(energy) ** 2 - q(energy) ** 2) \
               / (2 * k(energy) * q(energy)) \
               * sinh(len_b_m * k(energy)) * sin(len_a_m * q(energy)) \
               + (cosh(len_b_m * k(energy)) * cos(len_a_m * q(energy))) \
               - cos(phase_const * (len_a_m + len_b_m))

    roots = find_roots(func=eigenvalue_eq_n, start=start, stop=stop, num=num)
    return roots, eigenvalue_eq_n


def compute_eigenvalue_positive(well_depth, len_a_m, len_b_m, start, stop, num, propagation_const, num_periods):
    logging.info('Finding the Positive Eigenvalues (Unbound States) of the Kronig Penney Model')
    phase_const = compute_phase_const(len_a_m=len_a_m, len_b_m=len_b_m, propagation_const=propagation_const,
                                      num_periods=num_periods)

    def u(energy):
        return compute_wavenumber(energy=energy, potential=0.0, total_gt_potential=True)

    def w(energy):
        return compute_wavenumber(energy=energy, potential=-well_depth, total_gt_potential=True)

    def eigenvalue_eq_p(energy):
        return -(u(energy) ** 2 + w(energy) ** 2) \
               / (2 * u(energy) * w(energy)) \
               * sin(len_b_m * u(energy)) * sin(len_a_m * w(energy)) \
               + cos(len_b_m * u(energy)) * cos(len_a_m * w(energy)) \
               - cos(phase_const * (len_a_m + len_b_m))
    roots = find_roots(func=eigenvalue_eq_p, start=start, stop=stop, num=num)
    return roots, eigenvalue_eq_p


def find_zero_eigenvalue_depth(len_a_m, len_b_m, start, stop, num, propagation_const, num_periods):
    logging.info('Finding the Well Depths that Lead to an Eigenvalue of Zero in the Kronig Penney Model')
    phase_const = compute_phase_const(len_a_m=len_a_m, len_b_m=len_b_m, propagation_const=propagation_const,
                                      num_periods=num_periods)

    def wave_num_l(energy):
        return compute_wavenumber(energy=energy, potential=0.0, total_gt_potential=True)

    def zero_eigenvalue_eq(energy):
        return cos(len_a_m * wave_num_l(energy))\
               - len_b_m * wave_num_l(energy)/2 * sin(len_a_m * wave_num_l(energy)) \
               - cos(phase_const * (len_a_m + len_b_m))
    roots = find_roots(func=zero_eigenvalue_eq, start=start, stop=stop, num=num)
    return roots, zero_eigenvalue_eq


def normalize_neg_eigenvals(energy, well_depth, len_a_m, len_b_m, propagation_const, num_periods):
    phase_const = compute_phase_const(len_a_m, len_b_m, propagation_const, num_periods)
    ikab = 1j * phase_const * (len_a_m + len_b_m)
    a = len_a_m
    b = len_b_m
    k = compute_wavenumber(energy=energy, potential=0.0, total_gt_potential=False)
    q = compute_wavenumber(energy=energy, potential=-well_depth, total_gt_potential=True)

    m_mat = np.array([
        [1,                       1,                     -1,                        -1],
        [k,                      -k,                     -1j*q,                      1j*q],
        [exp(-b * k + ikab),      exp(b * k + ikab),     -exp(1j * a * q),          -exp(-1j * a * q)],
        [k * exp(-b * k + ikab), -k * exp(b * k + ikab), -1j * q * exp(1j * a * q),  1j * q * exp(-1j * a * q)]])

    x = nullspace(A=m_mat, rtol=1.0e-12)

    print(x)
    return x


def plot_wavefunction_neg(energy, well_depth, len_a_m, len_b_m, x):
    k = compute_wavenumber(energy=energy, potential=0.0, total_gt_potential=False)
    q = compute_wavenumber(energy=energy, potential=-well_depth, total_gt_potential=True)
    x_region1 = np.linspace(-len_b_m, 0.0, 100)
    x_region2 = np.linspace(0.0, len_a_m, 100)
    func1_n = x[0] * np.exp(k * x_region1) + x[1] * np.exp(-k * x_region1)
    func2_n = x[2] * np.exp(1j * q * x_region2) + x[3] * np.exp(-1j * q * x_region2)
    x = np.concatenate((x_region1[:-1], x_region2))
    func = np.concatenate((func1_n[:-1], func2_n))
    area = np.trapz(x=x, y=abs(func)**2)
    print(area)
    from matplotlib import pyplot as plt
    plt.plot(x, func.real)
    plt.show()
    input('whoaaaaaaaaa')


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s: %(asctime)s \n    %(message)s', level=logging.INFO)
    logging.getLogger()

    # V0: Potential well depth (eV)
    depth_eV = 5.0 * 1.6920644164445534e2
    depth_J = depth_eV * J_eV_CONVERSION_JpeV

    # a, b: Crystal lattice distances (m)
    len_a_m = 1e-10
    len_b_m = 1e-10

    # N: Number of periods in crystal before cycling back to other side
    num_periods = 10

    # n: Propagation constant, which determines phase of wavefunction
    propagation_const = 3

    roots_n, eigenvalue_eq_n = compute_eigenvalue_negative(
        well_depth=depth_J, len_a_m=len_a_m, len_b_m=len_b_m, start=-0.9999999*depth_J, stop=-0.0000001*depth_J,
        num=1000000, propagation_const=propagation_const, num_periods=num_periods)

    eig_n = roots_n[0][1]

    x = normalize_neg_eigenvals(energy=eig_n, well_depth=depth_J, len_a_m=len_a_m, len_b_m=len_b_m,
                                propagation_const=propagation_const, num_periods=num_periods)

    plot_wavefunction_neg(energy=eig_n, well_depth=depth_J, len_a_m=len_a_m, len_b_m=len_b_m, x=x)

    roots_0, zero_eigenvalue_eq = find_zero_eigenvalue_depth(
        len_a_m=len_a_m, len_b_m=len_b_m, start=0.75*depth_J, stop=1.25*depth_J, num=1000000,
        propagation_const=propagation_const, num_periods=num_periods)
    roots_p, eigenvalue_eq_p = compute_eigenvalue_positive(
        well_depth=depth_J, len_a_m=len_a_m, len_b_m=len_b_m, start=0.0001*depth_J, stop=2.0*depth_J,
        num=2000000, propagation_const=propagation_const, num_periods=num_periods)

    eigenvalues_p = roots_p[0]
    #from scipy.linalg.eig(a=M)
    # for eig in roots_n:

    # from matplotlib import pyplot as plt
    # x_m = np.linspace(start=-0.9999999*depth_J, stop=-0.0000001*depth_J, num=1000000)
    # eq_33_func = np.vectorize(eigenvalue_eq_neg)
    # plt.plot(x_m, eq_33_func(x_m), label='Equation 33: Eigenvalue Equation')
    # plt.title('Bound States of the Kronig-Penney Model')
    # fig = plt.plot(eigenvalues_n, eq_33_func(eigenvalues_n), linestyle=None, marker='o', label='Negative Eigenvalues')
    # plt.show()
    # print(eigenvalues_n)