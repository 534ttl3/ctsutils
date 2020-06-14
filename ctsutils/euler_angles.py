import math
import numpy as np

# ---- convention for accessing all 4 pi sr by just two angles, theta and phi ----


def gamma_convention():
    return 0


def beta_convention(theta):
    return -np.pi/2 + theta  # theta \in [0, pi]


def alpha_convention(phi):
    return phi  # phi \in [0, 2*pi]
# ----


def get_zxz_rot(alpha, beta, gamma):
    """
    calculate active rotation R_z(gamma) R_x(beta) R_z(alpha)
    think like this: the application is in the reverse order,
    and the order of arguments is the order of application,
    so, again reversed!
    and the index namings can also be flipped around without change
    """
    return np.einsum("ij,jk",
                     np.einsum("ij,jk", get_R_z(alpha), get_R_x(beta)),
                     get_R_z(gamma))

    # return np.dot(get_R_x(beta), get_R_z(alpha))

    # return np.dot(get_R_z(gamma), np.dot(get_R_x(beta), get_R_z(alpha)))  # this produces the wrong result
    # import ipdb; ipdb.set_trace()  # noqa BREAKPOINT
    # return np.einsum("ij,jk,kl", get_R_z(alpha), get_R_x(beta), get_R_z(gamma))


def get_R_x(angle):
    return np.array([[1, 0, 0],
                     [0, math.cos(angle), -math.sin(angle)],
                     [0, math.sin(angle), math.cos(angle)]])


def get_R_y(angle):
    return np.array([[math.cos(angle), 0, math.sin(angle)],
                     [0, 1, 0],
                     [-math.sin(angle), 0, math.cos(angle)]])


def get_R_z(angle):
    # import ipdb; ipdb.set_trace()  # noqa BREAKPOINT
    return np.array([[math.cos(angle), -math.sin(angle), 0],
                     [math.sin(angle), math.cos(angle), 0],
                     [0, 0, 1]])
