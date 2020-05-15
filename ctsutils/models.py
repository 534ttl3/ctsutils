import numpy as np


def lorentzian(x, A, x_0, Gamma):
    # See, Walck, p. 27
    # A: scaling factor
    # Gamma: HWHM
    # x_0: expectation value
    return A * (1./np.pi) * ((Gamma)/(Gamma**2. + (x - x_0)**2.))

def singlegaussian(x, A, mu, sigma):
    return A * np.exp( -(1./2.) * ((x - mu)/sigma)**2.)

def gaussiansum(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        A = params[i]
        mu = params[i+1]
        sigma = params[i+2]
        y = y + np.abs(A) * np.exp( -(1./2.) * ((x - mu)/sigma)**2.)
    return y


# define Model function
def Lfunc(B, x):
    return B[0]*x + B[1]

def polynomial2ndDegreefunc(B, x):
    return B[0]*x**2. + B[1]*x + B[2]

def linear(x, a, b):
    return a * x + b
