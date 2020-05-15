import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path, PurePath

import matplotlib.gridspec as gridspec

from scipy.optimize import curve_fit

class LorentzPeak:
    def __init__(self):
        self.A = None
        self.sigA = None
        self.x_0 = None
        self.sigx_0 = None
        self.Gamma = None
        self.sigGamma = None
        self.popt = None
        self.pcov = None

    def fit_lorentz_peak(self, x, y, xi=None, xf=None, show=False, annotate=False, plotinexistingax=None, plot_original_data_points=True):

        if xi is None:
            xi = min(x)

        if xf is None:
            xf = max(x)

        fit_indices = np.where((xi < x) & (x < xf))
        x_fit = x[fit_indices]
        y_fit = y[fit_indices]

        from models import lorentzian
        self.popt, self.pcov = curve_fit(lorentzian, xdata=x_fit, ydata=y_fit)

        self.A = self.popt[0]
        self.sigA = np.sqrt(np.diag(self.pcov)[0])
        self.x_0 = self.popt[1]
        self.sigx_0 = np.sqrt(np.diag(self.pcov)[1])
        self.Gamma = self.popt[2]
        self.sigGamma = np.sqrt(np.diag(self.pcov)[2])

        ax = None
        if plotinexistingax is None:
            fig = plt.figure(tight_layout=True)
            gs = gridspec.GridSpec(1, 1)
            ax = fig.add_subplot(gs[:, :])

            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$y$')
        else:
            ax = plotinexistingax

        if annotate:
            ax.plot(x,     y, '-', markersize=5,
                    label=r"Datenpunkte", alpha=0.5)
        else:
            if plot_original_data_points:
                ax.plot(x,     y, '-', markersize=1, alpha=0.5)

        ax.plot(x_fit, y_fit, '+', markersize=7.5,
                label=r"Datenpunkte verwendet für Fit", alpha=0.9)

        ax.plot(x_fit, lorentzian(x_fit, *self.popt), 'r:',
                label=r'Lorentz-Fit')

        print('----------------')
        print('fit_lorentz_peak(): ')
        print(r'Fit: ', "\n",
              '$A = {0} \pm {1}$, '.format(
                  self.popt[0], np.sqrt(np.diag(self.pcov)[0])), "\n",
              '$x_0 = {0} \pm {1}$, '.format(
                  self.popt[1], np.sqrt(np.diag(self.pcov)[1])), "\n",
              '$\Gamma = {0} \pm {1}$, '.format(self.popt[2], np.sqrt(np.diag(self.pcov)[2])), "\n")
        print('----------------')

        if plotinexistingax is None:
            ax.set_xlim(np.mean(x_fit) - np.abs(min(x_fit) - max(x_fit)),
                        np.mean(x_fit) + np.abs(min(x_fit) - max(x_fit)))

        if plotinexistingax is None and annotate is False:
            ax.axvline(self.popt[1])

        if annotate is True:
            plt.legend()
            ax.set_xlabel(r'$t\,/\,\si{s}$')
            ax.set_ylabel(r'$U_{\mathrm{pd, FPI}}\,/\,\si{mV}$')

        # plt.show()
        if plotinexistingax is None:
            if annotate is False:
                plt.close()


class GaussPeak: 
    def __init__(self):
        self.A        = None
        self.sigA     = None
        self.mu      = None
        self.sigmu   = None
        self.sig   = None
        self.sigsig   = None
        self.FWHM    = None
        self.sigFWHM = None

        self.popt     = None
        self.pcov     = None

    def assign_gaussian_data(self, A, sigA, mu, sigmu, sig, sigsig):
        self.A        = A
        self.sigA     = sigA
        self.mu       = mu
        self.sigmu    = sigmu
        self.sig      = sig
        self.sigsig   = sigsig

        self.calculate_FWHM()

    def calculate_FWHM(self): 
        self.FWHM = 2.* np.sqrt(2. * np.log(2.)) * self.sig
        self.sigFWHM = 2.* np.sqrt(2. * np.log(2.)) * self.sigsig


class BroadDopplerPeak(GaussPeak): 
    def __init__(self):
        super.__init__()
        self.nu_0 = None
        self.signu_0 = None
        self.transitions

    def calculate_FWHM(self): 
        super.calculate_FWHM()
        self.nu_0 = self.FWHM
        self.signu_0 = self.sigFWHM

