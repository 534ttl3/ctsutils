import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)

x = np.linspace(-3, 3, 51)
y = np.linspace(-2, 2, 41)
y2 = np.linspace(-1, 1, 31)
X, Y, Y2 = np.meshgrid(x, y, y2, indexing="ij")

Z = (1 - X/2 + X**5 + (Y+Y2)**3) * np.exp(-X**2 - (Y+Y2)**2) # calcul du tableau des valeurs de Z

plot = ax.pcolor(X[:, 5, :], Y2[:, 5, :], Z[:, 5, :], shading="auto")

axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])

sfreq = Slider(axfreq, 'Freq', 0., 50., valinit=0., valstep=1.)

def update(val):
    print(val)
    idx = int(val)
    ax.pcolor(X[:, idx, :], Y2[:, idx, :], Z[:, idx, :], shading="auto")

sfreq.on_changed(update)
plt.show()
