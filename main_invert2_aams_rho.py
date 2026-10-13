
import numpy as np
import matplotlib.pyplot as plt

from autils import autils, props, tools
from tfer import tfer

import bidias.tools
from bidias import kernel, invert
from bidias.Grid import Grid, PartialGrid
from bidias.Phantom import Phantom


# Create grid for Phantom and solution domain.
grid_x = Grid([[10,1000],[700,1600]], [80,70])
grid_x.type = ['dm', 'rho']


# Create phantom.
p = {
    'mu1': 127,
    'mu2': 1000,
    's1': 1.75,
    'pow': 0.001,
    's2|1': 1.1
}

pha = Phantom(p=p)

x = pha.eval(grid_x)
grid_x.plot2d(x)
plt.show()


# Define setpoints and generate a kernel.
prop_pma = props.pma()
prop_pma = autils.massmob_add(prop_pma, 'water')

prop_dma = props.dma()

prop_aac = props.aac()

d_star = np.logspace(1.2, 2.8, 60)
da_star = np.logspace(1.1, 3.2, 14)

grid_b = Grid(edges=[d_star, da_star])
grid_b.type = ['dm_star', 'da_star']
grid_b = grid_b.transpose()

A, Ac = kernel.build(grid_x, [['dma', grid_b.elements[:,1], prop_dma], 
                              ['aac', grid_b.elements[:,0], prop_aac, {}], 
                              ['charger']])


b0 = A @ x  # noiseless data
# b, Lb = tools.get_noise(b0, 1e5)  # add noise to data and get Lb
b = b0
Lb = np.eye(len(b)) * 1e7

grid_b.plot2d(b, cmap='inferno')
plt.show()


x1, _, _, _ = invert.tikhonov(Lb @ A, Lb @ b, 0.1, order=2, grid=grid_x)

_, ax = plt.subplots(1, 2)

plt.axes(ax[0])
grid_x.plot2d(x)
ax[0].set_box_aspect(1)

plt.axes(ax[1])
grid_x.plot2d(x1)
ax[1].set_box_aspect(1)

plt.show()


_, ax = plt.subplots(1, 2)

plt.axes(ax[0])
y, grid_y, T = bidias.tools.x2y(x, grid_x, lambda a, b: (np.pi/6) * (a*1e-9) * b**3, span_y=[5, 1000])
grid_y.plot2d(y, cmap='mako_r')
ax[0].set_box_aspect(1)

plt.axes(ax[1])
y, grid_y, T = bidias.tools.x2y(x1, grid_x, lambda a, b: (np.pi/6) * (a*1e-9) * b**3, span_y=[5, 1000])
grid_y.plot2d(y, cmap='mako_r')
ax[1].set_box_aspect(1)

plt.show()
