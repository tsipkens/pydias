
import numpy as np
import matplotlib.pyplot as plt

from autils import autils, props, tools
from tfer import tfer

from bidias import kernel, invert, tools
from bidias.Grid import Grid, PartialGrid
from bidias.Phantom import Phantom


# Create grid for Phantom and solution domain.
grid_x = Grid([[10,1000],[700,1600]], [80,70])
# grid_x = PartialGrid([[10,1000],[0.01,100]], [80,70], r=[[100,10]], slope=[1/3])
grid_x = Grid([[10,1000],[10,1000]], [80,70])
grid_x.type = ['dm', 'da']


# Create phantom.
p = {
    'mu1': 127,
    'mu2': 130,   # 1000
    's1': 1.75,
    'pow': 1,     # 0.001
    's2|1': 1.2   # 1.05
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

d_star = np.logspace(1.2, 2.8, 130)
da_star = np.logspace(1.1, 3.2, 25)

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
pha.overlay(color='w')
plt.show()


x1, _, _, _ = invert.tikhonov(Lb @ A, Lb @ b, 5.0, order=2, grid=grid_x)

_, ax = plt.subplots(1, 2)

plt.axes(ax[0])
grid_x.plot2d(x)
ax[0].set_box_aspect(1)

plt.axes(ax[1])
grid_x.plot2d(x1)
ax[1].set_box_aspect(1)

plt.show()


_, ax = plt.subplots(1, 2)

fun = lambda dm, da: autils.dm_da2rhoeff(dm*1e-9, da*1e-9)
span_y = [200, 4000]

# fun = lambda dm, da: autils.dm_da2mp(dm*1e-9, da*1e-9) * 1e18
# span_y = [0.01, 100]

plt.axes(ax[0])
y, grid_y, T = tools.x2y(x, grid_x, fun, span_y=span_y)
grid_y.plot2d(y, cmap='mako_r')
ax[0].set_box_aspect(1)

plt.axes(ax[1])
y, grid_y, T = tools.x2y(x1, grid_x, fun, span_y=span_y)
grid_y.plot2d(y, cmap='mako_r')
ax[1].set_box_aspect(1)

plt.show()
