
import numpy as np
import matplotlib.pyplot as plt

from autils import autils, props, tools
from tfer import tfer
from bidias import kernel, invert
from bidias.Grid import Grid, PartialGrid
from bidias.Phantom import Phantom


# Create grid for Phantom and solution domain.
grid_x = Grid([[10,1000],[0.01,100]], [80,70])
# grid_x = PartialGrid([[10,1000],[0.01,100]], [80,70], r=[[100,10]], slope=[1/3])
grid_x.type = ['dm', 'mp']


# Create phantom.
p = {}
p['dg'] = 127
p['mg'] = 0.6714
p['sd'] = 1.75
p['zet'] = 2.34
p['sm_d'] = 1.3

pha = Phantom('massmob', prop=p)

x = pha.eval(grid_x)
grid_x.plot2d(x)
plt.show()


# Define setpoints and generate a kernel.
prop_pma = props.pma()
prop_pma = autils.massmob_add(prop_pma, 'soot')

prop_dma = props.dma()

d_star = np.logspace(1.2, 2.8, 60)
m_star = np.logspace(-1.8, 1.8, 14)

grid_b = Grid(edges=[d_star, m_star])
grid_b = grid_b.transpose()

sp, _ = tfer.get_setpoint(prop_pma, 'm_star', grid_b.elements[:,0] * 1e-18, 'Rm', [3])

A, Ac = kernel.build(grid_x, [['dma', grid_b.elements[:,1], prop_dma], ['pma', sp, prop_pma], ['charger']])

grid_x.plot2d(A[510,:])
plt.show()


b0 = A @ x  # noiseless data
b, Lb = tools.get_noise(b0, 1e5)  # add noise to data and get Lb

grid_b.plot2d(b, cmap='inferno')
plt.show()


x1, _, _, _ = invert.tikhonov(Lb @ A, Lb @ b, 2.0, order=1, grid=grid_x)

_, ax = plt.subplots(1, 2)

plt.axes(ax[0])
grid_x.plot2d(x, cmap='viridis')
ax[0].set_box_aspect(1)

plt.axes(ax[1])
grid_x.plot2d(x1, cmap='viridis')
ax[1].set_box_aspect(1)

