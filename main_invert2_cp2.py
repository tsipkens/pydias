
import numpy as np
import matplotlib.pyplot as plt

from autils import autils, props, tools
from tfer import tfer
from bidias import kernel, invert
from bidias.Grid import Grid, PartialGrid
from bidias.Phantom import Phantom


# Create grid for Phantom and solution domain.
# grid_x = Grid([[0.01,100],[0.01,100]], [80,80])
grid_x = PartialGrid([[0.01,100],[0.01,100]], [80,80], r=[[10,10]], slope=1)
grid_x.type = ['mp', 'mrbc']


# Create phantom.
p = {}
p['dg'] = 0.6714
p['mg'] = 0.6714
p['sd'] = 2.3
p['zet'] = 1.0
p['sm_d'] = 1.3

pha = Phantom('massmob', prop=p)

x = pha.eval(grid_x)
grid_x.plot2d(x)
plt.show()


# Define setpoints and generate a kernel.
prop_pma = props.pma()
prop_pma = autils.massmob_add(prop_pma, 'soot')

mrbc_star = np.logspace(-1.8, 1.8, 120)
mp_star = np.logspace(-1.8, 1.8, 14)

grid_b = Grid(edges=[mp_star, mrbc_star])
grid_b.type = ['mp', 'mrbc']

sp, _ = tfer.get_setpoint(prop_pma, 'm_star', grid_b.elements[:,0] * 1e-18, 'Rm', [3])

A, Ac = kernel.build(grid_x, [['sp2', grid_b.elements[:,1]], ['pma', sp, prop_pma], ['charger']], z=np.arange(0, 4))
# A = np.maximum(A, 0)

grid_x.plot2d(A[810,:])
plt.colorbar()
plt.show()


b0 = A @ x  # noiseless data
b, Lb = tools.get_noise(b0, 1e6)  # add noise to data and get Lb

grid_b.plot2d(b, cmap='inferno')
plt.show()

x1, _, _, _ = invert.tikhonov(Lb @ A, Lb @ b, 4.0, order=1, grid=grid_x)

_, ax = plt.subplots(1, 2)

plt.axes(ax[0])
grid_x.plot2d(x)

plt.axes(ax[1])
grid_x.plot2d(x1)
