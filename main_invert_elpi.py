
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv
from scipy.interpolate import interp1d
from scipy.linalg import inv as np_inv

from autils import autils, props, tools
from tfer import tfer
from odias import invert, kernel

# Main computation

# Reconstruction points and setpoints
# Aerodynamic diameter grids (nm)
d = np.logspace(np.log10(10), np.log10(1e5), 300)

# Properties and calculations
prop = props.elpi()

# ELPI kernels return the matrix and the nominal cut-points (d_star)
A, d_star = tfer.elpi(d, prop)


# Parameters for data generation
mu = [700, 80]
s = [2.0, 1.4]
w = [1, 0.2]

# Data generation and plot
b, Lb, x0 = tools.gen_data(A, d, mu, s, w, d_star)
plt.show()

# Generate data. 
b, Lb, x0 = tools.gen_data(A, d, mu, s, w)
plt.show()

# Least-squares
print('Running least-squares ...')
x_lsq = invert.lsq(Lb @ A, Lb @ b)
autils.textdone()

# Twomey
print('Running Twomey:')
xi = invert.get_init(Lb @ A, Lb @ b, d, d_star)
x_two = invert.twomey(Lb @ A, Lb @ b, xi, show_progress=True)
autils.textdone()

# Twomey-Markowski
print('Running Twomey-Markowski:')
xi = invert.get_init(Lb @ A, Lb @ b, d, d_star)
x_twomark = invert.twomark(Lb @ A, Lb @ b, len(xi), xi)
autils.textdone()

# 1st order Tikhonov
print('Running Tikhonov (1st) ...')
lambda_tk1 = 3.8
x_tk1, _, _, _ = invert.tikhonov(Lb @ A, Lb @ b, lambda_tk1, order=1, bc=0)
# Gpo_tk1 = np_inv(Gpo_inv_tk1)
# e_tk1 = (x_tk1 - x0).T @ Gpo_inv_tk1 @ (x_tk1 - x0)
autils.textdone()

# 2nd order Tikhonov
print('Running Tikhonov (2nd) ...')
lambda_tk2 = 100
x_tk2, _, _, _ = invert.tikhonov(Lb @ A, Lb @ b, lambda_tk2, order=2, bc=0)
# Gpo_tk2 = np_inv(Gpo_inv_tk2)
# e_tk2 = (x_tk2 - x0).T @ Gpo_inv_tk2 @ (x_tk2 - x0)
autils.textdone()



plt.plot(d, x_two, label='Twomey')
plt.plot(d, x_twomark, label='Twomey-Markowski')
plt.plot(d, x_tk1, label='Tikhonov, 1st')
plt.plot(d, x_tk2, label='Tikhonov, 2nd')
# plt.plot(da, x_lsq, label='lsq')
plt.plot(d, x0, label='x0', color='k', linestyle='--')
plt.xscale('log')
plt.ylim(0, 1.5 * np.max(x0))
plt.legend()

# plt.savefig("test.svg", format="svg")
plt.show()

