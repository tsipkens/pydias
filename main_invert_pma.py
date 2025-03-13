
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
m = np.logspace(-3, 2, 500)
m_star = np.logspace(-3, 2, 80)

# Properties and calculations
prop = props.pma()
prop = autils.massmob_add(prop, 'soot')
d = (m * 1e-18 / prop['m0']) ** (1 / prop['Dm'])

# kernels generation
sp, _ = tfer.get_setpoint(prop, 'm_star', m_star * 1e-18, 'Rm', [3])
Af, _, _, _, _ = kernel.gen_pma(sp, m, d, np.arange(101), prop, None, model='fuchs')
Ab, _, _, _, _ = kernel.gen_pma(sp, m, d, np.arange(4), prop)

# Parameters for data generation
mu = [1, 0.1]
s = [2.5, 1.9]
w = [1, 0.5]

# Data generation and plot
b, Lb, x0 = tools.gen_data(Af, m, mu, s, w, m_star)
b, Lb, x0 = tools.gen_data(Ab, m, mu, s, w, m_star)
plt.show()

A = Af  # Choose which kernels to use
b, Lb, x0 = tools.gen_data(A, m, mu, s, w)
plt.show()

# Least-squares
print('Running least-squares ...')
x_lsq, _ = invert.lsq(Lb @ A, Lb @ b)
autils.textdone()

# Twomey
print('Running Twomey:')
xi = invert.get_init(Lb @ A, Lb @ b, m, m_star)
x_two = invert.twomey(Lb @ A, Lb @ b, xi, f_bar=True)
autils.textdone()

# Twomey-Markowski
print('Running Twomey-Markowski:')
xi = invert.get_init(Lb @ A, Lb @ b, m, m_star)
x_twomark = invert.twomark(Lb @ A, Lb @ b, len(xi), xi)
autils.textdone()

# 1st order Tikhonov
print('Running Tikhonov (1st) ...')
lambda_tk1 = 38
x_tk1, _, _, _ = invert.tikhonov(Lb @ A, Lb @ b, lambda_tk1, order=1, bc=0)
# Gpo_tk1 = np_inv(Gpo_inv_tk1)
# e_tk1 = (x_tk1 - x0).T @ Gpo_inv_tk1 @ (x_tk1 - x0)
autils.textdone()

# 2nd order Tikhonov
print('Running Tikhonov (2nd) ...')
lambda_tk2 = 1000
x_tk2, _, _, _ = invert.tikhonov(Lb @ A, Lb @ b, lambda_tk2, order=2, bc=0)
# Gpo_tk2 = np_inv(Gpo_inv_tk2)
# e_tk2 = (x_tk2 - x0).T @ Gpo_inv_tk2 @ (x_tk2 - x0)
autils.textdone()

# Two-step 2nd order Tikhonov
print('Running Tikhonov (2nd, two-step) ...')
lambda_tk2 = 3000
x_tk22, _, _, _ = invert.tikhonov(Lb @ A, Lb @ b, lambda_tk2, order=[2, 2], bc=0)
# Gpo_tk22 = np_inv(Gpo_inv_tk22)
# e_tk22 = (x_tk22 - x0).T @ Gpo_inv_tk2 @ (x_tk22 - x0)
autils.textdone()

# Exponential distance
# print('Running exponential distance ...')
# lambda_ed = 20
# ld = np.log10(s[0])
# Lpr_ed = exp_dist_lpr(ld, m, 0)  # Generate prior covariance
# x_ed, _, _, Gpo_inv_ed = exp_dist(Lb @ A, Lb @ b, lambda_ed, Lpr_ed)
# Gpo_ed = np_inv(Gpo_inv_ed)
# e_ed = (x_ed - x0).T @ Gpo_inv_ed @ (x_ed - x0)
# print('Done.')


plt.plot(d, x_two, label='Twomey')
plt.plot(d, x_twomark, label='Twomey-Markowski')
plt.plot(d, x_tk1, label='Tikhonov, 1st')
plt.plot(d, x_tk2, label='Tikhonov, 2nd')
plt.plot(d, x_tk22, label='Tikhonov, double 2nd')
# plt.plot(da, x_lsq, label='lsq')
plt.plot(d, x0, label='x0', color='k', linestyle='--')
plt.xscale('log')
plt.ylim(0, 1.5 * np.max(x0))
plt.legend()

# plt.savefig("test.svg", format="svg")
plt.show()

