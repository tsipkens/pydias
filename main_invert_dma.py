
import numpy as np
import matplotlib.pyplot as plt

# Assuming the custom modules are converted to Python
from autils import autils, props, tools
from odias import kernel, invert

# MAIN: Inversion of mobility distributions.
# Consider a truncated size distribution.

# Generate reconstruction points and mobility setpoints
d = np.logspace(np.log10(10), np.log10(1e3), 500)  # reconstruction points
d_star = np.logspace(np.log10(13.1), np.log10(225), 114)  # mobility setpoints

# Set device properties
prop = props.dma()

# Generate the kernels matrix
A, _, _, _ = kernel.gen_smps(d_star, d, None, prop)

# Parameters for size distribution
mu = 200
s = 1.2
w = 1

# Generate synthetic data
b, Lb, x0 = tools.gen_data(A, d, mu, s, w, d_star)

# Least-squares inversion
print('Running least-squares ...')
x_lsq, _ = invert.lsq(Lb @ A, Lb @ b)
autils.textdone()

# Twomey inversion
print('Running Twomey ...')
xi = invert.get_init(Lb @ A, Lb @ b, d, d_star)
x_two = invert.twomey(Lb @ A, Lb @ b, xi, f_bar=True)
autils.textdone()

# Twomey-Markowski inversion
print('Running Twomey-Markowski ...')
x_twomark = invert.twomark(Lb @ A, Lb @ b, len(xi), xi)
autils.textdone()

# 1st order Tikhonov regularization
print('Running Tikhonov (1st) ...')
lambda_tk1 = 3.8e1/2
x_tk1, sys1, _, _ = invert.tikhonov(Lb @ A, Lb @ b, lambda_tk1, order=1, bc=0)
# Gpo_tk1 = inv(Gpo_inv_tk1)
# e_tk1 = (x_tk1 - x0).T @ Gpo_inv_tk1 @ (x_tk1 - x0)
autils.textdone()

# 2nd order Tikhonov regularization
print('Running Tikhonov (2nd) ...')
lambda_tk2 = 8e2/2
x_tk2, _, _, _ = invert.tikhonov(Lb @ A, Lb @ b, lambda_tk2, order=2, bc=0)
# Gpo_tk2 = inv(Gpo_inv_tk2)
# e_tk2 = (x_tk2 - x0).T @ Gpo_inv_tk2 @ (x_tk2 - x0)
autils.textdone()

# 3rd order Tikhonov regularization
print('Running Tikhonov (3rd) ...')
lambda_tk3 = 5e3/2
x_tk3, _, _, _ = invert.tikhonov(Lb @ A, Lb @ b, lambda_tk3, order=3, bc=0)
# Gpo_tk3 = pinv(Gpo_inv_tk3)
# e_tk3 = (x_tk3 - x0).T @ Gpo_inv_tk3 @ (x_tk3 - x0)
autils.textdone()

# Exponential distance regularization
# print('Running exponential distance ...')
# lambda_ed = 1e1
# ld = np.log10(s_d)
# x_ed, _, _ = invert.exp_dist(Lb @ A, Lb @ b, lambda_ed, ld, d)
# Gpo_ed = inv(Gpo_inv_ed)
# e_ed = (x_ed - x0).T @ Gpo_inv_ed @ (x_ed - x0)
# autils.textdone()

plt.plot(d, x_two, label='Twomey')
plt.plot(d, x_twomark, label='Twomey-Markowski')
plt.plot(d, x_tk1, label='Tikhonov, 1st')
plt.plot(d, x_tk2, label='Tikhonov, 2nd')
plt.plot(d, x_tk3, label='Tikhonov, 3rd')
# plt.plot(da, x_lsq, label='lsq')
plt.plot(d, x0, label='x0', color='k', linestyle='--')
plt.xscale('log')
plt.ylim(0, np.max(x0) * 1.5)
plt.legend()
plt.show()
