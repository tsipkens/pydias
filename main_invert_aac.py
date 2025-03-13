
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm

from autils import autils, props, tools
from tfer import tfer
from odias.invert import tikhonov, lsq
from odias import invert

# Reconstruction points
d = np.logspace(np.log10(10), np.log10(1e3), 500)
d_star = np.logspace(np.log10(13.1), np.log10(700), 114)

# Initialize properties and add mass-mobility relationship for soot
prop = props.aac()
prop = autils.massmob_add(prop, 'soot')

# Calculate transfer function
opts = {'model': 'lt'}
A, _, _ = tfer.aac(d_star, d, prop, opts)

# Generate distribution
mu_d = 120
s_d = 1.7
x0 = norm.pdf(np.log(d), np.log(mu_d), np.log(s_d))

# Add noise
b0 = A @ x0
b, Lb = tools.get_noise(b0, 1e2, 1e-6)

# Plot initial distribution
plt.figure(1)
plt.semilogx(d_star, b.T, '.')
plt.semilogx(d_star, b0.T, color=[0.5, 0.5, 0.5])
plt.show()

# Least-squares inversion
print('Running least-squares ...')
x_lsq, _ = invert.lsq(Lb @ A, Lb @ b)
autils.textdone()

# Twomey inversion
print('Running Twomey ...')
xi = invert.get_init(Lb * A, Lb * b, d, d_star)
x_two = invert.twomey(Lb * A, Lb * b, xi, f_bar=True)
autils.textdone()

# Twomey-Markowski inversion
print('Running Twomey-Markowski ...')
x_twomark = invert.twomark(Lb @ A, Lb @ b, len(xi), xi)
autils.textdone()

# 1st order Tikhonov inversion
print('Running Tikhonov (1st) ...')
lambda_tk1 = 8e1
x_tk1, _, _, _ = invert.tikhonov(Lb @ A, Lb @ b, lambda_tk1, order=1, bc=0)
# Gpo_tk1 = np.linalg.inv(Gpo_inv_tk1)
# e_tk1 = (x_tk1 - x0).T @ Gpo_inv_tk1 @ (x_tk1 - x0)
autils.textdone()

# 2nd order Tikhonov inversion
print('Running Tikhonov (2nd) ...')
lambda_tk2 = 2e3
x_tk2, _, _, _ = invert.tikhonov(Lb @ A, Lb @ b, lambda_tk2, order=2, bc=0)
# Gpo_tk2 = np.linalg.inv(Gpo_inv_tk2)
# e_tk2 = (x_tk2 - x0).T @ Gpo_inv_tk2 @ (x_tk2 - x0)
autils.textdone()

# Two-step 2nd order Tikhonov inversion
print('Running Tikhonov (2nd, two-step) ...')
lambda_tk22 = 3e3
x_tk22, _, _, _ = invert.tikhonov(Lb @ A, Lb @ b, lambda_tk22, order=2, bc=0)
# Gpo_tk22 = np.linalg.inv(Gpo_inv_tk22)
# e_tk22 = (x_tk22 - x0).T @ Gpo_inv_tk2 @ (x_tk22 - x0)
autils.textdone()

# Exponential distance inversion
# print('Running exponential distance ...')
# lambda_ed = 1e1
# ld = np.log10(s_d)
# x_ed, _, _, Gpo_inv_ed = exp_dist(Lb @ A, Lb @ b, lambda_ed, ld, da)
# Gpo_ed = np.linalg.inv(Gpo_inv_ed)
# e_ed = (x_ed - x0).T @ Gpo_inv_ed @ (x_ed - x0)
# autils.textdone()

plt.plot(d, x_two, label='Twomey')
plt.plot(d, x_twomark, label='Twomey-Markowski')
plt.plot(d, x_tk1, label='Tikhonov, 1st')
plt.plot(d, x_tk2, label='Tikhonov, 2nd')
plt.plot(d, x_tk22, label='Tikhonov, double 2nd')
# plt.plot(da, x_lsq, label='lsq')
plt.plot(d, x0, label='x0', color='k', linestyle='--')
plt.xscale('log')
plt.legend()
plt.show()
