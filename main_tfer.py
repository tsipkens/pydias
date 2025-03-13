
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from autils import autils, props
from tfer import tfer

cmap_name = 'turbo'


#== DMA =================================================================#
d = np.logspace(np.log10(5), np.log10(3e3), 500)
d_star = np.logspace(np.log10(7), np.log10(800), 5)
z = np.arange(3) + 1

prop = props.dma()  # Replace with actual function or parameters
# Adjust properties if needed
# prop.Qa = np.ones_like(d_star) * prop.Qa
# prop.Qa[-3:] = 2.5e-5
# prop.Qc = np.ones_like(d_star) * prop.Qc
# prop.Qc[-3:] = 2.5e-4
# prop.Qs = prop.Qa
# prop.Qm = prop.Qc

Adma_z, _, prop, _ = tfer.dma(d_star, d, z, prop)
Adma = np.sum(Adma_z, axis=2)

Admat = tfer.tri(d_star, d, prop['Rd'])

plt.figure(figsize=(10, 6))
cm = plt.get_cmap(cmap_name, np.size(d_star) + 1).colors
for ii in range(np.size(d_star)):
    plt.semilogx(d, Adma[ii,:], color=cm[ii])
plt.semilogx(d.T, Admat.T, '--k')
plt.xlim(d[0], d[-1])
plt.xlabel('d_m [nm]')
plt.title('DMA')
plt.show()


#== PMA =================================================================#
m = np.logspace(-3, 3, 2000)
m_star = np.logspace(-2.5, 2, 5)
z = np.arange(5)

prop = props.pma()  # Replace with actual function or parameters
prop = autils.massmob_add(prop, 'soot')
d = (m * 1e-18 / prop['m0']) ** (1 / prop['Dm'])

sp, _ = tfer.get_setpoint(prop, 'm_star', m_star * 1e-18, 'Rm', [3])  # Replace with actual function
Af_z, _ = tfer.pma(sp, m, d, z, prop, '1C_diff')  # Replace with actual function
Af = np.sum(Af_z, axis=2)

plt.figure(figsize=(10, 6))
cm = plt.get_cmap(cmap_name, len(sp) + 1).colors
for ii in range(len(sp)):
    plt.semilogx(m, Af[ii,:], color=cm[ii])
plt.xlim(m[0], m[-1])
plt.xlabel('m_p [fg]')
plt.title('PMA')

Aft = tfer.tri(sp, m * 1e-18, prop['zet'], z)  # Replace with actual function

plt.semilogx(m.T, Aft[:, :, 2].T, 'k--')
plt.show()


#== CHARGING =============================================================#
d_star = np.logspace(np.log10(13.1), np.log10(500), 10)

z3 = np.arange(-6, 7)
Ac3, _, model = tfer.charger(d_star, z3)  # Replace with actual function

plt.figure(figsize=(10, 12))
plt.subplot(5, 1, 1)
cm = plt.get_cmap(cmap_name, np.size(d_star) + 1).colors
for ii in range(np.size(d_star)):
    plt.plot(z3, Ac3[ii,:], 'o-', color=cm[ii])
plt.xlabel('Gopal.-Wieds.')
plt.title('Charge distributions')

opts = {'eps': 13.5, 'nit': 4e13}
z4 = np.expand_dims(np.arange(101), 1)
Ac4, _, _ = tfer.charger(d_star, z4, model='fuchs', opts=opts)  # Replace with actual function

plt.subplot(5, 1, 2)
cm = plt.get_cmap(cmap_name, np.size(d_star) + 1).colors
for ii in range(np.size(d_star)):
    plt.semilogx(np.concatenate(([[0.5]], z4[1:])), Ac4[ii,:], 'o-', color=cm[ii])
plt.xlabel('Fuchs, nit = 4e13')

opts['nit'] = 5e11
Ac4, _, _ = tfer.charger(d_star, z4, model='fuchs', opts=opts)  # Replace with actual function

plt.subplot(5, 1, 4)
cm = plt.get_cmap(cmap_name, np.size(d_star) + 1).colors
for ii in range(np.size(d_star)):
    plt.semilogx(np.concatenate(([[0.5]], z4[1:])), Ac4[ii,:], 'o-', color=cm[ii])
plt.xlabel('Fuchs, nit = 5e11')

"""
Li et al. is not available.
opts = {'eps': 13.5, 'nit': 4e13}
z5 = np.arange(301)
Ac5 = tfer.charger(d_star, z5, [], 'li', opts)  # Replace with actual function

plt.subplot(5, 1, 3)
plt.semilogx(np.concatenate(([5e-1], z5[1:])), Ac5, 'o-')
plt.xlabel('z [-]')
plt.xlim([5e-1, 100])
plt.xlabel(f'Li, nit = {opts["nit"] / 1e12:.2f}x10^12')

opts['nit'] = 5e11
Ac5 = tfer.charger(d_star, z5, [], 'li', opts)  # Replace with actual function

plt.subplot(5, 1, 5)
plt.semilogx(np.concatenate(([5e-1], z5[1:])), Ac5, 'o-')
plt.xlabel('z [-]')
plt.xlim([5e-1, 100])
plt.xlabel(f'Li, nit = {opts["nit"] / 1e12:.2f}x10^12 / z [-]')
plt.show()
"""


#== BINNED DATA ==========================================================#
m = np.logspace(-3, 2, 500)
m_star = np.logspace(-2.5, 1.5, 20)

Abin = tfer.bin(m_star, m)  # Replace with actual function

plt.figure(figsize=(10, 6))
cm = plt.get_cmap(cmap_name, np.size(Abin, 0) + 1).colors
for ii in range(np.size(Abin, 0)):
    plt.semilogx(m, Abin[ii, :], color=cm[ii])
plt.xlim(m[0], m[-1])
plt.title('Binned')
plt.show()


#== ELPI/IMPACTOR =========================================================#
da = np.logspace(0.5, 4.5, 500)

Aimp, _ = tfer.elpi(da)  # Replace with actual function

plt.figure(figsize=(10, 6))
cm = plt.get_cmap(cmap_name, np.size(Aimp,0) + 1).colors
for ii in range(np.size(Aimp,0)):
    plt.semilogx(da, Aimp[ii, :], color=cm[ii])
plt.xlim(da[0], da[-1])
plt.xlabel('d_a [nm]')
plt.title('Impactor')
plt.show()


#== AAC ==================================================================#
da = np.logspace(0.8, 3.2, 1500)
da_star = np.logspace(1, 3, 10)
prop = props.aac()
prop = autils.massmob_add(prop, 'soot')

# Limited trajectory variant
opts = {'model': 'lt'}
Aa, _, _ = tfer.aac(da_star, da, prop, opts)  # Replace with actual function

# Particle streamline
opts = {'diffusion': False}
Aa2, _, _ = tfer.aac(da_star, da, prop, opts)  # Replace with actual function

# Scanning version (limited trajectory only)
opts = {'model': 'lt', 'scan': 1}
prop['tsc'] = 300
prop['omega_s'] = 4e3
prop['omega_e'] = 10
Aas, _, _ = tfer.aac(da_star, da, prop, opts)  # Replace with actual function

plt.figure(figsize=(10, 6))
cm = plt.get_cmap(cmap_name, np.size(Aa, 0) + 1).colors
for ii in range(np.size(Aa, 0)):
    plt.semilogx(da, Aa[ii, :], color=cm[ii])
plt.semilogx(da, Aa2.T, 'k--')
plt.semilogx(da, Aas.T, 'k:')
plt.xlim(da[0], da[-1])
plt.xlabel('d_a [nm]')
plt.title('AAC')
plt.show()
