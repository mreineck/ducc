# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2021 Max-Planck-Society

import ducc0
import numpy as np
import matplotlib.pyplot as plt

sigma = 1e-1
f_min=1e-4
f_knee=1e-1
f_samp=10.
slope=-1.7
nsamp = 1000000

gen = ducc0.misc.OofaNoise(sigma, f_min, f_knee, f_samp, slope)

power = np.zeros(nsamp)
nps = int(300)

for i in np.arange(nps):
    inp = np.random.normal(0.,1.,(nsamp,))
    noise = gen.filterGaussian(inp)

    ps = np.abs(np.fft.fft(noise))**2 / nsamp
    
    power = power + ps

ps = power/nps

time_step = 1. / f_samp
freqs = np.fft.fftfreq(noise.size, time_step)
ps_theory = sigma**2 * ((freqs**2+f_knee**2)/(freqs**2+f_min**2))**(-slope/2)

plt.title('Noise spectrum averaged over '+str(nps)+' realizations')
plt.loglog(freqs[:ps.size//2],ps[:ps.size//2],color='red')
plt.loglog(freqs[:ps.size//2],ps_theory[:ps.size//2],color='black')
plt.show()
