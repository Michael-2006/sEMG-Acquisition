import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

np.random.seed(0)

# ---- parameters ----
fs, T = 10000, 10                  # Hz, s
N = int(fs * T)
t = np.arange(N) / fs
M = 100                            # motor units
fr_min, fr_max = 8, 35             # Hz, firing-rate limits
gain = 30                          # Hz per unit excitation
cv = 0.2                           # ISI coefficient of variation
tau = 3.5e-3                       # s, MUAP width (peak at 1/(pi*tau) ~ 91 Hz)
rms_active = 0.5e-3                # V, target RMS where excitation > 0.5
noise_frac = 0.02                  # instrumentation noise relative to signal std

# ---- excitation drive (0..1) ----
e = np.clip(np.sin(2 * np.pi * 0.25 * t), 0, None)

# ---- motor unit pool ----
i = np.arange(M)
RT = 0.6 * (np.exp(np.log(30) * i / M) - 1) / (30 - 1)   # recruitment thresholds
P = np.exp(np.log(50) * i / M)                           # unit amplitudes

# ---- firing times ----
imp = np.zeros(N)
nxt = np.zeros(M)
for k in range(N):
    active = e[k] > RT
    due = active & (t[k] >= nxt)
    for u in np.where(due)[0]:
        imp[k] += P[u]
        rate = np.clip(fr_min + gain * (e[k] - RT[u]), fr_min, fr_max)
        nxt[u] = t[k] + (1 / rate) * max(0.3, 1 + cv * np.random.randn())
    nxt[~active] = t[k]

# ---- MUAP kernel ----
tk = np.arange(-5 * tau, 5 * tau, 1 / fs)
h = (1 - 2 * (tk / tau) ** 2) * np.exp(-(tk / tau) ** 2)
h /= np.abs(h).max()

# ---- assemble signal ----
sig = np.convolve(imp, h, mode='same')
sig += noise_frac * sig.std() * np.random.randn(N)
sig *= rms_active / sig[e > 0.5].std()

np.savetxt('emg.txt', np.c_[t, sig], fmt='%.9e')

# ---- validation ----
ff, Pxx = welch(sig, fs, nperseg=2048)
fig, ax = plt.subplots(2, 1, figsize=(9, 6))
ax[0].plot(t, sig); ax[0].set_xlabel('t (s)'); ax[0].set_ylabel('V')
ax[1].semilogy(ff, Pxx); ax[1].set_xlim(0, 500)
ax[1].set_xlabel('f (Hz)'); ax[1].set_ylabel('PSD (V²/Hz)')
plt.tight_layout()
plt.savefig('emg_check.png', dpi=120)
plt.show()