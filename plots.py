import numpy as np
import matplotlib.pyplot as plt

def dW(dt):
    '''brownian noise'''
    return np.random.normal(loc=0, scale=np.sqrt(dt))

def f(x, t=None):
    '''drift function'''
    kf = 6
    Kd = 10
    kd = 1
    R_bas = 0.4
    return (kf * x ** 2) / (x ** 2 + Kd) - kd*x + R_bas # biological model
    # return 4*x - x ** 3 # simple test case

def sigma(x):
    '''diffusion function'''
    return 0.1

import scipy.integrate as integrate
def F(f, x):
    res = np.zeros_like(x)
    for i, val in enumerate(x):
        y, err = integrate.quad(f, 0, val)
        res[i] = y
    return res

# %%
cmap = plt.get_cmap('rainbow')

fss = np.load('data/bio_fss.npy')

dx, num_runs, N = fss.shape
t0, tf = 0, 50
dt = (tf - t0) / N
ts = np.linspace(t0, tf, N)
xs = fss[:, :, 0][:, 0]
n_k = 10
split = N // n_k
dx = xs[1] - xs[0]

x_is = {'x+': 4.28343, 'x-': 0.62685, 'xu': 1.48971}

# %% plot simulation data

for i, xi in enumerate(xs):
    for run in range(num_runs):
        plt.plot(ts, fss[i][run], color=cmap(xi / max(xs)))

plt.axhline(x_is['x+'], label='x+', c='blue')
plt.axhline(x_is['x-'], label='x-', c='red')
plt.axhline(x_is['xu'], label='xu', c='black')

plt.xlabel('t')
plt.ylabel('x')
plt.legend()
plt.title('simulation')
plt.savefig('pics/simulated.png')
# %% sliced / zoomed in
for i, xi in enumerate(xs):
    for run in range(num_runs):
        plt.plot(ts, fss[i][run], color=cmap(xi / max(xs)))

# plt.axhline(x_is['x+'], label='x+', c='blue')
# plt.axhline(x_is['x-'], label='x-', c='red')
# plt.axhline(x_is['xu'], label='xu', c='black')

plt.axvline(0, c='gray')
plt.axvline(5, c='gray')

plt.xlabel('t')
plt.ylabel('x')
# plt.legend()
plt.title('simulation sliced')
plt.savefig('pics/paper/simulated_labeled_sliced.pdf')

# %% close up of X_{delta t}
interval = (0, 50)
for i in range(0, len(xs)):
    for run in range(num_runs):
        plt.plot(ts[interval[0]:interval[1]], fss[i][run][interval[0]:interval[1]], color=cmap(xs[i] / max(xs)))

plt.axhline(x_is['x+'], label='x+', c='blue')
plt.axhline(x_is['x-'], label='x-', c='red')
plt.axhline(x_is['xu'], label='xu', c='black')

plt.xlabel('t')
plt.ylabel('x')
plt.legend()
plt.title('close-up of simulation')
# plt.savefig('pics/paper/simulation_zoom.pdf')
# %%
def kramers_moyal(xs, fss, intervals=(0, 1)):
    '''implements eq(2) and eq (3) of Dai et al.
    approximates well when only considering first time steps'''
    # intervals defined as time intervals to consider

    f_approx = np.zeros_like(xs)
    s_approx = np.zeros_like(xs)
    fs = fss.mean(1) # mean over the runs at each x_i
    diffs = np.zeros((len(xs), split))
    for i in range(len(xs)):

        # splits data into split number of intervals over time
        diffs[i] = np.diff(np.split(fs[i][1:], split)).mean(1) # num of eq (2), (3) of Dai et al

        diff = diffs[i][intervals[0]:intervals[1]] # most relevant information is first couple of time steps
        f_approx[i] =  diff.mean() / dt # eq (2) of Dai et al
        s_approx[i] =  np.sqrt(diff.mean() ** 2) / dt # eq (3) of Dai et al

    return {'f': f_approx, 's': s_approx}
# %%
data = kramers_moyal(xs, fss, (0, 1))
polyf = np.poly1d(np.polyfit(xs, data['f'], deg=3))
polys = np.poly1d(np.polyfit(xs, data['s'], deg=0))

plt.plot(xs, np.polyval(polyf, xs), label='poly fit')
plt.plot(xs, f(xs), label='f')
# plt.plot(xs, -F(f, xs), label='F poly')
# plt.plot(xs, -F(polyf, xs), label='F')

plt.plot(xs, data['f'], label='poly data') # approximate drift

plt.xlabel('x')
plt.title('approximation and interpolation')
plt.legend()
# plt.savefig('pics/paper/polyfit_f.pdf')
# %%
zm = (polys ** 2 / 2) * np.polyder(polyf, m=2) + np.polyder(polyf, m=1)*polyf
loadzneg = np.load('data/bio_zss_xneg.npy')
loadvneg = np.load('data/bio_vss_xneg.npy')
target = x_is['x+']
initial = x_is['x-']

idx = np.where(loadzneg.any(2))

zz = loadzneg[idx]
vv = loadvneg[idx]

loss = (zz[:,-1] - target) ** 2

# vv[np.where(loss < 1e-2)]
min_loss = loss.argmin()
min_v = vv[min_loss]
# %% plotting
fig, axs = plt.subplots(1, 2, sharey=True, figsize=(18, 8))
# %% velocities
# axs[0].plot(vv[min_loss], zz[min_loss], color='black', linewidth=2, label='best')
# for v, z in zip(vv, zz):
#     axs[0].plot(v, z, color=cmap(v[0]/vv[:,0].max()))
#
# axs[0].plot(vv[min_loss], zz[min_loss], color='black', linewidth=2, label='best')
# axs[0].set_xlabel('v')
# axs[0].set_ylabel('x')
# %%
approx = kramers_moyal(xs, fss, (0, 1)) # most accurate data
polyf = np.poly1d(np.polyfit(xs, approx['f'], deg=3))
polys = np.poly1d(np.polyfit(xs, approx['s'], deg=0))
axs[0].plot(np.polyval(polyf, xs), xs, label='f poly')
# axs[0].plot(f(xs), xs, label='f')
axs[0].plot(-F(f, xs), xs, label='-U')
axs[0].plot(-F(polyf, xs), xs, label='-U poly')

axs[0].plot(approx['f'], xs, label='f approx') # approximate drift
# axs[0].plot(approx['s'], xs, label='s approx')
axs[0].set_xlabel('f(x)')
axs[0].set_ylabel('x')
axs[0].set_title('functions')
axs[0].legend()

# %%
for i, xi in enumerate(xs):
    for run in range(num_runs):
        axs[1].plot(ts, fss[i][run], color='black') # cmap(xi / max(xs))

axs[1].set_xlabel('t')
axs[1].set_ylabel('x')
axs[1].set_title('simulated')
# %%

for i, xi in enumerate(xs):
    for run in range(2):
        axs[1].plot(ts, fss[i][run], color='black')

# from x- to x+ shooting method
zneg_idx = np.where(loadzneg.any(2))
for i, z in enumerate(loadzneg[zneg_idx]):
    axs[1].plot(ts, z, color=cmap(loadvneg[zneg_idx][i, 0] / loadvneg[zneg_idx][:,0].max()))

# from x+ to x- shooting method
# zpos_idx = np.where(loadzpos.any(2))
# for i, z in enumerate(loadzpos[zpos_idx]):
#     plt.plot(ts, z, color=cmap(loadvpos[zpos_idx][i, 0] / loadvpos[zpos_idx][:,0].min()))

# min loss transition pathway
axs[1].plot(ts, zz[min_loss], color='purple', label=f'min action path', linewidth=2.75)
axs[1].legend()
axs[1].set_xlabel('t')
axs[1].set_ylabel('x')
axs[1].set_title('most probable transition pathway')

# %%
x_is
x_is_colors = ['blue', 'red', 'black']

i = 0
for xiname, xi in x_is.items():
    rgb = np.random.rand(3,)
    axs[0].axhline(xi, label=xiname, c=x_is_colors[i])
    axs[1].axhline(xi, label=xiname, c=x_is_colors[i])
    i += 1

axs[1].legend()

fig.tight_layout(pad=1)
fig
fig.savefig('pics/paper/dual_plot.pdf')
