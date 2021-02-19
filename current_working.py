import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

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

def euler(x, dt, a, b, t=None):
    '''one step of the ODE simulation'''
    # if b = 0 then we have ordinary ODE
    return x + a(x)*dt + b(x)*dW(dt)

def run_euler(x0, dt, N, f, sigma = lambda x: 0):
    '''simulate the ode $dX_t = f(X_dt, t)dt + sigma(X_dt, t)dW_t$'''
    fs = np.zeros(N + 1)
    fs[0] = x0
    for i in range(1, N + 1):
        fs[i] = euler(fs[i - 1], dt, a=f, b=sigma)
    return fs
# %%
num_of_runs = 2
N = 1000
t_0, t_f = 0, 50

n_k = 10
split = N // n_k # number of split time intervals
dt = (t_f - t_0) / N
ts = np.linspace(t_0, t_f, num=N + 1)

x_is = {'x+': 4.28343, 'x-': 0.62685, 'xu': 1.48971}
N_x = 100
xs = np.linspace(0, 6, num=N_x + 1)

fss = np.zeros((len(xs), num_of_runs, N + 1))

cmap = plt.get_cmap('rainbow')

for i, xi in tqdm(enumerate(xs)): # loop through initial values
    for n in range(num_of_runs): # num for each initial value
        run = run_euler(xi, dt, N, f, sigma)
        fss[i][n] = run
        plt.plot(ts, run, c=cmap(xi / max(xs)))

plt.xlabel('t')
plt.ylabel('x')
plt.title('simulated data')
# plt.savefig('pics/simulated.png')

# %%
import scipy.integrate as integrate

def F(f, x):
    res = np.zeros_like(x)
    for i, val in enumerate(x):
        y, err = integrate.quad(f, 0, val)
        res[i] = y
    return res

plt.plot(xs, f(xs), label='f')
plt.plot(xs, -F(f, xs), label='U')
# for xiname, xi in x_is.items(): # relevant stable and unstable points
#     rgb = np.random.rand(3,)
#     plt.axvline(xi, label=xiname, c=rgb)

plt.legend(loc='best')
plt.xlabel('x')
plt.title('drift and diffusion')
# plt.savefig('pics/actualfunc.png')

# %%
def kramers_moyal(xs, fss, intervals=(0, 1)):
    '''implements eq(2) and eq (3) of Dai et al
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
        s_approx[i] =  diff.mean() ** 2 / dt # eq (3) of Dai et al

    return {'f': f_approx, 's': s_approx}
# %% best approximation of $f$ and $\sigma$.
data = kramers_moyal(xs, fss, (0, 1))
plt.plot(xs, f(xs), label='f') # exact drift function
plt.plot(xs, -F(f, xs), label='U') # exact potential of drift
plt.plot(xs, data['f'], label='f approx') # approximate drift
plt.plot(xs, data['s'], label='s approx') # approximate diffusion
plt.legend()

# %% experiment when consider more intervals than first slice

plt.plot(xs, f(xs), label='f')
for i in range(1, split, 5):
    data = kramers_moyal(xs, fss, (0, i))
    plt.plot(xs, data['f'], label=f'f approx from (0, {i})')
    # plt.plot(xs, data['s'], label='s approx')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# %% polynomial interpolation and approximated data
approx = kramers_moyal(xs, fss, (0, 1)) # most accurate data
polyf = np.poly1d(np.polyfit(xs, approx['f'], deg=3))
polys = np.poly1d(np.polyfit(xs, approx['s'], deg=0))
plt.plot(xs, np.polyval(polyf, xs), label='f poly')
plt.plot(xs, f(xs), label='f')
plt.plot(xs, -F(f, xs), label='F poly')
plt.plot(xs, -F(polyf, xs), label='F')

plt.plot(xs, approx['f'], label='f approx') # approximate drift
plt.plot(xs, approx['s'], label='s approx')

plt.xlabel('x')
plt.title('approximation and interpolation')
plt.legend()
# plt.savefig('pics/approxpoly.png')

# %% shooting method

# Cheng et al. eq (7) gives formula for most probable transition pathway

zm = (polys ** 2 / 2) * np.polyder(polyf, m=2) + np.polyder(polyf, m=1)*polyf

for i, xi in tqdm(enumerate(xs)):
    plt.plot(ts, fss[i].mean(0), color='black')


initial = 0
target = x_is['x+']
success = {'vi': [], 'loss': [], 'xi': []}

vs = np.linspace(-1, 1, 51)
xs_skip = xs[::]
zss = np.zeros((len(xs_skip), len(vs), N + 1))
vss = np.zeros_like(zss)
with np.errstate(all='raise'):
    for i, xi in tqdm(enumerate(xs_skip)):
        initial = xi
        for j, vi in enumerate(vs):
            try: # catches overflows
                vss[i][j][0] = vi
                zss[i][j][0] = xi
                for k in range(1, N + 1): # shooting with initial veloctiy
                    vss[i][j][k] = vss[i][j][k - 1] + zm(zss[i][j][k - 1])*dt
                    zss[i][j][k] = zss[i][j][k - 1] + vss[i][j][k - 1]*dt
                    if max(xs) < zss[i][j][k] < min(xs):
                        raise ValueError # still outside of our bounds
                plt.plot(ts, zss[i][j], color=cmap(xi / max(xs_skip)), linewidth=1)
            except:
                # when overflow occurs
                # reset data to zero, i.e., all zeros in a row means bad data
                vss[i][j] = np.zeros(zss.shape[-1])
                zss[i][j] = np.zeros(zss.shape[-1])

plt.xlabel('t')
plt.ylabel('x')
plt.title('most probable pathway')

# np.save('data/bio_fss.npy', fss)
# np.save('data/bio_zss.npy', zss)
# np.save('data/bio_vss.npy', vss)
# plt.savefig('pics/simple z')
# plt.savefig('pics/bio_shooting.png')
