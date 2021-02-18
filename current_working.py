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
    # return x ** 3 - 4*x # simple test case

def sigma(x):
    '''diffusion function'''
    return 0.1

def euler(x, dt, a, b, t=None):
    '''one step of the ODE simulation'''
    # if b = 0 then we have ordinary ODE
    return x + a(x)*dt + b(x)*dW(dt)

def run_euler(x0, dt, N, f, sigma):
    '''simulate the ode $dX_t = f(X_dt, t)dt + sigma(X_dt, t)dW_t$'''
    fs = np.zeros(N + 1)
    fs[0] = x0
    for i in range(1, N + 1):
        fs[i] = euler(fs[i - 1], dt, a=f, b=sigma)
    return fs
# %%
num_of_runs = 3
N = 500
t_0, t_f = 0, 20
split = 100 # number of split time intervals
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

plt.savefig('pics/simulated.png')

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
for xiname, xi in x_is.items(): # relevant stable and unstable points
    rgb = np.random.rand(3,)
    plt.axvline(xi, label=xiname, c=rgb)

plt.legend(loc='best')
plt.xlabel('x')
plt.title('drift and diffusion')
# plt.savefig('pics/actualfunc.png')

# %%
def kramers_moyal(xs, fss, intervals=(0, 1)):
    '''implements eq(2) and eq (3) of Dai et al
    approximates well when only considering first time steps'''
    # range defined as time intervals to consider

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

# %% polynomial interpolation
best_approx = kramers_moyal(xs, fss, (0, 1))
best_poly = np.poly1d(np.polyfit(xs, best_approx['f'], deg=3))
plt.plot(xs, np.polyval(best_poly, xs), label='f poly')
plt.plot(xs, f(xs), label='f')
plt.plot(xs, -F(f, xs), label='F poly')
plt.plot(xs, -F(best_poly, xs), label='F')

plt.legend()
