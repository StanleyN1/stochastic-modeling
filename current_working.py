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
    # return (kf * x ** 2) / (x ** 2 + Kd) - kd*x + R_bas
    return 4*x - x ** 3

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
num_of_runs = 1
N = 500
t_0, t_f = 1, 1.05
n_k = 100 # number of runs to consider inbetween
dt_k = (t_f - t_0) / (N / n_k)
print(f'time interval: {(t_f - t_0) / (N / n_k)}')
dt = (t_f - t_0) / N
ts = np.linspace(t_0, t_f, num=N + 1)

x_is = {'x+': 4.28343, 'x-': 0.62685, 'xu': 1.48971}
xs = np.linspace(-1, 1, num=501)

fss = np.zeros((len(xs), num_of_runs, N + 1))

for i, xi in enumerate(xs): # loop through initial values
    for n in range(num_of_runs): # num for each initial value
        run = run_euler(xi, dt, N, f, sigma)
        fss[i][n] = run
        plt.plot(ts, run)

# %%
import scipy.integrate as integrate

def F(x):
    res = np.zeros_like(x)
    for i, val in enumerate(x):
        y, err = integrate.quad(f, 0, val)
        res[i] = y
    return res

plt.plot(xs, f(xs), label='f')
plt.plot(xs, -F(xs), label='U')
# for xiname, xi in x_is.items():
#     plt.axvline(xi, label=xiname)
plt.legend()

# %%
def kramers_moyal(xs, fss):
    '''approximates well when tf - t0 is very small'''
    f_approx = np.zeros_like(xs)
    s_approx = np.zeros_like(xs)
    # len(fss[-1][0])
    fs = fss.mean(1)
    for i in range(len(xs)):
        # sum_diff = 0
        # for j in range(1, N // n_k):
        #     sum_diff += (fs[i][j*n_k:(j + 1)*n_k] - fs[i][(j-1)*n_k:j*n_k]).mean(0)
        # diff = sum_diff // (N // n_k)
        diff = np.diff(fss[i]).mean(0)
        f_approx[i] =  diff.mean() / dt
        s_approx[i] =  diff.mean() ** 2 / dt
#-1.6956416560054222
    return {'f': f_approx, 's': s_approx}# np.polyfit(xs, f_approx, deg=4), np.polyfit(xs, s_approx, deg=0)

# %%

fss.mean()
# data['f']
data = kramers_moyal(xs, fss)
plt.plot(xs, f(xs), label='f')
plt.plot(xs, -F(xs), label='U')
plt.plot(xs, data['f'], label='f approx')
# plt.plot(xs, np.poly(data['f']), label='f poly')
plt.plot(xs, data['s'], label='s approx')
plt.legend()
