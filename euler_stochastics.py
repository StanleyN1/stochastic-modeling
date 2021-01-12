import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


c_theta = 0.7

c_mu    = 1.5

c_sigma = 0.06

def getruns():
    global num_of_runs
    return [f'rrun{i}' for i in range(num_of_runs)]

def sigma(x, t=None):
    return c_sigma

def mu(x, t=None):
    return c_theta * (c_mu - x)

def dW(dt):
    return np.random.normal(loc=0, scale=np.sqrt(dt))

def a(x, t=None):
    kf = 6
    Kd = 10
    kd = 1
    R_bas = 0.4
    return (kf * x ** 2) / (x ** 2 + Kd) - kd*x + R_bas

def b(x, t = None):
    return x

# Finds value of y for a given x using step size h
# and initial value y0 at x0.
def rungeKutta(x0, y0, n, h, f=lambda x:x):
    # Iterate for number of iterations
    y = y0
    for i in range(1, n + 1):
        # "Apply Runge Kutta Formulas to find next value of y"
        k1 = h * f(x0, y)
        k2 = h * f(x0 + 0.5 * h, y + 0.5 * k1)
        k3 = h * f(x0 + 0.5 * h, y + 0.5 * k2)
        k4 = h * f(x0 + h, y + k3)

        # Update next value of y
        y = y + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)

        # Update next value of x
        x0 = x0 + h
    return y

def euler(x, t, dt, a, b):
    return x + a(x)*dt + b(x)*dW(dt)

def euler_ode(x, t, dt, a, b):
    return x + a(x)*dt

def SDE(x, t, dt):
    a = lambda y, t: 2 * y
    b = lambda y, t: y
    return euler(x, t, dt, mu, sigma)
    # return x + f(x, t)*dt + sigma(x, t)*dW(dt)
    # return x + dW(dt)

def sim(x_0, t_0, t_f, a, b, N=1000):
    global num_of_runs

    ts = np.linspace(start=t_0, stop=t_f, num=N+1)
    dt = ts[1] - ts[0]

    xss = np.zeros(shape=(num_of_runs, N+1))
    xss[:, 0] = x_0
    for j in range(num_of_runs):
        for i in range(1, len(ts)):
            t = ts[i - 1]
            x = xss[j][i - 1]

            xss[j][i] = euler(x, t, dt, a=a, b=b)

    fs = np.zeros(N+1)
    fs[0] = x_0
    sigmas = np.zeros(N+1)
    sigmas[0] = x_0
    xs_avg = np.mean(xss, axis=0)
    for i in range(1, len(xss[0])):
        xdt = xss[:, i]
        x0 = xss[:, i - 1]
        diff = xdt - x0
        fs[i] = np.mean(diff / dt)
        sigmas[i] = np.mean(diff ** 2 / dt)

    func = {'ts': ts, 'f(x)': fs, 'sigma(x)': sigmas, 'xs_avg': xs_avg}
    run = {'ts': ts, **{f'rrun{i}': xss[i] for i in range(len(xss))}}
    all = {**run, **func}
    df = pd.DataFrame(all)
    return df
    #return {'xs' : xss, 'f(x)': fs, 'sigma(x)': sigmas, 'ts': ts, 'xs_avg': xs_avg}

# %% Run the simulation

# generate rainbow graphs
N = 500
num_of_runs = 15
x_is = {'x-': 0.62685, 'x+': 4.28343, 'xu': 1.48971} # first two stable, last is unstable
eps = [0.02, 0.04, 0.1]

x_0s = np.linspace(start=0, stop=6, num=76)
dfs = {(x_0, e): sim(x_0=x_0, t_0=0, t_f=50, a=a, b=lambda x: e, N=N) for e in eps for x_0 in x_0s}

t_0, t_f = 0, 50
ts = np.linspace(start=t_0, stop=t_f, num=N+1)
dt = ts[1] - ts[0]

# %% Plot the simulation
fig, axs = plt.subplots(1, len(eps), figsize=(15, 5), sharex=True)
cmap = plt.get_cmap('rainbow')

for (x_i, e), df in dfs.items():
    i = eps.index(e)
    runs = df[df.columns[1: num_of_runs + 1]]
    # df['x_0'] = x_i
    for run in runs:
        sns.lineplot(x='ts', y=run, color=cmap(x_i / max(x_0s)), ci=None, data=df[['ts', run]], legend=None, ax=axs[i])

    axs[i].set_ylabel('x(t)')
    axs[i].set_title(f'epsilon = {e}')

fig.savefig('graph')
fig
# %% Polynomial sim attempt 1
# take runs that only had a certain increase in concentration
skip = x_0s[::2]
df_us = {x0: sim(x_0=x0, t_0=0, t_f=50, a=a, b=lambda x: 0.02, N=N) for x0 in skip}
# {'x-': 0.62685, 'x+': 4.28343, 'xu': 1.48971}
# intervals = [(0, 1), (1, 3.75), (3.75, 6)]
#
# for ivt in intervals:
#     ivt[0] <= df_us.keys() <= ivt[1]

fss = np.zeros(shape=(num_of_runs, N+1))
df_u_runs = [df_u[getruns()] for df_u in df_us.values()]
# df_u_runs = df_u_runs[[run for run in (df_u_runs > 1.0).all().keys() if ((df_u_runs > 1.0).all())[run]]]

x_avg = np.zeros(N + 1)
fs = np.zeros(N + 1)
sigmas = np.zeros(N + 1)
len(x_avg)
x_avg = np.mean(df_u_runs, axis=2)

for j, df_u in enumerate(df_u_runs):
    for i in range(1, len(df_u_runs)):
        diff = df_u.iloc[i] - df_u.iloc[i-1]
        fs[i] = np.mean(diff / dt)
        sigmas[i] = np.mean(diff ** 2 / dt)

    # polynomial fitting of f and sigma
    fp = np.poly1d(np.polyfit(x_avg[j], fs, deg=3))
    sigmap = np.poly1d(np.polyfit(x_avg[j], sigmas, deg=0))

    fxs = []
    for i in range(1, len(x_avg[0]) + 1):
        t = ts[i - 1]
        x = x_avg[j][i - 1]
        fx = euler(x, t, dt, fp, sigmap)
        fxs.append(fx)

    # approximate solution (polynomial-fitted)
    axs[0].plot(ts, fxs, label='f, sigma', linewidth=2.5, color='k')

# %% polynomial attempt 2
def poly_fit(df, x_i=0, f_deg=2, s_deg=0, avg=None):
    if not avg:
        avg = np.mean(df, axis=1)
    fs = np.zeros(len(df))
    sigmas = np.zeros(len(df))
    fs[0] = sigmas[0] = x_i
    for i in range(1, len(df)):
        diff = df.iloc[i] - df.iloc[i - 1]
        fs[i] = np.mean(diff / dt)
        sigmas[i] = np.mean(diff ** 2 / dt)

    # polynomial fitting of f and sigma
    fp = np.poly1d(np.polyfit(avg, fs, deg=f_deg))
    sigmap = np.poly1d(np.polyfit(avg, sigmas, deg=s_deg))

    return {'f': fp, 's': sigmap, 'avg': avg}

def poly_sim(fp, sigmap, x_i=0):
    fxs = np.zeros(N + 1)
    fxs[0] = x_i
    for i in range(1, N + 1):
        t = ts[i - 1]
        x = fxs[i - 1]
        fx = euler(x, t, dt, fp, sigmap)
        fxs[i] = fx
    return fxs

def poly_sim_ode(fp, sigmap=lambda x: 0, x_i=0):
    fxs = np.zeros(N + 1)
    fxs[0] = x_i
    for i in range(1, N + 1):
        t = ts[i - 1]
        x = fxs[i - 1]
        fx = euler_ode(x, t, dt, fp, sigmap)
        fxs[i] = fx
    return fxs
# %% polynomial attempt 2 ctnd
epsilon = eps[1]

# tests the polynomial fitted curve with various initial points
runs = getruns()
fx_u = sim(x_0=x_is['xu'], t_0=0, t_f=50, a=a, b=lambda x: epsilon, N=N)
x_u_runs = fx_u[runs]

plt.plot(ts, fx_u.iloc[:, 1:])

rising_mask = x_u_runs[(x_u_runs > x_is['xu'])].any(0)

rising = x_u_runs.loc[:, rising_mask]
falling = x_u_runs.loc[:, ~rising_mask]

masks = [rising, falling]

skip = x_0s # [xi for xi in x_0s if 1 < xi < 1.75]
rising_poly = poly_fit(rising, x_is['xu'])

with np.errstate(all='raise'):
    for mask in masks:
        for x0 in skip:
            mask_poly = poly_fit(mask, x_is['xu'])
            try:
                data = poly_sim(mask_poly['f'], mask_poly['s'], x0)
                sns.lineplot(x=ts, y=data, color='gray', ax=axs[1], linewidth=0.75)
            except:
                print(x0)
fig
# %% creating the most probable transition pathway
capped_mask = x_u_runs[(x_u_runs > x_is['xu']) & (x_u_runs < x_is['x+'] + 0.5)].any(0)

rising_capped = x_u_runs.loc[:, capped_mask]

poly = poly_fit(rising_capped, x_is['xu'], f_deg=3, s_deg=0)
print(poly['f'])

zm = (poly['s']**2 / 2) * np.polyder(poly['f'], m=2) + np.polyder(poly['f'], m=1)*poly['f']
print(zm)

# z'(T) = 0
# z'(0) = v0
np.polyval([1, 2], 0)
zm_sim = poly_sim_ode(zm, poly['s'], x_i=x_is['xu']+1)
# %%
successes = {'v0':[], 'G':[]}
target = 0 # x_is['x+']
initial = x_is['xu']
with np.errstate(all='raise'):
    # we keep the initial x the same
    # but vary the velocity
    v_0s = np.linspace(0, 10/dt, num=101)
    for v0 in v_0s:
        try:
            # sim_final = rungeKutta(x0=0, y0=x_i, n=N, h=dt, f=zm)
            adjusted = initial + v0*dt
            sim = poly_sim_ode(zm, x_i=v0)

            # print('sim:', (sim_final - sim[-10:].mean()) ** 2)
            lines = axs[1].plot(ts, sim, color='k', linewidth=0.5)
            loss = (sim[-1].mean() - target) ** 2
            # loss = (sim_final.mean() - target) ** 2
            successes['v0'].append(v0)
            successes['G'].append(loss)

        except:
            pass

min_G = min(successes['G'])
midx = successes['G'].index(min_G)
min_v0 = successes['v0'][midx]
min_xi = min_v0*dt + initial
print(min_G, min_v0, min_xi)
# axs[1].lines
fig

# %% remove recent lines
for _ in range(len(successes['G'])):
    axs[1].lines.pop()
# %% integrate zm to give function
int_zm = np.polyint(zm, m=1, k=[x_is['xu'], x_is['x+']])
print(int_zm)

plt.plot(ts, poly_sim_ode(int_zm, poly['s'], x_i=x_is['xu']))

# %%

# success_sim = poly_sim(zm, poly['s'], x_i=min_v0)
#
# axs[1].plot(ts, success_sim, label='z(t)', linewidth=2.5, color='k')
#
# plt.plot(ts, success_sim, axs)
#
# fig
# %%
# data = np.polyint(zm, 2, [x_is['x+'], x_is['x-']])
#
# data(50)
#
# with np.errstate(all='raise'):
#     for x0 in skip:
#         try:
#             sns.lineplot(x=ts, y=data, color='black', ax=axs[0], linewidth=2)
#         except:
#             print(x0)


# %% shooting method
