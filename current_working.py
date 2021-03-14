import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

def dW(dt):
    '''brownian noise'''
    return np.random.normal(loc=0, scale=np.sqrt(dt))

kf = 6
Kd = 10
kd = 1
R_bas = 0.4
def f(x, t=None):
    '''drift function'''
    return (kf * x ** 2) / (x ** 2 + Kd) - kd*x + R_bas # biological model
    # return 4*x - x ** 3 # simple test case

def fprime(x, t=None):
    '''derivative of drift'''
    return -kd + (2*x**3 + 2*x*Kd*kf - 2*kf*x) / (x ** 2 + Kd) ** 2

def fprimeprime(x, t=None):
    '''second derivative of drift'''
    return (((x ** 2 + Kd) ** 2) * (6*x**2 + 2*kf*(Kd - 1)) - (2*x**3 + 2*x*Kd*kf - 2*kf*x)*4*x*(x**2+Kd)) / (x**2 + Kd) ** 4

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
num_of_runs = 20
N = 1000
t_0, t_f = 0, 50

n_k = 10
split = N // n_k # number of split time intervals
dt = (t_f - t_0) / N
ts = np.linspace(t_0, t_f, num=N + 1)

x_is = {'x+': 4.28343, 'x-': 0.62685, 'xu': 1.48971}
N_x = 100
xs = np.linspace(0, 6, num=N_x + 1)
cmap = plt.get_cmap('rainbow')
# %%
fss = np.zeros((len(xs), num_of_runs, N + 1))

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
# plt.plot(xs, [sigma(x) for x in xs], label='$\sigma$')

plt.axvline(x_is['x+'], label='x+', c='blue')
plt.axvline(x_is['x-'], label='x-', c='red')
plt.axvline(x_is['xu'], label='xu', c='black')

plt.legend(loc='best')
plt.xlabel('x')
plt.title('drift')
# plt.savefig('pics/paper/actualfunc.pdf')

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
        diffs[i] = np.diff(np.split(fs[i][1:], split)).mean(1) # numerator of eq (2), (3) of Dai et al
        diff = diffs[i][intervals[0]:intervals[1]] # most relevant information is first couple of time steps
        f_approx[i] =  diff.mean() / dt # eq (2) of Dai et al
        s_approx[i] =  np.sqrt(diff.mean() ** 2) / dt # eq (3) of Dai et al

    return {'f': f_approx, 's': s_approx}
# %% best approximation of $f$ and $\sigma$.
fss = np.load('data/bio_fss.npy')
data = kramers_moyal(xs, fss, (0, 1))
plt.plot(xs, f(xs), label='f') # exact drift function
# plt.plot(xs, -F(f, xs), label='U') # exact potential of drift
plt.plot(xs, data['f'], label='f approx') # approximate drift
# plt.plot(xs, data['s'], label='s approx') # approximate diffusion
plt.legend()
plt.xlabel('x')
plt.ylabel('f')
plt.title('f and approximation')
# plt.savefig('pics/paper/f_and_approx.pdf')
# %% experiment when consider more intervals than first slice

plt.plot(xs, f(xs), label='f')
for i in range(split):
    data = kramers_moyal(xs, fss, (i, i + 1))
    plt.plot(xs, data['f'], label=f'f approx from ({i}, {i + 5})', color=cmap(i / split))
plt.axvline(x_is['x+'], label='$x_+$', c='blue')
plt.axvline(x_is['x-'], label='$x_-$', c='red')
plt.axvline(x_is['xu'], label='$x_u$', c='black')
    # plt.plot(xs, data['s'], label='s approx')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# %% polynomial interpolation and approximated data
# fss = np.load('data/bio_fss.npy')

approx = kramers_moyal(xs, fss, (0, 1)) # most accurate data
polyf = np.poly1d(np.polyfit(xs, approx['f'], deg=3))
polys = np.poly1d(np.polyfit(xs, approx['s'], deg=0)) / 2
# plt.plot(xs, np.polyval(polyf, xs), label='f poly')
# plt.plot(xs, f(xs), label='f')
plt.plot(xs, [sigma(x) for x in xs], label='$\sigma$: 0.10')
plt.plot(xs, np.polyval(polys, xs), label=f'$\sigma$: {polys}')
# plt.plot(xs, -F(f, xs), label='F poly')
# plt.plot(xs, -F(polyf, xs), label='F')

# plt.plot(xs, approx['f'], label='f approx') # approximate drift
# plt.plot(xs, approx['s'], label='$\sigma$ approx')

plt.xlabel('x')
plt.title('approximation and interpolation')
plt.legend()
# %% shooting method not working
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# z = ...
# z(0) = x-
# z(T) = x+
def fun(x, y):
    return zm(y)

lo = -1
hi = 1
count = 0
target = x_is['x+']
initial = x_is['x-']
tol = 1e-4
while count < 100:
    guess = np.mean([lo, hi])
    # print(guess)
    sol = solve_ivp(fun, (t_0, t_f), [y0, guess], t_eval=ts)
    yf = sol.y[0][-5:].mean()
    print(guess)
    if abs(yf - target) < tol:
        break
    if yf < target:
        lo = guess
    else:
        hi = guess
    count += 1

print(guess)
# %% shooting method not working
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
# Cheng et al. eq (7) gives formula for most probable transition pathway
# fss = np.load('data/bio_fss.npy')
fss = np.load('data/bio_fss.npy')
data = kramers_moyal(xs, fss, (0, 1))
polyf = np.poly1d(np.polyfit(xs, data['f'], deg=3))
polys = np.poly1d(np.polyfit(xs, data['s'], deg=0)) / 2
zm = (polys ** 2) * np.polyder(polyf, m=2) + np.polyder(polyf, m=1)*polyf

# zm = lambda x: (sigma(x) ** 2 / 2) * fprimeprime(x) + fprime(x) * f(x)

plt.plot(xs, f(xs), label='f')
plt.plot(xs, fprime(xs), label="f'")
plt.plot(xs, fprimeprime(xs), label="f''")
plt.plot(xs, zm(xs), label="$\ddot{z}$")
plt.plot(xs, F(zm, xs), label="$\dot{z}$")
plt.plot(xs, F(zm, xs), label="$z$")
plt.legend()

# %% IVP solver test (not working currently)
from scipy.integrate import odeint
from scipy.optimize import minimize

fss = np.load('data/bio_fss.npy')
# odeint(fun, np.array([]))

target = x_is['x+']
initial = x_is['x-']

def func(t, y):
    # ys.append(y)
    return zm(y)

num_runs = 2
for i, xi in enumerate(xs):
    for run in range(num_runs):
        plt.plot(ts, fss[i][run], color='black') # cmap(xi / max(xs))
for x in [1, 2, 2.4, 3, 4]:
    ode_sol = odeint(func, x, ts, tfirst=True)

    plt.plot(ts, ode_sol.reshape(-1))

def fun(t, y):
    u, v = y
    return [zm(v), u]

def objective(v0):
    sol = solve_ivp(fun, (t_0, t_f), [initial, v0], t_eval=ts)
    z, v = sol.y
    return abs(z[-10:].mean() - target) # y[-1:].mean() - target (v[-5:].mean() - 0) ** 2 +

v0, = fsolve(objective, 0.0029)
sol = solve_ivp(fun, [t_0, t_f], [initial, v0])

plt.plot(sol.t, sol.y[0])
plt.show()
# %% IVP solver 2 test (not working currently)
target
vs = np.linspace(0.001, 0.01, 51)
for v0_guess in vs:
    v0, = fsolve(objective, v0_guess)
    loss = (objective(v0)) ** 2
    print(f'Init: {v0_guess}, Result: {v0}, Loss: {loss}')
# %% IVP shooting method (working)
for i, xi in tqdm(enumerate(xs)):
    for run in fss[i]:
        plt.plot(ts, run, color='black', alpha=0.5)

target = x_is['x+']

vs = np.linspace(0, 1, 5001)
xs_skip = [x_is['x-']] # xs[::]
zss = np.zeros((len(xs_skip), len(vs), N + 1))
vss = np.zeros_like(zss)
with np.errstate(all='raise'):
    for i, xi in enumerate(xs_skip):
        initial = xi
        for j, vi in tqdm(enumerate(vs)):
            try: # catches overflows
                vss[i][j][0] = vi
                zss[i][j][0] = xi
                for k in range(1, N + 1): # shooting with initial veloctiy
                    vss[i][j][k] = vss[i][j][k - 1] + zm(zss[i][j][k - 1])*dt
                    zss[i][j][k] = zss[i][j][k - 1] + vss[i][j][k - 1]*dt
                    if max(xs) < zss[i][j][k] or zss[i][j][k] < min(xs):
                       raise ValueError # still outside of our bounds
                plt.plot(ts, zss[i][j], color=cmap(xi / max(xs_skip)), linewidth=1)
            except:
                # when overflow occurs
                # reset data to zero, i.e., all zeros in a row means bad data
                # print('overflow at:', vi)
                vss[i][j] = np.zeros(zss.shape[-1])
                zss[i][j] = np.zeros(zss.shape[-1])

# np.where(a.any(2)) gives indicies for when it is non 0
# np.where(zss.any(2))

plt.xlabel('t')
plt.ylabel('x')
plt.title('most probable pathway')
# %%
np.save('data/bio_fss_xneg.npy', fss)
np.save('data/bio_zss_xneg.npy', zss)
np.save('data/bio_vss_xneg.npy', vss)
# plt.savefig('pics/simple z')
# plt.savefig('pics/bio_shooting.png')
# %%
zss = loadzneg = np.load('data/bio_zss_xneg.npy')
# loadzpos = np.load('data/bio_zss_xpos.npy')

vss = loadvneg = np.load('data/bio_vss_xneg.npy')
# loadvpos = np.load('data/bio_vss_xpos.npy')
fss = loadf = np.load('data/bio_fss_xneg.npy')

# %% investigating velocities and error to target
# zss = loadzneg = np.load('data/bio_zss_xneg.npy')
# vss = loadvneg = np.load('data/bio_vss_xneg.npy')
target = x_is['x+']
initial = x_is['x-']

idx = np.where(zss.any(2))

zz = zss[idx] # non trivial runs
vv = vss[idx] # non trivial runs

loss = (zz[:,-1] - target) ** 2

# vv[np.where(loss < 1e-2)]
min_loss = loss.argmin()
min_v = vv[min_loss]

plt.plot(vv[:, 0], loss, color='orange', label='loss')
plt.axvline(min_v[0], label=f'min velocity: {min_v[0]}')
plt.xlabel('velocity')
plt.ylabel('loss')
plt.title('loss given target and initial velocity')
plt.legend()
# %%
for i, xi in enumerate(xs):
    for run in range(2):
        plt.plot(ts, fss[i][run], color='black')

# from x- to x+ shooting method
zneg_idx = np.where(zss.any(2))
for i, z in enumerate(zss[zneg_idx]):
    plt.plot(ts, z, color=cmap(zss[zneg_idx][i, 0] / vss[zneg_idx][:,0].max()))

    # from x+ to x- shooting method
    # zpos_idx = np.where(loadzpos.any(2))
    # for i, z in enumerate(loadzpos[zpos_idx]):
    #     plt.plot(ts, z, color=cmap(loadvpos[zpos_idx][i, 0] / loadvpos[zpos_idx][:,0].min()))

    # min loss transition pathway
plt.plot(ts, zz[min_loss], color='purple', label=f'min action path', linewidth=2.75)
plt.legend()
plt.xlabel('t')
plt.ylabel('x')
plt.title('most probable transition pathway')

            # plt.savefig('pics/paper/bio_min_loss.pdf')
# %% final position z and initial velocity
plt.plot(vv[:, 0], zz[:, -1], color='red', label='loss')
plt.axvline(min_v[0], label=f'min velocity: {min_v[0]}', color='blue')
plt.axhline(target, label=f'target: {target}', color='brown')
plt.xlabel('velocity')
plt.ylabel('final z')
plt.title('final position given initial velocity')
plt.legend()
# %% plotted specifically wrt min loss
plt.plot(zz[min_loss], vv[min_loss], color='red', label='min z, v')

plt.xlim((0, 6))
plt.xlabel('x')
plt.ylabel('velocity')
plt.title('position vs velocity')
plt.legend()
# %%

for v, z in zip(vv, zz):
    plt.plot(z, v, color=cmap(v[0]/vv[:,0].max()))
plt.plot(zz[min_loss], vv[min_loss], color='black', linewidth=2.5, label='best')
plt.axvline(initial, color='brown', label=f'initial pos: {initial}')
plt.axvline(target, color='green', label=f'target pos: {target}')
plt.axvline(x_is['xu'], color='purple', label=f"unstable pos: {x_is['xu']}")
plt.xlim((0, 6))
plt.xlabel('x')
plt.ylabel('velocity')
plt.title('position vs velocity')
plt.legend()

# %%
init = x_is['x-']#  + vv[min_loss][0].mean()
init_sims = 50000
runs = []
for i in tqdm(range(init_sims)): # num for each initial value
    run = run_euler(init, dt, N, f, sigma)
    if run[-10:].mean() > 2:
        plt.plot(ts, run, alpha=0.5)
        runs.append(run)
print((len(runs) / 50000) * 100)
# %%
runs = np.load('data/neg_to_pos_runs.npy')

for run in runs:
    plt.plot(ts, run, alpha=0.3)
zneg_idx = np.where(loadzneg.any(2))
for i, z in enumerate(loadzneg[zneg_idx]):
    plt.plot(ts, z, color=cmap(loadvneg[zneg_idx][i, 0] / loadvneg[zneg_idx][:,0].max()), alpha=0.5)
vv[:, 0]
plt.plot(ts, np.mean(runs, 0), linewidth=3, color='black', label='mean path')
plt.plot(ts, zz[min_loss], color='purple', label=f'min action path', linewidth=2.75)
plt.axhline(x_is['x+'], label='x+', c='blue')
plt.axhline(x_is['x-'], label='x-', c='red')
plt.xlabel('t')
plt.ylabel('x')
plt.title('transition pathway simulation')
plt.legend()

plt.savefig('pics/pathway_velocities.pdf')
# plt.savefig('pics/pathway_mean.pdf')

# %% 3d pathway distribution plot data
from sklearn.neighbors import KernelDensity
import pandas as pd
from matplotlib import cm
# %matplotlib inline
# %%
# First import everthing you need
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

# Create an init function and the animate functions.
# Both are explained in the tutorial. Since we are changing
# the the elevation and azimuth and no objects are really
# changed on the plot we don't have to return anything from
# the init and animate function. (return value is explained
# in the tutorial.
ys = xs
position = np.array(runs)
# runs = pd.DataFrame(runs, index=time)

fig = plt.figure()
ax = fig.gca(projection='3d')
fig.set_size_inches(12, 8)

for i in range(len(runs)):
    ax.plot(xs=ts, ys=runs[i], zs=0, zdir='z', alpha=0.75)

# ax.plot(xs=ts, ys=np.mean(runs, 0), zs=0, zdir='z', linewidth=3, color='black', label='mean path')
ax.plot(xs=ts, ys=zz[min_loss], zs=0, zdir='z', color='purple', label=f'min action path', linewidth=4)

yss, probs, tss = [], [], []
count = 1
factor = (len(ts) // count + 1)
map_min2 = []
for i, t in enumerate(ts[::count]):
    slice = position[:, list(ts).index(t)][:, None]

    kde = KernelDensity(kernel='gaussian')
    kde.fit(slice)
    map_min2.append(np.exp(kde.score_samples([[zz[min_loss][i]]])))
    prob = np.exp(kde.score_samples(ys[:, None]))

    probs.append(prob)
    # ax.plot(ys=ys, zs=prob, xs=[t]*(len(ys)), c=cmap(i / factor))
probs = np.array(probs).transpose()
ax.plot_surface(TS, YS, probs, cmap='rainbow')


map_min2 = np.array(map_min2).reshape(-1)

ax.plot(xs=ts, ys=zz[min_loss], zs=map_min2, color='black', label=f'min action path', linewidth=10)

plt.title('3d pathway distribution')
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel('p(x)')
# ax.view_init(25, 160)
# plt.savefig()
# %%
# First import everthing you need
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

# Create an init function and the animate functions.
# Both are explained in the tutorial. Since we are changing
# the the elevation and azimuth and no objects are really
# changed on the plot we don't have to return anything from
# the init and animate function. (return value is explained
# in the tutorial.
ys = xs
position = np.array(runs)
# runs = pd.DataFrame(runs, index=time)

fig = plt.figure()
ax = fig.gca(projection='3d')
fig.set_size_inches(12, 8)

for i in range(len(runs)):
    ax.plot(xs=ts, ys=runs[i], zs=0, zdir='z', alpha=0.75)

# ax.plot(xs=ts, ys=np.mean(runs, 0), zs=0, zdir='z', linewidth=3, color='black', label='mean path')
ax.plot(xs=ts, ys=zz[min_loss], zs=0, zdir='z', color='black', label=f'min action path', linewidth=4)

yss, probs, tss = [], [], []
count = 1
factor = (len(ts) // count + 1)
map_min2 = []
for i, t in enumerate(ts[::count]):
    slice = position[:, list(ts).index(t)][:, None]

    kde = KernelDensity(kernel='gaussian')
    kde.fit(slice)
    map_min2.append(np.exp(kde.score_samples([[zz[min_loss][i]]])))
    prob = np.exp(kde.score_samples(ys[:, None]))

    ax.plot(ys=ys, zs=prob, xs=[t]*(len(ys)), c=cmap(i / factor))

map_min2 = np.array(map_min2).reshape(-1)

ax.plot(xs=ts, ys=zz[min_loss], zs=map_min2, color='black', label=f'min action path', linewidth=6)

plt.title('3d pathway distribution')
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel('p(x)')

# %% alt 3d graph
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

# Create an init function and the animate functions.
# Both are explained in the tutorial. Since we are changing
# the the elevation and azimuth and no objects are really
# changed on the plot we don't have to return anything from
# the init and animate function. (return value is explained
# in the tutorial.
ys = xs
position = np.array(runs)
# runs = pd.DataFrame(runs, index=time)

fig = plt.figure()
ax = fig.gca(projection='3d')
fig.set_size_inches(12, 8)

for i in range(len(runs)):
    ax.plot(xs=ts, ys=runs[i], zs=0, zdir='z', alpha=0.2)

# ax.plot(xs=ts, ys=np.mean(runs, 0), zs=0, zdir='z', linewidth=3, color='black', label='mean path')
# ax.plot(xs=ts, ys=zz[min_loss], zs=0, zdir='z', color='red', label=f'min action path', linewidth=4)

np.vstack([ts, zz[min_loss]]).transpose()

zz[min_loss]

start = 100
TS, YS = np.meshgrid(ts, ys)

map_min = []
ZS = []
for i in range(start, len(ts)):
    ZS.append(stats.gaussian_kde(position[:, i]).pdf(xs))
    map_min.append(stats.gaussian_kde(position[:, i]).pdf(zz[min_loss][i]))

ZS = np.array(ZS).transpose()
map_min = np.array(map_min)

ax.plot_surface(TS[:, start:], YS[:, start:], ZS, cmap='viridis')
ts.size, ys.size, map_min.size
ax.plot(xs=ts[start:], ys=zz[min_loss][start:], zs=map_min, color='red', label=f'min action path', linewidth=4)


plt.title('3d pathway distribution')
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel('p(x)')
ax.view_init(elev=25, azim=100)
# %%
import os

i = 0
for angle in range(0, 360):
    ax.view_init(elev=25, azim=angle)
    fig.savefig('pics/movie/movie-%d.png' % i)
    i += 1

# run this code in the cmd in the directory `pics/movie`
# ffmpeg -r 30 -i movie%d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p out.mp4
