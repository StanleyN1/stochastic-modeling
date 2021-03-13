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
num_of_runs = 10
N = 1000
t_0, t_f = 0, 50

n_k = 10
split = N // n_k # number of split time intervals
dt = (t_f - t_0) / N
ts = np.linspace(t_0, t_f, num=N + 1)

x_is = {'x+': 4.28343, 'x-': 0.62685, 'xu': 1.48971}
N_x = 100
xs = np.linspace(0, 6, num=N_x + 1)
# %%
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
plt.plot(xs, [sigma(x) for x in xs], label='$\sigma$')

plt.axvline(x_is['x+'], label='$x_+$', c='blue')
plt.axvline(x_is['x-'], label='$x_-$', c='red')
plt.axvline(x_is['xu'], label='$x_u$', c='black')

plt.legend(loc='best')
plt.xlabel('x')
plt.title('drift and diffusion')
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
data = kramers_moyal(xs, fss, (0, 1))
plt.plot(xs, f(xs), label='f') # exact drift function
# plt.plot(xs, -F(f, xs), label='U') # exact potential of drift
plt.plot(xs, data['f'], label='f approx') # approximate drift
# plt.plot(xs, data['s'], label='s approx') # approximate diffusion
plt.legend()
plt.xlabel('x')
plt.xlabel('f')
plt.title('f and approximation')
# plt.savefig('pics/paper/f_and_approx.pdf')
# %% experiment when consider more intervals than first slice

plt.plot(xs, f(xs), label='f')
for i in range(1, split, 10):
    data = kramers_moyal(xs, fss, (i, i + 5))
    plt.plot(xs, data['f'], label=f'f approx from (0, {i})')
plt.axvline(x_is['x+'], label='$x_+$', c='blue')
plt.axvline(x_is['x-'], label='$x_-$', c='red')
plt.axvline(x_is['xu'], label='$x_u$', c='black')
    # plt.plot(xs, data['s'], label='s approx')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# %% polynomial interpolation and approximated data
fss = np.load('data/bio_fss.npy')

approx = kramers_moyal(xs, fss, (0, 1)) # most accurate data
polyf = np.poly1d(np.polyfit(xs, approx['f'], deg=3))
polys = np.poly1d(np.polyfit(xs, approx['s'], deg=2))
# plt.plot(xs, np.polyval(polyf, xs), label='f poly')
# plt.plot(xs, f(xs), label='f')
plt.plot(xs, [sigma(x) for x in xs], label='$\sigma$: 0.10')
plt.plot(xs, np.polyval(polys, xs), label=f'$\sigma$: {polys}')
# plt.plot(xs, -F(f, xs), label='F poly')
# plt.plot(xs, -F(polyf, xs), label='F')

# plt.plot(xs, approx['f'], label='f approx') # approximate drift
# plt.plot(xs, approx['s'], label='$\sigma$ approx')
print(polys)
plt.xlabel('x')
plt.title('approximation and interpolation')
plt.legend()
# %%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# z = ...
# z(0) = x-
# z(T) = x+
def fun(x, y):
    return zm(y[1]), y[0]

lo = -1
hi = 1
count = 0
target = x_is['x+']
initial = x_is['x-']
tol = 1e-4
while count < 10:
    guess = np.mean([lo, hi])
    # print(guess)
    sol = solve_ivp(fun, (t_0, t_f), [initial, guess], t_eval=ts)
    yf = sol.y[0][-5:].mean()
    print(sol.y)
    if abs(yf - target) < tol:
        break
    if yf < target:
        hi = guess
    else:
        lo = guess
    count += 1
print()
print(guess)
# %%

xspan = (xs[0], xs[-1])
y0 = [3, 5]

sol = solve_ivp(fun, xspan, y0, t_eval=xs)

y_num = sol.y[0, :]

plt.plot(xs, y_num)

# def bc(u0, ub):
#     return u0[0] - x_is['x-'], ub[1] - x_is['x+']
#
# ya = np.zeros((2, xs.size))
# ya[0] = 0.1
# bvp = solve_bvp(fun, bc, xs, ya)
# x_is['x-']
# bvp['sol'](6)
# bvp['sol'](xs)


# %% shooting method
from scipy.integrate import solve_bvp, solve_ivp
# Cheng et al. eq (7) gives formula for most probable transition pathway
fss = np.load('data/bio_fss.npy')

data = kramers_moyal(xs, fss, (0, 1))
polyf = np.poly1d(np.polyfit(xs, data['f'], deg=3))
polys = np.poly1d(np.polyfit(xs, data['s'], deg=0))
zm = (polys ** 2 / 2) * np.polyder(polyf, m=2) + np.polyder(polyf, m=1)*polyf
# %%
for i, xi in tqdm(enumerate(xs)):
    for run in fss[i]:
        plt.plot(ts, run, color='black')


plt.plot(xs, bvp['y'][1])
target = x_is['x+']

vs = np.linspace(0, 3, 201)
xs_skip = [x_is['x-']] # xs[::]
zss = np.zeros((len(xs_skip), len(vs), N + 1))
vss = np.zeros_like(zss)
with np.errstate(all='raise'):
    for i, xi in enumerate(xs_skip):
        initial = xi
        for j, vi in tqdm(enumerate(vs)):
            try: # catches overflows
                solve_bvp(zm, )
                #vss[i][j][0] = vi
                #zss[i][j][0] = xi
                #for k in range(1, N + 1): # shooting with initial veloctiy
                    #vss[i][j][k] = vss[i][j][k - 1] + zm(zss[i][j][k - 1])*dt
                    #zss[i][j][k] = zss[i][j][k - 1] + vss[i][j][k - 1]*dt
                    #if max(xs) < zss[i][j][k] or zss[i][j][k] < min(xs):
                    #    raise ValueError # still outside of our bounds
                #plt.plot(ts, zss[i][j], color=cmap(xi / max(xs_skip)), linewidth=1)
            except:
                # when overflow occurs
                # reset data to zero, i.e., all zeros in a row means bad data
                print('overflow at:', vi)
                vss[i][j] = np.zeros(zss.shape[-1])
                zss[i][j] = np.zeros(zss.shape[-1])

# np.where(a.any(2)) gives indicies for when it is non 0

# np.where(zss.any(2))
# vss[0][76]

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
import seaborn as sns

loadzneg = np.load('data/bio_zss_xneg.npy')
# loadzpos = np.load('data/bio_zss_xpos.npy')

loadvneg = np.load('data/bio_vss_xneg.npy')
# loadvpos = np.load('data/bio_vss_xpos.npy')
loadf = np.load('data/bio_fss_xneg.npy')

for i, xi in enumerate(xs):
    for run in range(2):
        plt.plot(ts, fss[i][run], color='black')

# from x- to x+ shooting method
zneg_idx = np.where(loadzneg.any(2))
for i, z in enumerate(loadzneg[zneg_idx]):
    plt.plot(ts, z, color=cmap(loadvneg[zneg_idx][i, 0] / loadvneg[zneg_idx][:,0].max()))

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

# %% investigating velocities and error to target
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

plt.plot(vv[:, 0], loss, color='orange', label='loss')
plt.axvline(min_v[0], label=f'min velocity: {min_v[0]}')
plt.xlabel('velocity')
plt.ylabel('loss')
plt.title('loss given target and initial velocity')
plt.legend()
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
plt.plot(zz[min_loss], vv[min_loss], color='black', linewidth=2, label='best')
plt.axvline(initial, color='brown', label=f'initial pos: {initial}')
plt.xlim((0, 6))
plt.xlabel('x')
plt.ylabel('velocity')
plt.title('position vs velocity')
plt.legend()

# %%
init = x_is['x-'] + vv[min_loss][0].mean()/dt
init_sims = 1000
for i in tqdm(range(init_sims)): # num for each initial value
    run = run_euler(init, dt, N, f, sigma)
    plt.plot(ts, run)

vv[-5:]
plt.plot(ts, zz[min_loss], color='purple', label=f'min action path', linewidth=2.75)
