import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set()

#from diffusioncharacterization.ctrw.random_walks import advection_diffusion_random_walk
from temporal_normalizing_flows.neural_flow import neural_flow
from temporal_normalizing_flows.latent_distributions import gaussian
from temporal_normalizing_flows.preprocessing import prepare_data

try:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')  # enable for GPU
except:
    pass

# %%
data = np.load('data/bio_fss.npy')

data = data[:, :, 0:250] # take only first i to j time steps

plt.plot(data[:, 0, :].transpose());
# %%
x, runs, N = data.shape
N = N - 1
position = data[:, 0, :].transpose()
x_sample = data[:, 0, 0] # initial x's
t_sample = np.linspace(0, 50, N + 1) # times
time = t_sample
dataset = prepare_data(position, time, x_sample, t_sample)

iters = 5000
flow = neural_flow(gaussian)
flow.train(dataset, iters)

px, pz, jacob, z = flow.sample(dataset)

torch.save(flow.state_dict(), f'temporal_normalizing_flows/data/model_N={N}_iter={iters}.pth')

# %%
import matplotlib.animation as animation

Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
def animate(frame):
    plt.clf()
    x
    sns.distplot(position[frame, :], bins=25, label='KDE')
    plt.plot(x_sample, px[frame,:],'r', label='tNF')
    plt.xlim(0, 6)
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.title(f't={round(time[frame], 2)}')
    plt.legend(loc=0,ncol=1)

fig = plt.figure()

ani = animation.FuncAnimation(fig, animate, frames=N // 2, repeat=True)
ani.save('tnf.mp4')
