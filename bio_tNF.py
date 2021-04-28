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

data = data[:, :, 0:400] # take only first i to j time steps

plt.plot(data[:, 0, :].transpose());
# %%
x, runs, N = data.shape
N = N - 1
position = data[:, 0, :].transpose()
x_sample = data[:, 0, 0] # initial x's
t_sample = np.linspace(0, 50, N + 1) # times
time = t_sample
dataset = prepare_data(position, time, x_sample, t_sample)

# %%
iters = 1000
flow_dpx = neural_flow(gaussian)
flow = neural_flow(gaussian)
flow.load_state_dict(torch.load(f'temporal_normalizing_flows/data/model_N={N}_iter={iters}.pth', map_location=torch.device('cpu')))
flow_dpx.load_state_dict(torch.load(f'temporal_normalizing_flows/data/model_dpx_N={N}_iter={iters}.pth', map_location=torch.device('cpu')))
# %%
flow_dpx.train_dpx(dataset, iters)
flow.train(dataset, iters)

torch.save(flow.state_dict(), f'temporal_normalizing_flows/data/model_N={N}_iter={iters}.pth')

# %%
px, pz, jacob, z = flow.sample(dataset)
dpx, dpz, djacob, dz = flow_dpx.sample(dataset)

# %%
import matplotlib.animation as animation

Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=6000)
def animate(frame):
    plt.clf()
    sns.distplot(position[frame, :], bins=25, label='KDE')
    plt.plot(x_sample, px[frame,:],'r', label='tNF')
    plt.plot(x_sample, dpx[frame,:],'g', label='dtNF')
    plt.xlim(0, 6)
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.title(f't={round(time[frame], 2)}')
    plt.legend(loc=0,ncol=1)

fig = plt.figure()

ani = animation.FuncAnimation(fig, animate, frames=N - 1, repeat=True)
ani.save(f'tnf_N={N}_iter={iters}_dpx.mp4')

# %%
log_px_grid = self.forward(dataset)[0]

px, _, _, _ = self.sample(dataset)
plt.plot(px);
dpx = np.diff(px, axis=0)
plt.plot(dpx / np.max(dpx));
log_px_samples
torch.tensor(dpx / np.max(dpx)) * log_px_samples[:-1]
# print(log_px_grid.shape)
# print(dataset.grid_data.shape)
log_px_samples = self.sample_grid(log_px_grid, dataset)

plt.plot(np.exp(log_px_samples.detach().numpy())[300]);
dpx = np.gradient(px)
dpx = dpx[0]

log_px_adjusted = torch.tensor(dpx) * log_px_samples
loss = -torch.mean(log_px_adjusted)

# loss = -torch.mean(log_px_samples)
loss.backward()
optimizer.step()
optimizer.zero_grad()


dpx = np.exp(np.diff(log_px_samples.detach().numpy(), axis=0))
log_px_adjusted = torch.tensor(dpx) * log_px_samples[:-1]
loss = -torch.mean(log_px_adjusted)

plt.plot(np.diff(px);

# %% plot individual time slices
frame = 40

sns.distplot(position[frame, :], bins=25, label='KDE')
plt.plot(x_sample, px[frame,:],'r', label='tNF')
plt.plot(x_sample, dpx[frame,:],'g', label='dtNF')
plt.xlim(0, 6)
plt.xlabel('x')
plt.ylabel('p(x)')
plt.title(f't={round(time[frame], 2)}')
plt.legend(loc=0,ncol=1)

# %%
dpx = np.zeros((px.shape[0] - 1, px.shape[1]))

for i in range(dpx.shape[0]):
    dpx[i] = px[i + 1] - px[i]

dpx.mean(1)
plt.plot(dpx[100]);

plt.plot(px[101] - px[100]);
dpx = np.diff(px, axis=0)
dpx.shape
