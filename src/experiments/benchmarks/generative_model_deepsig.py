import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as torchdata

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from dataset.generative_model import gen_data
from models.deep_signature_transform.scripts import utils
from dataset import generative_model

plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

train_batch_size = 2 ** 10
val_batch_size = 2 ** 10
max_epochs = 100

optimizer_fn = lambda x: optim.Adam(x, lr=0.01)

n_points = 100

train_dataset = generative_model.get_noise(n_points=n_points, num_samples=train_batch_size)
eval_dataset = generative_model.get_noise(n_points=n_points, num_samples=val_batch_size)
signals = generative_model.get_signal(num_samples=train_batch_size, n_points=n_points,).tensors[0]

train_dataloader = torchdata.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=0)
eval_dataloader = torchdata.DataLoader(eval_dataset, batch_size=val_batch_size, shuffle=False, num_workers=0)

example_batch, _ = next(iter(train_dataloader))
example = example_batch[0]

print(f'Feature shape: {tuple(example.shape)}')
plt.plot(*example.numpy())
for path in signals[:100]:
    plt.plot(*path.numpy(), "orange", alpha=0.1)
plt.savefig('example_data.png')

loss_fn = generative_model.loss(signals, sig_depth=4, normalise_sigs=True)
model = generative_model.create_generative_model()

history = {}
train_model = utils.create_train_model_fn(max_epochs, optimizer_fn, loss_fn, train_dataloader, eval_dataloader, 
                                          example_batch)
train_model(model, 'generative_model', history)


fig, axs = plt.subplots(1, 2, gridspec_kw={'wspace': 0.6, 'hspace': 0.6}, figsize=(12, 4))
axs = axs.flatten()
for i, metric_name in enumerate(('train_loss', 'val_loss')):
    ax = axs[i]
    for model_history in history.values():
        metric = model_history[metric_name]

        # Moving average
        metric = np.convolve(metric, np.ones(10), 'valid') / 10.
        ax.semilogy(np.exp(metric))
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric_name)


batch, _ = next(iter(eval_dataloader))
batch = batch.to(device=next(model.parameters()).device)
generated = model(batch).cpu()
plt.figure(figsize=(12, 8))
plt.plot(generated[50:100].detach().numpy().T, "b", alpha=0.2)
plt.plot(signals[50:100, 1, :99].detach().numpy().T, "#ba0404", alpha=0.2)

orange_patch = mpatches.Patch(color='#ba0404', label='Ornstein–Uhlenbeck process')
blue_patch = mpatches.Patch(color='blue', label='Generated paths')
plt.legend(mode='expand', ncol=2, prop={'size': 18}, bbox_to_anchor=(0, 1, 1, 0), 
           handles=[blue_patch, orange_patch])
plt.ylim([-2,2])
plt.yticks([-2, -1, 0, 1, 2])

plt.savefig('generated_data.png')