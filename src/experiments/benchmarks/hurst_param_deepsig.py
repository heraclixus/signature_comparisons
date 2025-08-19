import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import iisignature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from models.deep_signature_transform.siglayer import examples
import torch
import torch.nn.functional as F
import torch.optim as optim

from models.deep_signature_transform import hurst_parameter, utils

plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

# dataset parameters
n_paths_train=600
n_paths_test=100 
n_samples=300
hurst_exponents=np.around(np.linspace(0.2, 0.8, 7), decimals=1).tolist()

# target shape
output_shape = (1,)

# batch and epoch sizes
train_batch_size = 128
val_batch_size = 128
max_epochs = 100

optimizer_fn = optim.Adam

def loss_fn(x,y):
    return torch.log(F.mse_loss(x, y))

history = {}
x_train, y_train, x_test, y_test = hurst_parameter.generate_data(n_paths_train, 
                                                                 n_paths_test, 
                                                                 n_samples, 
                                                                 hurst_exponents)

x_train_, x_test_ = hurst_parameter.preprocess_data(x_train, x_test)

(train_dataloader, test_dataloader, 
 example_batch_x, example_batch_y) = hurst_parameter.generate_torch_batched_data(x_train_, 
                                                                                 y_train, 
                                                                                 x_test_, 
                                                                                 y_test,
                                                                                 train_batch_size, 
                                                                                 val_batch_size)

train_model = utils.create_train_model_fn(max_epochs, optimizer_fn, loss_fn, train_dataloader, 
                                          test_dataloader, example_batch_x)


feedforward = examples.create_simple(output_shape, sig=False, augment_layer_sizes=(), 
                                     layer_sizes = (16, 16, 16),
                                     final_nonlinearity=torch.sigmoid)
train_model(feedforward, 'Feedforward', history)

rnn = hurst_parameter.deep_recurrent(output_shape, 
                                     sig=False,
                                     augment_layer_sizes=(), 
                                     layer_sizes_s=((64,64,32), (32,32,32)),
                                     lengths=(4,4), 
                                     strides=(2,4), 
                                     adjust_lengths=(0, 0),
                                     memory_sizes=(2,4),
                                     hidden_output_sizes=(4,),
                                     final_nonlinearity=torch.sigmoid)
train_model(rnn, 'RNN', history)


deepsignet = examples.create_simple(output_shape,
                                    sig=True,
                                    sig_depth=3,
                                    augment_layer_sizes=(3,),
                                    augment_kernel_size=3,
                                    layer_sizes = (32, 32, 32, 32, 32),
                                     final_nonlinearity=torch.sigmoid)
train_model(deepsignet, 'DeepSigNet', history)


deepersignet = hurst_parameter.deep_recurrent(output_shape, 
                                              sig=True, 
                                              sig_depth=3,
                                              augment_layer_sizes=(16, 16, 3), 
                                              augment_kernel_size=4,
                                              lengths=(10, 10, 10), 
                                              strides=(0, 0, 0), 
                                              adjust_lengths=(5, 5, 5),
                                              layer_sizes_s=((16, 16), (16, 16), (16, 16)), 
                                              memory_sizes=(8, 8, 8),
                                              hidden_output_sizes=(5, 5),
                                              final_nonlinearity=torch.sigmoid)
train_model(deepersignet, 'DeeperSigNet', history)


x_train_, x_test_ = hurst_parameter.preprocess_data(x_train, x_test, flag='lstm')

(train_dataloader_lstm, test_dataloader_lstm, 
 example_batch_lstm_x, example_batch_lstm_y) = hurst_parameter.generate_torch_batched_data(x_train_, 
                                                                                           y_train,
                                                                                           x_test_, 
                                                                                           y_test,
                                                                                           train_batch_size,
                                                                                           val_batch_size)

train_model_lstm = utils.create_train_model_fn(max_epochs, 
                                               optimizer_fn, 
                                               loss_fn, 
                                               train_dataloader_lstm, 
                                               test_dataloader_lstm, 
                                               example_batch_lstm_x)


lstmnet = hurst_parameter.LSTM(input_dim=1, 
                               num_layers=2,
                               hidden_dim=32,
                               output_dim=1,
                               final_nonlinearity=torch.sigmoid)
train_model_lstm(lstmnet, 'LSTM', history)


grunet = hurst_parameter.GRU(input_dim=1, 
                             num_layers=2, 
                             hidden_dim=32,
                             output_dim=1,
                             final_nonlinearity=torch.sigmoid)
train_model_lstm(grunet, 'GRU', history)



# generate dataset
x_train_, x_test_ = hurst_parameter.preprocess_data(x_train, x_test, flag='neuralsig')

# generate torch dataloaders
(train_dataloader_sig, test_dataloader_sig, 
 example_batch_sig_x, example_batch_sig_y) = hurst_parameter.generate_torch_batched_data(x_train_,
                                                                                         y_train,
                                                                                         x_test_,
                                                                                         y_test,
                                                                                         train_batch_size,
                                                                                         val_batch_size)

# trainer function
train_model_sig = utils.create_train_model_fn(max_epochs, 
                                              optimizer_fn, 
                                              loss_fn, 
                                              train_dataloader_sig, 
                                              test_dataloader_sig, 
                                              example_batch_sig_x)


neuralsig = examples.create_feedforward(output_shape, sig=False, 
                                        layer_sizes=(64, 64, 32, 32, 16, 16),
                                        final_nonlinearity=torch.sigmoid)
train_model_sig(neuralsig, 'Neural-Sig', history)



params = {}
for k, m in zip(('DeeperSigNet', 'DeepSigNet', 'Neural-Sig', 'LSTM', 'GRU', 'RNN', 'Feedforward'), 
                (deepersignet, deepsignet, neuralsig, lstmnet, grunet, rnn, feedforward)):
    params[k] = utils.count_parameters(m)

for key in history:
    print('{:12} {:6.4f} {}'.format(key, history[key]['val_loss'][-1], params[key]))

# Loss for the non-neural-network mathematically-derived rescaled range method
rescaled_range_pred = [hurst_parameter.hurst_rescaled_range(x_test_i) for x_test_i in x_test]
loss_fn(torch.Tensor(rescaled_range_pred), torch.Tensor(y_test))



# adapted from jet
colors = np.array([[0.5       , 0.5       , 0.5       , 1.        ],
                   [0.        , 0.06470588, 1.        , 1.        ],
                   [0.        , 0.64509804, 1.        , 1.        ],
                   [0.05882352, 0.51764705, 0.17647058, 1.        ],
                   [0.9       , 0.7       , 0.        , 1.        ],
                   [1.        , 0.18954248, 0.        , 1.        ],
                   [0.28627450, 0.18823529, 0.06666666, 1.        ]])

# define pd dataframe for losses
df_test_log = pd.DataFrame()
for k in ('Feedforward', 'RNN', 'GRU', 'LSTM', 'Neural-Sig', 'DeepSigNet', 'DeeperSigNet'):
    df_test_log[k] = history[k]['val_loss']

fig, axes = plt.subplots(figsize=(10, 8))
np.power(np.e, df_test_log.rolling(5).mean()).plot(grid=False, ax=axes, color=colors, lw=1.5, alpha=0.8)
plt.yscale('log', basey=10)
axes.set_xlabel('Epoch')
axes.set_ylabel('Test MSE')
plt.legend(mode='expand', bbox_to_anchor=(0, 1, 1, 0), ncol=3, prop={'size': 18})

plt.savefig('hurst_param_deepsig.png')