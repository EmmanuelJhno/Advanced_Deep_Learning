import numpy as np
import pickle
import os
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

class RosslerMap:
    """
    Rossler attractor
    With a=0.2, b=0.2, and c=5.7
    """

    def __init__(_, a=0.2, b=0.2, c=5.7, delta_t=1e-3):
        _.a, _.b, _.c = a, b, c
        _.delta_t = delta_t

    def v_eq(_, t=None, v=None):
        x, y, z = v[0], v[1], v[2]
        dot_x = -y - z
        dot_y = x + _.a*y
        dot_z = _.b + z*(x-_.c)
        return np.array([dot_x, dot_y, dot_z])

    def jacobian(_, v):
        if len(v.shape)==1:
            v = np.array([v])
        x, z = v[:, 0], v[:, 2]
        J = np.repeat([[[       0,      -1,       -1],
                        [       1,     _.a,        0],
                        [       0,       0,     -_.c]]], v.shape[0],
                                axis=2)
        J[:, 2, 0] += z
        J[:, 2, 2] += x
        return J
        res = np.array([[       0,      -1,       -1],
                        [       1,     _.a,        0],
                        [       z,       0,   x-_.c]])
        return res

    def full_traj(_, nb_steps, init_pos):
        t = np.linspace(0, nb_steps * _.delta_t, nb_steps)
        f = solve_ivp(_.v_eq, [0, nb_steps * _.delta_t], init_pos, method='RK45', t_eval=t)
        return np.moveaxis(f.y, -1, 0),t

    def equilibrium(_):
        x0 = (_.c - np.sqrt(_.c**2 - 4*_.a*_.b)) / 2
        y0 = (-_.c + np.sqrt(_.c**2 - 4*_.a*_.b)) / (2*_.a)
        z0 = (_.c - np.sqrt(_.c**2 - 4*_.a*_.b)) / (2*_.a)
        return np.array([x0,y0,z0])

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.l1 = nn.Linear(3, 100)
        self.l2 = nn.Linear(100, 100)
        self.l3 = nn.Linear(100, 3)

    def forward(self, inputs):

        outputs = torch.relu(self.l1(inputs))
        outputs = torch.relu(self.l2(outputs))
        outputs = torch.relu(self.l3(outputs))
        return outputs



class Feedforward(torch.nn.Module):
    def __init__(self, n_hidden_layers=2, n_neurons=7, hidden_activations=torch.relu,
                 output_activation=None):
        super(Feedforward, self).__init__()
        self.n_layers = n_hidden_layers
        if type(n_neurons)==int:
            self.n_neurons = [3] + [n_neurons for _ in range(n_hidden_layers)]
        else:
            assert type(n_neurons)==list
            assert len(n_neurons)==n_hidden_layers
            self.n_neurons = [3] + n_neurons

        for i in range(self.n_layers):
            input_dim = self.n_neurons[i]
            output_dim = self.n_neurons[i+1]
            setattr(self,
                    "layer_{}".format(i+1),
                    nn.Linear(input_dim, output_dim))

        self.out = nn.Linear(self.n_neurons[-1], 3)

        self.hidden_activations = hidden_activations
        self.output_activation = output_activation

    def forward(self, inputs):
        x = inputs
        for i in range(self.n_layers):
            x = getattr(self, "layer_{}".format(i+1))(x)
            if i < self.n_layers - 1:
                x = self.hidden_activations(x)
        outputs = self.out(x)
        if self.output_activation is not None:
            outputs = self.output_activation(outputs)
        return outputs


def my_custom_loss(y_true_t, y_pred_t, y_true_t1, y_pred_t1):
    loss = torch.sum(torch.abs(y_true_t - y_pred_t)) \
            + torch.sum(torch.abs(y_true_t1 - y_pred_t1)) \
            + torch.sum(torch.abs((y_true_t1 - y_true_t) \
                                    - (y_pred_t1 - y_pred_t)))
    return loss


def train(num_epochs, batch_size, criterion, optimizer, model, dataset, display=True):
    train_error = []
    train_loader = DataLoader(dataset, batch_size, shuffle=True)
    model.train()
    for epoch in range(num_epochs):
        epoch_average_loss = 0.0
        for (X_batch, y_real) in train_loader:
            if X_batch.shape[1]==6:
                X_batch_t, X_batch_t1 = X_batch[:,:3], X_batch[:,3:]
                y_real_t, y_real_t1 = y_real[:,:3], y_real[:,3:]
                y_pred_t = model(torch.tensor(X_batch_t.float()))
                y_pred_t1 = model(torch.tensor(X_batch_t1.float()))
                loss = criterion(y_real_t.float(), y_pred_t, y_real_t1.float(), y_pred_t1)
            else:
                pred = model(torch.tensor(X_batch.float()))
                loss = criterion(pred, y_real.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_average_loss += loss.item() * batch_size / len(dataset)
        train_error.append(epoch_average_loss)
        if display:
            print('Epoch [{}/{}], Loss: {:.4f}'
                          .format(epoch+1, num_epochs, epoch_average_loss))
    return train_error
