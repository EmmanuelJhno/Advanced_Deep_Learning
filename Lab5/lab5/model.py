from torch import nn, relu
from torch.utils.data import DataLoader, TensorDataset


class Feedforward(nn.Module):
    def __init__(self, n_hidden_layers=2, n_neurons=7, hidden_activations=relu,
                 output_activation=None):
        super(Feedforward, self).__init__()
        self.n_layers = n_hidden_layers
        if type(n_neurons) == int:
            self.n_neurons = [3] + [n_neurons for self in range(n_hidden_layers)]
        else:
            assert type(n_neurons) == list
            assert len(n_neurons) == n_hidden_layers
            self.n_neurons = [3] + n_neurons

        for i in range(self.n_layers):
            input_dim = self.n_neurons[i]
            output_dim = self.n_neurons[i + 1]
            setattr(self,
                    "layer_{}".format(i + 1),
                    nn.Linear(input_dim, output_dim))

        self.out = nn.Linear(self.n_neurons[-1], 3)

        self.hidden_activations = hidden_activations
        self.output_activation = output_activation

    def forward(self, inputs):
        x = inputs
        for i in range(self.n_layers):
            x = getattr(self, "layer_{}".format(i + 1))(x)
            if i < self.n_layers - 1:
                x = self.hidden_activations(x)
        outputs = self.out(x)
        if self.output_activation is not None:
            outputs = self.output_activation(outputs)
        return outputs.view((-1, 3))


def my_custom_loss(x_pred, x_true, dot_xt_pred, dot_xt_true, delta_t):
    loss = nn.functional.l1_loss(dot_xt_true, dot_xt_pred)
    for i in range(len(x_true)):
        loss += 1 / (len(x_true) - i) * nn.functional.mse_loss(x_true[i], x_pred[i])

    return loss


def train(num_epochs, batch_size, criterion, optimizer, model, dataset, delta_t, display=True):
    train_error = []
    train_loader = DataLoader(dataset, batch_size, shuffle=True)
    model.train()
    for epoch in range(num_epochs):
        epoch_average_loss = 0.0
        for (xt, xtp1, xtp2, xtp3, dot_xt, dot_xtp1, dot_xtp2) in train_loader:
            dot_xt_pred = model(xt.float())
            xtp1_pred = xt + dot_xt_pred * delta_t

            dot_xtp1_pred = model(xtp1_pred.float())
            xtp2_pred = xtp1 + dot_xtp1_pred * delta_t

            dot_xtp2_pred = model(xtp2_pred.float())
            xtp3_pred = xtp2 + dot_xtp2_pred * delta_t

            loss = criterion(x_pred=[xtp1_pred, xtp2_pred, xtp3_pred],
                             x_true=[xtp1, xtp2, xtp3],
                             dot_xt_true=dot_xt,
                             dot_xt_pred=dot_xt_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_average_loss += loss.item() * batch_size / len(dataset)
        train_error.append(epoch_average_loss)
        if display:
            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, epoch_average_loss))
    return train_error


def temporal_derivative(v_t, v_tp1, delta_t):
    return (v_tp1 - v_t) / delta_t
