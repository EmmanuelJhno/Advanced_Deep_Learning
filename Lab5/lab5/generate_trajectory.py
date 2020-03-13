from model import Feedforward
import numpy as np
import pickle
import torch


class Model:
    def __init__(self, delta_t, model_path, traj_path):
        self.delta_t = delta_t #if discrete model your delta_t
                              #if continuous model chose one <=1e-2
        self.nb_steps = int(10_000//self.delta_t)

        self.model_path = model_path
        self.traj_path = traj_path

        self.rosler_nn = pickle.load(open(self.model_path, 'rb'))

    def full_traj(self,initial_condition=[-5.75, -1.6,  0.02]):
        # run your model to generate the time series with nb_steps
        # just the y cordinate is necessary.
        y = [np.array(initial_condition)]
        with torch.no_grad():
            for k in range(self.nb_steps - 1):
                x = torch.from_numpy(y[-1]).float()
                dot_x = self.rosler_nn(x).squeeze().detach()
                next_x = (x + self.delta_t * dot_x).detach()
                y.append(next_x.numpy())
        return np.array(y)

    def save_traj(self, y):
        pickle.dump(y, open(self.traj_path, 'wb'), protocol=4)


if __name__ == '__main__':
    model = Model(delta_t=1e-3, model_path="model_Jehanno_Portier_Brandeis.pickle", traj_path="trajectory.pickle")
    traj = model.full_traj()
    model.save_traj(traj)
