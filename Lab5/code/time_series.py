import pickle
import torch


class Rossler_model:
    def __init__(self, delta_t, model_path, traj_path):
        self.delta_t = delta_t #if discrete model your delta_t
                              #if continuous model chose one <=1e-2
        self.nb_steps = int(10000//self.delta_t)

        self.model_path = model_path
        self.traj_path = traj_path

        self.rosler_nn = pickle.load(open(self.model_path, 'rb'))

    def full_traj(self,initial_condition=[-5.75, -1.6,  0.02]):
        # run your model to generate the time series with nb_steps
        # just the y cordinate is necessary.
        y=[]
        with torch.no_grad():

            y.append(torch.tensor(initial_condition))

            for k in range(self.nb_steps -1):

                y.append(self.rosler_nn(y[-1]))

        return y

    def save_traj(self,y):

        pickle.dump(y, open(self.traj_path, 'wb'),protocol=4)
