"""
Source code to generate figures
"""
import argparse
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import pickle
import statsmodels.api
from statsmodels.tsa.stattools import acf
import torch
from tqdm import tqdm

from rossler_map import RosslerMap
from model import Feedforward


DELTA_T = 1e-3
logging.basicConfig(level=logging.INFO)

def load_model(path_to_model):
    """
    Loads a pickled model
    Args:
        path_to_model (str): Path to the .pickle file containing the model

    Returns:
        torch.nn.Module: the pre-trained NN
    """
    model_path = os.path.join(path_to_model)
    model = pickle.load(open(model_path, 'rb'))
    return model


def generate_trajectories(start_pos, n_points, model, rossler_map, delta_t=DELTA_T):
    """
    Generates real and predicted trajectories

    Args:
        start_pos (numpy.ndarray): The initial position. (3,) shaped numpy array.
        n_points (int): Number of points to generate in the trajectories.
        model (torch.nn.Module): Pre-trained model used for prediction.
        rossler_map (rossler_map.RosslerMap): Rossler Map instance, used to generate the real
            trajectory
        delta_t (float): Time step used for discrete differentiation.

    Returns:
        (numpy.ndarray, numpy.ndarray): The real and predicted trajectories
    """
    trajectory_real, t = rossler_map.full_traj(n_points, start_pos)

    with torch.no_grad():
        trajectory_simul = [torch.tensor(start_pos).detach().numpy()]

    # Predict the next state and add it to the training
    with torch.no_grad():
        for k in tqdm(range(n_points - 1)):
            x = torch.from_numpy(trajectory_simul[-1])
            dot_x = model(x).squeeze().detach()
            next_x = (x + delta_t * dot_x).detach()
            trajectory_simul.append(next_x.numpy())
    trajectory_simul = np.array(trajectory_simul)
    return trajectory_real, trajectory_simul


def plot_pos_hist(traj_real, traj_pred, length=50_000, save_file=None):
    """
    Plots repartition of positions along coordinates for real and predicted trajectories,
    in the form of histogram plots.

    Args:
        traj_real (numpy.ndarray): Real trajectory
        traj_pred (numpy.ndarray): Predicted trajectory
        length (int): Length of trajectory to consider
        save_file (str): (Optional) Path where to save the figure as a .png image. Set to None for no saving.

    Returns:
        matplotlib.pyplot.Figure: The figure containing the plots
    """
    fig = plt.figure(figsize=(12, 12))
    for i, name in [(0, 'x'), (1, 'y'), (2, 'z')]:
        fig.suptitle('Repartition of position along coordinates')
        plt.subplot(3, 1, i + 1)
        plt.hist(traj_real[:length, i], label="real coordinate", color="lightcoral", alpha=0.8)
        plt.title('True trajectory - {}'.format(name))
        plt.legend()
        plt.grid()

        plt.hist(traj_pred[:length, i], label="simulated coordinate", color="lightblue", alpha=0.8)
        plt.title('Predicted trajectory - {}'.format(name))
        plt.legend()
        plt.grid()

    if save_file:
        plt.savefig(save_file)

    return fig


def plot_traj_axis(traj_real, traj_pred, length=50_000, save_file=None):
    """
    Plots trajectories (real and predicted) along each coordinate

    Args:
        traj_real (numpy.ndarray): Real trajectory
        traj_pred (numpy.ndarray): Predicted trajectory
        length (int): Length of trajectory to consider
        save_file (str): (Optional) Path where to save the figure as a .png image. Set to None for no saving.

    Returns:
        matplotlib.pyplot.Figure: The figure containing the plots
    """
    fig = plt.figure(figsize=(18, 6))
    for i, name in [(0, 'x'), (1, 'y'), (2, 'z')]:
        ax = plt.subplot(1, 3, i + 1)
        plt.plot(traj_real[:length, i], color="lightcoral", label='Real')
        plt.plot(traj_pred[:length, i], color="lightblue", label='Predicted')
        plt.title('Trajectories - {}'.format(name))
        plt.legend()
        plt.grid()

    if save_file:
        plt.savefig(save_file)
    return fig


def plot_traj_3d(traj_real, traj_pred, length=50_000, save_file=None):
    """
    Plots trajectories (real and predicted) in the 3d space

    Args:
        traj_real (numpy.ndarray): Real trajectory
        traj_pred (numpy.ndarray): Predicted trajectory
        length (int): Length of trajectory to consider
        save_file (str): (Optional) Path where to save the figure as a .png image. Set to None for no saving.

    Returns:
        matplotlib.pyplot.Figure: The figure containing the plots
    """
    fig = plt.figure(figsize=(16, 8))
    i = 1
    for trajectory, name in [(traj_real, "real"), (traj_pred, "predicted")]:
        ax = fig.add_subplot(1, 2, i, projection='3d')
        ax.plot(trajectory[:length, 0], trajectory[:length, 1], trajectory[:length, 2])
        ax.set_xlim(-11, 11)
        ax.set_ylim(-11, 11)
        ax.set_zlim(-1, 20)
        ax.set_title("{} trajectory".format(name.title()))
        i += 1

    if save_file:
        plt.savefig(save_file)
    return fig

def plot_autocorr(traj_real, traj_pred, length=50_000, save_file=None):
    """
    Plots autocorrelations along each coordinates of the (real and predicted) trajectories

    Args:
        traj_real (numpy.ndarray): Real trajectory
        traj_pred (numpy.ndarray): Predicted trajectory
        length (int): Length of trajectory to consider
        save_file (str): (Optional) Path where to save the figure as a .png image. Set to None for no saving.

    Returns:
        matplotlib.pyplot.Figure: The figure containing the plots
    """
    traj_real_x = statsmodels.tsa.stattools.acf(traj_real[:length, 0], nlags=length // 2)
    traj_pred_x = statsmodels.tsa.stattools.acf(traj_pred[:length, 0], nlags=length // 2)
    traj_real_y = statsmodels.tsa.stattools.acf(traj_real[:length, 1], nlags=length // 2)
    traj_pred_y = statsmodels.tsa.stattools.acf(traj_pred[:length, 1], nlags=length // 2)
    traj_real_z = statsmodels.tsa.stattools.acf(traj_real[:length, 2], nlags=length // 2)
    traj_pred_z = statsmodels.tsa.stattools.acf(traj_pred[:length, 2], nlags=length // 2)

    fig = plt.figure(figsize=(18, 6))
    for i, traj1, traj2, name in [(0, traj_real_x, traj_pred_x, 'x'),
                                  (1, traj_real_y, traj_pred_y, 'y'),
                                  (2, traj_real_z, traj_pred_z, 'z')]:
        ax = plt.subplot(1, 3, i + 1)
        plt.plot(traj1, color="lightcoral", label='Real')
        plt.plot(traj2, color="lightblue", label='Predicted')
        plt.title('Autocorrelations - {}'.format(name))
        plt.legend()
        plt.grid()

    if save_file:
        plt.savefig(save_file)

    return fig


def plot_fourier(traj_real, traj_pred, length=50_000, save_file=None, delta_t=DELTA_T):
    """
    Plots Fourier frequencies of the (real and predicted) trajectories

    Args:
        traj_real (numpy.ndarray): Real trajectory
        traj_pred (numpy.ndarray): Predicted trajectory
        length (int): Length of trajectory to consider
        save_file (str): (Optional) Path where to save the figure as a .png image. Set to None for no saving.
        delta_t (float): The timestep used to generate trajectories

    Returns:
        matplotlib.pyplot.Figure: The figure containing the plots
    """
    bar_width = 0.05
    fourier_real = np.fft.rfft(traj_real[:length,1])
    fourier_pred = np.fft.rfft(traj_pred[:length,1])
    n_freq = fourier_real.size
    f = np.linspace(0, 1 / delta_t, n_freq)

    fig = plt.figure(figsize=(12, 8))
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency [Hz]")

    plt.bar(f[:n_freq // 2] - bar_width,
            np.abs(fourier_real)[:n_freq // 2] * 1 / n_freq,
            width=bar_width / 2, color="lightcoral", label='Real', alpha=0.8)  # 1 / n_freq is a normalization factor
    plt.bar(f[:n_freq // 2] + bar_width,
            np.abs(fourier_pred)[:n_freq // 2] * 1 / n_freq,
            width=bar_width / 2, color="lightblue", label='Predicted', alpha=0.8) # 1 / n_freq is a normalization factor
    plt.legend()
    plt.grid()
    
    plt.xlim(0, 2.5)

    if save_file:
        plt.savefig(save_file)

    return fig


def main(args):
    path_to_model = args.model_path

    logging.info("Model successfully loaded")
    model = load_model(path_to_model)
    logging.info("Generating trajectories..."
                 "\n\tstart_pos = {}"
                 "\n\tn_points = {}"
                 "\n\tdelta_t = {}".format(start_pos, n_points, DELTA_T))
    traj_real, traj_pred = generate_trajectories(start_pos, n_points, model, rossler_map)
    logging.info("Done.")
    logging.info("Generating figures...")
    plot_pos_hist(traj_real, traj_pred, length=n_points, save_file="Position histograms.png")
    plot_autocorr(traj_real, traj_pred, length=n_points, save_file="Trajectory autocorrelations.png")
    plot_traj_axis(traj_real, traj_pred, length=n_points, save_file="Trajectories along coordinates.png")
    plot_traj_3d(traj_real, traj_pred, length=n_points, save_file="3d Trajectories.png")
    plot_fourier(traj_real, traj_pred, length=n_points, save_file="Fourier frequencies.png", delta_t=DELTA_T)
    logging.info("Done.")
    return 0


if __name__ == '__main__':
    start_pos = np.array(5*np.random.rand(3)).astype("float32")
    n_points = 100_000
    rossler_map = RosslerMap()
    parser = argparse.ArgumentParser(description="Generates analytical plots for the given pickled model.")
    parser.add_argument("model_path", help="Path to the pickled model", type=str,
                        default="./model_Jehanno_Portier_Brandeis")
    args = parser.parse_args()
    main(args)


