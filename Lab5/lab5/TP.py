import numpy as np
from numpy.linalg import qr, solve, norm
from scipy.linalg import expm
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from rossler_map import RosslerMap


def lyapunov_exponent(traj, jacobian, max_it=1000, delta_t=1e-3):

    n = traj.shape[1]
    w = np.eye(n)
    rs = []
    chk = 0

    for i in range(max_it):
        jacob = jacobian(traj[i,:])
        #WARNING this is true for the jacobian of the continuous system!
        w_next = np.dot(expm(jacob * delta_t),w)
        #if delta_t is small you can use:
        #w_next = np.dot(np.eye(n)+jacob * delta_t,w)

        w_next, r_next = qr(w_next)

        # qr computation from numpy allows negative values in the diagonal
        # Next three lines to have only positive values in the diagonal
        d = np.diag(np.sign(r_next.diagonal()))
        w_next = np.dot(w_next, d)
        r_next = np.dot(d, r_next.diagonal())

        rs.append(r_next)
        w = w_next
        # if i//(max_it/100)>chk:
        #     print(i//(max_it/100))
        #     chk +=1

    return  np.mean(np.log(rs), axis=0) / delta_t


def newton(f, jacob, x):
    #newton raphson method
    tol =1
    while tol>1e-5:
        #WARNING this is true for the jacobian of the continuous system!
        tol = x
        x = x-solve(jacob(x), f(v=x))
        tol = norm(tol-x)
    return x
