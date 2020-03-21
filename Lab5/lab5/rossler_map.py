import numpy as np
from scipy.integrate import solve_ivp


class RosslerMap:
    """
    Rossler attractor
    With a=0.2, b=0.2, and c=5.7
    """

    def __init__(self, a=0.2, b=0.2, c=5.7, delta_t=1e-3):
        self.a, self.b, self.c = a, b, c
        self.delta_t = delta_t

    def v_eq(self, t=None, v=None):
        x, y, z = v[0], v[1], v[2]
        dot_x = -y - z
        dot_y = x + self.a * y
        dot_z = self.b + z * (x - self.c)
        return np.array([dot_x, dot_y, dot_z])

    def jacobian(self, v):
        x, z = v[0], v[2]
        res = np.array([[0,     -1,      -1],
                       [1, self.a,       0],
                       [z,      0, x-self.c]])
        return res

    def full_traj(self, nb_steps, init_pos):
        t = np.linspace(0, nb_steps * self.delta_t, nb_steps)
        f = solve_ivp(self.v_eq, [0, nb_steps * self.delta_t], init_pos, method='RK45', t_eval=t)
        return np.moveaxis(f.y, -1, 0), t

    def equilibrium(self):
        x0 = (self.c - np.sqrt(self.c ** 2 - 4 * self.a * self.b)) / 2
        y0 = (-self.c + np.sqrt(self.c ** 2 - 4 * self.a * self.b)) / (2 * self.a)
        z0 = (self.c - np.sqrt(self.c ** 2 - 4 * self.a * self.b)) / (2 * self.a)
        return np.array([x0, y0, z0])

