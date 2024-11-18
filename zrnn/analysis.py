
import torch
import torch.nn as nn
import numpy as np
import warnings
from numdifftools.core import Jacobian
from torchdiffeq import odeint


DEF_PARAMS = {
            'W': np.array([[8.949, -7.4, -2.952],
                           [9.123, -7.386, -3.072],
                           [8.935, -4, -1]]),
            'W_in': np.array([0.048, 0.055, 0.068]),
            'I_b': np.array([0, 0, -0.1]),
            'tau': np.array([0.01, 0.01, 0.05]),
            'a': np.array([150.134, 62.873, 87.0]),
            'b': np.array([-0.476, 0.481, -0.781]),
            'c': np.array([0.007, 0.016, 0.012]),
}


class DynamicalSystem(nn.Module):
    def __init__(self, params):
        super(DynamicalSystem, self).__init__()
        self.params = params
        self.jacobian = Jacobian(self._eqs)

    def forward(self, t, state):
        return torch.tensor(self._eqs(state.numpy()), dtype=torch.float32)

    def _eqs(self, state):
        # Equations of motion
        i_cc = self.params['I_cc']
        weights = self.params['W']
        fx = self._soft_rect(state)
        w_in = self.params['W_in']
        i_b = self.params['I_b']
        scale = np.diag(1 / self.params['tau'])
        return scale @ (-state + weights @ fx + i_cc * w_in + i_b).T

    def _fixed_point(self, sphere: float = 1.):
        # Look for fixed point with randomly initiated point
        from scipy.optimize import fsolve
        warnings.filterwarnings("error", category=RuntimeWarning)
        init = sphere * np.random.rand(3) - sphere / 2
        try:
            return fsolve(self._eqs, init)
        except RuntimeWarning:
            return np.zeros_like(init)

    def get_fixed_points(self, n_iters: int = 1000, sphere: float = 1.):
        # Stochastically look for the fixed points by random initiations, then cluster similar points.
        from scipy.cluster.hierarchy import fclusterdata
        points = np.array([self._fixed_point(sphere) for _ in range(n_iters)])
        points = points[~np.all(points == 0, axis=1)]
        clusters = fclusterdata(points[:, -1].reshape(-1, 1), 1e-5, criterion='distance')
        fixed_points = [points[clusters==c].mean(axis=0) for c in np.unique(clusters)]
        stability, spirality = [], []
        for fp in fixed_points:
            t, p = self.determine_stability(fp)
            stability.append(t), spirality.append(p)
        return np.array(fixed_points), stability, spirality

    def determine_stability(self, point):
        # Determine the stability + spirality of a fixed point
        jac = self.jacobian(point)
        eigs = np.linalg.eigvals(jac)
        stable = np.all(np.real(eigs) < 0)
        oscillatory = np.any(np.imag(eigs) != 0)
        return stable, oscillatory

    def _soft_rect(self, state):
        # Non-linear activation
        return self.params['c'] * self._large_exp_handler((self.params['a'] * state - self.params['b']))

    @staticmethod
    def _large_exp_handler(x, threshold: float = 50.):
        # Helps avoiding explosions (log(1 + exp(x)) ~ x for x >> 1
        return np.where(x > threshold, x, np.log1p(np.exp(x)))


def solver(params, initial_conditions, t_final, method='rk4'):
    model = DynamicalSystem(params)
    initial_conditions = torch.tensor(initial_conditions, dtype=torch.float32)
    time = torch.linspace(0, t_final, int(t_final * 1e3 + 1))
    return odeint(model, initial_conditions, time, method=method).numpy()


def get_oscillation_period(solution):
    from scipy.signal import find_peaks
    return np.diff(find_peaks(solution[500:, 2], height=0)[0]).mean()
