import numpy as np
from scipy.integrate import odeint


class CoupledOscillator(object):

    def __init__(self, config, epsilon, cluster_vector, num_states_per_osc):

        dimension = config['dimension']

        if config is None:
            raise ValueError('Must provide the required parameters')

        if cluster_vector is None:
            cluster_vector = np.ones(dimension, dtype=bool)
        else:
            dimension = sum(cluster_vector)

        self.dimension = dimension
        self.epsilon = epsilon
        self.state_vector = np.r_[np.zeros(num_states_per_osc, dtype=bool), cluster_vector, False]
        self.num_states_per_osc = num_states_per_osc

        oscillator_matrix = cluster_vector.reshape(len(cluster_vector)/num_states_per_osc, num_states_per_osc)
        self.oscillator_matrix = np.any(oscillator_matrix, axis=1)

        self.index_states = np.where(self.state_vector)[0]

    def compute_parameters(self, parameter_list):

        self.parameter = np.zeros((len(self.oscillator_matrix), parameter_list.shape[1]))
        self.parameter[self.oscillator_matrix] = parameter_list

    def _velocity_function(self, x0):

        pass

    def _cluster_oscillators(self, cluster_matrix):

        pass

    def _compute_jacobian(self, x0):

        pass

    def _adjaceny_matrix(self, x0):

        pass

    def simulate(self, x0, t, dfun=None):

        def func(x0, t): return self._velocity_function(x0)
        y0 = odeint(func, x0, t, Dfun=dfun)
        return y0


class Lorenz3D(CoupledOscillator):

    def __init__(self, config, epsilon=0, cluster_vector=None):

        CoupledOscillator.__init__(self, config, epsilon, cluster_vector, 3)

        parameters_list = np.c_[config['sigma'][self.oscillator_matrix], config['rho'][self.oscillator_matrix],
                                     config['beta'][self.oscillator_matrix]]

        CoupledOscillator.compute_parameters(self, parameters_list)

    def _velocity_function(self, x0):

        X0 = np.zeros(self.state_vector.shape)
        X0[self.state_vector] = x0

        dy_dx = np.zeros(self.dimension)
        for k in range(3, self.dimension + 3):
            i = self.index_states[k - 3]
            osc_ind = int(i / 3)
            if i % 3 == 0:
                dy_dx[k - 3] = self.parameter[osc_ind - 1, 0] * (X0[i+1] - X0[i]) + self.epsilon * (X0[i+3] - 2*X0[i] + X0[i-3])
            elif i % 3 == 1:
                dy_dx[k - 3] = X0[i - 1] * (self.parameter[osc_ind - 1, 1] - X0[i + 1]) - X0[i]
            elif i % 3 == 2:
                dy_dx[k - 3] = X0[i - 2] * X0[i - 1] - self.parameter[osc_ind - 1, 2] * X0[i]

        return dy_dx

    def _cluster_oscillators(self, cluster_matrix):

        num_cluster = cluster_matrix.shape[0]

        config = dict()
        config['sigma'] = self.parameter[:, 0]
        config['rho'] = self.parameter[:, 1]
        config['beta'] = self.parameter[:, 2]
        config['dimension'] = self.dimension

        oscillator_clusters = []
        for k in range(num_cluster):
            cluster_vector = cluster_matrix[k]
            oscillator_clusters.append(Lorenz3D(config=config, epsilon=self.epsilon, cluster_vector=cluster_vector))

        return oscillator_clusters

    def _compute_jacobian(self, x0):

        num_oscillator = self.dimension/3
        xi = np.zeros(num_oscillator + 2)
        xi[1:-1] = x0[::3]
        yi = np.zeros(num_oscillator + 1)
        yi[1:] = x0[1::3]
        zi = np.zeros(num_oscillator + 1)
        zi[1:] = x0[2::3]

        dy = np.zeros((self.dimension, self.dimension))
        for k in range(num_oscillator):
            dy[3 * k, 3 * k] = -2 * self.epsilon - self.parameter[k, 0]
            if k <= num_oscillator - 2:
                dy[3 * k, 3 * (k + 1)] = self.epsilon
            if k != 0:
                dy[3 * k, 3 * (k - 1) + 1] = self.epsilon
            dy[3 * k, 3 * k + 1] = self.parameter[k, 0]
            dy[3 * k + 1, 3 * k] = self.parameter[k, 1] - zi[k + 1]
            dy[3 * k + 1, 3 * k + 1] = -1
            dy[3 * k + 1, 3 * k + 2] = -xi[k + 1]
            dy[3 * k + 2, 3 * k] = yi[k + 1]
            dy[3 * k + 2, 3 * k + 1] = xi[k + 1]
            dy[3 * k + 2, 3 * k + 2] = -self.parameter[k, 2]

        return dy


class Rossler3D(CoupledOscillator):

    def __init__(self, config, epsilon=0, cluster_vector=None):

        CoupledOscillator.__init__(self, config, epsilon, cluster_vector, 3)

        parameters_list = np.c_[config['a0'][self.oscillator_matrix],
                                config['b0'][self.oscillator_matrix], config['c0'][self.oscillator_matrix]]

        CoupledOscillator.compute_parameters(self, parameters_list)

    def _velocity_function(self, x0):

        X0 = np.zeros(self.state_vector.shape)
        X0[self.state_vector] = x0

        dy_dx = np.zeros(self.dimension)
        for k in range(3, self.dimension + 3):
            i = self.index_states[k - 3]
            osc_ind = int(i / 3)
            if i % 3 == 0:
                dy_dx[k - 3] = -X0[i+1] - X0[i+2] + self.epsilon * (X0[i + 3] - 2 * X0[i] + X0[i - 3])
            elif i % 3 == 1:
                dy_dx[k - 3] = X0[i - 1] + self.parameter[osc_ind - 1, 0] * X0[i]
            elif i % 3 == 2:
                dy_dx[k - 3] = self.parameter[osc_ind - 1, 1] + X0[i] * (X0[i - 2] - self.parameter[osc_ind - 1, 2])

        return dy_dx

    def _compute_jacobian(self, x0):

        num_oscillator = self.dimension/3
        xi = np.zeros(num_oscillator + 2)
        xi[1:-1] = x0[::3]
        yi = np.zeros(num_oscillator + 1)
        yi[1:] = x0[1::3]
        zi = np.zeros(num_oscillator + 1)
        zi[1:] = x0[2::3]

        dy = np.zeros((self.dimension, self.dimension))
        for k in range(num_oscillator):
            dy[3 * k, 3 * k] = -2 * self.epsilon
            if k <= num_oscillator - 2:
                dy[3 * k, 3 * (k + 1)] = self.epsilon
            if k != 0:
                dy[3 * k, 3 * (k - 1) + 1] = self.epsilon
            dy[3 * k, 3 * k + 1] = -1
            dy[3 * k, 3 * k + 2] = -1
            dy[3 * k + 1, 3 * k] = 1
            dy[3 * k + 1, 3 * k + 1] = self.parameter[k, 0]
            dy[3 * k + 2, 3 * k] = zi[k + 1]
            dy[3 * k + 2, 3 * k + 2] = -self.parameter[k, 2]

        return dy

    def _cluster_oscillators(self, cluster_matrix):

        num_cluster = cluster_matrix.shape[0]

        config = dict()
        config['a0'] = self.parameter[:, 0]
        config['b0'] = self.parameter[:, 1]
        config['c0'] = self.parameter[:, 2]
        config['dimension'] = self.dimension

        oscillator_clusters = []
        for k in range(num_cluster):
            cluster_vector = cluster_matrix[k]
            oscillator_clusters.append(Rossler3D(config=config, epsilon=self.epsilon, cluster_vector=cluster_vector))

        return oscillator_clusters

class Vanderpol2D(CoupledOscillator):

    def __init__(self, config, epsilon=0, cluster_vector=None):

        CoupledOscillator.__init__(self, config, epsilon, cluster_vector, 2)
        parameters_list = config['mu']
        CoupledOscillator.compute_parameters(self, parameters_list)

    def _velocity_function(self, x0):

        X0 = np.zeros(self.state_vector.shape)
        X0[self.state_vector] = x0

        dy_dx = np.zeros(self.dimension)
        for k in range(2, self.dimension + 2):
            i = self.index_states[k - 2]
            osc_ind = int(i / 2)
            if i % 2 == 0:
                dy_dx[k - 2] = self.parameter[osc_ind - 1, 0] * (X0[i] - (X0[i] ** 3) / 3. - X0[i + 1]) + self.epsilon * (X0[i + 2] - 2 * X0[i] + X0[i - 2])
            elif i % 2 == 1:
                dy_dx[k - 2] = 1./self.parameter[osc_ind - 1, 0] * X0[i - 1]

        return dy_dx

    def _compute_jacobian(self, x0):

        num_oscillator = self.dimension / 3
        xi = np.zeros(num_oscillator + 2)
        xi[1:-1] = x0[::3]
        yi = np.zeros(num_oscillator + 1)
        yi[1:] = x0[1::3]

        dy = np.zeros((self.dimension, self.dimension))
        for k in range(num_oscillator):
            dy[2 * k, 2 * k] = self.parameter[k, 0] * (1 - xi[k + 1] ^ 2) - 2 * self.epsilon
            if k <= num_oscillator - 2:
                dy[2 * k, 2 * (k + 1)] = self.epsilon
            if k != 0:
                dy[2 * k, 2 * (k - 1) + 1] = self.epsilon
            dy[2 * k, 2 * k + 1] = -self.parameter[k, 0]
            dy[2 * k + 1, 2 * k] = 1./self.parameter[k, 0]


class CoupledOscillatorExample:

    @staticmethod
    def model(model_name, config, epsilon=0, cluster_vector=None):
        if model_name == 'lorenz':
            return Lorenz3D(config=config, epsilon=epsilon, cluster_vector=cluster_vector)
        elif model_name == 'vanderpol':
            return Vanderpol2D(config=config, epsilon=epsilon, cluster_vector=cluster_vector)
        elif model_name == 'rossler':
            return Rossler3D(config=config, epsilon=epsilon, cluster_vector=cluster_vector)
        else:
            raise ValueError('Unknown model type or has not been developed yet')
