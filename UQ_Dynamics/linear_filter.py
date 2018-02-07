import numpy as np
from scipy import signal
from sampling import Moment


class LinearFilter:

    def __init__(self,ss):

        if not isinstance(ss, signal.StateSpace):
            raise ValueError('State Space model not valid')

        self._ss = ss

    def kf_update(self, x, P, H, R, z):

        Kk = np.dot(np.dot(P, H.T), np.linalg.inv(np.dot(np.dot(H, P), H.T) +R))
        x_update = x + np.dot(Kk, (z - np.dot(H, x)))
        P_update = np.dot((np.eye(x.size) - np.dot(Kk, H)), P)
        return Moment(x_update, P_update)

    def system_cluster(self, cluster_matrix):

        A = self._ss.A
        B = self._ss.B
        C = self._ss.C
        D = self._ss.D

        dt = self._ss.dt

        number_cluster = cluster_matrix.shape[0]

        _state_space_clust = []

        for k in range(number_cluster):
            cluster = cluster_matrix[k, :]
            cl = cluster.astype(bool)
            A_cl = A[cl][:, cl]
            B_cl = B[cl]

            C_cl = C[cl][:, cl]
            D_cl = D[cl]

            _state_space_clust.append(signal.StateSpace(A_cl, B_cl, C_cl, D_cl, dt=dt))

        return _state_space_clust

    def kf_predict(self, x_k, u, dt):

          x0 = x_k._mean
          P0 = x_k._covariance

          _, _, xk = signal.lsim(self._ss, u, np.array([0, dt]), X0=x0)







