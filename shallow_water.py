import numpy as np
from scipy import signal

class ShallowWater:

  def __init__(self, config):
    """
    Prepared by Arpan Mukherjee
    :param config: containing configurations pertaining to the 1D shallow water model
    
    """
    self.g = 9.81
    self.L = config['L']
    self.N = config['N']
    self.MaxTime = config['MaxTime']
    self.Timestep = config['Timestep']
    self.cf = config['cf']
    self.D = config['D']

    self._compute_system_parameters()
    self._compute_system_matrices()

  def _compute_system_parameters(self):

    self.dim = 2 * (self.N + 1)
    self.dx = self.L/(self.N + 0.5)
    self.dt = self.MaxTime/self.Timestep

  def _compute_system_matrices(self):
    """
    Prepared by Arpan Mukherjee
    :return: 
    """
    row1 = np.c_[-self.g/(2*self.dx), 1./self.dt + self.cf/2., self.g/(2*self.dx)]
    row3 = np.c_[self.g/(2*self.dx), 1./self.dt + self.cf/2., -self.g/(2*self.dx)]

    D_bar = np.eye(self.dim)
    A_bar = np.zeros((self.dim, self.dim))

    for ind_row in range(self.N):
      row2 = np.c_[-self.D / (2. * self.dx), 1. / self.dt, self.D / (2 * self.dx)]
      row4 = np.c_[self.D / (2. * self.dx), 1. / self.dt, -self.D / (2 * self.dx)]
      r_1 = ind_row * 2
      r_2 = ind_row * 2 + 1
      c_1 = ind_row * 2
      c_2 = ind_row * 2 + 1
      D_bar[r_1 + 1, c_1:c_1 + 3] = row1
      D_bar[r_2 + 1, c_2:c_2 + 3] = row2.copy()

      A_bar[r_1 + 1, c_1:c_1 + 3] = row3
      A_bar[r_2 + 1, c_2:c_2 + 3] = row4.copy()

    B = np.zeros((self.dim, 1))
    B[0, 0] = 1

    A = np.dot(np.linalg.pinv(D_bar), A_bar)

    '''
    C = np.zeros(self.dim)
    C[::2] = 1
    C = np.diagflat(C)
    C = C[::2, :]
    '''
    C = np.eye(self.dim)

    num_outputs = C.shape[0]
    D = np.zeros((num_outputs, 1))

    self.A = A
    self.B = B
    self.C = C
    self.D = D

  def _generate_system(self):

    """
    Prepared by Arpan Mukherjee
    :return: state-space representation of the 1-D Shallow Water Model
    """
    return signal.StateSpace(self.A, self.B, self.C, self.D, dt=self.dt)

  def _generate_system_clusters(self, clust_matrix):

    """
    Prepared by Arpan Mukherjee
    :param clust_matrix: 
    :return: list of shallow water equation objects for specific length of cluster
    """
    A = self.A
    B = self.B
    C = self.C
    D = self.D

    number_cluster = clust_matrix.shape[0]

    _state_space_clust = []

    for k in range(number_cluster):
      cluster = clust_matrix[k, :]
      cl = cluster.astype(bool)
      #length_cluster = np.count_nonzero(cluster)
      #cl = cl.reshape(length_cluster)
      A_cl = A[cl][:, cl]
      B_cl = B[cl]

      C_cl = C[cl][:, cl]
      D_cl = D[cl]

      _state_space_clust.append(signal.StateSpace(A_cl, B_cl, C_cl, D_cl, dt=self.dt))

    return _state_space_clust

  def _adjacency_matrix(self, tol=None):

    if tol is None:
      tol = 1e-12

    adj = np.absolute(self.A)
    adj[adj < tol] = 0

    return adj







