import numpy as np
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh

class KLExpansion:

    def __init__(self, cov_function=None, precomputed=True):

        """
        Initialize KL Expansion with covariance function
        if precomputed is set to True, then must provide cov_function
        :param cov_function:
        :param precomputed:
        """
        if precomputed and cov_function is None:
            raise ValueError('if precomputed is set to True, then must provide cov_function')

        if cov_function is not None:
            self.cov_function = cov_function

    def compute_k_matrix(self, domain, tol=1e-8):

        x = domain[0]
        y = domain[1]

        k = self.cov_function(x, y)

        k[k < tol] = 0

        return k

    def compute_eigen_functions(self, domain, num_eigens):

        X = self.compute_k_matrix(domain)
        return largest_eigsh(X, num_eigens, which='LM')

    def cluster_region(self, map_data, cluster_matrix):

        region = []
        eigs = []
        map_data = map_data.reshape(map_data.size)
        for cluster in cluster_matrix:
            clust = cluster.astype(bool)
            sub_map = map_data[clust]
            domain = np.meshgrid(sub_map, sub_map)
            region.append(self.compute_k_matrix(domain))
            eigs.append(self.compute_eigen_functions(domain, 5))

        return region, eigs
