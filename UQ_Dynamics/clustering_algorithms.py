import sklearn.cluster as skc
from sklearn.preprocessing import normalize
import community
import networkx as nx
from igraph import Graph
import logging
import numpy as np


class ClusteringMethod:

    def __init__(self,affinity_data, clustering_type, params=None):

        """
        Preared by Arpan Mukherjee
        Collection of available clustering methods on adjacency matrix
        Can return overlapping or non-overlapping clusters
        :param affinity_data: 
        :param clusatering_type: 
        :param params: 
        """
        if not isinstance(affinity_data, np.ndarray):
            raise ValueError('Affinity Information must be a 2D numpy.ndarray')
        self.dimension = affinity_data.shape[0]
        if not (affinity_data.transpose() == affinity_data).all():
            logging.info('Adjacency Matrix not symmetric, converted to Symmetric')
            affinity_data = (affinity_data + affinity_data.T)/2.

        if params is None:
            params = {}

        if 'Abs_Tol' not in params:
            params['Abs_Tol'] = 1e-12

        if 'n_cluster' not in params:
            params['n_cluster'] = int(affinity_data.shape[0] / 5)

        affinity_data[affinity_data < params['Abs_Tol']] = 0

        self._adjacency = normalize(affinity_data, axis=1, norm='l1')
        clustering_type = clustering_type.lower()
        if clustering_type == 'spectral':
            self._Graph = skc.SpectralClustering(n_clusters=params['n_cluster'],
                                            affinity='precomputed')
        elif clustering_type == 'louvain':
            self._Graph = nx.from_numpy_matrix(self._adjacency)
        else:
            self._Graph = Graph.Incidence(affinity_data.tolist())
        self.method = clustering_type

    def compute_cluster(self):

        comm = None
        if self.method == 'spectral':
            self._Graph.fit(self._adjacency, self._Graph.n_clusters)
            comm = self._Graph.labels_
        elif self.method == 'louvain':
            partition = community.best_partition(self._Graph)
            comm = np.array([val for (key, val) in partition.iteritems()],'int')
        elif self.method == 'fast_greedy':
            comm = self._Graph.community_fastgreedy()
            comm = comm.as_clustering()

        return comm

    def compute_cluster_matrix_skc(self, overlapping=False, overlap_tol=None, ensemble=False):

        comm = self.compute_cluster()

        clust_matrix = [comm == k for k in np.unique(comm)]

        if not overlapping:
          return np.array(clust_matrix).astype(np.uint8)

        D_cl = np.sum(self._adjacency, axis=1)
        clust_matrix_overlap = np.zeros((len(clust_matrix), self.dimension))

        for i in np.unique(comm):
          for j in range(self.dimension):
            clust_matrix_overlap[i, j] = np.sum(self._adjacency[j, comm == i])/D_cl[j]

        if overlap_tol is None:
          overlap_tol = 1e-3

        clust_matrix_overlap[clust_matrix_overlap < overlap_tol] = 0
        return normalize(clust_matrix_overlap, axis=0, norm='l1').astype(np.float16)