"""
Different sampling schemes for Multivariate Gaussian Distribution
"""
from abc import abstractmethod, ABCMeta
import numpy as np
import math
import itertools
from scipy.stats.distributions import invgauss

class Sample:

    def __init__(self, wc, wm, samples):

        self.Wc = wc
        self.Wm = wm
        self.samples = samples

        if samples.ndim == 2:
            self.dim = samples.shape[1]
        else:
            self.dim = 1

    def num_samples(self):

        return self.samples.shape[0]


class Moment:

    def __init__(self, mean, covariance):

        if (len(mean) != len(covariance)):
            raise ValueError('Inconsistent Dimensions')
        self._mean = mean
        self._covariance = covariance

        self.dim = len(mean)


class Hadamard:

      def __init__(self, cluster_matrix, overlapping=None):

        self.cluster_matrix = cluster_matrix
        self.num_clust = cluster_matrix.shape[0]
        self.dimension = cluster_matrix.shape[1]

        if overlapping is None:
          overlapping = cluster_matrix.dtype == float

        self.overlapping = overlapping

      def compute_trajectory_overlap(self, trajectory_list):

        trajectory_matrix = []
        for k in range(self.num_clust):
          temp = np.zeros(self.dimension)
          temp[self.cluster_matrix[k]] = trajectory_list[k]
          trajectory_matrix.append(temp)

        return np.sum(trajectory_matrix * self.cluster_matrix, axis=0)

      def compute_moment(self, momentlist):

        if self.overlapping:
          return self.compute_moment_overlap(momentlist)
        else:
          return self.compute_moment_nonoverlap(momentlist)

      def compute_moment_nonoverlap(self, momentlist):

        mean = np.zeros(self.dimension)
        covariance = np.zeros((self.dimension, self.dimension))

        for k in range(self.num_clust):
            clust = np.where(self.cluster_matrix[k])[0]
            ind_cl = np.array(list(itertools.product(clust, repeat=2)))
            ind_row = ind_cl[:, 0].reshape(len(clust), len(clust))
            ind_col = ind_cl[:, 1].reshape(len(clust), len(clust))
            covariance[ind_row, ind_col] = momentlist[k]._covariance
            mean[clust] = momentlist[k]._mean

        return Moment(mean, covariance)

      def compute_moment_overlap(self, momentlist):

        mean = np.zeros(self.dimension)
        covariance = np.zeros((self.dimension, self.dimension))

        for k in range(self.num_clust):

            clust = self.cluster_matrix[k]
            mean_cl = np.zeros(self.dimension)
            covariance_cl = np.zeros((self.dimension, self.dimension))
            covariance_cl[clust, clust] = momentlist[k]._covariance
            mean_cl[clust] = momentlist[k]._mean

            mean += clust * mean_cl
            covariance += (np.dot(clust, clust.T)) * covariance_cl

        return Moment(mean, covariance)

      def compute_sample_overlap(self, samplelist):

          sample_dimension = len(samplelist[0].Wc)
          dimension = self.cluster_matrix.shape[1]
          num_cluster = self.cluster_matrix.shape[0]

          sample = np.zeros((sample_dimension, dimension))
          for k in range(num_cluster):
              cluster = self.cluster_matrix[k].astype('bool')
              temp_sample = np.zeros((sample_dimension, dimension))
              temp_sample[:, cluster] = samplelist[k].samples
              sample += temp_sample * np.tile(self.cluster_matrix[k], (sample_dimension, 1))

          return sample



class SamplingMethod(object):

    __metaclass__ = ABCMeta

    def __init__(self,  mean, covariance):

        if isinstance(mean, np.ndarray) and isinstance(covariance, np.ndarray):
            if len(mean) != covariance.shape[0]:
                raise ValueError('Dimensions must be consistent')

        self._mean = mean
        self._covariance = covariance
        if isinstance(mean, np.ndarray):
            self.dim = len(self._mean)
        elif isinstance(mean, int) or isinstance(mean, float):
            self.dim = 1
        else:
            raise ValueError('Instance must be numpy array or float or int variable')

    @abstractmethod
    def generate_samples(self):

        pass

    @abstractmethod
    def cluster_sampling(self, cluster_matrix, equal_length=False):

        pass

    def compute_moments_from_cluster(self, sample, cluster_matrix=None, tol=1e-12, overlapping=None):

        if not isinstance(sample, list) and not isinstance(sample, Sample):
            raise ValueError('parameter sample must be of type Sample or list of Sample')

        if cluster_matrix is None:
            raise ValueError('Must provide cluster matrix for moment computation')

        num_cluster = cluster_matrix.shape[0]

        if isinstance(sample, Sample):
            moment_list = []
            for k in range(num_cluster):
                cluster = cluster_matrix[k].astype(bool)
                sample_cl = sample.samples[:, cluster]
                Wc = sample.Wc[k]
                Wm = sample.Wm[k]
                moment_cl = self.compute_moments(Sample(Wc, Wm, sample_cl))
                moment_list.append(moment_cl)
        else:
            moment_list = [self.compute_moments(sample_cl) for sample_cl in sample]
        hd = Hadamard(cluster_matrix, overlapping)

        return hd.compute_moment(moment_list)

    def compute_moments(self, sample, tol=1e-12):

        if not isinstance(sample, Sample):
            raise ValueError('parameter passed must be a Sample object')
        mean_est = np.average(sample.samples, weights=sample.Wm, axis=0)
        covariance_est = np.zeros((sample.dim, sample.dim))
        for k in range(sample.num_samples()):
            temp = (sample.samples[k] - mean_est).reshape((sample.dim, 1))
            covariance_est = covariance_est + sample.Wc[k] * np.dot(temp, temp.T)

        mean_est [np.abs(mean_est) < tol] = 0
        covariance_est [np.abs(covariance_est) < tol] = 0
        return Moment(mean_est, covariance_est)

    def compute_samplespace_cluster(self, cluster_matrix, overlapping=False):

        sample_cluster = self.cluster_sampling(cluster_matrix, equal_length=True, overlapping=overlapping)
        Wc = [sample.Wc for sample in sample_cluster]
        Wm = [sample.Wm for sample in sample_cluster]
        num_samples = len(sample_cluster[0].Wc)
        dimension = cluster_matrix.shape[1]

        sample = np.zeros((num_samples, dimension))
        for k in range(cluster_matrix.shape[0]):
            cluster = cluster_matrix[k, :]
            cl = cluster.astype(bool)
            if not overlapping:
                sample[:, cl] = sample_cluster[k].samples
            elif overlapping:
                temp_sample = np.zeros((num_samples, dimension))
                temp_sample[:, cl] = sample_cluster[k].samples
                sample += temp_sample * np.tile(cluster, (num_samples, 1))

        return Sample(Wc, Wm, sample), sample_cluster

    def compare_moment_list(self, moment_1, moment_2):

        return [self.compare_moment(x, y) for (x, y) in zip(moment_1, moment_2)]

    def compare_moment(self, moment_1, moment_2):

        if not isinstance(moment_1, Moment) and not isinstance(moment_2, Moment):
            raise ValueError('Parameters passed are not type Moment')
        if moment_1.dim != moment_2.dim:
            raise ValueError('Moments must have same dimension')

        np.seterr(divide='ignore', invalid='ignore')

        metric_mean = np.linalg.norm(moment_1._mean - moment_2._mean) / np.linalg.norm(moment_1._mean)
        #metric_mean[~ np.isfinite(metric_mean)] = 0
        norm_mean = np.linalg.norm(metric_mean)/moment_1.dim

        metric_covariance = np.linalg.norm(moment_1._covariance - moment_2._covariance)/np.linalg.norm(moment_1._covariance)
        #metric_covariance[~ np.isfinite(metric_covariance)] = 0
        norm_covariance = metric_covariance/moment_2.dim

        return norm_mean, norm_covariance


class UT(SamplingMethod):

    def __init__(self, mean, covariance, config=None, sampling_dimension=None):

        SamplingMethod.__init__(self, mean, covariance)
        if config is None:
            self.ki = 50
            self.beta = 2
            self.alpha = 0.25
        else:
            if config['ki']:
                self.ki = config['ki']
            else:
                self.ki = 50
            if config['beta']:
                self.beta = config['beta']
            else:
                self.beta = 2
            if config['alpha']:
                self.alpha = config['alpha']
            else:
                self.alpha = 0.25
        if sampling_dimension is None:
            self.num_samples = self.generate_num_samples()
            sampling_dimension = self.dim
        else:
            if sampling_dimension < self.dim:
                raise ValueError('Required number of sample points cannot be less than the minimum requirement')
            self.num_samples = 2 * sampling_dimension + 1

        self.dim_lam = sampling_dimension
        self._lambda = (self.alpha ** 2) * (self.dim + self.ki) - self.dim

    def generate_num_samples(self):

        return 2 * self.dim + 1

    def generate_samples(self):

        c = self._lambda + self.dim
        num_samples_mean = self.num_samples - self.generate_num_samples() + 1

        Wm = np.r_[(self._lambda/(c * num_samples_mean)) * np.ones(num_samples_mean), np.ones(2 * self.dim)*(0.5/c)]
        Wc = np.r_[((self._lambda/c + 1 - self.alpha**2 + self.beta) / num_samples_mean) * np.ones(num_samples_mean),
                   np.ones(2 * self.dim)*(0.5/c)]

        U = np.sqrt(c)*np.linalg.cholesky(self._covariance).T
        samples = np.vstack(
            (np.tile(self._mean, (num_samples_mean, 1)), np.tile(self._mean, (self.dim, 1)) + U, np.tile(self._mean, (self.dim, 1)) - U))

        return Sample(Wc, Wm, samples)

    def cluster_sampling(self, cluster_matrix, overlapping=False, equal_length=False):

        number_cluster = cluster_matrix.shape[0]
        sample_cluster = []

        max_dim = None
        if not overlapping and equal_length:
            max_dim = max(np.sum(cluster_matrix, axis=1)).astype(np.int)

        if overlapping and equal_length:
            max_dim = max(np.sum(cluster_matrix.astype('bool'), axis=1))

        for k in range(number_cluster):
            cluster = cluster_matrix[k, :]
            cl = cluster.astype(bool)
            cov_cl = self._covariance[cl][:, cl]
            mean_cl = self._mean[cl]
            method = UT(mean_cl, cov_cl, sampling_dimension=max_dim)
            sample_cluster.append(method.generate_samples())

        return sample_cluster

    @staticmethod
    def from_sample(sample, tol=1e-12):

        if not isinstance(sample, Sample):
            raise ValueError('sampled passed must be of type Sample')

        mean_est = np.average(sample.samples, weights=sample.Wm, axis=0)
        covariance_est = np.zeros((sample.dim, sample.dim))
        for k in range(sample.num_samples()):
            temp = (sample.samples[k] - mean_est).reshape((sample.dim, 1))
            covariance_est = covariance_est + sample.Wc[k] * np.dot(temp, temp.T)

        mean_est[np.abs(mean_est) < tol] = 0
        covariance_est[np.abs(covariance_est) < tol] = 0
        return UT(mean_est, covariance_est)


class MonteCarlo(SamplingMethod):

    def __init__(self, mean, covariance, config=None):

        SamplingMethod.__init__(self, mean, covariance)
        if config is None:
            self.num_samples = 1e6
        else:
            self.num_samples = config['num_samples']
        self.num_samples = int(self.num_samples)

    def generate_samples(self):

        Wm = np.ones(self.num_samples)*1./self.num_samples
        samples = np.random.multivariate_normal(self._mean, self._covariance, self.num_samples)

        return Sample(Wm, Wm, samples)

    def cluster_sampling(self, cluster_matrix, equal_length=False):

        number_cluster = cluster_matrix.shape[0]
        sample_cluster = []

        for k in range(number_cluster):
            cluster = cluster_matrix[k, :]
            cl = cluster.astype(bool)
            cov_cl = self._covariance[cl][:, cl]
            mean_cl = self._mean[cl]
            method = MonteCarlo(mean_cl, cov_cl)
            sample_cluster.append(method.generate_samples())

        return sample_cluster


class GaussQuadrature(SamplingMethod):

    def __init__(self, mean, covariance, config=None):

        SamplingMethod.__init__(self, mean, covariance)
        if config is None:
            self.deg = 3
        else:
            self.deg = config['deg']

    def generate_samples(self):

        sample_1d, wc_1d = np.polynomial.hermite.hermgauss(self.deg)
        sample_1d = sample_1d * math.sqrt(2)
        wc_1d = wc_1d / sum(wc_1d)

        sample = np.array(list(itertools.product(sample_1d, repeat=self.dim)))
        wc = np.array(list(itertools.product(wc_1d, repeat=self.dim)))
        wc = np.prod(wc, axis=1)

        sample = np.dot(sample, np.linalg.cholesky(self._covariance)) + self._mean

        return Sample(wc, wc, sample)

    def cluster_sampling(self, cluster_matrix, equal_length=False):

        number_cluster = cluster_matrix.shape[0]
        sample_cluster = []

        for k in range(number_cluster):
            cluster = cluster_matrix[k, :]
            cl = cluster.astype(bool)
            cov_cl = self._covariance[cl][:, cl]
            mean_cl = self._mean[cl]
            method = GaussQuadrature(mean_cl, cov_cl)
            method.deg = self.deg
            sample_cluster.append(method.generate_samples())

        return sample_cluster


class LatinHypercube(SamplingMethod):

    def __init__(self, mean, covariance, config=None):

        SamplingMethod.__init__(self, mean, covariance)
        if config is None:
            self.num_samples = 1e6
        else:
            self.num_samples = config['num_samples']
        self.num_samples = int(self.num_samples)

    def generate_samples(self):

        Wm = np.ones(self.num_samples) * 1. / self.num_samples
        z = np.random.multivariate_normal(self._mean, self._covariance, self.num_samples)

        p = self.dim
        samples = np.zeros(z.shape, dtype=z.dtype)

        for i in range(p):
            x = z[:, i]
            r = np.zeros(self.num_samples)
            rowidx = np.argsort(x)
            r[rowidx] = np.arange(self.num_samples)
            samples[:, i] = r

        samples = samples + np.random.rand(samples.shape)
        samples = samples / self.num_samples

        return Sample(Wm, Wm, samples)


class SamplingMethodGenerator:

    @staticmethod
    def sample(mean, covariance, sampling_type=None, config=None):

        if sampling_type is None:
            sampling_type = 'ut'

        sampling_type = sampling_type.lower()
        if sampling_type == 'ut':
            return UT(mean, covariance, config=config)
        elif sampling_type == 'gaussquadrature':
            return GaussQuadrature(mean, covariance, config=config)
        elif sampling_type == 'montecarlo':
            return MonteCarlo(mean, covariance, config=config)
