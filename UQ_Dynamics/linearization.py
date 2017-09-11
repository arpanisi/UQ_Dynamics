from sampling import UT
import numpy as np


class Linearization(object):

    def __init__(self, func_def, mean, method):

        self.func_def = func_def
        self.mean = mean
        self.method = method

    def linear_matrix(self):

        pass

    def generate_samples(self):

        pass

    def _adjacency_matrix(self, tol=1e-12):

        linear_matrix = self.linear_matrix()
        linear_matrix = abs(linear_matrix)

        linear_matrix[linear_matrix < tol] = 0
        if not (linear_matrix.transpose() == linear_matrix).all():
            linear_matrix = (linear_matrix + linear_matrix.T)/2.

        return linear_matrix

class JacobianLinearization(Linearization):

    def __init__(self, func_def, mean, epsilon=0.01, method='central', Abs_Tol = 1e-12):

        Linearization.__init__(self, func_def, mean, method)
        if self.method is None:
            self.method = 'central'
        self.mean = mean
        self.dimension = len(mean)
        ind = np.nonzero(self.mean == 0)
        self.mean[ind] = self.mean[ind] + Abs_Tol

    def linear_matrix(self):

        if self.method == 'forward':
            linear_matrix = self.forward_difference()
        elif self.method == 'backward':
            linear_matrix = self.backward_difference()
        else:
            linear_matrix = self.central_difference()

        return linear_matrix

    def forward_difference(self):

        if 'num_sample' not in self.__dict__:
            self.generate_num_samples()

        linear_matrix = np.zeros((self.dimension, self.dimension))
        for k in range(self.num_sample):

            x = self._samples[k]
            if k <= self.dimension:
                f = self.func_def(x)
            else:
                f = self.func_def(self.mean)
            linear_matrix += 2. * self._Wc[k] * np.dot(f.reshape(self.dimension, 1),
                                                  (x - self.mean).reshape(1, self.dimension))
        return linear_matrix

    def backward_difference(self):

        if 'num_sample' not in self.__dict__:
            self.generate_num_samples()

        linear_matrix = np.zeros((self.dimension, self.dimension))
        for k in range(self.num_sample):

            x = self._samples[k]
            if k > self.dimension:
                f = self.func_def(x)
            else:
                f = self.func_def(self.mean)
            linear_matrix += 2. * self._Wc[k] * np.dot(f.reshape(self.dimension, 1),
                                                  (x - self.mean).reshape(1, self.dimension))
        return linear_matrix

    def central_difference(self):

        if '_samples' not in self.__dict__:
            self.generate_samples()

        linear_matrix = np.zeros((self.dimension, self.dimension))
        for k in range(self.num_sample):

            x = self._samples[k]
            f = self.func_def(x)
            linear_matrix += self._Wc[k]*np.dot(f.reshape(self.dimension,1),
                                                             (x-self.mean).reshape(1,self.dimension))

        return linear_matrix

    def generate_samples(self):

        if 'num_sample' not in self.__dict__:
            self.generate_num_samples()

        h = np.diag(self.epsilon * self.mean)
        self._samples = np.r_[self.mean.reshape(1,self.dimension), np.tile(self.mean, (self.dimension, 1)) + h,
                                 np.tile(self.mean, (self.dimension, 1)) - h]
        self._Wc = np.r_[0, 0.5 / (np.r_[np.diag(h), np.diag(h)] ** 2)]

    def generate_num_samples(self):

        self.num_sample = 2 * self.dimension + 1


class StatisticalLinearization(Linearization):

    def __init__(self, func_def, mean, covariance, method='UT'):

        Linearization.__init__(self, func_def, mean, method)
        if self.method is None:
            self.method = UT
        self.covariance = covariance
        self.dimension = len(mean)

    def generate_samples(self):

        ut = UT(self.mean, self.covariance)
        return ut.generate_samples()

    def linear_matrix(self):

        ut_samples = self.generate_samples()
        Wc = ut_samples.Wc
        samples = ut_samples.samples
        linear_matrix = np.zeros((self.dimension, self.dimension))

        for k in range(ut_samples.num_samples()):
            x = samples[k]
            f = self.func_def(x)
            linear_matrix += Wc[k]*np.dot(f.reshape(self.dimension, 1), (x - self.mean).reshape(1, self.dimension))

        return linear_matrix

class LinearizationType(object):

    @staticmethod
    def linearization(func_def, mean, covariance=None, type='statistical', method=None):

        type = type.lower()
        if type == 'jacobian':
            return JacobianLinearization(func_def, mean, method)
        elif type == 'statistical':
            return StatisticalLinearization(func_def, mean, covariance, method)

