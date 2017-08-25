import logging
from sampling import UT
import numpy as np

class jac_linearization:

    def __init__(self,func_def,mean, epsilon=0.01, method='central',Abs_Tol = 1e-12):

        self.func_def = func_def
        self.epsilon = epsilon
        self.method = method
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

class stat_linearization:

    def __init__(self,func_def,mean,covariance, method='UT'):

        self.mean = mean
        self.func_def = func_def
        self.covariance = covariance
        self.method = method
        self.dimension = len(mean)

    def generate_samples(self):

        self._samples = UT(self.mean, self.covariance)
        self._samples.generate_num_samples()
        self._samples.generate_samples()

    def linear_matrix(self):

        self.generate_samples()
        linear_matrix = np.zeros((self.dimension, self.dimension))

        for k in range(self._samples.num_sample):
            x = self._samples.samples[k]
            f = self.func_def(x)
            linear_matrix += self._samples.Wc[k]*np.dot(f.reshape(self.dimension,1),(x-self.mean).reshape(1,self.dimension))

        return linear_matrix


class linearization:

    def __init__(self, lin_type, func, mean, params=None, covariance=None):

        lin_type = lin_type.lower()
        if lin_type == 'jacobian':

            jac_linearization.__init__(self,func_def=func, mean=mean,
                                  epsilon=params['epsilon'], method=params['method'])
        elif lin_type == 'statistical':

            if covariance is None:
                logging.warning('Default Covariance is assumed for Statistical Linearization')
            if params is None:
                params = {}
                params['method'] = None
            stat_linearization.__init__(self,func_def=func, mean=mean, covariance=covariance, method=params['method'])
        self.lin_type = lin_type

    def generate_samples(self):

        if self.lin_type == 'jacobian':
            jac_linearization.generate_samples(self)

        elif self.lin_type == 'statistical':
            stat_linearization.generate_samples(self)

    def linear_matrix(self):

        self.generate_samples()

        if self.lin_type == 'jacobian':
            linear_matrix = jac_linearization.linear_matrix(self)

        elif self.lin_type == 'statistical':
            linear_matrix = stat_linearization.linear_matrix(self)

        return linear_matrix


