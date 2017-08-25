import numpy as np

class Sample:

    def __init__(self,mean, covariance):

        self._mean = mean
        self._covariance = covariance
        self.dim = len(self._mean)

class UT(Sample):

    def __init__(self, mean, covariance, config=None):

        Sample.__init__(self, mean, covariance)
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
        self._lambda = (self.alpha**2)*(self.dim + self.ki) - self.dim

    def generate_num_samples(self):

        self.num_sample = 2 * self.dim + 1

    def generate_samples(self):

        self.generate_num_samples()
        self.c = self._lambda + self.dim

        self.Wm = np.zeros((self.num_sample,))
        self.Wc = np.zeros((self.num_sample,))

        self.Wm = np.r_[(self._lambda/self.c), np.ones((2*self.dim,))*(0.5/self.c)]
        self.Wc = np.r_[(self._lambda/self.c + 1 - self.alpha**2 + self.beta), np.ones((2*self.dim,))*(0.5/self.c)]

        U = np.sqrt(self.c)*np.linalg.cholesky(self._covariance).T
        self.samples = np.vstack(
            (self._mean, np.tile(self._mean, (self.dim, 1)) + U, np.tile(self._mean, (self.dim, 1)) - U))

    def compute_moment(self):

        self.mean_est = np.average(self.samples,weights=self.Wm,axis=0)
        self.covariance_est = np.zeros((self.dim,self.dim))
        for k in range(self.num_sample):
            temp = (self.samples[k] - self.mean_est).reshape((self.dim,1))
            self.covariance_est = self.covariance_est + self.Wc[k]*np.dot(temp,temp.T)


