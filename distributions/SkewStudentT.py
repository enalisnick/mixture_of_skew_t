import numpy as np
import numpy.random
from math import *
from StudentT import *


class SkewStudentT(StudentT):

    def __init__(self, mu=np.zeros((1,2)), Sigma=np.eye(2), delta=np.ones((1,2)), df=3.): 
        
        # init regular Student T
        super(SkewStudentT, self).__init__(mu=mu, Sigma=Sigma, df=df)
        self.delta = delta

        # compute auxiliary variables
        # for pdf
        self.Delta = np.diag(self.delta[0])
        self.Omega = self.Sigma + np.dot(self.Delta, self.Delta)

        # for cdf
        self.Lambda = np.eye(self.dim) - np.dot(np.dot(self.Delta, np.linalg.inv(self.Omega)), self.Delta)
        

    def get_d(self, y):
        return np.dot(np.dot((y - self.mu[0]).T, np.linalg.inv(self.Omega)), (y - self.mu[0]))


    def get_c(self, y):
        return np.dot(np.dot(self.Delta, np.linalg.inv(self.Omega)), (y - self.mu[0]))


    def update_aux_params(self):
        self.Delta = np.diag(self.delta)
        self.Omega = self.Sigma + np.dot(self.Delta, self.Delta)
        self.Lambda = np.eye(self.dim) - np.dot(np.dot(self.Delta, np.linalg.inv(self.Omega)), self.Delta)


    def pdf(self, y, n_samples=100):
        # unrestricted skew student T pdf

        q = np.dot(np.dot(self.Delta, np.linalg.inv(self.Omega)), y-self.mu[0])
        dy = self.get_d(y)
        y1 = q * np.sqrt( (self.df + self.dim)/(self.df + dy) )

        return 2**self.dim * super(SkewStudentT,self).pdf(y, Sigma=self.Omega) * self.impSamp_cdf(y1, mu=0.*self.mu, Sigma=self.Lambda, df=self.df+self.dim, n_samples=n_samples)


    def draw_sample(self):
        w = np.random.gamma(shape = self.df/2., scale = 2./self.df)
        u = np.abs(np.random.multivariate_normal(mean = np.zeros(self.dim,), cov = 1./w * np.eye(self.dim))) 
        y = np.random.multivariate_normal(mean = self.mu[0] + np.dot(self.Delta, u), cov = 1./w * self.Sigma)
        
        return y
