import numpy as np
from math import *
from StudentT import *


class SkewStudentT(object):

    def __init__(self, mu=np.zeros(2,), Sigma=np.eye(2), delta=np.ones(2,), deg_of_freedom=1): 
        
        # fancy init here
        dim = Sigma.shape[0]

        # check that parameters are correct sizes
        assert dim == mu.shape[0] 
        assert dim == delta.shape[0]
        assert deg_of_freedom > 0

        self.dim = dim
        self.mu = mu
        self.Sigma = Sigma
        self.delta = delta
        self.df = deg_of_freedom

        # multivariate student T
        self.stdT = StudentT(mu=self.mu, Sigma=self.Sigma, df=self.df)

        # compute auxiliary variables
        self.Omega = 


    def get_d(self, y):
        
        np.dot(np.dot((y - self.mu).T, np.linalg.inv(Omega)), (y - self.mu))


    def pdf(self, y, n_samples=10000):
        # unrestricted skew student T pdf
        
        # PDF variables
        Delta = np.diag(self.delta)
        Omega = self.Sigma + np.dot(Delta, Delta)

        # CDF variables
        Lambda = np.eye(self.dim) - np.dot(np.dot(Delta, np.linalg.inv(Omega)), Delta)
        q = np.dot(np.dot(Delta, np.linalg.inv(Omega)), y-self.mu)
        dy = np.dot(np.dot((y - self.mu).T, np.linalg.inv(Omega)), (y - self.mu))
        y1 = q * np.sqrt( (self.df + self.dim)/(self.df + dy) )

        return 2**self.dim * self.stdT.pdf(y, Sigma=self.Omega) * self.stdT.impSamp_cdf(y1, mu=0.*self.mu, Sigma=Lambda, df=self.df+self.dim, n_samples=n_samples)

