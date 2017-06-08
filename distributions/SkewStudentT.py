import numpy as np
from math import *


class SkewStudentT(object):

    def __init__(self, mu=np.zeros(2,), Sigma=np.eye(2), delta=np.ones(2,), deg_of_freedom=1): 
        
        # fancy init here
        dim = Sigma.shape[0]

        # check that parameters are correct sizes
        assert dim == mu.shape[0] 
        assert deg_of_freedom > 0

        self.dim = dim
        self.mu = mu
        self.Sigma = Sigma
        self.delta = delta
        self.df = deg_of_freedom


    def studentT_pdf(self, y, mu=None, Sigma=None, df=None):
        '''
        output:
            the density of the given element
        input:
            y = parameter (d dimensional numpy array or scalar)
        '''
        if mu is None: mu = self.mu
        if Sigma is None: Sigma = self.Sigma
        if df is None: df = self.df

        y = np.atleast_2d(y) # requires x as 2d
        nD = Sigma.shape[0] # dimensionality
        numerator = gamma((self.dim + df) / 2.0)

        if self.dim > 1:
            denominator = (
                gamma(df / 2.0) * 
                np.power(df * np.pi, self.dim / 2.0) *  
                np.sqrt(np.linalg.det(Sigma)) * 
                np.power(
                    1.0 + (1.0 / df) *
                    np.diagonal(
                        np.dot( np.dot(y - mu, np.linalg.inv(Sigma)), (y - mu).T)
                    ), 
                    (self.dim + df) / 2.0
                    )
                )

        else:
            denominator = (
                gamma(df / 2.0) *
                np.power(df * np.pi, self.dim / 2.0) *
                np.sqrt(Sigma) *
                np.power(
                    1.0 + (1.0 / df) *
                    (y - mu)**2 / Sigma,
                    (self.dim + df) / 2.0
                    )
                )

        return (numerator / denominator)[0] 


    def studentT_importance_sampled_cdf(self, y, mu=None, Sigma=None, df=None, n_samples=10000):
        # assumes a factorized std. Cauchy proposal
        if mu is None: mu = self.mu
        if Sigma is None: Sigma = self.Sigma
        if df is None: df = self.df

        approx = 0.
        N = n_samples

        for s_idx in range(n_samples):
            sample = [None] * self.dim
            reject = True
            while reject and N > 0:
                proposal_pdf = 1.
                reject = False
                for d in range(self.dim): 
                    u = np.random.uniform(low=0., high=1.)
                    s = np.tan(np.pi*(u-.5))
                    if y[d] > s: 
                        reject = True
                        break
                    sample[d] = s
                    proposal_pdf *= 1./(np.pi * (1 + (s)**2)) # standard cauchy pdf 
                N -= 1

            sample = np.array(sample)
            
            if not reject:
                approx += self.studentT_pdf(sample, mu, Sigma, df)/proposal_pdf

        return 1 - approx / n_samples


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

        return 2**self.dim * self.studentT_pdf(y, mu=self.mu, Sigma=Omega, df=self.df) * \
            self.studentT_importance_sampled_cdf(y1, mu=0.*mu, Sigma=Lambda, df=df+self.dim, n_samples=n_samples)

