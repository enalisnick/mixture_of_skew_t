import numpy as np
from math import *

def pdf(y, mu, Sigma, df):
    '''                                                                                                                                                                                                         
        output:                                                                                                                                                                                                     
            the density of the given element                                                                                                                                                                        
        input:                                                                                                                                                                                                      
            y = parameter (d dimensional numpy array or scalar)                                                                                                                                                     
    '''

    y = np.atleast_2d(y) # requires x as 2d                                                                                                                                                 
    dim = Sigma.shape[0]
    numerator = gamma((dim + df) / 2.0)

    if dim > 1:
        denominator = (
            gamma(df / 2.0) *
            np.power(df * np.pi, dim / 2.0) *
            np.sqrt(np.linalg.det(Sigma)) *
            np.power(
                1.0 + (1.0 / df) *
                np.diagonal(
                    np.dot( np.dot(y - mu, np.linalg.inv(Sigma)), (y - mu).T)
                ),
                (dim + df) / 2.0
                )
            )

    else:
        denominator = (
            gamma(df / 2.0) *
            np.power(df * np.pi, dim / 2.0) *
            np.sqrt(Sigma) *
            np.power(
                1.0 + (1.0 / df) *
                (y - mu)**2 / Sigma,
                (dim + df) / 2.0
                )
            )

    return (numerator / denominator)[0]


class StudentT(object):

    def __init__(self, mu=np.zeros((2,1)), Sigma=np.eye(2), df=3.): 
        
        # fancy init here
        dim = Sigma.shape[0]

        # check that parameters are correct sizes
        assert dim == mu.shape[1] 
        assert df > 2

        self.dim = dim
        self.mu = mu
        self.Sigma = Sigma
        self.df = df


    def pdf(self, y, mu=None, Sigma=None, df=None):
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


    def impSamp_cdf(self, y, mu=None, Sigma=None, df=None, n_samples=10000):
        # assumes a factorized std. Cauchy proposal
        if mu is None: mu = self.mu
        if Sigma is None: Sigma = self.Sigma
        if df is None: df = self.df

        dim = Sigma.shape[0]

        approx = 0.
        N = n_samples

        for s_idx in range(n_samples):
            sample = [None] * dim
            reject = True
            while reject and N > 0:
                proposal_pdf = 1.
                reject = False
                for d in range(dim): 
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
                approx += pdf(sample, mu, Sigma, df)/proposal_pdf

        return 1 - approx / n_samples
