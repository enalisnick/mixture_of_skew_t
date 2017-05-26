import numpy as np
from math import *

class SkewStudentT(object):

    def __init__(self, mu=np.zeros(2,), sigma=np.eye(2), deg_of_freedom=1): 
        
        # fancy init here
        dim = sigma.shape[0]

        # check that parameters are correct sizes
        assert dim == mu.shape[0] 
        #assert scale > 0
        assert deg_of_freedom > 0

        self.dim = dim
        self.mu = mu
        self.sigma = sigma
        self.df = deg_of_freedom


    def studentT_pdf(self, x):
        '''
        output:
            the density of the given element
        input:
            x = parameter (d dimensional numpy array or scalar)
        '''
        x = np.atleast_2d(x) # requires x as 2d
        nD = self.sigma.shape[0] # dimensionality

        numerator = gamma((self.dim + self.df) / 2.0)

        if self.dim > 1:
            denominator = (
                gamma(self.df / 2.0) * 
                np.power(self.df * np.pi, self.dim / 2.0) *  
                np.sqrt(np.linalg.det(self.sigma)) * 
                np.power(
                    1.0 + (1.0 / self.df) *
                    np.diagonal(
                        np.dot( np.dot(x - self.mu, np.linalg.inv(self.sigma)), (x - self.mu).T)
                    ), 
                    (self.dim + self.df) / 2.0
                    )
                )

        else:
            denominator = (
                gamma(self.df / 2.0) *
                np.power(self.df * np.pi, self.dim / 2.0) *
                np.sqrt(self.sigma) *
                np.power(
                    1.0 + (1.0 / self.df) *
                    (x - self.mu)**2 / self.sigma,
                    (self.dim + self.df) / 2.0
                    )
                )

        return (numerator / denominator)[0] 


    def studentT_importance_sampled_cdf(self, x, n_samples=10000):
        approx = 0.
        N = n_samples

        for s_idx in range(n_samples):
            sample = [None] * self.dim
            reject = True
            while reject and N > 0:
                sample_pdf = 1.
                reject = False
                for d in range(self.dim): 
                    u = np.random.uniform(low=0., high=1.)
                    s = np.tan(np.pi*(u-.5))
                    if x[d] > s: 
                        reject = True
                        break
                    sample[d] = s
                    sample_pdf *= 1./(np.pi * (1 + (s)**2)) # standard cauchy pdf 
                N -= 1

            sample = np.array(sample)
            
            if not reject:
                approx += self.studentT_pdf(sample)/sample_pdf

        return 1 - approx / n_samples


    #def pdf(self, x):
    #    return 2**self.dim * self.studentT_pdf(x) * self.

