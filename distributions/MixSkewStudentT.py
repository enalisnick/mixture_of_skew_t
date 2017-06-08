import numpy as np
from scipy.special import digamma

from SkewStudentT import SkewStudentT


class MixSkewStudentT(object):

    def __init__(self, rng, nb_components=2): 
        
        # check parameters are correct
        assert nb_components > 1
        
        self.dim = dim
        self.nb_components = nb_components
        self.weights = [1./self.nb_components for k in xrange(self.nb_components)]
        self.component_dists = [SkewStudentT() for k in range(self.nb_components)]


    def estimate(self, data, max_iterations=100):
        n,d = data.shape
        assert d == self.components_dists[0].dim

        params = {'tau':np.zeros(n,self.nb_components), 
                  'e':[np.zeros(n,self.nb_components) for i in xrange(4)]
                  }

        for k in range(max_iterations):
            params = self.perform_E_step(data, params)
            self.perform_M_step(data, params)

        return params


    def perform_E_step(self, Y, params):
        n,d = Y.shape

        for j in xrange(n):
            ### posterior membership prob
            for h in range(self.nb_components):
                params['tau'][j,h] = self.weights[h] * self.component_dists[h].pdf(Y[j])
            params['tau'][j,:] /= params['tau'][j,:].sum()

            ### update e variables
            for h in range(self.nb_components):
                S = 0 ### Infinite integral?

                params['e'][0][j,h] = digamma(self.component_dists[h].df/2. + self.dim) - np.log((self.component_dists[h].df + self.component_dists[h].get_d(Y[j]))/2.) - T_inv(y)*S

                params['e'][1][j,h] = (self.component_dists[h].df + self.dim)/(self.component_dists[h].df + self.component_dists[h].get_d(Y[j])) * \
                    self.component_dists[h].stdT.impSamp_cdf(Y[j], mu=0., Sigma=self.component_dists[h].Lambda, df=self.component_dists[h].df+self.dim+2)/\
                    self.component_dists[h].stdT.impSamp_cdf(Y[j], mu=0., Sigma=self.component_dists[h].Lambda, df=self.component_dists[h].df+self.dim)

                epsilon = 
                params['e'][2][j,h] = e[1][j,h] * (self.component_dists[h] + epsilon)

                params['e'][3][j,h] = e[1][j,h] * E(X * X.T | y)

        return params


    def perform_M_step(self):
        
        ### update means
        for h in range(self.nb_components):
            mu_numerator = 0.
            mu_denominator = 0.
            for j in range(N):
                mu_numerator += tau[h][j] * (e2[h][j]*y - p_delta[h]*e3[h][j])
                mu_denominator += tau[h][j] * e2[h][j]
            self.mu[h] = mu_numerator / mu_denominator

        for h in range(self.nb_components):
            delta_partial1 = 0.
            delta_partial2 = 0.
            for j in range(N):
                delta_partial1 += tau[h][j]*e4[h][j]
                delta_partial2 += tau[h][j]*(y[j]-mu[h][j])*e3[h][j]
            delta[h] = inv(sigma_inv * delta_partial1) * np.diag(sigma_inv * delta_partial2)

        ### update scale
        for h in range(self.nb_components):
            tmp1 = 0.
            tmp2 = 0.
            for j in range(N):
                tmp1 += p_delta[h]* e4[h][j] * p_delta[h].T - (y[j] - mu[h]) * e3[h][j] * p_delta[h] \\
                    - p_delta[h] * e3[h][j] * (y[j] - mu[h]).T + (y[j] - mu[h][j]) * (y[j] - mu[h]).T * e2[h][j]
                tmp2 += tau[h][j]
            sigma[h] = tmp1/tmp2


        ### update degrees of freedom
        for h in range(self.nb_components):
            tmp1 = 0.
            tmp2 = 0.
            for j in range(N):
                tmp1 += tau[h][j] * (e2[h][j] - e1[h][j])
                tmp2 += tau[h][h]

            degrees_of_freedom[h] = tmp1/tmp2 

        return 
