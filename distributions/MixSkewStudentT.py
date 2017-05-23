import numpy as np
from SkewStudentT import SkewStudentT


class MixSkewStudentT(object):

    def __init__(self, rng, nb_components=2): 
        
        # check parameters are correct
        assert nb_components > 1
        
        self.nb_components = nb_components
        self.component_dists = [SkewStudentT() for k in range(self.nb_components)]


    def estimate(self, max_iterations=100):
        
        for iter_idx in range(max_iterations):
            
            self.perform_E_step()
            self.perform_M_step()


    def perform_E_step(self, X):
        
        ### update mixture weights
        tau = [None]*self.nb_components
        mix_weight_normalizer = 0.

        for h in range(self.nb_components):
            tau[h] = self.pi[h] * self.component_dists[h].pdf(X)
            mix_weight_normalizer += tau[h]
        
        for h in range(self.nb_components):
            tau[h] /= mix_weight_normalizer


        ### update e variables
        for h in range(self.nb_components):
            S = 0 ### Infinite integral?
            e1[h] = psi(self.component_dists[h].degrees_of_freedom/2. + self.dims) - np.log( self.component_dists[h].degrees_of_freedom + d[h](y)/2. ) - T_inv(y)*S
            e2[h] = (self.component_dists[h].degrees_of_freedom + self.dims)/(self.component_dists[h].degrees_of_freedom + d[h]) * T()/T()
            e3[h] = e2[h] * E(x | y)
            e4[h] = e[2] * E(X * X.T | y)

        return 


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
