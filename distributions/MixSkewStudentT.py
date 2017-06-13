import numpy as np
from scipy.special import digamma, gamma

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
        a = np.amin(Y) # for computing E[x]
        
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

                # compute moment E[x]
                c = self.component_dists[h].stdT.impSamp_cdf(self.component_dists[h].mu - a, mu = 0.)
                xi = np.zeros(self.component_dists[h].mu.shape)

                for d in range(self.dim):
                    mu_minus_d = np.delete(self.component_dists[h].mu, d, axis=0)
                    Sigma_minus_d = np.delete(np.delete(self.component_dists[h].Sigma, d, axis=0), d, axis=1)
                    sigma_d = np.delete(self.component_dists[h].Sigma[:,d], d, axis=0)

                    a_star = (mu_temp-a) - (mu_temp-a) * 1./self.component_dists[h].Sigma[d,d]
                    Sigma_star = (self.component_dists[h].df + 1./self.component_dists[h].Sigma[d,d] * (self.component_dists[h].mu[d] - a)**2)/(self.component_dists[h].df-1) *\
                        Sigma_minus_d - 1./self.component_dists[h].Sigma[d,d] * np.dot(sigma_d, sigma_d.T)

                    xi[d] = 1./(2*np.pi*self.component_dists[h].Sigma[d,d]) * (self.component_dists[h].df/(self.component_dists[h].df+(1./self.component_dists[h].Sigma[d,d])*(self.component_dists[h].mu[d] - a)**2))**((self.component_dists[h].df-1)/2.) * np.sqrt(self.component_dists[h].df/2) * gamma((self.component_dists[h].df-1)/2.)/gamma(self.component_dists[h].df/2.) * self.component_dists[h].stdT.impSamp_cdf(a_star, mu = 0., Sigma = Sigma_star, df=self.component_dists[h].df-1)

                epsilon = 1./c * np.dot(self.component_dists[h].Sigma, xi)
                E_x = self.component_dists[h].mu + epsilon

                params['e'][2][j,h] = e[1][j,h] * E_x

                # compute moment E[xx]
                H = np.zeros((self.dim, self.dim))
                for i in range(self.dim):
                    for j in range(self.dim):
                        if j != i:
                            # precompute the necessary slices
                            mu_ij = self.component_dists[h].mu[[i,j]]
                            Sigma_ij = np.array([[self.component_dists[h].Sigma[i,i], self.component_dists[h].Sigma[i,j]], [self.component_dists[h].Sigma[j,i], self.component_dists[h].Si\
gma[j,j]]])
                            if j > i:
                                mu_negij = np.delete(np.delete(self.component_dists[h].mu,i, axis=0), j-1, axis=0)
                                Sigma_parenij = np.delete(np.delete(self.component_dists[h].Sigma[:,[i,j]], i, axis=0), j-1, axis=0)
                                Sigma_negij = np.delete(np.delete(np.delete(np.delete(self.component_dists[h].Sigma, i, axis=0), j-1, axis=0), i, axis=1), j-1, axis=1)
                            else:
                                mu_negij = np.delete(np.delete(self.component_dists[h].mu,i, axis=0), j, axis=0)
                                Sigma_parenij = np.delete(np.delete(self.component_dists[h].Sigma[:,[i,j]], i, axis=0), j, axis=0)
                                Sigma_negij = np.delete(np.delete(np.delete(np.delete(self.component_dists[h].Sigma, i, axis=0), j, axis=0), i, axis=1), j, axis=1)

                            df_star = self.component_dists[h].df + np.dot(np.dot((mu_ij - a).T, inv(Sigma_ij)), mu_ij - a) 
                            a_star_star = (mu_negij - a) - np.dot(np.dot(Sigma_parenij, inv(Sigma_ij)), mu_negij - a)
                            Sigma_star_star = df_star/(self.component_dists[h].df - 2) * (Sigma_negij - np.dot(np.dot(Sigma_parenij, inv(Sigma_ij)), Sigma_parenij.T))

                            H[i,j] = 1./(2 * np.pi * np.sqrt(self.component_dists[h].Sigma[i,i]*self.component_dists[h].Sigma[j,j] - self.component_dists[h].Sigma[i,j]**2))
                            H[i,j] *= (self.component_dists[h].df)/(self.component_dists[h].df-2) * (self.component_dists[h].df/df_star)**(self.component_dists[h].df/2 - 1)
                            H[i,j] *= self.component_dists[h].stdT.impSamp_cdf(a_star_star, mu = 0., Sigma = Sigma_star_star, df = self.component_dists[h].df-2)
                        
                    H[i,i] = 1./self.component_dists[h].Sigma[i,i] * ((self.component_dists[h].mu[i] - a) * xi[i] - np.sum([self.component_dists[h].Sigma[i,k]*h[i,k] for k in range(self.dim) if k!=i])) 

                E_xx = np.dot(self.component_dists[h].mu, self.component_dists[h].mu.T) + np.dot(self.component_dists[h].mu, epsilon.T) + np.dot(epsilon, self.component_dists[h].mu.T) - 1./c * np.dot(np.dot(self.component_dists[h].Sigma, H), self.component_dists[h].Sigma) + 1./c * (self.component_dists[h].df)/(self.component_dists[h].df-2) * self.component_dists[h].stdT.impSamp_cdf(self.component_dists[h].mu - a, mu = 0., Sigma = (self.component_dists[h].df)/(self.component_dists[h].df-2) * self.component_dists[h].Sigma, df = self.component_dists[h].df-2) * self.component_dists[h].Sigma 
                
                params['e'][3][j,h] = e[1][j,h] * E_xx

        return params


    def perform_M_step(self, Y, params):
        n,d = Y.shape

        for h in range(self.nb_components):

            ### update mean
            mu_numerator = 0.
            mu_denominator = 0.            
            for j in range(n):
                mu_numerator += params['tau'][j,h] * (params['e'][1][j,h]*Y[j] - self.component_dists[h].Delta * params['e'][2][j,h])
                mu_denominator += params['tau'][j,h] * params['e'][1][j,h]
            
            self.component_dists[h].mu = mu_numerator / mu_denominator

            ### update delta
            delta_partial1 = 0.
            delta_partial2 = 0.
            for j in range(n):
                delta_partial1 += params['tau'][j,h] * params['e'][3][j,h]
                delta_partial2 += params['tau'][j,h] * (Y[j] - self.component_dists[h].mu) * params['e'][2][j,h]

            self.component_dists[h].delta = np.dot(inv(inv(self.component_dists[h].Sigma) * delta_partial1), np.diag(inv(self.component_dists[h].Sigma) * delta_partial2))
            self.component_dists[h].update_aux_params()  # need to recompute Lambda, Delta, Omega

            ### update Sigma
            tmp = np.zeros(self.component_dists[h].Sigma.shape)
            for j in range(n):
                tmp += params['tau'][j,h] * np.dot(np.dot(self.component_dists[h].Delta, params['e'][3][j,h].T), self.component_dists[h].Delta.T)
                tmp -= params['tau'][j,h] * np.dot(np.dot(Y[j] - self.component_dists[h].mu, params['e'][2][j,h].T), self.component_dists[h].Delta) 
                tmp -= params['tau'][j,h] * np.dot(np.dot(self.component_dists[h].Delta, params['e'][2][j,h]), (Y[j] - self.component_dists[h].mu).T) 
                tmp += params['tau'][j,h] * np.dot(Y[j] - self.component_dists[h].mu, (Y[j] - self.component_dists[h].mu).T) * params['e'][1][j,h]
            self.component_dists[h].Sigma = tmp/np.sum(params['tau'][:,h])

        
            ### update degrees of freedom
            tmp = 0.
            for j in range(n):
                tmp1 += params['tau'][j,h] * (params['e'][1][j,h] - params['e'][0][j,h])

            # THIS IS NOT CORRECT; NEED ITERATIVE SOLVING
            self.component_dists[h].df = tmp/np.sum(params['tau'][:,h])

        return 
