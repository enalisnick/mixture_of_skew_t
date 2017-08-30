import numpy as np
from numpy.linalg import inv
from scipy.special import digamma, gamma
from scipy.optimize import minimize_scalar

from SkewStudentT import SkewStudentT


class MixSkewStudentT(object):

    def __init__(self, nb_components=2, weights=None, mus=None, Sigmas=None, deltas=None, dfs=None): 
        
        # check parameters are correct
        assert nb_components > 1
                
        self.nb_components = nb_components

        if weights == None:
            self.weights = [1./self.nb_components for k in xrange(self.nb_components)]
        else:
            self.weights = weights

        if mus == None:
            self.component_dists = [SkewStudentT() for k in range(self.nb_components)]
        else:
            self.component_dists = [SkewStudentT(mu=mus[k], Sigma=Sigmas[k], delta=deltas[k], df=dfs[k]) for k in range(self.nb_components)]

        self.dim = self.component_dists[0].dim


    def estimate(self, data, max_iterations=100):
        n,d = data.shape
        assert d == self.component_dists[0].dim

        print "Running E.M. for %d iterations..." %(max_iterations)
        for k in range(max_iterations):

            params = {'tau':np.zeros((n,self.nb_components)),
                  'e':[np.zeros((n,self.nb_components)), np.zeros((n,self.nb_components)), np.zeros((n, self.nb_components, self.dim)), np.zeros((n, self.nb_components, self.dim, self.dim))]}

            params = self.perform_E_step(data, params)
            self.perform_M_step(data, params)

            print "%d. Log Likelihood: %.4f" %(k+1, np.log(self.pdf(data)))

        return 


    def perform_E_step(self, Y, params, terms_in_int_approx=5):
        n,d = Y.shape
        a = np.amin(Y) # for computing E[x]
        
        for j in xrange(n):
            ### posterior membership prob
            for h in range(self.nb_components):
                params['tau'][j,h] = self.weights[h] * self.component_dists[h].pdf(Y[j])
            params['tau'][j,:] /= params['tau'][j,:].sum()

            ### update e variables
            for h in range(self.nb_components):

                # S integral
                S = gamma((self.component_dists[h].df + 2.*self.dim)/2.) / gamma((self.component_dists[h].df + self.dim)/2.)
                for r in xrange(terms_in_int_approx):
                    r += 1
                    for s in xrange(r):
                        S += ((-1)**(2*r-s-1) / r) * (gamma(r+1)/(gamma(s+1)*gamma(r-s+1))) * gamma((self.component_dists[h].df + self.dim)/2. + s) / gamma((self.component_dists[h].df + 2.*self.dim)/2. + s) * self.component_dists[h].impSamp_cdf(self.component_dists[h].get_c(Y[j]), mu=np.zeros((Y[j].shape[0],)), Sigma=((self.component_dists[h].df + self.component_dists[h].get_d(Y[j]))/(self.component_dists[h].df + self.dim + 2.*s))*self.component_dists[h].Lambda, df=self.component_dists[h].df+self.dim+2.*s)  

                params['e'][0][j,h] = digamma(self.component_dists[h].df/2. + self.dim) - np.log((self.component_dists[h].df + self.component_dists[h].get_d(Y[j]))/2.) - S/self.component_dists[h].impSamp_cdf(Y[j], mu=np.zeros((Y[j].shape[0],)), Sigma=self.component_dists[h].Lambda, df=self.component_dists[h].df+self.dim)

                params['e'][1][j,h] = (self.component_dists[h].df + self.dim)/(self.component_dists[h].df + self.component_dists[h].get_d(Y[j])) * \
                    self.component_dists[h].impSamp_cdf(Y[j], mu=np.zeros((Y[j].shape[0],)), Sigma=self.component_dists[h].Lambda, df=self.component_dists[h].df+self.dim+2)/\
                    self.component_dists[h].impSamp_cdf(Y[j], mu=np.zeros((Y[j].shape[0],)), Sigma=self.component_dists[h].Lambda, df=self.component_dists[h].df+self.dim)

                # compute moment E[x]
                c = self.component_dists[h].impSamp_cdf(self.component_dists[h].mu[0] - a, mu = np.zeros(self.component_dists[h].mu.shape))
                xi = np.zeros(self.component_dists[h].mu.shape)

                for d in range(self.dim):
                    mu_minus_d = np.delete(self.component_dists[h].mu, d, axis=1)
                    Sigma_minus_d = np.delete(np.delete(self.component_dists[h].Sigma, d, axis=0), d, axis=1)
                    sigma_d = np.delete(self.component_dists[h].Sigma[:,d], d, axis=0)
                    
                    a_star = (mu_minus_d-a) - (mu_minus_d-a) * 1./self.component_dists[h].Sigma[d,d]
                    Sigma_star = (self.component_dists[h].df + 1./self.component_dists[h].Sigma[d,d] * (self.component_dists[h].mu[0,d] - a)**2)/(self.component_dists[h].df-1) *\
                        Sigma_minus_d - 1./self.component_dists[h].Sigma[d,d] * np.dot(sigma_d, sigma_d.T)

                    xi[0,d] = 1./(2*np.pi*self.component_dists[h].Sigma[d,d]) * (self.component_dists[h].df/(self.component_dists[h].df+(1./self.component_dists[h].Sigma[d,d])*(self.component_dists[h].mu[0,d] - a)**2))**((self.component_dists[h].df-1)/2.) * np.sqrt(self.component_dists[h].df/2) * gamma((self.component_dists[h].df-1)/2.)/gamma(self.component_dists[h].df/2.) * self.component_dists[h].impSamp_cdf(a_star, mu = np.zeros((1, Sigma_star.shape[0])), Sigma = Sigma_star, df=self.component_dists[h].df-1)

                epsilon = 1./c * np.dot(xi, self.component_dists[h].Sigma)
                E_x = self.component_dists[h].mu + epsilon

                params['e'][2][j,h,:] = params['e'][1][j,h] * E_x

                # compute moment E[xx]
                H = np.zeros((self.dim, self.dim))
                for i in range(self.dim):
                    for j in range(self.dim):

                        if self.dim < 3: break

                        if j != i:
                            # precompute the necessary slices
                            mu_ij = np.array([[self.component_dists[h].mu[0,i], self.component_dists[h].mu[0,j]]])
                            Sigma_ij = np.array([[self.component_dists[h].Sigma[i,i], self.component_dists[h].Sigma[i,j]], [self.component_dists[h].Sigma[j,i], self.component_dists[h].Sigma[j,j]]])
                            if j > i:
                                mu_negij = np.delete(np.delete(self.component_dists[h].mu, i, axis=1), j-1, axis=1)
                                Sigma_parenij = np.delete(np.delete(np.array([self.component_dists[h].Sigma[:,i], self.component_dists[h].Sigma[:,j]]).T, i, axis=0), j-1, axis=0)
                                Sigma_negij = np.delete(np.delete(np.delete(np.delete(self.component_dists[h].Sigma, i, axis=0), j-1, axis=0), i, axis=1), j-1, axis=1)
                            else:
                                mu_negij = np.delete(np.delete(self.component_dists[h].mu, i, axis=1), j, axis=1)
                                Sigma_parenij = np.delete(np.delete(np.array([self.component_dists[h].Sigma[:,i], self.component_dists[h].Sigma[:,j]]).T, i, axis=0), j, axis=0)
                                Sigma_negij = np.delete(np.delete(np.delete(np.delete(self.component_dists[h].Sigma, i, axis=0), j, axis=0), i, axis=1), j, axis=1)

                            df_star = self.component_dists[h].df + np.dot(np.dot((mu_ij - a), inv(Sigma_ij)), (mu_ij - a).T)
 
                            
                            a_star_star = (mu_negij - a) - np.dot(np.dot(Sigma_parenij, inv(Sigma_ij)), mu_ij - a)
                            Sigma_star_star = df_star/(self.component_dists[h].df - 2) * (Sigma_negij - np.dot(np.dot(Sigma_parenij, inv(Sigma_ij)), Sigma_parenij.T))

                            H[i,j] = 1./(2 * np.pi * np.sqrt(self.component_dists[h].Sigma[i,i]*self.component_dists[h].Sigma[j,j] - self.component_dists[h].Sigma[i,j]**2))
                            H[i,j] *= (self.component_dists[h].df)/(self.component_dists[h].df-2) * (self.component_dists[h].df/df_star)**(self.component_dists[h].df/2 - 1)
                            H[i,j] *= self.component_dists[h].impSamp_cdf(a_star_star, mu = 0., Sigma = Sigma_star_star, df = self.component_dists[h].df-2)
                        
                    H[i,i] = 1./self.component_dists[h].Sigma[i,i] * ((self.component_dists[h].mu[0,i] - a) * xi[0,i] - np.sum([self.component_dists[h].Sigma[i,k]*H[i,k] for k in range(self.dim) if k!=i])) 

                E_xx = np.dot(self.component_dists[h].mu.T, self.component_dists[h].mu) + np.dot(self.component_dists[h].mu.T, epsilon) + np.dot(epsilon.T, self.component_dists[h].mu) - 1./c * np.dot(np.dot(self.component_dists[h].Sigma, H), self.component_dists[h].Sigma) + 1./c * (self.component_dists[h].df)/(self.component_dists[h].df-2) * self.component_dists[h].impSamp_cdf(self.component_dists[h].mu[0] - a, mu = np.zeros(self.component_dists[h].mu.shape), Sigma = (self.component_dists[h].df)/(self.component_dists[h].df-2) * self.component_dists[h].Sigma, df = self.component_dists[h].df-2) * self.component_dists[h].Sigma 
                
                params['e'][3][j, h, :, :] = params['e'][1][j, h] * E_xx

        return params


    def perform_M_step(self, Y, params):
        n,d = Y.shape

        updates = {'mu_s':[], 'Sigma_s':[], 'delta_s':[], 'df_s':[]}

        for h in range(self.nb_components):

            ### update mean
            mu_numerator = np.zeros((1, self.dim))
            mu_denominator = 0.            
            for j in range(n):
                mu_numerator += params['tau'][j,h] * (params['e'][1][j,h]*Y[j] - np.dot(self.component_dists[h].Delta, params['e'][2][j,h,:]))
                mu_denominator += params['tau'][j,h] * params['e'][1][j,h]
            
            self.component_dists[h].mu = mu_numerator / mu_denominator

            ### update delta
            delta_partial1 = 0.
            delta_partial2 = 0.
            for j in range(n):
                delta_partial1 += params['tau'][j,h] * params['e'][3][j,h,:,:]
                delta_partial2 += params['tau'][j,h] * np.dot((Y[j] - self.component_dists[h].mu).T, params['e'][2][j,h, :][np.newaxis])

            Sigma_inv = inv(self.component_dists[h].Sigma)
            self.component_dists[h].delta = np.diag( np.dot( inv(np.multiply(Sigma_inv,  delta_partial1) + .001*np.eye(self.dim)), np.diag(np.diag(np.dot(Sigma_inv, delta_partial2)))) )
            self.component_dists[h].update_aux_params()  # need to recompute Lambda, Delta, Omega

            ### update Sigma
            tmp = np.zeros(self.component_dists[h].Sigma.shape)
            for j in range(n):
                tmp += params['tau'][j,h] * np.dot(np.dot(self.component_dists[h].Delta, params['e'][3][j,h, :, :].T), self.component_dists[h].Delta.T)
                tmp -= params['tau'][j,h] * np.dot(np.dot((Y[j] - self.component_dists[h].mu).T, params['e'][2][j,h,:][np.newaxis]), self.component_dists[h].Delta) 
                tmp -= params['tau'][j,h] * np.dot(np.dot(self.component_dists[h].Delta, params['e'][2][j,h,:][np.newaxis].T), (Y[j] - self.component_dists[h].mu)) 
                tmp += params['tau'][j,h] * np.dot((Y[j] - self.component_dists[h].mu).T, Y[j] - self.component_dists[h].mu) * params['e'][1][j,h]

            self.component_dists[h].Sigma = tmp/np.sum(params['tau'][:,h])

        
            ### update degrees of freedom
            tmp = 0.
            for j in range(n):
                tmp += params['tau'][j,h] * (params['e'][1][j,h] - params['e'][0][j,h])
            tmp = tmp/np.sum(params['tau'][:,h])

            def df_eq(x):
                return np.log(x/2.) - digamma(x/2.) + 1. - tmp
            
            result = minimize_scalar(df_eq, bounds=(2, 100))
            self.component_dists[h].df = result.x

        return 


    def pdf(self, X):
        prob = 1.
        for n in range(X.shape[0]):
            point_prob = 0.
            for k in range(self.nb_components):
                point_prob += self.weights[k] * self.component_dists[k].pdf(X[n])
            prob *= point_prob

        return prob


    def draw_sample(self, nb_samples=500):
        samples = []
        comp_idxs = np.random.multinomial(n=nb_samples, pvals=self.weights)

        for k, dist in enumerate(self.component_dists):
            for i in range(comp_idxs[k]):
                samples.append(dist.draw_sample())

        samples = np.array(samples)
        np.random.shuffle(samples)

        return samples
        
