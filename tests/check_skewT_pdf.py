import numpy as np
from scipy.stats import t as scipyT
from mixture_of_skew_t.distributions.SkewStudentT import *


if __name__ == '__main__':
    random_seed = 1234

    ### 1D example
    print "One-dimensional example..."
    t = SkewStudentT(mu=np.array([0,]), Sigma=np.array([1,]), delta=np.array([1,]))
    x = np.array([0,])
    print t.pdf(x)

    ### 2D example
    print "Two-dimensional example..."
    t = SkewStudentT(mu=np.array([0,0]), Sigma=np.eye(2), delta=np.array([1,1]))
    x = np.array([0,0])
    print t.pdf(x)
    

