import numpy as np
from scipy.stats import t as scipyT
from mixture_of_skew_t.distributions.SkewStudentT import *


if __name__ == '__main__':
    random_seed = 1234

    ### 1D example
    print "One-dimensional example..."
    t = SkewStudentT(mu=np.array([0,]), sigma=np.array([1,]))
    x = np.array([0,])

    # compare pdfs
    print "Compare PDFs..."
    print "Scipy: %.5f" %(scipyT.pdf(x, 1, loc=0, scale=1))
    print "This: %.5f" %(t.studentT_pdf(x=x))

    print

    # compare cdfs
    print "Compare CDFs..."
    print "Scipy: %.5f" %(scipyT.cdf(x, 1, loc=0, scale=1))
    print "This: %.5f" %(t.studentT_importance_sampled_cdf(x=x))

    print
    print

    ### 2D example
    print "Two-dimensional example..."
    t = SkewStudentT()
    x = np.array([0,0])

    # compare pdfs                                                                                                                               
    print "This: %.5f" %(t.studentT_pdf(x=x))

    print

    # compare cdfs                                                                                                                               
    #print "Scipy: %.5f" %(scipyT.cdf(x, 1, loc=0, scale=1))
    print "This: %.5f" %(t.studentT_importance_sampled_cdf(x=x))
    

