import numpy as np


class SkewStudentT(object):

    def __init__(self, dims=1, loc=0., scale=1., deg_of_freedom=1): 
        
        # fancy init here

        # check that parameters are correct sizes
        assert dims > 0 
        assert scale > 0
        assert deg_of_freedom >= 1

        self.dims = dims
        self.loc = loc
        self.scale = scale
        self.v = deg_of_freedom
    



