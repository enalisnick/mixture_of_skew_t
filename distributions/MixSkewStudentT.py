import numpy as np
from SkewStudentT import SkewStudentT


class MixSkewStudentT(object):

    def __init__(self, rng, nb_components=2): 
        
        # check parameters are correct
        assert nb_components > 1
        
        self.component_dists = [SkewStudentT() for k in range(nb_components)]


    def estimate(self, max_iterations=100):
        
        for iter_idx in range(max_iterations):
            
            self.perform_E_step()
            self.perform_M_step()


    def perform_E_step(self):
        return None


    def perform_M_step(self):
        return None
