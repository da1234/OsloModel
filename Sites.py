import numpy as np 

import random as r 


class site:
    
    
    def __init__(self,threshold,height = 1):
        
        self.h =height 
        self.z_th = threshold
        self.z = 0
        self.stability = True 
        
        
    
    def get_height(self):
        
        return self.h
        
        
    def set_height(self, n_height):

        self.h = n_height 
        
        
        
    def set_threshold(self, n_threshold):
        
        self.z_th = n_threshold 
        
        
    def get_threshold(self):
        
        return self.z_th
        
    
    def get_slope(self):
        
        return self.z
        
    def set_slope(self, n_slope):
        
        self.z = n_slope 
        
        
    def is_stable(self):
        
        
        if self.get_slope() > self.get_threshold():
            
            self.stability =False
            
        else:
            
            self.stability =True
            
            
        return self.stability
        
        
        
        
    
    
        
        
        
        
        
        