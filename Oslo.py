import numpy as np 
import random as r
import Sites as S
import matplotlib.pyplot as plt
import sys 
from log_bin_CN_2016 import *
from scipy.optimize import curve_fit

import math 

class Oslo:
    
    
    def __init__(self,L,duration):
        
        self.List_size = L
        self.sys_time = 0 
        self.D_numbs = []   ## list of drop of numbers for each time step
        self.G_numbs = []   ## list of grain numbers for each time step
        self.duration = duration ## how long should the program run for i.e. max number of grains added  
        self.critical_time = L**2 + L
        self.avalanches = []     ## list of avalanches after each drive ##
        self.critical_height =0      ## for smoothed out heights ##
        
        self.average_height = 0     ## for task 2b ##
        self.std = 0 
        
        self.sps_max_val = 0            ##for task 3c##
        
        
        self.kth_moment = 0             ### for task 3d ......finding the kth_moment###
        
        self.list_of_sites = []
        
        for i in range(L):
            
            site = S.site(threshold =r.randint(1,2))
            self.list_of_sites.append(site)
            
        self.run()
            
                
    def drive(self):
        
    
  
        self.sys_time += 1
        self.G_numbs.append(1)
        print self.sys_time
             
        
        left_most = self.list_of_sites[0]
        
     
        self.list_of_sites[0].set_slope(left_most.get_slope() + 1)          

        #print self.list_of_sites[0].get_height()
     
        
    def relax(self):
        
        ## height of first site ##
        
        
        drop_of = 0 
        s = 0 
        height = 0
        
        while not all([x.is_stable() for x in self.list_of_sites]):
            
    
            for i in range(len(self.list_of_sites)):

                if not self.list_of_sites[i].is_stable():
                    
                    s +=1      ## adding each site that topples over ##
                    
                    drop_of += self._relax(i)
        
                
        self.avalanches.append(s)
        
        for i in range(len(self.list_of_sites)):
                height+= self.list_of_sites[i].get_slope()   
                    
        self.list_of_sites[0].set_height(height)  
        self.D_numbs.append(drop_of)
        
                
                
        
        
    def _relax(self,pos):
        
        drop_of = 0 
        

        #####relaxing ######
        
        site_i =  self.list_of_sites[pos]
        #site_i_next = self.list_of_sites[pos+1]
        #site_i_previous = self.list_of_sites[pos-1]
        
        if pos == 0:
            
            site_i_next = self.list_of_sites[pos+1]
            self.list_of_sites[pos].set_slope(site_i.get_slope()-2)
            self.list_of_sites[pos+1].set_slope(site_i_next.get_slope()+1)
            
        elif pos>0 and pos < (len(self.list_of_sites)-1):
            

            site_i_next = self.list_of_sites[pos+1]
            site_i_previous = self.list_of_sites[pos-1]
            self.list_of_sites[pos].set_slope(site_i.get_slope()-2)
            self.list_of_sites[pos+1].set_slope(site_i_next.get_slope()+1)
            self.list_of_sites[pos-1].set_slope(site_i_previous.get_slope()+1)
            
        elif pos == (len(self.list_of_sites)-1):
            
            site_i_previous = self.list_of_sites[pos-1]          
            self.list_of_sites[pos].set_slope(site_i.get_slope()-1)
            self.list_of_sites[pos-1].set_slope(site_i_previous.get_slope()+1)
            drop_of +=1 
            
        else:
            pass 
        
            
     
        ##### changing threshold#####
    
      
        self.list_of_sites[pos].set_threshold( r.randint(1,2))
     
            
            
        return drop_of
        
        
    ### getting key parameters ###
        
    def get_var(self, heights):
        
        var = np.std(heights)
        
        return var
        
    def set_critical_height(self, heights,critical_time): ## using the smoothed heights ##
        
        heights_to_av = heights[critical_time:]
        critical_height = np.mean(heights_to_av)
        self.critical_height = critical_height
        
    
    def get_critical_height(self):   ## for the smoothed heights ##
        
        return self.critical_height
        
        
        
    def set_critical_time(self, critical_time):
        
        self.critical_time = critical_time 


    def get_critical_time(self):
        
        return self.critical_time
                
                
    def get_average_height(self): ## for original heights ##
        
        
        return self.average_height
    
    
    def set_average_height(self, heights,critical_time):
        
        av = 0 
        total = np.sum(heights[critical_time+1:])
        T = len(heights[critical_time+1:])
        av = (1.0/float(T))*total 
        self.average_height = av 
        
        #print av 
    
    def set_std(self,heights,critical_time):
    
        #std = np.std(heights[critical_time+1:])
        #self.std = std
        
        std=0
        h2 = 0 
        heights_array = np.array(heights[critical_time+1:])
        #print "heights_array",heights_array
        total = np.sum((heights_array)**2)
        #print "total",total
        T = len(heights[critical_time+1:])
        h2 = (1.0/float(T))*total 
        av_squared = (np.array(self.get_average_height()))**2
        std = math.sqrt(h2 - av_squared)
        
        self.std = std
        
        
        
        
        
        
    def get_std(self):
        
        return self.std
        
        
                
        
        
        
    def h_prob(self,height,heights,critical_time):
        
        cnt = 0
        instances = 0  
        prob = 0
        
        for h in heights[critical_time+1:]:
            
            if h == height:
                instances+=1
               

            
            cnt+=1
            
        prob = float(instances)/float(cnt)
        print "this is prob of 27", prob
        
        return prob 
        
    def get_h_prob_plot_DC(self,heights):
        
        cnt = 0
        prob_dictionary = {}
        for height in heights[self.critical_time+1:]:
            cnt+=1
            prob_dictionary.setdefault(height,0)
            prob_dictionary[height]+=1
            
        for size,count in prob_dictionary.items():
            
            prob_dictionary[size] /= float(cnt)
            
        #print "these are the probs of avalanche for a sysyem size",self.List_size,":", prob_dictionary
        
    
        print "std:",self.get_std()
        print "mean:", self.get_average_height()
        plt.figure()
        plt.plot((prob_dictionary.keys()-np.array(self.get_average_height()))/(self.get_std()),np.array(prob_dictionary.values())*self.get_std(),'o-')
        plt.title("Prob of pile with hieght h")
        plt.xlabel("height, h")
        plt.ylabel("Probability")
        plt.show()
        
        
    def get_h_prob_plot(self,heights):
        
        cnt = 0
        prob_dictionary = {}
        for height in heights[self.critical_time+1:]:
            cnt+=1
            prob_dictionary.setdefault(height,0)
            prob_dictionary[height]+=1
            
        for size,count in prob_dictionary.items():
            
            prob_dictionary[size] /= float(cnt)
            
        #print "these are the probs of avalanche for a sysyem size",self.List_size,":", prob_dictionary
        
    
        print "std:",self.get_std()
        print "mean:", self.get_average_height()
        plt.figure()
        plt.plot(prob_dictionary.keys(),np.array(prob_dictionary.values()),'o-')
        plt.title("Prob of pile with hieght h")
        plt.xlabel("height, h")
        plt.ylabel("Probability")
        plt.show()
        
    
    
    ### for scaling function of heights ###
    
    def plot_collapsed_data(self,heights,times,critical_time):
        
        new_heights = np.array(heights, dtype=float)
        new_times = np.array(times, dtype=float)
        scaled_heights = new_heights/self.List_size
        scaled_times = new_times/((self.List_size)**2)
      
        plt.figure()
        plt.plot(scaled_times,scaled_heights)
        plt.title("Data collapse for processed pile heights against time")
        plt.xlabel("time, grains added")
        plt.ylabel("moving average height")
    
        plt.show()
        
        
    ### looking at avalanche sizes now ##
    
    def S_probability(self,s):
        
        cnt = 0
        instances = 0  
        prob = 0
        
        for avalanche in self.avalanches[self.critical_time+1:]:
            
            if avalanche == s:
                instances+=1
            cnt+=1
            
        prob = float(instances)/float(cnt)
        print "this is prob of an avalanche with size",s,":", prob
        
        return prob 
        
    
    def get_sPs(self):
        
        return self.sps_max_arg
    
        
    def get_S_prob_plot(self):
        
        cnt = 0
        prob_dictionary = {}
        for avalanche in self.avalanches[self.critical_time+1:]:
            cnt+=1
            prob_dictionary.setdefault(avalanche,0)
            prob_dictionary[avalanche]+=1
            
        for size,count in prob_dictionary.items():
            
            prob_dictionary[size] /= float(cnt)
            
        #print "these are the probs of avalanche for a sysyem size",self.List_size,":", prob_dictionary
        
        centres, counts = log_bin(self.avalanches[self.critical_time+1:], bin_start=1., first_bin_width=1., a=2., datatype='float')
        
        sPs = np.array(counts)*(np.array(centres))**1.53
        D = 2.2
        self.sps_max_arg = centres[np.argmax(np.log(sPs))]
        
        print "arg max",self.sps_max_arg 
        
            
        def func(x,m,c):
            
            return m*x + c
            
        popt, pcov = curve_fit(func,np.log(centres),np.log(counts))
        #print "this is popt",popt,"this is pcov", pcov
        
        
        
        plt.figure()
        plt.plot(np.log(prob_dictionary.keys()),np.log(prob_dictionary.values()),'o')
        #plt.plot(np.log(prob_dictionary.keys()),np.log(sPs),'o')
        plt.plot(np.log(centres),np.log(counts),'o-',label='binned data')
        plt.plot(np.log(centres),np.log(sPs),'o-',label='vertical collapse')
        plt.plot(np.log(np.array(centres)/(self.List_size)**D),np.log(sPs),'o-',label='horizontal & vertical collapse')
        plt.title("Prob of avalanche with size S")
        plt.xlabel("size, S")
        plt.ylabel("Probability")
        plt.legend(loc='lower left')
        plt.show()
        
                
    
    
    def get_kth_mom(self,k):
        
        k_th = 0 
        total = np.sum(np.array(self.avalanches[self.critical_time+1:],np.float64)**k)
        T = len(self.avalanches[self.critical_time+1:])
        k_th = (1.0/float(T))*total 
        self.kth_moment = k_th 
        return k_th
        
          
    
    def run(self):
        
        D_array  = np.array(self.D_numbs)
        G_array = np.array(self.G_numbs)
        heights = [] 
        time = np.arange(self.duration)  ### for normal heights 
        
    
        for i in range(self.duration):    
            self.drive()
            self.relax()
            
            D_array  = np.array(self.D_numbs)
            G_array = np.array(self.G_numbs)
            heights.append(self.list_of_sites[0].get_height())
     
        # print "next step"
          
        ### for the time average ##
        heights_av_array =[]
        bottom_height = 0
        
                                   
        ##  moving av ###
                                        
        time_av = np.arange(25, self.duration-25,1)    
        to_average = []
        
     
                ## looking at the moving average ##
        for i in range(51,len(heights)+1,1):
             
            to_average = heights[bottom_height:i]
            total = sum(to_average)
            average = ((1.0)/((2.0)*(25.0) + (1.0)))*total
            heights_av_array.append(average)
            bottom_height +=1
        
        self.set_critical_height(heights_av_array,self.critical_time)
        

        ### averaging the normal heights again for task 2b ###
        
        self.set_average_height(heights,self.critical_time)
       
     
        ### for the scaling function ###
        self.plot_collapsed_data(heights_av_array,time_av,self.critical_time)
        
        ## avalanche probability density ##
        self.get_S_prob_plot()
        
        self.set_std(heights,self.critical_time)
        
        
        ##plotting the height probabilities##
        self.get_h_prob_plot(heights)
        self.get_h_prob_plot_DC(heights)
           
        
        ##for plotting the heights normal without smoothing##
        plt.figure()           
        plt.plot(time,heights,label='Size:'+''+str(self.List_size))
        plt.title("Pile heights against time")
        plt.xlabel("time, grains added")
        plt.ylabel("Pile height")
        
        plt.figure()
        plt.plot(time_av,heights_av_array)
        plt.title("Averaged pile heights against time")
        plt.xlabel("time, grains added")
        plt.ylabel("moving average height")
        plt.show()
 
        
                     
os = Oslo(16,6000)
       
            
             
        
        
        
            
            
        
        
        
        
        