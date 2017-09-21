import Oslo as O 
import Sites as S
import numpy as np 
import random as r
import matplotlib.pyplot as plt
import sys
from scipy.optimize import curve_fit 

system_sizes = [8,16,32,64,128,256]   ### remeber this ###
critical_heights = []
av_heights = []
critical_times = []
std=[]
sPs=[]











        ## plotting the Average heights against sys sizes ##          

def get_Average_height_dep():
    for size in system_sizes:
    
       
        av_heights.append(O.Oslo(size,70000).get_critical_height())
        
    
    plt.figure()
    plt.plot(system_sizes,av_heights)
    plt.title("Average pile heights after cross-over time against System sizes")
    plt.xlabel("System size")
    plt.ylabel("Height")                                 
    plt.show()    
    
 
    
       ## plotting the critical times against sys sizes ##     
def get_critical_time_dep():
    
    for size in system_sizes:
        critical_times.append(size**2+size)
       
        
        
    
    plt.figure()
    plt.plot(system_sizes,critical_times)
    plt.title("Critical/Cross-over time against System sizes")
    plt.xlabel("System size")
    plt.ylabel("Time, tc")                                 
    plt.show()  
    

## plotting the first order scaling correction for 2c##
def get_w1():
    
    
    #loaded_av_heights = np.loadtxt('/Users/darrelladjei/python/C&N/data/tc_h_sys.txt')
    
    Beta = np.array(av_heights)/system_sizes
    
    
    
            
    def func(L,a0,b,w1):
        return a0 - b*(L**(-w1))
        
    popt, pcov = curve_fit(func,system_sizes,Beta)
    

    print "this is popt",popt,"this is pcov", pcov
    plt.plot(system_sizes, func(system_sizes,popt[0],popt[1],popt[2]))
    plt.title("average height / L against L")
    plt.ylabel("average height / L ")
    plt.xlabel("L") 
    plt.grid()
    plt.show()
    
    

def get_std_data():
    
        for size in system_sizes:
           std.append(O.Oslo(size,70000).get_std())
           
        
        #np.savetxt('/Users/darrelladjei/python/C&N/data/stdsMe.txt',std)
           
 
  
## plotting the stds for 2c##
def get_std_plot():
    
    #stds= np.loadtxt('/Users/darrelladjei/python/C&N/data/std.txt')
    
    stdss = np.array([ 0.96247271,  1.13778563,  1.32832039,  1.59046084,  1.86678003,
        2.17126726])
 
    
    def func(L,a0,D,b,w1):
        return a0*(L**D)*(1-b*(L**(-w1)))
        
    popt, pcov = curve_fit(func, system_sizes, stdss, p0=None, sigma=None, absolute_sigma=False, check_finite=True, method=None)
    

    print "this is popt",popt
    plt.plot(system_sizes, func(system_sizes,popt[0],popt[1],popt[2],popt[3]),'o')
    plt.plot(system_sizes,stdss)
    plt.title("Average Height-STDs against L")
    plt.xlabel(" system size")
    plt.ylabel("STD")        
    plt.grid()
    plt.show()

    
def get_D_for_3c():  ##for fitting to peaks on vertically collapsed data##
    
    for size in system_sizes:
    
        sPs.append(O.Oslo(size,70000).get_sPs())
      
    
    def func(L,b,D):
        
        return b*(L**(D))
        
    popt, pcov = curve_fit(func,system_sizes,sPs)
    
    print "this is D",popt[1]
        
    plt.figure()
    plt.plot(system_sizes,sPs,'o-')
    plt.grid()
    plt.title("plot of system sizes v Smax")
    plt.xlabel("system size")
    plt.ylabel("Smax")                                 
    plt.show()
        
    

def get_D_dependence():      #### ascertaing the relationship between k and D(1+k-ts) ####
    
    D_dependence = []
    
    for k in range(1,6):
        
        print "this is k",k
        
        k_th_mom = []
        
        for size in system_sizes:
            
            k_th_mom.append(O.Oslo(size,70000).get_kth_mom(k))
        #print k_th_mom
        def func(x,m,c):
            
            return m*x + c
            
        popt, pcov = curve_fit(func,np.log(system_sizes),np.log(k_th_mom))
        
        D_dependence.append(popt[0])
        k_th_mom = []
            
    #np.savetxt('/Users/darrelladjei/python/C&N/data/D_dependence.txt',D_dependence)
    return D_dependence

        
def get_D_and_ts():
    
    
    ##from saved textfile##
    D_dependence = [1.004789137368836904e+00, 3.159550334248789838e+00, 5.207625143715613802e+00, 7.843057928866399742e+00, 9.480698533484751778e+00]

    k_s = np.arange(1,6)
    
    def func(x,m,c):
            
        return m*x + c
            
    popt1, pcov1 = curve_fit(func,k_s,D_dependence)
    
    D = popt1[0]    
    print "this is D",D
    
    n_k_s =np.arange(0,6,0.01)
    
   
    plt.plot(n_k_s,func(n_k_s,popt1[0],popt1[1]),'o-')
    ax = plt.gca()

    line = ax.lines[0]

    xvalues = line.get_xdata()
    
    yvalues = line.get_ydata()
    
    idx = np.where(yvalues >= 0 ) 
    
    k_intercept = xvalues[idx[0][0]]
    
    print 'this is K_intercept', k_intercept
    plt.show()
    
    
  
       
 
    
    
    

    
    
                                                                                                                                                                                                                                                                                                      
  
                                                                                              
                                                                  
                                                                             
                                                                                                   
        