from re import A
import sys
import matplotlib as mpl
from ripser import ripser
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy
import scipy.integrate as integrate
import scipy.stats as stats
from scipy.optimize import curve_fit
import time
from tqdm import trange
import matplotlib.colors as colors
import matplotlib.cbook as cbook
import warnings
warnings.filterwarnings("ignore")

def run_metro(TI_np, n_iterations, r: float = 300, verbose: bool = False, flip: bool = False, SNR: float = 10000):
    n_iterations = int(n_iterations * 1.1) #burn in
    #Set the signal-to-noise ratio and standard deviation of the gaussian noise, which can be roughly estimated as 1/SNR
    #Normalization Constant to get the T_ij and c_j around the same range
    #Based on the decay time of pure water
    t_norm = 3000
    #Set up a 2D grid of points with a TI axis and a TE axis
    #TI: Inversion time, TE: Echo time, refer to time values over which MRI images are taken
    #_til refers to normalized values
    #Here TI and TE are a 2D meshgrid, while TI_set and TE_set are the 1D versions
    n_TE = 64
    TE_max = 512
    TE_set = np.linspace(TE_max/n_TE, TE_max, n_TE)
    #Define true values of the 6 parameters
    #For future usage, these parameters may be made into function parameters
    c1 = 0.5
    c2 = 0.5
    T11 = 600
    T12 = 1200
    T21 = 45
    T22 = 100
    noise_sd = (c1+c2)/SNR

    def S_4_param(TE, d1, d2, T21, T22):
        #Gives a 4 parameter, 1D model. Comes in when plugging in a constant value for TI
        #d1 is defined as c1*(1-2*exp(-TI/T11)) which is a constant for constant values of TI
        model = d1*np.exp(-TE/T21)+d2*np.exp(-TE/T22)
        return model

    def noise_TE(sd):
        #Adds noise to the 1D model across the TE axis with constant TI
        return np.random.normal(0, sd, n_TE)

    def pdf_np_4_param(data, sd):
        #Returns the Bayesian PDF of the 1D model, here with 4 parameters
        #Given the data (generated for a particular TI) and noise sd, 
        #returns Bayesian posterior joint PDF as a function of TE, d1, d2, T21, and T22
        def joint_pdf(TE, d1, d2, T21, T22):
            model = S_4_param(TE, d1, d2, T21, T22)
            residual = (data-model)**2
            return np.exp(1/(n_TE)*-(1/(2*sd**2))*residual.sum())
        return joint_pdf

    #This is the version of Metropolis most useful for us
    def prior_4P(d1, d2, T21, T22):
        #Set the prior as a 4D box with probability of zero outside
        isGreater = np.array([d1,d2,T21,T22]) > upper_bound_4P
        isLower = np.array([d1,d2,T21,T22]) < lower_bound_4P
        for greater in isGreater:
            if greater == True:
                return 0
        for lower in isLower:
            if lower == True:
                return 0
        if T21>T22:
            return 1
        return 1

    def transition_model_d_constrained(di_sum, d1, T21, T22):
        #Same as above, but requires that d1+d2 = di_sum, a constant
        #We can do this because generally d1+d2 will be a constant for a given TI
        dd1 = step_d1*np.random.normal(0,1)
        #dd2 = di_sum - dd1
        dT21 = step_T21*np.random.normal(0,1)
        dT22 = step_T22*np.random.normal(0,1)
        
        return(d1+dd1, di_sum-(d1+dd1), T21+dT21, T22+dT22)

    def acceptance_4P(r_current, r_temp):
        #returns true if new parameters are accepted
        #Basically same as above acceptance step
        if r_temp > r_current:
            return True
        else:
            accept = np.random.uniform(0,1)
            return (accept < r_temp/r_current)
        
    def metropolis_4P(likelihood, prior, transition_model, param_init, n_iterations, data, acceptance,verbose):
        #Takes the likelihood function from pdf_np_4_param function, and other metropolis helper functions as defined above
        #param_init are the initial parameters for Metropolis-Hastings
        #n_iterations is how many total steps the algorithm will take
        #Set current values to the initial parameters
        d1_c, d2_c, T21_c, T22_c = param_init
        di_sum = d1_c + d2_c
        estimates = np.zeros((n_iterations,4))
        #rejections = []
        if verbose==False:
            for n in range(n_iterations):
                #Define the possible next step
                d1_temp, d2_temp, T21_temp, T22_temp = transition_model(di_sum, d1_c, T21_c, T22_c)#transition_model(d1_c, d2_c, T21_c, T22_c)
        
                r_current = prior(d1_c, d2_c, T21_c, T22_c)*likelihood(TE_set, d1_c, d2_c, T21_c, T22_c)
                r_temp = prior(d1_temp, d2_temp, T21_temp, T22_temp)*likelihood(TE_set, d1_temp, d2_temp, T21_temp, T22_temp)
            
                #Compare current position and possible next position, determine if we move or not
                if acceptance(r_current, r_temp):
                    #If accepted, move to the new point. Otherwise, stay put
                    d1_c, d2_c, T21_c, T22_c = d1_temp, d2_temp, T21_temp, T22_temp
                #else:
                    #rejections.append(r_temp)
                estimates[n,:] = [d1_c, d2_c, T21_c, T22_c]
        if verbose==True:
            for n in trange(n_iterations):
                #Define the possible next step
                d1_temp, d2_temp, T21_temp, T22_temp = transition_model(di_sum, d1_c, T21_c, T22_c)#transition_model(d1_c, d2_c, T21_c, T22_c)
        
                r_current = prior(d1_c, d2_c, T21_c, T22_c)*likelihood(TE_set, d1_c, d2_c, T21_c, T22_c)
                r_temp = prior(d1_temp, d2_temp, T21_temp, T22_temp)*likelihood(TE_set, d1_temp, d2_temp, T21_temp, T22_temp)
            
                #Compare current position and possible next position, determine if we move or not
                if acceptance(r_current, r_temp):
                    #If accepted, move to the new point. Otherwise, stay put
                    d1_c, d2_c, T21_c, T22_c = d1_temp, d2_temp, T21_temp, T22_temp
                #else:
                    #rejections.append(r_temp)
                estimates[n,:] = [d1_c, d2_c, T21_c, T22_c]       
            
        return estimates    

    #Calculate the values of d1 and d2, given c1, c2, T11, and T12
    d1 = c1*(1-2*np.exp(-TI_np/T11))
    d2 = c2*(1-2*np.exp(-TI_np/T12))
    di_sum = d1+d2
    #print('d1, d2, di sum, and TI:', d1, d2, di_sum, TI_np)
    #Set random seed for reproducibility
    #np.random.seed(0)
    #Generate the data over TE with the given parameters
    data_4P = S_4_param(TE_set, d1, d2, T21, T22) + noise_TE(noise_sd)
    '''
    #Visualize the data
    plt.plot(TE_set, data_4P, label='data')
    plt.plot(TE_set, S_4_param(TE_set, d1, d2, T21, T22), label='Underlying curve')
    plt.title('Example data set, TI = %s' % (TI_np))
    plt.legend()
    plt.show()
    '''
    #Get the joint posterior as a function of the parameters
    #Metropolis-Hastings will sample this function
    joint_pdf_4P = pdf_np_4_param(data_4P, noise_sd)
    #Define step sizes for Metropolis for each of the parameters
    #Will need to adjust based on SNR, parameters chosen, etc, but these general values seem to work well
    step_d1 = 2/SNR
    step_T21 = 2*t_norm/SNR
    step_T22 = 2*t_norm/SNR
    #Define the box in which Metropolis will run
    #Needed for the Bayesian prior
    upper_bound_4P = np.array([1, 1, r, r])
    lower_bound_4P = np.array([-1, -1, 0, 0])

    #Define initial starting point
    d1_init = d1
    d2_init = di_sum - d1_init
    if flip:
        param_init_4P = (d2_init, d1_init, T22, T21)
    else:
        param_init_4P = (d1_init, d2_init, T21, T22)
    #Save metropolis steps as estimates_4P, an (n_iterations x 4) matrix
    #To run Metropolis, enter the joint PDF you would like to sample, 
    #the prior, transition model, initial parameters, number of steps, noisy data, and acceptance step
    #Verbose=True will give a progress bar, verbose=False will not give the progress bar
    estimates_4P = metropolis_4P(joint_pdf_4P,prior_4P,transition_model_d_constrained, 
                                param_init_4P, n_iterations, data_4P, acceptance_4P, verbose=verbose)

    #np.save("metrodata.npy", estimates_4P)
    #print("Data generated")
    return estimates_4P[int(0.1*n_iterations):]
