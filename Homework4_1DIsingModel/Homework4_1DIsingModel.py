# -------------------------------------------------------------------------------------------------------#
# INTRODUCTION TO MONTECARLO SIMULATION
# March 6, 20253 
# Simulation of the 2D Ising model, created by Jordi Faraudo
# Modification for the 1D Ising model, made by Hugo Soto 
# The algorithm is based on Dr Rajesh Singh (Cambridge University) blog with python resources in Physics. 
# -------------------------------------------------------------------------------------------------------#

import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt

#--------------------------------------------------------------------------------------------------------#
##  BLOCK OF FUNCTIONS USED IN THE MAIN CODE
#--------------------------------------------------------------------------------------------------------#
 
#Generation of a random initial state for N spins
def initialstate(N):  
    '''Generates a random spin configuration for initial condition'''
    state = 2*np.random.randint(2, size=(N))-1
    return state
 
#Here we define the interactions of the model (2D spin Ising model) and the solution method (Metropolis Monte Carlo)
def mcmove(config, beta):
    '''Monte Carlo move using Metropolis algorithm '''
    for i in range(N):
 
            #select random spin from N system  
            a = np.random.randint(0, N)
            s =  config[a]
            #calculate energy cost of flipping this spin (the % is for calculation of periodic boundary condition)
            nb = config[(a+1)%N] + config[(a-1)%N]
            cost = 2*s*nb
            #flip spin or not depending on the cost and its Boltzmann factor, acceptance probability is given by Boltzmann factor with beta = 1/kBT
            if cost < 0:
                s *= -1
            elif rand() < np.exp(-cost*beta):
                s *= -1
            config[a] = s
    return config
 
#This function calculates the energy of a given configuration for the plots of Energy as a function of T
def calcEnergy(config):
    '''Energy of a given configuration'''
    energy = 0
    for i in range(len(config)):
            S = config[i]
            nb = config[(i+1)%N] + config[(i-1)%N]
            energy += -nb*S
    return energy/2.
 
#This function calculates the magnetization of a given configuration
def calcMag(config):
    '''Magnetization of a given configuration'''
    mag = np.sum(config)
    return mag
 
#This function makes a plot of all data
def resultPlot(T,Energy,Magnetization,SpecificHeat,Susceptibility):
 #plot everything
    plt.clf()
    plt.subplot(2, 2, 1 );
    plt.plot(T, Energy, 'o', color="#A60628");
    plt.xlabel("Temperature (T)", fontsize=20);
    plt.ylabel("Energy ", fontsize=20);
 
    plt.subplot(2, 2, 2 );
    plt.plot(T, abs(Magnetization), 'o', color="#348ABD");
    plt.xlabel("Temperature (T)", fontsize=20);
    plt.ylabel("Magnetization ", fontsize=20);
 
    plt.subplot(2, 2, 3 );
    plt.plot(T, SpecificHeat, 'o', color="#A60628");
    plt.xlabel("Temperature (T)", fontsize=20);
    plt.ylabel("Specific Heat ", fontsize=20);
 
    plt.subplot(2, 2, 4 );
    plt.plot(T, Susceptibility, 'o', color="#348ABD");
    plt.xlabel("Temperature (T)", fontsize=20);
    plt.ylabel("Susceptibility", fontsize=20);
       
    plt.pause(0.1)
 
#----------------------------------------------------------------------
##  MAIN PROGRAM
#----------------------------------------------------------------------    
 
#Initial parameters for calculation
##change the parameters below to change system size and statistics
nt      = 100               #number of temperature points (recommended 2**8)
N       = 2**4              #size of the lattice,
eqSteps = 2**10             #number of MC sweeps for equilibration
mcSteps = 2**10             #number of MC sweeps for calculation
 
#Calculate normalization constants for future averages
n1  = 1.0/(mcSteps*N)
n2  = 1.0/(mcSteps*mcSteps*N)
 
#Generate a random distribution of temperatures centered around the most interesting one (tm) to make an exploration
tm = 2.269    
T=np.random.normal(tm, .64, nt)
 
#Keep only those in a reasonable interval
T  = T[(T>1.0) & (T<4.0)]
T.sort()
nt = np.size(T)
 
#Init calculation of physical quantities
Energy       = np.zeros(nt)
Magnetization  = np.zeros(nt)
SpecificHeat = np.zeros(nt)
Susceptibility = np.zeros(nt)
 
#----------------------------------------------------------------------
#  SIMULATION LOOP
#----------------------------------------------------------------------
print('Starting Simulations at ',len(T),' different temperatures.')
 
#Init interactive plot
plt.ion()
plt.figure(figsize=(18, 10)); # create figure to plot the calculated values    
 
for m in range(len(T)):
    E1 = M1 = E2 = M2 = 0
    config = initialstate(N)
    iT=1.0/T[m]
    iT2=iT*iT
    print('Running Simulation ',m+1,' of',len(T),' at reduced temperature T=',T[m])
   
    for i in range(eqSteps):         #equilibrate
        mcmove(config, iT)           #Monte Carlo moves
 
    for i in range(mcSteps):
        mcmove(config, iT)          
        Ene = calcEnergy(config)     #calculate the energy
        Mag = calcMag(config)        #calculate the magnetisation
 
        E1 = E1 + Ene
        M1 = M1 + Mag
        M2 = M2 + Mag*Mag
        E2 = E2 + Ene*Ene

        Energy[m]         = n1*E1
        Magnetization[m]  = n1*M1
        SpecificHeat[m]   = (n1*E2 - n2*E1*E1)*iT2
        Susceptibility[m] = (n1*M2 - n2*M1*M1)*iT
 
    #Plot final data for this T
    resultPlot(T,Energy,Magnetization,SpecificHeat,Susceptibility)

    #Plot spin configuration
    RD = []
    BD = []
    for k in config:
        if k==+1:
            RD.append(1)
            BD.append(0)
        else:
            RD.append(0)
            BD.append(1)

    X = np.linspace(0,N,N)
    plt.bar(X,RD, color='red', label='spin up')
    plt.bar(X,BD, color='blue', label='spin down')
    plt.yticks([0, 1])
    plt.ylim(0,1)
    plt.xlim(0,N)
    plt.legend()
    #plt.show()
       
#End interactive plot: final plot of everything
plt.ioff()
print("Finished. Plotting all results")
resultPlot(T,Energy,Magnetization,SpecificHeat,Susceptibility)
plt.show()
