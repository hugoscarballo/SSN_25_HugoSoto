# -----------------------------------------------------------------
# INTRODUCTION TO MONTECARLO SIMULATION
# March 6, 2025 
# Simulation of a sequence of configurations for the 2D Ising model, created by Jordi Faraudo
# Modification for the 1D Ising model, made by Hugo Soto
# The algorithm is based on Dr Rajesh Singh (Cambridge University) blog with python resources in Physics. 
# -----------------------------------------------------------------

import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt


#Function with the interactions of the model (1D spin Ising model) and the solution method (Metropolis Monte Carlo)
def mcmove(config, N, beta):
    #loop with a size equal to spins in the system
    for i in range(N):
        #pick a random spin: generate integer random number between 0 and N
        a = np.random.randint(0, N)
        #state of spin
        s =  config[a]
        #calculate energy cost of flipping that spin (the % is for calculation of periodic boundary condition)
        nb = config[(a+1)%N] + config[(a-1)%N]
        cost = 2*s*nb
        #flip spin or not depending on the cost and its Boltzmann factor
        #acceptance probability is given by Boltzmann factor with beta = 1/kBT
        if cost < 0:
            s = s*(-1)
        elif rand() < np.exp(-cost*beta):
            s = s*(-1)
            config[a] = s
    #return the new configuration
    return config

#This function makes an image of the spin configurations
def configPlot(config, i, N):
        ''' This modules plts the configuration '''
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
        plt.title('MC iteration=%d'%i)
        plt.legend()
        plt.show()

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

#----------------------------------------------------------------------
##  MAIN PROGRAM
#----------------------------------------------------------------------  

#Set initial conditions and control the flow of the simulation size of the lattice
N = 16
#Enter data for the simulation
temp = float(input("\n Please enter temperature in reduced units (suggestion 1.2): "))
msrmnt = int(input("\n Enter number of Monte Carlo iterations (suggestion 1000):"))

#Init Magnetization and Energy
step=[]
M=[]
E=[]

#Generate initial condition
config = 2*np.random.randint(2, size=(N))-1

#Calculate initial value of magnetization and Energy
Ene = calcEnergy(config)/(N)     #calculate average energy
Mag = calcMag(config)/(N)        #calculate average magnetisation
t=0
print('MC step=',t,' Energy=',Ene,' M=',Mag)

#Update 
step.append(t)
E.append(Ene)
M.append(Mag)

#Show initial condition
print('Initial configuration:')
#f = plt.figure(figsize=(15, 15), dpi=80);
f = plt.figure(dpi=100)
configPlot(config, 0, N)
plt.show()

#Turn on interactive mode for plots
print("Starting MC simulation")
plt.ion()

#Perform the MC iterations
for i in range(msrmnt):
            #call MC calculation
            mcmove(config, N, 1.0/temp)
            #update variables
            t=t+1                              #update MC step
            Ene = calcEnergy(config)/(N)     #calculate average energy
            Mag = calcMag(config)/(N)        #calculate average magnetisation
            #update 
            step.append(t)
            E.append(Ene)
            M.append(Mag)

            #plot only certain configurations
            if t%10 == 0:
                print('\nMC step=',t,' Energy=',Ene,' M=',Mag)
                configPlot(config, t, N)

#Print end
print('\nSimulation finished after',t, 'MC steps')

#Interactive plotting off
plt.ioff()

#Show final configuration
configPlot(config, t, N)
plt.show()

#Plot evolution of Energy and Magnetization during the simulation
plt.subplot(2, 1, 1)
plt.plot(step, E, 'r+-')
plt.ylabel('Energy')

plt.subplot(2, 1, 2)
plt.plot(step, M, 'b+-')
plt.ylabel('Magnetization')
plt.xlabel('MC step')

#Show the plot in screen
plt.show()
