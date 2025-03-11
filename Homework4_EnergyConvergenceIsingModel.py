# -----------------------------------------------------------------
# INTRODUCTION TO MONTECARLO SIMULATION
# March 6, 2025 
# Simulation of a sequence of configurations for the 2D Ising model, created by Jordi Faraudo 
# Energy convergence implementation by Hugo Soto
# The algorithm is based on Dr Rajesh Singh (Cambridge University) blog with python resources in Physics. 
# -----------------------------------------------------------------

import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt


#Function with the interactions of the model (2D spin Ising model) and the solution method (Metropolis Monte Carlo)
def mcmove(config, N, beta):
        #loop with a size equal to spins in the system
        for i in range(N):
            for j in range(N):
                    #pick a random spin: generate integer random number between 0 and N
                    a = np.random.randint(0, N)
                    b = np.random.randint(0, N)
                    #state of spin
                    s =  config[a, b]
                    #calculate energy cost of flipping that spin (the % is for calculation of periodic boundary condition)
                    nb = config[(a+1)%N,b] + config[a,(b+1)%N] + config[(a-1)%N,b] + config[a,(b-1)%N]
                    cost = 2*s*nb
                    #flip spin or not depending on the cost and its Boltzmann factor
                    #acceptance probability is given by Boltzmann factor with beta = 1/kBT
                    if cost < 0:
                        s = s*(-1)
                    elif rand() < np.exp(-cost*beta):
                        s = s*(-1)
                    config[a, b] = s
        #return the new configuration
        return config

#This function makes an image of the spin configurations
def configPlot(f, config, i, N):
        ''' This modules plts the configuration '''
        X, Y = np.meshgrid(range(N), range(N))
        plt.pcolormesh(X, Y, config, vmin=-1.0, vmax=1.0, cmap='RdBu_r');
        plt.title('MC iteration=%d'%i);
        plt.axis('tight')
        plt.pause(0.1)

#This function calculates the energy of a given configuration for the plots of Energy as a function of T
def calcEnergy(config):
    '''Energy of a given configuration'''
    energy = 0
    for i in range(len(config)):
        for j in range(len(config)):
            S = config[i,j]
            nb = config[(i+1)%N, j] + config[i,(j+1)%N] + config[(i-1)%N, j] + config[i,(j-1)%N]
            energy += -nb*S
    return energy/4.

#This function calculates the magnetization of a given configuration
def calcMag(config):
    '''Magnetization of a given configuration'''
    mag = np.sum(config)
    return mag

#----------------------------------------------------------------------
##  MAIN PROGRAM
#----------------------------------------------------------------------  

#Set initial conditions and control the flow of the simulation size of the lattice
N = 64
#Enter data for the simulation
temp = float(input("\n Enter temperature in reduced units (suggestion 1.2): "))
maximum_iterations = int(input("\n Enter maximum number of Monte Carlo iterations (suggeestion 50):"))
energy_convergence = float(input("\n Enter energy convergence for the system (suggestion 0.001):"))

#Consider number of decimals in convergence
energy_str = f"{energy_convergence:.10f}".rstrip('0')
energy_str = str(energy_str)
print(energy_str.split('.'))
decimal = len(energy_str.split('.')[1])

#Init Magnetization and Energy
step=[]
M=[]
E=[]

#Generate initial condition
config = 2*np.random.randint(2, size=(N,N))-1

#Calculate initial value of magnetization and Energy
Ene = calcEnergy(config)/(N*N)     #calculate average energy
Mag = calcMag(config)/(N*N)        #calculate average magnetisation
t=0
print('MC step=',t,' Energy=',Ene,' M=',Mag)

#Update 
step.append(t)
E.append(Ene)
M.append(Mag)

#Show initial condition
print('Initial configuration:')
print(config)
#f = plt.figure(figsize=(15, 15), dpi=80);
f = plt.figure(dpi=100)
configPlot(f, config, 0, N)
plt.show()

#Turn on interactive mode for plots
print("Starting MC simulation")
plt.ion()

#Perform the MC iterations

for i in range(maximum_iterations):
            #call MC calculation
            mcmove(config, N, 1.0/temp)
            #update variables
            t = t+1                            #update MC step
            Ene = calcEnergy(config)/(N*N)     #calculate average energy
            Mag = calcMag(config)/(N*N)        #calculate average magnetisation
            #update 
            step.append(t)
            E.append(Ene)
            M.append(Mag)

            #plot only certain configurations
            if t%10 == 0:
                print('\nMC step=',t,' Energy=',Ene,' M=',Mag)
                print(config)
                configPlot(f, config, t, N)
            
            energy_difference = abs(round(E[i]-E[i-1],decimal))  #calculate energy difference          
            if energy_difference>energy_convergence:
                #evaluate if converges
                print(f"MC step= {t} Absolute energy difference= {energy_difference:.{decimal}f} Energy convergence= {energy_convergence}")
                continue
            else:
                #evaluate if converges
                print(f"MC step= {t} Absolute energy difference= {energy_difference:.{decimal}f} Energy convergence= {energy_convergence}")
                break   
            
#Print end
if t==maximum_iterations:
    print('\nSimulation has not reached convergence with',t, 'MC steps')
else:
    print('\nSimulation has reached convergence after',t, 'MC steps')

#Interactive plotting off
plt.ioff()

#Show final configuration
configPlot(f, config, t, N)
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