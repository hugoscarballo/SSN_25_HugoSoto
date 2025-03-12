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
maximum_iterations = int(input("\n Enter maximum number of Monte Carlo iterations (suggeestion 1000):"))
energy_convergence = float(input("\n Enter energy convergence for the system (suggestion 0.001, lower could be possible but very difficult to achieve and higher to easy):"))
number_repetition = int(input("\n Minimum number of times convergence has to be reach consecutively to consider simulation converges (suggestion 5 if there isn't much noise, less if there is):"))

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
energy_difference = np.zeros(number_repetition)

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
                print(f"MC step= {t} Energy= {Ene:.{decimal}f} M= {Mag}")
                configPlot(f, config, t, N)
            
            #evaluate energy convergence in certain consecutive MC steps
            energy_diff = abs(E[i]-E[i-1])
            if t<number_repetition:
                print(f"MC step= {t} Energy= {Ene:.{decimal}f} M= {Mag}")
                print(f" Absolute energy difference= {energy_diff:.{decimal}f} Energy convergence= {energy_convergence}")
                continue
            else:
                for k in range(number_repetition):
                    energy_difference[k] = abs(round(E[i-k]-E[i-k-1],decimal))       
                if all(j<=energy_convergence for j in energy_difference):
                    #evaluate if converges
                    print(f"MC step= {t} Energy= {Ene:.{decimal}f} M= {Mag}")
                    print(f" Absolute energy difference= {energy_diff:.{decimal}f} Energy convergence= {energy_convergence}")
                    break
                else:
                    #evaluate if converges
                    print(f"MC step= {t} Energy= {Ene:.{decimal}f} M= {Mag}")
                    print(f" Absolute energy difference= {energy_diff:.{decimal}f} Energy convergence= {energy_convergence}")
                    continue
            
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
