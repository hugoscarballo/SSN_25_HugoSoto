
#
# Homework 1: Transmission coefficient in a barrier system 
# Created by Hugo Soto Carballo
#

# Libraries importation
import numpy as np
import matplotlib.pyplot as plt

# Function. Draw the transmission coefficient in a barrier system
# Input: effective mass constant (adimensional), barrier length (nm), barrier potential (eV) and number of points  

def barrier_transmission (alfa,L,V,N):
    q=1.602e-19                              # Càrrega de l'electró, en C
    hb=1.055e-34                             # Constant de Planck, en Js
    m=9.11e-31                               # Masa de l'electró, en kg
    X=[]                                     # Creació llista de posició energètica, en E/V
    TC=[]                                    # Creació llista de coeficient de transmissió
    NP=V/N                                   # Definir la distància entre dos punts, en energia
    E=np.arange(V+NP,2*V+NP,NP)              # Definir l'interval d'energies
    k=np.sqrt(2*alfa*m*q)/hb                 # Nombre d'ona a la barrera (sense tenir en compte l'energia, en eV)

    for i in E:
        K=k*np.sqrt(i-V)                     # Nombre d'ona a la barrera
        S=np.sin(K*L*1e-9)                   # Terme sinusoïdal de la transmissió 
        n=4*i*(i-V)                          # Valor del numerador
        T=n/(n+(V*S)**2)                     # Coeficient de transmissió
        TC.append(T)                         # Afegir els valors de transmissió
        X.append(i/V)                        # Afegir el terme d'energia corresponent

    plt.plot(X, TC, color='orange')          # Dibuixar la gràfica
    plt.title('Barrier system')              # Posar títol a la gràfica
    plt.xlabel('E/V')                        # Posar títol a l'eix X
    plt.ylabel('Transmission coefficient')   # Posar títol a l'eix Y
    plt.show()



