# 
# Quantum Potential Well (ATOMIC UNITS) with Finite differences method
# Created by Jordi Faraudo, modificated by Hugo Soto 
#

#Libraries importation
import numpy as np
import matplotlib.pyplot as plt

#Potential as a function of position
D = 2.0
B = 1.0
X0 = 1.0
V0 = 3.0
a = -1/B*np.log(1+(V0/D)**0.5)+X0
listV=[]
def getV(x):
    if (abs(x)>a):
        potvalue = D*(1-np.exp(-B*(abs(x)-X0)))**2
        listV.append(potvalue)
    else:
        potvalue = V0
        listV.append(potvalue)
    return potvalue


#Discretized Schrodinger equation in n points (FROM 0 to n-1)
def Eq(n,h,x):
    F = np.zeros([n,n])
    for i in range(0,n):
        F[i,i] = -2*((h**2)*getV(x[i]) + 1)
        if i > 0:
           F[i,i-1] = 1
           if i < n-1:
              F[i,i+1] = 1
    return F

#-------------------------
# Main program
#-------------------------
# Interval for calculating the wave function [-L/2,L/2]
L = 20
xlower = -L/2.0
xupper = +L/2.0

#Discretization options
h = 0.01  #discretization in space

#Create coordinates at which the solution will be calculated
x = np.arange(xlower,xupper+h,h)
#grid size (how many discrete points to use in the range [-L/2,L/2])
npoints=len(x)

print("Using",npoints, "grid points.")

#Calculation of discrete form of Schrodinger Equation
print("Calculating matrix...")
F=Eq(npoints,h,x)

#diagonalize the matrix F
print("Diagonalizing...")
eigenValues, eigenVectors = np.linalg.eig(F)

#Order results by eigenvalue
# w ordered eigenvalues and vs ordered eigenvectors
idx = eigenValues.argsort()[::-1]
w = eigenValues[idx]
vs = eigenVectors[:,idx]

#Energy Level
E = - w/(2.0*h**2)

#Print Energy Values
print("RESULTS:")
for k in range(0,4):
  Erel=E[k]/4.0  #ratio between energy level and potential well
  print("State ",k,": Energy = %.4f" %E[k],', E/V='+'{:.4f}'.format(Erel))

#Init Wavefunction (empty list with npoints elements)
psi = [None]*npoints

#Calculation of normalised Wave Functions
for k in range(0,len(w)):
    psi[k] = vs[:,k]
    integral = h*np.dot(psi[k],psi[k])
    psi[k] = psi[k]/integral**0.5

#Plot Wave functions
print("Plotting")

listE = []
llE = []
for e in range(0,4):
    listE = [round(E[e],2)]*len(x)
    llE.append(listE)
    arrayE = np.array(llE)

#v = int(input("\n Quantum Number (enter 0 for ground state):\n>"))
for v in range(0,4):
    plt.plot(x,listV, color='black')
    plt.plot(x,E[v]+psi[v],label=r'$\psi_v(x)$, v = ' + str(v), color='orange')
    plt.plot(x,arrayE[v],label=r'$E_n$ = ' + '{:.4f}'.format(E[v]),color='orange')
    plt.title("State " + str(v) + " for a double morse potential")
    plt.legend()
    plt.xlabel(r'$x$ (a.u.)')
    plt.ylabel(r'$\psi(x)$')
    plt.show()

print("Bye")
