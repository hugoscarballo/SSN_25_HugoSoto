#
# Homework 2: Numerical methods
# Created by Hugo Soto Carballo
#

import numpy as np
import matplotlib.pyplot as plt

#Input 
x0=float(input("\n Initial position x0 (in nm):"))                  #Initial position input (nm)
v0=float(input("\n Initial velocity v0 (in nm/s):"))                #Initial velocity input (nm/s)
m=float(input("\n System mass m (in ng):"))                         #System mass input (ng)
T=float(input("\n Period T (in ns):"))                              #Period input (ns)
dt=float(input("\n Time step dt (in ns):"))                         #Time step input (ns)
ntot=int(input("\n Number of time steps:"))                         #Total points input
w=2.0*np.pi/T                                                       #Angular velocity (rad/ns)
k=m*(w)**2                                                          #Elastic consant (N/nm)
print("Simulation time will be",dt*ntot," ns")                      #Print simulation time (ns)
 
#Initial conditions
t=np.zeros(ntot+1)                                                  #Time matrix
x=np.zeros(ntot+1)                                                  #Position matrix
v=np.zeros(ntot+1)                                                  #Velocity matrix
v_hs=np.zeros(ntot+1)                                               #Half-step velocity matrix
t[0]=0.0                                                            #Initial time
x[0]=x0                                                             #Initial position
v[0]=v0                                                             #Initial velocity
a=-(k/m)*x[0]                                                       #Initial acceleration
E0=(m/2.0)*(v0)**2+(k/2)*(x)**2                                     #Initial energy
v_hs[0]=v0+(dt/2.0)*a                                               #New half-step velocity after time dt/2
print(f"Step 0, t={t[0]}, x={x[0]}, v={v[0]}, a={a}")               #Print calculated data
print(f"                      v={v_hs[0]:.3}"      )

#Calculations
i=0
while i<ntot:
    x[i+1]=x[i]+dt*v_hs[i]                                          #New position after time t+dt
    t[i+1]=t[i]+dt                                                  #New time after t+dt
    a=-(k/m)*x[i+1]                                                 #New acceleration after time t+dt
    v_hs[1+i]=v_hs[i]+a*dt                                          #New velocity after time t+dt/2
    v[i+1]=(v_hs[i+1]+v_hs[i])/2                                    #New velocity after time t+dt                                         
    i=i+1                                                           #Update counter
    print(f"Step {i}, t={t[i]:.3}, x={x[i]:.3},         a={a:.3}")  #Print calculated data
    print(f"                       v={v_hs[i]:.3}")

#Graphics
 #Plot 1: position and velocity
xa=np.cos(w*t)                                                      #Position analytical solution
va=-w*np.sin(w*t)                                                   #Velocity analytical solution

plt.figure(1)

plt.subplot(211)
plt.plot(t,x, 'ro',t,xa, '-')                                       #Plot numerical and analytical position solutions
plt.ylabel('position x (nm)')                                       #Label y axis
plt.xlabel('time (ns)')                                             #Label x axis
plt.legend(['Numerical', 'Analytical'])                             #Legend

plt.subplot(212)
plt.plot(t,v, 'ro',t,va, '-')                                       #Plot numerical and analytical velocity solutions
plt.ylabel('velocity (nm/s)')                                       #Label y axis
plt.xlabel('time (ns)')                                             #Label x axis
plt.legend(['Numerical', 'Analytical'])                             #Legend

plt.show()                                                          #Show

 #Plot 2: phase space
plt.plot(x,v,'k')                                                   #Plot phase space
plt.ylabel('v (nm/s)')                                              #Label y axis
plt.xlabel('x (nm)')                                                #Label x axis
plt.show()                                                          #Show

 #Plot 3: energy
E=(m/2.0)*(v)**2+(k/2)*(x)**2                                       #Energy 
RE=E/E0[0]                                                          #Calculate relative energy
plt.plot(t,RE,'k')                                                  #Plot relative energy
plt.ylabel('E/E0')                                                  #Label y axis
plt.xlabel('time (ns)')                                             #Label x axis
plt.show()