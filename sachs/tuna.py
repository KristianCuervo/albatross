#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 15:28:02 2025

@author: uhth
"""

# Numerical solution of the Bigeye Tuna optimal control problem

import casadi
import numpy as np
import matplotlib.pyplot as plt


opti = casadi.Opti()

N = 128

Z = opti.variable(N)
Theta = opti.variable(N)
Phi = opti.variable(N)

dt = opti.variable(1)

## Model specification 
V = lambda theta: Um*Lf*(1-casadi.exp(l*(Td-theta)))
kappa = lambda dT: 0.5*( (kh+kl)*dT + (kh-kl)*casadi.sqrt(dT**2+tau2))
rho = lambda z: r0 + (rm-r0)*casadi.exp(-0.5*(z-zS)**2/s2)
Ta = lambda z: T0 - (p*z + dTclin)* z**lambdaClin / (zClin**lambdaClin + z**lambdaClin)

## Parameters. Unit: Meters, deg C, minute
T0 = 28           # deg
zClin = 160       # m
dTclin = 8        # deg C
p = 0.02          # deg C/ m
lambdaClin = 5    # (dimless)
zS = 450          # m
s2 = 25**2        # m**2
rm = 1            # J/m
r0 = 0.1          # J/m
Lf = 0.91          # m
kl = 0.015         # 1/min
kh = 0.1          # 1/min
tau2 = 0.2**2     # deg C**2
Um = 2*60         # m/min
Td = 11           # deg C
l = 0.75          # 1/deg Cax2.xaxis.set_inverted(True)

L = casadi.cos(Phi)*V(Theta)*rho(Z)*dt
opti.minimize(-casadi.sum1(L)/dt/N) # divide with casadi.sum1(dt) when dt changes)

## Leapfrog 

dZ = casadi.diff(casadi.vertcat(Z[-1],Z))
dTheta = casadi.diff(casadi.vertcat(Theta,Theta[0]))

opti.subject_to(dZ == casadi.sin(Phi)*V(Theta)*dt)
opti.subject_to(dTheta == kappa(Ta(Z)-Theta)*dt)

opti.subject_to(Z[0] == 300)
opti.subject_to(dTheta[0] > 0)
opti.subject_to(dt>0.1)

# opti.subject_to(casadi.sum1(dt)>10)
# opti.subject_to(casadi.sum1(dt)<200)
opti.subject_to(Z>0)
opti.subject_to(Theta>Td)
opti.subject_to(-np.pi/2 < Phi )
opti.subject_to(Phi < np.pi/2)

## Equidistance 
# dl2 = dt**2  + 0.01*dZ**2 + dTheta**2
# opti.subject_to(casadi.diff(dl2)==0)

T = 20
tv = np.linspace(0,T,N)
opti.set_initial(dt,0.5) # np.ones(N)*T/N)
opti.set_initial(Z,250-100*np.cos(2*np.pi*tv/T))
opti.set_initial(Theta,20+5*np.sin(2*np.pi*tv/T))
opti.set_initial(Phi,-np.pi/2*np.sin(2*np.pi*tv/T))

opti.solver('ipopt')

#%%

sol = opti.solve()

#%%

fig, (ax1,ax2) = plt.subplots(1,2)
zv = np.linspace(0,600,101)
ax1.plot(Ta(zv),zv)
ax1.yaxis.set_inverted(True)
ax2.plot(rho(zv),zv)
ax2.yaxis.set_inverted(True)

#%%
fig, (ax1,ax2) = plt.subplots(2,1)
# tv = np.cumsum(opti.value(dt)) / 60
tv = np.arange(N)*opti.value(dt)

#ax1.plot(tv,opti.value(Z,opti.initial()))
#ax2.plot(tv,opti.value(Theta,opti.initial()))

ax1.plot(np.concatenate((tv,tv+tv[-1])),np.concatenate((opti.value(Z),opti.value(Z))))

ax2.plot(np.concatenate((tv,tv+tv[-1])),np.concatenate((opti.value(Theta),opti.value(Theta))))

ax1.yaxis.set_inverted(True)
ax2.set_xlabel('Time [min]')
ax2.set_ylabel('Body temperature [C]')
ax1.set_ylabel('Depth [m]')

plt.savefig('tuna.pdf',bbox_inches='tight')
plt.close(plt.gcf())

