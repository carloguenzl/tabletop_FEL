# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 17:15:37 2024

@author: carlo
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
savepath = "C:/Users/carlo/Desktop/Dateidatei/6. SEMESTER/F-Praktikum/FEL/pics"


def get_dipole(B_rem,l,d):
    return B_rem * np.pi*(d/2)**2/const.mu_0*l #Joule/Tesla

m1 = get_dipole(1.3, 20E-3, 10E-3)

nx,ny = np.linspace(-1,1,1000),np.linspace(-.5,.5,1000)


class Magnet:
    def __init__(self,place,m):
        self.place = np.array(place)
        self.m = np.array(m)
    
    def B(self,r:np.ndarray):
        r = np.array(r-self.place)
        return const.mu_0/(4*np.pi) * ((3*np.dot(self.m,r)*r-self.m*np.linalg.norm(r)**2))/(np.linalg.norm(r)**5)

Mag = Magnet((0,0,0),(0,1,0))

x,y = np.meshgrid(nx,ny)

mx,my = np.array([Mag.B((xx,yy,0))[0] for xx,yy in zip(x.flatten(),y.flatten())]).reshape(1000,1000),np.array([Mag.B((xx,yy,0))[1] for xx,yy in zip(x.flatten(),y.flatten())]).reshape(1000,1000)
#%%
fig, ax = plt.subplots()
plt.streamplot(x,y,mx,my,color=(180/255,0,0),density=.8,broken_streamlines=True,arrowsize=1.4)
plt.plot((0,0),(0,.2),color=(204/255,0,0),linewidth=15)
plt.plot((0,0),(-0.03,-.2),color=(34/255,139/255,34/255),linewidth=15)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.tick_params(axis='both', which='both', length=0)  # Hides the ticks
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
#plt.savefig(savepath+"/magnetfield.png",dpi=300)
