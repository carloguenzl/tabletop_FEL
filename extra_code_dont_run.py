# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 12:37:19 2024

@author: carlo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import scipy.constants as const

#%%
def get_dipole(B_rem,l,d):
    return B_rem * np.pi*(d/2)**2/const.mu_0*l #Joule/Tesla

m1 = get_dipole(1.3, 20E-3, 10E-3)

#%% get Trajectory

def run_el(steps=1e3):
    steps=1E3
    i=0
    while el.place[0]**2+el.place[1]**2 <= 1**2 and i <= steps:
        el.rungestep()
        memory.append(el.place)
        vel_memory.append(el.vel)
        acc_memory.append(el.Lorenz_acc(el.place,el.vel))
        trajectory.append((el.place,el.vel))
        if i%(steps/10)==0:
            print(int(i/steps*100),"%")
        i+=1
    
    if el.place[0]**2+el.place[1]**2 > .07**2:
        print("hit border")

    global retardierung        
    retardierung = []
    for place in memory:
        retardierung.append(np.linalg.norm(np.array(detector_position)-place)/const.c)
    retardierung = np.array(retardierung)
    
    global timescale
    timescale = np.arange(0,(len(memory))*el.Dt,el.Dt)
    
    
    fig=plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    plt.xlim(-.05,.05)
    plt.ylim(-.05,.05)
    
    #mag.show_dipole(ax)
    cor_r.show_dipole(ax)
    cor_l.show_dipole(ax)
    #cor_h.show_dipole(ax)
    #cor_v.show_dipole(ax)
    
    ax.scatter(np.array(memory).T[0],np.array(memory).T[1],np.array(memory).T[2],color=(204/255,0,0),label="Trajektorie")
    
    ax.set_xlabel("x (m)",fontsize=20)
    ax.set_ylabel("y (m)",fontsize=20)
    ax.set_zlabel("z (m)",fontsize=20)
    ax.legend(fontsize=20)
    ax.view_init(20, 40)
    plt.tight_layout()
    
    #plt.savefig(savepath+"/trajectory_first.png",dpi=200)
    plt.show()
    plt.close()

run_el()

#%% Retry field. again.
R = np.array([(np.array(detector_position)-place) for place in memory])


def E_rad(r,t,ret):
    index = int(np.round((t - np.linalg.norm(ret)/const.c)/el.Dt))
    if index < 0:
        return np.zeros(3),index
        
    acc = acc_memory[index]
    acc_perp = np.array([0,acc[1],0]) #acc-np.dot(acc,R_unit)
    
    return const.e/(4*np.pi*const.epsilon_0*const.c**2)*1/np.linalg.norm(R[index])*acc_perp,index
# acc_memory = acc_memory

Ess = []
i=0
for t in (timescale):
    E,index = E_rad(detector_position,t,R[i])
    Ess.append(E)
    if index>=0:
        i+=1   

Ess = np.array(Ess)

fourier = np.abs(fft(Ess.T[1][:]))
freqs = fftfreq(len(fourier),timescale[-1]/len(timescale))



#%%
plt.plot(timescale[:]*1e9,Ess.T[1][:]*1e9,c=(204/255,0,0))
plt.xlabel("$t$ (ns)",fontsize=14)
plt.ylabel(r"$E_y$ (nV m$^{-1}$)",fontsize=14)
plt.title("Abgestrahltes Feld der Trajektorie bei $r=(0,0,1)$",fontsize=14)
plt.tight_layout()
plt.grid()
# plt.savefig(savepath+"/rad_field_wod_546keV.png",dpi=200)

#%%
plt.plot(freqs[:50]*1e-9,fourier[:50]*1e9,c=(204/255,0,0))
plt.xlabel(r"$f$ (GHz)",fontsize=14)
plt.ylabel(r"$FFT(E_y)$ (nV m$^{-1}$)",fontsize=14)
plt.title("FFT des Abgestrahlten Feldes bei $r=(0,0,1)$",fontsize=14)
plt.grid()
# plt.savefig(savepath+"/fourier_wod_546keV.png",dpi=200)

print("f_max = ",freqs[np.where(fourier==np.max(fourier))[0][0]]*1E-6,"MHz")
#%%Freq(Energy)
frequencies = []
energies = np.linspace(100,500,10)


def run_electron(elec,steps=5E3):
    trajectory = [(el.place,el.vel)]

    memory = [elec.place]
    vel_memory = [elec.vel]
    acc_memory = [elec.Lorenz_acc(el.place)]
    
    i=0
    while el.place[0]**2+el.place[1]**2 <= 1**2 and i <= steps:
        elec.rungestep()
        memory.append(elec.place)
        vel_memory.append(elec.vel)
        acc_memory.append(elec.Lorenz_acc(elec.place))
        trajectory.append((elec.place,elec.vel))
        if i%(steps/10)==0:
            print(int(i/steps*100),"%")
        i+=1

    if el.place[0]**2+el.place[1]**2 > .07**2:
        print("hit border")
        
    retardierung = []
    for place in memory:
        retardierung.append(np.linalg.norm(np.array(detector_position)-place)/const.c)
    retardierung = np.array(retardierung)

    timescale = np.arange(0,(len(memory))*el.Dt,el.Dt)
    return memory,vel_memory,acc_memory,timescale

for energy in energies:
    el = electron((0,0,0),energy,(.8,np.pi/2))
    memory,vel_memory,acc_memory,timescale = run_electron(el)
    
    R = np.array([(np.array(detector_position)-place) for place in memory])


    def E_rad(r,t,ret):
        index = int(np.round((t - np.linalg.norm(ret)/const.c)/el.Dt))
        if index < 0:
            return np.zeros(3),index
            
        acc = acc_memory[index]
        acc_perp = np.array([0,acc[1],0]) #acc-np.dot(acc,R_unit)
        
        return const.e/(4*np.pi*const.epsilon_0*const.c**2)*1/np.linalg.norm(R[index])*acc_perp,index
    acc_memory = acc_memory

    Ess = []
    i=0
    for t in (timescale):
        E,index = E_rad(detector_position,t,R[i])
        Ess.append(E)
        if index>=0:
            i+=1   
        
    Ess = np.array(Ess)

    fourier = np.abs(fft(Ess.T[1][:]))
    freqs = fftfreq(len(fourier),timescale[-1]/len(timescale))
    
    plt.plot(freqs[:50]*1e-9,fourier[:50]*1e9,c=(204/255,0,0))
    plt.xlabel(r"$f$ (GHz)",fontsize=14)
    plt.ylabel(r"$FFT(E_y)$ (nV m$^{-1}$)",fontsize=14)
    plt.title("FFT des Abgestrahlten Feldes bei $r=(0,0,1)$",fontsize=14)
    plt.grid()

    print("f_max = ",freqs[np.where(fourier==np.max(fourier))[0][0]]*1E-6,"MHz")

    
    
    frequencies.append(freqs[np.where(fourier==np.max(fourier))[0][0]])

#%%plot
plt.plot(energies,frequencies)


#%% Plotting 3D
for angle in np.linspace(0,90,100):
    fig=plt.figure(figsize=(15,15))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(np.array(memory).T[0],np.array(memory).T[1],np.array(memory).T[2],c="red")
    mag.show_dipole(ax)
    ax.view_init(angle, angle/2)
    plt.show()
    plt.close()


#%% E-Field and potential (retardiertes Feld)
R = []
for place in memory:
    R.append(np.array(detector_position)-place)
R = np.array(R)

def E_field(r,t,place,vel,acc,ret):
    index = int(np.round((t - ret)/el.Dt))
    
    if index < 0:
        return np.zeros(3)
    
    gamma = (1-(np.linalg.norm(vel)/const.c)**2)**(-1/2)
    beta = vel/const.c
    beta_dot = acc/const.c
    
    near_field = 0#-const.e/gamma**2 * R[index]-np.linalg.norm(R[index])*beta/(np.linalg.norm(R[index])-np.dot(R[index],beta))**3 
    
    rad_field = -const.e/const.c * np.cross(R[index],np.cross(R[index]-np.linalg.norm(R[index])*beta,beta_dot))/(np.linalg.norm(R[index])-np.dot(R[index],beta))**3
    return near_field+rad_field

def potential(r,t,place,vel,ret):
    index = int(np.round((t - ret)/el.Dt))
    
    if index < 0:
        return 0
    
    n=R[index]/np.linalg.norm(R[index])
    
    return -const.e / np.linalg.norm(R[index])* 1/(1-(np.dot(n,vel_memory[index])/const.c))


E_rad = []
B_rad = []
phis = []
for t,place,vel,acc,ret,r in zip(timescale,memory,vel_memory,acc_memory,retardierung,R):
    E_point = E_field(detector_position,t,place,vel,acc,ret)
    B_point = np.cross(r/np.linalg.norm(r),E_point)
    E_rad.append(E_point)
    B_rad.append(B_point)
    
    phi = potential(detector_position,t,place,vel,ret)
    phis.append(phi)
    
E_rad = np.array(E_rad).T
B_rad = np.array(B_rad).T
phis = np.array(phis).T
fourier = np.abs(fft(E_rad[1][2287:]))
freqs = fftfreq(len(fourier),timescale[-1]/len(timescale))
#%%
#plt.plot(timescale,np.linalg.norm(E_rad,axis=0))

plt.plot(timescale[2287:],E_rad[1][2287:])
#plt.plot(timescale[2000:],E_rad[1][2000:])
#plt.plot(freqs[:10],fourier[:10])
print("f_max = ",freqs[np.where(fourier==np.max(fourier))[0][0]]*1E-6,"MHz")

#%% Anderes berechnen vom Feld (wahrscheinlich falsch)
def rq(t,trajectory):
    i=int(np.round(t/1E-13))
    return trajectory[i]
def vq(t,trajectory):
    i=int(np.round(t/1E-13))
    return trajectory[i]
    
def potential(r,t,trajectory,ret):
    r = np.array(r)
    tr = t - ret
    
    if tr<0:
        return 0
    rqtr = rq(tr,trajectory)
    vqtr = vq(tr,trajectory)
    n = r-rqtr/np.linalg.norm(r-rqtr)
    return -const.e/np.linalg.norm(r-rqtr)*1/(1-np.dot(n,vqtr)/const.c)

pot=[]

for t,ret in zip(timescale,retardierung):
    pot.append(potential(detector_position,t,(memory,vel_memory),ret))
#%%
E = -np.gradient(pot)
frequencies = fft(pot[45000:70000])
#newpot = ifft(frequencies)
plt.plot(timescale[45000:],pot[45000:])
#plt.plot(frequencies)
#plt.plot(timescale[44000:],newpot[44000:])
#%%
mag=Magnetarray(10,[m1,0,0],distance=.07,pairdist=0.08)

def B_total(r):
    return mag.B(r)+cor_r.B(r)+cor_l.B(r)+cor_h.B(r)+cor_v.B(r)

ZZ = np.array([np.zeros(80),np.zeros(80),np.linspace(0,.7,80)]).T
BZZ = np.array([B_total(ZZZ) for ZZZ in ZZ])



fig=plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection='3d')
plt.xlim(-.05,.05)
plt.ylim(-.05,.05)



ax.quiver(ZZ.T[0],ZZ.T[1],ZZ.T[2],BZZ.T[0],BZZ.T[1],BZZ.T[2],color=(204/255,0,0),label="Magnetfeld")

ax.set_xlabel("x (m)",fontsize=20)
ax.set_ylabel("y (m)",fontsize=20)
ax.set_zlabel("z (m)",fontsize=20)
ax.legend(fontsize=20)
ax.view_init(20, 40)

mag.show_dipole(ax)
plt.tight_layout()

plt.savefig(savepath+"/Stack_with_magnets.png",dpi=200)
#plt.plot(ZZ.T[2],BZZ.T[0])




