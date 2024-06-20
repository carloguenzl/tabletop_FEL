# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 16:39:44 2024

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

#%%

class Magnet:
    def __init__(self,place,m):
        self.place = np.array(place)
        self.m = np.array(m)
    
    def B(self,r:np.ndarray):
        r = np.array(r-self.place)
        return const.mu_0/(4*np.pi) * ((3*np.dot(self.m,r)*r-self.m*np.linalg.norm(r)**2))/(np.linalg.norm(r)**5)
    
class Magnetpair:
    def __init__(self,place,m,distance=.12):
        self.distance = distance
        self.place = np.array(place)
        self.m = np.array(m)
    
    def B(self,r):
        magnet1=Magnet(self.place+(self.distance/2*self.m/np.linalg.norm(self.m)),self.m)
        magnet2=Magnet(self.place-(self.distance/2*self.m/np.linalg.norm(self.m)),self.m)
        return magnet1.B(r)+magnet2.B(r)

class Magnetarray:
    def __init__(self,num_of_pairs,m,distance=.02,pairdist=0.12):
        self.distance = distance
        self.num_of_pairs = int(num_of_pairs)
        self.m = m
        self.pairdist = pairdist
    
    def B(self,r):
        B=np.zeros(3)
        for i in range(self.num_of_pairs):
            place=np.array([0,0,i*self.distance])
            mag_pair=Magnetpair(place,self.m,distance=self.pairdist)
            B+=mag_pair.B(r)*(-1)**i
        return B
    
    def show_dipole(self, canvas):
        x,y,z=[-self.pairdist/2,self.pairdist/2]*self.num_of_pairs , [0,0]*self.num_of_pairs , [(1)**k*i*self.distance for i in range(self.num_of_pairs) for k in range(2) ]

        mx,my,mz = [(-1)**(i//2) for i in range(2*self.num_of_pairs)],[0,0]*self.num_of_pairs,[[0,0]*self.num_of_pairs]
        canvas.quiver(x,y,z,mx,my,mz,normalize=True,length = .01,linewidth=10,color=(34/255,139/255,34/255))
        canvas.quiver(x,y,z,-np.array(mx),my,mz,normalize=True,length = .01,linewidth=10,color=(204/255,0,0))

class Correctionstack:
    def __init__(self,num_of_magnets,m,root,distance):
        self.num_of_magnets = num_of_magnets
        self.m = np.array(m)
        self.distance = distance
        self.root = np.array(root)
    
    def B(self,r):
        B = np.zeros(3)
        for i in range(self.num_of_magnets):
            place = self.root+np.array([0,0,i*self.distance])
            mag = Magnet(place,self.m)
            B+=mag.B(r)
        return B


#%%        
mag=Magnetarray(10,[m1,0,0],distance=.06,pairdist=0.12)
cor_r = Correctionstack(0,[-m1,0,0],[0,.03,0],.013)
cor_l = Correctionstack(0,[m1,0,0],(0,-.03,0),.013)

def B_total(r):
    return mag.B(r)+cor_r.B(r)+cor_l.B(r)


detector_position = (0,0,1)

class electron:
    def __init__(self,place,energy,offset=(0,0)): #energy in keV
        self.place=np.array(place)
                
        self.vel_abs = np.sqrt(1-((const.m_e*const.c**2/const.e)/(energy*10**3+const.m_e*const.c**2/const.e))**2)*const.c
        self.vel = np.array([self.vel_abs*offset[0]*np.cos(offset[1]),self.vel_abs*offset[0]*np.sin(offset[1]),np.sqrt(1-offset[0]**2)*self.vel_abs])
        
        self.rel_mass = const.m_e/(np.sqrt(1-(self.vel_abs/const.c)**2))
        
        self.Dt = 1E-11
        
        self.timesteps = 0
    
    def Lorenz_acc(self,place):
        B = B_total(place)
        return -const.e*(np.cross(self.vel,B))/self.rel_mass
    
    def eulerstep(self):
        vel = self.vel
        place = self.place
        self.vel = vel + self.Lorenz_acc(self.place)*self.Dt
        self.vel = self.vel/np.linalg.norm(self.vel)*self.vel_abs
        self.place = place + vel*self.Dt
        
        self.timesteps+=1
    
    def rungestep(self):
        h = self.Dt
        vel = self.vel
        k1 = self.Lorenz_acc(self.place)
        k2 = self.Lorenz_acc(self.place+h*k1/2)
        k3 = self.Lorenz_acc(self.place+h*k2/2)
        k4 = self.Lorenz_acc(self.place+h*k3)
        
        self.vel = vel + h/6*(k1+k2*2+k3*2+k4)
        self.vel = self.vel/np.linalg.norm(self.vel)*self.vel_abs
        
        self.place = self.place+vel*self.Dt
        
        self.timesteps+=1

def get_offset():
    return min(1,abs(np.random.normal(0,.3))),np.random.rand()*2*np.pi

offset = get_offset()
el = electron((0,.003,0),10,(0,0)) #energy in keV

trajectory = [(el.place,el.vel)]

memory = [el.place]
vel_memory = [el.vel]
acc_memory = [el.Lorenz_acc(el.place)]


#%% get Trajectory
steps=1E3
i=0
while el.place[0]**2+el.place[1]**2 <= 1**2 and i <= steps:
    el.rungestep()
    memory.append(el.place)
    vel_memory.append(el.vel)
    acc_memory.append(el.Lorenz_acc(el.place))
    trajectory.append((el.place,el.vel))
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


fig=plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection='3d')
plt.xlim(-.05,.05)
plt.ylim(-.05,.05)

mag.show_dipole(ax)
ax.scatter(np.array(memory).T[0],np.array(memory).T[1],np.array(memory).T[2],color=(204/255,0,0),label="Trajektorie")

ax.set_xlabel("x (m)",fontsize=20)
ax.set_ylabel("y (m)",fontsize=20)
ax.set_zlabel("z (m)",fontsize=20)
ax.legend(fontsize=20)
ax.view_init(20, 40)
plt.tight_layout()
plt.show()
#plt.savefig("C:/Users/carlo/Desktop/Dateidatei/6. SEMESTER/F-Praktikum/FEL/pics/first_trajectory.png",dpi=200)
plt.close()

#%% Retry field. again.
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

#%%
plt.plot(timescale[:]*1e9,Ess.T[1][:]*1e9,c=(204/255,0,0))
plt.xlabel("$t$ (ns)",fontsize=14)
plt.ylabel(r"$E_y$ (nV m$^{-1}$)",fontsize=14)
plt.title("Abgestrahltes Feld der Trajektorie bei $r=(0,0,1)$",fontsize=14)
plt.tight_layout()
plt.grid()
plt.savefig("C:/Users/carlo/Desktop/Dateidatei/6. SEMESTER/F-Praktikum/FEL/pics/rad_field.png",dpi=200)

#%%
plt.plot(freqs[:50]*1e-9,fourier[:50]*1e9,c=(204/255,0,0))
plt.xlabel(r"$f$ (GHz)",fontsize=14)
plt.ylabel(r"$FFT(E_y)$ (nV m$^{-1}$)",fontsize=14)
plt.title("FFT des Abgestrahlten Feldes bei $r=(0,0,1)$",fontsize=14)
plt.grid()
plt.savefig("C:/Users/carlo/Desktop/Dateidatei/6. SEMESTER/F-Praktikum/FEL/pics/fourier.png",dpi=200)

print("f_max = ",freqs[np.where(fourier==np.max(fourier))[0][0]]*1E-6,"MHz")

#%% MONTE CARLO
def run_electron(electron,steps=1E3):
    memory = [electron.place]
    vel_memory = [electron.vel]
    acc_memory = [electron.Lorenz_acc(electron.place)]

    i=0
    while electron.place[0]**2+electron.place[1]**2 <= .04**2 and i <= steps:
        electron.rungestep()
        memory.append(electron.place)
        vel_memory.append(electron.vel)
        acc_memory.append(electron.Lorenz_acc(electron.place))
        i+=1
    return memory,vel_memory,acc_memory

clipboard = []

for run in range(1000):
    parameters = get_offset()
    contestant = electron((0,0,0),10,parameters)
    clipboard.append((run_electron(contestant),parameters))
    print(run*1/10,"%")
#%% check parameterraum
sucessful = []
unsucessful = []
for run in clipboard: # sort
    if run[0][0][-1][0]**2+run[0][0][-1][1]**2 <= .04**2:
        sucessful.append(np.array(run[1]))
    else:
        unsucessful.append(np.array(run[1]))

sucessful = np.array(sucessful).T
unsucessful = np.array(unsucessful).T

plt.xlabel("θ (°)")
plt.ylabel("φ (°)")
plt.xlim(0,40)
plt.ylim(0,360)
plt.scatter(np.arcsin(sucessful[0])*360/(2*np.pi),sucessful[1]*360/(2*np.pi),color=(34/255,139/255,34/255),s=3,label="Erfolgreiche Läufe")
plt.scatter(np.arcsin(unsucessful[0])*360/(2*np.pi),unsucessful[1]*360/(2*np.pi),color=(204/255,0,0),s=3,label="Erfolglose Läufe")
plt.grid()
plt.legend()
plt.savefig("C:/Users/carlo/Desktop/Dateidatei/6. SEMESTER/F-Praktikum/FEL/pics/parameterraum.png",dpi=200)
plt.show()


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
ZZ = np.array([np.zeros(1000),np.zeros(1000)-.12,np.linspace(0,.5,1000)]).T
BZZ = np.array([B_total(ZZZ) for ZZZ in ZZ])
plt.plot(ZZ.T[2],BZZ.T[0])

