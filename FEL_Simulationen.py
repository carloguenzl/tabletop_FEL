# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 16:39:44 2024

@author: carlo

Working params:
    el = electron((0,.003,0),10,(0,0))
    mag=Magnetarray(10,[m1,0,0],distance=.06,pairdist=0.12)



"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import scipy.constants as const
savepath = "C:/Users/carlo/Desktop/Dateidatei/6. SEMESTER/F-Praktikum/FEL/pics"
#%% Dipole strength
def get_dipole(B_rem,l,d):
    return B_rem * np.pi*(d/2)**2/const.mu_0*l #Joule/Tesla

m1 = get_dipole(1.3, 20E-3, 10E-3)

#%% Magnet implementation
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
        canvas.quiver(x,y,z,mx,my,mz,normalize=True,length = .01,linewidth=10,color=(204/255,0,0))
        canvas.quiver(x,y,z,-np.array(mx),my,mz,normalize=True,length = .01,linewidth=10,color=(34/255,139/255,34/255))

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
    
    def show_dipole(self,canvas):
        coords = np.array([self.root + i*np.array([0,0,self.distance]) for i in range(self.num_of_magnets)])
        x,y,z = coords.T[0],coords.T[1],coords.T[2]
        directs = np.array([self.m for _ in range(self.num_of_magnets)])
        mx,my,mz = directs.T[0],directs.T[1],directs.T[2]
        canvas.quiver(x,y,z,-mx,-my,-mz,normalize=True,length = .01,linewidth=10,color=(204/255,0,0))
        canvas.quiver(x,y,z,mx,my,mz,normalize=True,length = .01,linewidth=10,color=(34/255,139/255,34/255))

#%% calculate a global magnetic field from magnets
mag=Magnetarray(0,[m1,0,0],distance=.06,pairdist=0.12)
cor_r = Correctionstack(10,[-m1,0,0],[0,.05,0],.013)
cor_l = Correctionstack(10,[m1,0,0],[0,-.05,0],.013)
cor_v = Correctionstack(0,[0,m1,0],(.04,0,0),.013)
cor_h = Correctionstack(0,[0,-m1,0],(-.04,0,0),.013)


def B_total(r): 
    return mag.B(r)+cor_r.B(r)+cor_l.B(r)#+cor_h.B(r)+cor_v.B(r)

detector_position = (0,0,.3)
# %% electron classification
class electron:
    def __init__(self,place,energy,offset=(0,0)): #energy in keV
        self.place=np.array(place)
        
        self.vel_abs = np.sqrt(1-(((const.m_e*const.c**2)/const.e)/(energy*10**3+const.m_e*const.c**2/const.e))**2)*const.c
        self.vel = np.array([self.vel_abs*offset[0]*np.cos(offset[1]),self.vel_abs*offset[0]*np.sin(offset[1]),np.sqrt(1-offset[0]**2)*self.vel_abs])
        
        self.rel_mass = const.m_e/(np.sqrt(1-(self.vel_abs/const.c)**2))
        
        self.Dt = 1E-11
        
        self.timesteps = 0
    
    def Lorenz_acc(self,place,velocity):
        B = B_total(place)
        return -const.e*(np.cross(velocity,B))/self.rel_mass
    
    def eulerstep(self):
        vel = self.vel
        place = self.place
        self.vel = vel + self.Lorenz_acc(self.place,vel)*self.Dt
        self.vel = self.vel/np.linalg.norm(self.vel)*self.vel_abs
        self.place = place + vel*self.Dt
        
        self.timesteps+=1
    
    def rungestep(self):
        h = self.Dt
        vel = self.vel
        a1 = self.Lorenz_acc(self.place,vel)
        v1 = vel + a1 * h        
        a2 = self.Lorenz_acc(self.place + v1 * h + (a1/2)/2*h**2,v1)
        v2 = v1 + a2 * h
        a3 = self.Lorenz_acc(self.place + v2 * h + (a2/2)/2*h**2,v2)
        v3 = v2 + a3 * h
        a4 = self.Lorenz_acc(self.place + v3 * h + a3/2*h**2,v3)
        
        self.vel = vel + h/6*(a1+a2*2+a3*2+a4)
        self.vel = self.vel/np.linalg.norm(self.vel)*self.vel_abs
        
        self.place = self.place+vel*self.Dt
        
        self.timesteps+=1

def get_offset():
    return min(1,abs(np.random.normal(0,.3))),np.random.rand()*2*np.pi




# %% Sorted main simulator in global magnetic field 
class main:
    def __init__(self,place,energy,offset=(0,0),steps=1e3):
        self.steps = steps
        self.el = electron(place,energy,offset)
        
        self.trajectory = [(self.el.place,self.el.vel)]
        self.memory = [self.el.place]
        self.vel_memory = [self.el.vel]
        self.acc_memory = [self.el.Lorenz_acc(self.el.place,self.el.vel)]
    
    def run_el(self):
        steps = self.steps
        i=0
        while self.el.place[0]**2+self.el.place[1]**2 <= 1**2 and i <= steps:
            self.el.rungestep()
            self.memory.append(self.el.place)
            self.vel_memory.append(self.el.vel)
            self.acc_memory.append(self.el.Lorenz_acc(self.el.place,self.el.vel))
            self.trajectory.append((self.el.place,self.el.vel))
            if i%(steps/10)==0:
                print(int(i/steps*100),"%")
            i+=1
        
        if self.el.place[0]**2+self.el.place[1]**2 > .07**2:
            print("hit border")

        
        self.retardierung = []
        for place in self.memory:
            self.retardierung.append(np.linalg.norm(np.array(detector_position)-place)/const.c)
        self.retardierung = np.array(self.retardierung)
        
        self.timescale = np.arange(0,(len(self.memory))*self.el.Dt,self.el.Dt)
        
        fig=plt.figure(figsize=(10,10))
        ax = fig.add_subplot(projection='3d')
        plt.xlim(-.05,.05)
        plt.ylim(-.05,.05)
        
        #mag.show_dipole(ax)
        cor_r.show_dipole(ax)
        cor_l.show_dipole(ax)
        #cor_h.show_dipole(ax)
        #cor_v.show_dipole(ax)
        
        ax.scatter(np.array(self.memory).T[0],np.array(self.memory).T[1],np.array(self.memory).T[2],color=(204/255,0,0),label="Trajektorie")
        
        ax.set_xlabel("x (m)",fontsize=20)
        ax.set_ylabel("y (m)",fontsize=20)
        ax.set_zlabel("z (m)",fontsize=20)
        ax.legend(fontsize=20)
        ax.view_init(20, 40)
        plt.tight_layout()
        
        #plt.savefig(savepath+"/trajectory_first.png",dpi=200)
        plt.show()
        plt.close()
    
    def field_and_fourier(self):
        R = np.array([(np.array(detector_position)-place) for place in self.memory])
    
        def E_rad(r,t,ret):
            index = int(np.round((t - np.linalg.norm(ret)/const.c)/self.el.Dt))
            if index < 0:
                return np.zeros(3),index
                
            acc = self.acc_memory[index]
            acc_perp = np.array([0,acc[1],0]) #acc-np.dot(acc,R_unit)
            
            return const.e/(4*np.pi*const.epsilon_0*const.c**2)*1/np.linalg.norm(R[index])*acc_perp,index

    
        Ess = []
        i=0
        for t in (self.timescale):
            E,index = E_rad(detector_position,t,R[i])
            Ess.append(E)
            if index>=0:
                i+=1   
    
        Ess = np.array(Ess)
    
        fourier = np.abs(fft(Ess.T[1][:]))
        freqs = fftfreq(len(fourier),self.timescale[-1]/len(self.timescale))
        
        plt.plot(self.timescale[:]*1e9,Ess.T[1][:]*1e9,c=(204/255,0,0))
        plt.xlabel("$t$ (ns)",fontsize=14)
        plt.ylabel(r"$E_y$ (nV m$^{-1}$)",fontsize=14)
        plt.title("Abgestrahltes Feld der Trajektorie bei $r=(0,0,1)$",fontsize=14)
        plt.tight_layout()
        plt.grid()
        plt.show()
        # plt.savefig(savepath+"/rad_field_wod_546keV.png",dpi=200)


        plt.plot(freqs[:50]*1e-9,fourier[:50]*1e9,c=(204/255,0,0))
        plt.xlabel(r"$f$ (GHz)",fontsize=14)
        plt.ylabel(r"$FFT(E_y)$ (nV m$^{-1}$)",fontsize=14)
        plt.title("FFT des Abgestrahlten Feldes bei $r=(0,0,1)$",fontsize=14)
        plt.grid()
        plt.show()
        # plt.savefig(savepath+"/fourier_wod_546keV.png",dpi=200)
        self.fmax = freqs[np.where(fourier==np.max(fourier))[0][0]]*1E-6
        print("f_max = ",freqs[np.where(fourier==np.max(fourier))[0][0]]*1E-6,"MHz")


# -----------------------SIMULATIONS-BEGIN-HERE-------------------------------------
# %% Energy-frequency dependence 

Es = np.logspace(1,3,10)
fs = list()
for E in Es:
    run = main((0,.02,0),E,(0,0),2e3)
    run.run_el()
    run.field_and_fourier()
    fs.append(run.fmax)
    
    
plt.plot(Es,fs)


#%% monte carlo parameterraum

el = electron((0,.02,0),10,(0,0))


# offset = get_offset()
# #el = electron((0,0,0),10,(.8,np.pi/2)) #energy in keV

# trajectory = [(el.place,el.vel)]

# memory = [el.place]
# vel_memory = [el.vel]
# acc_memory = [el.Lorenz_acc(el.place,el.vel)]

def run_electron(electron,steps=1E3):
    memory = [electron.place]
    vel_memory = [electron.vel]
    acc_memory = [electron.Lorenz_acc(electron.place)]

    i=0
    while electron.place[0]**2+electron.place[1]**2 <= .04**2 and el.vel[2]>=0 and i <= steps:
        electron.rungestep()
        memory.append(electron.place)
        vel_memory.append(electron.vel)
        acc_memory.append(electron.Lorenz_acc(electron.place))
        i+=1
    return memory,vel_memory,acc_memory

clipboard = []

no_runs=100
for run in range(no_runs):
    parameters = get_offset()
    contestant = electron((0,.003,0),1,parameters)
    clipboard.append((run_electron(contestant),parameters))
    print(run*no_runs/100,"%")
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
plt.scatter(np.arcsin(sucessful[0])*360/(2*np.pi),sucessful[1]*360/(2*np.pi),color=(30/255,100/255,200/255),s=3,label="Erfolgreiche Läufe")
plt.scatter(np.arcsin(unsucessful[0])*360/(2*np.pi),unsucessful[1]*360/(2*np.pi),color=(204/255,0,0),s=3,label="Erfolglose Läufe")
plt.grid()
plt.legend()
plt.savefig(savepath+"/parameterraum.png",dpi=300)
plt.show()