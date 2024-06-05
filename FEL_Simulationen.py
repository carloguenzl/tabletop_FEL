# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 16:39:44 2024

@author: carlo
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
import scipy.constants as const
#%%
def get_dipole(B_rem,l,d):
    return B_rem * np.pi*(d/2)**2/const.mu_0*l #Joule/Tesla

m1 = get_dipole(1.3, 20E-3, 10E-3)

#%%
zl,zr=-.2,.5
yl,yr=-.2,.2
xl,xr=-.2,.2
ndens=10

nx,ny,nz = np.linspace(xl, xr,ndens),np.linspace(yl,yr,ndens),np.linspace(zl,zr,ndens)

class Magnet:
    def __init__(self,place,m):
        self.place = np.array(place)
        self.m = np.array(m)
    
    def B(self,r):
        r = np.array(r)
        r = np.array(r-self.place)  
        return const.mu_0/(4*np.pi) * ((3*np.dot(self.m,r)*r-self.m*np.linalg.norm(r)**2))/(np.linalg.norm(r)**5)
    
class Magnetpair:
    def __init__(self,place,m,distance=.04):
        self.distance = distance
        self.place = np.array(place)
        self.m = np.array(m)
    
    def B(self,r):
        magnet1=Magnet(self.place+(self.distance/2*self.m/np.linalg.norm(self.m)),self.m)
        magnet2=Magnet(self.place-(self.distance/2*self.m/np.linalg.norm(self.m)),self.m)
        return magnet1.B(r)+magnet2.B(r)

class Magnetarray:
    def __init__(self,num_of_pairs,m,distance=.013):
        self.distance = distance
        self.num_of_pairs = int(num_of_pairs)
        self.m = m
    
    def B(self,r):
        B=np.zeros(3)
        for i in range(self.num_of_pairs):
            place=np.array([0,0,i*self.distance])
            mag_pair=Magnetpair(place,self.m)
            B+=mag_pair.B(r)*(-1)**i
        return B

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
mag=Magnetarray(0,[m1,0,0],distance=.08)
cor_r = Correctionstack(20,[-m1,0,0],[0,.03,0],.013)
cor_l = Correctionstack(20,[m1,0,0],(0,-.03,0),.013)

def B_total(r):
    return mag.B(r)+cor_r.B(r)+cor_l.B(r)



B = np.zeros((ndens,ndens,ndens,3))
for i,x in enumerate(nx):
    for j,y in enumerate(ny):
        for k,z in enumerate(nz):
            B[i,j,k]=B_total((x,y,z))
B=B.T
Bx=B[0].T
By=B[1].T
Bz=B[2].T


detector_position = (0,0,1)

class electron:
    def __init__(self,place,energy,offset=(0,0)): #energy in keV
        self.place=np.array(place)
        
        #same thing...
        #E = energy*1E3*const.e
        #self.vel_abs1 = np.sqrt((E*(2*const.m_e*const.c**4+E*const.c**2))/(const.m_e**2*const.c**4 + 2*E*const.m_e*const.c**2+E**2))
        
        self.vel_abs = np.sqrt(1-((const.m_e*const.c**2/const.e)/(energy*10**3+const.m_e*const.c**2/const.e))**2)*const.c
        self.vel = np.array([self.vel_abs*offset[0]*np.cos(offset[1]),self.vel_abs*offset[0]*np.sin(offset[1]),np.sqrt(1-offset[0]**2)*self.vel_abs])
        
        self.rel_mass = const.m_e/(np.sqrt(1-(self.vel_abs/const.c)**2))
        
        self.Dt = 1E-12
        
        self.timesteps = 0
        
    def Lorenz_acc(self):
        B = B_total(self.place)
        return -const.e*(np.cross(self.vel,B))/self.rel_mass
    
    def step(self):
        #self.Dt = 1E-13#min(1E-10,1E-7/np.linalg.norm(mag.B(self.place))*1E-7)
        vel = self.vel
        place = self.place
        self.vel = vel + self.Lorenz_acc()*self.Dt
        self.vel = self.vel/np.linalg.norm(self.vel)*self.vel_abs
        self.place = place + vel*self.Dt
        
        self.timesteps+=1


def get_offset():
    return min(1,abs(np.random.normal(0,.3))),np.random.rand()*2*np.pi

offset = get_offset()
el = electron((0,.0,0),70,(.3,+np.pi/2)) #energy in keV pls  74.57   (offset[0],-np.pi/2)
print(offset)


trajectory = [(el.place,el.vel)]

memory = [el.place]
vel_memory = [el.vel]
acc_memory = [el.Lorenz_acc()]


#%% get Trajectory
steps=1E3
i=0
while el.place[0]**2+el.place[1]**2 <= 1**2 and i <= steps:
    el.step()
    memory.append(el.place)
    vel_memory.append(el.vel)
    acc_memory.append(el.Lorenz_acc())
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


fig=plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection='3d')
x,y,z=np.meshgrid(nx,ny,nz)

lengths=(np.sqrt(Bx**2+By**2+Bz**2))
lengths_normalized = (lengths-(abs(lengths).min())) / abs(lengths.max())
colors = (plt.cm.jet(lengths_normalized))
colors = colors.reshape(-1, 4)

ax.quiver(x,y,z,Bx,By,Bz,color=colors,length=0.05,normalize=True,linewidths=.5)
ax.scatter(np.array(memory).T[0],np.array(memory).T[1],np.array(memory).T[2],c="red")

plt.xlabel("x")
plt.ylabel("y")

ax.view_init(5, 0)
plt.show()
plt.close()
#%% continue for e-field
for i in range(int(1E3)):
    el.step()
    memory.append(el.place)
    vel_memory.append(el.vel)
    acc_memory.append(el.Lorenz_acc())
    trajectory.append((el.place,el.vel))
    if i%(steps/10)==0:
        print(int(i/steps*100),"%")

retardierung = []
for place in memory:
    retardierung.append(np.linalg.norm(np.array(detector_position)-place)/const.c)
retardierung = np.array(retardierung)

timescale = np.arange(0,(len(memory))*el.Dt,el.Dt)
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
fourier = np.abs(fft(E_rad[2][2287:]))
freqs = fftfreq(len(fourier),timescale[-1]/len(timescale))
#%%
#plt.plot(timescale,np.linalg.norm(E_rad,axis=0))

#plt.plot(timescale[2287:],E_rad[2][2287:])
plt.plot(timescale[:],phis)
#plt.plot(freqs[:10],fourier[:10])
#%% Retry field. again.
R = []
for place in memory:
    R.append(np.array(detector_position)-place)
R = np.array(R)

acc_memory = np.array(acc_memory)



def E_rad(r,t,ret):
    index = int(np.round((t - ret)/el.Dt))
    if index < 0:
        return np.zeros(3)
        
    acc = acc_memory[index]
    R_unit = R[index]/np.linalg.norm(R[index])
    acc_perp = acc-np.dot(acc,R_unit)
    
    return const.e/(4*np.pi*const.epsilon_0*const.c**2)*1/np.linalg.norm(R[index])*acc_perp
acc_memory = acc_memory

Ess = []
for t,ret in zip(timescale,retardierung):
    Ess.append(E_rad(detector_position,t,ret))
Ess = np.array(Ess)

fourier = np.abs(fft(Ess.T[0][2287:]))
freqs = fftfreq(len(fourier),timescale[-1]/len(timescale))

#%%
plt.plot(timescale[3000:],Ess.T[2][3000:])
#plt.plot(freqs[:10],fourier[:10])
plt.grid()

#%% MONTE CARLO
def run_electron(electron,steps=3.5E3):
    memory = [electron.place]
    vel_memory = [electron.vel]
    acc_memory = [electron.Lorenz_acc()]

    i=0
    while electron.place[0]**2+electron.place[1]**2 <= .1**2 and i <= steps:
        electron.step()
        memory.append(electron.place)
        vel_memory.append(electron.vel)
        acc_memory.append(electron.Lorenz_acc())
        i+=1
    return memory,vel_memory,acc_memory

clipboard = []

for run in range(100):
    parameters = get_offset()
    contestant = electron((0,0,0),200,parameters)
    clipboard.append((run_electron(contestant),parameters))
    print(run,"%")
#%% check parameterraum
sucessful = []
unsucessful = []
for run in clipboard: # sort
    if run[0][0][-1][0]**2+run[0][0][-1][1]**2 <= .07**2:
        sucessful.append(np.array(run[1]))
    else:
        unsucessful.append(np.array(run[1]))

sucessful = np.array(sucessful).T
unsucessful = np.array(unsucessful).T


plt.xlabel("radius")
plt.ylabel("angle")
plt.xlim(0,1)
plt.ylim(0,2*np.pi)
plt.scatter(sucessful[0],sucessful[1],c="green",s=3)
plt.scatter(unsucessful[0],unsucessful[1],c="red",s=3)
plt.show()



#%% 
for i in np.arange(50000,len(E_rad.T),100):
    e=E_rad.T[i]*3E17
    b=B_rad.T[i]*3E17
    fig=plt.figure(figsize=(15,15))
    ax = fig.add_subplot(projection='3d')
    
    ax.quiver(0, 0, 0, e[0],e[1],e[2], length=np.linalg.norm(e))
    ax.quiver(0, 0, 0, b[0],b[1],b[2], length=np.linalg.norm(b),color="red")
    ax.view_init((i-50000)/1000, i/2000)
    plt.show()
#ax.quiver(0, 0, 0, E_rad.T[50000,0], E_rad.T[50000,1], E_rad.T[50000,2], length=np.linalg.norm(E_rad.T[50000]))

#%% Plotting 3D
for angle in np.linspace(0,90,100):
    fig=plt.figure(figsize=(15,15))
    ax = fig.add_subplot(projection='3d')
    x,y,z=np.meshgrid(nx,ny,nz)
    
    lengths=(np.sqrt(Bx**2+By**2+Bz**2))
    lengths_normalized = (lengths-(abs(lengths).min())) / abs(lengths.max())
    colors = (plt.cm.jet(lengths))
    colors = colors.reshape(-1, 4)
    
    ax.quiver(x,y,z,Bx,By,Bz,color=colors,length=0.05,normalize=True,linewidths=.5)
    
    ax.scatter(np.array(memory).T[0],np.array(memory).T[1],np.array(memory).T[2],c="red")

    ax.view_init(angle, angle/2)
    plt.show()
    plt.close()
    

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


#%% test the magnet
ma = Magnet((0,0,0),(0,0,m1))

B_field = np.zeros((ndens,ndens,ndens,3))
for i,x in enumerate(nx):
    for j,y in enumerate(ny):
        for k,z in enumerate(nz):
            B_field[i,j,k]=ma.B((x,y,z))

B_field=B_field.T
Bx=B_field[0].T
By=B_field[1].T
Bz=B_field[2].T

# z,x=np.meshgrid(nz,nx)

# for i in range(ndens):
#     plt.streamplot(z,x,Bz[:,i,:],Bx[:,i,:])
#     plt.show()
# z,x=np.meshgrid(nz,nx)
# for i in range(ndens):
#     plt.streamplot(z,x,Bz[:,i,:],Bx[:,i,:])
#     plt.show()