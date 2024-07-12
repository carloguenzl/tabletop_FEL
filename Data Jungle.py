# -*- coding: utf-8 -*-
"""
Created on Fri May  3 19:24:33 2024

@author: carlo
"""
import numpy as np
import matplotlib.pyplot as plt

path ="C:/Users/carlo/Desktop/Dateidatei/6. SEMESTER/F-Praktikum/FEL_GIT/Spektren/MESSUNG/MESSUNG 1 MIT.mdr"
empty = "C:/Users/carlo/Desktop/Dateidatei/6. SEMESTER/F-Praktikum/FEL_GIT/Spektren/MESSUNG/Messung ohne Magnete.mdr"
def get_mdrdata(path):
    #Get the stuff
    Files = open(path)
    content = Files.readlines()
    
    count = content[47][28:32]

    Frequencies = np.array(content[47][34:-1].split("; ")[:-1],dtype=float)
    
    marker = f'        <Values count="{count}" unit="0">'
    Values = content[48].replace(marker,'')
    
    Values = [np.array(Values.split("; ")[:-1],dtype=float)]
    
    for i in range(50,len(content)):
        if marker in content[i]:
            morevalues = content[i].replace(marker,'')
            Values.append(np.array(morevalues.split("; ")[:-1],dtype=float))    
    return Frequencies,np.array(Values)

Frequencies,Values = get_mdrdata(path)

_,groundvalues = get_mdrdata(empty)
compare = np.mean(groundvalues,axis=0)

MeanValue = np.mean(Values,axis=0)
Stray = np.std(Values,axis=0)


# for i in Values:
#     plt.plot(Frequencies,i)
#     plt.show()
#%%
plt.figure(dpi=150)
plt.plot(Frequencies*1E-6,Stray,linewidth=.5,c=(204/255,0,0))
plt.grid()
plt.xlabel("Frequenz (MHz)")
plt.ylabel("Pegel (dB)")
#plt.savefig("C:/Users/carlo/Desktop/Dateidatei/6. SEMESTER/F-Praktikum/FEL/pics/spectran_rauschen.png",dpi=200)
#plt.plot(Frequencies,Stray)
