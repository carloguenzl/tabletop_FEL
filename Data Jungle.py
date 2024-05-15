# -*- coding: utf-8 -*-
"""
Created on Fri May  3 19:24:33 2024

@author: carlo
"""
import numpy as np
import matplotlib.pyplot as plt

path ="C:/Users/carlo/Desktop/Dateidatei/6. SEMESTER/F-Praktikum/FEL/Spektren/Test2.mdr"

def get_mdrdata(path):
    #Get the stuff
    Files = open(path)
    content = Files.readlines()
    
    count = content[48][28:32]
    Frequencies = np.array(content[48][34:-1].split("; ")[:-1],dtype=float)
    
    marker = '        <Values count="'+count+'" unit="0">'
    Values = content[49].replace(marker,'')
    
    Values = [np.array(Values.split("; ")[:-1],dtype=float)]
    
    for i in range(50,len(content)):
        if marker in content[i]:
            morevalues = content[i].replace(marker,'')
            Values.append(np.array(morevalues.split("; ")[:-1],dtype=float))    
    return Frequencies,np.array(Values)

Frequencies,Values = get_mdrdata(path)

MeanValue = np.mean(Values,axis=0)
Stray = np.std(Values,axis=0)

for i in Values:
    plt.plot(Frequencies,i)
    plt.show()

plt.plot(Frequencies,MeanValue)
plt.plot(Frequencies,Stray)
