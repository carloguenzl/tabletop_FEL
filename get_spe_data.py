# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:15:13 2024

@author: carlo
"""


import numpy as np
import matplotlib.pyplot as plt


path = "C:/Users/carlo/OneDrive/Fortgeschrittenenpraktikum/FEL/Erste Messungen/Kalibrierung.Spe"


def get_spedata(path):
    #Get the stuff
    Files = open(path)
    content = Files.readlines()
    Files.close()
    borders = (int(content[11].split(" ")[0]), int(content[11].split(" ")[1].replace("\n","")))
    data = np.array([int(content[i]) for i in range(borders[0]+12,borders[1]+12)])
    return data

result = get_spedata(path)

#%%
plt.figure(dpi=150)
plt.plot(result,linewidth=.5,c=(204/255,0,0))
plt.grid()
plt.xlabel("Energy")
plt.ylabel("count")
#plt.savefig("C:/Users/carlo/Desktop/Dateidatei/6. SEMESTER/F-Praktikum/FEL/pics/spectran_rauschen.png",dpi=200)