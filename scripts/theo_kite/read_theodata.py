import matplotlib.pyplot as plt
import numpy as np


theo3 = np.array([4450909.840, 6040800.456])
theo4 = np.array([4450713.646, 6040934.273])
hotel = np.array([4449525.439, 6041088.713]) 

#np.arctan2(y,x)

nordcorr3 = np.arctan2(hotel[1]-theo3[1],hotel[0]-theo3[0])*180/np.pi-90
nordcorr4 = np.arctan2(hotel[1]-theo4[1],hotel[0]-theo4[0])*180/np.pi-90

print(nordcorr3)
print(nordcorr4)
#az+nordcorrectur

if 0:
    plt.plot(0,0,'bo')
    plt.plot(theo4[0]-theo3[0],theo4[1]-theo3[1],'go')
    plt.plot(hotel[0]-theo3[0],hotel[1]-theo3[1],'rx')
    plt.xlim([-1400,0])
    plt.ylim([0,1400])
    plt.show()

dat_theo = "./Theo_daten/"
with open("")


