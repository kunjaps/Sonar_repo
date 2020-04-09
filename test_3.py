# Code to simulate the output of a sensor array that is picking up
# a target that is has a single frequency signature

import numpy as np
import matplotlib.pyplot as plt
import math


# Initialising variables
f      = 2000                                 # the main frequency
Fs     = 128000                                # sampling frequency
Ts     = 1/Fs                                 # sampling interval
N      = 64                                  # number of samples

num_sensors = 64                              # number of sensors
angle       = 60                              # input wave angle
c           = 1500                            # speed of sound in water
wavelength  = c/f                             # wavelength
x           = wavelength/2                    # interspace distance

t = np.zeros([N],dtype = float)               # a time array is created
for i in range(N):
  t[i] = i*Ts

unit_delay = x*np.cos(angle)/c                # unit delay
matrix    = np.empty([num_sensors,N],dtype='float')   # delayed pure signals

##########################################
SNR    = 10                                   #signal to noise ratio
SNR_weight = 10**(-1*SNR*0.05)                #SNR multiplying factor

##########################################
test_time = np.zeros(N)
k=0
while k<N:
  test_time[k] = k
  k+=1
##########################################

i=0
while i < num_sensors:
  matrix[i,] = np.sin(2*np.pi*(f/Fs)*(test_time - i*unit_delay))
  matrix[i,] += SNR_weight*np.random.rand(N)
  i+=1

#########################################
plt.plot(test_time, matrix[1,:],'r')
plt.plot(test_time, matrix[5,:],'b')
plt.plot(test_time,matrix[12,:],'g')
plt.show
