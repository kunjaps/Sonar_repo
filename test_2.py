# This code is to simulate a simple noisy sine wave
# We need plt,numpy

import numpy as np
import matplotlib.pyplot as plt
import math

## Simulate a sine wave + noise in MATLAB and see the effect of SNR on both time domain and frequency domain
# role: important.
# status : complete

# Initialising variables
f      = 200                                                  #the main frequency
Fs     = 32000                                                #sampling frequency
Ts     = 1/Fs                                                 #sampling interval
N      = 512                                                  #number of intervals

SNR    = 10                                                   #signal to noise ratio
SNR_weight = 10**(-1*SNR*0.05)                                #SNR multiplying factor

t = np.zeros([N],dtype = float)
for i in range(N):
  t[i] = i*Ts

new_mat    = np.zeros([N],dtype = float)                      #initialising noise included signal
SNR_weight*np.random.rand(N)

# creating the sine wave
y = np.sin(2*np.pi*f*t) + SNR_weight*np.random.rand(N) ;      #signal vector

plt.plot(t, y)
plt.show()
