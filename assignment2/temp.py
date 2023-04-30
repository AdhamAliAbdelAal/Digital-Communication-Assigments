import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

# Parameters
Eb = 1.0      # Energy per bit
M = 2        # Number of symbols
k = np.log2(M)  # Number of bits per symbol
SNR_dB = np.arange(-10, 21, 1)  # Range of E/N0 values in dB
SNR = 10**(SNR_dB/10.0)      # Range of E/N0 values in linear scale

# Theoretical BER
Pe = []
for snr in SNR:
    Es = Eb*k   # Energy per symbol
    sigma = np.sqrt(Es/(2*snr))  # Standard deviation of noise
    d = np.sqrt(Es)/2     # Distance between symbols
    Q = lambda x: 0.5*sp.erfc(x/np.sqrt(2))   # Q-function
    Pe.append(0.5*sp.erfc(snr**0.5))

# Plot results
plt.semilogy(SNR_dB, Pe, 'b.-')
plt.xlabel('E/N0 (dB)')
plt.ylabel('Bit Error Rate')
plt.grid(True)
plt.show()
