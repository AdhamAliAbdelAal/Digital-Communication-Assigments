from scipy.special import erfc
import numpy as np
from matplotlib import pyplot as plt

def draw_BER_vs_E_N0():
    E_N0s=np.arange(-10,21)
    SNRs=10**(E_N0s/10)

    BERs=0.5*erfc(np.sqrt(SNRs))
    plt.title("BER vs E/N0")
    plt.semilogy(E_N0s,BERs,'b.-')
    plt.xlabel('E/N0 (dB)')
    plt.ylabel('Bit Error Rate')
    plt.show()

draw_BER_vs_E_N0()