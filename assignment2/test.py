from scipy.special import erfc
import numpy as np
from matplotlib import pyplot as plt

def draw_BER_vs_E_N0():
    E_N0s=np.arange(-10,21)
    BERs=0.5*erfc(np.sqrt(10**(E_N0s/20)))
    plt.title("BER vs E/N0")
    plt.plot(E_N0s,BERs)
    plt.xlabel("E/N0")
    plt.ylabel("BER")
    plt.show()

draw_BER_vs_E_N0()