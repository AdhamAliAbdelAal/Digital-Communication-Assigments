import numpy as np
import matplotlib.pyplot as plt

DEBUG = False

# To run Mid-Rise, set requirement=1
# To run Mid-Tread, set requirement=2
# To run SNR, set requirement=3
requirement=3

def UniformQuantizer(in_val, n_bits, xmax, m):
    delta = 2 * xmax / (2 ** n_bits)
    indices = ((np.floor(in_val / delta) if m ==
               0 else np.round(in_val / delta)) + xmax/delta).astype(int)
    return indices


def UniformDequantizer(q_ind, n_bits, xmax, m):
    delta = 2 * xmax / (2 ** n_bits)
    out_val = (q_ind - xmax/delta) * delta
    return out_val if m == 1 else out_val + delta/2


def draw_quantizer(x, y):
    plt.step(x, y, where='post', color='blue')
    plt.grid(axis='x', color='0.95')
    plt.title('Indices')
    plt.show()


def draw_dequantizer(x, y):
    plt.step(x, y, where='post', color='blue')
    plt.plot(x, x, color='red')
    plt.grid(axis='x', color='0.95')
    plt.title('Dequantized Values')
    plt.show()

def runAll(val, bits, xmax, m):
    # Quantizer
    indices = UniformQuantizer(val, bits, xmax, m)
    if DEBUG:
        print(indices)
    draw_quantizer(val, indices)

    # Dequantizer
    dequantized = UniformDequantizer(indices, bits, xmax, m)
    if DEBUG:
        print(dequantized)
    draw_dequantizer(val, dequantized)

bits = 4

# Mid-Rise
if(requirement==1):
    m = 0
    val = 6.0
    x = np.arange(-val, val+0.01, 0.01)
    runAll(x, bits, val, m)

# Mid-Tread
if(requirement==2):
    m = 1
    val = 6.0
    x = np.arange(-val, val+0.01, 0.01)
    runAll(x, bits, val, m)

if(requirement==3):
    m = 0
    val=5
    n=np.arange(2,9,1)
    bits=n[:,np.newaxis]
    x=np.random.uniform(-val,val+0.01,10000)

    # Quantizer
    indices = UniformQuantizer(x, bits, val, m)
    # Dequantizer
    dequantized = UniformDequantizer(indices, bits, val, m)

    if DEBUG:
        plt.scatter(np.linspace(-5,5.01,10000), x, color='blue', s=0.1)
        plt.show()

        plt.scatter(np.linspace(-5,5.01,10000), dequantized, color='blue', s=0.1)
        plt.show()
    

    p = np.mean(x**2)
    # Calculate SNR
    snr_theo = 20*np.log(p*3*(2**(2*n))/(val**2))

    qp = np.mean((x-dequantized)**2, axis=1)
    snr_act= 20*np.log(p/qp)

    plt.plot(n, snr_theo, color='red', label='Theoretical')
    plt.title('Theoretical SNR')
    plt.show()

    plt.plot(n, snr_act, color='blue', label='Actual')
    plt.title('Actual SNR')
    plt.show()

