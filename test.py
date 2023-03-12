import numpy as np
import matplotlib.pyplot as plt

DEBUG = False


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
val = 6.0
x = np.arange(-val, val+0.01, 0.01)

# Mid-Rise
m = 0
runAll(x, bits, val, m)

# Mid-Tread
m = 1
runAll(x, bits, val, m)

