import numpy as np
import matplotlib.pyplot as plt


def UniformQuantizer(in_val, n_bits, xmax, m):
    delta = 2 * xmax / (2 ** n_bits)
    indices = (np.floor(in_val / delta) if m ==  # depending on m, we either mid-raise or mid-tread
               0 else np.round(in_val / delta)-1) + np.round(xmax/delta).astype(int)
    #print(indices, np.round((xmax-delta/2)/delta))
    indices[indices < 0] = 0
    return indices


# it returns the signal from the encoded signal to the original amplitudes after quantization.
def UniformDequantizer(q_ind, n_bits, xmax, m):
    delta = 2 * xmax / (2 ** n_bits)
    if(m == 1):
        q_ind = q_ind + 1
    out_val = (q_ind - xmax/delta) * delta
    return out_val if m == 1 else out_val + delta/2


def compressing_block(in_val, u):
    out_val = np.sign(in_val) * np.log(1 + u * np.abs(in_val)) / np.log(1 + u)
    return out_val


def expanding_block(in_val, u):
    out_val = np.sign(in_val) * (1 / u) * ((1 + u) ** np.abs(in_val) - 1)
    return out_val


x = np.random.exponential(1.0, 10000)
xmax = np.max(x)
array_probabilities = np.ones(10000)*0.5
out = (np.random.random(size=len(array_probabilities))
       > array_probabilities).astype(int)
out[out == 0] = -1
x = x*out
normalizedInput = x/xmax
u = np.array([0.000000000001, 5.0, 100.0, 200.0])
n_bits = np.array([2, 3, 4, 5, 6, 7, 8])
n_bits = n_bits[:, np.newaxis]
colors = ['red', 'blue', 'green', 'black']
for i in range(0, 4):
    compressed = compressing_block(normalizedInput, u[i])
    ymax = np.max(np.abs(compressed))
    quantized = UniformQuantizer(compressed, n_bits, ymax, 0)
    dequantized = UniformDequantizer(quantized, n_bits, ymax, 0)
    expanded = expanding_block(dequantized, u[i])
    denormalized_expanded = expanded * xmax
    signal_power = np.mean(x**2)
    noise = x - denormalized_expanded
    noise_power = np.mean(noise**2, axis=1)
    actual_snr = signal_power/noise_power
    actual_snr = 10 * np.log10(actual_snr)
    theoritical_snr = (3 * (2**n_bits)**2 / (np.log(1.000001 + u[i])) ** 2)
    theoritical_snr = 10 * np.log10(theoritical_snr)
    plt.plot(n_bits, theoritical_snr,
             color=colors[i % 4], label='Theo u = ' + str(int(u[i])))
    plt.plot(n_bits, actual_snr,
             color=colors[i % 4], linestyle='dashed', label='Actual u = ' + str(int(u[i])))
plt.title('SNRs')
plt.legend()
plt.show()
print(compressed)
