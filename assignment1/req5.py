from utilities import np, UniformDequantizer, UniformQuantizer, DEBUG, plt
m = 0
val = 5
n = np.arange(2, 9, 1)
bits = n[:, np.newaxis]
x= np.random.exponential(1.0,10000)
xmax = np.max(x)
array_probabilities=np.ones(10000)*0.5
out = (np.random.random(size=len(array_probabilities)) > array_probabilities).astype(int)
out[out==0]=-1
x=x*out

# Quantizer
indices = UniformQuantizer(x, bits, val, m)
# Dequantizer
dequantized = UniformDequantizer(indices, bits, val, m)

if DEBUG:
    plt.scatter(np.linspace(-5, 5.01, 10000), x, color='blue', s=0.1)
    plt.show()

    plt.scatter(np.linspace(-5, 5.01, 10000),
                dequantized, color='blue', s=0.1)
    plt.show()

p = np.mean(x**2)
# Calculate SNR
snr_theo = 10*np.log(p*3*(2**(2*n))/(val**2))  # in db

qp = np.mean((x-dequantized)**2, axis=1)
snr_act = 10*np.log(p/qp)  # in db

plt.plot(n, snr_theo, color='red', label='Theoretical')


plt.plot(n, snr_act, color='blue', label='Actual')
plt.title('SNRs')
plt.legend()
plt.show()
