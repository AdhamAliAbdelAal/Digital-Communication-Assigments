from utilities import rand_exp, UniformQuantizer, UniformDequantizer, getPower, calc_SNR, getMax, np, plt

# generating the samples
generatedSamples = rand_exp(100000)

# getting the x_max for the samples
xmax = getMax(generatedSamples)
noOfBits = np.arange(2, 9, 1)
L = 2 ** noOfBits
noOfBits = noOfBits[:, np.newaxis]

# passing the sample through the quantizer (apply it on one sample with n = 8)
quantized = UniformQuantizer(generatedSamples, noOfBits, xmax, 0)

# passing the sample through the dequantizer
dequantized = UniformDequantizer(quantized, noOfBits, xmax, 0)

# evaluating the SNR
signalPower = getPower(generatedSamples)
noise = generatedSamples - dequantized
noisePower = getPower(noise, axis=1)
actualSNR = calc_SNR(signalPower=signalPower, noisePower=noisePower)
actualSNR = 10 * np.log10(actualSNR)
theoriticalSNR = ((3 * (L**2) * signalPower / (xmax**2)))
theoriticalSNR = 10 * np.log10(theoriticalSNR)

# plotting the results on the same plot
plt.plot(noOfBits, theoriticalSNR, color='red', label='Theoretical')
plt.plot(noOfBits, actualSNR, color='blue', label='Actual')
plt.title('Actual SNR(blue) vs Theoretical SNR(red) (in dB)')
plt.legend()
plt.show()
