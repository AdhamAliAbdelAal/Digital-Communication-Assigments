from utilities import rand_exp, UniformQuantizer, UniformDequantizer, getPower, calc_SNR, getMax, np, plt, normalizeSignal, compressing_block, expanding_block
print('Requirement 9')
# generating the samples
generatedSamples = rand_exp(100000)
# getting the x_max for the samples
xmax = getMax(generatedSamples)
noOfBits = np.arange(2, 9, 1)
L = 2 ** noOfBits
noOfBits = noOfBits[:, np.newaxis]
u = [0, 5, 100, 200]
colors = ['red', 'green', 'blue', 'yellow',
          'black', 'orange', 'purple', 'pink']
# applying the compression block
normalizedInput = normalizeSignal(generatedSamples, xmax)
for i in range(0, 4):
    compressed = compressing_block(
        generatedSamples, u[i])  # u = 5 only for now
    # passing the sample through the quantizer (apply it on one sample with n = 8)
    quantized = UniformQuantizer(generatedSamples, noOfBits, xmax, 0)

    # passing the sample through the dequantizer
    dequantized = UniformDequantizer(quantized, noOfBits, xmax, 0)

    # expanding the result
    # denormalizing the result
    dequantized *= xmax
    expanded = expanding_block(dequantized, u[i])

    # evaluating the SNR
    signalPower = getPower(generatedSamples)
    noise = generatedSamples - expanded
    noisePower = getPower(noise, axis=1)
    actualSNR = calc_SNR(signalPower=signalPower, noisePower=noisePower)
    actualSNR = 10 * np.log10(actualSNR)
    theoriticalSNR = ((3 * (L**2) / (np.log(1 + u[i])) ** 2))
    theoriticalSNR = 10 * np.log10(theoriticalSNR)

    # plotting the results on the same plot
    plt.plot(noOfBits, theoriticalSNR,
             color=colors[i], label='u = ' + str(u[i]))
    plt.plot(noOfBits, actualSNR,
             color=colors[(i + 4) % 8], label='Actual u =' + str(u[i]))
plt.title('Actual SNR(blue) vs Theoretical SNR(red) (in dB)')
plt.legend(loc='upper left')
plt.show()
