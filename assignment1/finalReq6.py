from utilities import rand_exp, UniformQuantizer, UniformDequantizer, getPower, calc_SNR, getMax, np, plt, compressing_block, expanding_block
# requirment 6
plt.figure(figsize=(8, 7))
samples = rand_exp(10000, 1)
colors = ['red', 'blue', 'green', 'orange']
j = 0

for u in [0, 5, 100, 200]:
    expanded_samples = compressing_block(samples, u)

    SNR_theoretical = np.zeros(7)
    SNR_experimental = np.zeros(7)

    for i in range(2, 9):
        # calculate quntization level for each samples (midrise)
        expanded_levels = UniformDequantizer(
            UniformQuantizer(expanded_samples, i, 1, 0), i, 1, 0)
        q_levels = expanding_block(expanded_levels, u)
        SNR_theoretical[i - 2], SNR_experimental[i -
                                                 2] = calc_SNR(samples, q_levels, i, u, 1)

    # ploting
    plt.plot(range(2, 9), 10*np.log10(SNR_theoretical),
             label="u = " + str(u), color=colors[j])
    plt.plot(range(2, 9), 10*np.log10(SNR_experimental),
             "--",  color=colors[j])
    j += 1

plt.xlabel('x - n bits')
plt.ylabel('y - SNR')
plt.title('SNR non-uniform quantizer')

plt.legend()
plt.grid()
plt.show()
