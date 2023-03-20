from utilities import np, plt, rand_exp, UniformQuantizer, UniformDequantizer, calc_SNR
# requirment 5
samples = np.array(rand_exp(10000, 5))
# signs = np.array( [random.choice([-1, 1]) for _ in range(10000)])
# samples = samples * signs
# print(len(samples[samples<0]))
SNR_theoretical = np.zeros(7)
SNR_experimental = np.zeros(7)

for i in range(2, 9):
    # calculate quntization level for each samples (midrise)
    q_levels = UniformDequantizer(UniformQuantizer(samples, i, 5, 0), i, 5, 0)
    SNR_theoretical[i - 2], SNR_experimental[i -
                                             2] = calc_SNR(samples, q_levels, i, 0, 5)

# ploting
plt.figure(figsize=(8, 7))
plt.plot(range(2, 9), 10*np.log10(SNR_theoretical), label="SNR theoretical")
plt.plot(range(2, 9), 10*np.log10(SNR_experimental),
         "--", label="SNR experimental")

plt.xlabel('x - n bits')
plt.ylabel('y - SNR')
plt.title('SNR')

plt.legend()
plt.grid()
plt.show()
