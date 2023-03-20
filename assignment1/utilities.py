import random
import numpy as np
import matplotlib.pyplot as plt

DEBUG = False

# To run Mid-Rise, set requirement=1
# To run Mid-Tread, set requirement=2
# To run SNR, set requirement=3
####################
# For dynamic purpose, uncomment the following lines and comment the line below
# requirement = input(
#     "enter \n 1- Mid-Rise\n 2- Mid-Tread\n 3- SNR\n 5- nonUniform\n 6- Compander")
# requirement = int(requirement)
####################
requirement = 9

# it apply quantization to the signal and then encode it to levels.
# it returns indicies of the levels


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


bits = 3

# # Mid-Rise
# if(requirement == 1):  # 3a)
# m = 0
# val = 6.0
# x = np.arange(-val, val+0.01, 0.01)
# runAll(x, bits, val, m)

# # Mid-Tread
# if(requirement == 2):  # 3b)
#     m = 1
#     val = 6.0
#     x = np.arange(-val, val+0.01, 0.01)
#     runAll(x, bits, val, m)

# if(requirement == 3):  # 4a)
#     m = 0
#     val = 5
#     n = np.arange(2, 9, 1)
#     bits = n[:, np.newaxis]
#     x = np.random.uniform(-val, val+0.01, 10000)

#     # Quantizer
#     indices = UniformQuantizer(x, bits, val, m)
#     # Dequantizer
#     dequantized = UniformDequantizer(indices, bits, val, m)

#     if DEBUG:
#         plt.scatter(np.linspace(-5, 5.01, 10000), x, color='blue', s=0.1)
#         plt.show()

#         plt.scatter(np.linspace(-5, 5.01, 10000),
#                     dequantized, color='blue', s=0.1)
#         plt.show()

#     p = np.mean(x**2)
#     # Calculate SNR
#     snr_theo = 20*np.log(p*3*(2**(2*n))/(val**2))  # in db

#     qp = np.mean((x-dequantized)**2, axis=1)
#     snr_act = 20*np.log(p/qp)  # in db

#     plt.plot(n, snr_theo, color='red', label='Theoretical')
#     plt.title('Theoretical SNR')
#     plt.show()

#     plt.plot(n, snr_act, color='blue', label='Actual')
#     plt.title('Actual SNR')
#     plt.show()

##############################
# author : @AbdelazizSalah
# 5- Non-Uniform Quantizer
# 6- Compander
# date : 2023-3-13
##############################

#################################################################

# this is a utility function used to generate array
# contains negatives and positives with  probability 0.5


def generatingRandomNumbers(numOfRepeat):
    negatives = []
    for i in range(numOfRepeat):
        rndVal = random.randint(0, 1)
        if(rndVal == 1):
            negatives.append(1)
        else:
            negatives.append(-1)
    return negatives


def calc_SNR(samples, q_levels, n_bits, u, max_value):
    """
    calculate theoretical and experimental SNR
    Parameters:
    samples : vector of function samples 
    q_levels : vector of quantized samples
    n_bits : number of bits to decode the level
    u : expanding coefficient
    max_value : max value in samples
    return:
    SNR_theoretical : calculate the SNR with the equation
    SNR_experimental : calculate the SNR from the real values
    """
    error_diff = np.subtract(samples, q_levels)
    input_power = np.mean(np.square(samples))
    if(u > 0):
        SNR_theoretical = (3 * (1 << n_bits)**2) / (np.log(1 + u) ** 2)
    else:  # uniform quantization
        SNR_theoretical = (3 * (1 << n_bits)**2 * input_power) / max_value**2
    SNR_experimental = input_power / np.mean(np.square(error_diff))
    return SNR_theoretical, SNR_experimental


# this function is a utility function used to geenerate a random exponential signal.
# it take the size of the sample you want to generate.
def rand_exp(sz, max_value=5):
    # generatedSamples = np.random.exponential(
    #     scale=1.0, size=sz)  # 1/ e(X) = e(-X)
    # negatives = generatingRandomNumbers(sz)
    # generatedSamples *= negatives
    # return generatedSamples
    out_vect = np.random.exponential(1, sz)
    out_vect /= np.amax(out_vect)
    signs = np.random.choice([-1, 1], size=(sz,), p=[1./2, 1./2])
    out_vect *= (max_value * signs)
    return out_vect


def compressing_block(samples, u):
    """
    this expanding samples with log function to apply non-uniform quantizer
    Parameters:
    samples : vector of function samples
    max_value : max value in samples
    u : expanding coefficient
    return:
    y : vector of expanding samples
    """
    y = np.zeros_like(samples)
    if(u > 0):
        y = np.sign(samples) * np.log(1 + u *
                                      np.absolute(samples)) / np.log(1 + u)
    else:
        y = np.copy(samples)
    return y


def expanding_block(expanded_samples, u):
    """
    this compressing expanded samples to it's initial values
    Parameters:
    expanded_samples : vector of function expanded samples
    max_value : max value in samples
    u : expanding coefficient
    return:
    x : vector of real samples' value
    """
    x = np.zeros_like(expanded_samples)
    if(u > 0):
        x = ((1 + u) ** np.absolute(expanded_samples) - 1) / u
        x *= (np.sign(expanded_samples))
    else:
        x = np.copy(expanded_samples)
    return x

# def expanding_block(samples, u):
#     return (1 / u) * (np.power(1 + u, samples) - 1) if u != 0 else samples


# def compressing_block(samples, u):
#     return np.log(1 + u * np.abs(samples)) / np.log(1.0 + u) if u != 0 else samples


def normalizeSignal(signal, xmax):
    return signal / xmax


# def calc_SNR(signalPower, noisePower):
#     return signalPower / noisePower


def getPower(signal, axis=0):
    return np.mean(signal**2, axis=axis)


def getMax(generatedSamples):
    return abs(max(max(generatedSamples), min(generatedSamples), key=abs))
#################################################################
# this is a shit.
# def requirement5():
#     # generate random signal with 10000 samples with values =  e^(-x)
#     sz = 100000
#     generatedSamples = np.random.exponential(scale=1.0, size=sz)
#     negatives = generatingRandomNumbers(sz)
#     generatedSamples *= negatives
#     print(generatedSamples)
#     if DEBUG:
#         print("Generated Samples: ", generatedSamples)
#     # using xmax = 5 and m = 0
#     m = 0
#     # ageb el max absolut
#     print(min(generatedSamples))
#     print(max(generatedSamples))
#     xmax = abs(max(max(generatedSamples), min(generatedSamples), key=abs))
#     # number of bits -> req 4d) repeat for n = 2, 3, 4, 5, 6, 7, 8
#     n = np.arange(2, 9, 1)
#     bits = n[:, np.newaxis]

#     # Pass each sample through the quantizer
#     quantized = UniformQuantizer(generatedSamples, bits, xmax, m)
#     if DEBUG:
#         print("Quantized Samples: ", quantized, end='\n------------------\n')

#     # Pass it through the dequantizer
#     dequantized = UniformDequantizer(quantized, bits, xmax, m)

#     # evaluating the quantization error
#     quantizationError = dequantized - generatedSamples
#     if DEBUG:
#         print("Quantization Error: ", quantizationError,
#               end='\n------------------\n')

#     # evaluating the SNR in db -> 20 log10(Psignal/Pnoise)
#     signalPower = np.mean(generatedSamples**2)
#     noisePower = np.mean(quantizationError**2, axis=1)
#     if DEBUG:
#         print("Signal Power: ", signalPower, end='\n------------------\n')
#         print("Noise Power: ", noisePower, end='\n------------------\n')
#     # actualSNR = 20 * np.log10(signalPower / noisePower)
#     # theoriticalSNR = 20 * np.log10(signalPower * 3 * (2**(2*n)) / (xmax**2))
#     actualSNR = 10 * np.log10(signalPower / noisePower)
#     theoriticalSNR = 10 * np.log10(signalPower * 3 * (2**(2*n)) / (xmax**2))
#     if DEBUG:
#         print("Theoritical SNR: ", theoriticalSNR,
#               end='\n------------------\n')

#         print("Actual SNR: ", actualSNR, end='\n------------------\n')

#     # drawing the results on the same plot
#     plt.plot(n, theoriticalSNR, color='red', label='Theoretical')

#     plt.plot(n, actualSNR, color='blue', label='Actual')
#     plt.title('Actual SNR(blue) vs Theoretical SNR(red)')
#     plt.show()


# if(requirement == 5):
#     requirement5()

# # in this requirement we need to quantize the non-uniform signal using non-uniform u-law quantizer
# if(requirement == 6):
#     sz = 5
#     generatedSamples = np.random.exponential(scale=1.0, size=sz)
#     negatives = generatingRandomNumbers(sz)
#     generatedSamples *= negatives
#     if DEBUG:
#         print("Generated Samples: ", generatedSamples)
#     # using xmax = 5 and m = 0
#     m = 0
#     xmax = abs(max(max(generatedSamples), min(generatedSamples), key=abs))
#     # number of bits -> req 4d) repeat for n = 2, 3, 4, 5, 6, 7, 8
#     n = np.arange(2, 9, 1)
#     bits = n[:, np.newaxis]

#     # defining u values
#     # 0 is same as example 5
#     u = [0, 5, 100, 200]

#     # compressing the signal before inserting it to the quantizer
#     # u-law equation -> quantizerInput = (sign) ln(1 + u*normalized(signalInput))/ln(1+u)
#     # where normalized(signalInput) = signalInput/max(signalInput)
#     quantizerInput = [np.log(1 + u[i]*generatedSamples / np.max(generatedSamples)) / np.log(1+u[i])  # 2d
#                       if u[i] != 0 else generatedSamples for i in range(0, 4)]
#     if DEBUG:
#         # print(quantizerInput, sep='\n--------\n')
#         pass

#     # Pass each sample through the quantizer
#     quantized = [UniformQuantizer(  # 3d
#         quantizerInput[i], bits, xmax, m) for i in range(0, 4)]

#     # Pass it through the dequantizer
#     dequantized = UniformDequantizer(quantized, bits, xmax, m)

#     # expanding the result of the dequantizer
#     # u-law equation -> dequantizerOutput = (sign)(1+u)^dequantizerInput - 1 / u
#     # dequantizerOutput = [(1+u[i])**dequantized - 1 / u[i]
#     #                      if u[i] != 0 else dequantized for i in range(0, 4)]

#     dequantizerOutput = []
#     for i in range(0, 4):
#         dequantizerOutput.append((1+u[i])**dequantized[i] -
#                                  1 / u[i] if u[i] != 0 else dequantized[i])

#     if DEBUG:
#         print(dequantizerOutput, sep='\n--------\n')

#     normalizedInput = generatedSamples / np.max(generatedSamples)  # xmax
#     # TODO: this part still not correct, I need to draw on the same plot the results for different u values
#     # printing the results
#     # print(kiro.shape)
#     # for j in range(0, 7):
#     for i in range(0, 4):
#         plt.scatter(normalizedInput,
#                     dequantizerOutput[i][0], label='u = ' + str(u[i]))

#     plt.title('Dequantizer Output')
#     plt.legend()
#     plt.show()


# if (requirement == 7):
#     # requirment 6
#     plt.figure(figsize=(8, 7))
#     samples = rand_exp(10000, 1)
#     colors = ['red', 'blue', 'green', 'orange']
#     j = 0

#     for u in [0, 5, 100, 200]:
#         expanded_samples = expanding_block(samples, u)

#         SNR_theoretical = np.zeros(7)
#         SNR_experimental = np.zeros(7)

#         for i in range(2, 9):
#             # calculate quntization level for each samples (midrise)
#             expanded_levels = UniformDequantizer(
#                 UniformQuantizer(expanded_samples, i, 1, 0), i, 1, 0)
#             q_levels = compressing_block(expanded_levels, u)
#             SNR_theoretical[i - 2], SNR_experimental[i -
#                                                      2] = calc_SNR(samples, q_levels, i, u, 1)

#         # ploting
#         plt.plot(range(2, 9), 10*np.log10(SNR_theoretical),
#                  label="u = " + str(u), color=colors[j])
#         plt.plot(range(2, 9), 10*np.log10(SNR_experimental),
#                  "--",  color=colors[j])
#         j += 1

#     plt.xlabel('x - n bits')
#     plt.ylabel('y - SNR')
#     plt.title('SNR non-uniform quantizer')

#     plt.legend()
#     plt.grid()
#     plt.show()
