import numpy as np
import matplotlib.pyplot as plt

DEBUG = True

# To run Mid-Rise, set requirement=1
# To run Mid-Tread, set requirement=2
# To run SNR, set requirement=3
####################
# For dynamic purpose, uncomment the following lines and comment the line below
# requirement = input(
#     "enter \n 1- Mid-Rise\n 2- Mid-Tread\n 3- SNR\n 5- nonUniform\n 6- Compander")
# requirement = int(requirement)
####################
requirement = 6

# it apply quantization to the signal and then encode it to levels.
# it returns indicies of the levels


def UniformQuantizer(in_val, n_bits, xmax, m):
    delta = 2 * xmax / (2 ** n_bits)
    indices = ((np.floor(in_val / delta) if m ==  # depending on m, we either mid-raise or mid-tread
               0 else np.round(in_val / delta)) + xmax/delta).astype(int)
    return indices


# it returns the signal from the encoded signal to the original amplitudes after quantization.
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
if(requirement == 1):  # 3a)
    m = 0
    val = 6.0
    x = np.arange(-val, val+0.01, 0.01)
    runAll(x, bits, val, m)

# Mid-Tread
if(requirement == 2):  # 3b)
    m = 1
    val = 6.0
    x = np.arange(-val, val+0.01, 0.01)
    runAll(x, bits, val, m)

if(requirement == 3):  # 4a)
    m = 0
    val = 5
    n = np.arange(2, 9, 1)
    bits = n[:, np.newaxis]
    x = np.random.uniform(-val, val+0.01, 10000)

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
    snr_theo = 20*np.log(p*3*(2**(2*n))/(val**2))  # in db

    qp = np.mean((x-dequantized)**2, axis=1)
    snr_act = 20*np.log(p/qp)  # in db

    plt.plot(n, snr_theo, color='red', label='Theoretical')
    plt.title('Theoretical SNR')
    plt.show()

    plt.plot(n, snr_act, color='blue', label='Actual')
    plt.title('Actual SNR')
    plt.show()

##############################
# author : @AbdelazizSalah
# 5- Non-Uniform Quantizer
# 6- Compander
# date : 2023-3-13
##############################


def requirement5():
    # generate random signal with 10000 samples with values =  e^(-x)
    generatedSamples = np.exp(-np.random.uniform(-5, 5.01, 10000))
    if DEBUG:
        print("Generated Samples: ", generatedSamples)
    # using xmax = 5 and m = 0
    m = 0
    xmax = 5
    # number of bits -> req 4d) repeat for n = 2, 3, 4, 5, 6, 7, 8
    n = np.arange(2, 9, 1)
    bits = n[:, np.newaxis]

    # Pass each sample through the quantizer
    quantized = UniformQuantizer(generatedSamples, bits, xmax, m)
    if DEBUG:
        print("Quantized Samples: ", quantized, end='\n------------------\n')

    # Pass it through the dequantizer
    dequantized = UniformDequantizer(quantized, bits, xmax, m)

    # evaluating the quantization error
    quantizationError = dequantized - generatedSamples
    if DEBUG:
        print("Quantization Error: ", quantizationError,
              end='\n------------------\n')

    # evaluating the SNR in db -> 20 log10(Psignal/Pnoise)
    signalPower = np.mean(generatedSamples**2)
    noisePower = np.mean(quantizationError**2, axis=1)
    if DEBUG:
        print("Signal Power: ", signalPower, end='\n------------------\n')
        print("Noise Power: ", noisePower, end='\n------------------\n')
    actualSNR = 20 * np.log10(signalPower / noisePower)
    theoriticalSNR = 20 * np.log10(signalPower * 3 * (2**(2*n)) / (xmax**2))
    if DEBUG:
        print("Theoritical SNR: ", theoriticalSNR,
              end='\n------------------\n')
        print("Actual SNR: ", actualSNR, end='\n------------------\n')

    # drawing the results on the same plot
    plt.plot(n, theoriticalSNR, color='red', label='Theoretical')

    plt.plot(n, actualSNR, color='blue', label='Actual')
    plt.title('Actual SNR(blue) vs Theoretical SNR(red)')
    plt.show()


if(requirement == 5):
    requirement5()

# in this requirement we need to quantize the non-uniform signal using non-uniform u-law quantizer
if(requirement == 6):
    generatedSamples = np.exp(-np.random.uniform(-5, 5.01, 5))
    if DEBUG:
        print("Generated Samples: ", generatedSamples)
    # using xmax = 5 and m = 0
    m = 0
    xmax = 5
    # number of bits -> req 4d) repeat for n = 2, 3, 4, 5, 6, 7, 8
    n = np.arange(2, 9, 1)
    bits = n[:, np.newaxis]

    # defining u values
    # 0 is same as example 5
    u = [0, 5, 100, 200]

    # compressing the signal before inserting it to the quantizer
    # u-law equation -> quantizerInput = (sign) ln(1 + u*normalized(signalInput))/ln(1+u)
    # where normalized(signalInput) = signalInput/max(signalInput)
    quantizerInput = [np.log(1 + u[i]*generatedSamples / np.max(generatedSamples)) / np.log(1+u[i])
                      if u[i] != 0 else generatedSamples for i in range(0, 4)]
    if DEBUG:
        # print(quantizerInput, sep='\n--------\n')
        pass

    # Pass each sample through the quantizer
    quantized = [UniformQuantizer(
        quantizerInput[i], bits, xmax, m) for i in range(0, 4)]

    # Pass it through the dequantizer
    dequantized = UniformDequantizer(quantized, bits, xmax, m)

    # expanding the result of the dequantizer
    # u-law equation -> dequantizerOutput = (sign)(1+u)^dequantizerInput - 1 / u
    dequantizerOutput = [(1+u[i])**dequantized - 1 / u[i]
                         if u[i] != 0 else dequantized for i in range(0, 4)]
    if DEBUG:
        print(dequantizerOutput, sep='\n--------\n')

    normalizedInput = generatedSamples / np.max(generatedSamples)
    # TODO: this part still not correct, I need to draw on the same plot the results for different u values
    # printing the results
    for i in range(0, 4):
        plt.plot(normalizedInput,
                 dequantizerOutput[i][0][0], label='u = ' + str(u[i]))

    plt.title('Dequantizer Output')
    plt.legend()
    plt.show()
