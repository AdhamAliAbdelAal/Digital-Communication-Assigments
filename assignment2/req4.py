import numpy as np
from matplotlib import pyplot as plt
from scipy.special import erfc

E_N0s=np.arange(-10,21)
SNRs=10**(E_N0s/10)
T=3

def random_bit_stream_generator():
    return np.random.randint(0, 2, 10000)

def pulse_shaping_filter(bit_stream,T):
    temp= np.repeat(bit_stream, T)
    temp[temp==0]=-1
    return temp

def channel(signal):
    return signal+np.random.normal(0, N0, len(signal))

def draw(signal,title):
    plt.title(title)
    if(title=="Input Signal" or title=="Output"):
        plt.plot(signal)
    else:
        plt.plot(signal)
    plt.show()


def draw_after_matched_filter(signal,title):
    size=len(signal)//T
    samples=np.zeros(len(signal))
    for i in range(0,size):
        samples[(i+1)*T-1]=signal[(i+1)*T-1]
    samples[samples==0]=np.nan
    plt.title(title)
    plt.stem(samples,linefmt='grey', markerfmt='D',label="Samples")
    plt.plot(signal,c="g",label="Signal")
    plt.legend(loc="upper right")
    plt.show()

def draw_filter(filter,title):
    plt.title(title.capitalize()+" Filter")
    if(title=="impulse"):
        plt.stem(filter)
    else:
        plt.plot(filter)
    plt.show()

def receive_filter(signal,T,filter_type="matched"):
    if filter_type=="matched":
        matched_filter=np.ones(T)
    elif filter_type=="impulse":
        matched_filter= np.zeros(T)
        matched_filter[0]=1
    elif filter_type=="linear":
        matched_filter=np.linspace(0,1,T)*np.square(3)
    after_matched_filter=np.convolve(signal,matched_filter)
    return after_matched_filter

def sampling(signal,T):
    size=len(signal)//T
    samples=np.zeros(size)
    for i in range(0,size):
        samples[i]=signal[(i+1)*T-1]
    return np.sign(samples)

def decode(signal):
    signal[signal<0]=0
    return signal.astype(int)

def BER():
    return 0.5*erfc(np.sqrt(1/N0))

# Bit stream generator
bit_stream=random_bit_stream_generator()

# Pulse shaping filter
after_pulse_shape=pulse_shaping_filter(bit_stream,T)


def clac_error(filter,after_channel):
    # Receive filter
    after_receive_filter=receive_filter(after_channel,T,filter)

    # Sampling
    after_sampling=sampling(after_receive_filter,T)

    # Decode
    output=decode(after_sampling)
    
    # Error
    error=np.sum(output!=bit_stream)

    return error/len(bit_stream)

matched_filter_BERs=[]
impulse_filter_BERs=[]
linear_filter_BERs=[]
for snr in SNRs:
    N0=1/snr

    # Channel
    after_channel=channel(after_pulse_shape)

    error_matched=clac_error("matched",after_channel)
    error_impulse=clac_error("impulse",after_channel)
    error_linear=clac_error("linear",after_channel)

    matched_filter_BERs.append(error_matched)
    impulse_filter_BERs.append(error_impulse)
    linear_filter_BERs.append(error_linear)

# print(matched_filter_BERs)
# print(impulse_filter_BERs)
# print(linear_filter_BERs)
BERs=0.5*erfc(np.sqrt(SNRs))
print(BERs)

fig,(ax1,ax2)=plt.subplots(1,2)

ax1.set_title("Theoretical")
ax1.semilogy(E_N0s,BERs,'b.-')
ax1.set(xlabel='E/N0 (dB)', ylabel='Bit Error Rate Theoretical')


ax2.set_title("Actual")
ax2.semilogy(E_N0s,matched_filter_BERs,'b.-',label="Matched Filter")
ax2.semilogy(E_N0s,impulse_filter_BERs,'r.-',label="Impulse Filter")
ax2.semilogy(E_N0s,linear_filter_BERs,'g.-',label="Linear Filter")
ax2.legend()
ax2.set(xlabel='E/N0 (dB)', ylabel='Bit Error Rate Actual')
plt.show()

