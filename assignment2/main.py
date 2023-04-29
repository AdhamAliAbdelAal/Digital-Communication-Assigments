import numpy as np
from matplotlib import pyplot as plt
T=3

def random_bit_stream_generator():
    temp= np.random.randint(0, 2, 10)
    temp[temp==0]=-1
    return temp

def pulse_shaping_filter(bit_stream,T):
    return np.repeat(bit_stream, T)

def channel(signal):
    return signal+np.random.normal(0, 1, len(signal))

def receive_filter(signal,T):
    matched_filter=np.ones(T)
    after_matched_filter=np.zeros(0)
    for i in range(0,len(signal),T):
        convolution=np.convolve(signal[i:i+T],matched_filter)
        print(convolution.shape)
        after_matched_filter=np.concatenate((after_matched_filter,convolution))
    print(after_matched_filter.shape)
    return after_matched_filter

def sampling(signal,T):
    size=len(signal)//(2*T-1)
    print(size,signal.shape)
    samples=np.zeros(size)
    for i in range(0,size):
        samples[i]=signal[i*(2*T-1)+T]>0
    samples[samples==0]=-1
    return samples.astype(int)



bit_stream=random_bit_stream_generator()
x=pulse_shaping_filter(bit_stream,T)

signal=channel(x)
print(signal)

time=np.arange(0,len(signal),1)

# plt.step(time,x)
# plt.show()
after_matched_filter=receive_filter(signal,T)
plt.plot(time,signal)
plt.show()
time=np.arange(0,len(after_matched_filter),1)
plt.plot(time,after_matched_filter)
plt.show()
output=sampling(after_matched_filter,T)
print(output,bit_stream)
print(np.sum(output!=bit_stream))


# print(np.random.normal(0, 1, len(signal)))