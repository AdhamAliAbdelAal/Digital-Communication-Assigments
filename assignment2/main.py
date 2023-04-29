import numpy as np
from matplotlib import pyplot as plt
T=10

def random_bit_stream_generator():
    temp= np.random.randint(0, 2, 3)
    temp[temp==0]=-1
    return temp

def pulse_shaping_filter(bit_stream,T):
    return np.repeat(bit_stream, T)

def channel(signal):
    return signal+np.random.normal(0, .01, len(signal))

def receive_filter(signal,T):
    matched_filter=np.ones(T)
    after_matched_filter=np.convolve(signal,matched_filter)
    return after_matched_filter

def sampling(signal,T):
    size=len(signal)//T
    samples=np.zeros(size)
    for i in range(0,size):
        samples[i]=signal[(i+1)*T]
    return np.sign(samples)



bit_stream=random_bit_stream_generator()
print(bit_stream)
x=pulse_shaping_filter(bit_stream,T)
time=np.arange(0,len(x),1)
plt.title("Input Signal")
plt.step(time,x)
plt.show()

signal=channel(x)
# print(signal)
plt.title("Signal after channel")
plt.plot(signal)
plt.show()

time=np.arange(0,len(signal),1)

# plt.step(time,x)
# plt.show()
after_matched_filter=receive_filter(signal,T)
after_matched_filter=after_matched_filter/np.max(after_matched_filter)
print(after_matched_filter)

print(len(after_matched_filter))


# plt.plot(time,signal)
# plt.show()
time=np.arange(0,len(after_matched_filter),1)
plt.title("Signal after matched filter")
plt.plot(time,after_matched_filter)
plt.show()
output=sampling(after_matched_filter,T)
print(output)
# print(output,bit_stream)
print(np.sum(output!=bit_stream))


# print(np.random.normal(0, 1, len(signal)))