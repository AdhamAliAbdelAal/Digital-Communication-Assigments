import numpy as np
from matplotlib import pyplot as plt
T=3

def random_bit_stream_generator():
    return np.random.randint(0, 2, 3)

def pulse_shaping_filter(bit_stream,T):
    temp= np.repeat(bit_stream, T)
    temp[temp==0]=-1
    return temp

def channel(signal):
    return signal+np.random.normal(0, .1, len(signal))

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

def decode(signal):
    signal[signal<0]=0
    return signal

def draw(signal,title):
    plt.title(title)
    plt.stem(signal)
    plt.show()


# Bit stream generator
bit_stream=random_bit_stream_generator()
print(bit_stream)

# Pulse shaping filter
after_pulse_shape=pulse_shaping_filter(bit_stream,T)
draw(after_pulse_shape,"Input Signal")

# Channel
after_channel=channel(after_pulse_shape)
draw(after_channel,"Channel")

# Receive filter
after_receive_filter=receive_filter(after_channel,T)
draw(after_receive_filter,"Receive Filter")

# Sampling
after_sampling=sampling(after_receive_filter,T)
print(after_sampling)

# Decode
output=decode(after_sampling)
draw(output,"Output")

print(output)
# Error
error=np.sum(output!=bit_stream)
print("Error: ",error)