import numpy as np
from matplotlib import pyplot as plt
T=6
N0=1
def random_bit_stream_generator():
    return np.random.randint(0, 2, 50)

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
    draw_filter(matched_filter,filter_type)
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

# Bit stream generator
bit_stream=random_bit_stream_generator()
print(bit_stream)

# Pulse shaping filter
after_pulse_shape=pulse_shaping_filter(bit_stream,T)
draw(after_pulse_shape,"Input Signal")

# Channel
after_channel=channel(after_pulse_shape)
draw(after_channel,"Channel")


def filter_with(filter_type):
    # Receive filter
    after_receive_filter=receive_filter(after_channel,T,filter_type)
    draw_after_matched_filter(after_receive_filter,"Signal After Receive Filter")

    # Sampling
    after_sampling=sampling(after_receive_filter,T)
    print(after_sampling)

    # Decode
    output=decode(after_sampling)

    # Error
    error=np.sum(output!=bit_stream)/len(bit_stream)


    output=pulse_shaping_filter(output,T)
    draw(output,"Output")

    print("BER: ",error)

filter_types=["matched","impulse","linear"]
for filter_type in filter_types:
    filter_with(filter_type)

