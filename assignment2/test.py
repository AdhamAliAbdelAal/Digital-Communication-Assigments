import numpy as np
from scipy.signal import convolve

# Define the pulse shape
p = np.array([1, 0, 1, 0, 1])

# Define the received signal
t = np.linspace(0, 10, 1000)
y = np.sin(2*np.pi*t) + np.random.normal(0, 0.1, 1000)  # Example received signal

# Generate the matched filter impulse response
matched_filter = np.flip(p)

# Compute the matched filter output
output = convolve(y, matched_filter, mode='valid')

# Plot the received signal, matched filter impulse response, and output
import matplotlib.pyplot as plt
fig, axs = plt.subplots(3, 1, figsize=(8, 8))
axs[0].plot(t, y)
axs[0].set_ylabel('Received Signal')
axs[1].stem(p)
axs[1].set_ylabel('Pulse Shape')
axs[2].plot(t[2:998], output)
axs[2].set_ylabel('Matched Filter Output')
axs[2].set_xlabel('Time (s)')
plt.show()
