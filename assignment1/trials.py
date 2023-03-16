import random
import numpy as np


def generatingRandomNumbers():
    negatives = []
    for i in range(10):
        rndVal = random.randint(0, 1)
        if(rndVal == 1):
            negatives.append(1)
        else:
            negatives.append(-1)
    return negatives


generatedSamples = np.random.exponential(scale=1.0, size=10)
negatives = generatingRandomNumbers()
generatedSamples *= negatives
print(generatedSamples)
print(max(generatedSamples))
print(min(generatedSamples))
mx = abs(max(max(generatedSamples), min(generatedSamples), key=abs))
print(mx)
