from utilities import np, runAll, bits
m = 1
val = 6.0
x = np.arange(-val, val+0.01, 0.01)
runAll(x, bits, val, m)
