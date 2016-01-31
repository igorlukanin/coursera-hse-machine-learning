import numpy as np

print
print 'Random matrix'

matrix = np.random.normal(loc=1, scale=10, size=(1000, 50))

print matrix

print
print 'Normalized matrix'

mean = np.mean(matrix, axis=0)
std = np.std(matrix, axis=0)

normalizedMatrix = (matrix - mean) / std

print normalizedMatrix