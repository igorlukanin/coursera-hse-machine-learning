import numpy as np

print
print 'Matrices'

matrix1 = np.eye(3)
matrix2 = np.eye(3)

print matrix1
print matrix2

print
print 'Joined matrices'

matrix = np.vstack((matrix1, matrix2))

print matrix