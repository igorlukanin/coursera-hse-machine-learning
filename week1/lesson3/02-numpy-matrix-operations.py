import numpy as np

print
print 'Matrix'

matrix = np.array([[4, 5, 0],
                   [1, 9, 3],
                   [5, 1, 1],
                   [3, 3, 3],
                   [9, 9, 9],
                   [4, 7, 1]])

print matrix

print
print 'Row sums'

rowSums = np.sum(matrix, axis=1)

print rowSums

print
print 'Rows with sum of elements > 10'

rowSumsMoreThan10 = np.nonzero(rowSums > 10)

print rowSumsMoreThan10