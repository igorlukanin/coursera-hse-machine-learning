from __future__ import print_function

import pandas

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

survived = data['Survived']
survivedCounts = survived.value_counts()
survivedCount = survivedCounts[1]
survivedRatio = 1. * survivedCount / survived.size

print()
print('Ratio of survived:', survivedRatio)

file = open('04-result-02.txt', 'w')
print('{0:.2f}'.format(survivedRatio * 100), file=file, sep='', end='')
file.close()
