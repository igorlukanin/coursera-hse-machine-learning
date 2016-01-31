from __future__ import print_function

import pandas

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

passengerClass = data['Pclass']
passengerClassCounts = passengerClass.value_counts()
firstClassCount = passengerClassCounts[1]
firstClassCountRatio = 1. * firstClassCount / passengerClass.size

print()
print('Ratio of 1st class passengers:', firstClassCountRatio)

file = open('04-result-03.txt', 'w')
print('{0:.2f}'.format(firstClassCountRatio * 100), file=file, sep='', end='')
file.close()
