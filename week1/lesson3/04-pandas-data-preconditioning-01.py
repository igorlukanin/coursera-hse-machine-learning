from __future__ import print_function

import pandas

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

sexCounts = data['Sex'].value_counts()
maleCount = sexCounts['male']
femaleCount = sexCounts['female']

print()
print('Male:   ', maleCount)
print('Female: ', femaleCount)

file = open('04-result-01.txt', 'w')
print(maleCount, ' ', femaleCount, file=file, sep='', end='')
file.close()
