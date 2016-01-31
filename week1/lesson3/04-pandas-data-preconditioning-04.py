from __future__ import print_function

import pandas

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

age = data['Age']

meanAge = age.mean()
medianAge = age.median()

print('Mean age:  ', meanAge)
print('Median age:', medianAge)

file = open('04-result-04.txt', 'w')
print('{0:.2f}'.format(meanAge), ' ', '{0:.2f}'.format(medianAge), file=file, sep='', end='')
file.close()
