from __future__ import print_function

import pandas

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

sibSp = data['SibSp']
parCh = data['Parch']
correlation = sibSp.corr(parCh)

print()
print('Siblings/spouses to parents/children correlation:', correlation)

file = open('04-result-05.txt', 'w')
print('{0:.2f}'.format(correlation), file=file, sep='', end='')
file.close()
