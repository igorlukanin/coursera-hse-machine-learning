from __future__ import print_function
from pandas import read_csv
from numpy import corrcoef
from sklearn.decomposition import PCA


data = read_csv('close-prices.csv')
data = data[[x for x in data.columns if x != 'date']]

pca = PCA(n_components=10)
pca.fit(data)

print()
print('Variance ratios:')
print(pca.explained_variance_ratio_)


sum = 0
i = 0

while sum <= 0.9:
    sum += pca.explained_variance_ratio_[i]
    i += 1

print()
print('Components count for 90 % variance:', i)


djia = read_csv('djia_index.csv')['^DJI'].tolist()

component_0 = map(lambda x: x[0], pca.fit_transform(data))
c = corrcoef(component_0, djia)[0][1]

print()
print('Correlation coefficient:', c)


max = 0
j = 0

for x in range(0, len(pca.components_[0])):
    if max < pca.components_[0][x]:
        max = pca.components_[0][x]
        j = x

print('Max weight:', max)
print('Company:   ', data.columns[j])


file = open('01-result.txt', 'w')
print(i, file=file, sep='', end='')
file.close()

file = open('02-result.txt', 'w')
print('{0:.2f}'.format(c), file=file, sep='', end='')
file.close()

file = open('03-result.txt', 'w')
print(data.columns[j], file=file, sep='', end='')
file.close()