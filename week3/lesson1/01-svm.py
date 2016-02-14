from __future__ import print_function
from pandas import read_csv
from sklearn.svm import SVC


data = read_csv('svm.txt', header=None)

result = data[data.columns[0]]
features = data[data.columns[1:]]


svc = SVC(C=100000, kernel='linear', random_state=241)
svc.fit(features, result)

print()
print('Support vector indices:', svc.support_)


file = open('01-result.txt', 'w')
print(' '.join(str(x + 1) for x in svc.support_), file=file, sep='', end='')
file.close()