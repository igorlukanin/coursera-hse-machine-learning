from __future__ import print_function
from pandas import read_csv
from math import exp, sqrt
from sklearn.metrics import roc_auc_score


def e_dist(a, b):
    sum = 0

    for i in range(0, len(a)):
        delta = a[i] - b[i]
        sum += delta * delta

    return sqrt(sum)


def step(x, y, k, C, w, j):
    sum1 = 0
    for i in range(0, len(x)):
        sum2 = 0
        for ii in range(0, len(w)):
            sum2 += w[ii] * x[i][ii]

        sum1 += y[i] * x[i][j] * (1 - 1 / (1 + exp(-y[i] * sum2)))

    return w[j] + (k / len(x)) * sum1 - k * C * w[j]


def iterate(x, y, k, C):
    w = [0, 0]
    new_w = [0, 0]

    for i in range(0, 10000):
        # if i % 100 == 0:
            # print('Step', i, '- w:', w)

        for i in range(0, len(w)):
            new_w[i] = step(x.values, y, k, C, w, i)

        if e_dist(w, new_w) < 0.00001:
            break

        w = new_w[:]

    return w


def probability(w, x):
    sum = 0

    for i in range(0, len(x)):
        sum -= w[i] * x[i]

    return 1 / (1 + exp(sum))


data = read_csv('logistic.csv', header=None)

X = data[data.columns[1:]]
y = data[data.columns[0]]

w_L0 = iterate(X, y, k=0.1, C=0)
roc_auc_L0 = roc_auc_score(y, map(lambda x: probability(w_L0, x), X.values))

w_L2 = iterate(X, y, k=0.1, C=10)
roc_auc_L2 = roc_auc_score(y, map(lambda x: probability(w_L2, x), X.values))

print()
print('C=0,  w:', w_L0, ', AUC-ROC:', roc_auc_L0)
print('C=10, w:', w_L2, ', AUC-ROC:', roc_auc_L2)


file = open('01-result.txt', 'w')
print('{0:.3f} {1:.3f}'.format(roc_auc_L0, roc_auc_L2), file=file, sep='', end='')
file.close()