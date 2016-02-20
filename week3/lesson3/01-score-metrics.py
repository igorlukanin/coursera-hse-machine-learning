from __future__ import print_function
from pandas import read_csv
import sklearn.metrics as m


def calc_precision(y_true, probas_pred, recall_level):
    [precision, recall, thresholds] = m.precision_recall_curve(y_true, probas_pred)

    max_precision = 0

    for i in range(0, len(recall)):
        if (i == 0 or recall[i] >= recall_level) and max_precision < precision[i]:
            max_precision = precision[i]

    return max_precision


data = read_csv('classification.csv')

tp = 0
fp = 0
fn = 0
tn = 0

for i in range(0, len(data.index)):
    actu = data['true'][i]
    pred = data['pred'][i]

    tp += 1 if actu == 1 and pred == 1 else 0
    fn += 1 if actu == 1 and pred == 0 else 0
    fp += 1 if actu == 0 and pred == 1 else 0
    tn += 1 if actu == 0 and pred == 0 else 0

print()
print('TP:', tp)
print('FP:', fp)
print('FN:', fn)
print('TN:', tn)


accuracy = m.accuracy_score(data['true'], data['pred'])
precision = m.precision_score(data['true'], data['pred'])
recall = m.recall_score(data['true'], data['pred'])
f1 = m.f1_score(data['true'], data['pred'])

print()
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1:', f1)


scores = read_csv('scores.csv')

roc_auc_logreg = m.roc_auc_score(scores['true'], scores['score_logreg'])
roc_auc_svm = m.roc_auc_score(scores['true'], scores['score_svm'])
roc_auc_knn = m.roc_auc_score(scores['true'], scores['score_knn'])
roc_auc_tree = m.roc_auc_score(scores['true'], scores['score_tree'])

print()
print('AUC-ROC for LogReg:', roc_auc_logreg)
print('AUC-ROC for SVM:   ', roc_auc_svm)
print('AUC-ROC for kNN:   ', roc_auc_knn)
print('AUC-ROC for tree:  ', roc_auc_tree)


max_precision_logreg = calc_precision(scores['true'], scores['score_logreg'], recall_level=0.7)
max_precision_svm = calc_precision(scores['true'], scores['score_svm'], recall_level=0.7)
max_precision_knn = calc_precision(scores['true'], scores['score_knn'], recall_level=0.7)
max_precision_tree = calc_precision(scores['true'], scores['score_tree'], recall_level=0.7)

print()
print('Max precision for LogReg:', max_precision_logreg)
print('Max precision for SVM:   ', max_precision_svm)
print('Max precision for kNN:   ', max_precision_knn)
print('Max precision for tree:  ', max_precision_tree)


file = open('01-result-01.txt', 'w')
print(' '.join(str(x) for x in [tp, fp, fn, tn]), file=file, sep='', end='')
file.close()

file = open('01-result-02.txt', 'w')
print(' '.join('{0:.2f}'.format(x) for x in [accuracy, precision, recall, f1]), file=file, sep='', end='')
file.close()

file = open('01-result-03.txt', 'w')
print('score_logreg', file=file, sep='', end='')
file.close()

file = open('01-result-04.txt', 'w')
print('score_tree', file=file, sep='', end='')
file.close()