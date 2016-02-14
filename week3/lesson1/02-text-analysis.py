from __future__ import print_function
from sklearn import datasets
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
import numpy as np
import heapq
from pprint import pprint

newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])

X = newsgroups.data
y = newsgroups.target

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X, y)

# grid = {'C': np.power(10.0, np.arange(-5, 6))}
# cv = KFold(len(y), n_folds=5, shuffle=True, random_state=241)

clf = SVC(kernel='linear', random_state=241, C=1.0)
clf.fit(X_tfidf, y)

# gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
# gs.fit(X_tfidf, y)

# for a in gs.grid_scores_:
#     print()
#     print('Mean validation score:', a.mean_validation_score)
#     print('Params:', a.parameters)

words = vectorizer.get_feature_names()
coefs = clf.coef_.toarray()[0]

coefs = map(lambda x: abs(x), coefs)

ten = heapq.nlargest(10, coefs)
ten_words = []

for coef in ten:
    for i in range(0, len(coefs)):
        if coefs[i] == coef:
            ten_words.append(words[i])

ten_words.sort()
print(ten_words)


file = open('02-result.txt', 'w')
print(' '.join(ten_words), file=file, sep='', end='')
file.close()