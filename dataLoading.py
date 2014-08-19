import pandas as p
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
#from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
import resource

#resource.setrlimit(resource.RLIMIT_AS, (10000 * 1048576L, -1L))

paths = ['train.csv', 'test.csv']
t = p.read_csv(paths[0])
t2 = p.read_csv(paths[1])

tfidf = TfidfVectorizer(max_features=10000, strip_accents='unicode', analyzer='word')
tfidf.fit(t['tweet'])
X = tfidf.transform(t['tweet'])
print(X.shape)
X = X.toarray()
test = tfidf.transform(t2['tweet'])
y = np.array(t.ix[:,4])



#clf = RandomForestRegressor(verbose = 2)
clf = RandomForestRegressor(verbose = 2)
print("training classifier")
clf.fit(X,y)
test_prediction = clf.predict(test)

#RMSE:
print 'Train error: {0}'.format(np.sqrt(np.sum(np.array(np.array(clf.predict(X))-y)**2)/ (X.shape[0]*24.0)))