import pandas as p
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import math

#resource.setrlimit(resource.RLIMIT_AS, (10000 * 1048576L, -1L))
print('reading files')
paths = ['train.csv', 'test.csv']
t = p.read_csv(paths[0])
t2 = p.read_csv(paths[1])

print('creating tfidf matrices')
tfidf = TfidfVectorizer(max_features=300, strip_accents='unicode', analyzer='word')
tfidf.fit(t['tweet'])
X = tfidf.transform(t['tweet'])
X = X.toarray()
test = tfidf.transform(t2['tweet']).toarray()
y = np.array(t.ix[:,4:])

print('cross validating models')
#rmse = make_scorer(math.sqrt(mean_squared_error))
clf=clf = Pipeline([
  #('feature_selection', LinearSVC()),
  ('regression', linear_model.RidgeCV(alphas=[math.sqrt(2) ** i for i in range(-20,20)]))
])

#clf = RandomForestRegressor(verbose = 1, n_jobs =-1)
mse_scorer = make_scorer(mean_squared_error)
scores = cross_validation.cross_val_score(clf, X,y,
                                          scoring=mse_scorer, 
                                          cv=8,
                                          verbose =1,
                                          n_jobs =-1)
print([math.sqrt(val) for val in scores])