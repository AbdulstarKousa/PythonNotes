#####################
# sklearn
#####################

'PreProcessing'
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV
from sklearn import preprocessing

'Models'
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier

'Others'
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay


preprocessing.scale(X) 
"""
Just a function
standardize columns 0 mean and 1 std
"""

standardizer = preprocessing.StandardScaler()
""" 
is a class supporting the Transformer API
Standardize features by removing the mean and scaling to unit variance 

X_train = standardizer.fit_transform(X_train)
X_test = standardizer.transform(X_test)
"""

normalizer = preprocessing.Normalizer().fit(Xtrain.T) #Calculate normalizer
"""
normalize to unite norm
# Xtrain = normalizer.transform(Xtrain.T).T # normalize training data
# Xtest = normalizer.transform(Xtest.T).T # normalize test data
"""

kf = KFold(n_splits=K)
""" 
Better way of doing k fold cross validation 
for i, (train_index, test_index) in enumerate(kf.split(X)):
"""

skf = StratifiedKFold(n_splits=K,shuffle=True)
"""
This cross-validation object is a variation of KFold that returns stratified folds. 
The folds are made by preserving the percentage of samples for each class.

for i, (train_index, test_index) in enumerate(skf.split(Xtrain, Ytrain)):
"""


param_grid = {'min_samples_leaf': range(1,50)}
cv_grid = GridSearchCV(estimator = dtree, param_grid = param_grid, cv = 5, verbose=2, n_jobs=-1)
"""
grid search model

cv_grid.fit(X, y)
cv_grid.best_estimator_
cv_grid.cv_results_
cv_grid.cv_results_['mean_test_score']
cv_grid.cv_results_['std_test_score']

"""

 neigh = KNeighborsClassifier(n_neighbors=k, weights = 'uniform', metric = 'euclidean') 
"""
Use Scikit KNN classifier, as you have already tried implementing it youself        

neigh.fit(X_train, y_train)
yhat = neigh.predict(X_test)
"""

LinearDiscriminantAnalysis().fit(X, y)


reg = linear_model.Lars(n_nonzero_coefs=j, fit_path = False, fit_intercept = False, verbose = True)
"""
lars

reg.fit(Xtrain,ytrain)
beta = reg.coef_.ravel()
"""

with warnings.catch_warnings(): # done to disable all the convergence warnings from elastic net
    warnings.simplefilter("ignore")
    model = linear_model.ElasticNetCV(cv=5, l1_ratio = alpha, alphas=lambdas, normalize=True).fit(X, y)

"""
elastic

model.alphas_
model.mse_path_.mean(axis=-1)
"""

model = LogisticRegression(penalty = 'l1', C = 1/lambda_, solver='liblinear')
model = model.fit(X_train, y_train)
"""
LogisticRegression
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

.fit(X_train, y_train)
.predict(X_test)
.coef_
"""

dtree = DecisionTreeRegressor()
dtree = DecisionTreeClassifier()
dtree = DecisionTreeClassifier(min_samples_leaf=min_sample_leaf_opt)
dtree=DecisionTreeClassifier(ccp_alpha=0.02040816326530612, criterion='gini')
""" 
create a decisiontreeregressor/classifier 
See week 05: 
for how to tune the parameter, MinLeaf value, using cross validation.
or to find the tree size through cost complexity pruning of the best estimator

dtree.fit(x, y)
"""

bagging = BaggingClassifier(DecisionTreeClassifier(), bootstrap=True, oob_score = True)
"""
bagged trees
"""

from sklearn.tree import plot_tree
plot_tree(dtree,feature_names = feature_names,filled = True)
"""
to plot tree

A little description of the information at each plotted node
1. row: The condition
2. row: The impurity score of the node
3. row: The number of observations at this node
4. row: The number of samples for each class at this node
5. row: The class by majority voting
"""

sk.metrics.log_loss
"""
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
"""


clf = RandomForestClassifier(bootstrap=True, oob_score=True, criterion = 'gini',random_state=0)
"""
"""


s