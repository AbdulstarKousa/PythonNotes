# Project Part:
# ====================== P 01: Get to know data
# ====================== P 02: Reg Methods
# ====================== P 03: Ensembles Methods
# ====================== P 04: Extra investigation

# Project Template:
# 1. Prepare Problem
# a) Load libraries
# b) Load dataset

# 2. Summarize Data
# a) Descriptive statistics
# b) Data visualizations

# 3. Prepare Data
# a) Data Cleaning
# b) Features Selection
# c) Data Transforms

# 4. Evaluate Algorithms
# a) Split-out validation dataset
# b) Test options and evaluation metric
# c) Spot Check Algorithms
# d) Compare Algorithms

# 5. Improve Accuracy
# a) Algorithm Tuning
# b) Ensembles

# 6. Finalize Model (both reg ens)
# a) Predictions on validation dataset
# b) Save model for later use
 
# Load libraries
import numpy as np
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


# ====================== P 01: Get to know data

# Load dataset
filename = 'case1Data.csv'
dataset  = read_csv(filename, sep=', ', engine='python')

# shape
print(dataset.shape)

# types
print(dataset.dtypes)

# info 
dataset.info()

# head
print(dataset.head(5))

# random sample of features for Summarizing Data
seed = 123
random_sample = dataset.sample(frac=0.10, replace=False, random_state=seed, axis=1)

# descriptions
set_option('precision', 1)
print(random_sample.describe())

# correlation
set_option('precision', 2)
print(random_sample.corr(method='pearson'))

# histograms
random_sample.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1, layout=(4,3), figsize=(10,10))
pyplot.show()

# density
random_sample.plot(kind='density', subplots=True, sharex=False, legend=True, fontsize=1, layout=(4,3), figsize=(10,10))
pyplot.show()

# box and whisker plots
random_sample.plot(kind='box', subplots=True, sharex=False, sharey=False, fontsize=10, layout=(4,3), figsize=(10,10))
pyplot.show()

# scatter plot matrix
scatter_matrix(random_sample, figsize=(10,10))
pyplot.show()

# correlation matrix
fig = pyplot.figure(figsize=(10,10))
ax = fig.add_subplot(111)
cax = ax.matshow(random_sample.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
ticks = np.arange(0,len(random_sample.columns),1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(random_sample.columns)
ax.set_yticklabels(random_sample.columns)
pyplot.show()



# ====================== P 02: Reg Methods
# Split-out XY dataset
X = dataset.iloc[:,1:len(dataset.columns)] # FS: X = X[Important_Features]
Y = dataset.iloc[:,0]

# Split-out validation dataset
validation_size = 0.20
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,test_size=validation_size, random_state=seed)

# Test options and evaluation metric
num_folds = 10
scoring = 'neg_mean_squared_error'

# Spot-Check Algorithms
"""
LinearRegression()
Lasso()
ElasticNet()
KNeighborsRegressor()
DecisionTreeRegressor()
"""

# Prepare data
numeric_transformer = StandardScaler()

categorical_transformer = Pipeline(
    steps = [
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
        ('enc', OneHotEncoder(handle_unknown= 'ignore', sparse=False)),
        ('stand', StandardScaler())
        ])

preprocessor = ColumnTransformer(
        transformers=[
        ('num', numeric_transformer, selector(dtype_include="float64")),
        ('cat', categorical_transformer, selector(dtype_include="object"))
        ])

# regressors 
pipelines = []
pipelines.append(('LR', Pipeline([('preprocessor', preprocessor),('lr',LinearRegression())])))
pipelines.append(('LASSO', Pipeline([('preprocessor', preprocessor),('lasso',Lasso())])))
pipelines.append(('EN', Pipeline([('preprocessor', preprocessor),('en',ElasticNet())])))
pipelines.append(('KNN', Pipeline([('preprocessor', preprocessor),('knn',KNeighborsRegressor())])))
pipelines.append(('CART', Pipeline([('preprocessor', preprocessor),('cart',DecisionTreeRegressor())])))

results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds,shuffle= True, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results, showmeans=True)
ax.set_xticklabels(names)
pyplot.show()

# LASSO Algorithm tuning
model = Pipeline([('preprocessor', preprocessor),('lasso',Lasso())])

param_grid = dict()
alphas = np.arange(-5, 5, 0.2)
param_grid['lasso__alpha'] = alphas[np.where(alphas!=0.0)] 

rkfold = RepeatedKFold(n_splits=num_folds, n_repeats=3, random_state=seed)
grid = GridSearchCV(model, param_grid, scoring=scoring , cv=rkfold, n_jobs=-1)

grid_result = grid.fit(X_train, Y_train)
print('MSE:     %.3f' % grid_result.best_score_)
print('Alpha:   %.3f' % grid_result.best_params_['lasso__alpha'])
"""
> MSE:     -1210.65
> Alpha:   3.80
"""

# prepare the model
transformer = preprocessor.fit(X_train)
transformedX = preprocessor.transform(X_train)
model = Lasso(alpha = 3.800) # FS: model = Lasso(alpha = 3)
model.fit(transformedX, Y_train)

# transform the validation dataset
transformedValidationX = transformer.transform(X_validation)
predictions = model.predict(transformedValidationX)

# Report:
print('MSE = ', mean_squared_error(Y_validation, predictions))
"""
> 784.49
"""



# ====================== P 03: Ensembles Methods

# Spot-Check Algorithms
"""
AdaBoostRegressor()
RandomForestRegressor()
ExtraTreesRegressor()
GradientBoostingRegressor()
"""

# ensembles
ensembles = []
ensembles.append(('AB', Pipeline([('preprocessor', preprocessor),('ab',AdaBoostRegressor())])))
ensembles.append(('RF', Pipeline([('preprocessor', preprocessor),('rf',RandomForestRegressor())])))
ensembles.append(('ET', Pipeline([('preprocessor', preprocessor),('et',ExtraTreesRegressor())])))
ensembles.append(('GBM', Pipeline([('preprocessor', preprocessor),('gbm',GradientBoostingRegressor())])))

results = []
names = []
for name, model in ensembles:
    kfold = KFold(n_splits=num_folds, shuffle= True, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Ensemble Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results, showmeans=True)
ax.set_xticklabels(names)
pyplot.show()

# RandomForestRegressor tuning:
"""
    - Explore Number of Features (max_features) = 0.3  (Heuristic)
    - Explore Number of Samples (max_samples)   = None (default: number of samples, bootstrap with replacement)
    - Explore Tree Depth (max_depth)            = None (default: arbitrary depth do not prune)
    - Explore Number of Trees (n_estimators) = ?
    - Explore Minimum Node Size (min_samples_leaf) = ? 
"""

# Random Grid Search for hyperparameters focus  
model = Pipeline([('preprocessor', preprocessor),('rf',RandomForestRegressor(max_features = 0.3, max_samples=None, max_depth= None))])

n_estimators = [x for x in np.arange(100, 1100, 100)]
min_samples_leaf = [1, 2, 4]
random_grid = {'rf__n_estimators': n_estimators,
                'rf__min_samples_leaf':min_samples_leaf }

grid  = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 20, cv = 3, verbose=2, n_jobs = -1)
grid_result = grid.fit(X_train, Y_train)

print('MSE:     %.3f' % grid_result.best_score_)
print('rf__random_best_params_:   %s' % grid_result.best_params_)
"""
> MSE:     0.394
> rf__random_best_params_:   {'rf__n_estimators': 400, 
                              'rf__min_samples_leaf': 4}
"""


# Grid Search with Cross Validation 
model = Pipeline([('preprocessor', preprocessor),('rf',RandomForestRegressor(max_features = 0.3, max_samples=None, max_depth= None))])

n_estimators = [x for x in np.arange(300, 550, 50)]
min_samples_leaf = [3, 4, 5]
param_grid = {'rf__n_estimators': n_estimators,
                'rf__min_samples_leaf':min_samples_leaf }

rkfold = RepeatedKFold(n_splits=num_folds, n_repeats=3)
grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = rkfold, n_jobs = -1, verbose = 2)
grid_result = grid.fit(X_train, Y_train)

print('MSE:     %.3f' % grid_result.best_score_)
print('rf_random_best_params_:   %s' % grid_result.best_params_)
""""
> MSE:     0.407
> rf_random_best_params_:   {'rf__min_samples_leaf': 3,
                             'rf__n_estimators': 500}
"""

# prepare the model
transformer = preprocessor.fit(X_train)
transformedX = transformer.transform(X_train)
model = RandomForestRegressor(min_samples_leaf= 3, n_estimators= 500)
model.fit(transformedX, Y_train)

# transform the validation dataset
transformedValidationX = transformer.transform(X_validation)
predictions = model.predict(transformedValidationX)

# Report: average mean_squared_error over 10 runs :) 
print('MSE = ', mean_squared_error(Y_validation, predictions))

MSES=[]
for i in range(10):
    model = RandomForestRegressor(min_samples_leaf= 3, n_estimators= 500)
    model.fit(transformedX, Y_train)
    predictions = model.predict(transformedValidationX)
    MSE =  mean_squared_error(Y_validation, predictions)
    print('{}. MSE = {}'.format(i+1, MSE))
    MSES.append(MSE)
print("Average of MSEs over 10 runs :", np.mean(np.array(MSES)))
"""
> 1137.7
"""



# ====================== P 04: Extra investigation

# Feature selection with RF impurity-based feature importance (Gini importance) > 0.90:
nf = 38
for nf in range (len(model.feature_importances_)):    
    if (sum(model.feature_importances_[model.feature_importances_.argsort()[::-1][0:nf]]) > 0.90 ):
        print(nf)
        break  
"""
> 38
"""
Important_Features = X_train.columns[model.feature_importances_.argsort()[::-1][0:nf]]
print(Important_Features)
"""
['x_42', 'x_73', 'x_52', 'x_51', 'x_95', 'x_88', 'x_31', 'x_83', 'x_50',
 'x_28', 'x_ 4', 'x_37', 'x_90', 'x_65', 'x_18', 'x_20', 'x_45', 'x_87',
 'x_61', 'x_ 1', 'x_74', 'x_48', 'x_16', 'x_94', 'x_79', 'x_40', 'x_11',
 'x_ 2', 'x_43', 'x_14', 'x_71', 'x_54', 'x_59', 'x_47', 'x_84', 'x_56',
 'x_62', 'x_36']
"""


# Tune GBM
model = Pipeline([('preprocessor', preprocessor),('gbm',GradientBoostingRegressor())])

n_estimators=[400, 450, 500, 600,700]
param_grid = {'gbm__n_estimators': n_estimators}

rkfold = RepeatedKFold(n_splits=num_folds, n_repeats=3)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=rkfold, n_jobs = -1, verbose = 2)
grid_result = grid.fit(X_train, Y_train)

print('MSE:     %.3f' % grid_result.best_score_)
print('rf_random_best_params_:   %s' % grid_result.best_params_)

"""
MSE:     -1746.998
rf_random_best_params_:   {'gbm__n_estimators': 500}
"""

# prepare the model
transformer = preprocessor.fit(X_train)
transformedX = transformer.transform(X_train)
model = GradientBoostingRegressor(n_estimators= 500)
model.fit(transformedX, Y_train)

# transform the validation dataset
transformedValidationX = transformer.transform(X_validation)
predictions = model.predict(transformedValidationX)

# Report: average mean_squared_error over 10 runs :) 
print('MSE = ', mean_squared_error(Y_validation, predictions))

MSES=[]
for i in range(10):
    model = GradientBoostingRegressor(n_estimators= 500)
    model.fit(transformedX, Y_train)
    predictions = model.predict(transformedValidationX)
    MSE =  mean_squared_error(Y_validation, predictions)
    print('{}. MSE = {}'.format(i+1, MSE))
    MSES.append(MSE)
print("Average of MSEs over 10 runs :", np.mean(np.array(MSES)))
"""
> 1068.34
"""
