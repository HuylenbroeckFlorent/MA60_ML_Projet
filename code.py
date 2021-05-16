import numpy as np
import pandas as pd

# CTRL + / to comment out blocks of code.

### Display full dataframe columns.
pd.set_option("display.max_columns", None, "display.max_rows", None)

d_train = pd.read_csv("data/D_train.csv", index_col="index")
x_test = pd.read_csv("data/X_test.csv", index_col="index")

# print(d_train.head())
# print(d_train.isnull().sum())
# print(d_train.isna().sum())

from features import numeric_features, boolean_feature, categorical_nominal_features, categorical_ordinal_features, categorical_ordinal_features_mappers

for i in range(len(categorical_ordinal_features)):
	d_train[categorical_ordinal_features[i]] = d_train[categorical_ordinal_features[i]].replace(categorical_ordinal_features_mappers[i])
	x_test[categorical_ordinal_features[i]] = x_test[categorical_ordinal_features[i]].replace(categorical_ordinal_features_mappers[i])

d_train["Central_Air"] = d_train["Central_Air"].replace({'Y':1, 'N':0})
x_test["Central_Air"] = x_test["Central_Air"].replace({'Y':1, 'N':0})

d_train["Mo_Sold"] = d_train["Mo_Sold"].replace({1:"JAN",2:"FEV",3:"MAR",4:"APR",5:"MAY",6:"JUN",7:"JUL",8:"AUG",9:"SEP",10:"OCT",11:"NOV",12:"DEC"})
x_test["Mo_Sold"] = x_test["Mo_Sold"].replace({1:"JAN",2:"FEV",3:"MAR",4:"APR",5:"MAY",6:"JUN",7:"JUL",8:"AUG",9:"SEP",10:"OCT",11:"NOV",12:"DEC"})
d_train_tmp = d_train.copy()
d_train_tmp['y'] = d_train_tmp['y'].replace({'A':0,'B':1,'C':2,'D':3,'E':4})
d_train_tmp = pd.get_dummies(d_train_tmp, prefix=categorical_nominal_features)

y_corr = d_train_tmp.corr()['y'].abs().to_dict() #.sort_values(ascending=False)

corrs = {key: corr for key, corr in y_corr.items() if key in numeric_features+boolean_feature+categorical_ordinal_features}

for feature in categorical_nominal_features:
	occ = d_train[feature].value_counts().to_dict()
	tmp_sum = 0
	tmp_len = 0
	for key, value in occ.items():
		tmp_len += value
		tmp_sum += value*y_corr[feature+"_"+key]
	corrs[feature] = tmp_sum/tmp_len

x_train = d_train.drop("y", axis=1)
y_train = d_train.y

post_process_numeric_features = numeric_features+boolean_feature+categorical_ordinal_features
post_process_categorical_features = categorical_nominal_features

def build_and_score_model(	func,\
							numeric_features=[],\
							categorical_features=[],\
							export_path="output/tmp.csv",\
							folds=10, 
							to_dense=False):

	from sklearn.pipeline import Pipeline
	from sklearn.impute import SimpleImputer
	from sklearn.preprocessing import StandardScaler, OneHotEncoder

	numeric_transformer = Pipeline(steps=[
	    ('imputer', SimpleImputer(strategy='median')),
	    ('scaler', StandardScaler())])

	categorical_nominal_transformer = OneHotEncoder(handle_unknown='ignore')


	if to_dense == True:
		categorical_nominal_transformer = Pipeline(steps=[
			('onehot', OneHotEncoder(handle_unknown='ignore')),
			('to_dense', DenseTransformer())])

	from sklearn.compose import ColumnTransformer

	preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_nominal_transformer, categorical_features)])

	model = Pipeline(steps=[('preprocessor', preprocessor),
                    ('classifier', func)])

	from sklearn.model_selection import cross_validate

	model.fit(x_train, y_train)

	cv_results = cross_validate(model, x_train, y_train, scoring='neg_log_loss', n_jobs=4, cv=folds)

	pred_prob_test = pd.DataFrame(model.predict_proba(x_test))
	pred_prob_test.set_index(x_test.index, inplace = True)
	pred_prob_test.reset_index(inplace=True)
	pred_prob_test.rename(columns = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}, inplace = True)
	pred_prob_test.to_csv(export_path, index = False)

	return -(sum(cv_results["test_score"])/len(cv_results["test_score"]))

from sklearn.base import TransformerMixin

class DenseTransformer(TransformerMixin): #https://stackoverflow.com/a/28384887

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()


def test_in_correlation_order(func, i_min=0):
	import os

	best = 100
	most_correlated_features = []
	i=0
	for k, v in sorted(corrs.items(), key=lambda x: -x[1]):
		most_correlated_features.append(k)
		i+=1
		if i>i_min:
			numeric_subset = [str(f) for f in most_correlated_features if f in post_process_numeric_features]
			nominal_subset = [str(f) for f in most_correlated_features if f in post_process_categorical_features]
			tmp = build_and_score_model(func, numeric_features=numeric_subset, categorical_features=nominal_subset)

			if tmp<best:
				best=tmp
				print("==========")
				print("New best prediction using "+str(len(most_correlated_features))+" features.")
				print("Multiclass log loss score on training set : "+str(best))
				print(numeric_subset+nominal_subset)
				os.rename('output/tmp.csv', 'output/current_best_corr.csv')

def test_random_weighted_combinations(func):
	import random
	import os

	best = 100
	total_feature_length = len(post_process_numeric_features)+len(post_process_categorical_features)
	total_feature_list = post_process_numeric_features+post_process_categorical_features
	weights = []
	for feature in total_feature_list:
		weights.append(corrs[feature])
	weights_p = [w/sum(weights) for w in weights]

	while 1:

		tmp_len = random.randint(round(1*total_feature_length/4),round(3*total_feature_length/4))
		rdm_features = np.random.choice(total_feature_list, size=tmp_len, replace=False, p=weights_p).ravel()
		numeric_subset = [str(f) for f in rdm_features if f in post_process_numeric_features]
		nominal_subset = [str(f) for f in rdm_features if f in post_process_categorical_features]
		tmp = build_and_score_model(func, numeric_features=numeric_subset, categorical_features=nominal_subset)

		if tmp<best:
			best=tmp
			print("==========")
			print("New best prediction using "+str(tmp_len)+" features.")
			print("Multiclass log loss score on training set : "+str(best))
			print(numeric_subset+nominal_subset)
			os.rename('output/tmp.csv', 'output/current_best_rdm.csv')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB

test_random_weighted_combinations(BaggingClassifier(RandomForestClassifier(n_estimators=60, n_jobs=8), n_estimators=60, n_jobs=8))
