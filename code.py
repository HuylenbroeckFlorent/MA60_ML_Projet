import numpy as np
import pandas as pd

# CTRL + / to comment out blocks of code.

### Display full dataframe columns.
pd.set_option("display.max_columns", None)

def import_and_clean_housing_data(path):

	### Importing raw data.
	data_raw = pd.read_csv('data/X_test.csv',\
							index_col="index",\
							dtype = {'MS_SubClass': 'category',\
									'MS_Zoning': 'category',\
									'Street': 'category',\
									'Alley': 'category',\
									'Lot_Shape': 'category',\
									'Land_Contour': 'category',\
									'Utilities': 'category',\
									'Bldg_Type': 'category',\
									'House_Style': 'category',\
									'Overall_Qual': 'category',\
									'Overall_Cond': 'category',\
									'Roof_Style': 'category',\
									'Exter_Qual': 'category',\
									'Exter_Cond': 'category',\
									'Foundation': 'category',\
									'Bsmt_Qual': 'category',\
									'Bsmt_Cond': 'category',\
									'Bsmt_Exposure': 'category',\
									'Heating': 'category',\
									'Heating_QC': 'category',\
									'Electrical': 'category',\
									'Kitchen_Qual': 'category',\
									'Functional': 'category',\
									'Fireplace_Qu': 'category',\
									'Garage_Type': 'category',\
									'Garage_Qual': 'category',\
									'Garage_Cond': 'category',\
									'Paved_Drive': 'category',\
									'Pool_QC': 'category',\
									'Fence': 'category',\
									'Misc_Feature': 'category',\
									'Sale_Type': 'category',\
									'Sale_Condition': 'category'})

	# for colname in data_raw:
	# 	print(data_raw[colname].value_counts())
	# 	print()

	## Pre-processing.

	# print(data_raw.isnull().sum()) NO NULL
	# print(data_raw.isna().sum()) NO NA

	#### Nominal categorical features.
	from sklearn.preprocessing import OneHotEncoder
	ohe = OneHotEncoder(dtype=np.int64, sparse=False)
	data = pd.DataFrame(ohe.fit_transform(data_raw[['MS_SubClass',\
		'MS_Zoning',\
		'Street',\
		'Alley',\
		'Lot_Shape',\
		'Land_Contour',\
		'Utilities',\
		'Bldg_Type',\
		'House_Style',\
		'Roof_Style',\
		'Foundation',\
		'Bsmt_Exposure',\
		'Heating',\
		'Electrical',\
		'Functional',\
		'Garage_Type',\
		'Paved_Drive',\
		'Fence',\
		'Misc_Feature',\
		'Mo_Sold',\
		'Sale_Type',\
		'Sale_Condition']]),index=data_raw.index)


	data.columns = ohe.get_feature_names(['MS_SubClass',\
		'MS_Zoning',\
		'Street',\
		'Alley',\
		'Lot_Shape',\
		'Land_Contour',\
		'Utilities',\
		'Bldg_Type',\
		'House_Style',\
		'Roof_Style',\
		'Foundation',\
		'Bsmt_Exposure',\
		'Heating',\
		'Electrical',\
		'Functional',\
		'Garage_Type',\
		'Paved_Drive',\
		'Fence',\
		'Misc_Feature',\
		'Mo_Sold',\
		'Sale_Type',\
		'Sale_Condition']) #https://stackoverflow.com/a/55206934

	data_raw = data_raw.drop(['MS_SubClass',\
		'MS_Zoning',\
		'Street',\
		'Alley',\
		'Lot_Shape',\
		'Land_Contour',\
		'Utilities',\
		'Bldg_Type',\
		'House_Style',\
		'Roof_Style',\
		'Foundation',\
		'Bsmt_Exposure',\
		'Heating',\
		'Electrical',\
		'Functional',\
		'Garage_Type',\
		'Paved_Drive',\
		'Fence',\
		'Misc_Feature',\
		'Mo_Sold',\
		'Sale_Type',\
		'Sale_Condition'], axis=1) #Remove features from raw data set.


	#### Ordinal categorical features.
	# for colname in data_raw.columns:
	# 	print(data_raw[colname].value_counts())
	# 	print()

	#Overall_Qual
	ordinals_Overall_Qual= pd.Categorical(data_raw.Overall_Qual,\
								categories=['Very_Poor','Poor','Below_Average','Average','Above_Average','Fair','Good','Very_Good','Excellent','Very_Excellent'],\
								ordered=True)
	labels, unique = pd.factorize(ordinals_Overall_Qual, sort=True)
	data['Overall_Qual']=labels

	#Overall_Cond
	ordinals_Overall_Cond= pd.Categorical(data_raw.Overall_Cond,\
								categories=['Very_Poor','Poor','Below_Average','Average','Above_Average','Fair','Good','Very_Good','Excellent','Very_Excellent'],\
								ordered=True)
	labels, unique = pd.factorize(ordinals_Overall_Cond, sort=True)
	data['Overall_Cond']=labels

	#Exter_Qual
	ordinals_Exter_Qual= pd.Categorical(data_raw.Exter_Qual,\
								categories=['Poor','Typical','Fair','Good','Excellent'],\
								ordered=True)
	labels, unique = pd.factorize(ordinals_Exter_Qual, sort=True)
	data['Exter_Qual']=labels

	#Exter_Cond
	ordinals_Exter_Cond= pd.Categorical(data_raw.Exter_Cond,\
								categories=['Poor','Typical','Fair','Good','Excellent'],\
								ordered=True)
	labels, unique = pd.factorize(ordinals_Exter_Cond, sort=True)
	data['Exter_Cond']=labels

	#Bsmt_Qual
	ordinals_Bsmt_Qual= pd.Categorical(data_raw.Bsmt_Qual,\
								categories=['No_Basement','Poor','Typical','Fair','Good','Excellent'],\
								ordered=True)
	labels, unique = pd.factorize(ordinals_Bsmt_Qual, sort=True)
	data['Bsmt_Qual']=labels

	#Bsmt_Cond
	ordinals_Bsmt_Cond= pd.Categorical(data_raw.Bsmt_Cond,\
								categories=['No_Basement','Poor','Typical','Fair','Good','Excellent'],\
								ordered=True)
	labels, unique = pd.factorize(ordinals_Bsmt_Cond, sort=True)
	data['Bsmt_Cond']=labels

	#Heating_QC
	ordinals_Heating_QC= pd.Categorical(data_raw.Heating_QC,\
								categories=['Poor','Typical','Fair','Good','Excellent'],\
								ordered=True)
	labels, unique = pd.factorize(ordinals_Heating_QC, sort=True)
	data['Heating_QC']=labels

	#Kitchen_Qual
	ordinals_Kitchen_Qual= pd.Categorical(data_raw.Kitchen_Qual,\
								categories=['Poor','Typical','Fair','Good','Excellent'],\
								ordered=True)
	labels, unique = pd.factorize(ordinals_Kitchen_Qual, sort=True)
	data['Kitchen_Qual']=labels

	#Fireplace_Qu
	ordinals_Fireplace_Qu= pd.Categorical(data_raw.Fireplace_Qu,\
								categories=['No_Fireplace','Poor','Typical','Fair','Good','Excellent'],\
								ordered=True)
	labels, unique = pd.factorize(ordinals_Fireplace_Qu, sort=True)
	data['Fireplace_Qu']=labels

	#Garage_Qual
	ordinals_Garage_Qual= pd.Categorical(data_raw.Garage_Qual,\
								categories=['No_Garage','Poor','Typical','Fair','Good','Excellent'],\
								ordered=True)
	labels, unique = pd.factorize(ordinals_Garage_Qual, sort=True)
	data['Garage_Qual']=labels

	#Garage_Cond
	ordinals_Garage_Cond= pd.Categorical(data_raw.Garage_Cond,\
								categories=['No_Garage','Poor','Typical','Fair','Good','Excellent'],\
								ordered=True)
	labels, unique = pd.factorize(ordinals_Garage_Cond, sort=True)
	data['Garage_Cond']=labels

	#Pool_QC
	ordinals_Pool_QC= pd.Categorical(data_raw.Pool_QC,\
								categories=['No_Pool','Poor','Typical','Fair','Good','Excellent'],\
								ordered=True)
	labels, unique = pd.factorize(ordinals_Pool_QC, sort=True)
	data['Pool_QC']=labels

	#### Boolean features
	data_raw.Central_Air.replace(('Y','N'),(1,0), inplace=True) #https://stackoverflow.com/q/40901770
	data['Central_Air'] = data_raw.Central_Air

	return data

xtest = import_and_clean_housing_data('data/X_test.csv')
dtrain = import_and_clean_housing_data('data/D_train.csv')

sample_raw = pd.read_csv('data/sample.csv', index_col="index")

sample = pd.Series(sample_raw.columns[np.where(sample_raw!=0)[1]]) #https://stackoverflow.com/a/51275990

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='lbfgs', max_iter=10000)
lr.fit(xtest, sample)


from sklearn.naive_bayes import GaussianNB
naive_b = GaussianNB()
naive_b.fit(xtest, sample)

predicted_proba = naive_b.predict_proba(dtrain)

csv = pd.DataFrame(predicted_proba, index=dtrain.index, columns=['A','B','C','D','E'])
csv.to_csv('output/test.csv')