import os
import tarfile
import urllib.request
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

#load dataset
pathfolder = os.path.join(os.getcwd(), "dataset")
url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz"
file_name = "housing.tgz"

def load_dataset(url = url, path_folder = pathfolder, file_name = file_name):
    if not os.path.isdir(pathfolder):
        os.makedirs(pathfolder)
    tgz_path = os.path.join(pathfolder, file_name)
    urllib.request.urlretrieve(url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=pathfolder)
    housing_tgz.close()

def open_csv(pathfolder=pathfolder):
    csv_path = os.path.join(pathfolder, 'housing.csv')
    return pd.read_csv(csv_path)

load_dataset()

df = open_csv()

# train test data set
df["income_cat"] = pd.cut(df["median_income"],
                                  bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                  labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(df, df['income_cat']):
    strat_train_set=df.loc[train_index]
    strat_test_set=df.loc[test_index]

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


# Separete dataset

df_train = strat_train_set.drop("median_house_value", axis=1)
df_labels = strat_train_set["median_house_value"].copy()

# New attributes
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                    bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

#Pipeline
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
        ])

num_attribs = list(df_train.select_dtypes('number'))
cat_attribs = list(df_train.select_dtypes(exclude = 'number'))

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(df_train)

models=[]
models.append(('lin_reg', LinearRegression()))
models.append(('tree_reg', DecisionTreeRegressor()))
models.append(('forest_reg', RandomForestRegressor()))

results=[]
names=[]
i=0

for name, model in models:
    train_model=model.fit(housing_prepared, df_labels)
    predictions=train_model.predict(housing_prepared)
    error=mean_squared_error(df_labels,predictions)
    results.append(np.sqrt(error))
    names.append(name)
    print('Result mean squared error %s: %f' % (name, results[i]))
    i+=1