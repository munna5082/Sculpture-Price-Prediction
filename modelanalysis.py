import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from mlxtend.evaluate import bias_variance_decomp
import pickle

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("Artist Sculpture Cost.csv")
print(df.head(5))
print(df.shape)
print(df.info)
df['Scheduled Date'] = pd.to_datetime(df['Scheduled Date'], errors='coerce')
df['Delivery Date'] = pd.to_datetime(df['Delivery Date'], errors='coerce')

print(df.isna().sum())
df['Artist Reputation'] = df['Artist Reputation'].fillna(df['Artist Reputation'].median())
df['Height'] = df['Height'].fillna(df['Height'].median())
df['Width'] = df['Width'].fillna(df['Width'].median())
df['Weight'] = df['Weight'].fillna(df['Weight'].median())

imputer = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
df['Material'] = imputer.fit_transform(df[['Material']])
df['Transport'] = imputer.fit_transform(df[['Transport']])
df['Remote Location'] = imputer.fit_transform(df[['Remote Location']])
print(df.isna().sum())
print(df.info())

con = df['Cost'] > 0
df = df[con]
print(df.head(5))
print(df.info())


encoder = LabelEncoder()
print(np.sort(df['Material'].unique()))
df['Material'] = encoder.fit_transform(df['Material'])

print(df['International'].value_counts())
df['International'] = encoder.fit_transform(df['International'])

print(df['Express Shipment'].value_counts())
df['Express Shipment'] = encoder.fit_transform(df['Express Shipment'])

print(df['Installation Included'].value_counts())
df['Installation Included'] = encoder.fit_transform(df['Installation Included'])

print(df['Transport'].value_counts())
df['Transport'] = encoder.fit_transform(df['Transport'])

print(df['Fragile'].unique())
df['Fragile'] = encoder.fit_transform(df['Fragile'])

print(df['Customer Information'].unique())
df['Customer Information'] = encoder.fit_transform(df['Customer Information'])

print(df['Remote Location'].unique())
df['Remote Location'] = encoder.fit_transform(df['Remote Location'])
print(df.head(5))
print(df.describe())


for x in df.columns:
    sns.boxplot(df[x])
    plt.yticks(rotation=45)
    plt.show()

sns.boxplot(df['Height'])
plt.show()

q1 = df['Height'].quantile(0.25)
q3 = df['Height'].quantile(0.75)
iqr = q3-q1

upper_limit = q3 + (1.5*iqr)
lower_limit = q1 - (1.5*iqr)

df = df[(df['Height'] < upper_limit) & (df['Height'] > lower_limit)]

sns.boxplot(df['Height'])
plt.show()


sns.boxplot(df['Width'])
plt.show()

q1 = df['Width'].quantile(0.25)
q3 = df['Width'].quantile(0.75)
iqr = q3-q1

upper_limit = q3 + (1.5*iqr)
lower_limit = q1 - (1.5*iqr)

df = df[(df['Width'] < upper_limit) & (df['Width'] > lower_limit)]

sns.boxplot(df['Width'])
plt.show()


sns.boxplot(df['Price Of Sculpture'])
plt.show()

df = df[df['Price Of Sculpture'] < 4000]
df = df[df['Cost'] < 25000]
df = df.reset_index()
print(df.head(5))

df.drop(columns = ['index', 'Customer Id', 'Artist Name', 'Scheduled Date', 'Delivery Date', 'Customer Location'], inplace=True)
print(df.head(5))



cor = df.corr()
plt.figure(figsize=(12, 12))
sns.heatmap(cor, annot=True, cmap='viridis', linewidths=.5)
plt.show()
print(cor['Cost'].sort_values())

features = df.drop(columns='Cost')
target = df['Cost']

feature_selector = ExtraTreesRegressor(n_estimators=40)
feature_selector.fit(features, target)
s1 = pd.Series(feature_selector.feature_importances_)
s1.index = features.columns
s1 = s1.sort_values(ascending=False)
print(s1)

s1.plot.bar(figsize=(8, 8), width=0.8)
plt.show()

obj = mutual_info_regression(features, target)
s2 = pd.Series(obj)
s2.index = features.columns
s2 = s2.sort_values(ascending=False)
print(s2)

s2.plot.bar(figsize=(8, 8), width=0.8)
plt.show()


X = df.drop(columns=['Fragile', 'Customer Information', 'International', 'Installation Included', 'Remote Location', 'Cost'])
y = df['Cost']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor_model = Lasso(alpha=50, max_iter=100, tol=0.1)
rf_model = RandomForestRegressor(n_estimators=40)

models = [regressor_model, rf_model]
for model in models:
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(np.average(scores))

rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test).round()

print(r2_score(y_test, y_pred))

mse, bias, variance = bias_variance_decomp(rf_model, X_train.values, y_train.values, X_test.values, y_test.values, loss='mse', num_rounds=200, random_seed=1)
print(mse)
print(bias)
print(variance)

print(mean_absolute_error(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
print(np.sqrt(mean_squared_error(y_test, y_pred)))


with open('sculpturepredmodel.pkl', 'wb')as file:
    pickle.dump(rf_model, file)
