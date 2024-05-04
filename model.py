import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# save the parameters and model  to use when predicting 
import pickle
# Importing dataset
data = pd.read_csv('Loan_Data.csv')

# Identify columns with missing values for each data type 
missing_numeric_cols = data.select_dtypes(include=['number']).columns[data.select_dtypes(include=['number']).isnull().any()].tolist() 
missing_object_cols = data.select_dtypes(include=['object']).columns[data.select_dtypes(include=['object']).isnull().any()].tolist() 
# Fill missing values with mean for numeric columns and mode for object columns 
imputer_numeric = SimpleImputer(strategy='mean') 
imputer_object = SimpleImputer(strategy='most_frequent') 
data_imputed_numeric = imputer_numeric.fit_transform(data[missing_numeric_cols]) 
data_imputed_object = imputer_object.fit_transform(data[missing_object_cols])
# Update DataFrame with imputed values 
data[missing_numeric_cols] = data_imputed_numeric 
data[missing_object_cols] = data_imputed_object
 
train = data.set_index('Loan_ID')

# turning string data into floats 
le_Dependents = LabelEncoder()
train['Dependents'] = le_Dependents.fit_transform(train['Dependents'])

le_Gender = LabelEncoder()
train['Gender'] = le_Gender.fit_transform(train['Gender'])
# train["Gender"].unique()

le_Married = LabelEncoder()
train['Married'] = le_Married.fit_transform(train['Married'])


le_Education = LabelEncoder()
train['Education'] = le_Education.fit_transform(train['Education'])
# train["Education"].unique()

le_Self_Employed = LabelEncoder()
train['Self_Employed'] = le_Self_Employed.fit_transform(train['Self_Employed'])
# train["Self_Employed"].unique()


le_Property_Area = LabelEncoder()
train['Property_Area'] = le_Property_Area.fit_transform(train['Property_Area'])
# train["Property_Area"].unique()


le_Loan_Status = LabelEncoder()
train['Loan_Status'] = le_Loan_Status.fit_transform(train['Loan_Status'])
# create X and y
X = train.drop("Loan_Status", axis=1)
y = train["Loan_Status"]


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2, 
                                                    random_state=42)


model = RandomForestClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_pred,y_test)

data = {"model": model, "le_Dependents":le_Dependents,"le_Gender": le_Gender, "le_Married": le_Married, "le_Education": le_Education, "le_Self_Employed": le_Self_Employed }
# data = {"model": linear_reg, "le_Gender": le_Gender, "le_Married": le_Married,  "le_Dependents": le_Dependents, "le_Education": le_Education, "le_Self_Employed": le_Self_Employed, }

with open('model.pkl', 'wb') as file:
    pickle.dump(data, file)
    