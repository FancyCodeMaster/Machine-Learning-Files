'''
About the Data Preprocessing:
    Data Preprocessing is one of the important and critical step before creating our model and fitting it. It includes important steps like \:
    1. Importing the dataset
    2. Managing the missing values
    3. Encoding the categorical data
    4. Splitting the dataset into the Training Set and Test Set
    5. Feature Scaling 


    Explanation:

    1.Importing the dataset
        In this step , we simply load the csv file as our dataframe and separate it into Dependent Variable(y) and Independent variable(X).
    
    2. Managing the missing values
        In our dataframe(loaded from csv file) sometimes some values can be missing. The missing value entirely hinders the importance of the entire column.
        For that small problem , we cannot delete the entire column but we have to manage the missing values. For that we use the median of the entire column to fill
        the missing value's place.

    3. Encoding the categorical data
        Categorical data are the data that are simply strings(not numbers). But , Machine Learning models only work with numbers as many formulas are being run 
        behind the scenes. So to convert the categorical data into numbers we run through this process.

    4. Splitting the dataset into Training Set and Test Set
        Our dataset contains different number of dependent and independent variables. To check whether the predicted model(which we find later) matches with our 
        observed values , we make test set. Training set is made to train our model on the basis of which it creates a line like y = mx + c behind the scenes which
        ultimately helps the test set to get the predicted result.

    5. Features Scaling
        Machine Learning models are actually dumb interms of the comparison of the value of the variables. When they see two largely differing values(interms of magnitude)
        the model assumes as if the greater number had dominance over the small number. So to reduce the margin between different values , we use feature scaling.

        
    '''
# Data Preprocessing 

#Importing the libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#Importing the dataSet
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[: , 0:-1].values
y = dataset.iloc[: , -1].values


# Managing the missing values
from sklearn.impute import SimpleImputer
simpleimputer = SimpleImputer(missing_values = np.nan , strategy = "median")
'''imputer = imputer.fit(X[: , 1:3])
X[: , 1:3] = imputer.transform(X[: , 1:3]) This can be represe-
nted as:'''
X[: , 1:3] = simpleimputer.fit_transform(X[:,1:3])

# Encoding the categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
ct = ColumnTransformer([("encoder" , OneHotEncoder() , [0])] , remainder = "passthrough")
X = np.array(ct.fit_transform(X), dtype=np.float)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# Splitting the dataset into the Training Set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X ,y,test_size =0.2 , random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

