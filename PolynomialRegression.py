#Calculation of important metrics using Polynomial Regression algorithm

#Import all the libraries needed to construct the Polynomial Regression algorithm
%reset
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from scipy import stats

#Read data from the csv file. The inputs are Heart Rate, SV, dP/dt, SV,SW, TAU, EDP, EDV, ESP ,ESV, EDPVR, ESPVR, Ea, Ees. The output is infract size
Datas = pd.read_csv("E:/Infarct-data.csv")
X = Datas.iloc[0:,3:16].values
y = Datas.iloc[0:,18].values

#Number of iterations performed are 5000
n_iterations=5000

#Initialize metric arrays of size equal to the number of iterations
maxerror=np.zeros(n_iterations)
meansqerror=np.zeros(n_iterations)
sqrtmeansqerror=np.zeros(n_iterations)
r2score=np.zeros(n_iterations)
maxerror=np.zeros(n_iterations)
meanabs_error=np.zeros(n_iterations)
medianabs_error=np.zeros(n_iterations)
optimised_n_neighbors=np.zeros(n_iterations)
optimised_counts=np.zeros(n_iterations)

#Loop is used to calculate the metrics for n_iterations
for i in range(n_iterations):
    print(i)
    
    #Split the data randomly with 80% training data and 20% testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    
    #Generates polynomial features from original input features
    poly=PolynomialFeatures(degree=2, include_bias=True)
    Xtrain_poly=poly.fit_transform(X_train) 
    Xtest_poly=poly.transform(X_test)
    
    #Use linear regression model to fit mode to polynomial features
    model = LinearRegression(fit_intercept=False)
    model.fit(Xtrain_poly, y_train)
    
    #Make predictions using the testing dataset
    y_pred = model.predict(Xtest_poly)
    
    #Calculate metrics using predictions and expected output
    maxerror[i]=max_error(y_test, y_pred)
    meanabs_error[i]=mean_absolute_error(y_test, y_pred)
    medianabs_error[i]=median_absolute_error(y_test, y_pred)
    meansqerror[i]=mean_squared_error(y_test,y_pred)
    sqrtmeansqerror[i]=np.sqrt(meansqerror[i])
    r2score[i]=r2_score(y_test, y_pred)

#Calculate mean and standard deviation for all metrics
print (np.mean(maxerror))
print (np.mean(meanabs_error))
print (np.mean(medianabs_error))
print (np.mean(meansqerror))
print (np.mean(sqrtmeansqerror))
print (np.mean(r2score))
print (np.std(maxerror))
print (np.std(meanabs_error))
print (np.std(medianabs_error))
print (np.std(meansqerror))
print (np.std(sqrtmeansqerror))
print (np.std(r2score))
