#Calculation of important metrics using K Nearest Neighbor (KNN) algorithm

#Import all the libraries needed to construct the KNN algorithm
%reset
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy import stats
from sklearn.model_selection import GridSearchCV

#Read data from the csv file. The inputs are Heart Rate, SV, dP/dt, SV,SW, TAU, EDP, EDV, ESP ,ESV, EDPVR, ESPVR, Ea, Ees. The output is infract size
Datas = pd.read_csv("E:/Infarct-data.csv")
X = Datas.iloc[0:,3:16].values
y = Datas.iloc[0:,18].values

#Number of outer iterations performed are 500 and number of hyperparameter tuning iterations are 50
n_iterations=500
n_iter_tuning=50

#Initialize metric arrays of size equal to the number of iterations
maxerror=np.zeros(n_iterations)
meansqerror=np.zeros(n_iterations)
sqrtmeansqerror=np.zeros(n_iterations)
r2score=np.zeros(n_iterations)
maxerror=np.zeros(n_iterations)
meanabs_error=np.zeros(n_iterations)
medianabs_error=np.zeros(n_iterations)
n_neighbors_value=np.zeros(n_iter_tuning)
optimised_n_neighbors=np.zeros(n_iterations)
optimised_counts=np.zeros(n_iterations)

# Classifier "n_neighbors" is defined
n_neighb_est=np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15])

#Outer loop is used to calculate the metrics for n_iterations
for i in range(n_iterations):
    print(i)
    
    #Inner loop is used to tune the hyperparameter (n_neighbors) in the model
    for j in range(n_iter_tuning):
    
        #Split the data randomly with 80% training data and 20% testing (validation) data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
        
        #Use KNN model to fit the training datasets and find the best value of n_neighbors using GridSearchCV
        param_grid=dict(n_neighbors=n_neighb_est)
        knn = GridSearchCV(KNeighborsRegressor(), param_grid)
        knn.fit(X_train, y_train) 
        y_pred = knn.predict(X_test)
        optimised_neighbor=knn.best_estimator_.n_neighbors
        n_neighbors_value[j]=optimised_neighbor
 
    #Find optimal value of n_neighbors based on multiple iterations in the inner loop
    optimised_n_neighbors[i]=((stats.mode(n_neighbors_value))).mode[0]
    optimised_counts[i]=((stats.mode(n_neighbors_value))).count[0]
    n_neigh=[np.int(optimised_n_neighbors[i])]
    param_grid_new=dict(n_neighbors=n_neigh)
    
    #Use KNN model to fit the training datasets
    knn = GridSearchCV(KNeighborsRegressor(), param_grid_new)
    knn.fit(X_train, y_train)
    
    #Make predictions using the testing dataset
    y_pred = knn.predict(X_test)

    #Calculate metrics using predictions and expected output
    meansqerror[i]=mean_squared_error(y_test,y_pred)
    sqrtmeansqerror[i]=np.sqrt(meansqerror[i])
    r2score[i]=r2_score(y_test, y_pred)

#Calculate mean and standard deviation for all metrics
print (np.mean(meansqerror))
print (np.mean(sqrtmeansqerror))
print (np.mean(r2score))
print (np.std(meansqerror))
print (np.std(sqrtmeansqerror))
print (np.std(r2score))
