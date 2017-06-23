# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 13:24:46 2017

@author: Privat
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split


# Read data from the given csv file
def load_data(file_name):
    return pd.read_csv(file_name)

# Format the data in the given data_frame
def format_data(data):
    data['sex'] = data['sex'].map({'M': 1, 'F': 0})
    data['address'] = data['address'].map({'U': 1, 'R': 0})
    data['famsize'] = data['famsize'].map({'LE3': 1, 'GT3': 0})
    data['Pstatus'] = data['Pstatus'].map({'T': 1, 'A': 0})
    data['reason'] = data['reason'].map({'reputation': 0, 'home': 1, 'course': 2, 'other': 3})
    data['guardian'] = data['guardian'].map({'mother': 1, 'father': 0, 'other': 2})
    data['internet'] = data['internet'].map({'yes': 1, 'no': 0})
    data['romantic'] = data['romantic'].map({'yes': 1, 'no': 0})
    data['higher'] = data['higher'].map({'yes': 1, 'no': 0})
    data['activities'] = data['activities'].map({'yes': 1, 'no': 0})
    data['paid'] = data['paid'].map({'yes': 1, 'no': 0})
    data['famsup'] = data['famsup'].map({'yes': 1, 'no': 0})
    data['schoolsup'] = data['schoolsup'].map({'yes': 1, 'no': 0})
    
    return data

def main():
    # Read all data from csv
    all_data = load_data('data/train.csv')
    
    # Format the data to only have numbers no strings in it
    formated_data = format_data(all_data)
    
    # Drop the grades (Grade) column from data set
    X = formated_data.drop('Grade', axis=1)
    
    # Create a LinearRegression model
    lm = LinearRegression()
    
    # Fitting a linear model
    lm.fit(X, formated_data.Grade)
    
    print 'Estimated intercept coefficient: ', lm.intercept_
    print 'Number of coefficients: ', len(lm.coef_)

    data_with_coefficients = pd.DataFrame(zip(X.columns, lm.coef_), columns = ['features', 'estimatedCoefficients'])
    
    print data_with_coefficients
    
    # plot data with high correlation
    plt.scatter(formated_data.higher, formated_data.Grade)
    plt.xlabel('If he wants to study or not')
    plt.ylabel('Students grade')
    plt.title('Relationship between higher and Grade')
    plt.show()
    
    # Predict students grade and display first 5 of them
    print lm.predict(X)[0:5]
    
    # Display true Grade and predicted grade
    plt.scatter(formated_data.Grade, lm.predict(X))
    plt.xlabel('Grade')
    plt.ylabel('Predicted grade')
    plt.title('Grade vs Predicted grade')
    plt.show()
    
    # Calculate the mean squared error 
    mseFull = np.mean((formated_data.Grade - lm.predict(X)) ** 2)
    print mseFull
    
    # Fitting a linear model for one variables
    lm.fit(X[['traveltime']], formated_data.Grade)
    
    # Calculate the mean squeared error for only on column from dataset
    mseSex = np.mean((formated_data.Grade - lm.predict(X[['traveltime']])) ** 2)
    print mseSex
    
    """The whole model but with traint and test splited data"""
    
    # Split the data in trainig and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(
            X, formated_data.Grade, test_size=0.33, random_state=5)
    
    print X_train.shape
    print X_test.shape
    print Y_train.shape
    print Y_test.shape
    
    # Build a linear regression model using new data
    lm = LinearRegression()
    lm.fit(X_train, Y_train)
    pred_train = lm.predict(X_train)
    pred_test = lm.predict(X_test)
    
    print 'Fit a model X_train, and calculate MSE with Y_train: ', np.mean((Y_train - lm.predict(X_train))**2)
    print 'Fit a model X_train, and calculate MSE with X_test, Y_test: ', np.mean((Y_test - lm.predict(X_test))**2)
    
    # Residual plot
    plt.scatter(lm.predict(X_train), lm.predict(X_train)-Y_train, c='b', s=40, alpha=0.5)
    plt.scatter(lm.predict(X_test), lm.predict(X_test)-Y_test, c='g', s=40)
    plt.hlines(y=0, xmin=0, xmax=20)
    plt.title('Residual plot using training (blue) and test (green) data')
    plt.ylabel('Residuals')
    plt.show()
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    