
#import all package
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
get_ipython().run_line_magic('matplotlib', 'inline')

def lassoreg(Descriptors):
    """This function is used to plot RR coef and error vs lamdas throughs Lasso Regression method"""
    train,test = train_test_split(Descriptors, test_size=0.05, random_state=123)
    scaler = StandardScaler().fit(train)
    train_normalized = pd.DataFrame(data=scaler.transform(train), columns=train.columns)
    test_normalized = pd.DataFrame(data=scaler.transform(test), columns=test.columns)
    coefs = []
    trainerror = []
    testerror = []

    lambdas = np.logspace(-8,1,100) # Define lambda, May change
    model=linear_model.Lasso() 

    # loop over lambda values (strength of regularization)
    for k in lambdas:
        model.set_params(alpha=k,max_iter=1e7)
        model.fit(train_normalized[train.columns.values[2:15]],train_normalized[train.columns.values[1]])
        coefs.append(model.coef_)
        trainerror.append(mean_squared_error(train_normalized[train.columns.values[1]],model.predict(
        train_normalized[train.columns.values[2:15]])))
        testerror.append(mean_squared_error(test_normalized[train.columns.values[1]],model.predict(
        test_normalized[train.columns.values[2:15]])))

    # Plot The Fiigure
    fig = plt.figure(figsize=(10,3))
    plt.subplot(121)
    plt.plot(lambdas,coefs)
    plt.xscale('log')
    plt.xlabel('$\lambda$')
    plt.ylabel('$coefs$')
    plt.title('RR coefs vs $\lambda$')

    plt.subplot(122)
    plt.plot(lambdas,trainerror,label='train error')
    plt.plot(lambdas,testerror,label='test error')
    plt.xscale('log')
    plt.xlabel('$\lambda$')
    plt.ylabel('error')
    plt.legend(loc=1)
    plt.title('error vs $\lambda$')
    return
   
