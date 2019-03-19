# Orc Band!  

<div align=center> <img src="https://github.com/HongboQiao/Organic-Semi/blob/master/Organic-Semi/Documents/Logo%20square.jpg" width="400"> </div>

## Project Introduction
Our overall goal of the project is to predict the bandgap and bandgap position of the organic semiconductors, by using Machine Leaning Method. The dataset that we use is Harvard Clean Energy Project Database. To achieve this goal, Our tasks are: 
1. Calculate molecular descriptors for organic semiconductor from SMILES strings;
2. Determine predictors for mathine learning mehthod by LASSO regression;
3. Screen and optimization of regression model;
4. Build a wrapping function that help user to use our model

### Calculation by RDkit
RDkit is a very useful and opensource package which can be download very easily. By using the map calculation in the package, we can easily get thousands of descriptors from the SMILES strings. And use LASSO to screen the predictors.

### Machine Learning Model
#### Multiple Linear Regression
Import Linear Regression by using
```
from sklearn.linear_model import LinearRegression
```
75% of the data are used to train the model and 25% are used to test the model. The score of this model is 0.59 and the SSR is 461.51.
#### Polynominal Regression
Import Polynominal Regression by using
```
from sklearn.preprocessing import PolynomialFeatures
```
75% of the data are used to train the model and 25% are used to test the model. The SSR is 462.96.
#### Random Forest regression
Import Random Forest Regression by using
```
from sklearn.ensemble import RandomForestRegressor
```
75% of the data are used to train the model and 25% are used to test the model. The score of this model is 0.69 and the SSR is 130.36.
#### Neural Network
Import Keras by using
```
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
```
75% of the data are used to train the model and 25% are used to test the model. The SSR is 488.88.

#### Prediction Model
The Scatter Figure of the Predicted Bandgap for 4 Models are as follows.    
![Prediction Models](https://github.com/HongboQiao/Organic-Semi/blob/master/Organic-Semi/Documents/4ModelPlot.png)   
Therefore, we choose the Random Forest Regression as our Prediction Model.  
(Pending)
## Package Needed
### Rdkit - Calculate the descriptors
There are tow ways to install this package.
1. conda

    ```
    $ conda install -c rdkit -c mordred-descriptor mordred
    ```
    This is recommended and we install this in this way.
    Here is the link for [miniconda](http://conda.pydata.org/miniconda.html)

2. pip

    ```
    $ pip install 'mordred[full]'      #Or
    ```
    ```
    $ pip install mordred
    ```

### Scikit-Learn
There are tow ways to install this package.(If you already have numpy and scipy)
Here is the link for [scikit-learn](https://scikit-learn.org/stable/install.html)

1. conda

    ```
    conda install scikit-learn
    ```

2. pip

    ```
    $ pip install -U scikit-learn

    ```

## Github Organization
```
Organic-Semi/
    Organic-Semi/
        PreML/
        ML_model/
        tests/
        Document/
    README.md
    keras1_0.yml
    LICENSE
    setup.py
```
