# Orc Band!

<div align=center> <img src="https://github.com/HongboQiao/Orc_Band/blob/master/documentation/image/Logo.jpg" width="400"> </div>

## Project Introduction
Our overall goal of the project is to predict the bandgap of organic semiconductors with Machine Leaning Methods. The dataset that we use is Harvard Clean Energy Project Database. To achieve this goal, Our tasks are:
1. Calculate molecular descriptors for organic semiconductor from SMILES strings;
2. Determine predictors for mathine learning mehthod by LASSO regression;
3. Screen and optimize regression model;
4. Build a wrapping function that help user to use our model.


### Calculation by RDkit
[RDkit](https://www.rdkit.org/) is a very useful and opensource package which can be download very easily. By using the map calculation in the package, we can easily get thousands of descriptors from the SMILES strings. And use several methods to screen the predictors.

### Machine Learning Model
For all the regression models we choosed, 75% of the data are used to train the model and 25% are used to test the model. By choosing the model, we randomly choose a couple of small size of data to run it several times and calculate the average statistic data.
#### Multiple Linear Regression
Import Linear Regression by using
```
from sklearn.linear_model import LinearRegression
```
The score of this model is 0.59.
#### Polynominal Regression
Import Polynominal Regression by using
```
from sklearn.preprocessing import PolynomialFeatures
```
The score of this model is 0.54.
#### Random Forest regression
Import Random Forest Regression by using
```
from sklearn.ensemble import RandomForestRegressor
```
The score of this model is 0.67.
#### Neural Network
Import Keras by using
```
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
```
The score of this model is 0.57.

*Note: Tensor Flow Needed!

#### Prediction Model
The Scatter Figure of the Predicted Bandgap for 4 Models are as follows.
<div align=center>
<img src="https://github.com/HongboQiao/Orc_Band/blob/master/documentation/image/Model_Comparison.png" width="700">
</div>
The calculated statistic data are as follows:

<br>

| Error | Multiple Linear | Random Forest | Polynomial  | Neural Network |
| :---: | :-------------: | :-----------: | :---------: | :------------: |
| MSE   | 0.0450     | 0.0357   | 0.0503 | 0.1732    |
| MAE   | 0.1659     | 0.1425   | 0.1728 | 0.3349    |
| MAPE  | 0.0928     | 0.0792   | 0.0959 | 0.1883    |
| $R^2$ | 0.5933     | 0.6772   | 0.5458 | 0.5665    |
| Kfold | 0.5906     | 0.6819   | 0.5906 | -0.0620   |


According to the figure and the table above, we choose the Random Forest Regression as our Prediction Model. And by optimizing it, we have a really good model, which has $R^2=0.80$. (For whole data set)

<div align=center>
<img src="https://github.com/HongboQiao/Orc_Band/blob/master/documentation/image/Optimized_Random_Forest.png" width="300">
</div>

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

### Keras
Keras is a high-level neural networks API, you can find the infomation [here](https://keras.io/). Since the introduction on their website is very straight forward, we don't write it here.

## Github Organization
Our main work can be divided for 3 parts: datamining(Data Mining and Cleaning), ml_model(Machine Learning Models), tests(All test file). For each directory, we seperate the files into code(".py") and notebook(".ipynb", which is a usecase for the ".py" file).
```
Orc_Band/
    orcband/
        datamining/
            code/
            notebook/
        ml_model/
            code/
            notebook/
        tests/
    documentation/
        image/
        data/
    README.md
    keras1_0.yml
    LICENSE
    setup.py
```
  *Note: The example file ``usecase.ipynb`` is located in the documentation directory, it's an example about how to use our model. Also, you can put your SMILES strings to the function and get their bandgap predictions.
  
  *Note: The data we use is confidential, so you may be not able to run our jupyter notebook directly in each file.
