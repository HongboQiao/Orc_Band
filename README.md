# Organic-Semi

## Project Introduction
Our overall goal of the project is to predict the bandgap and bandgap position of the organic semiconductors, by using the Machine Leaning Model. And the data we use is from Harvard Clean Energy Project Database. We focus on using rdkit to calculate the fundamental properties for organic semiconductor from SMILES strings, and use LASSO regression to select the predictors for machine learning.

(Pending...)
### Calculation by RDkit
RDkit is a very useful and opensource package which can be download very easily. By using the map calculation in the package, we can easily get thousands of descriptors from the SMILES strings. And use LASSO to screen the predictors.
       
(Pending...)
### Machine Learning Model
(Pending...)
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
    Environmentpy36.yml
    LICENSE
    setup.py
```
