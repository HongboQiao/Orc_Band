import numpy as np
import pandas as pd
import pickle

from sklearn import linear_model
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from multiprocessing import freeze_support
from rdkit import Chem
from mordred import Calculator, descriptors


def sms_bandgap(sms):
    """Function from sms to predict Bandgap"""
    freeze_support()
    mols = [Chem.MolFromSmiles(sms)]
    # Create Calculator
    calc = Calculator(descriptors)
    # map method calculate multiple molecules (return generator)
    # pandas method calculate multiple molecules (return pandas DataFrame)
    raw_data=calc.pandas(mols)

    new = {'AXp-0d':raw_data['AXp-0d'].values,
        'AXp-1d':raw_data['AXp-1d'].values,
        'AXp-2d':raw_data['AXp-2d'].values,
        'ETA_eta_L':raw_data['ETA_eta_L'].values,
        'ETA_epsilon_3':raw_data['ETA_epsilon_3'].values}
        # Save the predictors (Pending)
    new_data=pd.DataFrame(index=[1],data=new)

    with open('Organic-Semi/regressor.pickle', 'rb') as f:
        regressor2 = pickle.load(f)
    # path and model need to rewrite

    bandgap = regressor2.predict(new_data)

    return bandgap
