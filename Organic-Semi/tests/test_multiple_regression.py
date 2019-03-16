import sys
sys.path.append("../ML_model")
import MLModels
import pandas as pd

def test_multiple_linaer_model():
    data = pd.read_csv('../PreML/DescriptorsDataset.csv')
    score = MLModels.Multiple_Linear_Model(data)
    assert score < 0.6, "should be 0.57"
    return

def test_Polynomial_Model():
    data = pd.read_csv('../PreML/DescriptorsDataset.csv')
    score1 = MLModels.Polynomial_Model(data)
    assert score1 < 0.65, "should be less than 0.65"
    return

def test_Random_Forest_Model():
    data = pd.read_csv('../PreML/DescriptorsDataset.csv')
    score2 = MLModels.Random_Forest_Model(data)
    assert score2 > 0.7, "should be larger than 0.7"
    return