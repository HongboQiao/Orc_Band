import sys
sys.path.append("../ml_model/code")
import random_forest_regression
import pandas as pd
def test_Random_Forest_Reg():
    data = pd.read_csv('../../documentation/data/DescriptorsDataset.csv')
    score = random_forest_regression.Random_Forest_Reg(data)
    assert score > 0.7, "should be 0.7"
    return
