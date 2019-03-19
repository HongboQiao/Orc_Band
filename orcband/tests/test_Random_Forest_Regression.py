import sys
sys.path.append("../ML_model")
import Random_Forest_Regression
import pandas as pd
def test_Random_Forest_Reg():
    data = pd.read_csv('../PreML/DescriptorsDataset.csv')
    score = Random_Forest_Regression.Random_Forest_Reg(data)
    assert score > 0.7, "should be 0.7"
    return