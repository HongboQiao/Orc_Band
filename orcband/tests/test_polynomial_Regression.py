import sys
sys.path.append("../ML_model")
import Polynomial_Regression_model
import pandas as pd
def test_polynomial_Reg():
    data = pd.read_csv('../PreML/DescriptorsDataset.csv')
    score = Polynomial_Regression_model.Polynomial_Model(data)
    assert score < 0.65, "should be 0.64"
    return