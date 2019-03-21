import sys
sys.path.append("../ml_model/code")
import polynomial_regression_model
import pandas as pd
def test_polynomial_Reg():
    data = pd.read_csv('../../documentation/data/DescriptorsDataset.csv')
    score = polynomial_regression_model.Polynomial_Model(data)
    assert score < 0.65, "should be 0.64"
    return
