#test 
import sys
sys.path.append("../ml_model/code")
import multiple_linear_model
import pandas as pd
def test_multiple_linaer_model():
    data = pd.read_csv('../../documentation/data/DescriptorsDataset.csv')
    score = multiple_linear_model.Multiple_Linear_Model(data)
    assert score < 0.6, "should be 0.57"
    return

