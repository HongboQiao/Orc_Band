#test 
import sys
sys.path.append("../ML_model")
import Multiple_linearReg_model
import pandas as pd
def test_multiple_linaer_model():
    data = pd.read_csv('../PreML/DescriptorsDataset.csv')
    score = Multiple_linearReg_model.Multiple_Linear_Model(data)
    assert score < 0.6, "should be 0.57"
    return

