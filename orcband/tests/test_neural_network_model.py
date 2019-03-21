import sys
sys.path.append("../ml_model/code")
import neural_network_model
import pandas as pd
def test_multiple_linaer_model():
    data = pd.read_csv('../PreML/DescriptorsDataset.csv')
    score = neural_network_model.neural_network_model(data)
    assert score < 0.6, "should be 0.56"
    return
