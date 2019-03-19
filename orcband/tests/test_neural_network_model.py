import sys
sys.path.append("../ML_model")
import Neural_Network_model
import pandas as pd
def test_multiple_linaer_model():
    data = pd.read_csv('../PreML/DescriptorsDataset.csv')
    score = Neural_Network_model.neural_network_model(data)
    assert score < 0.6, "should be 0.56"
    return