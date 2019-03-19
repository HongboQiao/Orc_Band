import sys
sys.path.append("../PreML")
import Random_Forest_Regression_Optimization

def test_rfr_optimization():
    result = Random_Forest_Regression_Optimization.search_hyperparameter()
    assert result is None, "Wrong!"
    return