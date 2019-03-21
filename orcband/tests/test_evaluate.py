import sys
sys.path.append("../ml_model/code")
import evaluation
import numpy as np

df1 = np.array([[2],[4],[6]])
df2 = np.array([[3],[5],[7]])
def test_mean_square_error():
    a = evaluation.mean_square_error(df1,df2)
    assert a == 1, "should be 1"
    return

def test_mean_absolute_error():
    b = evaluation.mean_absolute_error(df1,df2)
    assert b == 1, "should be 1"
    return

def test_mean_absolute_percentage_error():
    c = evaluation.mean_absolute_percentage_error(df1,df2)
    assert c < 0.5, "should be less than 0.5"
    return

def test_r2_score():
    d = evaluation.r2_score(df1,df2)
    assert d < 1, "should be less than 1"
    return
    