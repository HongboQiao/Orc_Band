import sys
sys.path.append("../dataming/code")
import sms_dcp_csv
import numpy as np
import pandas as pd

def test_sms_dcp_csv():
    """This function is used to test sms_dcp_csv function"""
    data = pd.read_csv('C:/Users/meng1/Desktop/data_science/HCEPDB_moldata/HCEPDB_moldata.csv')
    data_sample = data.sample(frac=0.001)
    result = sms_dcp_csv.sms_dcp(data_sample)
    assert len(result) > 100, "Wrong!"
    return