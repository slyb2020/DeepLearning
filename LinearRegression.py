import pandas as pd
from ID_DEFINE import *
def OpenDataFile(filename):
    Data = pd.read_csv(filename)
    x = Data['x'].values
    y = Data['y'].values

if __name__ == '__main__':
    filename = linearRegressionDataDir+'data.csv'
    OpenDataFile(filename)