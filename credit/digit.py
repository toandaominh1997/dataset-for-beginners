import numpy as np 
import pandas as pd 
import os 

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(train.shape)