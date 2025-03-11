# Importing libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib as mpl
import seaborn as sns
import os
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

print(data.shape)
print(data.sample(5))

# py -3.11 pip install 

# Read our stocks data into a DataFrame object
#  Specify delimiter to ',' to match CSV format
#  Skip errors in our CSV file because I do not want to look through that file for every little error :)
# date,open,high,low,close,volume,Name
data = pd.read_csv('stocks_5yrs.csv', delimiter=',', on_bad_lines='skip') 
