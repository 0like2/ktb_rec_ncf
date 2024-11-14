import pandas as pd
import os


print("Current working directory:", os.getcwd())


df1 = pd.read_csv('input/Creator_random25.csv')
df2 = pd.read_csv('input/Item_random25.csv')

print(df1.head())
print(df2.head())
