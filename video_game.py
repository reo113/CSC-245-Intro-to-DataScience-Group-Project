import numpy as np
import pandas as pd


df = pd.read_csv('dataset\\games.csv')
df.head()
#print(df.isnull().sum())
df = df.dropna()
df = df.fillna(0)
df.drop_duplicates(inplace=True)
df.info()
column_names = df.columns.tolist()
print(column_names)
df = df.drop('Unnamed: 0', axis=1)
print(df.head(20))


#https://www.kaggle.com/datasets/arnabchaki/popular-video-games-1980-2023

