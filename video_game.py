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

# What are the most popular video games?
# Which video game genres have been the most popular?
# Have there been any trends in video game popularity?
# Which video game team have been the most popular?
# Have there been any differences in video game popularity between different genres?
# How have video game sales and revenue changed over the past two decades?
# Are there any relationships between the popularity of a video game and its critical or commercial success(ratings)?
# Are there any patterns in the release dates of popular video games (e.g., are certain times of year or days of the week more popular for video game releases)?

# Title: Title of the game
# Release Date: Date of release of the game's first version
# Team: Game developer team
# Rating: Average rating
# Times Listed: Number of users who listed this game
# Number of Reviews: Number of reviews received from the users
# Genres: All genres pertaining to a specified game
# Summary: Summary provided by the team
# Reviews: User reviews
# Plays: Number of users that have played the game before
# Playing: Number of current users who are playing the game.
# Backlogs: Number of users who have access but haven't started with the game yet
# Wishlist: Number of users who wish to play the game 