import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from sklearn import linear_model
from sklearn import tree

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

df = pd.read_csv('games.csv') #Assigns the game.csv file to the dataframe

df.info() #Prints the database info
df.head() #Returns first 5 rows
print("This is the shape of the dataset: ", df.shape)
#df.shape #Returns the number of rows(x) and columns(y)

print(df.nunique()) #Count number of distinct elements in specified row.

#Replaces all missing data in the Rating/Team/Summary
#Data set with values
df['Rating'] = df['Rating'].replace(np.nan, 0.0)
df['Team'] = df['Team'].replace(np.nan, "['Unknown Team']")
df['Summary'] = df['Summary'].replace(np.nan, 'Unknown Summary')


#The sort_index() method sorts the DataFrame by the index.
#Drops all duplicates in dataset
df = df.drop_duplicates().sort_index()

# create a datetime object for the datetime module
dt = datetime.now()
# convert the datetime object to a string
dt_str = dt.strftime('%b %d, %Y')
print(dt_str)


df.loc[df['Release Date'] == 'releases on TBD']

df['Release Date'] = df['Release Date'].str.replace('releases on TBD', dt_str )
df['Release Date'] = pd.to_datetime(df['Release Date'], format='%b %d, %Y')
# format the datetime object as a string in the desired format
df['Release Date'] = df['Release Date'].dt.strftime('%Y-%-m-%-d')


# convert the date column to a datetime object
df['Release Date'] = pd.to_datetime(df['Release Date'])
# get the day from the date column
df['Day'] = df['Release Date'].dt.day
df['Month'] = df['Release Date'].dt.strftime('%b')
df['Year'] = df['Release Date'].dt.year
df['Week day'] = df['Release Date'].dt.day_name()

#Outputs top 5 rows
print(df[['Title', 'Release Date', 'Rating', 'Day', 'Month', 'Year', 'Week day']].head())
#outputs last 5 rows
print(df[['Title', 'Release Date', 'Rating', 'Day', 'Month', 'Year', 'Week day']].tail())

#for x in df['Rating']:
 #   if x >= 4.0:
  #      print(df[['Title', 'Release Date', 'Rating', 'Day', 'Month', 'Year', 'Week day']])

print(df.describe())

# create a sample DataFrame with a column containing multiple values
df_rate = pd.DataFrame({
    'Release Date': df['Release Date'].tolist(),
    'Rating': df['Rating'].tolist()
})
# use the explode method to transform the 'Rating' column
df_rate = df_rate.explode('Rating')
print(df_rate)

#Plots the df_rate dataframe
#df_rate.plot()
#plt.show()

#Example of hard testing top 10 games/genres
top_10_games = ['Fortnite', 'Minecraft', 'Grand Theft Auto V', 'Counter-Strike: Global Offensive', 'Apex Legends', 'League of Legends', 'Call of Duty: Warzone', 'Valorant', 'Dota 2', 'Roblox']
top_10_genres = ['Action', 'Shooter', 'Sports', 'Role-Playing', 'Adventure', 'Strategy', 'Simulation', 'Fighting', 'Racing', 'Puzzle']


top_rating = df[['Title','Rating']].sort_values(by = 'Rating', ascending = False)
top_rating = top_rating.loc[top_rating['Title'].isin(top_10_games)]
top_rating = top_rating.drop_duplicates()

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

sns.histplot(ax = axes[0], data = df['Rating'])
sns.barplot(ax = axes[1], data = top_rating, x = 'Rating', y = 'Title', palette = 'Blues_d')

axes[0].set_title('Distribution of ratings', pad = 10, fontsize = 15)
axes[0].set_xlabel('Rating', labelpad = 20)
axes[0].set_ylabel('Frequency', labelpad = 20)

axes[1].set_title('Top rated games in 2021', pad = 10, fontsize = 15)
axes[1].set_xlabel('Rating', labelpad = 20)
axes[1].set_ylabel('Title', labelpad = 20)
plt.tight_layout()

plt.show()







#MACHINE LEARNING
#df.columns
# select prediction target
#y = df.Rating
# select features
#features = ['Times Listed', 'Number of Reviews', 'Plays', 'Playing', 'Backlogs', 'Wishlist']
#X = df[features]

# Define model. Specify a number for random_state to ensure the same results each run
#melbourne_model = DecisionTreeRegressor(random_state=1)
# Fit model
#melbourne_model.fit(X, y)

#print("Making predictions for the following 5 houses:")
#print(X.head())
#print("The predictions are")
#print(melbourne_model.predict(X.head()))


df_playRate = pd.DataFrame({
    'Release Date': df['Release Date'].tolist(),
    'Rating': df['Rating'].tolist(),
    'Plays': df['Plays'].tolist(),
    'Playing': df['Playing'].tolist(),
})



X = df_rate[['Rating']]
y = df_rate[['Release Date']]



regr = linear_model.LinearRegression()
regr.fit(X, y)

predictedRate = regr.predict([[4.0]])

print(predictedRate)
#ATTEMPT at training the dataset
plt.plot(X, y)
plt.show()

plt.plot(predictedRate)
plt.show()


train_x = X[:80]
train_y = y[:80]

test_x = X[80:]
test_y = y[80:]

plt.scatter(train_x, train_y)
plt.show()

plt.scatter(test_x, test_y)
plt.show()



ohe_cars = pd.get_dummies(df_rate[['Release Date']])

#print(ohe_cars.to_string())


#I believe that this is summing all the rating together
#An outputting the result next to the most popular
#dates that released the most popular games

#CAUSING AN ERROR NEED TO FIX
#platform = df.groupby('Release Date').sum()['Rating'].reset_index()
#platform = platform.sort_values('Rating', ascending=False).head(10)
#print(platform)
#figure2 = plt.bar(platform, Y='Release Date', title="Most popular release dates")
#figure2.show()

print("The max value: ")
print(df_rate.loc[df_rate['Rating'].idxmax()])

print("The other max value: ")
print(df.groupby(['Release Date','Title'])['Rating'].max().head(20))

#Attempt to get all the info from the highest rating games
Highest_Rating = df.groupby(['Release Date','Title'])['Rating'].max()

print(Highest_Rating)
