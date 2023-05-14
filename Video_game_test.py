import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# import the data
df = pd.read_csv('games.csv')
df.head()

# data cleaning and exploring
print(df.isnull().sum())
df['Team'].fillna('Unknown', inplace=True)
df['Summary'].fillna('Unknown', inplace=True)
df = df.fillna(method='ffill')
df.drop_duplicates(inplace=True)
print(df.isnull().sum())
column_names = df.columns.tolist()
print(column_names)
df = df.drop('Unnamed: 0', axis=1)
# Combine Teams with same name
df.loc[df['Team'] == "['FromSoftware', 'Sony Computer Entertainment']", 'Team'] += ', ' + \
                                                                                   df.loc[df['Team'] ==
                                                                                          "['Sony Computer Entertainment', 'FromSoftware']", 'Team'].iloc[
                                                                                       0]
# Delete duplicate row
df = df[df['Team'] != "['Sony Computer Entertainment', 'FromSoftware']"]
# Change name of the Team
df['Team'] = df['Team'].replace(
    "['FromSoftware', 'Sony Computer Entertainment'], ['Sony Computer Entertainment', 'FromSoftware']",
    "['Sony Computer Entertainment', 'FromSoftware']")
# Save the updated dataset
df['Release Date'] = pd.to_datetime(
    df['Release Date'], format='%b %d, %Y', errors='coerce')
df.info()

# calculates the mean rating for each team
team_ratings = df.groupby('Team')['Rating'].mean()
# select only top ten teams by average rating
top_ten = team_ratings.nlargest(10)
# merge the top ten teams with the Team and Genre columns
top_teams = pd.merge(
    top_ten, df[['Team', 'Genres']].drop_duplicates(), on='Team')
# Create a barplot of the top ten teams by rating and their associated genres
plt.figure(figsize=(12, 8))
ax = sns.barplot(data=top_teams, x='Team', y='Rating',
                 hue='Genres', linewidth=3.5)
plt.xticks(rotation=90)
# Add axis labels and a title
plt.xlabel('Team')
plt.ylabel('Average Rating')
plt.title('Top Ten Teams by Average Rating and Their Associated Genres')
# Improve the legend
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
           borderaxespad=0, fontsize=12)
# Add labels to the top of each bar
for index, row in top_teams.iterrows():
    ax.annotate(format(row['Rating'], '.2f'),
                xy=(row.name, row['Rating']),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center',
                va='bottom')
# Add horizontal grid lines
plt.grid(axis='y')
# Remove the top and right spines
sns.despine()
# Show the Graph
plt.show()

# Calculate mean rating and standard deviation
rating_mean = df['Rating'].mean()
rating_std = df['Rating'].std()

# Specify the number of standard deviations to include
num_std = 1

# Create an array of values within the specified range
rating_range = np.linspace(rating_mean - num_std *
                           rating_std, rating_mean + num_std * rating_std, 100)

# Select only the data that falls within the specified range
selected_data = df[(df['Rating'] >= rating_range[0]) &
                   (df['Rating'] <= rating_range[-1])]

# Get top 3 games on each side of the mean
top_games_positive = selected_data.sort_values(
    by='Rating', ascending=False).drop_duplicates(subset=['Title']).iloc[:3]
top_games_negative = selected_data.sort_values(
    by='Rating', ascending=True).drop_duplicates(subset=['Title']).iloc[:3]
# Get top 3 games that equal the mean
top_games_at_mean = df[(df['Rating'] >= rating_mean - rating_std / 2) & (df['Rating']
                                                                         <= rating_mean + rating_std / 2)].drop_duplicates(
    subset=['Title']).iloc[:3]

# Print the top games
print(f'Top {num_std} standard deviations above the mean:')
for i, row in top_games_positive.iterrows():
    print(f"{row['Title']}: {row['Rating']}")

print(f'\nTop {num_std} standard deviations below the mean:')
for i, row in top_games_negative.iterrows():
    print(f"{row['Title']}: {row['Rating']}")

print(f'\nTop games at the mean:')
for i, row in top_games_at_mean.iterrows():
    print(f"{row['Title']}: {row['Rating']}")

# Plot the selected data
plt.hist(selected_data['Rating'], bins=20, color='blue', alpha=0.7)
plt.axvline(rating_mean, color='red', linestyle='--', linewidth=2)
plt.axvline(top_games_positive['Rating'].min(),
            color='green', linestyle='--', linewidth=2)
plt.axvline(top_games_positive['Rating'].max(),
            color='green', linestyle='--', linewidth=2)
plt.axvline(top_games_negative['Rating'].min(),
            color='orange', linestyle='--', linewidth=2)
plt.axvline(top_games_negative['Rating'].max(),
            color='orange', linestyle='--', linewidth=2)
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title(
    'Distribution of Ratings within {} Standard Deviations of the Mean'.format(num_std))
plt.grid(True, axis='y', alpha=0.5)
plt.show()

# Select relevant columns and drop rows with missing data
df['Plays'] = df['Plays'].str.replace('K', '000')

# convert Plays column to numeric data type
df['Plays'] = pd.to_numeric(df['Plays'], errors='coerce')

# fill NaN values with 0
df['Plays'] = df['Plays'].fillna(0)
data = df[['Team', 'Plays']].dropna()

# Create a binary matrix of genre values for each game
team = df["Team"].str.get_dummies("]")

# Sum the occurrences of each team across all games
popularity = team.sum().sort_values(ascending=False)
popularity.drop_duplicates(inplace=True)
popularity.dropna(inplace=True)
# Print the popularity of each genre
popularity.head()

# Setting the plot's figure size
fig, ax = plt.subplots(figsize=(10, 5))

# Plot the horizontal bar chart
sns.barplot(x=popularity[:10].values,
            y=popularity[:10].index, palette="Blues_r", ax=ax)

# Add labels showing the popularity of each genre
for i, v in enumerate(popularity[:10].values):
    ax.text(v + 3, i, str(f"{v} Games"),
            color="black", ha="center", va="center")

# Set the title and axis labels
ax.set_title("Top 10 Team Developers By Number of Titles", fontsize=15)
ax.set_xlabel("")
ax.set_ylabel("")

# Increase the fontsize of the y-axis tick labels
ax.tick_params(axis="y", labelsize=11)

# Remove spines from the right and top sides of the plot
sns.despine(right=True, top=True, bottom=True)

# Show the plot
plt.show()

avg_rating_by_team = df.groupby("Team")["Rating"].mean()

# Get the top 10 teams by average rating
top_teams = avg_rating_by_team.nlargest(10).index.tolist()

# Filter the dataset to include only games by the top 10 teams
top_teams_games = df[df["Team"].isin(top_teams)]

# Create a dictionary mapping each team to a unique color
# team_colors = {team: f"C{i}" for i, team in enumerate(top_teams)}
# Create a timeline plot of the top 10 teams' games
fig, ax = plt.subplots(figsize=(12, 8))

sns.scatterplot(data=top_teams_games, x="Release Date",
                y="Team", hue="Title", s=150, alpha=0.8, ax=ax)

# Set the title and axis labels
ax.set_title(
    "Top 10 Teams by Average Rating and Their Games Timeline", fontsize=18)
ax.set_xlabel("Release Date", fontsize=14)
ax.set_ylabel("Team", fontsize=14)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=12)

# Add a grid to the plot
ax.grid(axis="y", alpha=0.3)

# Show the plot
plt.show()

# Loop through each row in the DataFrame
for index, row in df.iterrows():

    # Get the value of the 'Column' column for the current row
    column_value = row['Team']

    # Split the column value by ',' into separate 'developer' and 'publisher' values

    if ',' in column_value:
        developer, publisher = column_value.split(',', maxsplit=1)
    else:
        developer, publisher = column_value, column_value

    # Update the 'Developer' and 'Publisher' columns for the current row
    df.at[index, 'Developer'] = developer.strip()
    df.at[index, 'Publisher'] = publisher.strip()

# Drop the original 'Column' column
df.drop('Team', axis=1, inplace=True)

df["Developer"] = df["Developer"].str.replace('[\[\]\'\"]', "")
df["Publisher"] = df["Publisher"].str.replace('[\[\]\'\"]', "")
publish = df["Publisher"].str.get_dummies(",")

# Sum the occurrences of each genre across all games
popularity = publish.sum().sort_values(ascending=False)

# Print the popularity of each genre
popularity.head()

counts = df.groupby(['Developer', 'Publisher']).size().reset_index(name='Count').sort_values(by='Count',
                                                                                             ascending=False)[:5]

# Merge counts dataframe with the original dataframe to get the release date for each game
merged_df = pd.merge(counts, df[['Developer', 'Publisher', 'Release Date']], on=['Developer', 'Publisher'], how='left')

# Create a timeline plot of the top 5 counts
fig, ax = plt.subplots(figsize=(12, 8))

sns.scatterplot(data=merged_df, x="Release Date", y="Publisher", hue="Developer", s=150, alpha=0.8, ax=ax)

# Set the title and axis labels
ax.set_title("Top 5 Developer-Publisher Pairs by Publisher and Their Games Timeline", fontsize=18)
ax.set_xlabel("Release Date", fontsize=14)
ax.set_ylabel("Publisher", fontsize=14)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=12)

# Add a grid to the plot
ax.grid(axis="y", alpha=0.3)

# Show the plot
plt.show()


#Steffan's  Contributions:

# convert Plays column to numeric data type
df['Plays'] = pd.to_numeric(df['Plays'], errors='coerce')

# fill NaN values with 0
df['Plays'] = df['Plays'].fillna(0)

# convert Playing column to numeric data type
df['Playing'] = pd.to_numeric(df['Playing'], errors='coerce')
# fill NaN values with 0
df['Playing'] = df['Playing'].fillna(0)

#Had to drop duplicates again due to more data being inputted
df.drop_duplicates(inplace=True)


#GETS THE TOP 10 GAME TITLES AND THEIR RATINGS
#And shows their release dates in the legend

top_10_game_of_all_time = df[['Title', 'Rating', 'Release Date']].sort_values(by = 'Rating', ascending = False).nlargest(10, columns='Rating')

print(top_10_game_of_all_time)

fig, axes = plt.subplots(1, figsize=(16, 5))
sns.barplot(ax=axes, data=top_10_game_of_all_time, x='Rating', y='Title', palette='GnBu')
axes.set_title('Top rated Games', pad=10, fontsize=15)
axes.set_xlabel('Rating', labelpad=20)
axes.set_ylabel('Title', labelpad=20)
plt.legend(top_10_game_of_all_time['Release Date'])
plt.show()

#BOTTOM 10 PERFORMING GAMES
#Drops the ratings that are equal to 0
#The reason why i did this is because almost all of the
#0 rated games have not been released yet so there are no ratings/reviews

df.drop(df[df['Rating'] == 0].index, inplace=True)
#Had to specifically drop these two columns since the
#.drop_duplicates(inplace=True) method was not dropping
#them for some reason. These are duplicate columns.
#df.drop(547, inplace=True)
#df.drop(761, inplace=True)

top_10_worst_game_of_all_time = df[['Title', 'Rating', 'Release Date']].sort_values(by = 'Rating', ascending = False).nsmallest(10, columns='Rating')

print(top_10_worst_game_of_all_time)

fig, axes = plt.subplots(1, figsize=(16, 6))
sns.barplot(ax=axes, data=top_10_worst_game_of_all_time, x='Title', y='Rating', palette='YlOrBr')
axes.set_title('Least rated Games', pad=10, fontsize=15)
axes.set_xlabel('Title', labelpad=100)
axes.set_ylabel('Rating', labelpad=100)
plt.legend(top_10_worst_game_of_all_time['Release Date'])
fig.tight_layout()
plt.show()



####GRAPHS THE GAMES WITH THE MOST PLAYS
#THE DATA POINTS ARE THE TOP 10 MOST PLAYED
#GAMES (not the top 10 highest rated games)
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
top_played_game_of_all_time = df[['Title', 'Rating', 'Plays']].sort_values(by = 'Plays', ascending = False).nlargest(10, columns='Plays')


fig, ax = plt.subplots(figsize=(12, 8))

sns.scatterplot(data=top_played_game_of_all_time, x="Rating",
                y="Plays", hue="Title", s=150, alpha=0.8, ax=ax)

# Set the title and axis labels
ax.set_title(
    "Top 10 Most Played Games and Their Ratings", fontsize=18)
ax.set_xlabel("Rating", fontsize=14)
ax.set_ylabel("Plays", fontsize=14)
ax.legend(bbox_to_anchor=(0.8, 0.8), loc="upper left", fontsize=12)

# Add a grid to the plot
ax.grid(axis="y", alpha=0.3)

# Show the plot
plt.show()

print(top_played_game_of_all_time)

##################################
#Graphs the games with the most people that are still playing them
#Seems like some duplicates are still in the dataframe
#will need to readjust
top_playing_game_of_all_time = df[['Title', 'Rating', 'Playing']].sort_values(by = 'Playing', ascending = False).nlargest(10, columns='Playing')


fig, ax = plt.subplots(figsize=(12, 8))

sns.scatterplot(data=top_playing_game_of_all_time, x="Rating",
                y="Playing", hue="Title", s=150, alpha=0.8, ax=ax)

# Set the title and axis labels
ax.set_title(
    "Top 10 Games That are still being played and their Ratings", fontsize=18)
ax.set_xlabel("Rating", fontsize=14)
ax.set_ylabel("Playing", fontsize=14)
ax.legend(bbox_to_anchor=(0.8, 0.8), loc="upper left", fontsize=12)

# Add a grid to the plot
ax.grid(axis="y", alpha=0.3)

# Show the plot
plt.show()

print(top_playing_game_of_all_time)






