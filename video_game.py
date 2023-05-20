
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate, train_test_split


# import the data
df = pd.read_csv('dataset\\games.csv')
df.drop_duplicates(inplace=True)
df.drop_duplicates(subset=['Title'], inplace=True)
df.head()

# data cleaning and exploring
print(df.isnull().sum())
df['Team'].fillna('Unknown', inplace=True)
df['Summary'].fillna('Unknown', inplace=True)
df = df.fillna(method='ffill')

print(df.isnull().sum())
column_names = df.columns.tolist()
print(column_names)
df = df.drop('Unnamed: 0', axis=1)
# Combine Teams with same name
df.loc[df['Team'] == "['FromSoftware', 'Sony Computer Entertainment']", 'Team'] += ', ' + \
    df.loc[df['Team'] ==
           "['Sony Computer Entertainment', 'FromSoftware']", 'Team'].iloc[0]
# Delete duplicate row
df = df[df['Team'] != "['Sony Computer Entertainment', 'FromSoftware']"]
# Change name of the Team
df['Team'] = df['Team'].replace("['FromSoftware', 'Sony Computer Entertainment'], ['Sony Computer Entertainment', 'FromSoftware']",
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
                           rating_std, rating_mean + num_std*rating_std, 100)

# Select only the data that falls within the specified range
selected_data = df[(df['Rating'] >= rating_range[0]) &
                   (df['Rating'] <= rating_range[-1])]

# Get top 3 games on each side of the mean
top_games_positive = selected_data.sort_values(
    by='Rating', ascending=False).drop_duplicates(subset=['Title']).iloc[:3]
top_games_negative = selected_data.sort_values(
    by='Rating', ascending=True).drop_duplicates(subset=['Title']).iloc[:3]
# Get top 3 games that equal the mean
top_games_at_mean = df[(df['Rating'] >= rating_mean - rating_std/2) & (df['Rating']
                                                                       <= rating_mean + rating_std/2)].drop_duplicates(subset=['Title']).iloc[:3]

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
df['Plays'] = df['Plays'].fillna(0, inplace=True)
data = df[['Team', 'Plays']].dropna(inplace=True)


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

counts = df.groupby(['Developer', 'Publisher']).size().reset_index(
    name='Count').sort_values(by='Count', ascending=False)[:5]

# Merge counts dataframe with the original dataframe to get the release date for each game
merged_df = pd.merge(counts, df[['Developer', 'Publisher', 'Release Date']], on=[
                     'Developer', 'Publisher'], how='left')

# Create a timeline plot of the top 5 counts
fig, ax = plt.subplots(figsize=(12, 8))

sns.scatterplot(data=merged_df, x="Release Date", y="Publisher",
                hue="Developer", s=150, alpha=0.8, ax=ax)

# Set the title and axis labels
ax.set_title(
    "Top 5 Developer-Publisher Pairs by Publisher and Their Games Timeline", fontsize=18)
ax.set_xlabel("Release Date", fontsize=14)
ax.set_ylabel("Publisher", fontsize=14)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=12)

# Add a grid to the plot
ax.grid(axis="y", alpha=0.3)

# Show the plot
plt.show()

# Remove games with missing ratings
df = df.dropna(subset=['Rating'])
# Drop rows with NaN values
df.dropna(subset=['Release Date'],inplace=True)
# Convert the Release Date column to Unix timestamp values
df['Release Date'] = df['Release Date'].apply(lambda x: int(x.timestamp()))

# Define the input and output variables
X = df[['Release Date', 'Rating']].values.reshape(-1,2)
y = (df['Rating'] > 4)

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest regressor model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict the release date for a high rated game in the future
next_rating = 4.5  # set the expected rating of the next high rated game
next_release_date = pd.to_datetime('2024-01-01')  # set a future date for the prediction input
next_release_date_unix = int(next_release_date.timestamp())
next_release_date_predicted = pd.to_datetime(rf.predict([[next_release_date_unix, next_rating]])[0], unit="s")
print(f'The predicted release date for the next high rated game (with rating {next_rating}) is {next_release_date_predicted}.')





# new_df = df.loc[:, ['Title', 'Rating', 'Release Date']].mean()
# new_df.head()
# # Specify the release date
# release_date = '2022-02-25'

# # Filter the dataframe to only include games with the specified release date
# games = df[df['Release Date'] == release_date]


# # Print the games with the specified release date
# games.head()
