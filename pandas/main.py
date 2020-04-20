import pandas as pandas
from sklearn.model_selection import train_test_split
import Recommenders

# Files we are going to work with
triplets_file = 'https://static.turi.com/datasets/millionsong/10000.txt'
songs_metadata_file = 'https://static.turi.com/datasets/millionsong/song_data.csv'

# Read the table of triplet_file using pandas and define the 3 columns as user_id, song_id and listen_count
# ( df here means dataframe)
song_df_1 = pandas.read_table(triplets_file, header=None)
song_df_1.columns = ['user_id', 'song_id', 'listen_count']

# Read the metadat_file and combine the metadata_file with triplets_file. We drop the duplicates between
# 2 datasets using song_id
song_df_2 = pandas.read_csv(songs_metadata_file)
song_df = pandas.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")

print(song_df.head())

# # We select a subset of this data (the first 10,000 songs). We then merge the song and artist_name
# # into one column, aggregated by number of time a particular song is listened too in general by all users.
# song_grouped = song_df.groupby(['song']).agg({'listen_count': 'count'}).reset_index()
# grouped_sum = song_grouped['listen_count'].sum()
# song_grouped['percentage'] = song_grouped['listen_count'].div(grouped_sum) * 100
# song_grouped.sort_values(['listen_count', 'song'], ascending=[0, 1])
#
# # We count the number of unique users and songs in our subset of data
# users = song_df['user_id'].unique()
# len(users)  # return 365 unique users
# songs = song_df['song'].unique()
# len(songs)  # return 5151 unique songs
#
# # We then create a song recommender by splitting our dataset into training and testing data.
# train_data, test_data = train_test_split(song_df, test_size=0.20, random_state=0)
#
# # We then used a popularity based recommender class as a blackbox to train our model.
# pm = Recommenders.popularity_recommender_py()
# pm.create(train_data, 'user_id', 'song')
#
# # User the popularity model to make some prediction
# user_id = users[5]
# pm.recommend(user_id)
#
# print(user_id)
