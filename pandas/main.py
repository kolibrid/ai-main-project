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

'''
print(song_df.head())
                                    user_id             song_id  listen_count            title                        release    artist_name  year
0  b80344d063b5ccb3212f76538f3d9e43d87dca9e  SOAKIMP12A8C130995             1         The Cove             Thicker Than Water   Jack Johnson     0
1  b80344d063b5ccb3212f76538f3d9e43d87dca9e  SOBBMDR12A8C13253B             2  Entre Dos Aguas            Flamenco Para Niños  Paco De Lucia  1976
2  b80344d063b5ccb3212f76538f3d9e43d87dca9e  SOBXHDL12A81C204C0             1         Stronger                     Graduation     Kanye West  2007

'''

# This step is needed and is not in the example
song_df['song'] = song_df["artist_name"] + ' - ' + song_df["title"]

# We select a subset of this data (the first 10,000 songs). We then merge the song and artist_name
# into one column, aggregated by number of time a particular song is listened too in general by all users.
song_grouped = song_df.groupby(['song']).agg({'listen_count': 'count'}).reset_index()
grouped_sum = song_grouped['listen_count'].sum()
song_grouped['percentage'] = song_grouped['listen_count'].div(grouped_sum) * 100
song_grouped.sort_values(['listen_count', 'song'], ascending=[0, 1])

'''
print(song_grouped)

                                                   song  listen_count  percentage
0                                      !!! - Sweet Life            90     0.00450
1     'N Sync/Phil Collins - Trashin' The Camp (Phil...            94     0.00470
2             + / - {Plus/Minus} - The Queen of Nothing           314     0.01570
3                                     +44 - Lycanthrope            96     0.00480
4                                  +44 - Make You Smile           130     0.00650
...                                                 ...           ...         ...
9948                       the bird and the bee - Witch           298     0.01490
9949                the bird and the bee - You're A Cad           317     0.01585
9950                     themselves - Skinning the Drum           108     0.00540
9951                        tobyMac - City On Our Knees            88     0.00440
9952  tobyMac - Lose My Soul feat. Kirk Franklin & M...            79     0.00395


[9953 rows x 3 columns]
'''

# We count the number of unique users and songs in our subset of data
users = song_df['user_id'].unique()
print(len(users))  # return 76353 unique users
songs = song_df['song'].unique()
print(len(songs))  # return 9953 unique songs

# We then create a song recommender by splitting our dataset into training and testing data.
train_data, test_data = train_test_split(song_df, test_size=0.20, random_state=0)

# We then used a popularity based recommender class as a blackbox to train our model.
pm = Recommenders.popularity_recommender_py()
pm.create(train_data, 'user_id', 'title')

# User the popularity model to make some prediction
user_id = users[5]
pm.recommend(user_id)

print(user_id)
