import pandas as pandas
from sklearn.model_selection import train_test_split
import Recommenders
import random

# We read the subset
song_df = pandas.read_csv('dataset/subset.csv')

# Aggregate by number of time a particular song is listened too in general by all users.
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

'''
FIRST PART
Simple song recommendation system
'''
# We count the number of unique users and songs in our subset of data
users = song_df['user_id'].unique()
print("There are ", len(users), " unique users in the system.")  # return 365 unique users
songs = song_df['song'].unique()
print("There are ", len(songs), " unique songs in the system.")  # return 5151 unique songs

# We then create a song recommender by splitting our dataset into training and testing data.
train_data, test_data = train_test_split(song_df, test_size=0.20, random_state=0)

# We then used a popularity based recommender class as a blackbox to train our model.
pm = Recommenders.popularity_recommender_py()
pm.create(train_data, 'user_id', 'song')

# User the popularity model to make some prediction
user_id = random.sample(list(users), 1)
user_id = user_id[0]

popularity = pm.recommend(user_id)

print("\nWe will proceed now to make song recommendations using the popularity model.")
print("\nThe user id chosen for the popularity recommendation is ", user_id)
print("The recommendation is:")
print(popularity)

'''
SECOND PART
Personalized song recommendation system
'''

print("\nWe will proceed now to make song recommendations using the co-occurrence matrix model, which personalizes the recommendation to each user.")

is_model = Recommenders.item_similarity_recommender_py()
is_model.create(train_data, 'user_id', 'song')

#Print the songs for the user in training data
user_items = is_model.get_user_items(user_id)
#
print("\n------------------------------------------------------------------------------------")
print("Training data songs for the user userid: %s:" % user_id)
print("------------------------------------------------------------------------------------")

for user_item in user_items:
    print(user_item)

print("----------------------------------------------------------------------")
print("Recommendation process going on:")
print("----------------------------------------------------------------------")

#Recommend songs for the user using personalized model
recommendations = is_model.recommend(user_id)

# We print the recommendations
print(recommendations)

# is_model.get_similar_items(['U Smile - Justin Bieber'])
