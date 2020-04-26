import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz

# songs_filename = 'song_data.csv'
# ratings_filename = '10000.txt'
#
# # read data
# df_songs = pd.read_csv(
#     songs_filename,
#     usecols=['song_id', 'title', 'release'],
#     dtype={'song_id': 'str', 'title': 'str', 'release': 'str'},
#     encoding='latin-1')
#
# df_ratings = pd.read_csv(
#     ratings_filename,
#     sep="	")
# df_ratings.columns = ["user_id", "song_id", "rating"]


# Files we are going to work with
songs_filename = 'https://static.turi.com/datasets/millionsong/song_data.csv'
ratings_filename = 'https://static.turi.com/datasets/millionsong/10000.txt'

df_ratings = pd.read_table(ratings_filename, header=None)
df_ratings.columns = ['user_id', 'song_id', 'rating']

# read data
df_songs = pd.read_csv(
    songs_filename,
    usecols=['song_id', 'title', 'release'],
    dtype={'song_id': 'str', 'title': 'str', 'release': 'str'},
    encoding='latin-1')

# Remove duplicates
#song_df = pd.merge(df_ratings, df_songs.drop_duplicates(['song_id']), on="song_id", how="left")

'''
print(print(song_df.head()))

              song_id              title                               release
0  SOQMMHC12AB0180CB8       Silent Night                 Monster Ballads X-Mas
1  SOVFVAK12A8C1350D9        Tanssi vaan                          KarkuteillÃ¤
2  SOGTUKN12AB017F4F1  No One Could Ever                                Butter
3  SOBNYVR12A8C13558C     Si Vos QuerÃ©s                               De Culo
4  SOHSBXH12A8C13B0DF   Tangle Of Aspens  Rene Ablaze Presents Winter Sessions

'''

# pivot ratings into song features
df_song_features = df_ratings.pivot(
    index='song_id',
    columns='user_id',
    values='rating'
).fillna(0)

print(df_song_features.head())

#
# mat_song_features = csr_matrix(df_song_features.values)
#
# model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
#
# num_users = len(df_ratings.user_id.unique())
# num_items = len(df_ratings.song_id.unique())
#
# # get rating frequency
# # number of ratings each song got.
# df_songs_cnt = pd.DataFrame(df_ratings.groupby('song_id').size(), columns=['count'])
# # df_songs_cnt.head()
#
# # now we need to take only songs that have been rated atleast 50 times to get some idea of the reactions of users towards it
#
# popularity_thres = 50
# popular_songs = list(set(df_songs_cnt.query('count >= @popularity_thres').index))
# df_ratings_drop_songs = df_ratings[df_ratings.song_id.isin(popular_songs)]
#
# # get number of ratings given by every user
# df_users_cnt = pd.DataFrame(df_ratings_drop_songs.groupby('user_id').size(), columns=['count'])
#
# # filter data to come to an approximation of user likings.
# ratings_thres = 50
# active_users = list(set(df_users_cnt.query('count >= @ratings_thres').index))
# df_ratings_drop_users = df_ratings_drop_songs[df_ratings_drop_songs.user_id.isin(active_users)]
#
# # pivot and create song-user matrix
# song_user_mat = df_ratings_drop_users.pivot(index='song_id', columns='user_id', values='rating').fillna(0)
# # map song titles to images
# # print(song_user_mat.head())
# # song_to_idx = {
# #    song: i for i, song in
# #    enumerate(list(df_songs.set_index('song_id').loc[song_user_mat.index].title))
# # }
# # for x in range(0,len(song_user_mat.index(song_id))
# data1 = list(df_songs.set_index('song_id').loc[song_user_mat.index].title)
# song_to_idx = {
#     song: i for i, song in
#     enumerate(data1)
# }
# print("THIS IS FROM ENUMERATE")
# lst = data1
#
#
# # for x in range(0,len(lst)):
# #    for y in range(x,x+1):
# #	    lst.insert(x+1+y, x)
#
# def Convert(lst):
#     res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)}
#     return res_dct
#
#
# # Driver code
#
# # print(Convert(lst))
# d = {}
# x = 0
# for item in lst:
#     d[item] = x
#     x = x + 1
#     print(x)
# print(d)
# song_to_idx = d
# # transform matrix to scipy sparse matrix
# song_user_mat_sparse = csr_matrix(song_user_mat.values)
#
# # define model
# model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
# # fit
# model_knn.fit(song_user_mat_sparse)
#
# # In[23]:
#
#
# # In[24]:
#
#
# def fuzzy_matching(mapper, fav_song, verbose=True):
#     # matches the song given by the user with the songs in database
#     # returns the closest mathces
#
#     # verbose: bool, prints log for True
#
#     match_tuple = []
#     # get match
#     for title, idx in mapper.items():
#         ratio = fuzz.ratio(title.lower(), fav_song.lower())
#         if ratio >= 60:
#             match_tuple.append((title, idx, ratio))
#     # sort
#     match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
#     if not match_tuple:
#         print('Oops! No match is found')
#         return
#     if verbose:
#         print('Found possible matches in our database: {0}\n'.format([x[0] for x in match_tuple]))
#     return match_tuple[0][1]
#
#
# def make_recommendation(model_knn, data, mapper, fav_song, n_recommendations):
#     """
#     return top n similar song recommendations based on user's input song
#
#
#     Parameters
#     ----------
#     model_knn: sklearn model, knn model
#
#     data: song-user matrix
#
#     mapper: dict, map song title name to index of the song in data
#
#     fav_song: str, name of user input song
#
#     n_recommendations: int, top n recommendations
#
#     Return
#     ------
#     list of top n similar song recommendations
#     """
#     # fit
#     model_knn.fit(data)
#     # get input song index
#     print('You have input song:', fav_song)
#     idx = fuzzy_matching(mapper, fav_song, verbose=True)
#
#     print('Recommendation system start to make inference')
#     print('......\n')
#     distances, indices = model_knn.kneighbors(data[idx], n_neighbors=n_recommendations + 1)
#
#     raw_recommends = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[
#                      :0:-1]
#     # get reverse mapper
#     reverse_mapper = {v: k for k, v in mapper.items()}
#     # print recommendations
#     print('Recommendations for {}:'.format(fav_song))
#     for i, (idx, dist) in enumerate(raw_recommends):
#         print('{0}: {1}, with distance of {2}'.format(i + 1, reverse_mapper[idx], dist))
#
#
# # In[26]:
#
#
# my_favorite = "I have seen"  # sys.argv[1]
#
# make_recommendation(
#     model_knn=model_knn,
#     data=song_user_mat_sparse,
#     fav_song=my_favorite,
#     mapper=song_to_idx,
#     n_recommendations=10)
#
# # In[27]:

