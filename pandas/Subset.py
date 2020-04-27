import pandas

class Subset:

    def createSubset(self):
        triplets_file = 'https://static.turi.com/datasets/millionsong/10000.txt'
        songs_metadata_file = 'https://static.turi.com/datasets/millionsong/song_data.csv'

        # Read the table of triplet_file using pandas and define the 3 columns as user_id, song_id and listen_count
        # ( df here means dataframe)
        song_df_1 = pandas.read_table(triplets_file, header=None)
        song_df_1.columns = ['user_id', 'song_id', 'listen_count']

        '''print(song_df_1.head())
                                            user_id             song_id  listen_count
        0  b80344d063b5ccb3212f76538f3d9e43d87dca9e  SOAKIMP12A8C130995             1
        1  b80344d063b5ccb3212f76538f3d9e43d87dca9e  SOBBMDR12A8C13253B             2
        2  b80344d063b5ccb3212f76538f3d9e43d87dca9e  SOBXHDL12A81C204C0             1
        3  b80344d063b5ccb3212f76538f3d9e43d87dca9e  SOBYHAJ12A6701BF1D             1
        4  b80344d063b5ccb3212f76538f3d9e43d87dca9e  SODACBL12A8C13C273             1'''


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

        '''
        print(song_df_2.head());
           song_id              title                               release       artist_name  year
        0  SOQMMHC12AB0180CB8       Silent Night                 Monster Ballads X-Mas  Faster Pussy cat  2003
        1  SOVFVAK12A8C1350D9        Tanssi vaan                           Karkuteillä  Karkkiautomaatti  1995
        2  SOGTUKN12AB017F4F1  No One Could Ever                                Butter    Hudson Mohawke  2006
        3  SOBNYVR12A8C13558C      Si Vos Quer2és                               De Culo       Yerba Brava  2003
        4  SOHSBXH12A8C13B0DF   Tangle Of Aspens  Rene Ablaze Presents Winter Sessions        Der Mystic     0
        '''

        # We create a subset of the dataset with 10k rows
        song_df = song_df.head(10000)

        # Export subdataset
        song_df.to_csv(r'/dataset/subset.csv', index=False, header=True)