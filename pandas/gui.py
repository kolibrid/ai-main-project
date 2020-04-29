import random
from tkinter import *
import tkinter.ttk as ttk
from sklearn.model_selection import train_test_split
import pandas
import Recommenders

'''
Generate Data
'''
song_df = pandas.read_csv('dataset/subset.csv')

song_df['song'] = song_df['title'].map(str) + " - " + song_df['artist_name']
song_grouped = song_df.groupby(['song']).agg({'listen_count': 'count'}).reset_index()
grouped_sum = song_grouped['listen_count'].sum()
song_grouped['percentage'] = song_grouped['listen_count'].div(grouped_sum) * 100
song_grouped.sort_values(['listen_count', 'song'], ascending=[0, 1])

all_users = song_df['user_id'].unique()

train_data, test_data = train_test_split(song_df, test_size=0.20, random_state=0)

is_model = Recommenders.item_similarity_recommender_py()
is_model.create(train_data, 'user_id', 'song')

#Print the songs for a random user in training data
user_id = random.sample(list(all_users), 1)
user_id = user_id[0]
user_items = is_model.get_user_items(user_id)

#Recommend songs for the user using personalized model
recommendations = is_model.recommend(user_id)

'''
GUI logic
'''

# Create canvas
root = Tk()
root.title("Song Recommendation System")
width = 1024
height = 700
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width/2) - (width/2)
y = (screen_height/2) - (height/2)
root.geometry("%dx%d+%d+%d" % (width, height, x, y))
root.resizable(True, True)

# Title
T1 = Text(root, height=1)
T1.tag_configure("center", justify='center')
T1.insert(END, "Song Recommendation System")
T1.tag_add("center", "1.0", "end")
T1.pack()

# Show songs listened by the user text
T2 = Text(root, height=1)
T2.tag_configure("left", justify='left')
T2.insert(END, "The user " + user_id + " has listened to " + str(len(user_items)) + " songs.")
T2.pack()

# Show list text
T3 = Text(root, height=1)
T3.tag_configure("left", justify='left')
T3.insert(END, "List of songs listened by the user:")
T3.pack()

# Create scrollable container for the list of songs since it can be long
container = Frame(root)
canvas = Canvas(container)
scrollbar = Scrollbar(container, orient="vertical", command=canvas.yview)
scrollable_frame = Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

canvas.configure(yscrollcommand=scrollbar.set)

for user_item in user_items:
    T4 = Text(scrollable_frame, height=1)
    T4.tag_configure("left", justify='left')
    T4.insert(END, user_item)
    T4.pack()

container.pack(fill="both", expand=True)
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# Recommended songs text
T5 = Text(root, height=2)
T5.tag_configure("left", justify='left')
T5.insert(END, "Recommended songs:")
T5.pack()

# Show table with the 10 top recommendation songs
TableMargin = Frame(root)
TableMargin.pack(side=TOP)
scrollbarx = Scrollbar(TableMargin, orient=HORIZONTAL)
scrollbary = Scrollbar(TableMargin, orient=VERTICAL)
tree = ttk.Treeview(TableMargin, columns=("UserID", "Song", "Score", "Rank"), height=100, selectmode="extended", yscrollcommand=scrollbary.set, xscrollcommand=scrollbarx.set)
scrollbary.config(command=tree.yview)
scrollbary.pack(side=RIGHT, fill=Y)
scrollbarx.config(command=tree.xview)
scrollbarx.pack(side=BOTTOM, fill=X)
tree.heading('UserID', text="UserID", anchor=W)
tree.heading('Song', text="Song", anchor=W)
tree.heading('Score', text="Score", anchor=W)
tree.heading('Rank', text="Rank", anchor=W)
tree.column('#0', stretch=NO, minwidth=0, width=0)
tree.column('#1', stretch=NO, minwidth=0, width=400)
tree.column('#2', stretch=NO, minwidth=0, width=200)
tree.column('#3', stretch=NO, minwidth=0, width=200)
tree.column('#4', stretch=NO, minwidth=0, width=50)
tree.pack()

# Reverse order (from rank 1 to 10)
recommendations = recommendations.iloc[::-1]

for index, row in recommendations.iterrows():
    userid = row['user_id']
    song = row['song']
    score = row['score']
    rank = row['rank']
    tree.insert("", 0, values=(userid, song, score, rank))

#============================INITIALIZATION==============================
if __name__ == '__main__':
    root.mainloop()