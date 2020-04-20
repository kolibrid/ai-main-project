## import libraries
##matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

## import keras models, layers and optimizers
from keras.models import Sequential, Model
from keras.layers import Embedding, Flatten, Dense, Dropout, concatenate, multiply, Input
from keras.optimizers import Adam

## read data
ratings = pd.read_csv("D:\Madu_files\Artifical Intelligence\Main Project\Tensorflow-Keras/ratings.csv")
books = pd.read_csv("D:\Madu_files\Artifical Intelligence\Main Project\Tensorflow-Keras/books.csv")

ratings.head()

## unisue users, books
n_users, n_books = len(ratings.user_id.unique()), len(ratings.book_id.unique())


print(f'The dataset includes {len(ratings)} ratings by {n_users} unique users on {n_books} unique books.')
## split the data to train and test dataframes
train, test = train_test_split(ratings, test_size=0.1)

print(f"The training and testing data include {len(train), len(test)} records.")

dim_embedddings = 30
bias = 1

# books
book_input = Input(shape=[1],name='Book')
book_embedding = Embedding(n_books+1, dim_embedddings, name="Book-Embedding")(book_input)
book_bias = Embedding(n_users + 1, bias, name="Book-Bias")(book_input)

# users
user_input = Input(shape=[1],name='User')
user_embedding = Embedding(n_users+1, dim_embedddings, name="User-Embedding")(user_input)
user_bias = Embedding(n_users + 1, bias, name="User-Bias")(user_input)


matrix_product = multiply([book_embedding, user_embedding])
matrix_product = Dropout(0.2)(matrix_product)

input_terms = concatenate([matrix_product, user_bias, book_bias])
input_terms = Flatten()(input_terms)

## add dense layers
dense_1 = Dense(50, activation="relu", name = "Dense1")(input_terms)
dense_1 = Dropout(0.2)(dense_1)
dense_2 = Dense(20, activation="relu", name = "Dense2")(dense_1)
dense_2 = Dropout(0.2)(dense_2)
result = Dense(1, activation='relu', name='Activation')(dense_2)

## define model with 2 inputs and 1 output
model_mf = Model(inputs=[book_input, user_input], outputs=result)

## show model summary
model_mf.summary()
   
## specify learning rate (or use the default)
#opt_adam = Adam(lr = 0.002)
opt_adam = tf.keras.optimizers.Adam(lr=0.002, amsgrad=True)


## compile model
model_mf.compile(optimizer = opt_adam, loss = ['mse'], metrics = ['mean_absolute_error'])

## fit model
history_mf = model_mf.fit([train['user_id'], train['book_id']],
                          train['rating'],
                          batch_size = 256,
                          validation_split = 0.005,
                          epochs = 4,
                          verbose = 0)

## show loss at each epoch
#pd.DataFrame(history_mf.history)

