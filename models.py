#!/usr/bin/env python
# coding: utf-8

# In[1]:
import streamlit as st
import pandas as pd
import warnings

import numpy as np
import time
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import math as mt
from scipy.sparse.linalg import * #used for matrix multiplication
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix
from scipy.stats import skew, norm, probplot
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
# from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
image = Image.open('logo.png')

st.image(image)

st.write(""" ### Hybrid Music Recommendation System """)
# Object notation


def get_data():
    path = "model2.csv"
    return pd.read_csv(path)

dataget = get_data()
#data
ti = dataget['title'].drop_duplicates(keep='last')
pn_choice = st.sidebar.selectbox("Select your song:", ti)
songid = dataget["ID"].loc[dataget["title"] == pn_choice]


if st.sidebar.button('Search'):
     st.write('Searching,...!')
else:
     st.write('')


st.sidebar.markdown("""---""")


if st.sidebar.button('About Project'):
    st.sidebar.image(image)
    st.sidebar.write("""
                    ##### Developed By: Hanan Hameed
                    ##### Supervised By: Assist. Prof. Dr. Mohammed Salih
                    ##### Informatics Inistitute for Postgraduate Studies - IIPS
                    """)
else:
     st.write('')
     

st.write("---")

st.write(""" ### Model 1 -  Popularity Recommendation""")
# model one dataset
data = pd.read_csv("model1.csv")

#model one recommendation
def create_popularity_recommendation(data, user_id, song_id, n=10):
    #Get a count of user_ids for each unique song as recommendation score
    data_grouped = data.groupby([song_id]).agg({user_id: 'count'}).reset_index()
    print(data_grouped)
    data_grouped.rename(columns = {user_id: 'score'},inplace=True)
    print(data_grouped)
    #Sort the songs based upon recommendation score
    data_sort = data_grouped.sort_values(['score', song_id], ascending = [0,1])
    
    #Generate a recommendation rank based upon score
    data_sort['Rank'] = data_sort.score.rank(ascending=False, method='first')
    print(data_sort) 
    #Get the top n recommendations
    popularity_recommendations = data_sort.head(n)
    return popularity_recommendations

recommendations = create_popularity_recommendation(data,'user_id','title', 20)
st.table(recommendations.head(10))



st.markdown("""---""")

st.write(""" ### Model 2 -  # Collaborative filtering Recommendation""")

# model three dataset
data2 = pd.read_csv("model2.csv")


Xcol = data2.iloc[0:,[12,13,14,15,16]]

#find cosine similarity and most similar items
cosine_similarities = cosine_similarity(Xcol)
results = {}
for index,value in data2.iloc[0:,].iterrows(): #iterates through all the rows
    similar_indices = cosine_similarities[index].argsort()[:-6:-1]
    similar_items = [(cosine_similarities[index][i], data2['ID'][i]) for i in similar_indices]
    results[value['ID']] = similar_items[1:]
    #print (cosine_similarities)
    
#Model 2 build
def artist(id):
    return data2.loc[data2['ID'] == id]['Artist'].tolist()[0]
def song(id):
    return data2.loc[data2['ID'] == id]['title'].tolist()[0]
def recommend(id, num):
    if (num == 0):
        st.write("Unable to recommend any songs as you have not chosen the number of songs to be recommended")    
    else :
        st.write("According to your choice similar song to "  + song(id) + " by " + artist(id) )
    st.write("This is the Recommended song list")
    recs = results[id][:num]
    i=0
    for rec in recs:
        i+=1
        st.write(str(i) +" - "+ song(rec[1]) + " By " + artist(rec[1]) + " (score:" + str(rec[0]) + ")")
    rsl=pd.DataFrame(recs,columns=["similarity score","song_id"])
    rsl["song_name"] =   data2.loc[data2.index.values, "Artist"]
    #return rsl.head()

recommend(int(songid),4)



st.markdown("""---""")
st.write(""" ### Model 2 -  # Content-Based Recommendation""")

# model three dataset
data3 = pd.read_csv("model3.csv")

#model 3
tfidf_cb = TfidfVectorizer(analyzer='word', stop_words='english')
lyrics_matrix_cb = tfidf_cb.fit_transform(data3['Lyrics'])
cosine_similarities_cb = cosine_similarity(lyrics_matrix_cb)
similarities_cb = {}



for i in range(len(cosine_similarities_cb)):
    # Now we'll sort each element in cosine_similarities and get the indexes of the songs. 
    similar_indices_cb = cosine_similarities_cb[i].argsort()[:-50:-1] 
    # After that, we'll store in similarities each name of the 50 most similar songs.
    # Except the first one that is the same song.
    similarities_cb[data3['title'].iloc[i]] = [(cosine_similarities_cb[i][x], data3['title'][x], data3['Artist'][x]) for x in similar_indices_cb][1:]


class ContentBasedRecommender:
    def __init__(self, matrix):
        self.matrix_similar = matrix

    def _print_message(self, song, recom_song):
        rec_items = len(recom_song)
        
        print(f'The {rec_items} recommended songs for {song} are:')
        for i in range(rec_items):
            st.write(f"Number {i+1}:")
            st.write(f"{recom_song[i][1]} by {recom_song[i][2]} with {round(recom_song[i][0], 3)} similarity score") 
            
    def recommend(self, recommendation_cb):
        # Get song to find recommendations for
        song = recommendation_cb['song']
        # Get number of songs to recommend
        number_songs = recommendation_cb['number_songs']
        # Get the number of songs most similars from matrix similarities
        recom_song = self.matrix_similar[song][:number_songs]
        # print each item
        self._print_message(song=song, recom_song=recom_song)

recommedations_cb = ContentBasedRecommender(similarities_cb)

recommendation_cb = {
    "song": data3['title'].iloc[int(songid)],
    "number_songs": 4
}

print(songid)
m3 = recommedations_cb.recommend(recommendation_cb)



st.markdown("""---""")

"""
##### Developed By: Hanan Hameed
##### Supervised By: Assist. Prof. Dr. Mohammed Salih
##### Informatics Inistitute for Postgraduate Studies - IIPS
"""
