#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import ast
import numpy as np
import tensorflow.compat.v1 as tf
import pickle

tf.disable_v2_behavior()


# In[6]:


from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#from google.colab import drive
#from surprise.model_selection import cross_validate
#drive.mount('/content/drive')


# Load Data

# In[12]:


movies = pd. read_csv('the-movies-dataset/movies_metadata.csv')
extlinks = pd.read_csv('the-movies-dataset/links_small.csv')
ratings = pd.read_csv('the-movies-dataset/ratings_small.csv')
extlinks.head()


# In[13]:


credits = pd.read_csv('the-movies-dataset/credits.csv')
keywords = pd.read_csv('the-movies-dataset/keywords.csv')
keywords.head()


# In[14]:


movies = movies.drop([19730, 29503, 35587])
extlinks = extlinks[extlinks['tmdbId'].notnull()]['tmdbId'].astype('int')


# In[15]:


movies['genres'] = movies['genres'].fillna('[]').apply(ast.literal_eval).apply(lambda genres: [genre['name'] for genre in genres] if isinstance(genres, list) else [])


# In[16]:


movies['id'] = movies['id'].astype('int')
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
movies = movies.merge(credits, on='id')
movies = movies.merge(keywords, on='id')


# In[17]:


movies_filtered = movies[movies['id'].isin(extlinks)]
movies_filtered.head()
movies_filtered.shape


# In[18]:


movies_filtered['cast'] = movies_filtered['cast'].apply(ast.literal_eval)
movies_filtered['crew'] = movies_filtered['crew'].apply(ast.literal_eval)
movies_filtered['keywords'] = movies_filtered['keywords'].apply(ast.literal_eval)


# In[19]:


movies_filtered['cast_size'] = movies_filtered['cast'].apply(lambda x: len(x))
movies_filtered['crew_size'] = movies_filtered['crew'].apply(lambda x: len(x))


# In[20]:


def getDirector(crew):
  for person in crew:
    if person['job'] == 'Director':
          return person['name']
  return np.nan
movies_filtered['director'] = movies_filtered['crew'].apply(getDirector)


# In[21]:


movies_filtered['cast']  = movies_filtered['cast'].apply(lambda cast: [person['name'] for person in cast] if isinstance(cast, list) else [])
movies_filtered['cast'] = movies_filtered['cast'].apply(lambda cast: cast[:3] if len(cast) >=3 else cast)
movies_filtered['cast']


# In[22]:


movies_filtered['keywords'] = movies_filtered['keywords'].apply(lambda keywords: [keyword['name'] for keyword in keywords] if isinstance(keywords, list) else [])
movies_filtered['keywords']


# In[23]:


movies_filtered['director'] = movies_filtered['director'].astype('str').apply(lambda name: str.lower(name.replace(" ", "")))
movies_filtered['director'] = movies_filtered['director'].apply(lambda name: [name, name, name])


# In[24]:


movies_filtered['director']


# In[25]:


keywords = movies_filtered.apply(lambda movie: pd.Series(movie['keywords']),axis=1).stack().reset_index(level=1, drop=True)
keywords = keywords.value_counts()


# In[26]:


keywords = keywords[keywords>1]
keywords.head()


# In[28]:


def filterKeywords(list):
  words = []
  for word in list:
    if word in keywords:
      words.append(word)
  return words


# In[29]:


movies_filtered['keywords'] = movies_filtered['keywords'].apply(filterKeywords)


# In[30]:


movies_filtered['keywords'] = movies_filtered['keywords'].apply(lambda movieKeywords: [SnowballStemmer('english').stem(word) for word in movieKeywords])


# In[31]:


movies_filtered['keywords'] = movies_filtered['keywords'].apply(lambda movieKeywords: [str.lower(word.replace(" ", "")) for word in movieKeywords])
movies_filtered['keywords']


# In[32]:


movies_filtered['document'] = movies_filtered['keywords'] + movies_filtered['cast'] + movies_filtered['director'] + movies_filtered['genres']
movies_filtered['document'] = movies_filtered['document'].apply(lambda x: ' '.join(x))
movies_filtered['document'] 


# In[33]:


movies_filtered['document'] 


# In[34]:


count_vector = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count_vector.fit_transform(movies_filtered['document'] )
similarity = cosine_similarity(count_matrix, count_matrix)


# In[35]:


movies_filtered = movies_filtered.reset_index()
titles = movies_filtered['title']
indices = pd.Series(movies_filtered.index, index=movies_filtered['title'])


# In[36]:


movies_filtered


# In[37]:


def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]


# In[38]:


get_recommendations('Iron Man').head(10)


# In[39]:


ratings.count()


# Convert String to numbers

# In[40]:


ratings.userId = ratings.userId.astype(str).astype(int)
ratings.movieId = ratings.movieId.astype(str).astype(int)
ratings.rating = ratings.rating.astype(str).astype(float)
ratings.timestamp = pd.to_datetime(ratings.timestamp.astype(int), unit='s')
ratings.head()


# In[41]:


len(movies_filtered)


# In[42]:


movies_filtered['List Index']=movies_filtered.index.astype(str).astype(int)
movies_filtered.head(6)


# In[43]:


data_combined=pd.merge(movies_filtered, ratings, left_on='id' , right_on='movieId')


# In[44]:


def getCombinedData():
  return data_combined


# In[45]:


data_combined.head()


# In[ ]:





# In[46]:


data_combined = data_combined[['movieId', 'List Index', 'userId', 'rating']]

data_combined['List Index']=data_combined['List Index'].astype(str).astype(int)
data_combined.head()


# Group By UserId

# In[47]:


user_group = data_combined.groupby('userId')


# In[48]:


user_group.head(50)


# In[49]:


n_users = user_group.first().shape[0]


# In[50]:


totalUsers=n_users
user_movie = [None]* n_users

for userId, curUser in user_group:
  temp = [0]* len(movies_filtered)

  for num, movie in curUser.iterrows():
    #print(movie)
    temp[int(movie['List Index'])] = movie['rating']/5.0
 # print(userId)
  user_movie[userId-1]=temp

  if totalUsers == 0:
    break
  totalUsers-=1


# In[51]:


# In[52]:


hidden_units=256
visible_units = len(movies_filtered)
visible = tf.placeholder("float", [visible_units])
hidden = tf.placeholder("float", [hidden_units])
w= tf.placeholder("float", [visible_units, hidden_units])


# Forward pass

# In[53]:


v0 = tf.placeholder("float", [None, visible_units])
_h0 = tf.nn.sigmoid(tf.matmul(v0, w)+hidden)
h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))


# Backward pass

# In[54]:


_v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(w))+visible)  
v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))
h1 = tf.nn.sigmoid(tf.matmul(v1, w)+ hidden)


# In[55]:


alpha = 0.5
positive_phase = tf.matmul(tf.transpose(v0), h0)
negative_phase = tf.matmul(tf.transpose(v1), h1)
contrastive_divergernce = positive_phase - negative_phase
contrastive_divergernce = contrastive_divergernce/tf.to_float(tf.shape(v0)[0])
update_w = w+ alpha * contrastive_divergernce
update_vb = visible + alpha * tf.reduce_mean(v0-v1, 0)
update_hb = hidden + alpha * tf.reduce_mean(h0-h1, 0)


# In[56]:


err = v0-v1
err_sum = tf.reduce_mean(err * err)


# In[57]:


cur_w = np.zeros([visible_units, hidden_units], np.float32)
cur_vb = np.zeros([visible_units], np.float32)
cur_hb = np.zeros([hidden_units], np.float32)

prev_w = np.zeros([visible_units, hidden_units], np.float32)
prev_vb = np.zeros([visible_units], np.float32)
prev_hb = np.zeros([hidden_units], np.float32)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[58]:


epochs=20
batchsize=20
errors=[]
for i in range(epochs):
  for start, end in zip(range(0, len(user_movie), batchsize), range(batchsize, len(user_movie), batchsize) ):
    batch = user_movie[start:end]
    cur_w = sess.run(update_w, feed_dict={v0 : batch, w : prev_w, visible: prev_vb , hidden : prev_hb})
    cur_vb = sess.run(update_vb, feed_dict={v0 : batch, w : prev_w, visible: prev_vb , hidden : prev_hb})
    cur_hb = sess.run(update_hb, feed_dict={v0 : batch, w : prev_w, visible: prev_vb , hidden : prev_hb})
    prev_w = cur_w
    prev_vb = prev_vb
    prev_hb = prev_hb
  errors.append(sess.run(err_sum, feed_dict={v0 : user_movie, w : cur_w, visible: cur_vb , hidden : cur_hb}))
  print(errors[-1])


# In[59]:


# In[60]:


testUserId =21
testUser = [user_movie[testUserId-1]]


# In[61]:


hh0 = tf.nn.sigmoid(tf.matmul(v0, w)+hidden)
vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(w))+visible)
feed = sess.run(hh0, feed_dict={v0 : testUser, w: prev_w, hidden : prev_hb})
rec = sess.run(vv1, feed_dict={hh0 : feed, w: prev_w, visible : prev_vb})


# In[62]:


scored_movies=movies_filtered
scored_movies['Recommended Scores']=rec[0]


# In[63]:


scored_movies.sort_values(['Recommended Scores'], ascending=False).head(10)


# In[64]:


def predict(userId):
  testUser = [user_movie[userId-1]]
  hh0 = tf.nn.sigmoid(tf.matmul(v0, w)+hidden)
  vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(w))+visible)
  feed = sess.run(hh0, feed_dict={v0 : testUser, w: prev_w, hidden : prev_hb})
  rec = sess.run(vv1, feed_dict={hh0 : feed, w: prev_w, visible : prev_vb})
  scored_movies['Recommended Scores']=rec[0]
  #recommendations = scored_movies.sort_values(['Recommended Scores'], ascending=False).head(10)
  #return recommendations
  return scored_movies


# In[65]:


def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan


# In[67]:


id_map = pd.read_csv('the-movies-dataset/links_small.csv')[['movieId', 'tmdbId']]
id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
id_map.columns = ['movieId', 'id']
id_map = id_map.merge(movies_filtered[['title', 'id']], on='id').set_index('title')


# In[68]:


indices_map = id_map.set_index('id')


# In[69]:


def hybrid(userId, title):
    idx = indices[title]
    tmdbId = id_map.loc[title]['id']
    #print(idx)
    movie_id = id_map.loc[title]['movieId']
    
    sim_scores = list(enumerate(similarity[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = movies_filtered.iloc[movie_indices][['title', 'id']]


    recommendations=predict(userId)
    
    #print(recommendations['id'].isin(movies['id']))
    recommendations1 = recommendations[recommendations['id'].isin(movies['id'])]
    #print(len(recommendations1))
    #movies['Recommended scores'] = movies['id'].apply(lambda id:  recommendations svd.predict(userId, indices_map.loc[x]['movieId']).est)
    recommendations1 = recommendations1.sort_values('Recommended Scores', ascending=False)
    return recommendations1['title'].tolist()[:10]


# In[70]:


#hybrid(7,"Avatar")

#In[80]


