#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
# # Dataset : Movielens
# https://grouplens.org/datasets/movielens/100k

# In[9]:


# Reading ratings file
ratings = pd.read_csv('C:/Users/vedan/OneDrive/Desktop/courses/CSE-573-Semnatic-Web-Mining/project/ml-100k/ratings.csv', sep=',', encoding='latin-1', usecols=['userId','movieId','rating','timestamp'])

# Reading movies file
movies = pd.read_csv('C:/Users/vedan/OneDrive/Desktop/courses/CSE-573-Semnatic-Web-Mining/project/ml-100k/movies.csv', sep=',', encoding='latin-1', usecols=['movieId','title','genres'])


# In[10]:


df_movies = movies 
df_ratings = ratings 


# ## Implementation of Item-Item Filtering

# In[11]:


df_movies_ratings=pd.merge(df_movies, df_ratings)


# In[12]:


df_movies_ratings


# Here Pivot table function is used as we want one to one maping between movies, user and their rating. 
# So by default pivot_table command takes average if we have multiple values of one combination.

# In[13]:


ratings_matrix_items = df_movies_ratings.pivot_table(index=['movieId'],columns=['userId'],values='rating').reset_index(drop=True)
ratings_matrix_items.fillna( 0, inplace = True )
ratings_matrix_items.shape


# In[14]:


ratings_matrix_items


# In[15]:


from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation
movie_similarity = 1 - pairwise_distances( ratings_matrix_items, metric="cosine" )
np.fill_diagonal( movie_similarity, 0 ) #Filling diagonals with 0s for future use when sorting is done
ratings_matrix_items = pd.DataFrame( movie_similarity )
ratings_matrix_items


# Below function will take the movie name as a input and will find the movies which are similar to this movie.
# This function first find the index of movie in movies frame and then take the similarity of movie and align in movies dataframe so that we can get the similarity of the movie with all other movies.

# In[16]:


def item_similarity(movieName): 
    """
    recomendates similar movies
   :param data: name of the movie 
   """
    try:
        #user_inp=input('Enter the reference movie title based on which recommendations are to be made: ')
        user_inp=movieName
        inp=df_movies[df_movies['title']==user_inp].index.tolist()
        inp=inp[0]

        df_movies['similarity'] = ratings_matrix_items.iloc[inp]
        df_movies.columns = ['movie_id', 'title', 'release_date','similarity']
    except:
        print("Sorry, the movie is not in the database!")


# Here we provide the user id of the user for which we have to recommend movies.
# Then we find the movies which are rated 5 or 4.5 by the user for whom we want to recommend movies.
# We are finding this because as we know that in Item-Item similarity approach we recommended movies to the user based on his previous selection.
# So to foster our algorithm we are finding movies which are liked by the user most and on bases of that we will recommend movies with are similar to movies highly rated by the user.
# Then our function has appended the similarity of the movie highly rated by the user to our movies data frame.
# Now we will sort the frame as per the similarity in descending order so that we can get the movies which are highly similar to movie highly rated bu our customer.
# Now we filter the movies which are most similar as per the similarity so if similarity is greater than 0.45 then we are considering the movies.
# Now the function goes ahead and see which all movies user has seen and then filter out the movies which he has not seen and than recommended that movies to him.

# In[17]:


def recommendedMoviesAsperItemSimilarity(user_id):
    """
     Recommending movie which user hasn't watched as per Item Similarity
    :param user_id: user_id to whom movie needs to be recommended
    :return: movieIds to user 
    """
    user_movie= df_movies_ratings[(df_movies_ratings.userId==user_id) & df_movies_ratings.rating.isin([5,4.5])][['title']]
    user_movie=user_movie.iloc[0,0]
    item_similarity(user_movie)
    sorted_movies_as_per_userChoice=df_movies.sort_values( ["similarity"], ascending = False )
    sorted_movies_as_per_userChoice=sorted_movies_as_per_userChoice[sorted_movies_as_per_userChoice['similarity'] >=0.45]['movie_id']
    
    df_recommended_item=pd.DataFrame()
    user2Movies= df_ratings[df_ratings['userId']== user_id]['movieId']
    for movieId in sorted_movies_as_per_userChoice:
            if movieId not in user2Movies:
                df_new= df_ratings[(df_ratings.movieId==movieId)]
                df_recommended_item=pd.concat([df_recommended_item,df_new])
            best10=df_recommended_item.sort_values(["rating"], ascending = False )[1:10] 
    return best10['movieId']


# In[18]:


def movieIdToTitle(listMovieIDs):
    """
     Converting movieId to titles
    :param user_id: List of movies
    :return: movie titles
    """
    movie_titles= list()
    for id in listMovieIDs:
        movie_titles.append(df_movies[df_movies['movieId']==id]['title'].values[0])
    return movie_titles


# In[19]:


# ## Implementation of User-Item Filtering

# In similar way as we did for ItemItem similarity we will create a matrix but here we will keep rows as user and columns as movieId as we want a vector of different users.
# Then in similar ways we will find distance and similarity between users.

# In[20]:


ratings_matrix_users = df_movies_ratings.pivot_table(index=['userId'],columns=['movieId'],values='rating').reset_index(drop=True)
ratings_matrix_users.fillna( 0, inplace = True )
movie_similarity = 1 - pairwise_distances( ratings_matrix_users, metric="cosine" )
np.fill_diagonal( movie_similarity, 0 ) #Filling diagonals with 0s for future use when sorting is done
ratings_matrix_users = pd.DataFrame( movie_similarity )
ratings_matrix_users


# Here now we have similarity of users in colums with respective users in row. So if we find maximum value in a column we will get the user with highest similarity. So now we can have a pair of users which are similar.

# In[21]:


ratings_matrix_users.idxmax(axis=1)


# In[22]:


ratings_matrix_users.idxmax(axis=1).sample( 10, random_state = 10 )


# In[23]:


similar_user_series= ratings_matrix_users.idxmax(axis=1)
df_similar_user= similar_user_series.to_frame()


# In[24]:


df_similar_user.columns=['similarUser']


# In[25]:


df_similar_user


# Below function takes id of the user to whom we have to recommend movies. On basis of that, we find the user which is similar to that user and then filter the movies which are highly rated by the user to recommend them to given user.

# In[26]:


movieId_recommended=list()
def getRecommendedMoviesAsperUserSimilarity(userId):
    """
     Recommending movies which user hasn't watched as per User Similarity
    :param user_id: user_id to whom movie needs to be recommended
    :return: movieIds to user 
    """
    user2Movies= df_ratings[df_ratings['userId']== userId]['movieId']
    sim_user=df_similar_user.iloc[userId,0]
    df_recommended=pd.DataFrame(columns=['movieId','title','genres','userId','rating','timestamp'])
    for movieId in df_ratings[df_ratings['userId']== sim_user]['movieId']:
        if movieId not in user2Movies:
            df_new= df_movies_ratings[(df_movies_ratings.userId==sim_user) & (df_movies_ratings.movieId==movieId)]
            df_recommended=pd.concat([df_recommended,df_new])
        best10=df_recommended.sort_values(['rating'], ascending = False )[1:10]  
    return best10['movieId']


# In[27]:






