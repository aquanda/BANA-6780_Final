'''
There are three parts within this python project. The first is a content based recommender that uses the plot of each
movie to determine the cosine similarity of the movies in the dataset. This is done using the Term Frequency-Inverse
Document Frequency (TFIDF) function to determine the uniqueness and importance of words within each movie plot. After
inputting a movie name into line 87 print(suggest_movie('Dune')), the process will return the 10 most similar movies.

The second part is a collaborative based recommender that uses the machine learning capabilities of PySpark to predict
user ratings of movies. It splits the data into a training and testing dataset (80/20) and uses Alternating Least
Squares (ALS) and Root Mean Squared Error (RMSE) in the process.


'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from ast import literal_eval

from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
spark = SparkSession.builder.appName('Recommender').config("spark.driver.memory", "8g").config("spark.executor.memory", "8g").getOrCreate()
spark

#content based recommendation
def has_meaningful_words(sentence):
    if isinstance(sentence, float):
        return False
    words = sentence.split()
    for word in words:
        if word.lower() not in ENGLISH_STOP_WORDS:
            return True
    return False


#import the movie metadata file using pandas
dfmeta = pd.read_csv('movies_metadata.csv', low_memory=False)
dfmeta = dfmeta[dfmeta['overview'].apply(has_meaningful_words)]
#determine the average rating across all movies
mean_rating = dfmeta['vote_average'].mean()

#establish a treshhold for the minimum number of votes a movie needs to have
min_vote_count = dfmeta['vote_count'].quantile(0.90)

#create a new df of movies based on min_vote_count
dfmovies = dfmeta.copy().loc[dfmeta['vote_count'] >= min_vote_count]

#define a function that calculates the weighted rating of a movie
def weighted_rating(x, C=min_vote_count, R=mean_rating):
    #retrieve the vote count of the movie
    count = x['vote_count']
    #retrieve the average rating of the movie
    average = x['vote_average']
    #calculate the weighted rating of the movie
    weighted_rating = (count/(count+C) * average) + (C/(count+C) * R)
    return weighted_rating

#create a new column 'score' in dfmovies that displays the weighted rating function
dfmovies['weighted rating'] = dfmovies.apply(weighted_rating, axis=1)

#sort movies by their weighted rating
dfmovies = dfmovies.sort_values('weighted rating', ascending=False)

#print(dfmovies[['title','vote_count', 'vote_average', 'weighted rating']].head(25))

#create a Term Frequency-Inverse Document Frequency (TFIDF) object that identifies rare words within movie plots. stop_words removes filler words (the, and, etc.)
tfidf = TfidfVectorizer(stop_words='english')

#replaces NaN's with empty values
dfmeta['overview'] = dfmeta['overview'].fillna('')

#creates a matrix using the TFIDF data
tfidf_matrix = tfidf.fit_transform(dfmeta['overview'])

#calculates the cosine similarity of each movie to one another
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#create an object that maps movie titles to their corresponding indices in dfmeta
indices = pd.Series(dfmeta.index, index=dfmeta['title']).drop_duplicates()

#define a function that inputs the title of a movie and returns 10 movies with the highest cosine similarity
def suggest_movie(title, similarity_matrix=cosine_sim):
    #get the index of the given title from the indices dictionary
    index = indices[title]
    #create a tuple for each movie where the first element is its index and the second is the similarity score
    similarity_scores = list(enumerate(similarity_matrix[index]))
    #sort the list of tuples based on the similarity score
    similarity_scores = sorted(similarity_scores, key=lambda score: score[1], reverse=True)
    #slice the sorted list to get the top 10 similar movie indices (0 being itself)
    similarity_scores = similarity_scores[1:11]
    #get the indices of each movie in similarity_scores
    movie_indices = [i[0] for i in similarity_scores]
    #return the titles of the top 10 similar movies
    return dfmeta['title'].iloc[movie_indices]

#test suggest_movie
print(suggest_movie('Dune'))

#collaborative based recommendation
spdata = spark.read.csv('ratings_small.csv', inferSchema=True, header=True)

#divide the data using random split into train_data and test_data (80/20)
train_data, test_data = spdata.randomSplit([0.8, 0.2])

#build the recommendation model using Alternating Least Squares (ALS) on the training data
als = ALS(maxIter=5,
          regParam=0.01,
          userCol="userId",
          itemCol="movieId",
          ratingCol="rating")

#fit the model on the train_data
model = als.fit(train_data)

#evaluate the model by computing the RMSE on test data
predictions = model.transform(test_data)
#drop any missing values from predictions
predictions = predictions.na.drop()

#display the initial prediction model
predictions.show()

#print and calculate RMSE
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

#filter user id which it has given reviews
user1 = test_data.filter(test_data['userId'] == 1).select(['movieId', 'userId', 'rating'])
#display user1 data
user1.show()

#training and evaluating for user1 with the trained model
recommendations = model.transform(user1)
#display the predictions of movies for user1
recommendations.orderBy('prediction', ascending=False).show()

#hybrid recommendation system that uses both collaborative and content for its recommendation
def hybrid_recommendation(userId, title):
    # create a list of movies recommended by collaborative filtering for the user
    user_movies = model.recommendForAllUsers(10)
    user_movies = user_movies.filter(user_movies.userId == userId).collect()[0]["recommendations"]
    # create a list of movieIds from the recommendations
    user_movieIds = [movie[0] for movie in user_movies]
    # filter dfmeta for only movies in the user_movieIds list
    user_meta = dfmeta[dfmeta["movieId"].isin(user_movieIds)]
    non_stopword_overviews = user_meta['overview'].apply(has_meaningful_words)
    if not any(non_stopword_overviews):
        print("No meaningful movie overviews for the recommended movies for this user.")
        return None
    else:
        user_meta = user_meta[non_stopword_overviews]
    # create a matrix using the TFIDF data
    tfidf_matrix = tfidf.fit_transform(user_meta['overview'])
    # calculates the cosine similarity of each movie to one another
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    # create an object that maps movie titles to their corresponding indices in dfmeta
    indices = pd.Series(user_meta.index, index=user_meta['title']).drop_duplicates()
    # recommend movies using the content based method
    recommended_movies = suggest_movie(title, cosine_sim)
    return recommended_movies

# test hybrid_recommendation
print(hybrid_recommendation(1, 'Alien'))

spark.stop()