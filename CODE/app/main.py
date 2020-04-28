# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 22:49:20 2020

@author: vedant
"""

import rbm_v4_recommender as rbm
import json
import svd as sd
from flask import Flask, request,  render_template
import user_item as ui
import item_item as ii

app = Flask(__name__)
recommendations = rbm.hybrid(21,"Avatar")
print(recommendations)


@app.route('/rbm', methods=[ 'POST'])
def getRBM():
    #print(request.form)
    #content = request.json
    recommendation = rbm.hybrid(int(request.form['userId']),request.form['title'])
    #print(recommendation)
    response = app.response_class(
        response=json.dumps(recommendation)    ,
        status=200,
        mimetype='application/json'
    )
    return response

@app.route('/svd', methods=[ 'POST'])
def getSVD():
    #print(request.form)
    #content = request.json
    recommendation = sd.predictForUser(int(request.form['userId']))
    #print(recommendation)
    response = app.response_class(
        response=json.dumps(recommendation)    ,
        status=200,
        mimetype='application/json'
    )
    return response


@app.route('/user-based', methods=[ 'POST'])
def getUB():
    user_id=int(request.form['userId'])
    recommend_movies= ui.movieIdToTitle(ui.getRecommendedMoviesAsperUserSimilarity(user_id))
    response = app.response_class(
        response=json.dumps(recommend_movies)    ,
        status=200,
        mimetype='application/json'
    )
    return response

@app.route('/item-item', methods=[ 'POST'])
def getII():
    title=request.form['title']
    recommend_movies= ii.predictMovies(title)
    response = app.response_class(
        response=json.dumps(recommend_movies)    ,
        status=200,
        mimetype='application/json'
    )
    return response

@app.route('/content-based', methods=[ 'POST'])
def getCB():
    title=request.form['title']
    recommend_movies= rbm.get_recommendations(title).head(10)
    response = app.response_class(
        response=json.dumps(recommend_movies.to_list())    ,
        status=200,
        mimetype='application/json'
    )
    return response

@app.route('/home', methods=[ 'GET'])
def home():
   return render_template('index.html')

if __name__ == '__main__':
  app.run()