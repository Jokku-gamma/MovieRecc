from flask import Flask,request,jsonify
import pandas as pd
import numpy as np
import joblib
from azure.storage.blob import BlobServiceClient
import os
from dotenv import load_dotenv

load_dotenv()
app=Flask(__name__)
STORAGE_ACCOUNT_NAME=os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
STORAGE_ACCOUNT_KEY=os.environ.get("STORAGE_ACCOUNT_KEY")
CONTAINER_NAME=os.environ.get("CONTAINER_NAME")
SIMILARITY_FILENAME=os.environ.get("SIMILARITY_FILENAME")
DATA_FILENAME=os.environ.get("DATA_FILENAME")
STORAGE_CONNECTION_STRING=os.environ.get("AZURE_STORAGE_CONNECTION_STRING")

movies=None
similarity=None
unique_genres=[]
data_loaded=False

def load_data_from_azure():
    global movies,similarity,unique_genres,data_loaded
    if data_loaded:
        print("Data already loaded")
        return True
    print(f"Connecting to container :{CONTAINER_NAME}")
    print(f"Downloading data file :{DATA_FILENAME}")
    print(f"Downloading similarity file :{SIMILARITY_FILENAME}")
    if not STORAGE_ACCOUNT_NAME or not STORAGE_ACCOUNT_KEY or not CONTAINER_NAME or not DATA_FILENAME or not SIMILARITY_FILENAME:
        print("Error in azure connection config variables")
        return False
    try:
        blob_service_client=BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
        container_client=blob_service_client.get_container_client(CONTAINER_NAME)

        blob_client=container_client.get_blob_client(DATA_FILENAME)
        with open(DATA_FILENAME,"wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        movies=pd.read_csv(DATA_FILENAME)
        os.remove(DATA_FILENAME)

        blob_client=container_client.get_blob_client(SIMILARITY_FILENAME)
        with open(SIMILARITY_FILENAME,"wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        similarity=joblib.load(SIMILARITY_FILENAME)
        os.remove(SIMILARITY_FILENAME)

        genre_columns=[col for col in movies.columns if col not in ['movieId','title','org_genres']]
        unique_genres=genre_columns
        print("Data and similarity matrix loaded from Azure")
        return True
    except Exception as e:
        print(e)
        return False
    
def reccomend_movies(genre,top_n=10):
    if not data_loaded:
        if not load_data_from_azure():
            return []
    if movies in None or similarity is None:
        return []
    genre_movies=movies[movies[genre]==1]
    if genre_movies.empty:
        return []
    avg_sim=similarity[genre_movies.index].mean(axis=0)
    ranked_indices=avg_sim.argsort()[::-1]
    return movies.iloc[ranked_indices[:top_n]][['title','org_genres']].to_dict('records')

@app.route('/recommend/genre',methods=['GET'])
def genre_recc():
    genre=request.args.get('genre')
    if not genre:
        return jsonify({"error":"Pls provide a genre parameter"}),400
    if not data_loaded:
        if not load_data_from_azure():
            return jsonify({"error":"Failed to load data from Azure"}),500
        recc=reccomend_movies(genre)
        if not recc:
            return jsonify({"message":"No reccomendations found for this genre or data not loaded"}),404
        return jsonify(recc)
    
if __name__=="__main__":
    app.run(debug=False)
    