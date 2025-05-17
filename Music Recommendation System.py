import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
df = pd.read_csv("C:/Users/Sunil Daggubati/Downloads/spotify_tracks.csv")
df.drop(columns=["id"],inplace=True)
df.duplicated(subset=df.drop(columns=["name"]).columns).sum()
df.drop_duplicates(subset=['genre', 'artists', 'album', 'popularity', 'duration_ms', 'explicit'],inplace=True)
df["artists"].nunique()
df["album"].nunique()
df["duration_ms"].describe()
df["popularity"].describe()
df["explicit"].value_counts()
plt.pie(df["explicit"].value_counts(),labels=df["explicit"].unique(),autopct='%1.2f%%')
plt.show()
for column in ["genre", "artists", "album", "explicit"]:
    LE = LabelEncoder()
    df[column] = LE.fit_transform(df[column])
scaler = StandardScaler()
df[["popularity", "duration_ms"]] = scaler.fit_transform(df[["popularity", "duration_ms"]])
similarity_matrix = cosine_similarity(df.drop(columns=["name"]))
def recommend(song_index, num_recommendations=5):
    similar_songs = list(enumerate(similarity_matrix[song_index]))
    similar_songs = sorted(similar_songs, key=lambda x: x[1], reverse=True)
    similar_songs = similar_songs[1:num_recommendations+1]
    recommendations = [df.iloc[i[0]]['name'] for i in similar_songs]
    return recommendations
cos_recommendations = recommend(song_index=0, num_recommendations=5)
cos_recommendations
knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(df.drop(columns=["name"]))
distances, indices = knn.kneighbors([df.drop(columns=['name']).iloc[0]], n_neighbors=5)
knn_recommendations = df.iloc[indices[0]]['name'].tolist()
for idx, rec in enumerate(knn_recommendations, start=1):
    print(f"{idx}. {rec}")
def recommend(song_index, num_recommendations=5):
    similar_songs = list(enumerate(similarity_matrix[song_index]))
    similar_songs = sorted(similar_songs, key=lambda x: x[1], reverse=True)
    similar_song_indices = [i[0] for i in similar_songs[1:num_recommendations+1]]
    recommended_song_names = df.iloc[similar_song_indices]['name'].tolist()
    return recommended_song_names

auto_encoder_recoms = recommend(song_index=0, num_recommendations=5)
for idx, rec in enumerate(auto_encoder_recoms, start=1):
    print(f"{idx}. {rec}")
