import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from kaggle_secrets import UserSecretsClient

# 1. Importation des données
df = pd.read_csv("data.csv")
genre_data = pd.read_csv("data_w_genres.csv")
year_data = pd.read_csv("data_by_year.csv")
artist_data = pd.read_csv("data_by_artist.csv")

# 2. Exploration des données
print(genre_data.info())

# 3. Création de la colonne 'decade' dans year_data
year_data["decade"] = year_data["year"].apply(lambda x: (x // 10) * 10)
print(year_data.head())

# 4. Visualisation des tendances par décennie
plt.figure(figsize=(10, 6))
sns.countplot(x='decade', data=year_data)
plt.title('Distribution des pistes par décennie')
plt.xlabel('Décennie')
plt.ylabel('Nombre de pistes')
plt.xticks(rotation=45)
plt.show()

# 5. Tendance des caractéristiques sonores au fil des décennies
caract_sonores = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence']
fig = px.line(year_data, x='year', y=caract_sonores,
              title='Tendance de diverses caractéristiques sonores au fil des décennies',
              labels={'year': 'Année', 'value': 'Valeur'},
              template='plotly_dark')
fig.update_layout(
    xaxis_title='Année',
    yaxis_title='Valeur',
    legend_title='Caractéristiques Sonores'
)
fig.show()

# 6. Tendance du volume sonore au fil des décennies
plt.figure(figsize=(14, 8))
plt.plot(year_data['year'], year_data['loudness'], label='Loudness', color='red')
plt.title('Tendance du volume sonore sur plusieurs décennies')
plt.xlabel('YEAR')
plt.ylabel('Loudness (dB)')
plt.grid(True)
plt.show()

# 7. Analyse de la popularité par genre
genre_popularity = genre_data.groupby('genres')['popularity'].mean().reset_index()
top10_genres = genre_popularity.nlargest(10, 'popularity')
sound_features = ['valence', 'energy', 'danceability', 'acousticness']
top10_genres = genre_data[genre_data['genres'].isin(top10_genres['genres'])][['genres'] + sound_features]
fig = px.bar(top10_genres, x='genres', y=sound_features,
             barmode='group',
             title='Tendance de diverses caractéristiques sonores sur les 10 principaux genres')
fig.update_layout(
    xaxis_title='Genres',
    yaxis_title='Valeur',
    legend_title='Caractéristiques Sonores'
)
fig.show()

# 8. Nuage de mots pour les genres
genres = genre_data['genres'].dropna()
comment_words = ' '.join(genres)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(width=800, height=800,
                      background_color='white',
                      stopwords=stopwords,
                      max_words=40,
                      min_font_size=10).generate(comment_words)
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

# 9. Nuage de mots pour les artistes
artists = genre_data['artists'].dropna()
comment_words = ' '.join(artists)
wordcloud = WordCloud(width=800, height=800,
                      background_color='white',
                      stopwords=stopwords,
                      min_word_length=3,
                      max_words=40,
                      min_font_size=10).generate(comment_words)
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

# 10. Artistes avec le plus de chansons produites
artist_song_count = genre_data['artists'].value_counts().reset_index()
artist_song_count.columns = ['artists', 'count']
top10_most_song_produced_artists = artist_song_count.nlargest(10, 'count')
print(top10_most_song_produced_artists[['count', 'artists']].sort_values('count', ascending=False))

# 11. Artistes les plus populaires
top_10_artists = genre_data[['popularity', 'artists']].sort_values('popularity', ascending=False).head(10)
print(top_10_artists)

# 12. Clustering des chansons
X = genre_data.select_dtypes(include=[np.number])
cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=12))
])
cluster_pipeline.fit(X)
genre_data['cluster'] = cluster_pipeline.predict(X)

# 13. Visualisation des clusters avec t-SNE
tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=1))])
genre_embedding = tsne_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)
projection['genres'] = genre_data['genres']
projection['cluster'] = genre_data['cluster']
fig = px.scatter(projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'genres'])
fig.show()

# 14. Clustering des chansons avec PCA
song_cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=25, verbose=False))
], verbose=False)
song_cluster_pipeline.fit(X)
song_cluster_labels = song_cluster_pipeline.predict(X)
genre_data['cluster_label'] = song_cluster_labels

pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
song_embedding = pca_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
projection['title'] = genre_data['artists']
projection['cluster'] = genre_data['cluster_label']
fig = px.scatter(projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'title'])
fig.show()

# 15. Intégration de l'API Spotify
user_secrets = UserSecretsClient()
spotify_client_id = user_secrets.get_secret("SPOTIFY_CLIENT_ID")
spotify_client_secret = user_secrets.get_secret("SPOTIFY_CLIENT_SECRET")
spotify_credentials = SpotifyClientCredentials(client_id=spotify_client_id, client_secret=spotify_client_secret)
sp = spotipy.Spotify(auth_manager=spotify_credentials)

# 16. Recherche de chansons sur Spotify
def find_song(spotify_client, song_name, artist_name=None):
    query = f"track:{song_name}"
    if artist_name:
        query += f" artist:{artist_name}"
    results = spotify_client.search(q=query, type='track')
    tracks = results['tracks']['items']
    songs = []
    for track in tracks:
        song_info = {
            'name': track['name'],
            'artists': ', '.join(artist['name'] for artist in track['artists']),
            'album': track['album']['name'],
            'url': track['external_urls']['spotify'],
            'id': track['id'],
        }
        songs.append(song_info)
    return songs

# 17. Obtenir les données d'une chanson
def get_song_data(spotify_client, dataset, song_name, artist_name=None):
    if artist_name:
        song_data = dataset[(dataset['song_name'].str.lower() == song_name.lower()) &
                            (dataset['artist_name'].str.lower() == artist_name.lower())]
    else:
        song_data = dataset[dataset['song_name'].str.lower() == song_name.lower()]
    if not song_data.empty:
        return song_data.iloc[0].to_dict()
    query = f"track:{song_name}"
    if artist_name:
        query += f" artist:{artist_name}"
    results = spotify_client.search(q=query, type='track')
    tracks = results['tracks']['items']
    if not tracks:
        return None
    track = tracks[0]
    song_info = {
        'name': track['name'],
        'artists': ', '.join(artist['name'] for artist in track['artists']),
        'album': track['album']['name'],
        'url': track['external_urls']['spotify'],
        'id': track['id'],
    }
    return song_info

# 18. Calculer le vecteur moyen pour une liste de chansons
def get_mean_vector(dataset, song_list):
    song_vectors = []
    for song in song_list:
        song_data = dataset[
            (dataset['song_name'].str.lower() == song['name'].lower()) &
            (dataset['artist_name'].str.lower() == song['artist'].lower())
        ]
        if not song_data.empty:
            song_vectors.append(song_data.select_dtypes(include=np.number).iloc[0])
    if not song_vectors:
        return None
    mean_vector = pd.concat(song_vectors)
