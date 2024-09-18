# Importation des bibliothèques
import streamlit as st
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
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy

# Utilisation directe des identifiants Spotify
SPOTIPY_CLIENT_ID = "32a36e74ba994588b8238efb75551ddf"
SPOTIPY_CLIENT_SECRET = "3ca67ec1b6ae48fc9e2df7d2ff3ee77a"
SPOTIPY_REDIRECT_URI = "http://localhost:8501/callback"

# Configuration de l'authentification Spotify
spotify_credentials = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=spotify_credentials)


# 1. Importation des données
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    genre_data = pd.read_csv("data_w_genres.csv")
    year_data = pd.read_csv("data_by_year.csv")
    artist_data = pd.read_csv("data_by_artist.csv")
    return df, genre_data, year_data, artist_data


df, genre_data, year_data, artist_data = load_data()

# Menu latéral
st.sidebar.title("Menu d'Analyse Musicale")
option = st.sidebar.selectbox(
    'Sélectionnez une analyse',
    ('Exploration des données', 'Tendances par décennie', 'Tendances des caractéristiques sonores',
     'Analyse de la popularité par genre', 'Nuage de mots - Genres', 'Nuage de mots - Artistes',
     'Clustering des chansons avec t-SNE', 'Clustering des chansons avec PCA', 'Recommandation musicale')
)


# Fonctions pour le système de recommandation musicale
def find_song(spotify_client, song_name, artist_name=None):
    query = f"track:{song_name}"
    if artist_name:
        query += f" artist:{artist_name}"
    results = spotify_client.search(q=query, type='track', limit=1)
    tracks = results['tracks']['items']
    if not tracks:
        return None
    track = tracks[0]
    audio_features = spotify_client.audio_features(track['id'])[0]
    song_data = {
        'name': track['name'],
        'artists': ', '.join([artist['name'] for artist in track['artists']]),
        'id': track['id'],
        'spotify_url': track['external_urls']['spotify'],
        'danceability': audio_features['danceability'],
        'energy': audio_features['energy'],
        'key': audio_features['key'],
        'loudness': audio_features['loudness'],
        'mode': audio_features['mode'],
        'speechiness': audio_features['speechiness'],
        'acousticness': audio_features['acousticness'],
        'instrumentalness': audio_features['instrumentalness'],
        'liveness': audio_features['liveness'],
        'valence': audio_features['valence'],
        'tempo': audio_features['tempo']
    }
    return song_data


def get_recommendations(spotify_client, track_id, num_recommendations=5):
    recommendations = spotify_client.recommendations(seed_tracks=[track_id], limit=num_recommendations)
    recommended_tracks = recommendations['tracks']
    recommended_songs = []
    for track in recommended_tracks:
        audio_features = spotify_client.audio_features(track['id'])[0]
        song_data = {
            'name': track['name'],
            'artists': ', '.join([artist['name'] for artist in track['artists']]),
            'id': track['id'],
            'spotify_url': track['external_urls']['spotify'],
            'danceability': audio_features['danceability'],
            'energy': audio_features['energy'],
            'valence': audio_features['valence']
        }
        recommended_songs.append(song_data)
    return recommended_songs


# Exploration des données
if option == 'Exploration des données':
    st.header('Exploration des données')
    st.write('Aperçu des données :')
    st.write(genre_data.head())
    st.write('Informations sur le dataset :')
    st.write(genre_data.info())
    st.write('Statistiques descriptives :')
    st.write(genre_data.describe())

# Tendances par décennie
elif option == 'Tendances par décennie':
    st.header('Tendances par décennie')
    year_data["decade"] = year_data["year"].apply(lambda x: (x // 10) * 10)
    plt.figure(figsize=(10, 6))
    sns.countplot(x='decade', data=year_data)
    plt.title('Distribution des pistes par décennie')
    plt.xlabel('Décennie')
    plt.ylabel('Nombre de pistes')
    plt.xticks(rotation=45)
    st.pyplot(plt)

# Recommandation musicale
elif option == 'Recommandation musicale':
    st.header('Système de Recommandation Musicale')

    st.write('Choisissez une option de recommandation :')
    menu_option = st.radio(
        'Comment souhaitez-vous obtenir des recommandations ?',
        ('Par nom de chanson', 'Recommandations basées sur une chanson existante')
    )

    if menu_option == 'Par nom de chanson':
        song_name = st.text_input('Entrez le nom de la chanson :')
        artist_name = st.text_input('Entrez le nom de l\'artiste (optionnel) :')

        if st.button('Trouver la chanson et recommander'):
            song_data = find_song(sp, song_name, artist_name)
            if song_data:
                st.write(f"Chanson trouvée : {song_data['name']} par {song_data['artists']}")
                st.write(f"[Écouter sur Spotify]({song_data['spotify_url']})")
                st.write('Caractéristiques de la chanson :')
                st.json(song_data)

                recommended_songs = get_recommendations(sp, song_data['id'])
                st.write('Recommandations de chansons similaires :')
                for song in recommended_songs:
                    st.write(f"{song['name']} par {song['artists']} [Écouter sur Spotify]({song['spotify_url']})")
                    st.write(
                        f"Danceability: {song['danceability']}, Energy: {song['energy']}, Valence: {song['valence']}")
                    st.write('---')
            else:
                st.write("Chanson non trouvée, veuillez vérifier les informations saisies.")

    elif menu_option == 'Recommandations basées sur une chanson existante':
        song_name = st.text_input('Entrez le nom de la chanson :')
        artist_name = st.text_input('Entrez le nom de l\'artiste (optionnel) :')

        if st.button('Trouver et recommander des chansons similaires'):
            song_data = find_song(sp, song_name, artist_name)
            if song_data:
                st.write(f"Chanson trouvée : {song_data['name']} par {song_data['artists']}")
                st.write(f"[Écouter sur Spotify]({song_data['spotify_url']})")
                st.write('Caractéristiques de la chanson :')
                st.json(song_data)

                recommended_songs = get_recommendations(sp, song_data['id'])
                st.write('Recommandations de chansons similaires :')
                for song in recommended_songs:
                    st.write(f"{song['name']} par {song['artists']} [Écouter sur Spotify]({song['spotify_url']})")
                    st.write(
                        f"Danceability: {song['danceability']}, Energy: {song['energy']}, Valence: {song['valence']}")
                    st.write('---')
            else:
                st.write("Chanson non trouvée, veuillez vérifier les informations saisies.")
