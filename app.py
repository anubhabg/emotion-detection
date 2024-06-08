from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import json
from urllib.parse import quote
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load the models and data
faceCascade = cv2.CascadeClassifier("datasets/haarcascade_frontalface_default.xml")
CNN_Model = load_model("model/trained_CNN_model.h5")
imdb_data = pd.read_csv('datasets/imdb_top_1000.csv')
music_data = pd.read_csv("datasets/data_moods.csv")

with open('datasets/recommendations.json', 'r') as file:
    recommendations = json.load(file)

# Preprocess IMDb data
imdb_data = imdb_data[['Series_Title', 'Genre', 'IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Gross']]
imdb_data['Meta_score'] = imdb_data['Meta_score'].fillna(imdb_data['Meta_score'].mean())
imdb_data['Gross'] = imdb_data['Gross'].replace(',', '', regex=True)
imdb_data['Gross'] = pd.to_numeric(imdb_data['Gross'], errors='coerce')
imdb_data['Gross'] = imdb_data['Gross'].fillna(imdb_data['Gross'].mean())

# Genre Processing
genre_list = imdb_data['Genre'].str.split(',').explode().str.strip().unique().tolist()
for genre in genre_list:
    imdb_data[genre] = imdb_data['Genre'].apply(lambda x: 1 if genre in x else 0)

# Calculate Average Ratings
avg_movie_rating = imdb_data[['Series_Title', 'IMDB_Rating', 'No_of_Votes', 'Meta_score', 'Gross']]
min_reviews = int(np.percentile(avg_movie_rating['No_of_Votes'], 70))
avg_rating_all = imdb_data['IMDB_Rating'].mean()

def weighted_rating(x, m=min_reviews, C=avg_rating_all):
    v = x['No_of_Votes']
    R = x['IMDB_Rating']
    return (v / (v + m) * R) + (m / (m + v) * C)

imdb_data['weighted_score'] = avg_movie_rating.apply(weighted_rating, axis=1)

def best_movies_by_genre(genre, top_n):
    if genre not in imdb_data.columns:
        return pd.DataFrame()  # Return an empty DataFrame if genre does not exist
    genre_movies = imdb_data[imdb_data[genre] == 1]
    top_movies = genre_movies.sort_values('weighted_score', ascending=False).head(top_n)
    return top_movies[['Series_Title', 'IMDB_Rating', 'No_of_Votes', 'weighted_score', 'Genre']]

def get_genres_for_mood(mood):
    mood_genres = {
        'Happy': ['Comedy', 'Music', 'Romance', 'Animation', 'Fantasy'],
        'Sad': ['Drama', 'Romance', 'War', 'Mystery'],
        'Fear': ['Horror', 'Thriller', 'Mystery', 'Sci-Fi'],
        'Surprise': ['Thriller', 'Mystery', 'Fantasy', 'Sci-Fi'],
        'Neutral': ['Biography', 'Drama', 'Animation', 'Adventure'],
        'Angry': ['Action', 'Crime', 'Thriller', 'War'],
        'Disgust': ['Horror', 'Crime', 'Drama', 'Family']
    }
    return mood_genres.get(mood, [])

def recommend_movie_links_by_mood(mood, top_n=5):
    genres = get_genres_for_mood(mood)
    recommended_movies = pd.DataFrame()
    
    for genre in genres:
        top_movies_by_genre = best_movies_by_genre(genre, top_n)
        recommended_movies = pd.concat([recommended_movies, top_movies_by_genre]).drop_duplicates().head(top_n)
    
    movie_links = []
    for title in recommended_movies['Series_Title']:
        encoded_title = quote(title)
        imdb_link = f"https://www.imdb.com/find?q={encoded_title}"
        youtube_link = f"https://www.youtube.com/results?search_query={encoded_title}+trailer"
        wikipedia_link = f"https://en.wikipedia.org/wiki/{encoded_title}"
        movie_links.append({
            'title': title,
            'imdb': imdb_link,
            'youtube': youtube_link,
            'wikipedia': wikipedia_link
        })
    
    return movie_links

def Recommend_Songs(pred_class):
    mood_map = {
        'Disgust': 'Sad',
        'Happy': 'Happy',
        'Sad': 'Happy',
        'Fear': 'Calm',
        'Angry': 'Calm',
        'Surprise': 'Energetic',
        'Neutral': 'Energetic'
    }
    mood = mood_map.get(pred_class, 'Happy')
    songs = music_data[music_data['mood'] == mood]
    songs = songs.sort_values(by="popularity", ascending=False)
    return songs.head(5)[['name', 'artist']].to_dict(orient='records')

def Recommend_Books_Yogas(pred_class):
    return recommendations.get(pred_class, {})

def load_and_prep_image(image):
    img_shape = 48
    GrayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(GrayImg, 1.1, 4)

    for x, y, w, h in faces:
        roi_GrayImg = GrayImg[y: y + h, x: x + w]
        roi_Img = image[y: y + h, x: x + w]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        faces = faceCascade.detectMultiScale(roi_Img, 1.1, 4)
        if len(faces) == 0:
            print("No Faces Detected")
        else:
            for (ex, ey, ew, eh) in faces:
                img = roi_Img[ey: ey + eh, ex: ex + ew]

    RGBImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    RGBImg = cv2.resize(RGBImg, (img_shape, img_shape))
    RGBImg = RGBImg / 255.0
    return RGBImg

def predict_emotion(image):
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    img = load_and_prep_image(image)
    if img is None:
        return "No face detected"

    pred = CNN_Model.predict(np.expand_dims(img, axis=0))
    pred_class = class_names[pred.argmax()]
    return pred_class

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if 'image' in request.files and request.files['image'].filename != '':
        image_file = request.files['image']
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_file.filename))
        image_file.save(image_path)
        image = cv2.imread(image_path)
    elif 'captured_image' in request.form and request.form['captured_image'] != '':
        image_data = request.form['captured_image']
        image_data = image_data.split(',')[1]
        image_data = base64.b64decode(image_data)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'captured_image.png')
        with open(image_path, 'wb') as f:
            f.write(image_data)
        image = cv2.imread(image_path)
    else:
        return "No image provided."

    if image is None:
        return "Failed to read the image file."
    
    predicted_emotion = predict_emotion(image)
    songs = Recommend_Songs(predicted_emotion)
    movies = recommend_movie_links_by_mood(predicted_emotion)
    books_yogas = Recommend_Books_Yogas(predicted_emotion)
    
    return render_template('result.html', mood=predicted_emotion, songs=songs, movies=movies, books_yogas=books_yogas)

@app.route('/developers')
def developers():
    developers = [
        {
            "name": "Utsab Dutta",
            "email": "utsab_2303res118@iitp.ac.in",
            "roll": "2303res118",
            "phone": "+91-9739244583",
            "image": "images/developers/utsab.jpg"
        },
        {
            "name": "Anubhab Ghosh",
            "email": "anubhab_2303res135@iitp.ac.in",
            "roll": "2303res135",
            "phone": "+91-8017658216",
            "image": "images/developers/anubhab.jpg"
        },
        {
            "name": "Vatsala Singh",
            "email": "vatsala_2303res198@iitp.ac.in",
            "roll": "2303res198",
            "phone": "+91-6386669478",
            "image": "images/developers/vatsala.jpeg"
        }
    ]
    return render_template('developers.html', developers=developers)

if __name__ == '__main__':
    app.run(debug=True, port=5001)